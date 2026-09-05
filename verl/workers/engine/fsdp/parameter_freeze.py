# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Fail-closed value-model parameter freezing and bounded update evidence."""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import torch
from torch import nn

MANIFEST_SCHEMA = "verl_critic_parameter_freeze_v3"
LOADER_PROVENANCE_SCHEMA = "verl_model_loader_provenance_v1"
QWEN35_DENSE_TOKEN_MIXERS_POLICY = "qwen35_dense_token_mixers_v1"

_CORE_FROZEN_CATEGORIES = ("self_attention", "linear_attention")
_AUXILIARY_FROZEN_CATEGORIES = ("visual_auxiliary", "lm_head_auxiliary")
_FROZEN_CATEGORIES = _CORE_FROZEN_CATEGORIES + _AUXILIARY_FROZEN_CATEGORIES
_TRAINABLE_CATEGORIES = ("value_head", "mlp", "norm", "embedding")
_ALL_CATEGORIES = _FROZEN_CATEGORIES + _TRAINABLE_CATEGORIES
_VALUE_HEAD_COMPONENTS = frozenset({"score", "classifier", "v_head", "value_head"})
_FSDP_WRAPPER_COMPONENTS = frozenset({"_fsdp_wrapped_module"})
_FORBIDDEN_COMPONENTS = frozenset({"visual", "vision_model", "lm_head", "mtp"})
_TRL_VALUE_HEAD_WRAPPER_CLASS = "AutoModelForCausalLMWithValueHead"
_QWEN35_CONDITIONAL_GENERATION_CLASS = "Qwen3_5ForConditionalGeneration"
_TRL_VISUAL_PARAMETER_ROOT = "pretrained_model.model.visual."
_TRL_LM_HEAD_PARAMETER_ROOT = "pretrained_model.lm_head."
_TRL_LANGUAGE_EMBEDDING_COMPONENT = "embed_tokens"
_VALUE_MODEL_LOADER_ENTRYPOINT = "verl.utils.model.load_valuehead_model"
_SAFE_TENSORS_INDEX = "model.safetensors.index.json"
_PYTORCH_INDEX = "pytorch_model.bin.index.json"
_SINGLE_WEIGHT_FILES = ("model.safetensors", "pytorch_model.bin")


class ParameterFreezePolicyError(RuntimeError):
    """Raised when a configured freeze policy cannot be proven safe."""


@dataclass
class _ParameterProbeSample:
    name: str
    category: str
    parameter: nn.Parameter
    indices: torch.Tensor
    before: torch.Tensor


@dataclass
class FirstOptimizerStepProbe:
    """Bounded parameter samples captured immediately before an optimizer step."""

    frozen_samples: list[_ParameterProbeSample]
    trainable_samples: list[_ParameterProbeSample]
    reduction_device: torch.device


@dataclass
class ParameterFreezeRuntime:
    """Runtime evidence state for one configured value-model freeze policy."""

    manifest_path: Path
    manifest: dict[str, Any]
    category_by_name: dict[str, str]
    requires_grad_by_name: dict[str, bool]
    probe_elements_per_rank: int
    rank: int
    optimizer_steps_completed: int = 0
    first_optimizer_step_complete: bool = False

    def write_manifest(self) -> None:
        if self.rank == 0:
            _atomic_write_json(self.manifest_path, self.manifest)


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _name_sha256(names: list[str]) -> str:
    return _sha256_json(sorted(names))


def _canonical_parameter_name(name: str) -> str:
    components = [component for component in name.split(".") if component not in _FSDP_WRAPPER_COMPONENTS]
    return ".".join(components)


def _named_parameters_by_canonical_name(module: nn.Module) -> dict[str, nn.Parameter]:
    result: dict[str, nn.Parameter] = {}
    for raw_name, parameter in module.named_parameters():
        name = _canonical_parameter_name(raw_name)
        if not name:
            raise ParameterFreezePolicyError(f"empty canonical parameter name for {raw_name!r}")
        if name in result:
            raise ParameterFreezePolicyError(f"duplicate canonical parameter name: {name}")
        result[name] = parameter
    return result


def _owner_module(module_by_name: dict[str, nn.Module], parameter_name: str) -> nn.Module:
    owner_name, _, _ = parameter_name.rpartition(".")
    try:
        return module_by_name[owner_name]
    except KeyError as exc:
        raise ParameterFreezePolicyError(
            f"cannot resolve owner module {owner_name!r} for parameter {parameter_name!r}"
        ) from exc


def _is_scalar_value_head(parameter_name: str, parameter: nn.Parameter, owner: nn.Module) -> bool:
    components = set(parameter_name.split("."))
    if not components.intersection(_VALUE_HEAD_COMPONENTS):
        return False

    out_features = getattr(owner, "out_features", None)
    if out_features == 1:
        return True

    # Covers a direct scalar head parameter while rejecting a language-model
    # projection that merely happens to use a familiar wrapper name.
    leaf_name = parameter_name.rsplit(".", 1)[-1]
    if leaf_name == "weight" and parameter.ndim >= 1 and parameter.shape[0] == 1:
        return True
    if leaf_name == "bias" and parameter.numel() == 1:
        return True
    return False


def _classify_parameter(parameter_name: str, parameter: nn.Parameter, owner: nn.Module) -> str | None:
    components = parameter_name.split(".")
    component_set = set(components)
    owner_class = type(owner).__name__.lower()

    # This policy is intentionally limited to a text-only scalar critic. In
    # particular, generic names such as ``mlp`` or ``norm`` must not make a
    # retained vision tower or MTP block look like an allowed text parameter.
    if _forbidden_parameter_component(parameter_name) is not None:
        return None

    # Token-mixer ownership dominates nested projection/norm class names.
    if "self_attn" in component_set:
        return "self_attention"
    if "linear_attn" in component_set:
        return "linear_attention"

    if component_set.intersection(_VALUE_HEAD_COMPONENTS):
        return "value_head" if _is_scalar_value_head(parameter_name, parameter, owner) else None
    if "mlp" in component_set:
        return "mlp"
    if "embed_tokens" in component_set or "embedding" in owner_class:
        return "embedding"
    if "norm" in owner_class or any("norm" in component.lower() for component in components):
        return "norm"
    return None


def _forbidden_parameter_component(parameter_name: str) -> str | None:
    return next(
        (component for component in parameter_name.split(".") if component in _FORBIDDEN_COMPONENTS),
        None,
    )


def _is_exact_qwen35_trl_value_head_wrapper(module: nn.Module) -> bool:
    pretrained_model = getattr(module, "pretrained_model", None)
    return (
        type(module).__name__ == _TRL_VALUE_HEAD_WRAPPER_CLASS
        and pretrained_model is not None
        and type(pretrained_model).__name__ == _QWEN35_CONDITIONAL_GENERATION_CLASS
    )


def _parameter_alias_records(
    module: nn.Module,
    *,
    canonical_name_by_id: dict[int, str],
    exact_trl_wrapper: bool,
) -> list[dict[str, Any]]:
    """Validate and describe shared parameter aliases hidden by named_parameters().

    Dense Qwen3.5 ties ``lm_head.weight`` to ``embed_tokens.weight``.  That
    single tensor cannot simultaneously be a frozen auxiliary projection and a
    trainable embedding, so the policy classifies the unique tensor as the
    embedding and records the LM-head name as an explicit trainable alias.
    Every other shared-parameter topology remains fail-closed.
    """

    aliases_by_id: dict[int, list[str]] = {}
    for raw_name, parameter in module.named_parameters(remove_duplicate=False):
        name = _canonical_parameter_name(raw_name)
        aliases_by_id.setdefault(id(parameter), []).append(name)

    records: list[dict[str, Any]] = []
    for parameter_id, raw_names in aliases_by_id.items():
        names = sorted(set(raw_names))
        if len(names) == 1:
            continue
        canonical_name = canonical_name_by_id.get(parameter_id)
        lm_head_names = [name for name in names if name.startswith(_TRL_LM_HEAD_PARAMETER_ROOT)]
        embedding_names = [
            name for name in names if _TRL_LANGUAGE_EMBEDDING_COMPONENT in name.split(".")
        ]
        allowed = (
            exact_trl_wrapper
            and len(names) == 2
            and len(lm_head_names) == 1
            and len(embedding_names) == 1
            and canonical_name == embedding_names[0]
            and lm_head_names[0].endswith(".weight")
            and embedding_names[0].endswith(".weight")
        )
        if not allowed:
            raise ParameterFreezePolicyError(
                "unsupported shared parameter aliases: " + ", ".join(names)
            )
        records.append(
            {
                "canonical_name": canonical_name,
                "canonical_category": "embedding",
                "alias_name": lm_head_names[0],
                "alias_role": "tied_lm_head",
                "requires_grad": True,
            }
        )
    return sorted(records, key=lambda record: (record["canonical_name"], record["alias_name"]))


def _classify_frozen_auxiliary_parameter(module: nn.Module, parameter_name: str) -> str | None:
    category = None
    if parameter_name.startswith(_TRL_VISUAL_PARAMETER_ROOT):
        category = "visual_auxiliary"
    elif parameter_name.startswith(_TRL_LM_HEAD_PARAMETER_ROOT):
        category = "lm_head_auxiliary"

    if category is not None and not _is_exact_qwen35_trl_value_head_wrapper(module):
        raise ParameterFreezePolicyError(
            f"{category} is only supported on the exact "
            f"{_TRL_VALUE_HEAD_WRAPPER_CLASS}({_QWEN35_CONDITIONAL_GENERATION_CLASS}) wrapper: "
            f"{parameter_name}"
        )
    return category


def _model_config_dict(hf_config: Any) -> dict[str, Any]:
    if hasattr(hf_config, "to_dict"):
        config = hf_config.to_dict()
    elif isinstance(hf_config, dict):
        config = dict(hf_config)
    else:
        config = dict(vars(hf_config))
    if not isinstance(config, dict):
        raise ParameterFreezePolicyError("hf_config did not produce a mapping")
    return config


def _regular_loader_file(root: Path, relative: str, *, role: str) -> dict[str, Any]:
    pure = PurePosixPath(relative)
    if pure.is_absolute() or ".." in pure.parts or len(pure.parts) != 1:
        raise ParameterFreezePolicyError(f"unsafe loader artifact path: {relative!r}")
    path = root / relative
    if not path.is_file() or path.is_symlink():
        raise ParameterFreezePolicyError(f"loader artifact is missing, non-regular, or a symlink: {path}")
    size_bytes = path.stat().st_size
    if size_bytes <= 0:
        raise ParameterFreezePolicyError(f"loader artifact is empty: {path}")
    return {
        "relative_path": relative,
        "role": role,
        "size_bytes": size_bytes,
        "sha256": _sha256_file(path),
    }


def _weight_artifact_paths(root: Path) -> tuple[str | None, list[str]]:
    """Resolve exactly the weight files selected by ``from_pretrained``.

    Transformers prefers safetensors over PyTorch checkpoints.  For a sharded
    checkpoint the index's weight map, rather than a directory glob, is the
    authoritative loader file set.  This matters for Qwen3.5 roots that retain
    compatibility hard links under names the active index never reads.
    """

    for index_name in (_SAFE_TENSORS_INDEX, _PYTORCH_INDEX):
        index_path = root / index_name
        if not index_path.exists():
            continue
        if not index_path.is_file() or index_path.is_symlink():
            raise ParameterFreezePolicyError(f"weight index is not a regular file: {index_path}")
        try:
            raw_index = json.loads(index_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ParameterFreezePolicyError(f"cannot parse weight index {index_path}: {exc}") from exc
        weight_map = raw_index.get("weight_map") if isinstance(raw_index, dict) else None
        if not isinstance(weight_map, dict) or not weight_map:
            raise ParameterFreezePolicyError(f"weight index has no non-empty weight_map: {index_path}")
        shard_names = sorted(set(weight_map.values()))
        if not shard_names or any(not isinstance(name, str) or not name for name in shard_names):
            raise ParameterFreezePolicyError(f"weight index contains invalid shard names: {index_path}")
        for name in shard_names:
            pure = PurePosixPath(name)
            if pure.is_absolute() or ".." in pure.parts or len(pure.parts) != 1:
                raise ParameterFreezePolicyError(f"unsafe weight shard path in {index_path}: {name!r}")
        return index_name, shard_names

    present = [name for name in _SINGLE_WEIGHT_FILES if (root / name).exists()]
    if len(present) != 1:
        raise ParameterFreezePolicyError(
            "loader root must contain exactly one supported unsharded weight file "
            f"when no weight index is present; found {present}"
        )
    return None, present


def _loader_provenance(
    *,
    configured_path: str | None,
    loader_path: str | None,
    loader_entrypoint: str,
    resolved_config_sha256: str,
    source_revision: str | None,
) -> dict[str, Any]:
    if not isinstance(loader_entrypoint, str) or not loader_entrypoint:
        raise ParameterFreezePolicyError("loader entrypoint must be a non-empty string")
    if not loader_path:
        raise ParameterFreezePolicyError("actual loader path is required by the parameter freeze policy")
    root = Path(loader_path)
    if not root.is_absolute():
        raise ParameterFreezePolicyError(f"actual loader path must be absolute: {root}")
    if not root.is_dir() or root.is_symlink():
        raise ParameterFreezePolicyError(f"actual loader path must be a non-symlink directory: {root}")
    real_root = root.resolve(strict=True)

    index_name, weight_names = _weight_artifact_paths(real_root)
    records = [_regular_loader_file(real_root, "config.json", role="config")]
    if index_name is not None:
        records.append(_regular_loader_file(real_root, index_name, role="weight_index"))
    records.extend(
        _regular_loader_file(real_root, name, role="weight_shard") for name in weight_names
    )
    records.sort(key=lambda record: record["relative_path"])
    weight_records = [record for record in records if record["role"] == "weight_shard"]
    artifact_manifest_sha256 = _sha256_json(records)
    weight_manifest_sha256 = _sha256_json(weight_records)

    configured_realpath = None
    configured_samefile = False
    if configured_path:
        configured = Path(configured_path)
        if configured.is_absolute() and configured.exists():
            configured_realpath = str(configured.resolve(strict=True))
            try:
                configured_samefile = configured.samefile(real_root)
            except OSError:
                configured_samefile = False

    resolved_revision = (
        f"huggingface:{source_revision}"
        if source_revision
        else f"artifact-sha256:{artifact_manifest_sha256}"
    )
    return {
        "schema": LOADER_PROVENANCE_SCHEMA,
        "loader_entrypoint": loader_entrypoint,
        "configured_path": str(configured_path) if configured_path is not None else None,
        "configured_realpath": configured_realpath,
        "loader_path": str(root),
        "loader_realpath": str(real_root),
        "configured_path_samefile": configured_samefile,
        "resolved_config_sha256": resolved_config_sha256,
        "source_revision": source_revision,
        "resolved_revision": resolved_revision,
        "artifact_files": records,
        "artifact_manifest_sha256": artifact_manifest_sha256,
        "weight_manifest_sha256": weight_manifest_sha256,
        "weight_file_count": len(weight_records),
        "weight_total_bytes": sum(record["size_bytes"] for record in weight_records),
        "readback": {
            "status": "pass",
            "phase": "post_from_pretrained",
            "file_count": len(records),
            "total_bytes": sum(record["size_bytes"] for record in records),
            "all_regular": True,
            "symlink_count": 0,
        },
    }


def _validate_manifest_path(path: Path) -> None:
    if not path.is_absolute():
        raise ParameterFreezePolicyError(f"parameter freeze manifest path must be absolute: {path}")
    if path.exists() and not path.is_file():
        raise ParameterFreezePolicyError(f"parameter freeze manifest path is not a regular file: {path}")
    if path.is_symlink():
        raise ParameterFreezePolicyError(f"refusing to replace symlink parameter freeze manifest: {path}")


def _atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    _validate_manifest_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    temporary_path = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with temporary_path.open("w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _summary_for_names(records_by_name: dict[str, dict[str, Any]], names: list[str]) -> dict[str, Any]:
    return {
        "parameter_count": len(names),
        "numel": sum(records_by_name[name]["numel"] for name in names),
        "name_sha256": _name_sha256(names),
    }


def configure_parameter_freeze_policy(
    module: nn.Module,
    *,
    policy: str | None,
    manifest_path: str | None,
    probe_elements_per_rank: int,
    model_role: str,
    hf_config: Any,
    model_path: str | None,
    loader_path: str | None,
    rank: int,
    loader_entrypoint: str = _VALUE_MODEL_LOADER_ENTRYPOINT,
) -> ParameterFreezeRuntime | None:
    """Validate, classify, and apply a supported parameter-freezing policy.

    No parameter is mutated until every parameter has a recognized category and
    all policy invariants have passed.
    """

    if policy is None:
        if manifest_path is not None:
            raise ParameterFreezePolicyError("parameter_freeze_manifest_path requires parameter_freeze_policy")
        return None
    if policy != QWEN35_DENSE_TOKEN_MIXERS_POLICY:
        raise ParameterFreezePolicyError(f"unsupported parameter freeze policy: {policy!r}")
    if model_role != "value_model":
        raise ParameterFreezePolicyError(
            f"{QWEN35_DENSE_TOKEN_MIXERS_POLICY} is critic-only; got model role {model_role!r}"
        )
    if not manifest_path:
        raise ParameterFreezePolicyError("parameter_freeze_manifest_path is required when the policy is enabled")
    resolved_manifest_path = Path(manifest_path)
    _validate_manifest_path(resolved_manifest_path)
    if (
        not isinstance(probe_elements_per_rank, int)
        or isinstance(probe_elements_per_rank, bool)
        or probe_elements_per_rank < 2
    ):
        raise ParameterFreezePolicyError("parameter_freeze_probe_elements_per_rank must be an integer >= 2")

    config = _model_config_dict(hf_config)
    hf_model_type = config.get("model_type", getattr(hf_config, "model_type", None))
    if hf_model_type != "qwen3_5":
        raise ParameterFreezePolicyError(
            f"{QWEN35_DENSE_TOKEN_MIXERS_POLICY} requires dense qwen3_5, got {hf_model_type!r}"
        )

    module_by_name = dict(module.named_modules())
    is_exact_trl_wrapper = _is_exact_qwen35_trl_value_head_wrapper(module)
    canonical_name_by_id = {
        id(parameter): _canonical_parameter_name(name)
        for name, parameter in module.named_parameters()
    }
    parameter_aliases = _parameter_alias_records(
        module,
        canonical_name_by_id=canonical_name_by_id,
        exact_trl_wrapper=is_exact_trl_wrapper,
    )
    classified: list[tuple[str, nn.Parameter, nn.Module, str]] = []
    forbidden: list[tuple[str, str]] = []
    unknown: list[str] = []
    seen_parameter_ids: set[int] = set()
    for parameter_name, parameter in sorted(module.named_parameters(), key=lambda item: item[0]):
        if id(parameter) in seen_parameter_ids:
            raise ParameterFreezePolicyError(f"shared parameter is not supported by this policy: {parameter_name}")
        seen_parameter_ids.add(id(parameter))
        owner = _owner_module(module_by_name, parameter_name)
        auxiliary_category = _classify_frozen_auxiliary_parameter(module, parameter_name)
        if auxiliary_category is not None:
            classified.append((parameter_name, parameter, owner, auxiliary_category))
            continue
        forbidden_component = _forbidden_parameter_component(parameter_name)
        if forbidden_component is not None:
            forbidden.append((parameter_name, forbidden_component))
            continue
        category = _classify_parameter(parameter_name, parameter, owner)
        if category is None:
            unknown.append(parameter_name)
        else:
            classified.append((parameter_name, parameter, owner, category))

    if forbidden:
        preview = ", ".join(f"{name} ({component})" for name, component in forbidden[:8])
        suffix = "" if len(forbidden) <= 8 else f" (+{len(forbidden) - 8} more)"
        raise ParameterFreezePolicyError(f"forbidden parameter subtrees under {policy}: {preview}{suffix}")
    if unknown:
        preview = ", ".join(unknown[:8])
        suffix = "" if len(unknown) <= 8 else f" (+{len(unknown) - 8} more)"
        raise ParameterFreezePolicyError(f"unclassified parameters under {policy}: {preview}{suffix}")
    if not classified:
        raise ParameterFreezePolicyError("model has no parameters")

    tied_lm_head = any(record["alias_role"] == "tied_lm_head" for record in parameter_aliases)
    required_categories = (
        tuple(category for category in _ALL_CATEGORIES if category != "lm_head_auxiliary")
        if is_exact_trl_wrapper and tied_lm_head
        else (_ALL_CATEGORIES if is_exact_trl_wrapper else _CORE_FROZEN_CATEGORIES + _TRAINABLE_CATEGORIES)
    )
    category_counts = {category: 0 for category in _ALL_CATEGORIES}
    for _, _, _, category in classified:
        category_counts[category] += 1
    missing_categories = [
        category for category in required_categories if category_counts[category] == 0
    ]
    if missing_categories:
        raise ParameterFreezePolicyError(
            f"{policy} requires every parameter category; missing {', '.join(missing_categories)}"
        )

    revision = config.get("_commit_hash") or getattr(hf_config, "_commit_hash", None)
    config_sha256 = _sha256_json(config)
    # Hash and validate the actual post-load artifact before changing any
    # parameter flags.  Only rank zero reads the multi-gigabyte checkpoint;
    # only rank zero owns the persisted manifest.
    loader_provenance = None
    if rank == 0:
        loader_provenance = _loader_provenance(
            configured_path=model_path,
            loader_path=loader_path,
            loader_entrypoint=loader_entrypoint,
            resolved_config_sha256=config_sha256,
            source_revision=str(revision) if revision is not None else None,
        )

    records: list[dict[str, Any]] = []
    category_by_name: dict[str, str] = {}
    requires_grad_by_name: dict[str, bool] = {}
    for parameter_name, parameter, owner, category in classified:
        requires_grad = category in _TRAINABLE_CATEGORIES
        parameter.requires_grad_(requires_grad)
        category_by_name[parameter_name] = category
        requires_grad_by_name[parameter_name] = requires_grad
        records.append(
            {
                "name": parameter_name,
                "shape": list(parameter.shape),
                "numel": parameter.numel(),
                "dtype": str(parameter.dtype).removeprefix("torch."),
                "module_class": type(owner).__name__,
                "category": category,
                "requires_grad": requires_grad,
                "optimizer_member": None,
            }
        )

    records_by_name = {record["name"]: record for record in records}
    trainable_names = sorted(name for name, trainable in requires_grad_by_name.items() if trainable)
    frozen_names = sorted(name for name, trainable in requires_grad_by_name.items() if not trainable)
    trainable_summary = _summary_for_names(records_by_name, trainable_names)
    frozen_summary = _summary_for_names(records_by_name, frozen_names)

    categories: dict[str, dict[str, Any]] = {}
    for category in _ALL_CATEGORIES:
        names = sorted(name for name, value in category_by_name.items() if value == category)
        category_summary = _summary_for_names(records_by_name, names)
        category_summary["requires_grad"] = category in _TRAINABLE_CATEGORIES
        categories[category] = category_summary

    manifest: dict[str, Any] = {
        "schema": MANIFEST_SCHEMA,
        "name_sha256_encoding": "sha256(canonical-json(sorted(parameter_names)))",
        "status": "pending_first_optimizer_step",
        "policy": policy,
        "model": {
            "class": type(module).__name__,
            "pretrained_model_class": type(getattr(module, "pretrained_model", None)).__name__
            if getattr(module, "pretrained_model", None) is not None
            else None,
            "model_type": hf_model_type,
            "role": model_role,
            "config_sha256": config_sha256,
            "path": str(model_path) if model_path is not None else None,
            "revision": str(revision) if revision is not None else None,
        },
        "loader_provenance": loader_provenance,
        "parameter_aliases": parameter_aliases,
        "parameters": records,
        "categories": categories,
        "frozen_parameter_count": frozen_summary["parameter_count"],
        "frozen_parameter_numel": frozen_summary["numel"],
        "frozen_parameter_name_sha256": frozen_summary["name_sha256"],
        "trainable_parameter_count": trainable_summary["parameter_count"],
        "trainable_parameter_numel": trainable_summary["numel"],
        "trainable_parameter_name_sha256": trainable_summary["name_sha256"],
        "optimizer": {
            "membership_exact": False,
            "parameter_count": None,
            "numel": None,
            "name_sha256": None,
        },
        "first_optimizer_step": {
            "status": "pending",
            "completed_step": 0,
            "frozen": {"changed_count": None, "l2": None, "max_abs": None, "sampled_elements": None},
            "trainable": {"changed_count": None, "l2": None, "max_abs": None, "sampled_elements": None},
            "probe_elements_per_rank": probe_elements_per_rank,
            "learning_rates": None,
        },
        "optimizer_step_learning_rates": {
            "required_steps": 2,
            "status": "pending",
            "steps": [],
        },
    }
    runtime = ParameterFreezeRuntime(
        manifest_path=resolved_manifest_path,
        manifest=manifest,
        category_by_name=category_by_name,
        requires_grad_by_name=requires_grad_by_name,
        probe_elements_per_rank=probe_elements_per_rank,
        rank=rank,
    )
    runtime.write_manifest()
    return runtime


def finalize_optimizer_membership(
    runtime: ParameterFreezeRuntime,
    module: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> None:
    """Prove that the optimizer contains exactly the trainable parameters."""

    named_parameters = _named_parameters_by_canonical_name(module)
    expected_names = set(runtime.category_by_name)
    if set(named_parameters) != expected_names:
        missing = sorted(expected_names - set(named_parameters))[:8]
        unexpected = sorted(set(named_parameters) - expected_names)[:8]
        raise ParameterFreezePolicyError(
            f"parameter names changed after FSDP wrapping; missing={missing}, unexpected={unexpected}"
        )

    for name, parameter in named_parameters.items():
        expected = runtime.requires_grad_by_name[name]
        if parameter.requires_grad != expected:
            raise ParameterFreezePolicyError(
                f"requires_grad drift after FSDP wrapping for {name}: {parameter.requires_grad} != {expected}"
            )

    optimizer_parameters = [parameter for group in optimizer.param_groups for parameter in group["params"]]
    optimizer_parameter_ids = [id(parameter) for parameter in optimizer_parameters]
    if len(optimizer_parameter_ids) != len(set(optimizer_parameter_ids)):
        raise ParameterFreezePolicyError("optimizer contains duplicate parameter objects")

    trainable_parameter_ids = {id(parameter) for parameter in named_parameters.values() if parameter.requires_grad}
    if set(optimizer_parameter_ids) != trainable_parameter_ids:
        raise ParameterFreezePolicyError("optimizer membership does not exactly match requires_grad=True parameters")

    name_by_parameter_id = {id(parameter): name for name, parameter in named_parameters.items()}
    optimizer_names = sorted(name_by_parameter_id[parameter_id] for parameter_id in optimizer_parameter_ids)
    expected_trainable_names = sorted(name for name, value in runtime.requires_grad_by_name.items() if value)
    if optimizer_names != expected_trainable_names:
        raise ParameterFreezePolicyError("optimizer parameter names do not match the classified trainable name list")

    optimizer_name_set = set(optimizer_names)
    records_by_name = {record["name"]: record for record in runtime.manifest["parameters"]}
    for name, record in records_by_name.items():
        record["optimizer_member"] = name in optimizer_name_set
    runtime.manifest["optimizer"] = {
        "membership_exact": True,
        "parameter_count": len(optimizer_names),
        "numel": sum(records_by_name[name]["numel"] for name in optimizer_names),
        "name_sha256": _name_sha256(optimizer_names),
    }
    runtime.write_manifest()


def _local_parameter_tensor(parameter: nn.Parameter) -> torch.Tensor:
    tensor = parameter.detach()
    to_local = getattr(tensor, "to_local", None)
    if callable(to_local):
        tensor = to_local()
    return tensor.reshape(-1)


def _evenly_spaced_indices(numel: int, count: int) -> torch.Tensor:
    if count <= 0 or numel <= 0:
        return torch.empty(0, dtype=torch.long)
    if count >= numel:
        return torch.arange(numel, dtype=torch.long)
    return torch.div(torch.arange(count, dtype=torch.long) * numel, count, rounding_mode="floor")


def _sample_parameter_group(
    named_parameters: dict[str, nn.Parameter],
    runtime: ParameterFreezeRuntime,
    *,
    categories: tuple[str, ...],
    budget: int,
) -> list[_ParameterProbeSample]:
    candidates: list[tuple[str, nn.Parameter, torch.Tensor]] = []
    for category in categories:
        for name in sorted(runtime.category_by_name):
            if runtime.category_by_name[name] != category:
                continue
            parameter = named_parameters[name]
            local_tensor = _local_parameter_tensor(parameter)
            if local_tensor.numel() > 0:
                candidates.append((name, parameter, local_tensor))

    if not candidates or budget <= 0:
        return []
    if len(candidates) > budget:
        selected_positions = _evenly_spaced_indices(len(candidates), budget).tolist()
        candidates = [candidates[position] for position in selected_positions]

    samples: list[_ParameterProbeSample] = []
    remaining_budget = budget
    for position, (name, parameter, local_tensor) in enumerate(candidates):
        remaining_parameters = len(candidates) - position
        count = min(local_tensor.numel(), max(1, remaining_budget // remaining_parameters))
        indices = _evenly_spaced_indices(local_tensor.numel(), count)
        selected = local_tensor.index_select(0, indices.to(local_tensor.device)).cpu().clone()
        samples.append(
            _ParameterProbeSample(
                name=name,
                category=runtime.category_by_name[name],
                parameter=parameter,
                indices=indices,
                before=selected,
            )
        )
        remaining_budget -= count
    return samples


def begin_first_optimizer_step_probe(
    runtime: ParameterFreezeRuntime,
    module: nn.Module,
) -> FirstOptimizerStepProbe:
    """Capture a deterministic bounded sample immediately before step one."""

    if runtime.first_optimizer_step_complete or runtime.optimizer_steps_completed != 0:
        raise ParameterFreezePolicyError("the first optimizer-step probe has already completed")

    named_parameters = _named_parameters_by_canonical_name(module)
    if set(named_parameters) != set(runtime.category_by_name):
        raise ParameterFreezePolicyError("cannot probe a model whose parameter names drifted")

    frozen_budget = runtime.probe_elements_per_rank // 2
    trainable_budget = runtime.probe_elements_per_rank - frozen_budget
    frozen_samples = _sample_parameter_group(
        named_parameters,
        runtime,
        categories=_FROZEN_CATEGORIES,
        budget=frozen_budget,
    )
    trainable_samples = _sample_parameter_group(
        named_parameters,
        runtime,
        categories=_TRAINABLE_CATEGORIES,
        budget=trainable_budget,
    )

    reduction_device = torch.device("cpu")
    for sample in frozen_samples + trainable_samples:
        reduction_device = _local_parameter_tensor(sample.parameter).device
        break
    else:
        for parameter in named_parameters.values():
            reduction_device = _local_parameter_tensor(parameter).device
            break
    return FirstOptimizerStepProbe(
        frozen_samples=frozen_samples,
        trainable_samples=trainable_samples,
        reduction_device=reduction_device,
    )


def _local_probe_metrics(samples: list[_ParameterProbeSample]) -> tuple[int, float, float, int]:
    changed_count = 0
    squared_l2 = 0.0
    max_abs = 0.0
    sampled_elements = 0
    for sample in samples:
        local_tensor = _local_parameter_tensor(sample.parameter)
        after = local_tensor.index_select(0, sample.indices.to(local_tensor.device)).cpu()
        if after.shape != sample.before.shape:
            raise ParameterFreezePolicyError(f"local parameter shape changed during optimizer step: {sample.name}")
        changed_count += int(torch.count_nonzero(after != sample.before).item())
        delta = after.to(torch.float64) - sample.before.to(torch.float64)
        squared_l2 += float(torch.sum(delta * delta).item())
        max_abs = max(max_abs, float(torch.max(torch.abs(delta)).item()) if delta.numel() else 0.0)
        sampled_elements += delta.numel()
    return changed_count, squared_l2, max_abs, sampled_elements


def _aggregate_probe_metrics(
    frozen: tuple[int, float, float, int],
    trainable: tuple[int, float, float, int],
    reduction_device: torch.device,
) -> tuple[dict[str, Any], dict[str, Any]]:
    sum_values = torch.tensor(
        [frozen[0], frozen[1], frozen[3], trainable[0], trainable[1], trainable[3]],
        dtype=torch.float64,
        device=reduction_device,
    )
    max_values = torch.tensor([frozen[2], trainable[2]], dtype=torch.float64, device=reduction_device)
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.all_reduce(sum_values, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(max_values, op=torch.distributed.ReduceOp.MAX)

    summed = sum_values.cpu().tolist()
    maxima = max_values.cpu().tolist()
    frozen_metrics = {
        "changed_count": int(summed[0]),
        "l2": math.sqrt(summed[1]),
        "max_abs": maxima[0],
        "sampled_elements": int(summed[2]),
    }
    trainable_metrics = {
        "changed_count": int(summed[3]),
        "l2": math.sqrt(summed[4]),
        "max_abs": maxima[1],
        "sampled_elements": int(summed[5]),
    }
    return frozen_metrics, trainable_metrics


def finish_first_optimizer_step_probe(
    runtime: ParameterFreezeRuntime,
    probe: FirstOptimizerStepProbe,
    *,
    completed_step: int,
    learning_rates: list[float],
) -> dict[str, Any]:
    """Record and enforce the first completed optimizer-step delta gate."""

    frozen_metrics, trainable_metrics = _aggregate_probe_metrics(
        _local_probe_metrics(probe.frozen_samples),
        _local_probe_metrics(probe.trainable_samples),
        probe.reduction_device,
    )
    errors: list[str] = []
    if completed_step != 1:
        errors.append(f"expected completed optimizer step 1, got {completed_step}")
    if frozen_metrics["sampled_elements"] <= 0:
        errors.append("frozen probe sampled zero elements")
    if trainable_metrics["sampled_elements"] <= 0:
        errors.append("trainable probe sampled zero elements")
    if frozen_metrics["changed_count"] != 0 or frozen_metrics["max_abs"] != 0.0:
        errors.append("frozen parameter probe changed")
    if trainable_metrics["changed_count"] <= 0 or trainable_metrics["max_abs"] <= 0.0:
        errors.append("trainable parameter probe did not change")

    step_manifest = {
        "status": "fail" if errors else "pass",
        "completed_step": completed_step,
        "frozen": frozen_metrics,
        "trainable": trainable_metrics,
        "probe_elements_per_rank": runtime.probe_elements_per_rank,
        "learning_rates": learning_rates,
    }
    if errors:
        step_manifest["errors"] = errors
    runtime.manifest["first_optimizer_step"] = step_manifest
    if errors:
        runtime.manifest["status"] = "fail"
        runtime.write_manifest()
        raise ParameterFreezePolicyError("; ".join(errors))
    runtime.first_optimizer_step_complete = True
    record_optimizer_step_learning_rates(
        runtime,
        completed_step=completed_step,
        learning_rates=learning_rates,
    )
    return step_manifest


def record_optimizer_step_learning_rates(
    runtime: ParameterFreezeRuntime,
    *,
    completed_step: int,
    learning_rates: list[float],
) -> dict[str, Any]:
    """Record the LR used by each of the first two successful optimizer steps."""

    evidence = runtime.manifest["optimizer_step_learning_rates"]
    required_steps = evidence["required_steps"]
    if completed_step > required_steps:
        return evidence

    errors: list[str] = []
    expected_step = len(evidence["steps"]) + 1
    if completed_step != expected_step:
        errors.append(f"expected completed optimizer step {expected_step}, got {completed_step}")
    if runtime.optimizer_steps_completed != completed_step:
        errors.append(
            "optimizer-step runtime count does not match the LR evidence step: "
            f"{runtime.optimizer_steps_completed} != {completed_step}"
        )
    if not learning_rates:
        errors.append("optimizer step has no learning-rate parameter groups")
    elif any(not math.isfinite(value) or value < 0.0 for value in learning_rates):
        errors.append(f"optimizer step has invalid learning rates: {learning_rates}")

    if errors:
        evidence["status"] = "fail"
        evidence["errors"] = errors
        runtime.manifest["status"] = "fail"
        runtime.write_manifest()
        raise ParameterFreezePolicyError("; ".join(errors))

    evidence["steps"].append(
        {
            "completed_step": completed_step,
            "learning_rates": learning_rates,
        }
    )
    evidence["status"] = "pass" if len(evidence["steps"]) == required_steps else "pending"
    evidence.pop("errors", None)
    runtime.manifest["status"] = (
        "pass" if evidence["status"] == "pass" else "pending_optimizer_step_learning_rates"
    )
    runtime.write_manifest()
    return evidence
