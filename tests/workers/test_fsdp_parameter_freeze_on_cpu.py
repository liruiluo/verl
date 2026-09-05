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

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard

from verl.workers.engine.fsdp.parameter_freeze import (
    MANIFEST_SCHEMA,
    QWEN35_DENSE_TOKEN_MIXERS_POLICY,
    ParameterFreezePolicyError,
    begin_first_optimizer_step_probe,
    configure_parameter_freeze_policy,
    finalize_optimizer_membership,
    finish_first_optimizer_step_probe,
)
from verl.workers.engine.fsdp.transformer_impl import FSDPEngine


class _FakeQwen35Config:
    model_type = "qwen3_5"
    _commit_hash = "test-revision"

    def to_dict(self):
        return {
            "model_type": self.model_type,
            "architectures": ["Qwen3_5ForTokenClassification"],
            "hidden_size": 8,
            "_commit_hash": self._commit_hash,
        }


class _TinyMixerLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = nn.Linear(8, 8, bias=False)
        self.linear_attn = nn.Linear(8, 8, bias=False)
        self.mlp = nn.Sequential(nn.Linear(8, 16), nn.GELU(), nn.Linear(16, 8))
        self.input_layernorm = nn.LayerNorm(8)
        self.post_attention_layernorm = nn.LayerNorm(8)

    def forward(self, hidden_states):
        mixed = self.self_attn(hidden_states) + self.linear_attn(hidden_states)
        hidden_states = self.input_layernorm(hidden_states + mixed)
        return self.post_attention_layernorm(hidden_states + self.mlp(hidden_states))


class _TinyQwen35ValueModel(nn.Module):
    def __init__(self, *, include_unknown=False, include_retained_multimodal_heads=False):
        super().__init__()
        self.embed_tokens = nn.Embedding(32, 8)
        self.layers = nn.ModuleList([_TinyMixerLayer(), _TinyMixerLayer()])
        self.norm = nn.LayerNorm(8)
        self.score = nn.Linear(8, 1)
        if include_unknown:
            self.mystery_projection = nn.Linear(8, 8)
        if include_retained_multimodal_heads:
            self.visual = nn.ModuleDict({"mlp": nn.Linear(8, 8), "norm": nn.LayerNorm(8)})
            self.lm_head = nn.Linear(8, 32, bias=False)

    def forward(self, input_ids):
        hidden_states = self.embed_tokens(input_ids).mean(dim=1)
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return self.score(self.norm(hidden_states))


class _TinyQwen35TextModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(32, 8)
        self.layers = nn.ModuleList([_TinyMixerLayer(), _TinyMixerLayer()])
        self.norm = nn.LayerNorm(8)

    def forward(self, input_ids):
        hidden_states = self.embed_tokens(input_ids).mean(dim=1)
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return self.norm(hidden_states)


class _TinyQwen35ModelContainer(nn.Module):
    def __init__(self):
        super().__init__()
        # Nested mixer/MLP/norm names prove that the exact visual root takes
        # precedence over otherwise trainable text-component names.
        self.visual = nn.ModuleDict({"blocks": nn.ModuleList([_TinyMixerLayer()])})
        self.language_model = _TinyQwen35TextModel()


class Qwen3_5ForConditionalGeneration(nn.Module):
    def __init__(self, *, tie_lm_head=False):
        super().__init__()
        self.model = _TinyQwen35ModelContainer()
        self.lm_head = nn.Linear(8, 32, bias=False)
        if tie_lm_head:
            self.lm_head.weight = self.model.language_model.embed_tokens.weight


class _TinyValueHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.summary = nn.Linear(8, 1)


class AutoModelForCausalLMWithValueHead(nn.Module):
    def __init__(self, *, tie_lm_head=False):
        super().__init__()
        self.pretrained_model = Qwen3_5ForConditionalGeneration(tie_lm_head=tie_lm_head)
        self.v_head = _TinyValueHead()

    def forward(self, input_ids):
        hidden_states = self.pretrained_model.model.language_model(input_ids)
        return self.v_head.summary(hidden_states)


class _WrongValueHeadWrapper(AutoModelForCausalLMWithValueHead):
    pass


def _configure(model: nn.Module, manifest_path: Path, *, model_role="value_model", model_type="qwen3_5"):
    config = _FakeQwen35Config()
    config.model_type = model_type
    loader_root = manifest_path.parent / "loader-model"
    if manifest_path.is_absolute():
        loader_root.mkdir(parents=True, exist_ok=True)
        (loader_root / "config.json").write_text(
            json.dumps(config.to_dict(), sort_keys=True) + "\n", encoding="utf-8"
        )
        (loader_root / "model-00001-of-00002.safetensors").write_bytes(b"first-weight-shard")
        (loader_root / "model-00002-of-00002.safetensors").write_bytes(b"second-weight-shard")
        (loader_root / "model.safetensors.index.json").write_text(
            json.dumps(
                {
                    "metadata": {"total_size": 37},
                    "weight_map": {
                        "model.embed_tokens.weight": "model-00001-of-00002.safetensors",
                        "model.layers.0.weight": "model-00002-of-00002.safetensors",
                    },
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    return configure_parameter_freeze_policy(
        model,
        policy=QWEN35_DENSE_TOKEN_MIXERS_POLICY,
        manifest_path=str(manifest_path),
        probe_elements_per_rank=64,
        model_role=model_role,
        hf_config=config,
        model_path=str(loader_root),
        loader_path=str(loader_root),
        rank=0,
    )


def _build_engine(model, optimizer, runtime):
    engine = object.__new__(FSDPEngine)
    engine.module = model
    engine.optimizer = optimizer
    engine.optimizer_config = SimpleNamespace(clip_grad=1.0)
    engine.scaler = None
    engine._qat_enabled = False
    engine._parameter_freeze_runtime = runtime
    engine._optimizer_steps_completed = 0
    return engine


class ParameterFreezeTests(unittest.TestCase):
    def test_policy_classifies_freezes_and_records_first_real_optimizer_step(self):
        with tempfile.TemporaryDirectory() as raw_dir:
            torch.manual_seed(7)
            model = AutoModelForCausalLMWithValueHead()
            manifest_path = Path(raw_dir) / "parameter-freeze.json"
            runtime = _configure(model, manifest_path)
            self.assertIsNotNone(runtime)

            frozen_before = {
                name: parameter.detach().clone()
                for name, parameter in model.named_parameters()
                if runtime.category_by_name[name]
                in {"self_attention", "linear_attention", "visual_auxiliary", "lm_head_auxiliary"}
            }
            trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
            optimizer = torch.optim.AdamW(trainable_parameters, lr=0.05, weight_decay=0.0)
            finalize_optimizer_membership(runtime, model, optimizer)

            pending_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(pending_manifest["schema"], MANIFEST_SCHEMA)
            self.assertEqual(pending_manifest["status"], "pending_first_optimizer_step")
            self.assertTrue(pending_manifest["optimizer"]["membership_exact"])
            self.assertIsNone(pending_manifest["first_optimizer_step"]["learning_rates"])
            self.assertEqual(
                pending_manifest["optimizer_step_learning_rates"],
                {"required_steps": 2, "status": "pending", "steps": []},
            )
            self.assertIsNone(pending_manifest["first_optimizer_step"]["frozen"]["sampled_elements"])
            self.assertIsNone(pending_manifest["first_optimizer_step"]["trainable"]["sampled_elements"])
            self.assertEqual(
                set(pending_manifest["categories"]),
                {
                    "self_attention",
                    "linear_attention",
                    "visual_auxiliary",
                    "lm_head_auxiliary",
                    "value_head",
                    "mlp",
                    "norm",
                    "embedding",
                },
            )
            self.assertEqual(pending_manifest["model"]["class"], "AutoModelForCausalLMWithValueHead")
            self.assertEqual(
                pending_manifest["model"]["pretrained_model_class"],
                "Qwen3_5ForConditionalGeneration",
            )
            provenance = pending_manifest["loader_provenance"]
            self.assertEqual(provenance["schema"], "verl_model_loader_provenance_v1")
            self.assertEqual(
                provenance["loader_entrypoint"], "verl.utils.model.load_valuehead_model"
            )
            self.assertTrue(provenance["configured_path_samefile"])
            self.assertEqual(provenance["loader_realpath"], str((Path(raw_dir) / "loader-model").resolve()))
            self.assertEqual(provenance["resolved_config_sha256"], pending_manifest["model"]["config_sha256"])
            self.assertEqual(provenance["source_revision"], "test-revision")
            self.assertEqual(provenance["resolved_revision"], "huggingface:test-revision")
            self.assertEqual(provenance["weight_file_count"], 2)
            self.assertEqual(
                {record["relative_path"] for record in provenance["artifact_files"]},
                {
                    "config.json",
                    "model.safetensors.index.json",
                    "model-00001-of-00002.safetensors",
                    "model-00002-of-00002.safetensors",
                },
            )
            self.assertEqual(provenance["readback"]["status"], "pass")
            self.assertEqual(pending_manifest["parameter_aliases"], [])
            self.assertTrue(
                all(
                    not record["requires_grad"] and not record["optimizer_member"]
                    for record in pending_manifest["parameters"]
                    if record["category"]
                    in {"self_attention", "linear_attention", "visual_auxiliary", "lm_head_auxiliary"}
                )
            )
            self.assertTrue(
                all(
                    record["requires_grad"] and record["optimizer_member"]
                    for record in pending_manifest["parameters"]
                    if record["category"]
                    not in {"self_attention", "linear_attention", "visual_auxiliary", "lm_head_auxiliary"}
                )
            )
            self.assertEqual(
                pending_manifest["optimizer"]["name_sha256"],
                pending_manifest["trainable_parameter_name_sha256"],
            )

            engine = _build_engine(model, optimizer, runtime)
            input_ids = torch.tensor([[1, 2, 3], [4, 5, 6]])
            model(input_ids).square().mean().backward()
            grad_norm = engine.optimizer_step()
            self.assertGreater(grad_norm, 0)
            self.assertEqual(engine.optimizer_steps_completed, 1)

            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "pending_optimizer_step_learning_rates")
            self.assertEqual(manifest["first_optimizer_step"]["status"], "pass")
            self.assertEqual(manifest["first_optimizer_step"]["completed_step"], 1)
            self.assertEqual(manifest["first_optimizer_step"]["frozen"]["changed_count"], 0)
            self.assertEqual(manifest["first_optimizer_step"]["frozen"]["l2"], 0.0)
            self.assertEqual(manifest["first_optimizer_step"]["frozen"]["max_abs"], 0.0)
            self.assertGreater(manifest["first_optimizer_step"]["trainable"]["changed_count"], 0)
            self.assertGreater(manifest["first_optimizer_step"]["trainable"]["l2"], 0.0)
            self.assertGreater(manifest["first_optimizer_step"]["trainable"]["max_abs"], 0.0)
            self.assertEqual(
                manifest["optimizer_step_learning_rates"],
                {
                    "required_steps": 2,
                    "status": "pending",
                    "steps": [{"completed_step": 1, "learning_rates": [0.05]}],
                },
            )
            named_parameters = dict(model.named_parameters())
            self.assertTrue(all(torch.equal(named_parameters[name], value) for name, value in frozen_before.items()))

            optimizer.zero_grad()
            model(input_ids).square().mean().backward()
            engine.optimizer_step()
            self.assertEqual(engine.optimizer_steps_completed, 2)
            final_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(final_manifest["status"], "pass")
            self.assertEqual(final_manifest["first_optimizer_step"], manifest["first_optimizer_step"])
            self.assertEqual(
                final_manifest["optimizer_step_learning_rates"],
                {
                    "required_steps": 2,
                    "status": "pass",
                    "steps": [
                        {"completed_step": 1, "learning_rates": [0.05]},
                        {"completed_step": 2, "learning_rates": [0.05]},
                    ],
                },
            )

    def test_engine_scheduler_records_first_two_optimizer_step_learning_rates(self):
        with tempfile.TemporaryDirectory() as raw_dir:
            torch.manual_seed(9)
            model = _TinyQwen35ValueModel()
            manifest_path = Path(raw_dir) / "manifest.json"
            runtime = _configure(model, manifest_path)
            optimizer = torch.optim.SGD(
                [parameter for parameter in model.parameters() if parameter.requires_grad],
                lr=5e-6,
            )
            finalize_optimizer_membership(runtime, model, optimizer)
            engine = _build_engine(model, optimizer, runtime)
            engine.rank = 0
            engine.optimizer_config = SimpleNamespace(
                clip_grad=1.0,
                total_training_steps=100,
                lr_warmup_steps=10,
                lr_warmup_steps_ratio=0.0,
                lr_scheduler_type="constant",
                min_lr_ratio=0.0,
                num_cycles=0.5,
                zero_indexed_step=False,
            )
            engine.lr_scheduler = engine._build_lr_scheduler(optimizer)

            input_ids = torch.tensor([[1, 2, 3], [4, 5, 6]])
            observed_learning_rates = []
            for _ in range(2):
                observed_learning_rates.append(optimizer.param_groups[0]["lr"])
                optimizer.zero_grad()
                model(input_ids).square().mean().backward()
                engine.optimizer_step()
                engine.lr_scheduler_step()

            self.assertEqual(len(observed_learning_rates), 2)
            self.assertAlmostEqual(observed_learning_rates[0], 5e-7)
            self.assertAlmostEqual(observed_learning_rates[1], 1e-6)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["schema"], "verl_critic_parameter_freeze_v3")
            self.assertEqual(manifest["status"], "pass")
            lr_evidence = manifest["optimizer_step_learning_rates"]
            self.assertEqual(lr_evidence["required_steps"], 2)
            self.assertEqual(lr_evidence["status"], "pass")
            self.assertEqual([step["completed_step"] for step in lr_evidence["steps"]], [1, 2])
            self.assertAlmostEqual(lr_evidence["steps"][0]["learning_rates"][0], 5e-7)
            self.assertAlmostEqual(lr_evidence["steps"][1]["learning_rates"][0], 1e-6)

    def test_exact_qwen35_tied_lm_head_is_recorded_as_trainable_embedding_alias(self):
        with tempfile.TemporaryDirectory() as raw_dir:
            model = AutoModelForCausalLMWithValueHead(tie_lm_head=True)
            manifest_path = Path(raw_dir) / "parameter-freeze.json"
            runtime = _configure(model, manifest_path)
            manifest = runtime.manifest

            self.assertEqual(
                manifest["parameter_aliases"],
                [
                    {
                        "canonical_name": (
                            "pretrained_model.model.language_model.embed_tokens.weight"
                        ),
                        "canonical_category": "embedding",
                        "alias_name": "pretrained_model.lm_head.weight",
                        "alias_role": "tied_lm_head",
                        "requires_grad": True,
                    }
                ],
            )
            self.assertEqual(
                manifest["categories"]["lm_head_auxiliary"]["parameter_count"], 0
            )
            self.assertTrue(model.pretrained_model.lm_head.weight.requires_grad)
            self.assertIs(
                model.pretrained_model.lm_head.weight,
                model.pretrained_model.model.language_model.embed_tokens.weight,
            )

    def test_policy_rejects_unknown_parameters_without_partial_mutation(self):
        with tempfile.TemporaryDirectory() as raw_dir:
            model = _TinyQwen35ValueModel(include_unknown=True)
            requires_grad_before = {name: parameter.requires_grad for name, parameter in model.named_parameters()}
            manifest_path = Path(raw_dir) / "manifest.json"
            with self.assertRaisesRegex(ParameterFreezePolicyError, "unclassified parameters"):
                _configure(model, manifest_path)
            self.assertEqual(
                {name: parameter.requires_grad for name, parameter in model.named_parameters()},
                requires_grad_before,
            )
            self.assertFalse(manifest_path.exists())

    def test_policy_rejects_visual_and_lm_head_outside_exact_trl_roots_without_partial_mutation(self):
        with tempfile.TemporaryDirectory() as raw_dir:
            model = _TinyQwen35ValueModel(include_retained_multimodal_heads=True)
            requires_grad_before = {name: parameter.requires_grad for name, parameter in model.named_parameters()}
            manifest_path = Path(raw_dir) / "manifest.json"
            with self.assertRaisesRegex(ParameterFreezePolicyError, "forbidden parameter subtrees"):
                _configure(model, manifest_path)
            self.assertEqual(
                {name: parameter.requires_grad for name, parameter in model.named_parameters()},
                requires_grad_before,
            )
            self.assertFalse(manifest_path.exists())

    def test_policy_rejects_auxiliary_roots_on_non_exact_trl_wrapper_without_partial_mutation(self):
        with tempfile.TemporaryDirectory() as raw_dir:
            model = _WrongValueHeadWrapper()
            requires_grad_before = {name: parameter.requires_grad for name, parameter in model.named_parameters()}
            manifest_path = Path(raw_dir) / "manifest.json"
            with self.assertRaisesRegex(ParameterFreezePolicyError, "only supported on the exact"):
                _configure(model, manifest_path)
            self.assertEqual(
                {name: parameter.requires_grad for name, parameter in model.named_parameters()},
                requires_grad_before,
            )
            self.assertFalse(manifest_path.exists())

    def test_policy_rejects_relative_manifest_path_without_partial_mutation(self):
        model = _TinyQwen35ValueModel()
        requires_grad_before = {name: parameter.requires_grad for name, parameter in model.named_parameters()}
        with self.assertRaisesRegex(ParameterFreezePolicyError, "must be absolute"):
            _configure(model, Path("relative-parameter-freeze.json"))
        self.assertEqual(
            {name: parameter.requires_grad for name, parameter in model.named_parameters()},
            requires_grad_before,
        )

    def test_policy_rejects_symlink_manifest_path_without_partial_mutation(self):
        with tempfile.TemporaryDirectory() as raw_dir:
            model = _TinyQwen35ValueModel()
            requires_grad_before = {name: parameter.requires_grad for name, parameter in model.named_parameters()}
            target_path = Path(raw_dir) / "target.json"
            target_path.write_text("untouched\n", encoding="utf-8")
            manifest_path = Path(raw_dir) / "manifest.json"
            manifest_path.symlink_to(target_path)
            with self.assertRaisesRegex(ParameterFreezePolicyError, "symlink"):
                _configure(model, manifest_path)
            self.assertEqual(
                {name: parameter.requires_grad for name, parameter in model.named_parameters()},
                requires_grad_before,
            )
            self.assertEqual(target_path.read_text(encoding="utf-8"), "untouched\n")

    def test_policy_rejects_wrong_model_role(self):
        with tempfile.TemporaryDirectory() as raw_dir:
            with self.assertRaisesRegex(ParameterFreezePolicyError, "critic-only"):
                _configure(
                    _TinyQwen35ValueModel(),
                    Path(raw_dir) / "manifest.json",
                    model_role="language_model",
                )

    def test_policy_rejects_wrong_model_family(self):
        with tempfile.TemporaryDirectory() as raw_dir:
            with self.assertRaisesRegex(ParameterFreezePolicyError, "requires dense qwen3_5"):
                _configure(
                    _TinyQwen35ValueModel(),
                    Path(raw_dir) / "manifest.json",
                    model_type="qwen3_5_moe",
                )

    def test_optimizer_membership_must_exclude_every_frozen_parameter(self):
        with tempfile.TemporaryDirectory() as raw_dir:
            model = _TinyQwen35ValueModel()
            runtime = _configure(model, Path(raw_dir) / "manifest.json")
            optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
            with self.assertRaisesRegex(ParameterFreezePolicyError, "exactly match"):
                finalize_optimizer_membership(runtime, model, optimizer)

    def test_non_finite_gradient_does_not_increment_completed_step(self):
        with tempfile.TemporaryDirectory() as raw_dir:
            model = _TinyQwen35ValueModel()
            manifest_path = Path(raw_dir) / "manifest.json"
            runtime = _configure(model, manifest_path)
            optimizer = torch.optim.SGD(
                [parameter for parameter in model.parameters() if parameter.requires_grad],
                lr=0.1,
            )
            finalize_optimizer_membership(runtime, model, optimizer)
            engine = _build_engine(model, optimizer, runtime)

            for parameter in model.parameters():
                if parameter.requires_grad:
                    parameter.grad = torch.full_like(parameter, float("inf"))
            grad_norm = engine.optimizer_step()

            self.assertEqual(grad_norm, float("inf"))
            self.assertEqual(engine.optimizer_steps_completed, 0)
            self.assertEqual(runtime.optimizer_steps_completed, 0)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "pending_first_optimizer_step")
            self.assertEqual(manifest["first_optimizer_step"]["completed_step"], 0)

    def test_scaler_skip_does_not_increment_completed_step(self):
        class SkippingScaler:
            def unscale_(self, optimizer):
                return None

            def step(self, optimizer):
                return None

            def update(self):
                return None

        with tempfile.TemporaryDirectory() as raw_dir:
            model = _TinyQwen35ValueModel()
            manifest_path = Path(raw_dir) / "manifest.json"
            runtime = _configure(model, manifest_path)
            optimizer = torch.optim.SGD(
                [parameter for parameter in model.parameters() if parameter.requires_grad],
                lr=0.1,
            )
            finalize_optimizer_membership(runtime, model, optimizer)
            engine = _build_engine(model, optimizer, runtime)
            engine.scaler = SkippingScaler()
            trainable_before = {
                name: parameter.detach().clone()
                for name, parameter in model.named_parameters()
                if parameter.requires_grad
            }

            model(torch.tensor([[1, 2, 3], [4, 5, 6]])).square().mean().backward()
            grad_norm = engine.optimizer_step()

            self.assertTrue(torch.isfinite(torch.tensor(grad_norm)))
            self.assertEqual(engine.optimizer_steps_completed, 0)
            self.assertEqual(runtime.optimizer_steps_completed, 0)
            named_parameters = dict(model.named_parameters())
            self.assertTrue(all(torch.equal(named_parameters[name], value) for name, value in trainable_before.items()))
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "pending_first_optimizer_step")
            self.assertEqual(manifest["first_optimizer_step"]["completed_step"], 0)

    def test_fsdp2_cpu_membership_and_first_step_probe(self):
        if dist.is_initialized():
            self.skipTest("test requires ownership of the default process group")

        with tempfile.TemporaryDirectory() as raw_dir:
            rendezvous_path = Path(raw_dir) / "gloo-rendezvous"
            dist.init_process_group(
                "gloo",
                init_method=f"file://{rendezvous_path}",
                rank=0,
                world_size=1,
            )
            try:
                torch.manual_seed(11)
                model = _TinyQwen35ValueModel()
                manifest_path = Path(raw_dir) / "manifest.json"
                runtime = _configure(model, manifest_path)
                fully_shard(model, mesh=init_device_mesh("cpu", (1,)))
                optimizer = torch.optim.AdamW(
                    [parameter for parameter in model.parameters() if parameter.requires_grad],
                    lr=0.05,
                    weight_decay=0.0,
                )
                finalize_optimizer_membership(runtime, model, optimizer)

                model(torch.tensor([[1, 2, 3], [4, 5, 6]])).square().mean().backward()
                probe = begin_first_optimizer_step_probe(runtime, model)
                optimizer.step()
                runtime.optimizer_steps_completed += 1
                finish_first_optimizer_step_probe(
                    runtime,
                    probe,
                    completed_step=runtime.optimizer_steps_completed,
                    learning_rates=[float(group["lr"]) for group in optimizer.param_groups],
                )

                self.assertEqual(runtime.optimizer_steps_completed, 1)
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                self.assertEqual(manifest["status"], "pending_optimizer_step_learning_rates")
                self.assertTrue(manifest["optimizer"]["membership_exact"])
                self.assertEqual(manifest["first_optimizer_step"]["frozen"]["changed_count"], 0)
                self.assertGreater(manifest["first_optimizer_step"]["trainable"]["changed_count"], 0)
            finally:
                dist.destroy_process_group()

    def test_fsdp1_mixed_freeze_gate_runs_before_model_construction(self):
        engine = object.__new__(FSDPEngine)
        engine.model_config = SimpleNamespace(parameter_freeze_policy=QWEN35_DENSE_TOKEN_MIXERS_POLICY)
        engine.engine_config = SimpleNamespace(forward_only=False, strategy="fsdp", use_orig_params=False)
        engine._is_lora = False
        engine._qat_enabled = False
        engine._build_module = lambda: self.fail("model construction must not run before the FSDP1 gate")

        with self.assertRaisesRegex(ParameterFreezePolicyError, "use_orig_params=true"):
            engine._build_model_optimizer()


if __name__ == "__main__":
    unittest.main()
