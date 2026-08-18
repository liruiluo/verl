# Copyright 2026 The verl team.
# Licensed under the Apache License, Version 2.0.

"""Fail-closed resolved-config/runtime contract for AMG continuous async PPO.

This is a contract oracle, not a new veRL control plane. The eventual launcher
validator remains external to veRL core and must match these checks. In
particular, behavior-policy provenance is established by veRL's real
``calculate_log_probs + bypass_mode`` path and by comparing retained tensors;
a friendly receipt boolean or an inert compatibility field is not sufficient.
"""

from __future__ import annotations

import copy
import hashlib
import unittest
from pathlib import Path

import torch

from verl.trainer.ppo.core_algos import ADV_ESTIMATOR_REGISTRY, get_adv_estimator_fn, register_adv_est

ROOT = Path(__file__).resolve().parents[3]
ACTION_ESTIMATOR = "amg_action_axis_gae"


if ACTION_ESTIMATOR not in ADV_ESTIMATOR_REGISTRY:

    @register_adv_est(ACTION_ESTIMATOR)
    def _fixture_registered_action_estimator(**kwargs):
        """Registration witness only; numerical action-GAE has a separate oracle."""

        return kwargs


def _get(config: dict, path: str):
    value = config
    for part in path.split("."):
        if not isinstance(value, dict) or part not in value:
            raise ValueError(f"missing resolved config field: {path}")
        value = value[part]
    return value


def _validate_resolved_ppo_config(config: dict) -> None:
    if _get(config, "actor_rollout_ref.rollout.n") != 1:
        raise ValueError("actor_rollout_ref.rollout.n must equal 1")

    estimator = str(_get(config, "algorithm.adv_estimator"))
    if estimator in {"grpo", "gae"}:
        raise ValueError("registered action-axis estimator is required; GRPO/stock GAE is forbidden")
    try:
        get_adv_estimator_fn(estimator)
    except ValueError as exc:
        raise ValueError(f"registered action-axis estimator is required: {estimator!r}") from exc

    if _get(config, "critic.enable") is not True:
        raise ValueError("critic.enable must be true for a custom action-axis estimator")
    if _get(config, "actor_rollout_ref.rollout.calculate_log_probs") is not True:
        raise ValueError("actor_rollout_ref.rollout.calculate_log_probs must be true")

    correction = _get(config, "algorithm.rollout_correction")
    if not isinstance(correction, dict) or correction.get("bypass_mode") is not True:
        raise ValueError("algorithm.rollout_correction.bypass_mode must be true to use rollout behavior logprobs")
    if correction.get("loss_type") != "ppo_clip":
        raise ValueError("resolved rollout correction must dispatch clipped PPO, not REINFORCE")

    policy_loss = _get(config, "actor_rollout_ref.actor.policy_loss.loss_mode")
    if policy_loss != "bypass_mode":
        raise ValueError("resolved policy loss must be bypass_mode clipped PPO")
    if _get(config, "trainer.val_before_train") is not False:
        raise ValueError("trainer.val_before_train must be false")
    if _get(config, "trainer.test_freq") != -1:
        raise ValueError("trainer.test_freq must equal -1")
    if _get(config, "async_training.use_trainer_do_validate") is not False:
        raise ValueError("async_training.use_trainer_do_validate must be false")


def _masked_tensor_sha256(tensor: torch.Tensor, response_mask: torch.Tensor) -> str:
    selected = tensor.detach().cpu()[response_mask.detach().cpu().to(dtype=torch.bool)].contiguous()
    header = f"dtype={selected.dtype};shape={tuple(selected.shape)};".encode()
    return hashlib.sha256(header + selected.numpy().tobytes()).hexdigest()


def _validate_one_update_evidence(
    receipt: dict,
    *,
    rollout_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
) -> None:
    required_true = {"actor_initialized", "critic_initialized", "clipped_ppo_dispatched"}
    for key in required_true:
        if receipt.get(key) is not True:
            raise ValueError(f"one-update receipt must prove {key}=true")

    if rollout_log_probs.shape != old_log_probs.shape or rollout_log_probs.shape != response_mask.shape:
        raise ValueError("rollout_log_probs, old_log_probs, and response_mask must have identical shapes")
    mask = response_mask.to(dtype=torch.bool)
    compared_tokens = int(mask.sum().item())
    if compared_tokens <= 0:
        raise ValueError("at least one sampled response token is required for logprob provenance")

    rollout_selected = rollout_log_probs[mask]
    old_selected = old_log_probs[mask]
    if not torch.equal(rollout_selected, old_selected):
        max_abs_diff = float((rollout_selected.float() - old_selected.float()).abs().max().item())
        raise ValueError(f"runtime old_log_probs differ from rollout behavior logprobs: max_abs_diff={max_abs_diff}")

    rollout_digest = _masked_tensor_sha256(rollout_log_probs, mask)
    old_digest = _masked_tensor_sha256(old_log_probs, mask)
    if receipt.get("rollout_log_probs_sha256") != rollout_digest:
        raise ValueError("rollout logprob tensor digest is missing or stale")
    if receipt.get("old_log_probs_sha256") != old_digest:
        raise ValueError("old logprob tensor digest is missing or stale")
    if receipt.get("rollout_old_log_probs_compared_tokens") != compared_tokens:
        raise ValueError("compared response-token count is missing or stale")

    if receipt.get("policy_version_min") is None or receipt.get("policy_version_max") is None:
        raise ValueError("policy-version evidence is required")
    if not float(receipt.get("actor_parameter_delta_l2", 0.0)) > 0.0:
        raise ValueError("actor parameter delta must be nonzero")
    if not float(receipt.get("critic_parameter_delta_l2", 0.0)) > 0.0:
        raise ValueError("critic parameter delta must be nonzero")
    if int(receipt.get("validation_rpc_count", -1)) != 0:
        raise ValueError("validation RPC must remain zero before endpoint")
    if int(receipt.get("validation_metric_count", -1)) != 0:
        raise ValueError("validation metrics must remain zero before endpoint")


def _valid_config() -> dict:
    return {
        # top-level rollout.n is deliberately 8 to prove it is not the live
        # legacy fully-async repeat namespace.
        "rollout": {"n": 8, "nnodes": 1, "n_gpus_per_node": 4},
        "actor_rollout_ref": {
            "rollout": {"n": 1, "calculate_log_probs": True},
            "actor": {
                # Retained for upstream config compatibility, but the validator
                # does not mistake this field for the real bypass control.
                "use_rollout_log_probs": True,
                "policy_loss": {"loss_mode": "bypass_mode"},
            },
        },
        "algorithm": {
            "adv_estimator": ACTION_ESTIMATOR,
            "rollout_correction": {"bypass_mode": True, "loss_type": "ppo_clip"},
        },
        "critic": {"enable": True},
        "trainer": {"val_before_train": False, "test_freq": -1},
        "async_training": {"use_trainer_do_validate": False},
    }


def _valid_tensors() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rollout = torch.tensor([[-0.1, -0.2, 99.0], [-0.3, 88.0, 77.0]], dtype=torch.float32)
    old = rollout.clone()
    response_mask = torch.tensor([[1, 1, 0], [1, 0, 0]], dtype=torch.bool)
    return rollout, old, response_mask


def _valid_receipt(
    rollout_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
) -> dict:
    return {
        "actor_initialized": True,
        "critic_initialized": True,
        "clipped_ppo_dispatched": True,
        "rollout_log_probs_sha256": _masked_tensor_sha256(rollout_log_probs, response_mask),
        "old_log_probs_sha256": _masked_tensor_sha256(old_log_probs, response_mask),
        "rollout_old_log_probs_compared_tokens": int(response_mask.sum().item()),
        "policy_version_min": 0,
        "policy_version_max": 1,
        "actor_parameter_delta_l2": 1e-4,
        "critic_parameter_delta_l2": 2e-4,
        "validation_rpc_count": 0,
        "validation_metric_count": 0,
    }


class TestResolvedAMGPPOConfig(unittest.TestCase):
    def test_positive_contract_uses_live_namespace_registered_estimator_and_clipped_ppo(self) -> None:
        _validate_resolved_ppo_config(_valid_config())
        rollout, old, mask = _valid_tensors()
        _validate_one_update_evidence(
            _valid_receipt(rollout, old, mask),
            rollout_log_probs=rollout,
            old_log_probs=old,
            response_mask=mask,
        )

    def test_top_level_rollout_n_cannot_replace_live_repeat_namespace(self) -> None:
        config = _valid_config()
        del config["actor_rollout_ref"]["rollout"]["n"]
        config["rollout"]["n"] = 1
        with self.assertRaisesRegex(ValueError, "actor_rollout_ref.rollout.n"):
            _validate_resolved_ppo_config(config)

    def test_non_ppo_or_missing_runtime_controls_fail_closed(self) -> None:
        mutations = [
            ("actor_rollout_ref.rollout.n", 8, "must equal 1"),
            ("algorithm.adv_estimator", "grpo", "action-axis"),
            ("algorithm.adv_estimator", "unregistered_action_axis", "registered action-axis"),
            ("critic.enable", False, "critic.enable"),
            ("actor_rollout_ref.rollout.calculate_log_probs", False, "calculate_log_probs"),
            ("algorithm.rollout_correction.bypass_mode", False, "bypass_mode"),
            ("algorithm.rollout_correction.loss_type", "reinforce", "clipped PPO"),
            ("actor_rollout_ref.actor.policy_loss.loss_mode", "vanilla", "bypass_mode"),
            ("trainer.val_before_train", True, "val_before_train"),
            ("trainer.test_freq", 10, "test_freq"),
            ("async_training.use_trainer_do_validate", True, "use_trainer_do_validate"),
        ]
        for path, value, message in mutations:
            with self.subTest(path=path, value=value):
                config = copy.deepcopy(_valid_config())
                node = config
                parts = path.split(".")
                for part in parts[:-1]:
                    node = node[part]
                node[parts[-1]] = value
                with self.assertRaisesRegex(ValueError, message):
                    _validate_resolved_ppo_config(config)

    def test_recomputed_or_mutated_old_logprobs_fail_tensor_comparison(self) -> None:
        rollout, old, mask = _valid_tensors()
        receipt = _valid_receipt(rollout, old, mask)
        old[0, 1] += 0.25
        # Even stale/equal receipt booleans could not hide this tensor mismatch.
        with self.assertRaisesRegex(ValueError, "differ from rollout behavior"):
            _validate_one_update_evidence(
                receipt,
                rollout_log_probs=rollout,
                old_log_probs=old,
                response_mask=mask,
            )

    def test_runtime_evidence_requires_digests_versions_deltas_and_no_eval(self) -> None:
        rollout, old, mask = _valid_tensors()
        base = _valid_receipt(rollout, old, mask)
        for field, bad_value in (
            ("clipped_ppo_dispatched", False),
            ("rollout_log_probs_sha256", "bad"),
            ("old_log_probs_sha256", "bad"),
            ("rollout_old_log_probs_compared_tokens", 0),
            ("policy_version_min", None),
            ("actor_parameter_delta_l2", 0.0),
            ("critic_parameter_delta_l2", 0.0),
            ("validation_rpc_count", 1),
            ("validation_metric_count", 1),
        ):
            with self.subTest(field=field):
                receipt = copy.deepcopy(base)
                receipt[field] = bad_value
                with self.assertRaises(ValueError):
                    _validate_one_update_evidence(
                        receipt,
                        rollout_log_probs=rollout,
                        old_log_probs=old,
                        response_mask=mask,
                    )


class TestPinnedSourceNamespaceAndRecipeBoundary(unittest.TestCase):
    def test_detach_uses_actor_rollout_ref_repeat_namespace(self) -> None:
        source = (ROOT / "verl/experimental/fully_async_policy/detach_utils.py").read_text()
        self.assertIn("config.actor_rollout_ref.rollout.n", source)

    def test_main_only_copies_rollout_resource_fields_not_top_level_n(self) -> None:
        source = (ROOT / "verl/experimental/fully_async_policy/fully_async_main.py").read_text()
        self.assertIn("rollout.nnodes = config.rollout.nnodes", source)
        self.assertIn("rollout.n_gpus_per_node = config.rollout.n_gpus_per_node", source)
        self.assertNotIn("rollout.n = config.rollout.n", source)

    def test_bypass_mode_is_the_actual_old_logprob_substitution_path(self) -> None:
        trainer = (ROOT / "verl/experimental/separation/ray_trainer.py").read_text()
        helper = (ROOT / "verl/trainer/ppo/rollout_corr_helper.py").read_text()
        self.assertIn('rollout_corr_config.get("bypass_mode", False)', trainer)
        self.assertIn('batch.batch["old_log_probs"] = batch.batch["rollout_log_probs"]', helper)

    def test_official_qwen35_dynamic_recipe_is_grpo_topology_donor_only(self) -> None:
        source = (
            ROOT / "verl/experimental/fully_async_policy/shell/run_qwen35_35b_a3b_math_dynamic_megatron.sh"
        ).read_text()
        self.assertIn("GRPO", source)
        self.assertIn("algorithm.adv_estimator=grpo", source)
        self.assertIn("actor_rollout_ref.rollout.n=8", source)


if __name__ == "__main__":
    unittest.main()
