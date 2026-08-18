# Copyright 2026 The verl team.
# Licensed under the Apache License, Version 2.0.

"""Ensure legacy fully-async reuses veRL's current unified critic contract."""

from __future__ import annotations

import ast
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SEPARATION = ROOT / "verl/experimental/separation/ray_trainer.py"
FULL_ASYNC = ROOT / "verl/experimental/fully_async_policy/fully_async_trainer.py"
V1_DONOR = ROOT / "verl/trainer/ppo/v1/trainer_base.py"


def _method_source(path: Path, class_name: str, method_name: str) -> str:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for child in node.body:
                if isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef) and child.name == method_name:
                    return ast.get_source_segment(source, child) or ""
    raise AssertionError(f"missing {class_name}.{method_name} in {path}")


class TestNativeFullyAsyncCriticWiring(unittest.TestCase):
    def test_separation_critic_uses_current_engine_owner_for_fsdp_and_fsdp2(self) -> None:
        candidate = _method_source(SEPARATION, "SeparateRayPPOTrainer", "_create_critic_class")
        donor = _method_source(V1_DONOR, "PPOTrainer", "_setup")

        # The current CriticConfig owns one selected engine. FSDPEngineConfig itself
        # validates both fsdp and fsdp2; no legacy model.fsdp_config branch is needed.
        self.assertIn("critic_cfg.engine", donor)
        self.assertIn("self.orig_critic_cfg.engine", candidate)
        self.assertNotIn("self.orig_critic_cfg.model.fsdp_config", candidate)
        self.assertNotIn('self.orig_critic_cfg.strategy == "fsdp"', candidate)
        self.assertIn("engine_config=engine_config", candidate)

    def test_fully_async_critic_uses_training_worker_reset_and_value_loss(self) -> None:
        candidate = _method_source(FULL_ASYNC, "FullyAsyncTrainer", "_init_models")
        donor = _method_source(V1_DONOR, "PPOTrainer", "_setup")

        self.assertIn("self.critic_wg.reset()", donor)
        self.assertIn("partial(value_loss, config=critic_cfg)", donor)
        self.assertIn("self.critic_wg.set_loss_fn(value_loss_)", donor)

        self.assertIn("self.critic_wg.reset()", candidate)
        self.assertIn("partial(value_loss, config=self.orig_critic_cfg)", candidate)
        self.assertIn("self.critic_wg.set_loss_fn(value_loss_)", candidate)
        self.assertNotIn("self.critic_wg.init_model()", candidate)


if __name__ == "__main__":
    unittest.main()
