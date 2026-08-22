# Copyright 2026 The verl team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Source-level regressions for the native fully-async PPO critic path.

These tests intentionally use only the Python standard library. They guard two
wiring failures before an expensive distributed runtime is started.
"""

from __future__ import annotations

import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


def _method_source(path: Path, class_name: str, method_name: str) -> str:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) and child.name == method_name:
                    return ast.get_source_segment(source, child) or ""
    raise AssertionError(f"missing {class_name}.{method_name} in {path}")


class TestFullyAsyncCriticCompatibility(unittest.TestCase):
    def test_separation_critic_reads_the_current_engine_owner(self) -> None:
        method = _method_source(
            ROOT / "verl/experimental/separation/ray_trainer.py",
            "SeparateRayPPOTrainer",
            "_create_critic_class",
        )

        self.assertIn("self.orig_critic_cfg.engine", method)
        self.assertNotIn("self.orig_critic_cfg.model.fsdp_config", method)
        self.assertNotIn('self.orig_critic_cfg.strategy == "fsdp"', method)

    def test_fully_async_critic_uses_training_worker_lifecycle(self) -> None:
        method = _method_source(
            ROOT / "verl/experimental/fully_async_policy/fully_async_trainer.py",
            "FullyAsyncTrainer",
            "_init_models",
        )

        self.assertIn("self.critic_wg.reset()", method)
        self.assertIn("partial(value_loss, config=self.orig_critic_cfg)", method)
        self.assertIn("self.critic_wg.set_loss_fn(value_loss_)", method)
        self.assertNotIn("self.critic_wg.init_model()", method)


if __name__ == "__main__":
    unittest.main()
