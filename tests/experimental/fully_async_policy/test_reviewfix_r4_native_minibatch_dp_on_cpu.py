# Copyright 2026 The verl team.
# Licensed under the Apache License, Version 2.0.

"""Regression for native global-mini-batch/data-parallel validation."""

from __future__ import annotations

from types import SimpleNamespace
from unittest import TestCase

from verl.experimental.fully_async_policy.fully_async_trainer import FullyAsyncTrainer


def _local_actor_class(actor_class):
    return actor_class.__ray_metadata__.modified_class


def _worker_group(role: str, dp_size: int):
    return SimpleNamespace(_dispatch_info={role: list(range(dp_size))})


def _trainer(
    *, actor_mini_batch: int, actor_dp: int, critic_mini_batch: int = 4, critic_dp: int = 2, use_critic: bool = False
):
    trainer = object.__new__(_local_actor_class(FullyAsyncTrainer))
    trainer.config = SimpleNamespace(
        actor_rollout_ref=SimpleNamespace(
            actor=SimpleNamespace(ppo_mini_batch_size=actor_mini_batch),
            rollout=SimpleNamespace(n=1),
        ),
        critic=SimpleNamespace(ppo_mini_batch_size=critic_mini_batch),
        trainer=SimpleNamespace(critic_warmup=0),
    )
    trainer.global_steps = 0
    trainer.use_critic = use_critic
    trainer.actor_rollout_wg = _worker_group("actor", actor_dp)
    trainer.critic_wg = _worker_group("train", critic_dp)
    return trainer


class TestNativeMiniBatchDPContract(TestCase):
    def test_actor_global_mini_batch_must_be_divisible_by_actor_dp(self):
        trainer = _trainer(actor_mini_batch=3, actor_dp=2)
        with self.assertRaisesRegex(ValueError, "actor.*mini-batch.*3.*DP.*2"):
            trainer._get_required_batch_multiple()

    def test_critic_global_mini_batch_must_be_divisible_by_critic_dp(self):
        trainer = _trainer(
            actor_mini_batch=4,
            actor_dp=2,
            critic_mini_batch=3,
            critic_dp=2,
            use_critic=True,
        )
        with self.assertRaisesRegex(ValueError, "critic.*mini-batch.*3.*DP.*2"):
            trainer._get_required_batch_multiple()

    def test_valid_role_contract_keeps_shared_lcm_alignment(self):
        trainer = _trainer(
            actor_mini_batch=6,
            actor_dp=2,
            critic_mini_batch=4,
            critic_dp=2,
            use_critic=True,
        )
        self.assertEqual(trainer._get_required_batch_multiple(), 12)


if __name__ == "__main__":
    import unittest

    unittest.main()
