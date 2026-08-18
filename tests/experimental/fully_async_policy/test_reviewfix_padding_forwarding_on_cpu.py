# Copyright 2026 The verl team.
# Licensed under the Apache License, Version 2.0.

"""Focused regressions for zero-loss padding and custom-estimator forwarding."""

from __future__ import annotations

from types import SimpleNamespace
from unittest import TestCase, mock

import numpy as np
import torch
from omegaconf import OmegaConf
from tensordict import TensorDict

from verl.experimental.fully_async_policy.fully_async_trainer import FullyAsyncTrainer
from verl.experimental.separation.ray_trainer import SeparateRayPPOTrainer
from verl.protocol import DataProto
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.padding_utils import upsample_dataproto_to_divisible_size
from verl.trainer.ppo.ray_trainer import RayPPOTrainer, compute_advantage

ESTIMATOR_NAME = "test_reviewfix_kwargs_estimator"
REAL_ROWS = 1001


def _local_actor_class(actor_class):
    return actor_class.__ray_metadata__.modified_class


def _variable_row_batch() -> DataProto:
    prompts = torch.arange(REAL_ROWS, dtype=torch.long).unsqueeze(1).repeat(1, 2)
    responses = prompts + 1
    batch = TensorDict(
        {
            "prompts": prompts,
            "responses": responses,
            "input_ids": torch.cat([prompts, responses], dim=1),
            "attention_mask": torch.ones((REAL_ROWS, 4), dtype=torch.long),
            "position_ids": torch.arange(4, dtype=torch.long).repeat(REAL_ROWS, 1),
            "response_mask": torch.ones((REAL_ROWS, 2), dtype=torch.long),
            "loss_mask": torch.ones((REAL_ROWS, 2), dtype=torch.long),
            "token_level_rewards": torch.ones((REAL_ROWS, 2), dtype=torch.float32),
        },
        batch_size=REAL_ROWS,
    )
    group_sizes = [16] * 41 + [15] * 23
    uids = [f"episode-{group}" for group, size in enumerate(group_sizes) for _ in range(size)]
    assert len(uids) == REAL_ROWS
    return DataProto(
        batch=batch,
        non_tensor_batch={
            "uid": np.array(uids, dtype=object),
            "is_padding": np.zeros(REAL_ROWS, dtype=bool),
            "multi_modal_inputs": np.array(
                [{"pixel_values": torch.tensor([float(index)])} for index in range(REAL_ROWS)], dtype=object
            ),
        },
    )


def _near_boundary_batch() -> DataProto:
    """1,023 action rows from 64 trajectories: only one pad row is needed."""
    rows = 1023
    group_sizes = [16] * 63 + [15]
    group_ids = [group for group, size in enumerate(group_sizes) for _ in range(size)]
    prompts = torch.tensor(group_ids, dtype=torch.long).unsqueeze(1).repeat(1, 2)
    responses = prompts + 1
    batch = TensorDict(
        {
            "prompts": prompts,
            "responses": responses,
            "input_ids": torch.cat([prompts, responses], dim=1),
            "attention_mask": torch.ones((rows, 4), dtype=torch.long),
            "position_ids": torch.arange(4, dtype=torch.long).repeat(rows, 1),
            "response_mask": torch.ones((rows, 2), dtype=torch.long),
            "loss_mask": torch.ones((rows, 2), dtype=torch.long),
            "token_level_rewards": torch.ones((rows, 2), dtype=torch.float32),
        },
        batch_size=rows,
    )
    uids = [f"episode-{group}" for group, size in enumerate(group_sizes) for _ in range(size)]
    assert len(uids) == rows
    return DataProto(
        batch=batch,
        non_tensor_batch={
            "uid": np.array(uids, dtype=object),
            "is_padding": np.zeros(rows, dtype=bool),
            "multi_modal_inputs": np.array(
                [{"pixel_values": torch.tensor([float(index)])} for index in range(rows)], dtype=object
            ),
        },
    )


class TestPaddingNativeCompatibility(TestCase):
    def test_near_boundary_padding_reuses_a_prefix_group_without_creating_eight_rows(self) -> None:
        padded = upsample_dataproto_to_divisible_size(
            _near_boundary_batch(),
            batch_multiple=1024,
            eos_token_id=0,
            uid_group_divisor=8,
        )

        is_padding = padded.non_tensor_batch["is_padding"].astype(bool)
        self.assertEqual(len(padded), 1024)
        self.assertEqual(int(is_padding.sum()), 1)
        self.assertEqual(len(set(padded.non_tensor_batch["uid"])), 64)

        pad_index = int(np.flatnonzero(is_padding)[0])
        pad_uid = padded.non_tensor_batch["uid"][pad_index]
        real_index = int(np.flatnonzero((padded.non_tensor_batch["uid"] == pad_uid) & ~is_padding)[0])
        self.assertTrue(torch.equal(padded.batch["prompts"][pad_index], padded.batch["prompts"][real_index]))
        self.assertEqual(int(padded.batch["response_mask"][pad_index].sum()), 0)

        trainer = object.__new__(RayPPOTrainer)
        trainer.use_prefix_grouper = True
        trainer.actor_rollout_wg = SimpleNamespace(_dispatch_info={"actor": list(range(8))})
        trainer._balance_batch(padded, metrics={})

        uid_to_rank = {}
        for rank, chunk in enumerate(padded.chunk(8)):
            self.assertEqual(len(chunk), 128)
            for uid in set(chunk.non_tensor_batch["uid"]):
                self.assertNotIn(uid, uid_to_rank)
                uid_to_rank[uid] = rank

    def test_multimodal_payloads_are_cleared_and_prefix_groups_remain_dp_divisible(self) -> None:
        padded = upsample_dataproto_to_divisible_size(
            _variable_row_batch(),
            batch_multiple=1024,
            eos_token_id=0,
            uid_group_divisor=8,
        )

        is_padding = padded.non_tensor_batch["is_padding"].astype(bool)
        self.assertEqual(int(is_padding.sum()), 23)
        self.assertTrue(all(payload == {} for payload in padded.non_tensor_batch["multi_modal_inputs"][is_padding]))
        self.assertEqual(len(set(padded.non_tensor_batch["uid"])), 72)
        self.assertEqual(len(set(padded.non_tensor_batch["uid"][is_padding])), 8)

        trainer = object.__new__(RayPPOTrainer)
        trainer.use_prefix_grouper = True
        trainer.actor_rollout_wg = SimpleNamespace(_dispatch_info={"actor": list(range(8))})
        trainer._balance_batch(padded, metrics={})

        uid_to_rank = {}
        for rank, chunk in enumerate(padded.chunk(8)):
            self.assertEqual(len(chunk), 128)
            for uid in set(chunk.non_tensor_batch["uid"]):
                self.assertNotIn(uid, uid_to_rank)
                uid_to_rank[uid] = rank

    def test_realigning_an_existing_lcm_batch_keeps_less_than_one_role_minibatch_of_padding(self) -> None:
        shared = upsample_dataproto_to_divisible_size(_variable_row_batch(), batch_multiple=1536, eos_token_id=0)
        critic = upsample_dataproto_to_divisible_size(shared, batch_multiple=384, eos_token_id=0)
        actor = upsample_dataproto_to_divisible_size(shared, batch_multiple=512, eos_token_id=0)

        self.assertEqual(len(critic), 1152)
        self.assertLess(int(critic.non_tensor_batch["is_padding"].sum()), 384)
        self.assertEqual(len(actor), 1024)
        self.assertLess(int(actor.non_tensor_batch["is_padding"].sum()), 512)

    def test_critic_warmup_omits_actor_minibatch_and_role_updates_preserve_shared_alignment(self) -> None:
        trainer = object.__new__(_local_actor_class(FullyAsyncTrainer))
        trainer.use_critic = True
        trainer.global_steps = 0
        trainer.config = SimpleNamespace(
            trainer=SimpleNamespace(critic_warmup=2, balance_batch=False),
            actor_rollout_ref=SimpleNamespace(
                actor=SimpleNamespace(ppo_mini_batch_size=512),
                rollout=SimpleNamespace(n=1),
            ),
            critic=SimpleNamespace(ppo_mini_batch_size=384),
        )
        trainer.actor_rollout_wg = SimpleNamespace(_dispatch_info={"actor": list(range(8))})
        trainer.critic_wg = SimpleNamespace(_dispatch_info={"train": list(range(8))})
        trainer.tokenizer = SimpleNamespace(eos_token_id=0)

        self.assertEqual(trainer._get_required_batch_multiple(), 384)
        trainer.global_steps = 2
        self.assertEqual(trainer._get_required_batch_multiple(), 1536)

        shared = upsample_dataproto_to_divisible_size(_variable_row_batch(), batch_multiple=1536, eos_token_id=0)
        with (
            mock.patch.object(SeparateRayPPOTrainer, "_update_critic", return_value="critic") as update_critic,
            mock.patch.object(SeparateRayPPOTrainer, "_update_actor", return_value="actor") as update_actor,
        ):
            self.assertEqual(trainer._update_critic(shared), "critic")
            self.assertEqual(trainer._update_actor(shared), "actor")

        critic_batch = update_critic.call_args.args[0]
        actor_batch = update_actor.call_args.args[0]
        self.assertIs(critic_batch, shared)
        self.assertIs(actor_batch, shared)
        self.assertEqual(len(critic_batch), 1536)
        self.assertEqual(len(actor_batch), 1536)


class TestKwargsAdvantageCompatibility(TestCase):
    def tearDown(self) -> None:
        core_algos.ADV_ESTIMATOR_REGISTRY.pop(ESTIMATOR_NAME, None)

    def test_var_kwargs_estimator_receives_legacy_guaranteed_fields(self) -> None:
        seen = {}

        @core_algos.register_adv_est(ESTIMATOR_NAME)
        def _estimator(**kwargs):
            seen.update(kwargs)
            return torch.zeros_like(kwargs["token_level_rewards"]), torch.zeros_like(kwargs["token_level_rewards"])

        tensor_batch = TensorDict(
            {
                "token_level_rewards": torch.ones((2, 2), dtype=torch.float32),
                "response_mask": torch.ones((2, 2), dtype=torch.long),
                "unrelated_tensor": torch.ones((2, 2), dtype=torch.float32),
            },
            batch_size=2,
        )
        data = DataProto(
            batch=tensor_batch,
            non_tensor_batch={"uid": np.array(["a", "b"], dtype=object), "unrelated_metadata": np.array([1, 2])},
        )
        config = OmegaConf.create({"name": "kwargs-compat"})

        compute_advantage(data, adv_estimator=ESTIMATOR_NAME, config=config)

        self.assertIs(seen["token_level_rewards"], tensor_batch["token_level_rewards"])
        self.assertIs(seen["response_mask"], tensor_batch["response_mask"])
        self.assertIs(seen["config"], config)
        self.assertEqual(seen["index"].tolist(), ["a", "b"])
        self.assertNotIn("unrelated_tensor", seen)
        self.assertNotIn("unrelated_metadata", seen)


if __name__ == "__main__":
    import unittest

    unittest.main()
