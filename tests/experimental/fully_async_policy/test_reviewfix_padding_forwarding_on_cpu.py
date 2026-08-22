# Copyright 2026 The verl team.
# Licensed under the Apache License, Version 2.0.

"""Focused regressions for zero-loss padding and custom-estimator forwarding."""

from __future__ import annotations

from types import SimpleNamespace
from unittest import IsolatedAsyncioTestCase, TestCase, mock

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

    def test_shared_lcm_padding_rejects_different_actor_critic_step_counts(self) -> None:
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

        # During critic warmup the inactive actor contract must not enlarge the
        # batch. Once both roles are active, differing global mini-batches are
        # rejected instead of padding to their LCM and adding optimizer steps.
        self.assertEqual(trainer._get_required_batch_multiple(), 384)
        trainer.global_steps = 2
        with self.assertRaisesRegex(ValueError, "equal active actor/critic"):
            trainer._get_required_batch_multiple()

    def test_equal_shared_padding_preserves_optimizer_step_count_and_parameters(self) -> None:
        trainer = object.__new__(_local_actor_class(FullyAsyncTrainer))
        trainer.use_critic = True
        trainer.global_steps = 2
        trainer.config = SimpleNamespace(
            trainer=SimpleNamespace(critic_warmup=2, balance_batch=False),
            actor_rollout_ref=SimpleNamespace(
                actor=SimpleNamespace(ppo_mini_batch_size=512),
                rollout=SimpleNamespace(n=1),
            ),
            critic=SimpleNamespace(ppo_mini_batch_size=512),
        )
        trainer.actor_rollout_wg = SimpleNamespace(_dispatch_info={"actor": list(range(8))})
        trainer.critic_wg = SimpleNamespace(_dispatch_info={"train": list(range(8))})

        batch_multiple = trainer._get_required_batch_multiple()
        self.assertEqual(batch_multiple, 512)
        shared = upsample_dataproto_to_divisible_size(
            _variable_row_batch(), batch_multiple=batch_multiple, eos_token_id=0
        )
        expected_steps = (REAL_ROWS + batch_multiple - 1) // batch_multiple
        self.assertEqual(len(shared) // batch_multiple, expected_steps)

        with (
            mock.patch.object(SeparateRayPPOTrainer, "_update_critic", return_value="critic") as update_critic,
            mock.patch.object(SeparateRayPPOTrainer, "_update_actor", return_value="actor") as update_actor,
        ):
            self.assertEqual(trainer._update_critic(shared), "critic")
            self.assertEqual(trainer._update_actor(shared), "actor")
        self.assertIs(update_critic.call_args.args[0], shared)
        self.assertIs(update_actor.call_args.args[0], shared)

        real_values = torch.linspace(-1.0, 1.0, REAL_ROWS, dtype=torch.float64)
        padded_values = torch.cat([real_values, torch.zeros(len(shared) - REAL_ROWS, dtype=torch.float64)])
        padded_mask = torch.cat(
            [torch.ones(REAL_ROWS, dtype=torch.float64), torch.zeros(len(shared) - REAL_ROWS, dtype=torch.float64)]
        )

        def _optimize(values: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, int]:
            parameter = torch.nn.Parameter(torch.tensor(0.25, dtype=torch.float64))
            optimizer = torch.optim.AdamW([parameter], lr=0.01, weight_decay=0.1)
            step_count = 0
            for start in range(0, len(values), batch_multiple):
                chunk = values[start : start + batch_multiple]
                chunk_mask = mask[start : start + batch_multiple]
                loss = (((parameter - chunk) ** 2) * chunk_mask).sum() / chunk_mask.sum()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                step_count += 1
            return parameter.detach(), step_count

        unpadded_parameter, unpadded_steps = _optimize(real_values, torch.ones_like(real_values))
        padded_parameter, padded_steps = _optimize(padded_values, padded_mask)
        self.assertEqual(unpadded_steps, expected_steps)
        self.assertEqual(padded_steps, expected_steps)
        torch.testing.assert_close(padded_parameter, unpadded_parameter, rtol=0.0, atol=0.0)

    def test_synthetic_padding_rejects_seq_mean_loss(self) -> None:
        trainer = object.__new__(_local_actor_class(FullyAsyncTrainer))
        trainer.use_critic = True
        trainer.global_steps = 0
        trainer.config = SimpleNamespace(
            trainer=SimpleNamespace(critic_warmup=0),
            actor_rollout_ref=SimpleNamespace(actor=SimpleNamespace(loss_agg_mode="seq-mean-token-mean")),
            critic=SimpleNamespace(loss_agg_mode="token-mean"),
            algorithm={"use_kl_in_reward": False},
        )
        with self.assertRaisesRegex(ValueError, "actor loss_agg_mode=token-mean"):
            trainer._validate_synthetic_padding_contract(_variable_row_batch(), batch_multiple=512)

        trainer.config.actor_rollout_ref.actor.loss_agg_mode = "token-mean"
        trainer.config.critic.loss_agg_mode = "seq-mean-token-mean"
        with self.assertRaisesRegex(ValueError, "critic loss_agg_mode=token-mean"):
            trainer._validate_synthetic_padding_contract(_variable_row_batch(), batch_multiple=512)

    def test_synthetic_padding_rejects_adaptive_kl(self) -> None:
        trainer = object.__new__(_local_actor_class(FullyAsyncTrainer))
        trainer.use_critic = True
        trainer.global_steps = 0
        trainer.config = SimpleNamespace(
            trainer=SimpleNamespace(critic_warmup=0),
            actor_rollout_ref=SimpleNamespace(actor=SimpleNamespace(loss_agg_mode="token-mean")),
            critic=SimpleNamespace(loss_agg_mode="token-mean"),
            algorithm={"use_kl_in_reward": True},
        )
        with self.assertRaisesRegex(ValueError, "adaptive in-reward KL"):
            trainer._validate_synthetic_padding_contract(_variable_row_batch(), batch_multiple=512)

    def test_multi_output_distinct_prompts_are_not_prefix_shared(self) -> None:
        trainer = object.__new__(_local_actor_class(FullyAsyncTrainer))
        trainer.use_prefix_grouper = True
        batch = _variable_row_batch()[:2]
        batch.non_tensor_batch["uid"] = np.array(["trajectory-0", "trajectory-0"], dtype=object)
        with self.assertRaisesRegex(ValueError, "distinct multi-output action prompts"):
            trainer._validate_prefix_group_contract(batch)

    def test_variable_uid_groups_are_rejected(self) -> None:
        trainer = object.__new__(_local_actor_class(FullyAsyncTrainer))
        trainer.use_prefix_grouper = True
        batch = _variable_row_batch()[:3]
        batch.non_tensor_batch["uid"] = np.array(["trajectory-0", "trajectory-0", "trajectory-1"], dtype=object)
        with self.assertRaisesRegex(ValueError, "equal UID-group sizes"):
            trainer._validate_prefix_group_contract(batch)

    def test_metrics_and_rollout_dump_exclude_synthetic_rows_without_dropping_real_extras(self) -> None:
        trainer = object.__new__(_local_actor_class(FullyAsyncTrainer))
        padded = upsample_dataproto_to_divisible_size(
            _variable_row_batch(),
            batch_multiple=1024,
            eos_token_id=0,
        )
        original_extras = {
            "row_receipt": [f"receipt-{index}" for index in range(len(padded))],
            "aggregate": "kept-as-is",
        }
        trainer.reward_extra_infos_dict = original_extras

        seen = {}

        def _capture_dump(_self, batch):
            seen["dump_batch"] = batch
            seen["dump_extras"] = _self.reward_extra_infos_dict

        def _capture_metrics(_self, batch):
            seen["metrics_batch"] = batch

        with (
            mock.patch.object(SeparateRayPPOTrainer, "_fit_dump_data", autospec=True, side_effect=_capture_dump),
            mock.patch.object(
                SeparateRayPPOTrainer,
                "_fit_collect_metrics",
                autospec=True,
                side_effect=_capture_metrics,
            ),
        ):
            trainer._fit_dump_data(padded)
            trainer._fit_collect_metrics(padded)

        self.assertEqual(len(seen["dump_batch"]), REAL_ROWS)
        self.assertEqual(len(seen["metrics_batch"]), REAL_ROWS)
        self.assertEqual(seen["dump_extras"]["row_receipt"], original_extras["row_receipt"][:REAL_ROWS])
        self.assertEqual(seen["dump_extras"]["aggregate"], "kept-as-is")
        self.assertIs(trainer.reward_extra_infos_dict, original_extras)


class TestFullyAsyncMetricPublication(IsolatedAsyncioTestCase):
    async def test_current_cycle_metrics_are_logged_before_reset(self) -> None:
        trainer = object.__new__(_local_actor_class(FullyAsyncTrainer))
        trainer.config = SimpleNamespace(global_profiler=SimpleNamespace(steps=None))
        trainer.current_param_version = 0
        trainer.local_trigger_step = 1
        trainer.trigger_parameter_sync_step = 1
        trainer.global_steps = 1
        trainer.epoch = 0
        trainer.dynamic_schedule_enabled = False
        trainer.dynamic_resource_controller = SimpleNamespace()
        trainer.required_samples = 64
        trainer.message_queue_client = SimpleNamespace(get_queue_size_sync=lambda: 7)
        trainer.progress_bar = mock.Mock()

        events: list[str] = []

        class _Aggregator:
            def __init__(self) -> None:
                self.steps: list[dict] = []

            def add_step_metrics(self, *, metrics, sample_count, timestamp) -> None:
                events.append("add")
                self.steps.append(dict(metrics))

            def get_aggregated_metrics(self, **kwargs) -> dict:
                events.append("get")
                if not self.steps:
                    return {}
                return dict(self.steps[-1])

            def reset(self) -> None:
                events.append("reset")
                self.steps.clear()

        trainer.metrics_aggregator = _Aggregator()
        trainer.logger = SimpleNamespace(log=lambda *, data, step: events.append(("log", dict(data), step)))
        sample = object()
        trainer._fit_start_profile = mock.Mock()
        trainer._fit_stop_profile = mock.Mock()
        trainer._fit_generate = mock.AsyncMock(return_value=sample)
        for name in (
            "_fit_compute_reward",
            "_fit_compute_log_prob",
            "_fit_compute_ref_log_prob",
            "_fit_compute_critic",
            "_fit_compute_advantage",
            "_fit_update_critic",
            "_fit_update_actor",
        ):
            setattr(trainer, name, mock.Mock(side_effect=lambda value: value))
        trainer._fit_update_weights = mock.AsyncMock(
            return_value={"dynamic_resource/rollout_resource_utilization": 0.5}
        )
        trainer._fit_dump_data = mock.Mock()
        trainer._record_train_resource_utilization = mock.Mock()
        trainer._fit_validate = mock.AsyncMock()
        trainer._fit_save_checkpoint = mock.Mock()

        def _collect(_batch) -> None:
            trainer.metrics["actor/grad_norm"] = 3.5

        trainer._fit_collect_metrics = mock.Mock(side_effect=_collect)

        await trainer.fit_step()

        self.assertLess(events.index("add"), events.index("get"))
        self.assertLess(events.index("get"), events.index("reset"))
        logged = [event for event in events if isinstance(event, tuple) and event[0] == "log"]
        self.assertEqual(len(logged), 1)
        self.assertEqual(logged[0][1]["actor/grad_norm"], 3.5)
        self.assertEqual(logged[0][1]["dynamic_resource/mq_size"], 7)
        self.assertEqual(logged[0][2], 1)


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
