# Copyright 2026 The verl team.
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

"""Probe whether upstream require_batches already separates collection size.

AMG needs 64 complete episodes per learner collection while retaining a
512-row PPO mini-batch.  Before adding a new collection_size interface, this
contract exercises the native arithmetic with require_batches=0.125.
"""

from __future__ import annotations

import asyncio
import unittest
from types import SimpleNamespace
from unittest import IsolatedAsyncioTestCase, mock

from omegaconf import OmegaConf

from verl.experimental.fully_async_policy import fully_async_trainer as trainer_module
from verl.experimental.fully_async_policy.detach_utils import MetricsAggregator
from verl.experimental.fully_async_policy.dynamic_schedule.base import DynamicScheduleContext
from verl.experimental.fully_async_policy.dynamic_schedule.default_policy import DefaultDynamicSchedulePolicy
from verl.experimental.fully_async_policy.fully_async_rollouter import FullyAsyncRollouter
from verl.experimental.fully_async_policy.fully_async_trainer import FullyAsyncTrainer
from verl.protocol import DataProto

PPO_MINI_BATCH_SIZE = 512
REQUIRE_BATCHES = 0.125
EXPECTED_EPISODES = 64
SYNC_INTERVAL = 4


def _local_actor_class(actor_class):
    """Return Ray's wrapped implementation class for dependency-only unit probes."""
    return actor_class.__ray_metadata__.modified_class


def _fractional_config():
    return OmegaConf.create(
        {
            "async_training": {
                "require_batches": REQUIRE_BATCHES,
                "trigger_parameter_sync_step": SYNC_INTERVAL,
            },
            "actor_rollout_ref": {"actor": {"ppo_mini_batch_size": PPO_MINI_BATCH_SIZE}},
        }
    )


class TestFractionalRequireBatchesArithmetic(unittest.TestCase):
    def test_omegaconf_preserves_fraction_and_integral_collection_count(self) -> None:
        config = _fractional_config()
        self.assertEqual(config.async_training.require_batches, REQUIRE_BATCHES)
        required_samples = config.actor_rollout_ref.actor.ppo_mini_batch_size * config.async_training.require_batches
        self.assertEqual(required_samples, EXPECTED_EPISODES)
        self.assertEqual(int(required_samples), required_samples)

    def test_dynamic_schedule_uses_the_same_collection_unit(self) -> None:
        required_samples = PPO_MINI_BATCH_SIZE * REQUIRE_BATCHES
        ctx = DynamicScheduleContext(
            required_samples=required_samples,
            trigger_parameter_sync_step=SYNC_INTERVAL,
            total_generated_samples=0,
            expected_samples=0,
            buffer_samples=0,
        )
        self.assertEqual(ctx.step_required_samples, EXPECTED_EPISODES * SYNC_INTERVAL)
        policy = DefaultDynamicSchedulePolicy(deactivate_ratio=0.5)
        self.assertEqual(policy.deactivate_wait_samples(ctx), 128)

    def test_metrics_count_the_episode_collection_without_rounding(self) -> None:
        aggregator = MetricsAggregator(total_gpus=8)
        aggregator.add_step_metrics({"perf/time_per_step": 1.0}, sample_count=PPO_MINI_BATCH_SIZE * REQUIRE_BATCHES)
        stats = aggregator.get_current_stats()
        self.assertEqual(stats["total_samples"], EXPECTED_EPISODES)


class TestFractionalRequireBatchesRuntime(IsolatedAsyncioTestCase):
    async def test_rollouter_capacity_and_train_horizon_accept_fractional_config(self) -> None:
        rollouter = object.__new__(_local_actor_class(FullyAsyncRollouter))
        rollouter.lock = asyncio.Lock()
        rollouter.required_samples = PPO_MINI_BATCH_SIZE * REQUIRE_BATCHES
        rollouter.staleness_threshold = 0.5
        rollouter.total_rollout_steps = 6400
        rollouter.concurrent_samples_per_replica = 16
        rollouter.config = SimpleNamespace(async_training=SimpleNamespace(trigger_parameter_sync_step=SYNC_INTERVAL))
        rollouter.llm_server_manager = SimpleNamespace(get_active_server_count=lambda: 8)
        rollouter.max_required_samples = None
        rollouter.max_concurrent_samples = None
        rollouter.max_queue_size = None
        rollouter._active_count_history = [(0.0, 0, 0)]

        await rollouter.set_max_required_samples()

        self.assertEqual(rollouter.max_required_samples, 384)
        self.assertEqual(rollouter.max_queue_size, 384)
        self.assertEqual(rollouter.max_concurrent_samples, 128)
        self.assertEqual(rollouter.total_train_steps, 25)
        self.assertEqual(rollouter._active_count_history, [(0.0, 0, 128)])

    async def test_trainer_collects_exactly_64_queue_items(self) -> None:
        class _Queue:
            def __init__(self):
                self.calls = 0

            async def get_sample(self):
                self.calls += 1
                return {"sample": self.calls}, EXPECTED_EPISODES - self.calls

        queue = _Queue()
        trainer = object.__new__(_local_actor_class(FullyAsyncTrainer))
        trainer.required_samples = PPO_MINI_BATCH_SIZE * REQUIRE_BATCHES
        trainer.dynamic_schedule_enabled = False
        trainer.message_queue_client = queue
        trainer.config = SimpleNamespace(trainer=SimpleNamespace(balance_batch=False))
        trainer.tokenizer = None
        trainer._step_wait_times = []
        trainer._step_wait_samples = []
        captured = {}

        def _assemble(samples, tokenizer, config, balance_batch):
            captured["count"] = len(samples)
            return DataProto(meta_info={})

        with (
            mock.patch.object(trainer_module.ray.cloudpickle, "loads", side_effect=lambda value: value),
            mock.patch.object(trainer_module, "assemble_batch_from_rollout_samples", side_effect=_assemble),
        ):
            epoch, batch = await trainer._get_samples_from_queue()

        self.assertEqual(epoch, 0)
        self.assertIsInstance(batch, DataProto)
        self.assertEqual(queue.calls, EXPECTED_EPISODES)
        self.assertEqual(captured["count"], EXPECTED_EPISODES)
        self.assertEqual(trainer._step_wait_samples, [0])
        self.assertEqual(len(trainer._step_wait_times), 1)

    async def test_rollouter_checkpoint_arithmetic_keeps_the_same_numeric_unit(self) -> None:
        rollouter = object.__new__(_local_actor_class(FullyAsyncRollouter))
        rollouter.required_samples = PPO_MINI_BATCH_SIZE * REQUIRE_BATCHES
        rollouter.config = SimpleNamespace(
            trainer=SimpleNamespace(
                resume_mode="resume_path",
                resume_from_path="/tmp/no-data/global_step_2",
                default_hdfs_dir=None,
                default_local_dir="/tmp/no-data",
            ),
            async_training=SimpleNamespace(trigger_parameter_sync_step=SYNC_INTERVAL),
        )
        rollouter.train_dataloader = SimpleNamespace(load_state_dict=lambda state: None)
        rollouter.total_rollout_steps = 6400

        rollouter.load_checkpoint()
        self.assertEqual(rollouter.global_steps, 2 * EXPECTED_EPISODES * SYNC_INTERVAL + 1)


if __name__ == "__main__":
    unittest.main()
