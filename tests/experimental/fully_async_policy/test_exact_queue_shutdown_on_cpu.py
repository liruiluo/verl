# Copyright 2026 The verl team.
# Licensed under the Apache License, Version 2.0.

"""Regressions for lossless fully-async queue termination."""

from __future__ import annotations

import asyncio
import pickle
from types import MethodType, SimpleNamespace
from unittest import IsolatedAsyncioTestCase, TestCase, mock
from unittest.mock import patch

import numpy as np
import torch
from omegaconf import OmegaConf
from tensordict import TensorDict

from verl import DataProto

from verl.experimental.fully_async_policy.detach_utils import (
    MetricsAggregator,
    RolloutSample,
    assemble_batch_from_rollout_samples,
    extract_scalar_sample_metadata,
)
from verl.experimental.fully_async_policy.fully_async_main import FullyAsyncTaskRunner
from verl.experimental.fully_async_policy.fully_async_rollouter import FullyAsyncRollouter
from verl.experimental.fully_async_policy.fully_async_trainer import (
    FullyAsyncTrainer,
    TrainingStopException,
)
from verl.experimental.fully_async_policy.message_queue import MessageQueue


FORMAL_EPISODES = 25_600
EPISODES_PER_UPDATE = 64


def _local_actor_class(actor_class):
    return actor_class.__ray_metadata__.modified_class


def _rollout_sample(data_source: str, response_masks: list[list[int]], sample_id: str) -> RolloutSample:
    response_mask = torch.tensor(response_masks, dtype=torch.long)
    rows, response_width = response_mask.shape
    prompts = torch.ones((rows, 1), dtype=torch.long)
    responses = torch.ones((rows, response_width), dtype=torch.long)
    full_batch = DataProto(
        batch=TensorDict(
            {
                "prompts": prompts,
                "responses": responses,
                "input_ids": torch.cat([prompts, responses], dim=1),
                "attention_mask": torch.cat(
                    [torch.ones_like(prompts), response_mask], dim=1
                ),
                "position_ids": torch.arange(
                    1 + response_width, dtype=torch.long
                ).repeat(rows, 1),
                "response_mask": response_mask,
            },
            batch_size=rows,
        ),
        non_tensor_batch={
            "data_source": np.array([data_source] * rows, dtype=object),
            "min_global_steps": np.zeros(rows, dtype=np.int64),
            "max_global_steps": np.zeros(rows, dtype=np.int64),
        },
        meta_info={
            "metrics": [
                {"generate_sequences": 1.0, "tool_calls": float(index + 1)}
                for index in range(rows)
            ]
        },
    )
    return RolloutSample(
        full_batch=full_batch,
        sample_id=sample_id,
        epoch=0,
        rollout_status={},
        sample_metadata={"data_source": data_source},
    )


def _optimizer_batch(sample_metadata: list[dict], response_masks: list[list[int]], is_padding=None) -> DataProto:
    response_mask = torch.tensor(response_masks, dtype=torch.long)
    rows = response_mask.shape[0]
    non_tensor_batch = {}
    if is_padding is not None:
        non_tensor_batch["is_padding"] = np.asarray(is_padding, dtype=bool)
    return DataProto(
        batch=TensorDict({"response_mask": response_mask}, batch_size=rows),
        non_tensor_batch=non_tensor_batch,
        meta_info={
            "fully_async/sample_metadata": sample_metadata,
            "fully_async/real_episode_count": len(sample_metadata),
        },
    )


class _ClosedQueueClient:
    async def get_sample(self):
        return None


class _LegacySentinelQueueClient:
    async def get_sample(self):
        return None, 0


class _RecordingQueueClient:
    def __init__(self):
        self.shutdown_calls = 0
        self.samples = []
        self.closed = False
        self.post_shutdown_puts = 0

    async def put_sample(self, sample):
        if self.closed:
            self.post_shutdown_puts += 1
            raise RuntimeError("post-shutdown write")
        self.samples.append(sample)
        return True

    async def shutdown(self):
        self.shutdown_calls += 1
        self.closed = True


class _FiniteQueueClient:
    def __init__(self, samples):
        self.samples = list(samples)

    async def get_sample(self):
        if not self.samples:
            return None
        sample = self.samples.pop(0)
        return sample, len(self.samples)


class TestMessageQueueShutdown(IsolatedAsyncioTestCase):
    async def test_shutdown_drains_exact_multitask400_schedule_without_data_sentinel(self) -> None:
        queue_cls = _local_actor_class(MessageQueue)
        queue = queue_cls(OmegaConf.create({}), max_queue_size=FORMAL_EPISODES)

        data_sources = ("webshop", "swesmith", "literesearcher", "openmle_fast")
        for episode_id in range(FORMAL_EPISODES):
            accepted = await queue.put_sample(
                {"episode_id": episode_id},
                data_source=data_sources[episode_id % len(data_sources)],
            )
            self.assertTrue(accepted)

        await queue.shutdown()

        consumed = []
        while True:
            item = await queue.get_sample()
            if item is None:
                break
            sample, _queue_len = item
            consumed.append(sample["episode_id"])

        stats = await queue.get_statistics()
        self.assertEqual(consumed, list(range(FORMAL_EPISODES)))
        self.assertEqual(stats["total_produced"], FORMAL_EPISODES)
        self.assertEqual(stats["total_consumed"], FORMAL_EPISODES)
        self.assertEqual(stats["dropped_samples"], 0)
        self.assertEqual(stats["queue_size"], 0)
        self.assertEqual(
            stats["enqueued_by_data_source"],
            {source: FORMAL_EPISODES // 4 for source in data_sources},
        )
        self.assertEqual(
            stats["consumed_by_data_source"],
            {source: FORMAL_EPISODES // 4 for source in data_sources},
        )
        self.assertEqual(stats["evicted_by_data_source"], {})
        self.assertEqual(len(consumed) // EPISODES_PER_UPDATE, 400)
        self.assertEqual(len(consumed) % EPISODES_PER_UPDATE, 0)

    async def test_shutdown_rejects_new_samples_without_touching_drained_data(self) -> None:
        queue_cls = _local_actor_class(MessageQueue)
        queue = queue_cls(OmegaConf.create({}), max_queue_size=1)
        self.assertTrue(await queue.put_sample(b"accepted"))

        # A mixed-version producer may still send the old in-band sentinel.
        # Treat it as native shutdown so it cannot evict the accepted sample.
        self.assertTrue(await queue.put_sample(None))
        self.assertEqual(await queue.get_sample(), (b"accepted", 0))
        self.assertIsNone(await queue.get_sample())

        with self.assertRaisesRegex(RuntimeError, "shut down"):
            await queue.put_sample(b"orphan")


class TestFullyAsyncTermination(IsolatedAsyncioTestCase):
    async def test_trainer_accepts_native_shutdown_none_without_unpacking(self) -> None:
        trainer = object.__new__(_local_actor_class(FullyAsyncTrainer))
        trainer.required_samples = EPISODES_PER_UPDATE
        trainer.dynamic_schedule_enabled = False
        trainer.message_queue_client = _ClosedQueueClient()

        epoch, batch = await trainer._get_samples_from_queue()

        self.assertIsNone(epoch)
        self.assertIsNone(batch)

    async def test_trainer_accepts_legacy_in_band_sentinel_during_upgrade(self) -> None:
        trainer = object.__new__(_local_actor_class(FullyAsyncTrainer))
        trainer.required_samples = EPISODES_PER_UPDATE
        trainer.dynamic_schedule_enabled = False
        trainer.message_queue_client = _LegacySentinelQueueClient()

        epoch, batch = await trainer._get_samples_from_queue()

        self.assertIsNone(epoch)
        self.assertIsNone(batch)

    async def test_partial_final_batch_is_returned_for_zero_loss_alignment(self) -> None:
        trainer = object.__new__(_local_actor_class(FullyAsyncTrainer))
        trainer.required_samples = 3
        trainer.dynamic_schedule_enabled = False
        trainer.message_queue_client = _FiniteQueueClient(
            [pickle.dumps({"episode": 0}), pickle.dumps({"episode": 1})]
        )
        trainer.tokenizer = object()
        trainer.config = OmegaConf.create({})
        trainer._step_wait_times = []
        trainer._step_wait_samples = []
        assembled = SimpleNamespace(meta_info={})

        with patch(
            "verl.experimental.fully_async_policy.fully_async_trainer.assemble_batch_from_rollout_samples",
            return_value=assembled,
        ) as assemble:
            epoch, batch = await trainer._get_samples_from_queue()

        self.assertEqual(epoch, 0)
        self.assertIs(batch, assembled)
        self.assertEqual(len(assemble.call_args.args[0]), 2)
        self.assertTrue(batch.meta_info["fully_async/final_partial_batch"])
        self.assertEqual(batch.meta_info["fully_async/real_episode_count"], 2)

    async def test_fit_lifecycle_consumes_exactly_400_complete_batches(self) -> None:
        queue_cls = _local_actor_class(MessageQueue)
        queue = queue_cls(OmegaConf.create({}), max_queue_size=FORMAL_EPISODES)
        with patch("builtins.print"):
            for episode_id in range(FORMAL_EPISODES):
                self.assertTrue(await queue.put_sample(pickle.dumps(episode_id)))
            await queue.shutdown()

            trainer = object.__new__(_local_actor_class(FullyAsyncTrainer))
            trainer.required_samples = EPISODES_PER_UPDATE
            trainer.dynamic_schedule_enabled = False
            trainer.message_queue_client = queue
            trainer.rollouter = object()
            trainer.tokenizer = object()
            trainer.config = OmegaConf.create({"trainer": {"test_freq": 1}})
            trainer._step_wait_times = []
            trainer._step_wait_samples = []
            trainer.global_steps = 0
            trainer.current_param_version = 0
            trainer.local_trigger_step = 1
            trainer.progress_bar = SimpleNamespace(close=lambda: None)
            trainer.optimizer_updates = 0

            async def _fit_step(self, batch_dict=None):
                _epoch, batch = await self._get_samples_from_queue()
                if batch is None:
                    raise TrainingStopException
                self.optimizer_updates += 1
                self.current_param_version += 1

            trainer.fit_step = MethodType(_fit_step, trainer)
            trainer._fit_save_checkpoint = lambda force=False: None
            assembled = SimpleNamespace(meta_info={})
            with patch(
                "verl.experimental.fully_async_policy.fully_async_trainer.assemble_batch_from_rollout_samples",
                return_value=assembled,
            ):
                await trainer.fit()

        stats = await queue.get_statistics()
        self.assertEqual(trainer.optimizer_updates, 400)
        self.assertEqual(stats["total_consumed"], FORMAL_EPISODES)
        self.assertEqual(stats["queue_size"], 0)

    async def test_rollouter_quiesces_inflight_producers_before_shutdown(self) -> None:
        rollouter = object.__new__(_local_actor_class(FullyAsyncRollouter))
        queue_client = _RecordingQueueClient()
        producer_started = asyncio.Event()
        producer_cancelled = asyncio.Event()
        processor_release = asyncio.Event()

        async def _active_producer():
            producer_started.set()
            try:
                await asyncio.Event().wait()
            finally:
                if queue_client.closed:
                    await queue_client.put_sample(b"orphan")
                producer_cancelled.set()

        async def _failed_feed(self):
            await producer_started.wait()
            raise RuntimeError("synthetic feed failure")

        async def _processor(self):
            task = asyncio.create_task(_active_producer(), name="inflight-producer")
            self.active_tasks.add(task)
            await producer_started.wait()
            await processor_release.wait()

        rollouter.async_rollout_manager = SimpleNamespace()
        rollouter.max_concurrent_samples = 1
        rollouter._feed_samples = MethodType(_failed_feed, rollouter)
        rollouter._processor_worker = MethodType(_processor, rollouter)
        rollouter.pending_queue = asyncio.Queue()
        rollouter.active_tasks = set()
        rollouter._active_count_history = []
        rollouter.feed_task = None
        rollouter.processor_task = None
        rollouter.message_queue_client = queue_client
        rollouter.lock = asyncio.Lock()
        rollouter.running = True

        with self.assertRaisesRegex(RuntimeError, "synthetic feed failure"):
            await rollouter._streaming_generation_main()

        self.assertTrue(producer_cancelled.is_set())
        self.assertEqual(queue_client.shutdown_calls, 1)
        self.assertEqual(queue_client.post_shutdown_puts, 0)
        self.assertFalse(rollouter.active_tasks)

    async def test_rollouter_uses_native_shutdown_instead_of_enqueuing_none(self) -> None:
        rollouter = object.__new__(_local_actor_class(FullyAsyncRollouter))
        queue_client = _RecordingQueueClient()

        async def _finished(self):
            return None

        rollouter.async_rollout_manager = SimpleNamespace()
        rollouter.max_concurrent_samples = 1
        rollouter._feed_samples = MethodType(_finished, rollouter)
        rollouter._processor_worker = MethodType(_finished, rollouter)
        rollouter.pending_queue = asyncio.Queue()
        rollouter.active_tasks = set()
        rollouter._active_count_history = []
        rollouter.feed_task = None
        rollouter.processor_task = None
        rollouter.message_queue_client = queue_client
        rollouter.lock = asyncio.Lock()
        rollouter.running = True

        await rollouter._streaming_generation_main()

        self.assertEqual(queue_client.shutdown_calls, 1)
        self.assertEqual(queue_client.samples, [])
        self.assertFalse(rollouter.running)


class TestSampleMetadata(TestCase):
    def test_extracts_one_data_source_from_multi_action_episode(self) -> None:
        batch = SimpleNamespace(
            non_tensor_batch={
                "data_source": np.array(["webshop", "webshop"], dtype=object)
            }
        )
        self.assertEqual(
            extract_scalar_sample_metadata(batch), {"data_source": "webshop"}
        )

    def test_rejects_mixed_data_sources_inside_one_episode(self) -> None:
        batch = SimpleNamespace(
            non_tensor_batch={
                "data_source": np.array(["webshop", "swesmith"], dtype=object)
            }
        )
        with self.assertRaisesRegex(ValueError, "multiple values"):
            extract_scalar_sample_metadata(batch)

    def test_assembler_preserves_route_and_exact_real_work(self) -> None:
        batch = assemble_batch_from_rollout_samples(
            [
                _rollout_sample("webshop", [[1, 1], [1, 0]], "episode-0"),
                _rollout_sample("openmle_fast", [[1, 1]], "episode-1"),
            ],
            tokenizer=None,
            config=None,
        )

        self.assertEqual(len(batch), 3)
        self.assertEqual(batch.meta_info["fully_async/real_episode_count"], 2)
        self.assertEqual(
            batch.meta_info["fully_async/sample_metadata"],
            [
                {
                    "data_source": "webshop",
                    "real_action_rows": 2,
                    "policy_response_tokens": 3,
                },
                {
                    "data_source": "openmle_fast",
                    "real_action_rows": 1,
                    "policy_response_tokens": 2,
                },
            ],
        )


class TestQueueDataSourceAccounting(IsolatedAsyncioTestCase):
    async def test_queue_attributes_overflow_to_evicted_source(self) -> None:
        queue_cls = _local_actor_class(MessageQueue)
        queue = queue_cls(OmegaConf.create({}), max_queue_size=2)

        self.assertTrue(await queue.put_sample("slow-0", data_source="slow"))
        self.assertTrue(await queue.put_sample("fast-0", data_source="fast"))
        self.assertFalse(await queue.put_sample("fast-1", data_source="fast"))

        first, _ = await queue.get_sample()
        self.assertEqual(first, "fast-0")
        stats = await queue.get_statistics()
        self.assertEqual(stats["total_produced"], 3)
        self.assertEqual(stats["total_consumed"], 1)
        self.assertEqual(stats["dropped_samples"], 1)
        self.assertEqual(stats["total_cleared"], 0)
        self.assertEqual(stats["enqueued_by_data_source"], {"fast": 2, "slow": 1})
        self.assertEqual(stats["consumed_by_data_source"], {"fast": 1})
        self.assertEqual(stats["evicted_by_data_source"], {"slow": 1})
        self.assertEqual(stats["resident_by_data_source"], {"fast": 1})
        self.assertEqual(
            stats["enqueued_by_data_source"],
            {
                source: stats["consumed_by_data_source"].get(source, 0)
                + stats["evicted_by_data_source"].get(source, 0)
                + stats["cleared_by_data_source"].get(source, 0)
                + stats["resident_by_data_source"].get(source, 0)
                for source in stats["enqueued_by_data_source"]
            },
        )

        await queue.clear_queue()
        stats = await queue.get_statistics()
        self.assertEqual(stats["queue_size"], 0)
        self.assertEqual(stats["total_cleared"], 1)
        self.assertEqual(stats["cleared_by_data_source"], {"fast": 1})
        self.assertEqual(stats["resident_by_data_source"], {})
        self.assertEqual(
            stats["total_produced"],
            stats["total_consumed"]
            + stats["dropped_samples"]
            + stats["total_cleared"]
            + stats["queue_size"],
        )

    async def test_rollouter_exposes_cumulative_source_events(self) -> None:
        class _QueueStatsClient:
            async def get_statistics(self):
                return {
                    "queue_size": 2,
                    "total_produced": 7,
                    "total_consumed": 4,
                    "dropped_samples": 1,
                    "total_cleared": 0,
                    "enqueued_by_data_source": {"fast": 5, "slow": 2},
                    "consumed_by_data_source": {"fast": 3, "slow": 1},
                    "evicted_by_data_source": {"slow": 1},
                    "cleared_by_data_source": {},
                    "resident_by_data_source": {"fast": 2},
                }

        rollouter = object.__new__(_local_actor_class(FullyAsyncRollouter))
        rollouter.message_queue_client = _QueueStatsClient()
        rollouter.active_tasks = set()
        rollouter.pending_queue = asyncio.Queue()
        rollouter.total_generated_samples = 7
        rollouter.rollout_completed_samples = 7
        rollouter.rollout_failed_samples = 1
        rollouter.rollout_completed_by_data_source = {"fast": 5, "slow": 2}
        rollouter.rollout_failed_by_data_source = {"slow": 1}
        rollouter.queue_overflow_evictions = 1
        rollouter.staleness_samples = 2
        rollouter.dropped_stale_samples = 0
        rollouter.max_required_samples = 128
        rollouter.required_samples = 64
        rollouter.staleness_threshold = 1
        rollouter.max_queue_size = 128
        rollouter.max_concurrent_samples = 32

        stats = await rollouter.get_statistics()

        self.assertEqual(stats["count/rollout_completed/data_source/fast"], 5)
        self.assertEqual(stats["count/rollout_failed/data_source/slow"], 1)
        self.assertEqual(stats["count/queue_enqueued/data_source/slow"], 2)
        self.assertEqual(stats["count/queue_dequeued/data_source/fast"], 3)
        self.assertEqual(stats["count/queue_overflow_evicted/data_source/slow"], 1)
        self.assertEqual(stats["count/queue_resident/data_source/fast"], 2)
        self.assertEqual(stats["count/queue_resident_samples"], 2)
        self.assertEqual(stats["count/queue_overflow_evictions"], 1)
        self.assertEqual(stats["count/dropped_stale_samples"], 0)

    async def test_optimizer_consumption_counts_episode_action_and_token_work_by_route(self) -> None:
        trainer = object.__new__(_local_actor_class(FullyAsyncTrainer))
        trainer.required_samples = 4
        trainer.metrics = {}
        batch = _optimizer_batch(
            sample_metadata=[
                {
                    "data_source": "webshop",
                    "real_action_rows": 2,
                    "policy_response_tokens": 3,
                },
                {
                    "data_source": "swesmith",
                    "real_action_rows": 1,
                    "policy_response_tokens": 1,
                },
                {
                    "data_source": "literesearcher",
                    "real_action_rows": 1,
                    "policy_response_tokens": 2,
                },
                {
                    "data_source": "openmle_fast",
                    "real_action_rows": 2,
                    "policy_response_tokens": 2,
                },
            ],
            response_masks=[
                [1, 1],
                [1, 0],
                [1, 0],
                [1, 1],
                [1, 0],
                [1, 0],
            ],
        )

        trainer._record_optimizer_consumption(batch)

        self.assertEqual(trainer.optimizer_consumed_episodes, 4)
        self.assertEqual(trainer.optimizer_consumed_action_rows, 6)
        self.assertEqual(trainer.optimizer_consumed_policy_response_tokens, 8)
        self.assertEqual(
            dict(trainer.optimizer_consumed_episodes_by_data_source),
            {
                "webshop": 1,
                "swesmith": 1,
                "literesearcher": 1,
                "openmle_fast": 1,
            },
        )
        self.assertEqual(trainer.optimizer_consumed_action_rows_by_data_source["webshop"], 2)
        self.assertEqual(
            trainer.optimizer_consumed_policy_response_tokens_by_data_source["webshop"], 3
        )
        self.assertEqual(trainer.metrics["fully_async/sum/optimizer_consumed_episodes"], 4)
        self.assertEqual(trainer.metrics["fully_async/sum/optimizer_consumed_action_rows"], 6)
        self.assertEqual(
            trainer.metrics["fully_async/sum/optimizer_consumed_policy_response_tokens"], 8
        )

    async def test_partial_tail_excludes_synthetic_padding_from_consumption(self) -> None:
        trainer = object.__new__(_local_actor_class(FullyAsyncTrainer))
        trainer.required_samples = 4
        trainer.metrics = {}
        batch = _optimizer_batch(
            sample_metadata=[
                {
                    "data_source": "webshop",
                    "real_action_rows": 2,
                    "policy_response_tokens": 3,
                },
                {
                    "data_source": "openmle_fast",
                    "real_action_rows": 1,
                    "policy_response_tokens": 1,
                },
            ],
            response_masks=[[1, 1], [1, 0], [1, 0], [1, 1]],
            is_padding=[False, False, False, True],
        )

        trainer._record_optimizer_consumption(batch)

        self.assertEqual(trainer.optimizer_consumed_episodes, 2)
        self.assertEqual(trainer.optimizer_consumed_action_rows, 3)
        self.assertEqual(trainer.optimizer_consumed_policy_response_tokens, 4)
        self.assertEqual(
            trainer.metrics["fully_async/count/optimizer_consumed_episodes/data_source/webshop"],
            1,
        )


class _ImmediateRemoteMethod:
    def __init__(self, value, events=None, event=None):
        self.value = value
        self.events = events
        self.event = event

    def remote(self):
        if self.events is not None:
            self.events.append(self.event)
        return self.value


class TestFinalStatistics(TestCase):
    def _runner(self, *, queue_value, rollouter_value, trainer_value, events=None):
        runner = object.__new__(_local_actor_class(FullyAsyncTaskRunner))
        runner.components = {
            "message_queue": SimpleNamespace(
                clear_queue=_ImmediateRemoteMethod(
                    "queue-cleared", events=events, event="clear"
                ),
                get_statistics=_ImmediateRemoteMethod(
                    queue_value, events=events, event="queue_stats"
                ),
            ),
            "rollouter": SimpleNamespace(
                get_statistics=_ImmediateRemoteMethod(
                    rollouter_value, events=events, event="rollouter_stats"
                )
            ),
            "trainer": SimpleNamespace(
                get_runtime_statistics=_ImmediateRemoteMethod(
                    trainer_value, events=events, event="trainer_stats"
                )
            ),
        }
        return runner

    def test_task_runner_collects_all_three_true_owner_snapshots_once(self) -> None:
        runner = self._runner(
            queue_value={"total_produced": 4},
            rollouter_value={"count/rollout_completed_samples": 4},
            trainer_value={"optimizer_consumed_episodes": 4},
        )

        with (
            mock.patch(
                "verl.experimental.fully_async_policy.fully_async_main.ray.wait",
                side_effect=lambda values, num_returns, timeout: (values, []),
            ) as wait,
            mock.patch(
                "verl.experimental.fully_async_policy.fully_async_main.ray.get",
                side_effect=lambda value, *args, **kwargs: value,
            ),
        ):
            statistics = runner._collect_final_statistics()

        wait.assert_called_once()
        self.assertEqual(statistics["schema"], "verl_fully_async_final_statistics_v1")
        self.assertEqual(statistics["queue"]["total_produced"], 4)
        self.assertEqual(statistics["rollouter"]["count/rollout_completed_samples"], 4)
        self.assertEqual(statistics["trainer"]["optimizer_consumed_episodes"], 4)

    def test_owner_timeout_uses_one_shared_wait_budget(self) -> None:
        queue_value, rollouter_value, trainer_value = object(), object(), object()
        runner = self._runner(
            queue_value=queue_value,
            rollouter_value=rollouter_value,
            trainer_value=trainer_value,
        )

        with (
            mock.patch(
                "verl.experimental.fully_async_policy.fully_async_main.ray.wait",
                return_value=([queue_value], [rollouter_value, trainer_value]),
            ) as wait,
            mock.patch(
                "verl.experimental.fully_async_policy.fully_async_main.ray.get",
                return_value={"total_produced": 4},
            ) as get,
        ):
            statistics = runner._collect_final_statistics(timeout_s=7.0)

        wait.assert_called_once_with(
            [queue_value, rollouter_value, trainer_value], num_returns=3, timeout=7.0
        )
        get.assert_called_once_with(queue_value)
        self.assertEqual(statistics["rollouter"]["error_type"], "TimeoutError")
        self.assertEqual(statistics["trainer"]["error_type"], "TimeoutError")

    def test_finalizer_clears_before_collecting_cleanup_inclusive_stats(self) -> None:
        events = []
        runner = self._runner(
            queue_value={"total_cleared": 2, "resident_by_data_source": {}},
            rollouter_value={},
            trainer_value={},
            events=events,
        )

        with (
            mock.patch(
                "verl.experimental.fully_async_policy.fully_async_main.ray.wait",
                side_effect=lambda values, num_returns, timeout: (values, []),
            ),
            mock.patch(
                "verl.experimental.fully_async_policy.fully_async_main.ray.get",
                side_effect=lambda value, *args, **kwargs: value,
            ),
        ):
            statistics = runner._finalize_statistics()

        self.assertEqual(events[0], "clear")
        self.assertEqual(events[1:], ["queue_stats", "rollouter_stats", "trainer_stats"])
        self.assertEqual(statistics["queue_cleanup"], {"status": "completed"})
        self.assertEqual(statistics["queue"]["total_cleared"], 2)


class TestCumulativeMetricAggregation(TestCase):
    def test_dynamic_fully_async_count_uses_last_snapshot(self) -> None:
        aggregator = MetricsAggregator(total_gpus=8)
        key = "fully_async/count/queue_enqueued/data_source/webshop"
        aggregator.add_step_metrics(metrics={key: 3}, sample_count=32, timestamp=1.0)
        aggregator.add_step_metrics(metrics={key: 5}, sample_count=32, timestamp=2.0)

        self.assertEqual(aggregator.get_aggregated_metrics()[key], 5)


if __name__ == "__main__":
    import unittest

    unittest.main()
