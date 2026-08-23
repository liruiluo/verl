# Copyright 2026 The verl team.
# Licensed under the Apache License, Version 2.0.

"""Regressions for lossless fully-async queue termination."""

from __future__ import annotations

import asyncio
import pickle
from types import MethodType, SimpleNamespace
from unittest import IsolatedAsyncioTestCase
from unittest.mock import patch

from omegaconf import OmegaConf

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

        for episode_id in range(FORMAL_EPISODES):
            accepted = await queue.put_sample({"episode_id": episode_id})
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


if __name__ == "__main__":
    import unittest

    unittest.main()
