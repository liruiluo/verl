# Copyright 2026 The verl team.
# Licensed under the Apache License, Version 2.0.

"""Regressions for lossless fully-async queue termination."""

from __future__ import annotations

import asyncio
from types import MethodType, SimpleNamespace
from unittest import IsolatedAsyncioTestCase

from omegaconf import OmegaConf

from verl.experimental.fully_async_policy.fully_async_rollouter import FullyAsyncRollouter
from verl.experimental.fully_async_policy.fully_async_trainer import FullyAsyncTrainer
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

    async def put_sample(self, sample):
        self.samples.append(sample)
        return True

    async def shutdown(self):
        self.shutdown_calls += 1


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
