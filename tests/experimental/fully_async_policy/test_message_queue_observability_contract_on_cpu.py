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

"""FIFO/drop-oldest queue semantics plus attribution and conservation.

The test intentionally preserves the native freshness policy: when full, evict
the oldest resident and admit the incoming sample.  It only requires truthful
receipts, close/drain semantics, and enough counters to audit every real item.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from types import SimpleNamespace
from unittest import IsolatedAsyncioTestCase, TestCase

from omegaconf import OmegaConf

from verl.experimental.fully_async_policy.fully_async_rollouter import FullyAsyncRollouter
from verl.experimental.fully_async_policy.fully_async_trainer import FullyAsyncTrainer
from verl.experimental.fully_async_policy.message_queue import MessageQueue


def _local_actor_class(actor_class):
    return actor_class.__ray_metadata__.modified_class


@dataclass(frozen=True)
class _Sample:
    uid: str
    min_version: int
    max_version: int


def _new_queue(max_size: int):
    queue = object.__new__(_local_actor_class(MessageQueue))
    queue.__init__(OmegaConf.create({}), max_queue_size=max_size)
    return queue


def _receipt_field(receipt, name, default=None):
    if isinstance(receipt, dict):
        return receipt.get(name, default)
    return getattr(receipt, name, default)


class TestMessageQueueFreshnessSemantics(IsolatedAsyncioTestCase):
    async def test_fifo_drop_oldest_keeps_incoming(self) -> None:
        queue = _new_queue(2)
        a = _Sample("a", 1, 1)
        b = _Sample("b", 1, 1)
        c = _Sample("c", 2, 2)

        await queue.put_sample(a)
        await queue.put_sample(b)
        await queue.put_sample(c)

        first, remaining = await queue.get_sample()
        second, remaining_after = await queue.get_sample()
        self.assertEqual((first, second), (b, c))
        self.assertEqual((remaining, remaining_after), (1, 0))

    async def test_drop_receipt_attributes_victim_not_incoming(self) -> None:
        queue = _new_queue(2)
        a = _Sample("a", 1, 1)
        b = _Sample("b", 1, 1)
        c = _Sample("c", 2, 3)

        await queue.put_sample(a)
        await queue.put_sample(b)
        receipt = await queue.put_sample(c)

        self.assertTrue(_receipt_field(receipt, "incoming_admitted"), receipt)
        self.assertEqual(_receipt_field(receipt, "incoming_uid"), "c")
        self.assertEqual(_receipt_field(receipt, "incoming_min_version"), 2)
        self.assertEqual(_receipt_field(receipt, "incoming_max_version"), 3)
        self.assertEqual(_receipt_field(receipt, "evicted_uid"), "a")
        self.assertEqual(_receipt_field(receipt, "evicted_min_version"), 1)
        self.assertEqual(_receipt_field(receipt, "evicted_max_version"), 1)
        self.assertIsNotNone(_receipt_field(receipt, "enqueue_sequence"))

    async def test_real_item_conservation_includes_clear_and_residence(self) -> None:
        queue = _new_queue(3)
        await queue.put_sample(_Sample("a", 1, 1))
        await queue.put_sample(_Sample("b", 1, 2))
        await queue.put_sample(_Sample("c", 2, 2))
        await queue.get_sample()
        await queue.clear_queue()

        stats = await queue.get_statistics()
        self.assertEqual(stats["real_enqueued"], 3)
        self.assertEqual(stats["real_consumed"], 1)
        self.assertEqual(stats["real_evicted"], 0)
        self.assertEqual(stats["real_cleared"], 2)
        self.assertEqual(stats["real_resident"], 0)
        self.assertEqual(
            stats["real_enqueued"],
            stats["real_consumed"] + stats["real_evicted"] + stats["real_cleared"] + stats["real_resident"],
        )
        self.assertGreaterEqual(stats["last_dequeue_residence_s"], 0.0)

    async def test_shutdown_is_out_of_band_and_drains_without_evicting_real_items(self) -> None:
        queue = _new_queue(1)
        await queue.put_sample(_Sample("a", 1, 1))
        await queue.shutdown()

        sample, remaining = await queue.get_sample()
        terminal, terminal_remaining = await queue.get_sample()
        rejected = await queue.put_sample(_Sample("late", 2, 2))
        stats = await queue.get_statistics()

        self.assertEqual((sample.uid, remaining), ("a", 0))
        self.assertIsNone(terminal)
        self.assertEqual(terminal_remaining, 0)
        self.assertFalse(_receipt_field(rejected, "incoming_admitted", rejected))
        self.assertTrue(stats["closed"])
        self.assertEqual(stats["real_enqueued"], 1)
        self.assertEqual(stats["real_consumed"], 1)
        self.assertEqual(stats["real_evicted"], 0)
        self.assertEqual(stats["control_signals_enqueued"], 0)


class TestFullyAsyncQueueLifecycle(IsolatedAsyncioTestCase):
    async def test_terminal_underfill_is_accounted(self) -> None:
        class _Queue:
            def __init__(self):
                self.items = [(b"one", 0), (None, 0)]

            async def get_sample(self):
                return self.items.pop(0)

        trainer = object.__new__(_local_actor_class(FullyAsyncTrainer))
        trainer.required_samples = 2
        trainer.dynamic_schedule_enabled = False
        trainer.message_queue_client = _Queue()
        trainer.config = SimpleNamespace(trainer=SimpleNamespace(balance_batch=False))
        trainer.tokenizer = None
        trainer._step_wait_times = []
        trainer._step_wait_samples = []
        trainer.terminal_underfill_samples = 0

        epoch, batch = await trainer._get_samples_from_queue()

        self.assertIsNone(epoch)
        self.assertIsNone(batch)
        self.assertEqual(trainer.terminal_underfill_samples, 1)

    async def test_pause_reason_distinguishes_queue_and_staleness_budget(self) -> None:
        class _Queue:
            def __init__(self):
                self.size = 2

            async def get_statistics(self):
                return {"queue_size": self.size}

        queue = _Queue()
        rollouter = object.__new__(_local_actor_class(FullyAsyncRollouter))
        rollouter.message_queue_client = queue
        rollouter.max_queue_size = 2
        rollouter.max_required_samples = 4
        rollouter.staleness_samples = 0
        rollouter.paused = False
        rollouter.pause_reason = None

        self.assertTrue(await rollouter._should_pause_generation())
        self.assertEqual(rollouter.pause_reason, "queue_full")

        queue.size = 0
        rollouter.staleness_samples = 4
        self.assertTrue(await rollouter._should_pause_generation())
        self.assertEqual(rollouter.pause_reason, "staleness_budget")

        rollouter.staleness_samples = 0
        self.assertFalse(await rollouter._should_pause_generation())
        self.assertIsNone(rollouter.pause_reason)


class TestNoInBandTerminalSentinel(TestCase):
    def test_rollouter_uses_existing_queue_shutdown_instead_of_putting_none(self) -> None:
        source = inspect.getsource(_local_actor_class(FullyAsyncRollouter)._streaming_generation_main)
        self.assertIn("shutdown", source)
        self.assertNotIn("put_sample(sample=None)", source)


if __name__ == "__main__":
    import unittest

    unittest.main()
