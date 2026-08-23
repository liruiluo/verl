# Copyright 2025 Meituan Ltd. and/or its affiliates
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

import asyncio
import logging
from collections import Counter, deque
from dataclasses import dataclass
from typing import Any

import ray
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


_UNLABELED_DATA_SOURCE = "__unlabeled__"


@dataclass(frozen=True)
class _QueueEntry:
    payload: Any
    data_source: str


def _normalize_data_source(data_source: Any) -> str:
    if data_source is None:
        return _UNLABELED_DATA_SOURCE
    normalized = str(data_source).strip()
    if not normalized:
        return _UNLABELED_DATA_SOURCE
    if len(normalized) > 256 or "\n" in normalized or "\r" in normalized:
        raise ValueError("data_source queue label must be a single line of at most 256 characters")
    return normalized


def _counter_dict(counter: Counter[str]) -> dict[str, int]:
    return {key: int(counter[key]) for key in sorted(counter)}


@ray.remote(num_cpus=2, max_concurrency=20)
class MessageQueue:
    """
    Simplified Ray-based asynchronous message queue for communication between Rollouter and Trainer
    """

    def __init__(self, config: DictConfig, max_queue_size: int = 1000):
        self.config = config
        if max_queue_size is None:
            raise ValueError(f"max_queue_size cannot be None, got: {max_queue_size}")
        self.max_queue_size = int(max_queue_size)
        self.queue = deque(maxlen=self.max_queue_size)

        self.val_queue = deque()

        # Asyncio for message handling
        self.running = True

        # async safe
        self._lock = asyncio.Lock()
        self._consumer_condition = asyncio.Condition(self._lock)

        # statistic message
        self.total_produced = 0
        self.total_consumed = 0
        self.dropped_samples = 0
        self.total_cleared = 0
        self.enqueued_by_data_source: Counter[str] = Counter()
        self.consumed_by_data_source: Counter[str] = Counter()
        self.evicted_by_data_source: Counter[str] = Counter()
        self.cleared_by_data_source: Counter[str] = Counter()

        print(f"[MessageQueue] initialized with max_queue_size={max_queue_size}")

    async def put_sample(self, sample: Any, data_source: str | None = None) -> bool:
        """
        Put a batch sample into the queue

        Args:
            sample: Sample data
            data_source: Optional opaque source label used only for accounting.

        Returns:
            bool: True when enqueued without eviction; False when an older
                queued sample was evicted before this sample was accepted.
        """
        data_source = _normalize_data_source(data_source)
        async with self._lock:
            if not self.running:
                raise RuntimeError("cannot put a sample after the message queue is shut down")
            # Older rollouters used an in-band ``None`` sentinel. Translate it
            # into native shutdown without consuming bounded queue capacity or
            # evicting an accepted rollout.
            if sample is None:
                self.running = False
                self._consumer_condition.notify_all()
                return True

            # If queue is full, remove the oldest sample (rarely happens).
            # Keep the evicted sample's label so overflow is not misattributed
            # to the newly accepted sample.
            is_drop = False
            if len(self.queue) >= self.max_queue_size:
                evicted = self.queue.popleft()
                self.dropped_samples += 1
                self.evicted_by_data_source[evicted.data_source] += 1
                is_drop = True
                logger.warning("Queue full, dropped sample")
            self.queue.append(_QueueEntry(payload=sample, data_source=data_source))
            self.total_produced += 1
            self.enqueued_by_data_source[data_source] += 1

            # Notify waiting consumers
            self._consumer_condition.notify_all()

            if self.total_produced % 100 == 0:
                print(f"MessageQueue stats: produced={self.total_produced}, queue_size={len(self.queue)}")
            if is_drop:
                return False
            return True

    async def get_sample(self) -> Any | None:
        """
        Get a single sample from the queue, wait until one is available

        Returns:
            Any: Single sample data or None if queue is closed
        """
        async with self._lock:
            while len(self.queue) == 0 and self.running:
                await self._consumer_condition.wait()

            # If queue is closed and empty, return None
            if not self.running and len(self.queue) == 0:
                return None

            # Get one sample
            entry = self.queue.popleft()
            self.total_consumed += 1
            self.consumed_by_data_source[entry.data_source] += 1
            return entry.payload, len(self.queue)

    async def get_queue_size(self) -> int:
        """Get current queue length"""
        async with self._lock:
            return len(self.queue)

    async def get_statistics(self) -> dict[str, Any]:
        """Get queue statistics"""
        async with self._lock:
            resident_by_data_source = Counter(entry.data_source for entry in self.queue)
            return {
                "queue_size": len(self.queue),
                "total_produced": self.total_produced,
                "total_consumed": self.total_consumed,
                "dropped_samples": self.dropped_samples,
                "total_cleared": self.total_cleared,
                "max_queue_size": self.max_queue_size,
                "enqueued_by_data_source": _counter_dict(self.enqueued_by_data_source),
                "consumed_by_data_source": _counter_dict(self.consumed_by_data_source),
                "evicted_by_data_source": _counter_dict(self.evicted_by_data_source),
                "cleared_by_data_source": _counter_dict(self.cleared_by_data_source),
                "resident_by_data_source": _counter_dict(resident_by_data_source),
            }

    async def clear_queue(self):
        """Clear the queue"""
        async with self._lock:
            cleared_count = len(self.queue)
            for entry in self.queue:
                self.cleared_by_data_source[entry.data_source] += 1
            self.total_cleared += cleared_count
            self.queue.clear()
            logger.info(f"Cleared {cleared_count} samples from queue")

    async def shutdown(self):
        """Shutdown the message queue"""
        async with self._lock:
            self.running = False
            # Notify all waiting coroutines so they can exit
            self._consumer_condition.notify_all()
        logger.info("MessageQueue shutdown")

    async def get_memory_usage(self) -> dict:
        """Get memory usage statistics"""
        async with self._lock:
            # Estimate memory usage of samples in queue
            import sys

            total_size = 0
            sample_count = len(self.queue)

            if sample_count > 0:
                # Estimate size of a single sample (simplified estimation)
                sample = list(self.queue)[0].payload
                try:
                    sample_size = sys.getsizeof(sample)
                    # Since we now store RolloutSample directly, estimate based on its components
                    if hasattr(sample, "original_batch_dict") and sample.original_batch_dict:
                        # Estimate batch data size
                        batch_data = sample.original_batch_dict.get("batch", {})
                        sample_size += len(batch_data) * 1000  # Roughly estimate 1KB per batch entry
                    if hasattr(sample, "agent_loop_output"):
                        # Estimate AgentLoopOutput size
                        sample_size += 5000  # Roughly estimate 5KB for AgentLoopOutput
                    total_size = sample_size * sample_count
                except Exception:
                    total_size = sample_count * 15000  # Roughly estimate 15KB per RolloutSample

            return {
                "queue_samples": sample_count,
                "estimated_memory_bytes": total_size,
                "estimated_memory_mb": total_size / (1024 * 1024),
            }

    async def put_validate(self, data):
        async with self._lock:
            self.val_queue.append(data)

    async def get_validate(self):
        async with self._lock:
            if self.val_queue:
                return self.val_queue.popleft()
            else:
                return None


class MessageQueueClient:
    """Asyncio-compatible MessageQueue client for communicating with MessageQueue Actor"""

    def __init__(self, queue_actor: Any):
        self.queue_actor = queue_actor

    async def put_sample(self, sample: Any, data_source: str | None = None) -> bool:
        """Put batch into queue (async)."""
        future = self.queue_actor.put_sample.remote(sample, data_source=data_source)
        return await asyncio.wrap_future(future.future())

    async def put_validate(self, data: Any) -> bool:
        future = self.queue_actor.put_validate.remote(data)
        return await asyncio.wrap_future(future.future())

    def get_validate_sync(self) -> Any | None:
        return ray.get(self.queue_actor.get_validate.remote())

    async def get_sample(self) -> Any | None:
        """Get single sample from queue, wait until one is available (async)"""
        future = self.queue_actor.get_sample.remote()
        return await asyncio.wrap_future(future.future())

    async def get_queue_size(self) -> int:
        """Get queue size (async)"""
        future = self.queue_actor.get_queue_size.remote()
        return await asyncio.wrap_future(future.future())

    async def get_statistics(self) -> dict[str, Any]:
        """Get statistics (async)"""
        future = self.queue_actor.get_statistics.remote()
        return await asyncio.wrap_future(future.future())

    async def clear_queue(self):
        """Clear queue (async)"""
        future = self.queue_actor.clear_queue.remote()
        await asyncio.wrap_future(future.future())

    async def shutdown(self):
        """Shutdown queue (async)"""
        future = self.queue_actor.shutdown.remote()
        await asyncio.wrap_future(future.future())

    async def get_memory_usage(self) -> dict:
        """Get memory usage statistics (async)"""
        future = self.queue_actor.get_memory_usage.remote()
        return await asyncio.wrap_future(future.future())

    def get_sample_sync(self) -> Any | None:
        """Get single sample from queue (sync - deprecated, use get_sample instead)"""
        return ray.get(self.queue_actor.get_sample.remote())

    def get_statistics_sync(self) -> dict[str, Any]:
        """Get statistics (sync - deprecated, use get_statistics instead)"""
        return ray.get(self.queue_actor.get_statistics.remote())

    def get_queue_size_sync(self) -> int:
        """Get current queue length (sync, blocking).

        Used by callers running outside an event loop (e.g. synchronous
        post-processing hooks) that need an immediate snapshot of the queue size.
        """
        return ray.get(self.queue_actor.get_queue_size.remote())
