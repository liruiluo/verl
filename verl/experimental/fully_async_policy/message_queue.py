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
import time
from collections import deque
from dataclasses import dataclass
from typing import Any

import ray
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


def _sample_field(sample: Any, name: str):
    if isinstance(sample, dict):
        return sample.get(name)
    return getattr(sample, name, None)


@dataclass(frozen=True)
class QueuePutReceipt:
    incoming_admitted: bool
    incoming_uid: Any = None
    incoming_min_version: Any = None
    incoming_max_version: Any = None
    evicted_uid: Any = None
    evicted_min_version: Any = None
    evicted_max_version: Any = None
    enqueue_sequence: int | None = None

    def __bool__(self) -> bool:
        return self.incoming_admitted


@dataclass(frozen=True)
class _QueuedSample:
    sample: Any
    enqueued_at: float
    sequence: int
    uid: Any
    min_version: Any
    max_version: Any


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
        self.real_cleared = 0
        self.last_dequeue_residence_s = 0.0
        self._enqueue_sequence = 0

        print(f"[MessageQueue] initialized with max_queue_size={max_queue_size}")

    async def put_sample(
        self,
        sample: Any,
        *,
        uid: Any = None,
        min_version: Any = None,
        max_version: Any = None,
    ) -> QueuePutReceipt:
        """
        Put a batch sample into the queue

        Args:
            sample: Sample data
            uid: Stable identity for receipt/conservation accounting.
            min_version: Minimum policy version used by the sample.
            max_version: Maximum policy version used by the sample.

        Returns:
            Admission and optional resident-eviction metadata.
        """
        async with self._lock:
            incoming_uid = uid if uid is not None else _sample_field(sample, "uid")
            if incoming_uid is None:
                incoming_uid = _sample_field(sample, "sample_id")
            incoming_min_version = min_version if min_version is not None else _sample_field(sample, "min_version")
            incoming_max_version = max_version if max_version is not None else _sample_field(sample, "max_version")
            incoming_metadata = {
                "incoming_uid": incoming_uid,
                "incoming_min_version": incoming_min_version,
                "incoming_max_version": incoming_max_version,
            }
            if not self.running:
                return QueuePutReceipt(incoming_admitted=False, **incoming_metadata)

            evicted = None
            if len(self.queue) >= self.max_queue_size:
                evicted = self.queue.popleft()
                self.dropped_samples += 1
                logger.warning("Queue full, dropped sample")
            self._enqueue_sequence += 1
            self.queue.append(
                _QueuedSample(
                    sample=sample,
                    enqueued_at=time.monotonic(),
                    sequence=self._enqueue_sequence,
                    uid=incoming_uid,
                    min_version=incoming_min_version,
                    max_version=incoming_max_version,
                )
            )
            self.total_produced += 1

            # Notify waiting consumers
            self._consumer_condition.notify_all()

            if self.total_produced % 100 == 0:
                print(f"MessageQueue stats: produced={self.total_produced}, queue_size={len(self.queue)}")
            return QueuePutReceipt(
                incoming_admitted=True,
                evicted_uid=evicted.uid if evicted is not None else None,
                evicted_min_version=evicted.min_version if evicted is not None else None,
                evicted_max_version=evicted.max_version if evicted is not None else None,
                enqueue_sequence=self._enqueue_sequence,
                **incoming_metadata,
            )

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
                return None, 0

            # Get one sample
            queued_sample = self.queue.popleft()
            self.total_consumed += 1
            self.last_dequeue_residence_s = max(0.0, time.monotonic() - queued_sample.enqueued_at)
            return queued_sample.sample, len(self.queue)

    async def get_queue_size(self) -> int:
        """Get current queue length"""
        async with self._lock:
            return len(self.queue)

    async def get_statistics(self) -> dict[str, Any]:
        """Get queue statistics"""
        async with self._lock:
            return {
                "queue_size": len(self.queue),
                "total_produced": self.total_produced,
                "total_consumed": self.total_consumed,
                "dropped_samples": self.dropped_samples,
                "max_queue_size": self.max_queue_size,
                "closed": not self.running,
                "real_enqueued": self.total_produced,
                "real_consumed": self.total_consumed,
                "real_evicted": self.dropped_samples,
                "real_cleared": self.real_cleared,
                "real_resident": len(self.queue),
                "control_signals_enqueued": 0,
                "last_dequeue_residence_s": self.last_dequeue_residence_s,
            }

    async def clear_queue(self):
        """Clear the queue"""
        async with self._lock:
            cleared_count = len(self.queue)
            self.queue.clear()
            self.real_cleared += cleared_count
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
                sample = self.queue[0].sample
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

    async def put_sample(
        self,
        sample: Any,
        *,
        uid: Any = None,
        min_version: Any = None,
        max_version: Any = None,
    ) -> QueuePutReceipt:
        """Put batch into queue (async)"""
        future = self.queue_actor.put_sample.remote(
            sample,
            uid=uid,
            min_version=min_version,
            max_version=max_version,
        )
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
