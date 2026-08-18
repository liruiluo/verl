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

"""Contract for fully-async scheduler horizon and initialization ordering.

The rollouter reports parameter-publication cycles. A cycle contains
``trigger_parameter_sync_step`` learner collections, while TrainingWorker
advances its LR scheduler once after all optimizer mini-batches in one
collection. Therefore actor/critic optimizer configs must receive the
learner-collection/LR-scheduler horizon before engine construction. Optimizer
mini-batch calls are a separate count derived from real rows, batch size and
epochs.
"""

from __future__ import annotations

from unittest import IsolatedAsyncioTestCase, TestCase, mock

from omegaconf import OmegaConf

from verl.experimental.fully_async_policy.fully_async_trainer import FullyAsyncTrainer

SYNC_CYCLES = 25
LOCAL_UPDATES_PER_CYCLE = 4
LR_SCHEDULER_STEPS = SYNC_CYCLES * LOCAL_UPDATES_PER_CYCLE


def _local_actor_class(actor_class):
    """Return Ray's wrapped implementation class for dependency-only probes."""
    return actor_class.__ray_metadata__.modified_class


def _config():
    return OmegaConf.create(
        {
            "actor_rollout_ref": {
                "actor": {
                    "optim": {
                        "total_training_steps": -1,
                    }
                }
            },
            "critic": {
                "optim": {
                    "total_training_steps": -1,
                }
            },
        }
    )


def _trainer():
    trainer = object.__new__(_local_actor_class(FullyAsyncTrainer))
    trainer.config = _config()
    trainer.total_train_steps = None
    trainer.trigger_parameter_sync_step = LOCAL_UPDATES_PER_CYCLE
    trainer.progress_bar = None
    return trainer


class TestFullyAsyncSchedulerHorizon(TestCase):
    def test_sync_cycle_horizon_is_converted_to_lr_scheduler_steps(self) -> None:
        trainer = _trainer()

        trainer.set_total_train_steps(SYNC_CYCLES)

        # Progress/checkpoint/publication use sync cycles.
        self.assertEqual(trainer.total_train_steps, SYNC_CYCLES)
        # Actor and critic schedulers step once per learner collection.
        self.assertEqual(
            trainer.config.actor_rollout_ref.actor.optim.total_training_steps,
            LR_SCHEDULER_STEPS,
        )
        self.assertEqual(trainer.config.critic.optim.total_training_steps, LR_SCHEDULER_STEPS)


class TestFullyAsyncSchedulerInitialization(IsolatedAsyncioTestCase):
    async def test_worker_initialization_fails_closed_without_horizon(self) -> None:
        trainer = _trainer()
        trainer._init_resource_pools = mock.Mock()
        trainer._create_worker_classes = mock.Mock()
        trainer._init_worker_groups = mock.Mock()
        trainer._init_models = mock.Mock()

        with self.assertRaisesRegex(RuntimeError, "total.*train.*steps|horizon|scheduler"):
            await trainer.init_workers()

        trainer._create_worker_classes.assert_not_called()
        trainer._init_worker_groups.assert_not_called()
        trainer._init_models.assert_not_called()

    async def test_worker_configs_see_horizon_before_engine_construction(self) -> None:
        trainer = _trainer()
        trainer.set_total_train_steps(SYNC_CYCLES)
        seen = []

        trainer._init_resource_pools = mock.Mock()

        def _capture_worker_config() -> None:
            seen.append(
                (
                    trainer.config.actor_rollout_ref.actor.optim.total_training_steps,
                    trainer.config.critic.optim.total_training_steps,
                )
            )

        trainer._create_worker_classes = mock.Mock(side_effect=_capture_worker_config)
        trainer._init_worker_groups = mock.Mock()
        trainer._init_models = mock.Mock()

        await trainer.init_workers()

        self.assertEqual(seen, [(LR_SCHEDULER_STEPS, LR_SCHEDULER_STEPS)])
        trainer._init_worker_groups.assert_called_once_with()
        trainer._init_models.assert_called_once_with()


if __name__ == "__main__":
    import unittest

    unittest.main()
