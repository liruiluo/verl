# Copyright 2026 The verl team.
# Licensed under the Apache License, Version 2.0.

"""Focused regression for #6987 explicit-or-derived horizon semantics."""

from __future__ import annotations

from types import SimpleNamespace
from unittest import IsolatedAsyncioTestCase, TestCase, mock

from omegaconf import OmegaConf

from verl.experimental.fully_async_policy import fully_async_main as main_module
from verl.experimental.fully_async_policy.fully_async_main import FullyAsyncTaskRunner
from verl.experimental.fully_async_policy.fully_async_trainer import FullyAsyncTrainer


class _StopAfterHorizon(Exception):
    pass


class _RemoteMethod:
    def __init__(self, value):
        self.value = value

    def remote(self, *args, **kwargs):
        return self.value


class _RollouterHandle:
    def __init__(self, derived_steps: int):
        self.get_total_train_steps = _RemoteMethod(derived_steps)


class _RollouterFactory:
    def __init__(self):
        self.received_rollout_steps = None

    def remote(self, config, **kwargs):
        self.received_rollout_steps = config.rollout.total_rollout_steps
        samples_per_publication = (
            config.actor_rollout_ref.actor.ppo_mini_batch_size
            * config.async_training.require_batches
            * config.async_training.trigger_parameter_sync_step
        )
        return _RollouterHandle(self.received_rollout_steps // samples_per_publication)


def _config(total_training_steps):
    return OmegaConf.create(
        {
            "actor_rollout_ref": {
                "model": {"path": "/tmp/model", "use_shm": False},
                "actor": {"ppo_mini_batch_size": 2},
            },
            "async_training": {"require_batches": 1, "trigger_parameter_sync_step": 2},
            "data": {"trust_remote_code": False},
            "rollout": {"total_rollout_steps": 100},
            "trainer": {"device": "cpu", "total_training_steps": total_training_steps},
        }
    )


class TestExplicitHorizonResolution(TestCase):
    @staticmethod
    def _resolve_until_trainer_creation(total_training_steps):
        runner_cls = FullyAsyncTaskRunner.__ray_metadata__.modified_class
        runner = object.__new__(runner_cls)
        runner.components = {}
        runner._create_trainer = mock.Mock(side_effect=_StopAfterHorizon)
        rollouter_factory = _RollouterFactory()

        with (
            mock.patch.object(main_module, "copy_to_local", return_value="/tmp/model"),
            mock.patch.object(main_module, "create_role_worker_mapping", return_value=({}, object())),
            mock.patch.object(main_module, "FullyAsyncRollouter", rollouter_factory),
            mock.patch.object(main_module.ray, "get", side_effect=lambda value: value),
            mock.patch("verl.utils.hf_tokenizer", return_value=object()),
            mock.patch("verl.utils.hf_processor", return_value=None),
        ):
            with TestCase().assertRaises(_StopAfterHorizon):
                runner._initialize_components(_config(total_training_steps))
        return runner.components["total_train_steps"], rollouter_factory.received_rollout_steps

    def test_explicit_positive_horizon_is_preserved(self) -> None:
        total_train_steps, rollout_steps = self._resolve_until_trainer_creation(7)
        self.assertEqual(total_train_steps, 7)
        self.assertEqual(rollout_steps, 28)

    def test_absent_horizon_uses_rollouter_derivation(self) -> None:
        total_train_steps, rollout_steps = self._resolve_until_trainer_creation(None)
        self.assertEqual(total_train_steps, 25)
        self.assertEqual(rollout_steps, 100)

    def test_nonpositive_explicit_horizon_fails_closed(self) -> None:
        runner_cls = FullyAsyncTaskRunner.__ray_metadata__.modified_class
        runner = object.__new__(runner_cls)
        runner.components = {}

        with (
            mock.patch.object(main_module, "copy_to_local", return_value="/tmp/model"),
            mock.patch.object(main_module, "create_role_worker_mapping", return_value=({}, object())),
            mock.patch.object(main_module, "FullyAsyncRollouter", _RollouterFactory()),
            mock.patch.object(main_module.ray, "get", side_effect=lambda value: value),
            mock.patch("verl.utils.hf_tokenizer", return_value=object()),
            mock.patch("verl.utils.hf_processor", return_value=None),
            self.assertRaisesRegex(RuntimeError, "positive|greater than zero"),
        ):
            runner._initialize_components(_config(0))


class TestTrainerExecutionBound(IsolatedAsyncioTestCase):
    async def test_trainer_stops_at_explicit_publication_horizon_without_waiting_for_queue_shutdown(self) -> None:
        trainer_cls = FullyAsyncTrainer.__ray_metadata__.modified_class
        trainer = object.__new__(trainer_cls)
        trainer.message_queue_client = object()
        trainer.rollouter = object()
        trainer.global_steps = 0
        trainer.current_param_version = 0
        trainer.total_train_steps = 2
        trainer.local_trigger_step = 1
        trainer.progress_bar = mock.Mock()
        trainer.config = SimpleNamespace(trainer=SimpleNamespace(test_freq=1))
        trainer._fit_update_weights = mock.AsyncMock(return_value=None)
        trainer._fit_validate = mock.AsyncMock()
        trainer._fit_save_checkpoint = mock.Mock()

        calls = 0

        async def fit_step():
            nonlocal calls
            calls += 1
            if calls > 2:
                raise AssertionError("trainer ran past explicit horizon")
            trainer.current_param_version += 1

        trainer.fit_step = fit_step
        await trainer.fit()

        self.assertEqual(calls, 2)
        self.assertEqual(trainer.current_param_version, 2)
        trainer._fit_save_checkpoint.assert_called_once_with(force=True)


if __name__ == "__main__":
    import unittest

    unittest.main()
