# Copyright 2026 The verl team.
# Licensed under the Apache License, Version 2.0.

"""CPU contracts for generic fully-async terminal and update evidence."""

from __future__ import annotations

import json
import os
from copy import deepcopy
from types import SimpleNamespace
from unittest import TestCase, mock

import numpy as np
import torch
from omegaconf import OmegaConf
from tensordict import TensorDict

from verl.experimental.fully_async_policy import fully_async_main as main_module
from verl.experimental.fully_async_policy.fully_async_main import FullyAsyncTaskRunner
from verl.experimental.fully_async_policy.fully_async_trainer import FullyAsyncTrainer
from verl.experimental.fully_async_policy.runtime_evidence import compare_parameter_update_probes
from verl.experimental.separation.ray_trainer import SeparateRayPPOTrainer
from verl.protocol import DataProto
from verl.trainer.ppo.rollout_corr_helper import compute_bypass_log_prob_evidence
from verl.workers.engine_workers import collect_parameter_update_probe


def _local_actor_class(actor_class):
    return actor_class.__ray_metadata__.modified_class


class _Tokenizer:
    def batch_decode(self, rows, skip_special_tokens=True):
        del skip_special_tokens
        return [" ".join(str(int(token)) for token in row) for row in rows]


def _rollout_dump_batch() -> DataProto:
    return DataProto(
        batch=TensorDict(
            {
                "prompts": torch.tensor([[1, 2], [9, 9]], dtype=torch.long),
                "responses": torch.tensor([[3, 4], [8, 8]], dtype=torch.long),
                "token_level_scores": torch.tensor([[0.25, 0.75], [99.0, 99.0]]),
            },
            batch_size=2,
        ),
        non_tensor_batch={
            "is_padding": np.array([False, True]),
            "reward_model": np.array(
                [{"ground_truth": "real-ground-truth"}, {"ground_truth": "padding-ground-truth"}], dtype=object
            ),
            "evidence": np.array(
                [{"trace_id": "real", "nested": [1, {"ok": True}]}, {"trace_id": "padding"}], dtype=object
            ),
            "request_id": np.array(["request-real", "request-padding"], dtype=object),
            "secret": np.array(["must-not-leak", "padding-secret"], dtype=object),
        },
    )


class TestRealOnlyRolloutDump(TestCase):
    def test_real_dump_path_excludes_padding_and_preserves_exact_whitelist(self) -> None:
        trainer = object.__new__(_local_actor_class(FullyAsyncTrainer))
        trainer.tokenizer = _Tokenizer()
        trainer.global_steps = 7
        trainer.rollout_data_non_tensor_keys = ("evidence",)
        trainer._init_dump_executor()

        with self.subTest("real background JSONL writer"):
            from tempfile import TemporaryDirectory

            with TemporaryDirectory() as dump_dir:
                trainer._log_rollout_data(
                    _rollout_dump_batch(),
                    reward_extra_infos_dict={"reward_label": ["real-reward", "padding-reward"]},
                    timing_raw={},
                    rollout_data_dir=dump_dir,
                )
                trainer._shutdown_dump_executor()

                with open(os.path.join(dump_dir, "7.jsonl"), encoding="utf-8") as fp:
                    rows = [json.loads(line) for line in fp]

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["input"], "1 2")
        self.assertEqual(rows[0]["output"], "3 4")
        self.assertEqual(rows[0]["gts"], "real-ground-truth")
        self.assertEqual(rows[0]["reward_label"], "real-reward")
        self.assertEqual(rows[0]["request_id"], "request-real")
        self.assertEqual(rows[0]["evidence"], {"trace_id": "real", "nested": [1, {"ok": True}]})
        self.assertNotIn("secret", rows[0])

    def test_whitelist_has_configurable_and_hard_bounds(self) -> None:
        trainer_cls = _local_actor_class(FullyAsyncTrainer)
        self.assertEqual(trainer_cls._validate_rollout_data_non_tensor_keys(["a", "b"], 2), ("a", "b"))

        with self.assertRaisesRegex(ValueError, "maximum|at most"):
            trainer_cls._validate_rollout_data_non_tensor_keys(["a", "b"], 1)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            trainer_cls._validate_rollout_data_non_tensor_keys(["a", "a"], 2)
        with self.assertRaisesRegex(ValueError, "hard maximum"):
            trainer_cls._validate_rollout_data_non_tensor_keys([], 10_000)

    def test_malformed_padding_mask_fails_closed(self) -> None:
        trainer = object.__new__(_local_actor_class(FullyAsyncTrainer))
        trainer.tokenizer = _Tokenizer()
        trainer.global_steps = 1
        trainer.rollout_data_non_tensor_keys = ()
        batch = _rollout_dump_batch()
        batch.non_tensor_batch["is_padding"] = np.array([False])

        with self.assertRaisesRegex(ValueError, "is_padding"):
            trainer._log_rollout_data(batch, {}, {}, "/unused")

    def test_failed_rollout_dump_is_drained_before_finalization_returns(self) -> None:
        trainer = object.__new__(_local_actor_class(FullyAsyncTrainer))
        trainer._init_dump_executor()

        def fail_dump():
            raise OSError("rollout dump failed")

        trainer._dump_futures.append(trainer._dump_executor.submit(fail_dump))
        with self.assertRaisesRegex(OSError, "rollout dump failed"):
            trainer.finalize_runtime_evidence()

        self.assertTrue(trainer._dump_executor._shutdown)


def _bypass_batch(old_log_probs: torch.Tensor, rollout_log_probs: torch.Tensor) -> DataProto:
    return DataProto(
        batch=TensorDict(
            {
                "old_log_probs": old_log_probs,
                "rollout_log_probs": rollout_log_probs,
                "response_mask": torch.tensor([[1, 1, 0], [1, 1, 1]], dtype=torch.long),
            },
            batch_size=2,
        ),
        non_tensor_batch={"is_padding": np.array([False, True])},
    )


class TestBypassLogProbEvidence(TestCase):
    def test_counts_real_tokens_and_ignores_masked_and_padding_rows(self) -> None:
        rollout = torch.tensor([[-0.1, -0.2, -9.0], [-1.0, -2.0, -3.0]])
        old = rollout.clone()
        old[0, 2] = 123.0
        old[1] = torch.tensor([7.0, 8.0, 9.0])

        evidence = compute_bypass_log_prob_evidence(_bypass_batch(old, rollout))

        self.assertEqual(evidence["rollout_corr/bypass_real_token_count"], 2)
        self.assertEqual(evidence["rollout_corr/bypass_max_abs_diff"], 0.0)

    def test_real_token_mismatch_fails(self) -> None:
        rollout = torch.tensor([[-0.1, -0.2, -9.0], [-1.0, -2.0, -3.0]])
        old = rollout.clone()
        old[0, 1] += 0.01

        with self.assertRaisesRegex(RuntimeError, "old_log_probs.*rollout_log_probs|mismatch"):
            compute_bypass_log_prob_evidence(_bypass_batch(old, rollout))

    def test_trainer_persists_identity_before_actor_dispatch(self) -> None:
        rollout = torch.tensor([[-0.1, -0.2, -9.0], [-1.0, -2.0, -3.0]])
        trainer = object.__new__(_local_actor_class(FullyAsyncTrainer))
        trainer.config = SimpleNamespace(
            algorithm={"rollout_correction": {"bypass_mode": True}},
            actor_rollout_ref=SimpleNamespace(
                actor=SimpleNamespace(ppo_mini_batch_size=2),
                rollout=SimpleNamespace(n=1),
            ),
        )
        trainer.parameter_update_probe_config = {"enabled": False}
        trainer.latest_bypass_log_prob_evidence = {}
        trainer.metrics = {}
        batch = _bypass_batch(rollout.clone(), rollout)

        def assert_evidence_is_already_persisted(_trainer, dispatched_batch):
            self.assertIs(dispatched_batch, batch)
            self.assertEqual(_trainer.metrics["rollout_corr/bypass_real_token_count"], 2)
            return "actor-output"

        with mock.patch.object(
            SeparateRayPPOTrainer,
            "_update_actor",
            autospec=True,
            side_effect=assert_evidence_is_already_persisted,
        ):
            self.assertEqual(trainer._update_actor(batch), "actor-output")

        self.assertEqual(
            trainer.latest_bypass_log_prob_evidence,
            {
                "rollout_corr/bypass_real_token_count": 2,
                "rollout_corr/bypass_max_abs_diff": 0.0,
            },
        )


def _probe_payload(values: list[float]) -> list[dict]:
    return [
        {
            "max_parameters": 1,
            "max_elements_per_parameter": len(values),
            "sampled_parameters": 1,
            "sampled_elements": len(values),
            "parameters": [
                {
                    "name": "weight",
                    "numel": 8,
                    "sample_indices": list(range(len(values))),
                    "values": values,
                }
            ],
        }
    ]


class TestBoundedParameterUpdateProbe(TestCase):
    def test_unified_worker_sampler_is_hard_bounded_without_state_dict(self) -> None:
        module = torch.nn.Sequential(torch.nn.Linear(4, 3), torch.nn.Linear(3, 2))
        module.state_dict = mock.Mock(side_effect=AssertionError("full model copy forbidden"))

        probe = collect_parameter_update_probe(module, max_parameters=1, max_elements_per_parameter=2)

        self.assertEqual(probe["sampled_parameters"], 1)
        self.assertLessEqual(probe["sampled_elements"], 2)
        self.assertEqual(len(probe["parameters"]), 1)
        self.assertLessEqual(len(probe["parameters"][0]["values"]), 2)
        module.state_dict.assert_not_called()

        with self.assertRaisesRegex(ValueError, "hard maximum"):
            collect_parameter_update_probe(module, max_parameters=10_000, max_elements_per_parameter=2)
        with self.assertRaisesRegex(ValueError, "hard maximum"):
            collect_parameter_update_probe(module, max_parameters=1, max_elements_per_parameter=10_000)

    def test_sampler_spreads_across_trainable_parameters(self) -> None:
        class SpreadModule(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                for index in range(5):
                    self.register_parameter(f"p{index}", torch.nn.Parameter(torch.tensor([float(index)])))

        module = SpreadModule()
        before = collect_parameter_update_probe(module, max_parameters=3, max_elements_per_parameter=1)
        module.p4.data.add_(1.0)
        after = collect_parameter_update_probe(module, max_parameters=3, max_elements_per_parameter=1)

        self.assertEqual([record["name"] for record in before["parameters"]], ["p0", "p2", "p4"])
        comparison = compare_parameter_update_probes([before], [after], atol=0.0)
        self.assertTrue(comparison["changed"])
        self.assertEqual(comparison["changed_elements"], 1)

    def test_sampler_spreads_within_each_local_shard(self) -> None:
        module = torch.nn.Module()
        module.register_parameter("weight", torch.nn.Parameter(torch.arange(9.0)))

        before = collect_parameter_update_probe(module, max_parameters=1, max_elements_per_parameter=3)
        module.weight.data[-1].add_(1.0)
        after = collect_parameter_update_probe(module, max_parameters=1, max_elements_per_parameter=3)

        self.assertEqual(before["parameters"][0]["sample_indices"], [0, 4, 8])
        self.assertEqual(before["parameters"][0]["values"], [0.0, 4.0, 8.0])
        comparison = compare_parameter_update_probes([before], [after], atol=0.0)
        self.assertTrue(comparison["changed"])
        self.assertEqual(comparison["changed_elements"], 1)

    def test_comparison_binds_exact_sample_positions(self) -> None:
        before = _probe_payload([1.0, 2.0])
        after = deepcopy(before)
        after[0]["parameters"][0]["sample_indices"] = [0, 2]

        with self.assertRaisesRegex(ValueError, "structure changed"):
            compare_parameter_update_probes(before, after, atol=0.0)

        invalid = deepcopy(before)
        invalid[0]["parameters"][0]["sample_indices"] = [0, 8]
        with self.assertRaisesRegex(ValueError, "out-of-range"):
            compare_parameter_update_probes(invalid, before, atol=0.0)

    def test_actor_and_critic_zero_and_nonzero_behavior(self) -> None:
        trainer = object.__new__(_local_actor_class(FullyAsyncTrainer))
        trainer.metrics = {}
        trainer.latest_parameter_update_probe = {}
        trainer.parameter_update_probe_config = {
            "enabled": True,
            "max_parameters": 1,
            "max_elements_per_parameter": 2,
            "atol": 0.0,
            "require_change": True,
        }

        unchanged = _probe_payload([1.0, 2.0])
        changed = _probe_payload([1.0, 3.0])
        for role in ("actor", "critic"):
            with self.subTest(role=role, update="zero"):
                with self.assertRaisesRegex(RuntimeError, role):
                    trainer._record_parameter_update_probe(role, unchanged, deepcopy(unchanged))
            with self.subTest(role=role, update="nonzero"):
                evidence = trainer._record_parameter_update_probe(role, unchanged, changed)
                self.assertTrue(evidence["changed"])
                self.assertEqual(evidence["changed_elements"], 1)
                self.assertEqual(evidence["max_abs_diff"], 1.0)

    def test_disabled_probe_performs_no_worker_rpc(self) -> None:
        trainer = object.__new__(_local_actor_class(FullyAsyncTrainer))
        trainer.parameter_update_probe_config = {"enabled": False}
        worker_group = mock.Mock()

        self.assertIsNone(trainer._capture_parameter_update_probe(worker_group))
        worker_group.parameter_update_probe.assert_not_called()

    def test_comparison_rejects_oversized_worker_payload(self) -> None:
        payload = _probe_payload([1.0])
        payload[0]["parameters"] *= 65
        payload[0]["sampled_parameters"] = 65
        with self.assertRaisesRegex(ValueError, "hard maximum|payload"):
            compare_parameter_update_probes(payload, payload, atol=0.0)

    def test_actor_and_critic_updates_are_bracketed_by_worker_probes(self) -> None:
        trainer = object.__new__(_local_actor_class(FullyAsyncTrainer))
        trainer.config = SimpleNamespace(
            actor_rollout_ref=SimpleNamespace(
                actor=SimpleNamespace(ppo_mini_batch_size=2),
                rollout=SimpleNamespace(n=1),
            ),
            critic=SimpleNamespace(ppo_mini_batch_size=2),
        )
        trainer.metrics = {}
        trainer.latest_parameter_update_probe = {}
        trainer.parameter_update_probe_config = {
            "enabled": True,
            "max_parameters": 1,
            "max_elements_per_parameter": 2,
            "atol": 0.0,
            "require_change": True,
        }
        actor_worker_group = mock.Mock()
        critic_worker_group = mock.Mock()
        actor_worker_group.parameter_update_probe.side_effect = [
            _probe_payload([1.0, 2.0]),
            _probe_payload([1.0, 3.0]),
        ]
        critic_worker_group.parameter_update_probe.side_effect = [
            _probe_payload([4.0, 5.0]),
            _probe_payload([6.0, 5.0]),
        ]
        trainer.actor_rollout_wg = actor_worker_group
        trainer.critic_wg = critic_worker_group
        batch = DataProto(batch=TensorDict({}, batch_size=2))

        with (
            mock.patch.object(SeparateRayPPOTrainer, "_update_actor", return_value="actor-output"),
            mock.patch.object(SeparateRayPPOTrainer, "_update_critic", return_value="critic-output"),
        ):
            self.assertEqual(trainer._update_actor(batch), "actor-output")
            self.assertEqual(trainer._update_critic(batch), "critic-output")

        self.assertEqual(actor_worker_group.parameter_update_probe.call_count, 2)
        self.assertEqual(critic_worker_group.parameter_update_probe.call_count, 2)
        self.assertEqual(trainer.latest_parameter_update_probe["actor"]["changed_elements"], 1)
        self.assertEqual(trainer.latest_parameter_update_probe["critic"]["changed_elements"], 1)


class _RemoteMethod:
    def __init__(self, callback):
        self.callback = callback

    def remote(self, *args, **kwargs):
        return self.callback(*args, **kwargs)


class _TrainerHandle:
    def __init__(self, statistics, finalize_failure=None):
        self.statistics = statistics
        self.finalize_failure = finalize_failure
        self.finalize_calls = 0
        self.finalize_runtime_evidence = _RemoteMethod(self._finalize)
        self.get_statistics = _RemoteMethod(self._statistics)

    def _finalize(self):
        self.finalize_calls += 1
        if self.finalize_failure is not None:
            raise self.finalize_failure

    def _statistics(self):
        if isinstance(self.statistics, Exception):
            raise self.statistics
        return deepcopy(self.statistics)


class _RollouterHandle:
    def __init__(self, statistics, after_statistics=None):
        self.statistics = statistics
        self.after_statistics = after_statistics if after_statistics is not None else statistics
        self.calls = 0
        self.get_statistics = _RemoteMethod(self._statistics)

    def _statistics(self):
        self.calls += 1
        value = self.statistics if self.calls == 1 else self.after_statistics
        if isinstance(value, Exception):
            raise value
        return deepcopy(value)


class _QueueClient:
    def __init__(self):
        self.statistics = {
            "queue_size": 2,
            "real_enqueued": 3,
            "real_consumed": 1,
            "real_evicted": 0,
            "real_cleared": 0,
            "real_resident": 2,
            "closed": True,
        }

    async def get_statistics(self):
        return deepcopy(self.statistics)

    async def clear_queue(self):
        self.statistics["real_cleared"] += self.statistics["real_resident"]
        self.statistics["real_resident"] = 0
        self.statistics["queue_size"] = 0


def _runtime_config(receipt_path: str):
    return OmegaConf.create(
        {
            "async_training": {"runtime_receipt_path": receipt_path},
            "trainer": {
                "project_name": "runtime-evidence-tests",
                "experiment_name": "terminal",
                "default_local_dir": os.path.dirname(receipt_path),
            },
        }
    )


def _read_receipt(path: str) -> dict:
    with open(path, encoding="utf-8") as fp:
        wrapper = json.loads(fp.read())
    return wrapper["data"]


class TestAtomicTerminalRuntimeReceipt(TestCase):
    def _run_case(self, tmpdir, *, terminal_underfill_events=0, failure=None):
        receipt_path = os.path.join(tmpdir, "runtime-receipt.jsonl")
        runner = object.__new__(_local_actor_class(FullyAsyncTaskRunner))
        runner.__init__()
        trainer = _TrainerHandle(
            {
                "current_param_version": 2,
                "total_train_steps": 2,
                "terminal_underfill_events": terminal_underfill_events,
                "terminal_underfill_samples": int(terminal_underfill_events > 0),
            }
        )
        rollouter = _RollouterHandle({"count/total_generated_samples": 3})
        queue = _QueueClient()

        def initialize(config):
            runner.components.update(
                config=config,
                trainer=trainer,
                rollouter=rollouter,
                message_queue_client=queue,
            )

        runner._initialize_components = mock.Mock(side_effect=initialize)
        runner._run_training_loop = mock.Mock(side_effect=failure)
        config = _runtime_config(receipt_path)

        with mock.patch.object(main_module.ray, "get", side_effect=lambda value: value):
            if failure is None:
                runner.run(config)
            else:
                with self.assertRaisesRegex(type(failure), str(failure)):
                    runner.run(config)

        return receipt_path, _read_receipt(receipt_path), trainer, queue

    def test_success_receipt_is_atomic_and_conserves_queue(self) -> None:
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as tmpdir:
            receipt_path, receipt, trainer, queue = self._run_case(tmpdir)
            leftovers = [name for name in os.listdir(tmpdir) if name != os.path.basename(receipt_path)]

        self.assertEqual(receipt["outcome"], "success")
        self.assertEqual(receipt["status"], "completed")
        self.assertIsNone(receipt["exception"])
        self.assertEqual(trainer.finalize_calls, 1)
        self.assertEqual(queue.statistics["real_cleared"], 2)
        self.assertEqual(leftovers, [])
        self.assertTrue(receipt["queue_conservation"]["before_clear"])
        self.assertTrue(receipt["queue_conservation"]["after_clear"])
        self.assertTrue(receipt["queue_conservation"]["clear_delta_matches_resident"])
        self.assertEqual(
            receipt["snapshots"]["before_clear"]["queue"]["statistics"]["real_resident"],
            2,
        )
        self.assertEqual(
            receipt["snapshots"]["after_clear"]["queue"]["statistics"]["real_resident"],
            0,
        )

    def test_terminal_underfill_receipt_is_partial(self) -> None:
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as tmpdir:
            _path, receipt, _trainer, _queue = self._run_case(tmpdir, terminal_underfill_events=1)

        self.assertEqual(receipt["outcome"], "terminal_underfill")
        self.assertEqual(receipt["status"], "partial")
        self.assertIsNone(receipt["exception"])

    def test_component_crash_wins_over_underfill_and_is_classified(self) -> None:
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as tmpdir:
            _path, receipt, _trainer, _queue = self._run_case(
                tmpdir,
                terminal_underfill_events=1,
                failure=RuntimeError("component exploded"),
            )

        self.assertEqual(receipt["outcome"], "crash")
        self.assertEqual(receipt["status"], "failed")
        self.assertEqual(receipt["exception"]["type"], "RuntimeError")
        self.assertEqual(receipt["exception"]["message"], "component exploded")

    def test_initialization_crash_still_writes_unavailable_snapshots(self) -> None:
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as tmpdir:
            receipt_path = os.path.join(tmpdir, "init-crash.jsonl")
            runner = object.__new__(_local_actor_class(FullyAsyncTaskRunner))
            runner.__init__()
            runner._initialize_components = mock.Mock(side_effect=ValueError("init failed"))

            with self.assertRaisesRegex(ValueError, "init failed"):
                runner.run(_runtime_config(receipt_path))
            receipt = _read_receipt(receipt_path)

        self.assertEqual(receipt["outcome"], "crash")
        for boundary in ("before_clear", "after_clear"):
            for component in ("trainer", "rollouter", "queue"):
                record = receipt["snapshots"][boundary][component]
                self.assertFalse(record["available"])
                self.assertIn("error", record)

    def test_after_clear_statistics_failure_is_recorded_and_fails_run(self) -> None:
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as tmpdir:
            receipt_path = os.path.join(tmpdir, "stats-crash.jsonl")
            runner = object.__new__(_local_actor_class(FullyAsyncTaskRunner))
            runner.__init__()
            trainer = _TrainerHandle(
                {"current_param_version": 1, "total_train_steps": 1, "terminal_underfill_events": 0}
            )
            rollouter = _RollouterHandle(
                {"count/total_generated_samples": 1},
                after_statistics=RuntimeError("rollouter stats unavailable"),
            )

            def initialize(config):
                runner.components.update(
                    config=config,
                    trainer=trainer,
                    rollouter=rollouter,
                    message_queue_client=_QueueClient(),
                )

            runner._initialize_components = mock.Mock(side_effect=initialize)
            runner._run_training_loop = mock.Mock()
            with mock.patch.object(main_module.ray, "get", side_effect=lambda value: value):
                with self.assertRaisesRegex(RuntimeError, "rollouter stats unavailable"):
                    runner.run(_runtime_config(receipt_path))
            receipt = _read_receipt(receipt_path)

        self.assertEqual(receipt["outcome"], "crash")
        record = receipt["snapshots"]["after_clear"]["rollouter"]
        self.assertFalse(record["available"])
        self.assertEqual(record["error"]["type"], "RuntimeError")

    def test_corrupt_queue_accounting_cannot_publish_success(self) -> None:
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as tmpdir:
            receipt_path = os.path.join(tmpdir, "queue-corruption.jsonl")
            runner = object.__new__(_local_actor_class(FullyAsyncTaskRunner))
            runner.__init__()
            queue = _QueueClient()
            queue.statistics["real_enqueued"] += 1

            def initialize(config):
                runner.components.update(
                    config=config,
                    trainer=_TrainerHandle(
                        {"current_param_version": 1, "total_train_steps": 1, "terminal_underfill_events": 0}
                    ),
                    rollouter=_RollouterHandle({"count/total_generated_samples": 1}),
                    message_queue_client=queue,
                )

            runner._initialize_components = mock.Mock(side_effect=initialize)
            runner._run_training_loop = mock.Mock()
            with mock.patch.object(main_module.ray, "get", side_effect=lambda value: value):
                with self.assertRaisesRegex(RuntimeError, "queue conservation"):
                    runner.run(_runtime_config(receipt_path))
            receipt = _read_receipt(receipt_path)

        self.assertEqual(receipt["outcome"], "crash")
        self.assertFalse(receipt["queue_conservation"]["before_clear"])
        self.assertFalse(receipt["queue_conservation"]["after_clear"])

    def test_malformed_statistics_cannot_publish_success(self) -> None:
        from tempfile import TemporaryDirectory

        class _MalformedQueue(_QueueClient):
            async def get_statistics(self):
                return []

            async def clear_queue(self):
                return None

        with TemporaryDirectory() as tmpdir:
            receipt_path = os.path.join(tmpdir, "malformed-statistics.jsonl")
            runner = object.__new__(_local_actor_class(FullyAsyncTaskRunner))
            runner.__init__()

            def initialize(config):
                runner.components.update(
                    config=config,
                    trainer=_TrainerHandle(
                        {"current_param_version": 1, "total_train_steps": 1, "terminal_underfill_events": 0}
                    ),
                    rollouter=_RollouterHandle({"count/total_generated_samples": 1}),
                    message_queue_client=_MalformedQueue(),
                )

            runner._initialize_components = mock.Mock(side_effect=initialize)
            runner._run_training_loop = mock.Mock()
            with mock.patch.object(main_module.ray, "get", side_effect=lambda value: value):
                with self.assertRaisesRegex(TypeError, "queue statistics must be a mapping"):
                    runner.run(_runtime_config(receipt_path))
            receipt = _read_receipt(receipt_path)

        self.assertEqual(receipt["outcome"], "crash")
        for boundary in ("before_clear", "after_clear"):
            queue_record = receipt["snapshots"][boundary]["queue"]
            self.assertFalse(queue_record["available"])
            self.assertEqual(queue_record["error"]["type"], "TypeError")

    def test_training_exception_survives_dump_finalization_failure(self) -> None:
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as tmpdir:
            receipt_path = os.path.join(tmpdir, "double-failure.jsonl")
            runner = object.__new__(_local_actor_class(FullyAsyncTaskRunner))
            runner.__init__()
            trainer = _TrainerHandle(
                {"current_param_version": 1, "total_train_steps": 2, "terminal_underfill_events": 0},
                finalize_failure=OSError("dump drain failed"),
            )

            def initialize(config):
                runner.components.update(
                    config=config,
                    trainer=trainer,
                    rollouter=_RollouterHandle({"count/total_generated_samples": 1}),
                    message_queue_client=_QueueClient(),
                )

            runner._initialize_components = mock.Mock(side_effect=initialize)
            runner._run_training_loop = mock.Mock(side_effect=RuntimeError("training failed first"))
            with mock.patch.object(main_module.ray, "get", side_effect=lambda value: value):
                with self.assertRaisesRegex(RuntimeError, "training failed first"):
                    runner.run(_runtime_config(receipt_path))
            receipt = _read_receipt(receipt_path)

        self.assertEqual(receipt["exception"]["type"], "RuntimeError")
        self.assertEqual(receipt["exception"]["message"], "training failed first")
        self.assertEqual(receipt["finalization_errors"][0]["stage"], "trainer.rollout_dump")


if __name__ == "__main__":
    import unittest

    unittest.main()
