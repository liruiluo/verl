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
import os
import socket
import threading
from pprint import pprint

import hydra
import ray
from omegaconf import OmegaConf, open_dict

from verl.experimental.fully_async_policy.fully_async_rollouter import FullyAsyncRollouter
from verl.experimental.fully_async_policy.fully_async_trainer import FullyAsyncTrainer
from verl.experimental.fully_async_policy.message_queue import MessageQueue, MessageQueueClient
from verl.experimental.fully_async_policy.runtime_evidence import (
    atomic_write_runtime_receipt,
    classify_exception,
    queue_clear_delta_matches,
    queue_conservation_holds,
    utc_timestamp,
)
from verl.experimental.reward_loop import migrate_legacy_reward_impl
from verl.experimental.separation.utils import create_resource_pool_manager, create_role_worker_mapping
from verl.trainer.ppo.utils import Role
from verl.utils.device import auto_set_device
from verl.utils.fs import copy_to_local


@ray.remote(num_cpus=1)
class FullyAsyncTaskRunner:
    """
    Ray remote class for executing distributed PPO training tasks.
    """

    def __init__(self):
        self.running = False
        self.components = {}
        self.shutdown_event = threading.Event()
        self.run_started_at = None

    def run(self, config):
        print("[ASYNC MAIN] Starting fully async PPO training...")
        self.run_started_at = utc_timestamp()
        self.components["config"] = config
        primary_exception = None
        primary_traceback = None
        try:
            self._initialize_components(config)
            self._run_training_loop()
        except BaseException as exception:
            primary_exception = exception
            primary_traceback = exception.__traceback__

        finalization_exception = None
        try:
            finalization_exception = self._finalize_runtime_evidence(config, primary_exception)
        except BaseException as exception:
            finalization_exception = exception

        if primary_exception is not None:
            if finalization_exception is not None:
                print(
                    "[ASYNC MAIN] Runtime-evidence finalization also failed: "
                    f"{classify_exception(finalization_exception)}"
                )
            raise primary_exception.with_traceback(primary_traceback)
        if finalization_exception is not None:
            raise finalization_exception

    @staticmethod
    def _unavailable_statistics_record(exception: BaseException) -> dict:
        return {
            "timestamp": utc_timestamp(),
            "available": False,
            "error": classify_exception(exception),
        }

    def _capture_statistics(self, component_name: str) -> tuple[dict, BaseException | None]:
        component_key = "message_queue_client" if component_name == "queue" else component_name
        component = self.components.get(component_key)
        if component is None:
            exception = RuntimeError(
                f"{component_name} statistics unavailable because the component is not initialized"
            )
            return self._unavailable_statistics_record(exception), exception
        try:
            if component_name == "queue":
                statistics = asyncio.run(component.get_statistics())
            else:
                statistics = ray.get(component.get_statistics.remote())
            if not isinstance(statistics, dict):
                raise TypeError(f"{component_name} statistics must be a mapping, got {type(statistics).__name__}")
            return {
                "timestamp": utc_timestamp(),
                "available": True,
                "statistics": statistics,
            }, None
        except BaseException as exception:
            return self._unavailable_statistics_record(exception), exception

    def _capture_statistics_boundary(self, boundary: str) -> tuple[dict, list[tuple[str, BaseException]]]:
        snapshot = {"timestamp": utc_timestamp()}
        errors = []
        for component_name in ("trainer", "rollouter", "queue"):
            record, exception = self._capture_statistics(component_name)
            snapshot[component_name] = record
            if exception is not None:
                errors.append((f"{boundary}.{component_name}", exception))
        return snapshot, errors

    @staticmethod
    def _statistics_from_record(record: dict) -> dict | None:
        if not record.get("available", False):
            return None
        statistics = record.get("statistics")
        return statistics if isinstance(statistics, dict) else None

    @staticmethod
    def _runtime_receipt_path(config) -> str:
        configured_path = config.async_training.get("runtime_receipt_path", None)
        if configured_path:
            return configured_path
        return os.path.join(config.trainer.default_local_dir, "fully_async_runtime_receipt.jsonl")

    def _finalize_runtime_evidence(self, config, primary_exception: BaseException | None) -> BaseException | None:
        finalization_started_at = utc_timestamp()
        finalization_errors: list[tuple[str, BaseException]] = []

        trainer = self.components.get("trainer")
        if trainer is not None:
            try:
                ray.get(trainer.finalize_runtime_evidence.remote())
            except BaseException as exception:
                finalization_errors.append(("trainer.rollout_dump", exception))

        before_clear, boundary_errors = self._capture_statistics_boundary("before_clear")
        finalization_errors.extend(boundary_errors)

        message_queue_client = self.components.get("message_queue_client")
        if message_queue_client is not None:
            try:
                asyncio.run(message_queue_client.clear_queue())
            except BaseException as exception:
                finalization_errors.append(("queue.clear", exception))

        after_clear, boundary_errors = self._capture_statistics_boundary("after_clear")
        finalization_errors.extend(boundary_errors)

        before_queue_statistics = self._statistics_from_record(before_clear["queue"])
        after_queue_statistics = self._statistics_from_record(after_clear["queue"])
        before_queue_conserved = queue_conservation_holds(before_queue_statistics)
        after_queue_conserved = queue_conservation_holds(after_queue_statistics)
        clear_delta_matches_resident = queue_clear_delta_matches(before_queue_statistics, after_queue_statistics)
        if before_queue_statistics is not None and not before_queue_conserved:
            finalization_errors.append(
                ("queue.conservation.before_clear", RuntimeError("queue conservation failed before clear"))
            )
        if after_queue_statistics is not None and not after_queue_conserved:
            finalization_errors.append(
                ("queue.conservation.after_clear", RuntimeError("queue conservation failed after clear"))
            )
        if (
            before_queue_statistics is not None
            and after_queue_statistics is not None
            and not clear_delta_matches_resident
        ):
            finalization_errors.append(
                ("queue.conservation.clear_delta", RuntimeError("queue clear delta does not match resident samples"))
            )

        effective_exception = primary_exception
        if effective_exception is None and finalization_errors:
            effective_exception = finalization_errors[0][1]

        trainer_statistics = self._statistics_from_record(before_clear["trainer"])
        if trainer_statistics is None:
            trainer_statistics = self._statistics_from_record(after_clear["trainer"])
        terminal_underfill_events = (trainer_statistics or {}).get("terminal_underfill_events", 0)
        if effective_exception is not None:
            outcome, status = "crash", "failed"
        elif terminal_underfill_events:
            outcome, status = "terminal_underfill", "partial"
        else:
            outcome, status = "success", "completed"

        receipt = {
            "schema_version": 1,
            "outcome": outcome,
            "status": status,
            "exception": classify_exception(effective_exception) if effective_exception is not None else None,
            "timestamps": {
                "run_started_at": self.run_started_at,
                "finalization_started_at": finalization_started_at,
                "finalized_at": utc_timestamp(),
            },
            "snapshots": {
                "before_clear": before_clear,
                "after_clear": after_clear,
            },
            "queue_conservation": {
                "before_clear": before_queue_conserved,
                "after_clear": after_queue_conserved,
                "clear_delta_matches_resident": clear_delta_matches_resident,
            },
            "finalization_errors": [
                {"stage": stage, "exception": classify_exception(exception)} for stage, exception in finalization_errors
            ],
            "trainer_step": (trainer_statistics or {}).get("current_param_version", 0),
        }
        atomic_write_runtime_receipt(
            self._runtime_receipt_path(config),
            receipt,
            project_name=config.trainer.project_name,
            experiment_name=config.trainer.experiment_name,
        )
        return finalization_errors[0][1] if finalization_errors else None

    def _initialize_components(self, config) -> None:
        print(f"[ASYNC MAIN] TaskRunner hostname: {socket.gethostname()}, PID: {os.getpid()}")
        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)

        print("[ASYNC MAIN] Initializing model and tokenizer...")
        local_path = copy_to_local(
            config.actor_rollout_ref.model.path, use_shm=config.actor_rollout_ref.model.get("use_shm", False)
        )
        from verl.utils import hf_processor, hf_tokenizer

        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)

        # Used for multimodal LLM, could be None
        processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)

        self.components["tokenizer"] = tokenizer
        self.components["processor"] = processor
        self.components["config"] = config

        print("[ASYNC MAIN] Creating worker mapping and resource pools...")
        role_worker_mapping, ray_worker_group_cls = create_role_worker_mapping(config)
        self.components["role_worker_mapping"] = role_worker_mapping
        self.components["ray_worker_group_cls"] = ray_worker_group_cls

        configured_train_steps = config.trainer.get("total_training_steps", None)
        if configured_train_steps is not None:
            if isinstance(configured_train_steps, bool) or configured_train_steps <= 0:
                raise RuntimeError(
                    "trainer.total_training_steps must be a positive integer when explicit; "
                    f"got {configured_train_steps!r}"
                )
            samples_per_publication = (
                config.actor_rollout_ref.actor.ppo_mini_batch_size
                * config.async_training.require_batches
                * config.async_training.trigger_parameter_sync_step
            )
            explicit_rollout_steps = configured_train_steps * samples_per_publication
            if int(explicit_rollout_steps) != explicit_rollout_steps:
                raise RuntimeError(
                    "Explicit training horizon does not map to an integral rollout-sample horizon: "
                    f"{configured_train_steps} * {samples_per_publication} = {explicit_rollout_steps}"
                )
            # Reuse the native producer horizon instead of introducing a second
            # stop channel.  FullyAsyncRollouter still applies the dataset/epoch
            # cap and the readback below fails closed if it cannot honor the
            # explicit trainer horizon.
            with open_dict(config.rollout):
                config.rollout.total_rollout_steps = int(explicit_rollout_steps)

        print("[ASYNC MAIN] Constructing FullyAsyncRollouter to resolve the scheduler horizon...")
        self.components["rollouter"] = FullyAsyncRollouter.remote(
            config=config,
            tokenizer=self.components["tokenizer"],
            processor=self.components["processor"],
            device_name=config.trainer.device,
        )
        derived_train_steps = ray.get(self.components["rollouter"].get_total_train_steps.remote())
        if configured_train_steps is not None and derived_train_steps != configured_train_steps:
            raise RuntimeError(
                "FullyAsyncRollouter cannot honor explicit trainer.total_training_steps under the configured "
                "dataset/epoch horizon: "
                f"requested {configured_train_steps}, resolved {derived_train_steps}"
            )
        total_train_steps = derived_train_steps
        if total_train_steps <= 0:
            raise RuntimeError(
                "trainer.total_training_steps must be positive when explicit, and the derived horizon must be "
                f"positive when it is absent; got {total_train_steps}"
            )
        self.components["rollouter_total_train_steps"] = derived_train_steps
        self.components["total_train_steps"] = total_train_steps

        print("[ASYNC MAIN] Creating FullyAsyncTrainer first (needed for hybrid worker group injection)...")
        self._create_trainer(config)

        print("[ASYNC MAIN] Injecting trainer's worker group into rollouter for hybrid replicas...")
        self._setup_hybrid_worker_group(config)

        print("[ASYNC MAIN] Initializing FullyAsyncRollouter...")
        self._create_rollouter(config)

        print("[ASYNC MAIN] Setting up rollouter reference on trainer")
        ray.get(self.components["trainer"].set_rollouter.remote(self.components["rollouter"]))

        # sync total_train_steps between rollouter and trainer
        total_train_steps = ray.get(self.components["rollouter"].get_total_train_steps.remote())
        print(f"total_train_steps {total_train_steps}")
        if total_train_steps != self.components["rollouter_total_train_steps"]:
            raise RuntimeError(
                "Precomputed rollouter horizon does not match the initialized rollouter: "
                f"{self.components['rollouter_total_train_steps']} != {total_train_steps}"
            )

        # max_queue_size
        max_queue_size = ray.get(self.components["rollouter"].get_max_queue_size.remote())
        print(f"[ASYNC MAIN] Creating MessageQueue... max_queue_size {max_queue_size}")
        message_queue = MessageQueue.remote(config, max_queue_size)
        message_queue_client = MessageQueueClient(message_queue)
        self.components["message_queue"] = message_queue
        self.components["message_queue_client"] = message_queue_client

        ray.get(self.components["rollouter"].set_message_queue_client.remote(self.components["message_queue_client"]))
        ray.get(self.components["trainer"].set_message_queue_client.remote(self.components["message_queue_client"]))

        # param_version resume from ckpt or default 0
        ray.get(self.components["trainer"].load_checkpoint.remote())
        ray.get(self.components["rollouter"].load_checkpoint.remote())

        print("[ASYNC MAIN] Param sync before fit..")
        ray.get(self.components["trainer"]._fit_update_weights.remote())

        if config.trainer.get("val_before_train", True):
            ray.get(self.components["trainer"]._fit_validate.remote(True))

        print("[ASYNC MAIN] All components initialized successfully")

    def _create_rollouter(self, config) -> None:
        print("[ASYNC MAIN] Starting rollouter worker initialization...")
        rollouter = self.components["rollouter"]

        # set_hybrid_worker_group must be called BEFORE init_workers() so that
        # _init_async_rollout_manager can pass the hybrid WG to ALM.create().
        if "hybrid_worker_group" in self.components:
            ray.get(rollouter.set_hybrid_worker_group.remote(self.components["hybrid_worker_group"]))
            print("[ASYNC MAIN] Hybrid worker group injected into rollouter")

        ray.get(rollouter.init_workers.remote())
        ray.get(rollouter.set_max_required_samples.remote())

        self.components["rollouter"] = rollouter
        print("[ASYNC MAIN] Rollouter created and initialized successfully")

    def _create_trainer(self, config) -> None:
        print("[ASYNC MAIN] Starting create trainer...")
        trainer_role_mapping = {
            role: worker_cls
            for role, worker_cls in self.components["role_worker_mapping"].items()
            if role != Role.Rollout
        }

        trainer = FullyAsyncTrainer.remote(
            config=config,
            tokenizer=self.components["tokenizer"],
            role_worker_mapping=trainer_role_mapping,
            resource_pool_manager=create_resource_pool_manager(config, roles=list(trainer_role_mapping.keys())),
            ray_worker_group_cls=self.components["ray_worker_group_cls"],
            device_name=config.trainer.device,
        )

        ray.get(trainer.set_total_train_steps.remote(self.components["total_train_steps"]))
        ray.get(trainer.init_workers.remote())
        self.components["trainer"] = trainer
        print("[ASYNC MAIN] FullyAsyncTrainer created and initialized successfully")

    def _setup_hybrid_worker_group(self, config) -> None:
        """Extract the trainer's actor_rollout_wg for hybrid rollout replicas.

        Hybrid replicas are needed when *either*:
        - ``use_trainer_do_validate`` is enabled (trainer-side validation), or
        - ``use_dynamic_resource_scheduling`` is enabled (dynamic resource scheduling).

        The extracted worker group is stored in ``self.components["hybrid_worker_group"]``
        and injected into the rollouter before ``init_workers()`` is called.
        """
        trainer = self.components["trainer"]
        needs_hybrid = config.async_training.use_trainer_do_validate or config.async_training.get(
            "use_dynamic_resource_scheduling", False
        )
        if needs_hybrid:
            trainer_wg = ray.get(trainer.get_actor_wg.remote())
            self.components["hybrid_worker_group"] = trainer_wg
            print(
                f"[ASYNC MAIN] Hybrid worker group extracted from trainer "
                f"(world_size={getattr(trainer_wg, 'world_size', '?')})"
            )
        else:
            print(
                "[ASYNC MAIN] Neither use_trainer_do_validate nor use_dynamic_resource_scheduling enabled, "
                "skipping hybrid worker group setup"
            )

    def _run_training_loop(self):
        self.running = True

        print("[ASYNC MAIN] Starting Rollouter and Trainer...")
        rollouter_future = self.components["rollouter"].fit.remote()
        trainer_future = self.components["trainer"].fit.remote()

        futures = [rollouter_future, trainer_future]

        try:
            while futures:
                # Use ray.wait to monitor all futures and return when any one is completed.
                done_futures, remaining_futures = ray.wait(futures, num_returns=1, timeout=None)

                for future in done_futures:
                    try:
                        ray.get(future)
                        print("[ASYNC MAIN] One component completed successfully")
                    except Exception as e:
                        print(f"[ASYNC MAIN] Component failed with error: {e}")
                        for remaining_future in remaining_futures:
                            ray.cancel(remaining_future)
                        raise e

                futures = remaining_futures

        except Exception as e:
            print(f"[ASYNC MAIN] Training failed: {e}")
            for future in futures:
                ray.cancel(future)
            raise
        finally:
            print("[ASYNC MAIN] Training completed or interrupted")


@hydra.main(config_path="config", config_name="fully_async_ppo_trainer", version_base=None)
def main(config):
    from verl.trainer.main_ppo import run_ppo

    # Ensure async training config exists
    if not hasattr(config, "async_training"):
        raise RuntimeError("must set async_training config")

    from time import time

    start_time = time()
    auto_set_device(config)
    # TODO: unify rollout config with actor_rollout_ref
    config.actor_rollout_ref.rollout.nnodes = config.rollout.nnodes
    config.actor_rollout_ref.rollout.n_gpus_per_node = config.rollout.n_gpus_per_node
    config = migrate_legacy_reward_impl(config)
    run_ppo(config, task_runner_class=FullyAsyncTaskRunner)
    print(f"total time: {time() - start_time:.2f} seconds")


if __name__ == "__main__":
    main()
