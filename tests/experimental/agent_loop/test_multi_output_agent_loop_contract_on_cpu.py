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

"""Generic contract for one AgentLoop invocation yielding multiple train rows.

The test deliberately exercises veRL's standard AgentLoopWorker/DataProto path.
It does not introduce an AMG-specific carrier, manager, queue, or materializer.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest import IsolatedAsyncioTestCase, mock

import numpy as np
import torch

from verl.experimental.agent_loop import agent_loop as agent_loop_module
from verl.experimental.agent_loop.agent_loop import (
    AgentLoopMetrics,
    AgentLoopOutput,
    AgentLoopWorker,
    _InternalAgentLoopOutput,
)
from verl.experimental.fully_async_policy.detach_utils import RolloutSample
from verl.protocol import DataProto


def _internal_output(prompt_marker: int, row_marker: int, reward: float) -> _InternalAgentLoopOutput:
    prompt_ids = torch.tensor([[prompt_marker, prompt_marker + 1]], dtype=torch.long)
    response_ids = torch.tensor([[row_marker, 0]], dtype=torch.long)
    response_mask = torch.tensor([[1, 0]], dtype=torch.long)
    attention_mask = torch.tensor([[1, 1, 1, 0]], dtype=torch.long)
    input_ids = torch.cat([prompt_ids, response_ids], dim=1)
    position_ids = torch.tensor([[0, 1, 2, 0]], dtype=torch.long)
    response_logprobs = torch.tensor([[-0.01 * row_marker, 0.0]], dtype=torch.float32)
    return _InternalAgentLoopOutput(
        prompt_ids=prompt_ids,
        response_ids=response_ids,
        input_ids=input_ids,
        position_ids=position_ids,
        response_mask=response_mask,
        attention_mask=attention_mask,
        response_logprobs=response_logprobs,
        reward_score=reward,
        num_turns=row_marker,
        metrics=AgentLoopMetrics(generate_sequences=float(row_marker)),
        extra_fields={
            "trajectory_row_uid": f"p{prompt_marker}-r{row_marker}",
            "trajectory_row_order": row_marker,
            "immediate_reward": reward,
        },
    )


class TestAgentLoopPluginListOutput(IsolatedAsyncioTestCase):
    async def test_standard_worker_postprocesses_each_plugin_output(self) -> None:
        """An upstream AgentLoop plugin may return ordered action-row outputs."""

        plugin_outputs = [
            AgentLoopOutput(
                prompt_ids=[1, 2],
                response_ids=[10],
                response_mask=[1],
                response_logprobs=[-0.10],
                reward_score=0.1,
                metrics=AgentLoopMetrics(),
                extra_fields={"trajectory_row_uid": "episode-0-row-0"},
            ),
            AgentLoopOutput(
                prompt_ids=[3, 4],
                response_ids=[11],
                response_mask=[1],
                response_logprobs=[-0.11],
                reward_score=0.2,
                metrics=AgentLoopMetrics(),
                extra_fields={"trajectory_row_uid": "episode-0-row-1"},
            ),
        ]

        class _Plugin:
            async def run(self, sampling_params, **kwargs):
                return plugin_outputs

        worker = object.__new__(AgentLoopWorker)
        worker.config = SimpleNamespace(data=SimpleNamespace())
        worker.llm_client = object()
        worker.tokenizer = object()
        worker.processor = None
        # Current main passes this generic model hint through Hydra; v0.9.0 ignores it.
        worker.hf_model_type = None
        worker.dataset_cls = object
        worker.tools = []

        seen: list[str] = []

        async def _postprocess_one(output, validate, **kwargs):
            if not isinstance(output, AgentLoopOutput):
                raise TypeError(f"expected one AgentLoopOutput, got {type(output).__name__}")
            seen.append(output.extra_fields["trajectory_row_uid"])
            return output.extra_fields["trajectory_row_uid"]

        worker._agent_loop_postprocess = _postprocess_one
        trajectory = {"step": 1, "sample_index": 0, "rollout_n": 0, "validate": False}

        with (
            mock.patch.dict(agent_loop_module._agent_loop_registry, {"multi_output_test": {"_target_": "unused"}}),
            mock.patch.object(agent_loop_module.hydra.utils, "instantiate", return_value=_Plugin()),
        ):
            result = await worker._run_agent_loop(
                {"temperature": 1.0},
                trajectory,
                agent_name="multi_output_test",
                trace=False,
                raw_prompt="prompt-0",
            )

        self.assertEqual(result, ["episode-0-row-0", "episode-0-row-1"])
        self.assertEqual(seen, ["episode-0-row-0", "episode-0-row-1"])


class TestAgentLoopWorkerMultiOutputBatch(IsolatedAsyncioTestCase):
    def _worker(self) -> AgentLoopWorker:
        worker = object.__new__(AgentLoopWorker)
        worker.rollout_config = SimpleNamespace(
            temperature=1.0,
            top_p=1.0,
            top_k=-1,
            calculate_log_probs=True,
            val_kwargs=SimpleNamespace(top_p=1.0, top_k=-1, temperature=0.0),
            agent=SimpleNamespace(default_agent_loop="unused"),
        )
        worker.reward_loop_worker_handles = None
        return worker

    @staticmethod
    def _prompts() -> DataProto:
        return DataProto(
            non_tensor_batch={
                "raw_prompt": np.array(["prompt-0", "prompt-1"], dtype=object),
                "agent_name": np.array(["test", "test"], dtype=object),
                "index": np.array([100, 200], dtype=object),
                "source_family": np.array(["family-a", "family-b"], dtype=object),
                "priority": np.array([7, 8], dtype=object),
            },
            meta_info={"global_steps": 3, "validate": False},
        )

    async def test_prompt_major_flatten_expands_all_prompt_metadata(self) -> None:
        """Output counts 2/1 become three independent rows in prompt-major order."""

        worker = self._worker()

        async def _run_agent_loop(sampling_params, trajectory, *, agent_name, trace=True, **kwargs):
            if kwargs["raw_prompt"] == "prompt-0":
                return [
                    _internal_output(prompt_marker=100, row_marker=10, reward=0.1),
                    _internal_output(prompt_marker=100, row_marker=11, reward=0.2),
                ]
            return [_internal_output(prompt_marker=200, row_marker=20, reward=0.3)]

        worker._run_agent_loop = _run_agent_loop
        trace_config = SimpleNamespace(max_samples_per_step_per_worker=None)
        with mock.patch.object(agent_loop_module.RolloutTraceConfig, "get_instance", return_value=trace_config):
            output = await worker.generate_sequences(self._prompts())

        self.assertEqual(len(output), 3)
        output.check_consistency()
        self.assertEqual(output.batch["responses"][:, 0].tolist(), [10, 11, 20])
        for got, expected in zip(output.batch["rm_scores"][:, 0].tolist(), [0.1, 0.2, 0.3], strict=True):
            self.assertAlmostEqual(got, expected)
        self.assertEqual(output.batch["rm_scores"][:, 1].tolist(), [0.0, 0.0, 0.0])
        self.assertAlmostEqual(float(output.batch["rollout_log_probs"][0, 0]), -0.10)
        self.assertAlmostEqual(float(output.batch["rollout_log_probs"][1, 0]), -0.11)
        self.assertAlmostEqual(float(output.batch["rollout_log_probs"][2, 0]), -0.20)
        self.assertEqual(output.non_tensor_batch["raw_prompt"].tolist(), ["prompt-0", "prompt-0", "prompt-1"])
        self.assertEqual(output.non_tensor_batch["index"].tolist(), [100, 100, 200])
        self.assertEqual(
            output.non_tensor_batch["source_family"].tolist(),
            ["family-a", "family-a", "family-b"],
        )
        self.assertEqual(output.non_tensor_batch["priority"].tolist(), [7, 7, 8])
        self.assertEqual(
            output.non_tensor_batch["trajectory_row_uid"].tolist(),
            ["p100-r10", "p100-r11", "p200-r20"],
        )
        self.assertEqual(output.non_tensor_batch["trajectory_row_order"].tolist(), [10, 11, 20])
        self.assertEqual(output.non_tensor_batch["immediate_reward"].tolist(), [0.1, 0.2, 0.3])

        queue_item = RolloutSample(
            full_batch=output,
            sample_id="episode-bundle-0",
            epoch=0,
            rollout_status={"policy_version": 3},
        )
        self.assertIs(queue_item.full_batch, output)
        self.assertEqual(len(queue_item.full_batch), 3)

    async def test_empty_plugin_output_fails_closed(self) -> None:
        worker = self._worker()

        async def _run_agent_loop(sampling_params, trajectory, *, agent_name, trace=True, **kwargs):
            if kwargs["raw_prompt"] == "prompt-0":
                return []
            return [_internal_output(prompt_marker=200, row_marker=20, reward=0.3)]

        worker._run_agent_loop = _run_agent_loop
        trace_config = SimpleNamespace(max_samples_per_step_per_worker=None)
        with (
            mock.patch.object(agent_loop_module.RolloutTraceConfig, "get_instance", return_value=trace_config),
            self.assertRaisesRegex(ValueError, "empty.*AgentLoop|AgentLoop.*empty|zero outputs"),
        ):
            await worker.generate_sequences(self._prompts())


if __name__ == "__main__":
    unittest.main()
