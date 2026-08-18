# Copyright 2026 The verl team.
# Licensed under the Apache License, Version 2.0.

"""Custom advantage plugins receive only the fields they explicitly declare."""

from __future__ import annotations

import unittest

import numpy as np
import torch
from omegaconf import OmegaConf
from tensordict import TensorDict

from verl.protocol import DataProto
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.ray_trainer import compute_advantage

ESTIMATOR_NAME = "test_signature_aware_action_axis_gae"


class TestCustomAdvantageForwarding(unittest.TestCase):
    def tearDown(self) -> None:
        core_algos.ADV_ESTIMATOR_REGISTRY.pop(ESTIMATOR_NAME, None)

    def test_registered_estimator_receives_declared_values_and_row_metadata(self) -> None:
        seen = {}

        @core_algos.register_adv_est(ESTIMATOR_NAME)
        def _estimator(
            token_level_rewards,
            response_mask,
            values,
            trajectory_uid,
            trajectory_row_uid,
            trajectory_row_order,
            immediate_reward,
            rollout_done_flag,
            trajectory_terminal,
            is_padding,
            batch,
            non_tensor_batch,
            config,
        ):
            seen.update(
                values=values,
                trajectory_uid=trajectory_uid,
                trajectory_row_uid=trajectory_row_uid,
                trajectory_row_order=trajectory_row_order,
                immediate_reward=immediate_reward,
                rollout_done_flag=rollout_done_flag,
                trajectory_terminal=trajectory_terminal,
                is_padding=is_padding,
                batch=batch,
                non_tensor_batch=non_tensor_batch,
                config=config,
            )
            return torch.zeros_like(token_level_rewards), values.clone()

        tensor_batch = TensorDict(
            {
                "token_level_rewards": torch.tensor([[0.0, 0.2], [0.0, 0.4]], dtype=torch.float32),
                "response_mask": torch.tensor([[1, 1], [1, 1]], dtype=torch.long),
                "values": torch.tensor([[0.1, 0.2], [0.3, 0.4]], dtype=torch.float32),
            },
            batch_size=2,
        )
        non_tensor_batch = {
            "trajectory_uid": np.array(["episode-0", "episode-0"], dtype=object),
            "trajectory_row_uid": np.array(["episode-0-row-0", "episode-0-row-1"], dtype=object),
            "trajectory_row_order": np.array([0, 1], dtype=object),
            "immediate_reward": np.array([0.0, 0.4], dtype=object),
            "rollout_done_flag": np.array([False, True], dtype=object),
            "trajectory_terminal": np.array([False, True], dtype=object),
            "is_padding": np.array([False, False], dtype=object),
        }
        data = DataProto(batch=tensor_batch, non_tensor_batch=non_tensor_batch)
        config = OmegaConf.create({"name": "action-axis-test"})

        result = compute_advantage(data, adv_estimator=ESTIMATOR_NAME, config=config)

        self.assertIs(seen["values"], tensor_batch["values"])
        self.assertIs(seen["batch"], tensor_batch)
        self.assertIs(seen["non_tensor_batch"], non_tensor_batch)
        self.assertIs(seen["config"], config)
        self.assertEqual(seen["trajectory_uid"].tolist(), ["episode-0", "episode-0"])
        self.assertEqual(seen["trajectory_row_order"].tolist(), [0, 1])
        self.assertEqual(seen["rollout_done_flag"].tolist(), [False, True])
        self.assertEqual(result.batch["returns"].tolist(), tensor_batch["values"].tolist())

    def test_action_gae_uses_first_response_state_value_after_physical_row_reorder(self) -> None:
        """Freeze AMG's causal state-value and padding semantics numerically.

        The local estimator below is a test oracle only.  Production must live in
        the external AMG plugin and reach this call through veRL's generic,
        signature-aware advantage dispatch.
        """

        estimator_name = f"{ESTIMATOR_NAME}_causal_numeric"

        @core_algos.register_adv_est(estimator_name)
        def _causal_action_gae(
            token_level_rewards,
            response_mask,
            values,
            trajectory_uid,
            trajectory_row_uid,
            trajectory_row_order,
            immediate_reward,
            rollout_done_flag,
            trajectory_terminal,
            is_padding,
            batch,
            non_tensor_batch,
            config,
        ):
            del token_level_rewards, trajectory_row_uid, trajectory_terminal, batch, non_tensor_batch
            gamma = float(config.gamma)
            lam = float(config.lam)
            valid_mask = response_mask.to(dtype=torch.bool)
            advantages = torch.zeros_like(values)
            returns = torch.zeros_like(values)
            trajectories = {}
            for physical_row in range(values.shape[0]):
                if bool(is_padding[physical_row]):
                    continue
                token_indices = torch.nonzero(valid_mask[physical_row], as_tuple=False).flatten()
                if token_indices.numel() == 0:
                    raise AssertionError("a real action row must have sampled response tokens")
                trajectories.setdefault(str(trajectory_uid[physical_row]), []).append(
                    (
                        int(trajectory_row_order[physical_row]),
                        physical_row,
                        int(token_indices[0]),
                        token_indices,
                    )
                )

            for rows in trajectories.values():
                rows.sort(key=lambda item: item[0])
                next_advantage = torch.zeros((), dtype=values.dtype)
                for reverse_index in range(len(rows) - 1, -1, -1):
                    _order, physical_row, state_index, token_indices = rows[reverse_index]
                    state_value = values[physical_row, state_index]
                    has_next = reverse_index + 1 < len(rows) and not bool(rollout_done_flag[physical_row])
                    if has_next:
                        _next_order, next_row, next_state_index, _next_tokens = rows[reverse_index + 1]
                        next_value = values[next_row, next_state_index]
                    else:
                        next_value = torch.zeros((), dtype=values.dtype)
                    reward = torch.as_tensor(float(immediate_reward[physical_row]), dtype=values.dtype)
                    continuation = 1.0 if has_next else 0.0
                    delta = reward + gamma * continuation * next_value - state_value
                    action_advantage = delta + gamma * lam * continuation * next_advantage
                    action_return = action_advantage + state_value
                    advantages[physical_row, token_indices] = action_advantage
                    returns[physical_row, token_indices] = action_return
                    next_advantage = action_advantage
            return advantages, returns

        try:
            tensor_batch = TensorDict(
                {
                    # Physical order: B1, PAD, A1, B0, A0.  The large value at
                    # each row's final token makes a final-token implementation fail.
                    "values": torch.tensor(
                        [
                            [0.4, 8.4, 0.0],
                            [99.0, 99.0, 99.0],
                            [0.5, 0.0, 0.0],
                            [0.2, 0.0, 0.0],
                            [0.1, 9.1, 0.0],
                        ],
                        dtype=torch.float32,
                    ),
                    "response_mask": torch.tensor(
                        [
                            [1, 1, 0],
                            [0, 0, 0],
                            [1, 0, 0],
                            [1, 0, 0],
                            [1, 1, 0],
                        ],
                        dtype=torch.long,
                    ),
                    "token_level_rewards": torch.tensor(
                        [
                            [0.0, 2.0, 0.0],
                            [0.0, 0.0, 0.0],
                            [1.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0],
                        ],
                        dtype=torch.float32,
                    ),
                },
                batch_size=5,
            )
            non_tensor_batch = {
                "trajectory_uid": np.array(["B", "PAD", "A", "B", "A"], dtype=object),
                "trajectory_row_uid": np.array(["B1", "PAD", "A1", "B0", "A0"], dtype=object),
                "trajectory_row_order": np.array([1, 0, 1, 0, 0], dtype=object),
                "immediate_reward": np.array([2.0, 0.0, 1.0, 0.0, 0.0], dtype=object),
                "rollout_done_flag": np.array([True, True, True, False, False], dtype=object),
                "trajectory_terminal": np.array([True, True, True, False, False], dtype=object),
                "is_padding": np.array([False, True, False, False, False], dtype=object),
            }
            data = DataProto(batch=tensor_batch, non_tensor_batch=non_tensor_batch)
            config = OmegaConf.create({"gamma": 1.0, "lam": 1.0})

            result = compute_advantage(data, adv_estimator=estimator_name, config=config)

            expected_advantages = torch.tensor(
                [
                    [1.6, 1.6, 0.0],
                    [0.0, 0.0, 0.0],
                    [0.5, 0.0, 0.0],
                    [1.8, 0.0, 0.0],
                    [0.9, 0.9, 0.0],
                ],
                dtype=torch.float32,
            )
            expected_returns = torch.tensor(
                [
                    [2.0, 2.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [2.0, 0.0, 0.0],
                    [1.0, 1.0, 0.0],
                ],
                dtype=torch.float32,
            )
            torch.testing.assert_close(result.batch["advantages"], expected_advantages)
            torch.testing.assert_close(result.batch["returns"], expected_returns)
            self.assertEqual(float(result.batch["advantages"][1].abs().sum()), 0.0)
            self.assertEqual(float(result.batch["returns"][1].abs().sum()), 0.0)
        finally:
            core_algos.ADV_ESTIMATOR_REGISTRY.pop(estimator_name, None)


if __name__ == "__main__":
    unittest.main()
