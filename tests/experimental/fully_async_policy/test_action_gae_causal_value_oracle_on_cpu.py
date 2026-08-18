# Copyright 2026 The verl team.
# Licensed under the Apache License, Version 2.0.

"""Numeric oracle for AMG action-axis GAE causal value selection.

Production remains an external AMG advantage plugin.  This fixture freezes the
math independently from generic veRL signature forwarding so a dispatch error
cannot prevent the value-selection invariant from being tested.
"""

from __future__ import annotations

import unittest

import numpy as np
import torch


def reference_action_gae(
    *,
    values: torch.Tensor,
    response_mask: torch.Tensor,
    trajectory_uid: np.ndarray,
    trajectory_row_order: np.ndarray,
    immediate_reward: np.ndarray,
    done: np.ndarray,
    is_padding: np.ndarray,
    gamma: float,
    lam: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    valid_mask = response_mask.to(dtype=torch.bool)
    advantages = torch.zeros_like(values)
    returns = torch.zeros_like(values)
    trajectories: dict[str, list[tuple[int, int, int, torch.Tensor]]] = {}
    for physical_row in range(values.shape[0]):
        if bool(is_padding[physical_row]):
            continue
        token_indices = torch.nonzero(valid_mask[physical_row], as_tuple=False).flatten()
        if token_indices.numel() == 0:
            raise ValueError("real action row has no response token")
        trajectories.setdefault(str(trajectory_uid[physical_row]), []).append(
            (int(trajectory_row_order[physical_row]), physical_row, int(token_indices[0]), token_indices)
        )

    for rows in trajectories.values():
        rows.sort(key=lambda item: item[0])
        next_advantage = torch.zeros((), dtype=values.dtype)
        for reverse_index in range(len(rows) - 1, -1, -1):
            _order, physical_row, state_index, token_indices = rows[reverse_index]
            state_value = values[physical_row, state_index]
            has_next = reverse_index + 1 < len(rows) and not bool(done[physical_row])
            next_value = (
                values[rows[reverse_index + 1][1], rows[reverse_index + 1][2]]
                if has_next
                else torch.zeros((), dtype=values.dtype)
            )
            continuation = 1.0 if has_next else 0.0
            reward = torch.as_tensor(float(immediate_reward[physical_row]), dtype=values.dtype)
            delta = reward + gamma * continuation * next_value - state_value
            action_advantage = delta + gamma * lam * continuation * next_advantage
            action_return = action_advantage + state_value
            advantages[physical_row, token_indices] = action_advantage
            returns[physical_row, token_indices] = action_return
            next_advantage = action_advantage
    return advantages, returns


class TestActionGAECausalValueOracle(unittest.TestCase):
    def test_reordered_unequal_rows_use_first_valid_response_state_and_ignore_padding(self) -> None:
        # Physical order is B1, PAD, A1, B0, A0.  Large later-token values
        # deliberately make final-token or physical-row-order implementations fail.
        values = torch.tensor(
            [
                [0.4, 8.4, 0.0],
                [99.0, 99.0, 99.0],
                [0.5, 0.0, 0.0],
                [0.2, 0.0, 0.0],
                [0.1, 9.1, 0.0],
            ],
            dtype=torch.float32,
        )
        response_mask = torch.tensor(
            [[1, 1, 0], [0, 0, 0], [1, 0, 0], [1, 0, 0], [1, 1, 0]],
            dtype=torch.long,
        )
        advantages, returns = reference_action_gae(
            values=values,
            response_mask=response_mask,
            trajectory_uid=np.array(["B", "PAD", "A", "B", "A"], dtype=object),
            trajectory_row_order=np.array([1, 0, 1, 0, 0], dtype=object),
            immediate_reward=np.array([2.0, 0.0, 1.0, 0.0, 0.0], dtype=object),
            done=np.array([True, True, True, False, False], dtype=object),
            is_padding=np.array([False, True, False, False, False], dtype=object),
            gamma=1.0,
            lam=1.0,
        )
        torch.testing.assert_close(
            advantages,
            torch.tensor(
                [[1.6, 1.6, 0.0], [0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [1.8, 0.0, 0.0], [0.9, 0.9, 0.0]],
                dtype=torch.float32,
            ),
        )
        torch.testing.assert_close(
            returns,
            torch.tensor(
                [[2.0, 2.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [1.0, 1.0, 0.0]],
                dtype=torch.float32,
            ),
        )


if __name__ == "__main__":
    unittest.main()
