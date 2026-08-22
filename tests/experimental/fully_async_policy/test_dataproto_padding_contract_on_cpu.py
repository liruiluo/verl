# Copyright 2026 The verl team.
# Licensed under the Apache License, Version 2.0.

"""Zero-loss DataProto padding for variable action-row PPO batches."""

from __future__ import annotations

import unittest

import numpy as np
import torch
from tensordict import TensorDict

from verl.protocol import DataProto, pad_dataproto_to_divisor
from verl.trainer.ppo import padding_utils
from verl.utils import tensordict_utils as tu

REAL_ROWS = 1001
BATCH_MULTIPLE = 512
PADDED_ROWS = 1024


def _real_batch() -> DataProto:
    row_ids = torch.arange(REAL_ROWS, dtype=torch.long)
    prompts = torch.stack([row_ids, row_ids + 1], dim=1)
    responses = torch.stack([row_ids + 2, row_ids + 3], dim=1)
    response_mask = torch.ones((REAL_ROWS, 2), dtype=torch.long)
    attention_mask = torch.ones((REAL_ROWS, 4), dtype=torch.long)
    batch = TensorDict(
        {
            "prompts": prompts,
            "responses": responses,
            "input_ids": torch.cat([prompts, responses], dim=1),
            "attention_mask": attention_mask,
            "position_ids": torch.arange(4, dtype=torch.long).repeat(REAL_ROWS, 1),
            "response_mask": response_mask,
            "loss_mask": response_mask.clone(),
            "rm_scores": torch.ones((REAL_ROWS, 2), dtype=torch.float32),
            "rollout_log_probs": torch.full((REAL_ROWS, 2), -0.5, dtype=torch.float32),
            "values": torch.full((REAL_ROWS, 2), 0.25, dtype=torch.float32),
            "advantages": torch.full((REAL_ROWS, 2), 0.75, dtype=torch.float32),
            "returns": torch.full((REAL_ROWS, 2), 1.0, dtype=torch.float32),
            "token_level_rewards": torch.full((REAL_ROWS, 2), 0.4, dtype=torch.float32),
        },
        batch_size=REAL_ROWS,
    )
    return DataProto(
        batch=batch,
        non_tensor_batch={
            "uid": np.array([f"real-{i}" for i in range(REAL_ROWS)], dtype=object),
            "trajectory_uid": np.array([f"episode-{i // 16}" for i in range(REAL_ROWS)], dtype=object),
            "trajectory_row_uid": np.array([f"row-{i}" for i in range(REAL_ROWS)], dtype=object),
            "trajectory_row_order": np.array(list(range(REAL_ROWS)), dtype=object),
            "immediate_reward": np.array([0.4] * REAL_ROWS, dtype=object),
            "rollout_done_flag": np.array([False] * REAL_ROWS, dtype=object),
            "trajectory_terminal": np.array([False] * REAL_ROWS, dtype=object),
            "is_padding": np.array([False] * REAL_ROWS, dtype=object),
        },
    )


def _two_token_mean_updates(features: torch.Tensor, targets: torch.Tensor, valid: torch.Tensor, splits):
    weight = torch.nn.Parameter(torch.tensor(0.2, dtype=torch.float64))
    optimizer = torch.optim.Adam([weight], lr=0.01)
    gradients = []
    optimizer_mini_batch_calls = 0
    for indices in splits:
        optimizer.zero_grad(set_to_none=True)
        mask = valid[indices]
        losses = (weight * features[indices] - targets[indices]).square()
        loss = (losses * mask).sum() / mask.sum()
        loss.backward()
        gradients.append(float(weight.grad))
        optimizer.step()
        optimizer_mini_batch_calls += 1
    # Native TrainingWorker advances the LR scheduler once after the final
    # optimizer mini-batch of one learner collection, not once per split.
    lr_scheduler_steps = 1 if optimizer_mini_batch_calls else 0
    return float(weight.detach()), gradients, optimizer_mini_batch_calls, lr_scheduler_steps


class TestDataProtoPaddingGap(unittest.TestCase):
    def test_unpadded_batch_cannot_enter_two_512_row_updates(self) -> None:
        data = _real_batch()
        with self.assertRaisesRegex(AssertionError, "1001 % 512"):
            list(tu.make_iterator(data.batch, mini_batch_size=BATCH_MULTIPLE, epochs=1))

    def test_existing_transport_padding_repeats_real_loss_rows(self) -> None:
        data = _real_batch()
        padded, pad_size = pad_dataproto_to_divisor(data, BATCH_MULTIPLE)
        self.assertEqual(pad_size, PADDED_ROWS - REAL_ROWS)
        self.assertEqual(len(padded), PADDED_ROWS)
        # This generic transport helper intentionally repeats real rows. It must
        # not be reused as PPO zero-loss padding without an explicit adapter.
        self.assertEqual(int(padded.batch["response_mask"][REAL_ROWS:].sum()), 2 * pad_size)
        self.assertEqual(padded.non_tensor_batch["uid"][REAL_ROWS], "real-0")


class TestDataProtoZeroLossPaddingContract(unittest.TestCase):
    def test_1001_to_1024_preserves_real_membership_and_partial_tail_update(self) -> None:
        adapter = getattr(padding_utils, "upsample_dataproto_to_divisible_size", None)
        self.assertIsNotNone(
            adapter,
            "missing thin DataProto counterpart of native V1 upsample_batch_to_divisible_size",
        )

        real = _real_batch()
        padded = adapter(real, batch_multiple=BATCH_MULTIPLE, eos_token_id=0)
        padded.check_consistency()

        self.assertEqual(len(padded), PADDED_ROWS)
        self.assertEqual(padded.non_tensor_batch["uid"][:REAL_ROWS].tolist(), [f"real-{i}" for i in range(REAL_ROWS)])
        self.assertEqual(padded.non_tensor_batch["is_padding"][:REAL_ROWS].tolist(), [False] * REAL_ROWS)
        self.assertEqual(padded.non_tensor_batch["is_padding"][REAL_ROWS:].tolist(), [True] * (PADDED_ROWS - REAL_ROWS))
        self.assertTrue(all(str(uid).startswith("pad-") for uid in padded.non_tensor_batch["uid"][REAL_ROWS:]))

        for key in (
            "response_mask",
            "loss_mask",
            "rm_scores",
            "rollout_log_probs",
            "values",
            "advantages",
            "returns",
            "token_level_rewards",
        ):
            self.assertEqual(float(padded.batch[key][REAL_ROWS:].abs().sum()), 0.0, key)

        # Physical optimizer membership is two 512-row updates. Real rows appear
        # once and in order; only the second update carries 23 synthetic rows.
        first = padded.non_tensor_batch["uid"][:512].tolist()
        second = padded.non_tensor_batch["uid"][512:].tolist()
        self.assertEqual(first, [f"real-{i}" for i in range(512)])
        self.assertEqual(second[: REAL_ROWS - 512], [f"real-{i}" for i in range(512, REAL_ROWS)])
        self.assertTrue(all(str(uid).startswith("pad-") for uid in second[REAL_ROWS - 512 :]))

        # Under token-mean aggregation, padding is optimizer-equivalent to the
        # synchronous partial-tail oracle: same two optimizer mini-batch calls,
        # gradients and parameter. Native LR scheduling advances once for the
        # enclosing learner collection, after the final mini-batch.
        features = torch.linspace(0.1, 2.0, REAL_ROWS, dtype=torch.float64)
        targets = torch.linspace(-0.5, 0.5, REAL_ROWS, dtype=torch.float64)
        baseline = _two_token_mean_updates(
            features,
            targets,
            torch.ones(REAL_ROWS, dtype=torch.float64),
            [torch.arange(0, 512), torch.arange(512, REAL_ROWS)],
        )
        pad_count = PADDED_ROWS - REAL_ROWS
        padded_features = torch.cat([features, torch.zeros(pad_count, dtype=torch.float64)])
        padded_targets = torch.cat([targets, torch.zeros(pad_count, dtype=torch.float64)])
        padded_valid = torch.cat(
            [torch.ones(REAL_ROWS, dtype=torch.float64), torch.zeros(pad_count, dtype=torch.float64)]
        )
        adapted = _two_token_mean_updates(
            padded_features,
            padded_targets,
            padded_valid,
            [torch.arange(0, 512), torch.arange(512, PADDED_ROWS)],
        )
        self.assertEqual(baseline[2], adapted[2])
        self.assertEqual(baseline[2], 2)
        self.assertEqual(baseline[3], adapted[3])
        self.assertEqual(baseline[3], 1)
        self.assertAlmostEqual(baseline[0], adapted[0], places=12)
        for expected, actual in zip(baseline[1], adapted[1], strict=True):
            self.assertAlmostEqual(expected, actual, places=12)


if __name__ == "__main__":
    unittest.main()
