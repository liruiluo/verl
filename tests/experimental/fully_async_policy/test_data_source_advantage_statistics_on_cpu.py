# Copyright 2026 The verl team.
# Licensed under the Apache License, Version 2.0.

"""CPU regressions for fully-async per-data-source PPO input statistics."""

from __future__ import annotations

from unittest import TestCase, mock

import numpy as np
import torch
from tensordict import TensorDict

from verl.experimental.fully_async_policy.detach_utils import MetricsAggregator
from verl.experimental.fully_async_policy.fully_async_trainer import FullyAsyncTrainer
from verl.experimental.separation.ray_trainer import SeparateRayPPOTrainer
from verl.protocol import DataProto


def _local_actor_class(actor_class):
    return actor_class.__ray_metadata__.modified_class


def _route_batch() -> DataProto:
    return DataProto(
        batch=TensorDict(
            {
                "response_mask": torch.tensor(
                    [
                        [1, 1, 0],
                        [1, 0, 0],
                        [0, 0, 0],
                        [1, 1, 1],
                    ],
                    dtype=torch.long,
                ),
                "advantages": torch.tensor(
                    [
                        [1.0, 1.0, 99.0],
                        [-2.0, 99.0, 99.0],
                        [999.0, 999.0, 999.0],
                        [0.5, 0.5, 0.5],
                    ],
                    dtype=torch.float32,
                ),
                "returns": torch.tensor(
                    [
                        [1.5, 7.0, 0.0],
                        [-1.0, 0.0, 0.0],
                        [999.0, 999.0, 999.0],
                        [0.75, 8.0, 8.0],
                    ],
                    dtype=torch.float32,
                ),
                "values": torch.tensor(
                    [
                        [0.5, 6.0, 0.0],
                        [0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0],
                        [0.25, 7.5, 7.5],
                    ],
                    dtype=torch.float32,
                ),
            },
            batch_size=4,
        ),
        non_tensor_batch={
            # Deliberately interleaved: metrics must use these row-aligned labels,
            # not queue-time episode ordering.
            "data_source": np.array(["webshop", "swesmith", "padding", "webshop"], dtype=object),
            "is_padding": np.array([False, False, True, False], dtype=bool),
        },
    )


class TestDataSourceAdvantageStatistics(TestCase):
    def test_statistics_follow_balanced_row_labels_and_exclude_padding(self) -> None:
        trainer_class = _local_actor_class(FullyAsyncTrainer)
        real_batch = trainer_class._without_synthetic_padding(_route_batch())

        metrics = trainer_class._compute_data_source_advantage_statistics(real_batch)
        webshop = "fully_async/sum/route_advantage/data_source/webshop"
        swesmith = "fully_async/sum/route_advantage/data_source/swesmith"

        self.assertEqual(metrics[f"{webshop}/action_rows"], 2.0)
        self.assertEqual(metrics[f"{webshop}/response_tokens"], 5.0)
        self.assertAlmostEqual(metrics[f"{webshop}/pre_whiten_action_sum"], 1.5)
        self.assertAlmostEqual(metrics[f"{webshop}/pre_whiten_action_squared_sum"], 1.25)
        self.assertAlmostEqual(metrics[f"{webshop}/pre_whiten_action_abs_sum"], 1.5)
        self.assertAlmostEqual(metrics[f"{webshop}/post_whiten_action_sum"], 1.5)
        self.assertAlmostEqual(metrics[f"{webshop}/post_whiten_action_squared_sum"], 1.25)
        self.assertAlmostEqual(metrics[f"{webshop}/post_whiten_action_abs_sum"], 1.5)
        self.assertAlmostEqual(metrics[f"{webshop}/post_whiten_token_sum"], 3.5)
        self.assertAlmostEqual(metrics[f"{webshop}/post_whiten_token_squared_sum"], 2.75)
        self.assertAlmostEqual(metrics[f"{webshop}/post_whiten_token_abs_sum"], 3.5)
        self.assertAlmostEqual(metrics[f"{webshop}/unit_ratio_pg_numerator_sum"], -3.5)

        self.assertEqual(metrics[f"{swesmith}/action_rows"], 1.0)
        self.assertEqual(metrics[f"{swesmith}/response_tokens"], 1.0)
        self.assertAlmostEqual(metrics[f"{swesmith}/pre_whiten_action_sum"], -1.0)
        self.assertAlmostEqual(metrics[f"{swesmith}/post_whiten_action_sum"], -2.0)
        self.assertAlmostEqual(metrics[f"{swesmith}/post_whiten_token_sum"], -2.0)
        self.assertAlmostEqual(metrics[f"{swesmith}/unit_ratio_pg_numerator_sum"], 2.0)
        self.assertFalse(any("padding" in key for key in metrics))

    def test_fit_collect_metrics_filters_padding_before_parent_and_route_stats(self) -> None:
        trainer_class = _local_actor_class(FullyAsyncTrainer)
        trainer = object.__new__(trainer_class)
        trainer.metrics = {}
        batch = _route_batch()
        seen = {}

        def _parent_collect(_self, real_batch) -> None:
            seen["rows"] = len(real_batch)
            seen["sources"] = real_batch.non_tensor_batch["data_source"].tolist()

        with mock.patch.object(SeparateRayPPOTrainer, "_fit_collect_metrics", new=_parent_collect):
            trainer._fit_collect_metrics(batch)

        self.assertEqual(seen["rows"], 3)
        self.assertEqual(seen["sources"], ["webshop", "swesmith", "webshop"])
        self.assertEqual(
            trainer.metrics["fully_async/sum/route_advantage/data_source/webshop/action_rows"],
            2.0,
        )

    def test_statistics_are_additive_across_parameter_sync_cycle(self) -> None:
        trainer_class = _local_actor_class(FullyAsyncTrainer)
        real_batch = trainer_class._without_synthetic_padding(_route_batch())
        metrics = trainer_class._compute_data_source_advantage_statistics(real_batch)
        # A route label may itself contain aggregation-looking substrings.  The
        # explicit fully_async/sum namespace must take precedence over them.
        metrics["fully_async/sum/route_advantage/data_source/admin/response_tokens"] = 3.0
        aggregator = MetricsAggregator(total_gpus=8, hybrid_gpus=6, standalone_gpus=2)

        aggregator.add_step_metrics(metrics=metrics, sample_count=2, timestamp=1.0)
        aggregator.add_step_metrics(metrics=metrics, sample_count=2, timestamp=2.0)
        aggregated = aggregator.get_aggregated_metrics()

        key = "fully_async/sum/route_advantage/data_source/webshop/post_whiten_token_abs_sum"
        self.assertEqual(aggregated[key], 7.0)
        self.assertEqual(
            aggregated["fully_async/sum/route_advantage/data_source/admin/response_tokens"],
            6.0,
        )
        self.assertTrue(all("/sum/" in name for name in metrics))

    def test_misaligned_data_source_fails_closed(self) -> None:
        trainer_class = _local_actor_class(FullyAsyncTrainer)
        batch = trainer_class._without_synthetic_padding(_route_batch())
        batch.non_tensor_batch["data_source"] = np.array(["webshop", "swesmith"], dtype=object)

        with self.assertRaisesRegex(ValueError, "one label per real learner row"):
            trainer_class._compute_data_source_advantage_statistics(batch)

    def test_missing_route_label_is_backward_compatible(self) -> None:
        trainer_class = _local_actor_class(FullyAsyncTrainer)
        batch = trainer_class._without_synthetic_padding(_route_batch())
        batch.non_tensor_batch.pop("data_source")

        self.assertEqual(trainer_class._compute_data_source_advantage_statistics(batch), {})

    def test_critic_free_batch_skips_pre_whiten_statistics(self) -> None:
        trainer_class = _local_actor_class(FullyAsyncTrainer)
        batch = trainer_class._without_synthetic_padding(_route_batch())
        batch.batch.pop("values")

        metrics = trainer_class._compute_data_source_advantage_statistics(batch)

        self.assertTrue(metrics)
        self.assertFalse(any("pre_whiten" in name for name in metrics))
        self.assertEqual(
            metrics["fully_async/sum/route_advantage/data_source/webshop/response_tokens"],
            5.0,
        )


if __name__ == "__main__":
    import unittest

    unittest.main()
