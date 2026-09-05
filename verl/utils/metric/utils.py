# Copyright 2025 Bytedance Ltd. and/or its affiliates
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
"""
Metrics utils.
"""

import math
from enum import Enum
from typing import Any, Optional, Union

import numpy as np
import torch


def reduce_metrics(metrics: dict[str, Union["Metric", list[Any]]]) -> dict[str, Any]:
    """
    Reduces a dictionary of metric lists by computing the mean, max, or min of each list.
    The reduce operation is determined by the key name:
    - If the key contains "max", np.max is used
    - If the key contains "min", np.min is used
    - Otherwise, np.mean is used

    Args:
        metrics: A dictionary mapping metric names to lists of metric values.

    Returns:
        A dictionary with the same keys but with each list replaced by its reduced value.

    Example:
        >>> metrics = {
        ...     "loss": [1.0, 2.0, 3.0],
        ...     "accuracy": [0.8, 0.9, 0.7],
        ...     "max_reward": [5.0, 8.0, 6.0],
        ...     "min_error": [0.1, 0.05, 0.2]
        ... }
        >>> reduce_metrics(metrics)
        {"loss": 2.0, "accuracy": 0.8, "max_reward": 8.0, "min_error": 0.05}
    """
    for key, val in metrics.items():
        if isinstance(val, Metric):
            metrics[key] = val.aggregate()
        elif "max" in key:
            metrics[key] = np.max(val)
        elif "min" in key:
            metrics[key] = np.min(val)
        else:
            metrics[key] = np.mean(val)
    return metrics


class AggregationType(Enum):
    MEAN = "mean"
    SUM = "sum"
    MIN = "min"
    MAX = "max"
    STD = "std"
    EFFECTIVE_SAMPLE_SIZE = "effective_sample_size"


NumericType = int, float, torch.Tensor, np.ndarray
Numeric = int | float | torch.Tensor | np.ndarray


class Metric:
    """
    A metric aggregator for collecting and aggregating numeric values.

    This class accumulates numeric values (int, float, or scalar tensors) and computes
    an aggregate statistic based on the specified aggregation type (MEAN, SUM, MIN, or MAX).

    Args:
        aggregation: The aggregation method to use. Can be a string ("mean", "sum", "min", "max")
            or an AggregationType enum value.
        value: Optional initial value(s) to add. Can be a single numeric value or a list of values.

    Example:
        >>> metric = Metric(aggregation="mean", value=1.0)
        >>> metric.append(2.0)
        >>> metric.append(3.0)
        >>> metric.aggregate()
        2.0
    """

    def __init__(
        self,
        aggregation: str | AggregationType,
        value: Optional[Numeric | list[Numeric]] = None,
        *,
        weight: Numeric = 1.0,
        second_moment: Optional[Numeric] = None,
        ddof: int = 0,
    ) -> None:
        if isinstance(aggregation, str):
            self.aggregation = AggregationType(aggregation)
        else:
            self.aggregation = aggregation
        if not isinstance(self.aggregation, AggregationType):
            raise ValueError(f"Unsupported aggregation type: {aggregation}")
        if isinstance(ddof, bool) or not isinstance(ddof, int) or ddof < 0:
            raise ValueError(f"ddof must be a non-negative integer, got {ddof!r}")
        self.ddof = ddof
        self.values = []
        # ``weights`` are observation counts, not optimization weights.  They
        # let a local micro-batch publish sufficient statistics without making
        # a short micro-batch count as much as a long one.  A zero weight marks
        # synthetic padding and is ignored by every reduction.
        self.weights = []
        # The second raw moment is needed to merge standard deviation and ESS
        # exactly across dynamic micro-batches and unequal DP-rank workloads.
        self.second_moments = []
        if value is not None:
            if isinstance(value, list):
                for item in value:
                    self.append(item)
            else:
                self.append(value, weight=weight, second_moment=second_moment)

    @staticmethod
    def _as_scalar(value: Numeric, *, name: str) -> int | float:
        if isinstance(value, torch.Tensor):
            if value.numel() != 1:
                raise ValueError(f"Only scalar tensors can be used as {name}")
            value = value.detach().item()
        elif isinstance(value, np.ndarray):
            if value.size != 1:
                raise ValueError(f"Only scalar arrays can be used as {name}")
            value = value.item()
        if not isinstance(value, (int, float)):
            raise ValueError(f"Unsupported {name} type: {type(value)}")
        return value

    def append(
        self,
        value: Union[Numeric, "Metric"],
        *,
        weight: Numeric = 1.0,
        second_moment: Optional[Numeric] = None,
    ) -> None:
        if isinstance(value, Metric):
            if weight != 1.0 or second_moment is not None:
                raise ValueError("weight/second_moment cannot override an appended Metric")
            self.extend(value)
            return
        value = self._as_scalar(value, name="value")
        weight = self._as_scalar(weight, name="weight")
        if weight < 0:
            raise ValueError(f"weight must be non-negative, got {weight}")
        if second_moment is None:
            second_moment = value * value
        else:
            second_moment = self._as_scalar(second_moment, name="second_moment")
        self.values.append(value)
        self.weights.append(float(weight))
        self.second_moments.append(second_moment)

    def extend(self, values: Union["Metric", list[Numeric]]) -> None:
        if isinstance(values, Metric):
            if values.aggregation != self.aggregation:
                raise ValueError(f"Aggregation type mismatch: {self.aggregation} != {values.aggregation}")
            if values.ddof != self.ddof:
                raise ValueError(f"ddof mismatch: {self.ddof} != {values.ddof}")
            for value, weight, second_moment in zip(
                values.values, values.weights, values.second_moments, strict=True
            ):
                self.append(value, weight=weight, second_moment=second_moment)
            return
        for value in values:
            self.append(value)

    def aggregate(self) -> float:
        return self._aggregate(
            self.values,
            self.aggregation,
            weights=self.weights,
            second_moments=self.second_moments,
            ddof=self.ddof,
        )

    @classmethod
    def _aggregate(
        cls,
        values: list[Numeric],
        aggregation: AggregationType,
        *,
        weights: Optional[list[Numeric]] = None,
        second_moments: Optional[list[Numeric]] = None,
        ddof: int = 0,
    ) -> float:
        if weights is None:
            weights = [1.0] * len(values)
        if second_moments is None:
            second_moments = [value * value for value in values]
        if not (len(values) == len(weights) == len(second_moments)):
            raise ValueError("values, weights and second_moments must have equal lengths")

        active = [
            (float(value), float(weight), float(second_moment))
            for value, weight, second_moment in zip(values, weights, second_moments, strict=True)
            if float(weight) > 0.0
        ]
        match aggregation:
            case AggregationType.MEAN:
                if not active:
                    raise ValueError("Cannot aggregate a Metric with no positive-weight observations")
                total_weight = sum(weight for _, weight, _ in active)
                return sum(value * weight for value, weight, _ in active) / total_weight
            case AggregationType.SUM:
                if not active:
                    raise ValueError("Cannot aggregate a Metric with no positive-weight observations")
                return sum(value for value, _, _ in active)
            case AggregationType.MIN:
                if not active:
                    raise ValueError("Cannot aggregate a Metric with no positive-weight observations")
                return min(value for value, _, _ in active)
            case AggregationType.MAX:
                if not active:
                    raise ValueError("Cannot aggregate a Metric with no positive-weight observations")
                return max(value for value, _, _ in active)
            case AggregationType.STD:
                if not active:
                    raise ValueError("Cannot aggregate a Metric with no positive-weight observations")
                total_weight = sum(weight for _, weight, _ in active)
                if total_weight <= ddof:
                    return 0.0
                total = sum(value * weight for value, weight, _ in active)
                total_squares = sum(second_moment * weight for _, weight, second_moment in active)
                variance_numerator = max(0.0, total_squares - total * total / total_weight)
                return math.sqrt(variance_numerator / (total_weight - ddof))
            case AggregationType.EFFECTIVE_SAMPLE_SIZE:
                if not active:
                    raise ValueError("Cannot aggregate a Metric with no positive-weight observations")
                total_weight = sum(weight for _, weight, _ in active)
                total = sum(value * weight for value, weight, _ in active)
                total_squares = sum(second_moment * weight for _, weight, second_moment in active)
                if total_weight <= 0.0 or total_squares <= 0.0:
                    return 0.0
                # Normalized ESS in [0, 1]: (sum w)^2 / (N * sum w^2).
                return total * total / (total_weight * total_squares)

    @classmethod
    def aggregate_dp(cls, metric_lists: list["Metric"]) -> float:
        if not metric_lists:
            raise ValueError("Cannot aggregate an empty list of metrics.")
        aggregation = metric_lists[0].aggregation
        ddof = metric_lists[0].ddof
        if any(metric.aggregation != aggregation for metric in metric_lists):
            raise ValueError("All Metric instances must use the same aggregation type")
        if any(metric.ddof != ddof for metric in metric_lists):
            raise ValueError("All Metric instances must use the same ddof")

        if aggregation == AggregationType.SUM:
            # SUM-valued losses are already normalized for DDP.  Preserve the
            # fixed DP-rank denominator even if a synthetic-padding rank emits
            # no local metric for one dynamic micro-batch.
            rank_sums = [
                sum(
                    float(value)
                    for value, weight in zip(metric.values, metric.weights, strict=True)
                    if float(weight) > 0.0
                )
                for metric in metric_lists
            ]
            if not any(
                float(weight) > 0.0
                for metric in metric_lists
                for weight in metric.weights
            ):
                raise ValueError("Cannot aggregate Metrics with no positive-weight observations")
            return sum(rank_sums) / len(metric_lists)

        values = [value for metric in metric_lists for value in metric.values]
        weights = [weight for metric in metric_lists for weight in metric.weights]
        second_moments = [
            second_moment for metric in metric_lists for second_moment in metric.second_moments
        ]
        return cls._aggregate(
            values,
            aggregation,
            weights=weights,
            second_moments=second_moments,
            ddof=ddof,
        )

    @classmethod
    def from_dict(
        cls, data: dict[str, Union[Numeric, "Metric"]], aggregation: str | AggregationType
    ) -> dict[str, "Metric"]:
        metrics = {}
        for key, value in data.items():
            if isinstance(value, Metric):
                metric = cls(aggregation=value.aggregation, ddof=value.ddof)
                metric.extend(value)
                metrics[key] = metric
            else:
                metrics[key] = cls(value=value, aggregation=aggregation)
        return metrics

    def init_list(self) -> "Metric":
        return Metric(aggregation=self.aggregation, ddof=self.ddof)
