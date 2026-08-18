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

"""Bounded, task-agnostic evidence helpers for fully asynchronous training."""

from __future__ import annotations

import math
import os
import tempfile
from datetime import datetime, timezone
from typing import Any

from verl.utils.tracking import FileLogger

PARAMETER_PROBE_HARD_MAX_PARAMETERS = 64
PARAMETER_PROBE_HARD_MAX_ELEMENTS_PER_PARAMETER = 256


def utc_timestamp() -> str:
    """Return an RFC 3339-compatible UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


def classify_exception(exception: BaseException) -> dict[str, str]:
    """Return a stable, JSON-serializable exception classification."""
    return {
        "type": type(exception).__name__,
        "module": type(exception).__module__,
        "message": str(exception),
    }


def queue_conservation_holds(statistics: dict[str, Any] | None) -> bool:
    """Check the native real-sample queue accounting identity."""
    if not statistics:
        return False
    keys = ("real_enqueued", "real_consumed", "real_evicted", "real_cleared", "real_resident")
    if any(key not in statistics for key in keys):
        return False
    values = [statistics[key] for key in keys]
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in values):
        return False
    enqueued, consumed, evicted, cleared, resident = values
    return statistics.get("queue_size") == resident and enqueued == consumed + evicted + cleared + resident


def queue_clear_delta_matches(
    before_statistics: dict[str, Any] | None,
    after_statistics: dict[str, Any] | None,
) -> bool:
    """Check that the existing queue clear moved every resident row to ``real_cleared``."""
    if not queue_conservation_holds(before_statistics) or not queue_conservation_holds(after_statistics):
        return False
    stable_keys = ("real_enqueued", "real_consumed", "real_evicted")
    if any(before_statistics[key] != after_statistics[key] for key in stable_keys):
        return False
    cleared_delta = after_statistics["real_cleared"] - before_statistics["real_cleared"]
    resident_delta = before_statistics["real_resident"] - after_statistics["real_resident"]
    return cleared_delta == resident_delta == before_statistics["real_resident"]


def _validate_probe_payload(payload: Any, *, label: str) -> list[dict[str, Any]]:
    if not isinstance(payload, list):
        raise ValueError(f"{label} parameter probe payload must be a list of worker records")

    validated = []
    for rank, record in enumerate(payload):
        if not isinstance(record, dict):
            raise ValueError(f"{label} parameter probe rank {rank} must be a mapping")
        max_parameters = record.get("max_parameters")
        max_elements = record.get("max_elements_per_parameter")
        limits = (
            ("max_parameters", max_parameters, PARAMETER_PROBE_HARD_MAX_PARAMETERS),
            (
                "max_elements_per_parameter",
                max_elements,
                PARAMETER_PROBE_HARD_MAX_ELEMENTS_PER_PARAMETER,
            ),
        )
        for limit_name, value, hard_maximum in limits:
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0 or value > hard_maximum:
                raise ValueError(
                    f"{label} parameter probe rank {rank} has invalid {limit_name}; hard maximum is {hard_maximum}"
                )
        parameters = record.get("parameters")
        if not isinstance(parameters, list):
            raise ValueError(f"{label} parameter probe rank {rank} has an invalid parameters payload")
        if len(parameters) > max_parameters:
            raise ValueError(f"{label} parameter probe payload exceeds its configured parameter bound")
        if record.get("sampled_parameters") != len(parameters):
            raise ValueError(f"{label} parameter probe rank {rank} has inconsistent sampled_parameters")

        sampled_elements = 0
        names = set()
        for parameter in parameters:
            if not isinstance(parameter, dict):
                raise ValueError(f"{label} parameter probe rank {rank} contains a non-mapping parameter")
            name = parameter.get("name")
            values = parameter.get("values")
            sample_indices = parameter.get("sample_indices")
            numel = parameter.get("numel")
            shape = parameter.get("shape")
            if not isinstance(name, str) or not name or name in names:
                raise ValueError(f"{label} parameter probe rank {rank} contains an invalid or duplicate name")
            names.add(name)
            if not isinstance(numel, int) or isinstance(numel, bool) or numel < 0:
                raise ValueError(f"{label} parameter probe {name!r} has invalid numel")
            if shape is not None and (
                not isinstance(shape, list)
                or len(shape) > 16
                or any(
                    isinstance(dimension, bool) or not isinstance(dimension, int) or dimension < 0
                    for dimension in shape
                )
            ):
                raise ValueError(f"{label} parameter probe {name!r} has invalid shape")
            if not isinstance(values, list) or len(values) > max_elements:
                raise ValueError(f"{label} parameter probe payload exceeds its configured element bound")
            if not isinstance(sample_indices, list) or len(sample_indices) != len(values):
                raise ValueError(f"{label} parameter probe {name!r} has invalid sample_indices")
            if any(
                isinstance(index, bool) or not isinstance(index, int) or index < 0 or index >= numel
                for index in sample_indices
            ):
                raise ValueError(f"{label} parameter probe {name!r} has out-of-range sample_indices")
            if sample_indices != sorted(set(sample_indices)):
                raise ValueError(f"{label} parameter probe {name!r} sample_indices must be unique and increasing")
            if len(values) > numel:
                raise ValueError(f"{label} parameter probe {name!r} samples more elements than it owns")
            if any(isinstance(value, bool) or not isinstance(value, int | float) for value in values):
                raise ValueError(f"{label} parameter probe {name!r} contains a non-numeric value")
            sampled_elements += len(values)

        if record.get("sampled_elements") != sampled_elements:
            raise ValueError(f"{label} parameter probe rank {rank} has inconsistent sampled_elements")
        validated.append(record)
    return validated


def compare_parameter_update_probes(
    before: Any,
    after: Any,
    *,
    atol: float,
) -> dict[str, int | float | bool]:
    """Compare bounded per-rank parameter samples without materializing a model copy."""
    if isinstance(atol, bool) or not isinstance(atol, int | float) or not math.isfinite(atol) or atol < 0:
        raise ValueError(f"parameter probe atol must be a finite non-negative number, got {atol!r}")

    before_records = _validate_probe_payload(before, label="before")
    after_records = _validate_probe_payload(after, label="after")
    if len(before_records) != len(after_records):
        raise ValueError("parameter probe worker count changed across the update")

    changed_elements = 0
    sampled_elements = 0
    sampled_parameters = 0
    max_abs_diff = 0.0
    for rank, (before_record, after_record) in enumerate(zip(before_records, after_records, strict=True)):
        for bound_key in ("max_parameters", "max_elements_per_parameter"):
            if before_record[bound_key] != after_record[bound_key]:
                raise ValueError(f"parameter probe rank {rank} changed {bound_key} across the update")
        before_parameters = before_record["parameters"]
        after_parameters = after_record["parameters"]
        if len(before_parameters) != len(after_parameters):
            raise ValueError(f"parameter probe rank {rank} changed parameter count across the update")
        sampled_parameters += len(before_parameters)

        for before_parameter, after_parameter in zip(before_parameters, after_parameters, strict=True):
            structural_keys = ("name", "numel", "shape", "sample_indices")
            if any(before_parameter.get(key) != after_parameter.get(key) for key in structural_keys):
                raise ValueError(f"parameter probe structure changed across the update on rank {rank}")
            before_values = before_parameter["values"]
            after_values = after_parameter["values"]
            if len(before_values) != len(after_values):
                raise ValueError(f"parameter probe sample length changed across the update on rank {rank}")
            for before_value, after_value in zip(before_values, after_values, strict=True):
                difference = abs(float(after_value) - float(before_value))
                if not math.isfinite(difference):
                    raise RuntimeError(f"parameter probe observed a non-finite difference on rank {rank}")
                sampled_elements += 1
                max_abs_diff = max(max_abs_diff, difference)
                if difference > atol:
                    changed_elements += 1

    return {
        "changed": changed_elements > 0,
        "changed_elements": changed_elements,
        "sampled_elements": sampled_elements,
        "sampled_parameters": sampled_parameters,
        "worker_count": len(before_records),
        "max_abs_diff": max_abs_diff,
        "atol": float(atol),
    }


def atomic_write_runtime_receipt(
    filepath: str,
    receipt: dict[str, Any],
    *,
    project_name: str,
    experiment_name: str,
) -> None:
    """Write one FileLogger JSONL record and atomically publish it at ``filepath``."""
    final_path = os.path.abspath(os.path.expanduser(filepath))
    directory = os.path.dirname(final_path)
    os.makedirs(directory, exist_ok=True)
    fd, temporary_path = tempfile.mkstemp(prefix=f".{os.path.basename(final_path)}.", suffix=".tmp", dir=directory)
    os.close(fd)
    logger = None
    try:
        logger = FileLogger(project_name, experiment_name, filepath=temporary_path)
        logger.log(receipt, step=receipt.get("trainer_step", 0))
        logger.fp.flush()
        os.fsync(logger.fp.fileno())
        logger.finish()
        logger = None
        os.replace(temporary_path, final_path)
        directory_fd = os.open(directory, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if logger is not None:
            logger.finish()
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)
