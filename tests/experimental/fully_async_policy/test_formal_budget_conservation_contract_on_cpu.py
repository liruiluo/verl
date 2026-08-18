# Copyright 2026 The verl team.
# Licensed under the Apache License, Version 2.0.

"""External formal100 conservation contract over native queue telemetry.

The checker intentionally does not change MessageQueue FIFO/drop-oldest
semantics and does not authorize refill, reissue, a batch coordinator, or a
longer producer horizon. It only decides whether a finished lineage is the
declared 6,400-episode/100-collection experiment. Scheduled, produced, and
consumed identities are checked independently; equal scalar counts are not
enough.
"""

from __future__ import annotations

import copy
import json
import os
import unittest
from collections import Counter
from pathlib import Path

FORMAL_EPISODES = 6_400
EPISODES_PER_COLLECTION = 64
COMPLETE_COLLECTIONS = 100
COLLECTIONS_PER_PUBLICATION = 4
PUBLICATION_CYCLES = 25


def _validate_formal_conservation(
    scheduled_ids: list[str],
    produced_ids: list[str],
    consumed_ids: list[str],
    receipt: dict,
) -> None:
    scheduled = Counter(scheduled_ids)
    produced = Counter(produced_ids)
    consumed = Counter(consumed_ids)

    if len(scheduled_ids) != FORMAL_EPISODES or len(scheduled) != FORMAL_EPISODES:
        raise ValueError("frozen schedule must contain 6400 unique episode item IDs")
    if produced != scheduled:
        raise ValueError("scheduled episode IDs must be produced exactly once with no substitution or duplication")
    if consumed != scheduled:
        raise ValueError("scheduled episode IDs must be consumed exactly once")

    expected = {
        "scheduled_real": FORMAL_EPISODES,
        "produced_real": FORMAL_EPISODES,
        "consumed_real": FORMAL_EPISODES,
        "refill_count": 0,
        "reissue_count": 0,
        "real_evicted": 0,
        "real_cleared": 0,
        "real_resident": 0,
        "terminal_underfill": 0,
        "complete_learner_collections": COMPLETE_COLLECTIONS,
        "episodes_per_collection": EPISODES_PER_COLLECTION,
        "publication_cycles": PUBLICATION_CYCLES,
        "collections_per_publication": COLLECTIONS_PER_PUBLICATION,
        "learner_collection_horizon": COMPLETE_COLLECTIONS,
        "actor_lr_scheduler_steps": COMPLETE_COLLECTIONS,
        "critic_lr_scheduler_steps": COMPLETE_COLLECTIONS,
    }
    for key, value in expected.items():
        if receipt.get(key) != value:
            raise ValueError(f"formal conservation requires {key}={value}, got {receipt.get(key)!r}")
    if receipt.get("queue_closed") is not True:
        raise ValueError("formal queue must be closed")
    if receipt.get("generation_horizon_extended") is not False:
        raise ValueError("generation horizon must remain the frozen 6400 episodes")
    if receipt.get("validation_rpc_count") != 0 or receipt.get("validation_metric_count") != 0:
        raise ValueError("no validation may run before the declared endpoint")


def _valid_receipt() -> dict:
    return {
        "scheduled_real": FORMAL_EPISODES,
        "produced_real": FORMAL_EPISODES,
        "consumed_real": FORMAL_EPISODES,
        "refill_count": 0,
        "reissue_count": 0,
        "real_evicted": 0,
        "real_cleared": 0,
        "real_resident": 0,
        "terminal_underfill": 0,
        "complete_learner_collections": COMPLETE_COLLECTIONS,
        "episodes_per_collection": EPISODES_PER_COLLECTION,
        "publication_cycles": PUBLICATION_CYCLES,
        "collections_per_publication": COLLECTIONS_PER_PUBLICATION,
        "learner_collection_horizon": COMPLETE_COLLECTIONS,
        "actor_lr_scheduler_steps": COMPLETE_COLLECTIONS,
        "critic_lr_scheduler_steps": COMPLETE_COLLECTIONS,
        "queue_closed": True,
        "generation_horizon_extended": False,
        "validation_rpc_count": 0,
        "validation_metric_count": 0,
    }


class TestFormalBudgetConservation(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        explicit = os.environ.get("AMG_SYNC_BASELINE_R14")
        if explicit:
            cls.baseline = Path(explicit)
        else:
            workspace = next(
                parent for parent in Path(__file__).resolve().parents if parent.name == "agentmemorygym-rl-workspace"
            )
            cls.baseline = workspace / "operations/verl-fully-async-20260817/sync-baseline-r14"
        cls.certificate = json.loads((cls.baseline / "formal100-schedule-certificate.json").read_text())
        cls.schedule = [
            json.loads(line) for line in (cls.baseline / "formal100-train-schedule.jsonl").read_text().splitlines()
        ]
        cls.scheduled_ids = [row["item_id"] for row in cls.schedule]

    def test_frozen_comparator_schedule_matches_declared_budget(self) -> None:
        self.assertEqual(self.certificate["scheduled_episode_count"], FORMAL_EPISODES)
        self.assertEqual(self.certificate["optimizer_updates"], COMPLETE_COLLECTIONS)
        self.assertEqual(self.certificate["train_batch_size"], EPISODES_PER_COLLECTION)
        self.assertEqual(len(self.scheduled_ids), FORMAL_EPISODES)
        self.assertEqual(len(set(self.scheduled_ids)), FORMAL_EPISODES)

    def test_exact_three_way_identity_conservation_is_a_valid_endpoint(self) -> None:
        _validate_formal_conservation(
            self.scheduled_ids,
            list(self.scheduled_ids),
            list(self.scheduled_ids),
            _valid_receipt(),
        )

    def test_scalar_or_queue_failures_invalidate_lineage(self) -> None:
        receipt_mutations = (
            ("produced_real", FORMAL_EPISODES - 1),
            ("consumed_real", FORMAL_EPISODES - 1),
            ("refill_count", 1),
            ("reissue_count", 1),
            ("real_evicted", 1),
            ("terminal_underfill", 1),
            ("complete_learner_collections", COMPLETE_COLLECTIONS - 1),
            ("publication_cycles", PUBLICATION_CYCLES - 1),
            ("real_resident", 1),
            ("queue_closed", False),
            ("generation_horizon_extended", True),
        )
        for key, value in receipt_mutations:
            with self.subTest(key=key):
                receipt = copy.deepcopy(_valid_receipt())
                receipt[key] = value
                with self.assertRaises(ValueError):
                    _validate_formal_conservation(
                        self.scheduled_ids,
                        list(self.scheduled_ids),
                        list(self.scheduled_ids),
                        receipt,
                    )

    def test_produced_substitution_or_duplication_cannot_hide_behind_equal_counts(self) -> None:
        substituted = list(self.scheduled_ids)
        substituted[-1] = "not-in-frozen-schedule"
        with self.assertRaisesRegex(ValueError, "produced exactly once"):
            _validate_formal_conservation(
                self.scheduled_ids,
                substituted,
                list(self.scheduled_ids),
                _valid_receipt(),
            )

        duplicated = list(self.scheduled_ids)
        duplicated[-1] = duplicated[0]
        with self.assertRaisesRegex(ValueError, "produced exactly once"):
            _validate_formal_conservation(
                self.scheduled_ids,
                duplicated,
                list(self.scheduled_ids),
                _valid_receipt(),
            )

    def test_consumed_duplication_is_rejected_independently(self) -> None:
        duplicate = list(self.scheduled_ids)
        duplicate[-1] = duplicate[0]
        with self.assertRaisesRegex(ValueError, "consumed exactly once"):
            _validate_formal_conservation(
                self.scheduled_ids,
                list(self.scheduled_ids),
                duplicate,
                _valid_receipt(),
            )


if __name__ == "__main__":
    unittest.main()
