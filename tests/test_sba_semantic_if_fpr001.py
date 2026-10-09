"""Regression tests for 0.1% FPR and suppressed-evidence experimental protocol."""
from __future__ import annotations

import sys
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "5g"))

from run_semantic_if_fpr001 import (  # noqa: E402
    ORIGINS, SCENARIOS, TARGET_FPR,
    make_world, choose_special_window, bools_from_scores, rate_row,
)
from sba_semantic_v5 import OriginObservation  # noqa: E402
from sba_semantic_v8 import EvidenceState, SemanticFact  # noqa: E402


class Fpr001Tests(unittest.TestCase):
    def reports(self):
        return {
            origin: OriginObservation(
                event_id="event-001",
                fact=SemanticFact.TOKEN,
                origin=origin,
                trust_root=origin.value,
                state=EvidenceState.CONSISTENT,
                observed_at=0.05 + i * 0.01,
                confidence=1.0,
            )
            for i, origin in enumerate(ORIGINS)
        }

    def test_alpha_target_is_point_one_percent(self):
        self.assertEqual(TARGET_FPR, 0.001)

    def test_single_origin_flip_and_suppress_provenance(self):
        reports = self.reports()
        amended = make_world(reports, list(ORIGINS), "flip1_suppress1", rotation=0)
        self.assertEqual(len(amended), len(reports)-1)
        self.assertEqual(
            sum(x.state == EvidenceState.INCONSISTENT for x in amended.values()), 1
        )
        self.assertEqual(reports[ORIGINS[0]].state, EvidenceState.CONSISTENT)
        self.assertEqual(amended[ORIGINS[0]].state, EvidenceState.INCONSISTENT)
        self.assertNotIn(ORIGINS[1], amended)

    def test_unknown_mutations_keep_original_unknown(self):
        reports = self.reports()
        reports[ORIGINS[-1]] = replace(reports[ORIGINS[-1]], state=EvidenceState.UNKNOWN)
        consistent = list(ORIGINS[:-1])
        for scenario in SCENARIOS[:7]:
            changed = make_world(reports, consistent, scenario, rotation=0)
            self.assertIn(ORIGINS[-1], changed)
            self.assertEqual(changed[ORIGINS[-1]].state, EvidenceState.UNKNOWN)

    def test_rotation_spreads_interventions_over_sources(self):
        reports = self.reports()
        seen = set()
        for k in range(5):
            changed = make_world(reports, list(ORIGINS), "flip1", rotation=k)
            seen.add(next(origin for origin in ORIGINS
                          if changed[origin].state == EvidenceState.INCONSISTENT))
        self.assertEqual(seen, set(ORIGINS))

    def test_special_selection_uses_event_id_only_in_evaluation(self):
        reports = self.reports()
        window = SimpleNamespace(
            observations=list(reports.values()),
            operational_event_ids=frozenset(["event-001"]),
            attack_event_ids=frozenset(["event-001"]),
        )
        self.assertEqual(
            choose_special_window(window, "operational_token_refresh"), reports
        )
        self.assertEqual(choose_special_window(window, "attack_no_token"), reports)
        window.operational_event_ids = frozenset(["other"])
        self.assertIsNone(
            choose_special_window(window, "operational_token_refresh")
        )

    def test_joint_score_calibrated_separately(self):
        # No assumption that raw and masked p-values are independent.
        score_calib = {"normal_indirect": {
            "raw": [0.1,0.2,0.3,0.4,0.5],
            "loo": [0.1,0.2,0.3,0.4,0.5]}}
        fusion_calib = {"normal_indirect": [0.1,0.2,0.3,0.4,0.5]}
        raw, loo, combined = bools_from_scores(
            0.8,0.8,"normal_indirect",score_calib,fusion_calib,0.2
        )
        self.assertTrue(raw)
        self.assertTrue(loo)
        self.assertTrue(combined)

    def test_rate_row_wilson(self):
        row = rate_row(301,"normal_indirect","normal","if_raw",0,10000)
        self.assertEqual(row["rate"],0.0)
        self.assertGreater(row["wilson95_hi"],0.0)


if __name__ == "__main__":
    unittest.main()
