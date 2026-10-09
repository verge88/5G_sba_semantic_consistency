"""Unit and integration smoke tests for trained IF two-channel experiment."""
from __future__ import annotations

import sys
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "5g"))

from run_semantic_if_two_channel import (  # noqa: E402
    N_TREES, ORIGINS, batch_scores, features, fit_normal_only,
    fused_stat, rank_p,
)
from sba_semantic_v5 import OriginObservation  # noqa: E402
from sba_semantic_v8 import EvidenceState, SemanticFact  # noqa: E402


class IfTwoChannelTests(unittest.TestCase):
    def _case(self):
        return {
            origin: OriginObservation(
                event_id="test-event",
                fact=SemanticFact.TOKEN,
                origin=origin,
                trust_root=origin.value,
                state=EvidenceState.CONSISTENT,
                observed_at=1.0 + i * 0.02,
                confidence=0.9,
            )
            for i, origin in enumerate(ORIGINS)
        }

    def test_dimension_and_masking(self):
        case = self._case()
        normal = features(case)
        self.assertEqual(len(normal), len(ORIGINS) * 7 + 4)
        for origin in ORIGINS:
            masked = features(case, excluded=origin)
            self.assertEqual(len(normal), len(masked))
            i = list(ORIGINS).index(origin)
            self.assertEqual(masked[7 * i + 4], 1.0)
            self.assertEqual(masked[7 * i + 1], 0.0)

    def test_unknown_and_absence_are_separate(self):
        case = self._case()
        origin = ORIGINS[0]
        unknown = dict(case)
        unknown[origin] = replace(case[origin], state=EvidenceState.UNKNOWN)
        missing = dict(case)
        del missing[origin]
        self.assertEqual(features(unknown)[2], 1.)
        self.assertEqual(features(unknown)[3], 0.)
        self.assertEqual(features(missing)[2], 0.)
        self.assertEqual(features(missing)[3], 1.)

    def test_rank_ties_conservative(self):
        self.assertEqual(rank_p([0.] * 100, 0.), 1.)
        self.assertAlmostEqual(rank_p([0.] * 100, 1.), 1. / 101.)
        self.assertAlmostEqual(rank_p([0., 1., 2.], 2.), 0.5)
        self.assertAlmostEqual(rank_p([0., 1., 2.], 3.), 0.25)

    def test_fusion_depends_on_both_channels(self):
        calibrators = {"normal_indirect": {
            "raw": [0., 1., 2.], "loo": [0., 1., 2.]}}
        both = fused_stat(3., 3., "normal_indirect", calibrators)
        raw_only = fused_stat(3., 0., "normal_indirect", calibrators)
        self.assertAlmostEqual(both, raw_only)
        self.assertGreater(raw_only, fused_stat(0., 0., "normal_indirect", calibrators))

    def test_training_and_scores_use_benign_features(self):
        case = self._case()
        train = [(case, list(ORIGINS))] * 300
        raw, loo = fit_normal_only(9001, train)
        self.assertEqual(raw.n_estimators, N_TREES)
        direct, robust = batch_scores(raw, loo, [case])
        self.assertEqual(direct.shape, (1,))
        self.assertEqual(robust.shape, (1,))
        self.assertTrue(np.isfinite(direct[0]))
        self.assertTrue(np.isfinite(robust[0]))


if __name__ == "__main__":
    unittest.main()
