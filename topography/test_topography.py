#!/usr/bin/env python3
"""Short tests of the scalp-topography code. Reads only; writes nothing.

Usage (from the archive root):
    python -B topography/test_topography.py
"""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np
from scipy.stats import rankdata

import run_topography as rt


def float_path_z(mark, indices):
    """General floating-point weighted-midrank path of rt.bootstrap_z for columns without ties."""
    n, c, d, _ = mark.shape
    left = rt.prepare_ranks(mark[..., 0].reshape(n, c * d))
    right = rt.prepare_ranks(mark[..., 1].reshape(n, c * d))
    counts = np.array([np.bincount(i, minlength=n) for i in indices], dtype=float)
    x = rt.weighted_ranks(left, counts)
    y = rt.weighted_ranks(right, counts)
    variance12 = n * (n * n - 1) - np.sum(counts ** 3 - counts, axis=1)
    rho = 1 - 6 * np.sum(counts[:, :, None] * (x - y) ** 2, axis=1) / variance12[:, None]
    return np.arctanh(rho).reshape(len(indices), c, d).mean(axis=2)


def explicit_z(mark, indices):
    """Resample participants, rank with scipy.stats.rankdata and correlate the ranks."""
    n, c, d, _ = mark.shape
    out = np.empty((len(indices), c))
    for b, i in enumerate(indices):
        x = rankdata(mark[i, ..., 0].reshape(n, -1), axis=0)
        y = rankdata(mark[i, ..., 1].reshape(n, -1), axis=0)
        x -= x.mean(axis=0)
        y -= y.mean(axis=0)
        rho = (x * y).sum(axis=0) / np.sqrt((x * x).sum(axis=0) * (y * y).sum(axis=0))
        out[b] = np.arctanh(rho).reshape(c, d).mean(axis=1)
    return out


class TopographyTests(unittest.TestCase):
    def test_channel_list(self):
        self.assertEqual(len(rt.CHANNELS), 62)
        self.assertNotIn("TP9", rt.CHANNELS)
        self.assertNotIn("TP10", rt.CHANNELS)
        self.assertEqual(len(set(rt.CHANNELS)), 62)
        self.assertEqual(rt.CHANNELS, tuple(c for c in rt.EEG_CHANNELS if c not in ("TP9", "TP10")))
        self.assertTrue(set(rt.ROI) <= set(rt.CHANNELS))

    def test_integer_bootstrap_matches_float_path(self):
        rng = np.random.default_rng(20261107)
        n = 41
        mark = rng.normal(size=(n, 2, 7, 2))
        mark[..., 1] += 0.6 * mark[..., 0]
        indices = rng.integers(0, n, size=(30, n))
        exact = rt.bootstrap_z_unique(mark, indices)
        np.testing.assert_allclose(exact, float_path_z(mark, indices), rtol=0, atol=1e-12)
        np.testing.assert_allclose(exact, explicit_z(mark, indices), rtol=0, atol=1e-12)
        np.testing.assert_allclose(rt.bootstrap_z(mark, indices), exact, rtol=0, atol=0)

    def test_general_path_with_ties(self):
        rng = np.random.default_rng(7)
        n = 33
        mark = rng.integers(0, 9, size=(n, 2, 5, 2)).astype(float)
        indices = rng.integers(0, n, size=(20, n))
        np.testing.assert_allclose(rt.bootstrap_z(mark, indices), explicit_z(mark, indices), rtol=0, atol=1e-12)

    def test_ratio_of_means_marker(self):
        powers = np.zeros((2, 42, 2))
        powers[..., 1] = 10.0
        powers[0, :, 0] = np.arange(42) % 5 + 1.0
        powers[0, :, 1] = np.arange(42) % 3 + 4.0
        powers[1, :, 0] = 2.0
        schedule = np.array([np.arange(10), np.arange(20, 30)])
        expected = [powers[p, schedule[p], 0].mean() / powers[p, schedule[p], 1].mean() for p in range(2)]
        np.testing.assert_allclose(rt.markers(powers, schedule), expected, rtol=1e-15, atol=0)
        ratio_mean = (powers[0, schedule[0], 0] / powers[0, schedule[0], 1]).mean()
        self.assertGreater(abs(rt.markers(powers, schedule)[0] - ratio_mean), 1e-6)
        self.assertAlmostEqual(rt.markers(powers, schedule)[1], 0.2, places=15)

    def test_plan_checksums(self):
        record = json.loads((HERE / "PLAN_RECORD.json").read_text(encoding="utf-8"))
        self.assertEqual(set(record["files"]), {"ANALYSIS_PLAN.md", "ANALYSIS_PLAN.json"})
        for name, data in record["files"].items():
            self.assertEqual(rt.sha(HERE / name), data["sha256"], name)
        self.assertEqual(json.loads((HERE / "ANALYSIS_PLAN.json").read_text(encoding="utf-8")), rt.PLAN)
        deviation = json.loads((HERE / "DEVIATION_RECORD.json").read_text(encoding="utf-8"))
        self.assertEqual(rt.sha(HERE / "PLAN_DEVIATION.md"), deviation["sha256"])


if __name__ == "__main__":
    unittest.main()
