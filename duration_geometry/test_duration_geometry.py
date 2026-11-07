#!/usr/bin/env python3
"""Short invariant tests for the duration-by-geometry analysis (synthetic data only)."""
import hashlib
import sys
import unittest
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from scipy.stats import rankdata

import run_duration_geometry as dg


class Sections(unittest.TestCase):
    def test_sizes_and_order(self):
        expected = {5: [9, 9, 8, 8, 8], 10: [5, 5] + [4] * 8, 20: [3, 3] + [2] * 18}
        for n, sizes in expected.items():
            sections = dg.split_sections(42, n)
            self.assertEqual([len(s) for s in sections], sizes)
            self.assertEqual([c for s in sections for c in s], list(range(42)))

    def test_distributed_one_cell_per_section(self):
        for n in dg.DURATIONS:
            sections = dg.split_sections(42, n)
            for d in range(50):
                cells = dg.draw_cells("sub-test", f"distributed_{n}", d)
                self.assertEqual(len(cells), n)
                for c, s in zip(cells, sections):
                    self.assertIn(c, s)


class Contiguous(unittest.TestCase):
    def test_block_bounds_and_all_starts(self):
        for n in dg.DURATIONS:
            starts = set()
            for d in range(2000):
                cells = dg.draw_cells(f"sub-{d % 7}", f"contiguous_{n}", d)
                self.assertEqual(cells, tuple(range(cells[0], cells[0] + n)))
                self.assertGreaterEqual(cells[0], 0)
                self.assertLessEqual(cells[-1], 41)
                starts.add(cells[0])
            self.assertEqual(starts, set(range(42 - n + 1)))


class Seeds(unittest.TestCase):
    def test_seed_rule(self):
        text = "duration-geometry-2026-11-07|sub-001|contiguous_10|0"
        expected = int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big")
        self.assertEqual(dg.draw_seed("sub-001", "contiguous_10", 0), expected)
        start = int(np.random.default_rng(expected).integers(0, 33))
        self.assertEqual(dg.draw_cells("sub-001", "contiguous_10", 0), tuple(range(start, start + 10)))

    def test_reproducible_and_stream_specific(self):
        a = dg.build_schedules(["sub-001", "sub-002"])
        b = dg.build_schedules(["sub-001", "sub-002"])
        for d in dg.DESIGNS:
            np.testing.assert_array_equal(a[d], b[d])
        self.assertFalse(np.array_equal(a["distributed_10"][0], a["distributed_10"][1]))

    def test_bootstrap_seed(self):
        text = "OSF-nh4d8|participant_bootstrap|duration_geometry_bootstrap"
        seed = int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big")
        expected = np.random.default_rng(seed).integers(0, 9, size=(3, 9), dtype=np.int64)
        np.testing.assert_array_equal(dg.bootstrap_indices(9, resamples=3), expected)


class Ratios(unittest.TestCase):
    def test_ratio_of_sums(self):
        rng = np.random.default_rng(1)
        total = rng.uniform(2, 3, (4, 42))
        powers = np.stack([total * rng.uniform(0.1, 0.9, (4, 42)), total], axis=-1)
        cells = np.array([[0, 1, 2], [5, 9, 41], [3, 3, 3], [10, 20, 30]])
        got = dg.ratio_of_sums(powers, cells)
        for i, c in enumerate(cells):
            self.assertAlmostEqual(got[i], powers[i, c, 0].sum() / powers[i, c, 1].sum(), places=14)
        self.assertFalse(np.allclose(got, (powers[np.arange(4)[:, None], cells, 0] / powers[np.arange(4)[:, None], cells, 1]).mean(1)))
        mask = rng.uniform(size=(4, 42)) > 0.2
        ref = dg.masked_ratio(powers, mask)
        for i in range(4):
            self.assertAlmostEqual(ref[i], powers[i, mask[i], 0].sum() / powers[i, mask[i], 1].sum(), places=14)


class Bootstrap(unittest.TestCase):
    def test_weighted_midranks_match_direct_resampling(self):
        rng = np.random.default_rng(2)
        x = rng.normal(size=(15, 4))
        x[3, 1] = x[7, 1]  # an original tie
        ref = rng.normal(size=15)
        idx = rng.integers(0, 15, size=(6, 15))
        got = dg.bootstrap_draw_rho(x, ref, idx, batch=4)
        for b, i in enumerate(idx):
            np.testing.assert_allclose(got[b], dg.spearman_columns(x[i], ref[i]), rtol=0, atol=1e-13)

    def test_spearman_matches_rankdata(self):
        rng = np.random.default_rng(3)
        x = rng.normal(size=(20, 3))
        ref = rng.normal(size=20)
        for j in range(3):
            a, b = rankdata(x[:, j]), rankdata(ref)
            self.assertAlmostEqual(dg.spearman_columns(x, ref)[j], np.corrcoef(a, b)[0, 1], places=13)


if __name__ == "__main__":
    unittest.main(verbosity=2)
