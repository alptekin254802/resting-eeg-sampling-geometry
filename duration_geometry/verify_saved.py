#!/usr/bin/env python3
"""Check the saved duration-by-geometry results against a recomputation.

Reads only the archive inputs and the saved files in this folder; writes nothing.
By default the bootstrap is checked on its index checksum, the saved
percentiles and the first and last resamples recomputed by explicit
resampling. --full repeats all 5000 resamples (about five minutes).

Usage (from the archive root):
    python -B duration_geometry/verify_saved.py [--full]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np

import run_duration_geometry as dg

ATOL = 1e-12  # computed floating-point values; identifiers, schedules, seeds and checksums stay exact


def close(actual, expected) -> bool:
    """Floats within ATOL; strings, integers and booleans exactly; dicts and lists element by element."""
    if isinstance(expected, dict):
        return isinstance(actual, dict) and actual.keys() == expected.keys() and all(close(actual[k], expected[k]) for k in expected)
    if isinstance(expected, (list, tuple)):
        return isinstance(actual, (list, tuple)) and len(actual) == len(expected) and all(close(a, e) for a, e in zip(actual, expected))
    if isinstance(expected, float) or isinstance(actual, float):
        return not isinstance(actual, (bool, str)) and not isinstance(expected, (bool, str)) and abs(float(actual) - float(expected)) <= ATOL
    return type(actual) is type(expected) and actual == expected


def allclose(actual, expected) -> bool:
    actual, expected = np.asarray(actual), np.asarray(expected)
    return actual.shape == expected.shape and bool(np.allclose(actual, expected, rtol=0, atol=ATOL))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--archive", type=Path, default=HERE.parent)
    parser.add_argument("--results", type=Path, default=HERE)
    parser.add_argument("--full", action="store_true", help="recompute all bootstrap resamples")
    args = parser.parse_args(argv)
    archive, folder = args.archive.resolve(), args.results.resolve()
    saved = json.loads((folder / "results_summary.json").read_text(encoding="utf-8"))
    with np.load(folder / "duration_geometry_draws.npz") as z:
        draws = {k: z[k] for k in z.files}
    with np.load(folder / "duration_geometry_inference.npz") as z:
        inf = {k: z[k] for k in z.files}
    checks = []

    def check(name, ok):
        checks.append((name, bool(ok)))
        print(("PASS " if ok else "FAIL ") + name, flush=True)

    cohort = dg.load_cohort(archive)
    check("compatibility checks pass and match saved record", close(cohort["checks"], saved["compatibility_checks"]))
    for key, path in [(dg.PRE_FILE, dg.PRE_FILE), (dg.POST_FILE, dg.POST_FILE), (dg.SCIENTIFIC_FILE, dg.SCIENTIFIC_FILE)]:
        check(f"input checksum {key}", dg.file_sha256(archive / path) == saved["input_sha256"][key])
    pids = cohort["participants"]
    check("participant identifiers", np.array_equal(pids, draws["participant_ids"]) and saved["n"] == len(pids) == dg.EXPECTED_N)
    check("design labels", list(draws["design_labels"]) == list(dg.DESIGNS) == list(inf["design_labels"]))

    schedules = dg.build_schedules(pids)
    check("schedules regenerate exactly", all(np.array_equal(schedules[d], draws[f"cells_{d}"]) for d in dg.DESIGNS))
    check("schedule checksums", dg.schedule_hashes(schedules) == saved["schedule_sha256_uint8_c_order"])
    ok = True
    for d in dg.DESIGNS:
        kind, n = dg.parse_design(d)
        a = schedules[d].astype(int)
        if kind == "contiguous":
            ok &= bool(np.all(np.diff(a, axis=-1) == 1) and a.min() >= 0 and a.max() <= 41)
        else:
            for k, section in enumerate(dg.split_sections(42, n)):
                ok &= bool(np.isin(a[..., k], section).all())
    check("schedule construction rules", ok)

    reference = dg.masked_ratio(cohort["post_power"], cohort["post_mask"])
    check("reference values", allclose(reference, draws["reference"]))
    values = np.stack([dg.ratio_of_sums(cohort["pre"], schedules[d]) for d in dg.DESIGNS], axis=1)
    check("sample values", allclose(values, draws["sample_values"]))
    n = len(pids)
    rho = dg.spearman_columns(values.reshape(n, -1), reference).reshape(len(dg.DESIGNS), dg.NDRAWS)
    check("draw correlations", allclose(rho, inf["draw_rho"]) and allclose(np.arctanh(rho), inf["draw_z"]))
    sys.path.insert(0, str(archive / "scripts"))
    from preregistered_analysis import spearman_midrank_columns
    other = spearman_midrank_columns(values.reshape(n, -1), np.repeat(reference[:, None], values[0].size, axis=1))
    check("draw correlations match archive Spearman function", allclose(other.reshape(rho.shape), rho))

    indices = dg.bootstrap_indices(n)
    check("bootstrap index checksum", dg.index_sha256(indices) == saved["bootstrap_index_sha256"])
    check("bootstrap seed", dg.seed64(saved["bootstrap_seed_string"]) == saved["bootstrap_seed"])
    flat = values.reshape(n, -1)
    for b in (0, len(indices) - 1):
        i = indices[b]
        direct = dg.spearman_columns(flat[i], reference[i]).reshape(len(dg.DESIGNS), dg.NDRAWS)
        check(f"bootstrap resample {b} by explicit resampling",
              allclose(np.arctanh(direct).mean(axis=1), inf["bootstrap_mean_z"][b])
              and allclose(direct.mean(axis=1), inf["bootstrap_mean_rho"][b]))
    if args.full:
        res = dg.analyse(cohort, indices=indices, log=lambda s: None)
        check("full bootstrap recomputation", allclose(res["bootstrap_mean_z"], inf["bootstrap_mean_z"])
              and allclose(res["bootstrap_mean_rho"], inf["bootstrap_mean_rho"]))

    res = dict(draw_rho=inf["draw_rho"], draw_z=inf["draw_z"], bootstrap_mean_z=inf["bootstrap_mean_z"],
               bootstrap_mean_rho=inf["bootstrap_mean_rho"])
    summary = dg.summarise(res)
    check("design summaries (within 1e-12)", close(summary["designs"], saved["designs"]))
    check("contrast summaries (within 1e-12)", close(summary["contrasts"], saved["contrasts"]))
    check("contrast list matches plan", [c["contrast"] for c in saved["contrasts"]] == [f"{a} - {b}" for _, a, b in dg.CONTRASTS])

    failed = [name for name, ok in checks if not ok]
    print(f"{len(checks) - len(failed)}/{len(checks)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
