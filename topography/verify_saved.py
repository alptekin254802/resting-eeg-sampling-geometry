#!/usr/bin/env python3
"""Check the saved scalp-topography results against a recomputation.

Reads only the archive inputs and the saved files in this folder; writes nothing.
The validation checks and the channel-level point estimates are recomputed in
full. The bootstrap is checked on its index checksums, the saved percentiles
and the first and last resamples recomputed by explicit resampling with
scipy.stats.rankdata. --full repeats all 5000 resamples for every channel
(about 10 to 40 minutes, depending on --workers).

Usage (from the archive root):
    python -B topography/verify_saved.py [--full] [--workers N]
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np

import run_topography as rt

ATOL = 1e-12  # computed floating-point values; identifiers, schedules and checksums stay exact


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


def recompute_validation():
    """Run rt.validate() with its file writes captured in memory."""
    captured = {}
    write_json, log = rt.write_json, rt.log
    rt.write_json = lambda path, value: captured.setdefault(Path(path).name, json.loads(json.dumps(value, default=lambda v: v.tolist() if isinstance(v, np.ndarray) else v.item())))
    rt.log = lambda message: None
    try:
        rt.validate()
    except rt.StopAnalysis:
        pass
    finally:
        rt.write_json, rt.log = write_json, log
    return captured.get("VALIDATION.json")


def explicit_bootstrap_z(marks, index):
    """Mean Fisher z per channel and geometry for one resample, by resampling and ranking directly."""
    from scipy.stats import rankdata
    n = len(index)
    out = np.empty(marks.shape[1:3])
    for j in range(marks.shape[1]):
        sample = marks[index, j]
        x = rankdata(sample[..., 0].reshape(n, -1), axis=0)
        y = rankdata(sample[..., 1].reshape(n, -1), axis=0)
        x -= x.mean(axis=0)
        y -= y.mean(axis=0)
        rho = (x * y).sum(axis=0) / np.sqrt((x * x).sum(axis=0) * (y * y).sum(axis=0))
        out[j] = np.arctanh(rho).reshape(marks.shape[2:4]).mean(axis=-1)
    return out


def summarise(mean_z, draw_rho, boot, n, validation_reference):
    """Rebuild one comparison of results_summary.json as rt.report() does."""
    from scipy.stats import spearmanr
    delta = mean_z[:, 1] - mean_z[:, 0]
    delta_boot = boot[:, :, 1] - boot[:, :, 0]
    draw_delta = np.arctanh(draw_rho[:, 1]) - np.arctanh(draw_rho[:, 0])
    mcse = draw_delta.std(axis=1, ddof=1) / np.sqrt(200)
    ci = rt.interval(delta_boot)
    rho_ci = np.tanh(rt.interval(boot))
    per_channel = [{"channel": c, "n": n,
                    "rho_C": float(np.tanh(mean_z[j, 0])), "rho_C_ci95": rho_ci[:, j, 0].tolist(),
                    "rho_F10": float(np.tanh(mean_z[j, 1])), "rho_F10_ci95": rho_ci[:, j, 1].tolist(),
                    "delta_z": float(delta[j]), "delta_z_ci95": ci[:, j].tolist(), "delta_z_mcse": float(mcse[j])}
                   for j, c in enumerate(rt.CHANNELS)]
    imin, imax = int(np.argmin(delta)), int(np.argmax(delta))
    roi = [rt.CHANNELS.index(c) for c in rt.ROI]
    return {
        "n": n,
        "channel_mean_delta_z": float(delta.mean()),
        "channel_mean_delta_z_ci95": rt.interval(delta_boot.mean(axis=1)).tolist(),
        "channels_delta_z_positive": int((delta > 0).sum()),
        "channels_ci_wholly_above_zero": int((ci[0] > 0).sum()),
        "channels_ci_wholly_below_zero": int((ci[1] < 0).sum()),
        "min_delta_z": {"channel": rt.CHANNELS[imin], "delta_z": float(delta[imin]), "ci95": ci[:, imin].tolist()},
        "max_delta_z": {"channel": rt.CHANNELS[imax], "delta_z": float(delta[imax]), "ci95": ci[:, imax].tolist()},
        "spearman_rhoC_vs_delta_z_across_channels": float(spearmanr(np.tanh(mean_z[:, 0]), delta).statistic),
        "rho_C_range": [float(np.tanh(mean_z[:, 0]).min()), float(np.tanh(mean_z[:, 0]).max())],
        "rho_F10_range": [float(np.tanh(mean_z[:, 1]).min()), float(np.tanh(mean_z[:, 1]).max())],
        "single_channel_O1_Oz_O2_delta_z": {rt.CHANNELS[j]: float(delta[j]) for j in roi},
        "posterior_roi_reference": validation_reference,
        "per_channel": per_channel,
    }


def csv_text(summary):
    """channel_results.csv as rt.report() writes it, from the saved summary values."""
    rows = []
    for comparison in rt.COMPARISONS:
        for item in summary["comparisons"][comparison]["per_channel"]:
            rows.append({"comparison": comparison, **{k: (f"{v:.6f}" if isinstance(v, float) else v)
                                                       for k, v in item.items() if not isinstance(v, list)},
                         "rho_C_ci_low": f"{item['rho_C_ci95'][0]:.6f}", "rho_C_ci_high": f"{item['rho_C_ci95'][1]:.6f}",
                         "rho_F10_ci_low": f"{item['rho_F10_ci95'][0]:.6f}", "rho_F10_ci_high": f"{item['rho_F10_ci95'][1]:.6f}",
                         "delta_z_ci_low": f"{item['delta_z_ci95'][0]:.6f}", "delta_z_ci_high": f"{item['delta_z_ci95'][1]:.6f}"})
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--full", action="store_true", help="recompute all 5000 bootstrap resamples for every channel")
    parser.add_argument("--workers", type=int, default=2, help="processes for --full")
    args = parser.parse_args(argv)
    folder = rt.OUT
    checks = []

    def check(name, ok):
        checks.append((name, bool(ok)))
        print(("PASS " if ok else "FAIL ") + name, flush=True)

    def load(name):
        return json.loads((folder / name).read_text(encoding="utf-8"))

    # Plan, deviation and run records.
    plan_record = load("PLAN_RECORD.json")
    check("plan checksums match PLAN_RECORD.json",
          all(rt.sha(folder / name) == data["sha256"] for name, data in plan_record["files"].items())
          and set(plan_record["files"]) == {"ANALYSIS_PLAN.md", "ANALYSIS_PLAN.json"})
    check("plan text matches the script", load("ANALYSIS_PLAN.json") == rt.PLAN
          and (folder / "ANALYSIS_PLAN.md").read_text(encoding="utf-8") == rt.plan_md())
    deviation = load("DEVIATION_RECORD.json")
    check("deviation checksum matches DEVIATION_RECORD.json", rt.sha(folder / "PLAN_DEVIATION.md") == deviation["sha256"])
    run_record = load("RUN_RECORD.json")
    check("run record checksums", run_record["plan_record_sha256"] == rt.sha(folder / "PLAN_RECORD.json")
          and run_record["validation_sha256"] == rt.sha(folder / "VALIDATION.json")
          and all(rt.sha(folder / name) == digest for name, digest in run_record["files_sha256"].items()))
    check("input checksums (schedule and census)", rt.sha(rt.SCHEDULE_FILE) == rt.SCHEDULE_SHA256
          and rt.sha(rt.CENSUS_FILE) == rt.CENSUS_SHA256)

    # Channels.
    summary = load("results_summary.json")
    with np.load(folder / "topography_inference.npz", allow_pickle=False) as z:
        inf = {k: z[k] for k in z.files}
    channel_ok = len(rt.CHANNELS) == 62 and not set(rt.REFERENCE) & set(rt.CHANNELS)
    channel_ok &= summary["channels"] == list(rt.CHANNELS) and tuple(inf["channels"].astype(str)) == rt.CHANNELS
    for cohort in ("pre", "post"):
        with np.load(folder / f"{cohort}_channel_band_powers.npz", allow_pickle=False) as z:
            channel_ok &= tuple(z["channels"].astype(str)) == rt.CHANNELS
    check("62 channels without TP9 and TP10 in every file", channel_ok)

    # Validation, recomputed in full.
    saved_validation = load("VALIDATION.json")
    validation = recompute_validation()
    ok = validation is not None and validation["passed"] and saved_validation["passed"]
    ok = ok and all(validation[k] <= v for k, v in validation["limits"].items())
    ok = ok and close({k: v for k, v in validation.items() if k != "created_at_utc"},
                      {k: v for k, v in saved_validation.items() if k != "created_at_utc"})
    check("validation recomputed: all limits met and saved values reproduced", ok)

    # Channel-level point estimates, recomputed in full.
    data = rt.comparison_markers()
    points = {}
    for comparison in rt.COMPARISONS:
        ids, marks = data[comparison]
        mean_z = np.empty((len(rt.CHANNELS), 2))
        draw_rho = np.empty((len(rt.CHANNELS), 2, 200))
        for j in range(len(rt.CHANNELS)):
            mean_z[j], draw_rho[j] = rt.point(marks[:, j])
        points[comparison] = mean_z, draw_rho
        check(f"{comparison}: participant identifiers", np.array_equal(ids, inf[f"{comparison}_participant_ids"].astype(str)))
        check(f"{comparison}: channel draw correlations and mean Fisher z",
              allclose(draw_rho, inf[f"{comparison}_draw_rho"]) and allclose(mean_z, inf[f"{comparison}_mean_z"]))

    # Bootstrap.
    indices = {c: rt.bootstrap_indices(c) for c in rt.COMPARISONS}
    for comparison in rt.COMPARISONS:
        ids, marks = data[comparison]
        index = indices[comparison]
        check(f"{comparison}: bootstrap index checksum",
              index.shape == (5000, len(ids)) and rt.array_sha(index, "<i8") == str(inf[f"{comparison}_bootstrap_indices_sha256"]))
        saved_boot = inf[f"{comparison}_bootstrap_z"]
        for b in (0, len(index) - 1):
            check(f"{comparison}: bootstrap resample {b} by explicit resampling",
                  allclose(explicit_bootstrap_z(marks, index[b]), saved_boot[b]))
    if args.full:
        for comparison in rt.COMPARISONS:
            saved_boot = inf[f"{comparison}_bootstrap_z"]
            ok = True
            with ProcessPoolExecutor(max_workers=args.workers, initializer=rt._init_worker, initargs=(comparison,)) as pool:
                for j, _, _, boot in pool.map(rt._channel_job, range(len(rt.CHANNELS))):
                    ok &= allclose(boot, saved_boot[:, j])
            check(f"{comparison}: all 5000 bootstrap resamples, every channel", ok)

    # Summaries: point estimates recomputed above, saved bootstrap, saved percentile rule.
    for comparison in rt.COMPARISONS:
        mean_z, draw_rho = points[comparison]
        rebuilt = summarise(mean_z, draw_rho, inf[f"{comparison}_bootstrap_z"], len(data[comparison][0]),
                            saved_validation["posterior_reference"][comparison])
        check(f"{comparison}: summary and per-channel values (within 1e-12)", close(rebuilt, summary["comparisons"][comparison]))
    check("summary status and plan deviation record", summary["status"] == rt.PLAN["status"]
          and summary.get("plan_deviations") == deviation)
    check("channel_results.csv matches the summary",
          (folder / "channel_results.csv").read_bytes().decode("utf-8").replace("\r\n", "\n")
          == csv_text(summary).replace("\r\n", "\n"))

    failed = [name for name, ok in checks if not ok]
    print(f"{len(checks) - len(failed)}/{len(checks)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
