#!/usr/bin/env python3
"""Exploratory duration-by-geometry analysis.

One pre-battery sample per participant and draw is compared with one
post-battery reference value per participant. Designs: contiguous or
distributed placement of 5, 10 or 20 four-second cells (20, 40, 80 s).
The full specification is in ANALYSIS_PLAN.md in this folder.

Usage (from the archive root):
    python -B duration_geometry/run_duration_geometry.py --output outputs/duration_geometry
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import platform
import sys
from pathlib import Path

sys.dont_write_bytecode = True
for _key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_key, "2")

import numpy as np
from scipy.stats import rankdata

HERE = Path(__file__).resolve().parent
NCELLS = 42
NDRAWS = 200
NBOOT = 5000
EXPECTED_N = 509
DURATIONS = (5, 10, 20)
DESIGNS = tuple(f"{kind}_{n}" for kind in ("contiguous", "distributed") for n in DURATIONS)
DRAW_NAMESPACE = "duration-geometry-2026-11-07|{participant}|{design}|{draw}"
BOOTSTRAP_LABEL = "duration_geometry_bootstrap"
BOOTSTRAP_NAMESPACE = "OSF-nh4d8|participant_bootstrap|{label}"
PRE_FILE = "results/exploratory_2026-09-20/cell_band_powers.npz"
POST_FILE = "additional/results/post_prepared.npz"
SCIENTIFIC_FILE = "results/preregistered/scientific_results.npz"
TOLERANCE = 1e-12
CONTRASTS = (
    ("a_placement", "distributed_5", "contiguous_5"),
    ("a_placement", "distributed_10", "contiguous_10"),
    ("a_placement", "distributed_20", "contiguous_20"),
    ("b_duration", "contiguous_10", "contiguous_5"),
    ("b_duration", "contiguous_20", "contiguous_10"),
    ("b_duration", "distributed_10", "distributed_5"),
    ("b_duration", "distributed_20", "distributed_10"),
    ("c_placement_versus_duration", "distributed_5", "contiguous_10"),
    ("c_placement_versus_duration", "distributed_10", "contiguous_20"),
)


class AnalysisStop(RuntimeError):
    """A plan rule requires the analysis to stop."""


# ---------------------------------------------------------------- sampling
def split_sections(n_cells: int, k: int) -> list[tuple[int, ...]]:
    """Existing split_strata rule: divmod; the extra cells go to the first sections."""
    q, r = divmod(n_cells, k)
    out, start = [], 0
    for i in range(k):
        size = q + (1 if i < r else 0)
        out.append(tuple(range(start, start + size)))
        start += size
    return out


def seed64(text: str) -> int:
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big", signed=False)


def draw_seed(participant: str, design: str, draw: int) -> int:
    return seed64(DRAW_NAMESPACE.format(participant=participant, design=design, draw=draw))


def parse_design(design: str) -> tuple[str, int]:
    kind, n = design.split("_")
    if kind not in ("contiguous", "distributed") or int(n) not in DURATIONS:
        raise ValueError(design)
    return kind, int(n)


def draw_cells(participant: str, design: str, draw: int) -> tuple[int, ...]:
    kind, n = parse_design(design)
    rng = np.random.default_rng(draw_seed(participant, design, draw))
    if kind == "contiguous":
        start = int(rng.integers(0, NCELLS - n + 1))
        return tuple(range(start, start + n))
    cells = []
    for section in split_sections(NCELLS, n):
        cells.append(section[int(rng.integers(0, len(section)))])
    return tuple(cells)


def build_schedules(participants) -> dict[str, np.ndarray]:
    out = {}
    for design in DESIGNS:
        _, n = parse_design(design)
        arr = np.empty((len(participants), NDRAWS, n), dtype=np.uint8)
        for i, pid in enumerate(participants):
            for d in range(NDRAWS):
                arr[i, d] = draw_cells(str(pid), design, d)
        out[design] = arr
    return out


# ---------------------------------------------------------------- markers
def ratio_of_sums(powers: np.ndarray, cells: np.ndarray) -> np.ndarray:
    """powers: N x 42 x (alpha,total); cells: N x ... x n. Sum alpha / sum total."""
    powers = np.asarray(powers, dtype=float)
    cells = np.asarray(cells)
    rows = np.arange(len(powers)).reshape((-1,) + (1,) * (cells.ndim - 1))
    return powers[..., 0][rows, cells].sum(axis=-1) / powers[..., 1][rows, cells].sum(axis=-1)


def masked_ratio(powers: np.ndarray, mask: np.ndarray) -> np.ndarray:
    powers = np.asarray(powers, dtype=float)
    mask = np.asarray(mask, dtype=bool)
    return (powers[..., 0] * mask).sum(axis=1) / (powers[..., 1] * mask).sum(axis=1)


# ---------------------------------------------------------------- statistics
def spearman_columns(x: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Spearman (average ranks) of each column of x (N x m) with ref (N,)."""
    rx = rankdata(x, method="average", axis=0)
    ry = rankdata(ref, method="average")
    rx = rx - rx.mean(axis=0, keepdims=True)
    ry = ry - ry.mean()
    with np.errstate(divide="ignore", invalid="ignore"):
        rho = (rx * ry[:, None]).sum(axis=0) / np.sqrt((rx**2).sum(axis=0) * (ry**2).sum())
    return rho


def checked_fisher(rho: np.ndarray, label: str) -> np.ndarray:
    rho = np.asarray(rho, dtype=float)
    if not np.isfinite(rho).all() or np.any(np.abs(rho) >= 1):
        raise AnalysisStop(f"{label}: undefined correlation or infinite Fisher transform")
    return np.arctanh(rho)


def bootstrap_indices(n: int, label: str = BOOTSTRAP_LABEL, resamples: int = NBOOT) -> np.ndarray:
    rng = np.random.default_rng(seed64(BOOTSTRAP_NAMESPACE.format(label=label)))
    return rng.integers(0, n, size=(resamples, n), dtype=np.int64)


def index_sha256(indices: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(indices, dtype="<i8").tobytes(order="C")).hexdigest()


def _prepare(values: np.ndarray):
    """Sort order and tie-group bounds of each column (N x m)."""
    order = np.argsort(values, axis=0, kind="stable")
    ordered = np.take_along_axis(values, order, axis=0)
    n, m = values.shape
    index = np.broadcast_to(np.arange(n)[:, None], (n, m))
    first = np.r_[np.ones((1, m), bool), ordered[1:] != ordered[:-1]]
    last = np.r_[ordered[:-1] != ordered[1:], np.ones((1, m), bool)]
    start = np.maximum.accumulate(np.where(first, index, 0), axis=0)
    end = np.minimum.accumulate(np.where(last, index, n - 1)[::-1], axis=0)[::-1]
    inverse = np.argsort(order, axis=0)
    return order, inverse, start, end


def weighted_midranks(prepared, counts: np.ndarray) -> np.ndarray:
    """Average ranks of each original participant inside resampled data.

    counts: B x N multiplicities. Returns B x N x m. A participant drawn c times
    takes the average rank of its c copies; ties in the original values are
    pooled across their whole tie group.
    """
    order, inverse, start, end = prepared
    w = counts[:, order]  # B x N x m, in sorted order
    cumulative = np.cumsum(w, axis=1)
    hi = np.take_along_axis(cumulative, np.broadcast_to(end[None], w.shape), axis=1)
    lo = np.take_along_axis(cumulative - w, np.broadcast_to(start[None], w.shape), axis=1)
    ranks = (hi + lo + 1.0) * 0.5
    return np.take_along_axis(ranks, np.broadcast_to(inverse[None], w.shape), axis=1)


def bootstrap_draw_rho(samples: np.ndarray, reference: np.ndarray, indices: np.ndarray, batch: int = 8):
    """samples: N x m; returns B x m Spearman correlations recomputed per resample."""
    n, m = samples.shape
    left = _prepare(samples)
    right = _prepare(reference[:, None])
    center = (n + 1) / 2.0
    out = np.empty((len(indices), m))
    for s in range(0, len(indices), batch):
        e = min(s + batch, len(indices))
        counts = np.stack([np.bincount(i, minlength=n) for i in indices[s:e]]).astype(float)
        x = weighted_midranks(left, counts) - center
        y = weighted_midranks(right, counts)[..., 0] - center
        w = counts
        num = np.einsum("bn,bnm,bn->bm", w, x, y)
        den = np.sqrt(np.einsum("bn,bnm->bm", w, x * x) * np.einsum("bn,bn->b", w, y * y)[:, None])
        out[s:e] = num / den
    return out


def percentile(values: np.ndarray, axis: int = 0):
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).all():
        raise AnalysisStop("nonfinite bootstrap value")
    return np.percentile(values, [2.5, 97.5], axis=axis, method="linear")


# ---------------------------------------------------------------- data
def load_cohort(archive: Path) -> dict:
    with np.load(archive / PRE_FILE) as z:
        pre_ids, pre_power, pre_schedule = z["participant_ids"], z["cell_band_powers"], z["schedule"]
    with np.load(archive / POST_FILE) as z:
        post = {k: z[k] for k in z.files}
    with np.load(archive / SCIENTIFIC_FILE) as z:
        sens_ids, sens_schedule, sens_markers = (
            z["sensitivity_participant_ids"], z["sensitivity_schedule"], z["sensitivity_markers_C_F10"])
    checks = {}
    if not np.array_equal(pre_ids, sens_ids):
        raise AnalysisStop("pre-battery participant list differs from the complete-support list")
    if not np.array_equal(pre_schedule, sens_schedule[:, [0, 2]]):
        raise AnalysisStop("pre-battery schedule differs from the stored complete-support C/F10 schedule")
    checks["pre_marker_max_abs_diff"] = float(np.abs(ratio_of_sums(pre_power, pre_schedule) - sens_markers).max())
    checks["post_marker_max_abs_diff"] = float(np.abs(ratio_of_sums(post["power"], post["schedule"]) - post["markers"]).max())
    for key in ("pre_marker_max_abs_diff", "post_marker_max_abs_diff"):
        if not checks[key] <= TOLERANCE:
            raise AnalysisStop(f"{key} = {checks[key]} exceeds {TOLERANCE}")
    common = np.intersect1d(pre_ids, post["participant_ids"])
    if len(common) != EXPECTED_N:
        raise AnalysisStop(f"cohort N = {len(common)}, expected {EXPECTED_N}")
    ip = np.searchsorted(pre_ids, common)
    iq = np.searchsorted(post["participant_ids"], common)
    if not (np.array_equal(pre_ids[ip], common) and np.array_equal(post["participant_ids"][iq], common)):
        raise AnalysisStop("identifier alignment failed")
    pre = pre_power[ip]
    post_power, post_mask = post["power"][iq], post["mask"][iq]
    if not (np.isfinite(pre).all() and (pre[..., 0] > 0).all() and (pre[..., 1] > pre[..., 0]).all()):
        raise AnalysisStop("invalid pre-battery cell powers")
    used = post_power[post_mask]
    if not (np.isfinite(used).all() and (used[:, 0] > 0).all() and (used[:, 1] > used[:, 0]).all()):
        raise AnalysisStop("invalid post-battery cell powers")
    checks.update(n_pre=int(len(pre_ids)), n_post=int(len(post["participant_ids"])), n_common=int(len(common)),
                  post_eligible_cells_min=int(post_mask.sum(1).min()),
                  post_eligible_cells_max=int(post_mask.sum(1).max()),
                  post_participants_with_all_42_cells=int((post_mask.sum(1) == NCELLS).sum()))
    return dict(participants=common, pre=pre, post_power=post_power, post_mask=post_mask, checks=checks)


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


# ---------------------------------------------------------------- analysis
def analyse(cohort: dict, indices: np.ndarray | None = None, log=print) -> dict:
    pids = cohort["participants"]
    n = len(pids)
    reference = masked_ratio(cohort["post_power"], cohort["post_mask"])
    schedules = build_schedules(pids)
    values = np.stack([ratio_of_sums(cohort["pre"], schedules[d]) for d in DESIGNS], axis=1)  # N x 6 x 200
    if not (np.isfinite(values).all() and np.isfinite(reference).all()):
        raise AnalysisStop("nonfinite marker")
    draw_rho = spearman_columns(values.reshape(n, -1), reference).reshape(len(DESIGNS), NDRAWS)
    draw_z = checked_fisher(draw_rho, "draw correlations")
    if indices is None:
        indices = bootstrap_indices(n)
    log("bootstrap start")
    boot_rho = bootstrap_draw_rho(values.reshape(n, -1), reference, indices).reshape(len(indices), len(DESIGNS), NDRAWS)
    boot_z = checked_fisher(boot_rho, "bootstrap correlations")
    log("bootstrap done")
    return dict(participants=pids, reference=reference, schedules=schedules, values=values,
                draw_rho=draw_rho, draw_z=draw_z, bootstrap_mean_z=boot_z.mean(axis=2),
                bootstrap_mean_rho=boot_rho.mean(axis=2), indices=indices)


def summarise(res: dict) -> dict:
    z, rho = res["draw_z"], res["draw_rho"]
    mean_z, mean_rho = z.mean(axis=1), rho.mean(axis=1)
    mcse = z.std(axis=1, ddof=1) / np.sqrt(NDRAWS)
    bz, br = res["bootstrap_mean_z"], res["bootstrap_mean_rho"]
    qz, qr = percentile(bz), percentile(br)
    designs = []
    for j, d in enumerate(DESIGNS):
        kind, n = parse_design(d)
        designs.append(dict(design=d, placement=kind, cells=n, duration_s=4 * n,
                            mean_z=float(mean_z[j]), rho_tilde=float(np.tanh(mean_z[j])),
                            rho_tilde_ci95=[float(np.tanh(qz[0, j])), float(np.tanh(qz[1, j]))],
                            mean_z_ci95=[float(qz[0, j]), float(qz[1, j])],
                            mean_raw_rho=float(mean_rho[j]), mean_raw_rho_ci95=[float(qr[0, j]), float(qr[1, j])],
                            mcse_mean_z=float(mcse[j]),
                            draw_rho_min=float(rho[j].min()), draw_rho_max=float(rho[j].max())))
    pos = {d: j for j, d in enumerate(DESIGNS)}
    contrasts = []
    for family, a, b in CONTRASTS:
        i, k = pos[a], pos[b]
        dz = percentile(bz[:, i] - bz[:, k])
        dr = percentile(br[:, i] - br[:, k])
        contrasts.append(dict(family=family, contrast=f"{a} - {b}",
                              delta_z=float(mean_z[i] - mean_z[k]), delta_z_ci95=[float(dz[0]), float(dz[1])],
                              raw_difference=float(mean_rho[i] - mean_rho[k]),
                              raw_difference_ci95=[float(dr[0]), float(dr[1])],
                              difference_of_rho_tilde=float(np.tanh(mean_z[i]) - np.tanh(mean_z[k])),
                              mcse_delta_z=float(np.sqrt(mcse[i] ** 2 + mcse[k] ** 2))))
    return dict(designs=designs, contrasts=contrasts)


def schedule_hashes(schedules: dict) -> dict:
    return {d: hashlib.sha256(np.ascontiguousarray(a, dtype=np.uint8).tobytes(order="C")).hexdigest()
            for d, a in schedules.items()}


def write_outputs(res: dict, summary: dict, cohort: dict, archive: Path, out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    draws_file = out / "duration_geometry_draws.npz"
    np.savez_compressed(draws_file, participant_ids=res["participants"], reference=res["reference"],
                        design_labels=np.array(DESIGNS), sample_values=res["values"],
                        **{f"cells_{d}": res["schedules"][d] for d in DESIGNS})
    inference_file = out / "duration_geometry_inference.npz"
    np.savez_compressed(inference_file, design_labels=np.array(DESIGNS), draw_rho=res["draw_rho"],
                        draw_z=res["draw_z"], bootstrap_mean_z=res["bootstrap_mean_z"],
                        bootstrap_mean_rho=res["bootstrap_mean_rho"])
    import scipy
    doc = dict(
        analysis="duration_by_geometry",
        status="exploratory",
        multiplicity_correction="none",
        interpretation_note=("The outcome is agreement with a measurement from a separate post-battery recording. "
                             "It is not a biological true level and includes any change in state after the battery. "
                             "Interval overlap is not used to claim that two designs are equal."),
        n=int(len(res["participants"])),
        draws=NDRAWS, bootstrap_resamples=int(len(res["indices"])),
        draw_seed_namespace=DRAW_NAMESPACE,
        bootstrap_seed_string=BOOTSTRAP_NAMESPACE.format(label=BOOTSTRAP_LABEL),
        bootstrap_seed=seed64(BOOTSTRAP_NAMESPACE.format(label=BOOTSTRAP_LABEL)),
        bootstrap_index_sha256=index_sha256(res["indices"]),
        interval="2.5th and 97.5th percentiles, NumPy method linear",
        mcse_rule="design: SD(ddof=1) of 200 draw z / sqrt(200); contrast: sqrt(sum of squared design MCSE), independent seed streams",
        participant_ids_sha256=hashlib.sha256("\n".join(map(str, res["participants"])).encode()).hexdigest(),
        schedule_sha256_uint8_c_order=schedule_hashes(res["schedules"]),
        input_sha256={PRE_FILE: file_sha256(archive / PRE_FILE), POST_FILE: file_sha256(archive / POST_FILE),
                      SCIENTIFIC_FILE: file_sha256(archive / SCIENTIFIC_FILE)},
        compatibility_checks=cohort["checks"],
        designs=summary["designs"], contrasts=summary["contrasts"],
        software=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__),
    )
    (out / "results_summary.json").write_text(json.dumps(doc, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    return doc


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--archive", type=Path, default=HERE.parent, help="archive root (default: parent folder)")
    parser.add_argument("--output", type=Path, required=True, help="output folder")
    args = parser.parse_args(argv)
    archive, out = args.archive.resolve(), args.output.resolve()
    if (out / "results_summary.json").exists():
        raise SystemExit(f"{out} already holds results; choose an empty output folder")
    log = lambda s: print(datetime.datetime.now(datetime.timezone.utc).isoformat(), s, flush=True)
    cohort = load_cohort(archive)
    log(f"compatibility checks passed: {cohort['checks']}")
    res = analyse(cohort, log=log)
    doc = write_outputs(res, summarise(res), cohort, archive, out)
    log(f"wrote {out} (N = {doc['n']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
