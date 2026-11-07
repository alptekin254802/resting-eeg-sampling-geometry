"""Exploratory scalp topography of the sampling-geometry contrast for relative alpha.

Stages (run in order from any directory; paths are relative to this file):
  python run_topography.py lock
  python run_topography.py extract pre  [--budget SECONDS] [--workers N]
  python run_topography.py extract post [--budget SECONDS] [--workers N]
  python run_topography.py assemble
  python run_topography.py validate
  python run_topography.py infer [--workers N]
  python run_topography.py report

The extract stage reads raw EDF files and can be resumed; every other stage
reads only saved arrays. Requires numpy, scipy and mne (extract only).

Inputs are found in either the software-archive layout (this folder at the
archive root) or the original analysis workspace. Raw ds005385 EDF folders
can be set with TOPOGRAPHY_RAW_PRE and TOPOGRAPHY_RAW_POST; each must contain
the sub-XXX/ses-1/eeg/ paths of the OpenNeuro dataset (version 1.0.3).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

sys.dont_write_bytecode = True
for _key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_key, "1")
import numpy as np

OUT = Path(__file__).resolve().parent
BASE = OUT.parent
sys.path.insert(0, str(BASE / "scripts"))
CACHE = OUT / "cache"
SCHEDULE_SHA256 = "95c0da1f105ff9f92a64010d120b8ebecc760c75540b448f209dc391e91649c1"
CENSUS_SHA256 = "90f425ba446daac8db3daa73e23035d4a6e9f6efc3a232671efad16cdfd04c82"
if (BASE / "inputs" / "preanalysis").is_dir():
    # Software archive layout (this folder sits at the archive root as topography/).
    SCHEDULE_FILE = BASE / "inputs/preanalysis/structural/ds005385_sampling_schedule.npz"
    CENSUS_FILE = BASE / "inputs/preanalysis/DS005385_PARTICIPANT_CENSUS.csv"
    PRIMARY_RESULTS = BASE / "results/preregistered/scientific_results.npz"
    PRE_ROI_CELLS = BASE / "results/exploratory_2026-09-20/cell_band_powers.npz"
    POST_PREPARED = BASE / "additional/results/post_prepared.npz"
    POST_METADATA = BASE / "additional/metadata/post_metadata.json"
    CROSS_INFERENCE = BASE / "additional/results/pre_post_cross_inference.npz"
    RAW_PRE = Path(os.environ.get("TOPOGRAPHY_RAW_PRE", BASE / "raw/ds005385"))
    RAW_POST = Path(os.environ.get("TOPOGRAPHY_RAW_POST", BASE / "raw/ds005385"))
else:
    # Analysis workspace layout used for the original run.
    SCHEDULE_FILE = BASE / "osf_upload_package/structural/ds005385_sampling_schedule.npz"
    CENSUS_FILE = BASE / "osf_upload_package/DS005385_PARTICIPANT_CENSUS.csv"
    PRIMARY_RESULTS = BASE / "public_release/eeg-temporal-sampling/results/preregistered/scientific_results.npz"
    PRE_ROI_CELLS = BASE / "public_release/eeg-temporal-sampling/results/exploratory_2026-09-20/cell_band_powers.npz"
    POST_PREPARED = BASE / "followup_2026-11-01/results/post_prepared.npz"
    POST_METADATA = BASE / "followup_2026-11-01/metadata/post_metadata.json"
    CROSS_INFERENCE = BASE / "followup_2026-11-01/results/pre_post_cross_inference.npz"
    RAW_PRE = Path(os.environ.get("TOPOGRAPHY_RAW_PRE", BASE / "raw_cache/production/ds005385"))
    RAW_POST = Path(os.environ.get("TOPOGRAPHY_RAW_POST", BASE / "followup_2026-11-01/raw_post"))

EEG_CHANNELS = (
    "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "FC5", "FC1", "FC2", "FC6", "T7", "C3", "Cz", "C4", "T8",
    "TP9", "CP5", "CP1", "CP2", "CP6", "TP10", "P7", "P3", "Pz", "P4", "P8", "PO9", "O1", "Oz", "O2", "PO10",
    "AF7", "AF3", "AF4", "AF8", "F5", "F1", "F2", "F6", "FT9", "FT7", "FC3", "FC4", "FT8", "FT10", "C5", "C1",
    "C2", "C6", "TP7", "CP3", "CPz", "CP4", "TP8", "P5", "P1", "P2", "P6", "PO7", "PO3", "POz", "PO4", "PO8",
)
REFERENCE = ("TP9", "TP10")
CHANNELS = tuple(c for c in EEG_CHANNELS if c not in REFERENCE)
ROI = ("O1", "Oz", "O2")
GEOMETRIES = ("C", "F10")
PRIMARY_REGIME_INDEX = {"C": 0, "F5": 1, "F10": 2}
COMPARISONS = ("within", "between")

PLAN = {
    "status": "Exploratory. The local timestamp and SHA-256 values in PLAN_RECORD.json are an audit record. No multiplicity correction is applied; channel-level intervals are descriptive.",
    "question": "Is the difference in rank reproducibility between distributed 10x4-s and contiguous 40-s sampling of relative alpha limited to the posterior channels of the main analysis (O1, Oz, O2), or does it extend across the scalp? Examined within the pre-battery recording (the main A/B comparison) and between the pre- and post-battery recordings (the comparison in which the timing of paired samples within a recording plays no role).",
    "scope": "One feature (relative alpha, 8-13 Hz power over 1-30 Hz power), one contrast (distributed 10x4 s minus contiguous 40 s) and one cohort (ds005385). The 5x8-s geometry, other features and the ds004148 replication cohort are not part of this analysis.",
    "channels": "The 62 EEG channels of ds005385 other than TP9 and TP10, each referenced to the mean of TP9 and TP10 as in the main analysis. The 64 EEG labels and their order are checked in every file; a missing or reordered channel stops the analysis.",
    "data": "Pre-battery: the 531 primary EDF files in raw_cache/production/ds005385, verified against the registered byte counts and SHA-256 digests. Post-battery: the 524 post-battery EDF files of post_prepared.npz in followup_2026-11-01/raw_post, verified against the byte counts and SHA-256 digests in post_metadata.json.",
    "processing": "Support [8,176) s and 42 non-overlapping 4-s cells. Per channel and cell, the registered elementary periodogram (linear detrend, periodic Hann, one-sided density, native sampling rate, 0.25-Hz grid). Band powers by trapezoidal integration over 8-13 Hz and 1-30 Hz, endpoints included. No filtering, resampling, ICA, channel interpolation or additional cleaning. The existing cell-eligibility masks and A/B schedules are reused unchanged; they were defined from the posterior channels and the reference, so other channels are not screened separately.",
    "marker": "For each channel, sample and draw: mean 8-13 Hz power over the sample's ten cells divided by mean 1-30 Hz power over the same cells.",
    "comparisons": "within: 531 participants; contiguous (C) and distributed 10x4-s (F10) original schedules; sample A against sample B within the pre-battery recording. between: 524 participants; pre-battery sample A (original schedule) against post-battery sample A (post-battery schedule of the 2026-11-01 follow-up), as in the published between-recording comparison.",
    "statistics": "Per channel, comparison and geometry: Spearman midrank correlation across participants in each of 200 draws, Fisher z without clipping, mean over draws, rho_tilde = tanh(mean z). Delta_z = mean z(F10) - mean z(C). MCSE = sample SD of the draw-level contrast / sqrt(200).",
    "bootstrap": "The existing 5000 x N participant index matrices, shared by all channels: within = generate_bootstrap_indices(531, 'primary_ds005385'), the primary-analysis matrix; between = bootstrap_indices stored in followup_2026-11-01/results/pre_post_cross_inference.npz. Exact weighted-midrank Spearman in every resample. Unadjusted 95% percentile intervals (NumPy method linear).",
    "summaries": "All reported: (1) per-channel Delta_z scalp maps for both comparisons on one colour scale centred at zero, with O1, Oz and O2 marked; (2) the mean of the 62 channel Delta_z values with a bootstrap interval (mean over channels within each resample); (3) the number of channels with Delta_z > 0 and the numbers whose interval lies wholly above or wholly below zero; (4) the smallest and largest channel Delta_z and their channels; (5) across channels, the Spearman correlation between contiguous rho_tilde and Delta_z, descriptive only; (6) rho_tilde maps for C and F10 in both comparisons and a per-channel CSV.",
    "validation": "Before any channel-level outcome: (a) per participant and cell, the mean of the O1, Oz and O2 band powers reproduces the stored posterior cell band powers (pre: exploratory_2026-09-20/cell_band_powers.npz for its 516 participants; post: post_prepared.npz power) within relative 1e-10; (b) posterior markers from the stored schedules reproduce primary_markers (531) and the post_prepared markers (524) within absolute 1e-12; (c) the inference code reproduces the stored primary draw-level Fisher z for C and F10 within 1e-12 and primary_bootstrap_delta_z within 1e-10, and the stored between-recording cross_A mean z and bootstrap z for C and F10 within 1e-10. Any failure stops the analysis.",
    "stop_rules": "A missing or reordered channel, a hash or byte-count mismatch, a sampling-rate mismatch, a nonfinite sample, a nonfinite or nonpositive band power in an eligible cell, alpha power not below 1-30 Hz power, an undefined correlation or an infinite Fisher transform stops the analysis. No clipping, dropping, repair or replacement.",
    "interpretation": "Descriptive. No channel is called significant and no region is selected after seeing the maps. The within-recording map carries the same timing confound as the main within-recording comparison; the between-recording map does not. Channels outside the posterior region were not separately screened for artifacts; this applies equally to both geometries but may lower reproducibility at those channels.",
    "reporting": "All results are reported whatever their direction: one main-text figure with the two Delta_z maps and one Results paragraph; the rho_tilde maps and the summaries in the Online Resource; per-channel values in the archive.",
    "outputs": "All new files go only to followup_topography_2026-11. Original results, schedules, figures and manuscript files are not changed by this script.",
}


class StopAnalysis(RuntimeError):
    pass


def stamp():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def array_sha(a, dtype=None):
    return hashlib.sha256(np.ascontiguousarray(a, dtype=dtype).tobytes()).hexdigest()


def write_json(path, value):
    def convert(v):
        if isinstance(v, np.ndarray):
            return v.tolist()
        if isinstance(v, np.generic):
            return v.item()
        raise TypeError(type(v))
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False, default=convert) + "\n",
                          encoding="utf-8")


def log(message):
    line = f"{stamp()} {message}"
    print(line, flush=True)
    with (OUT / "run_log.txt").open("a", encoding="utf-8") as f:
        f.write(line + "\n")


# ----------------------------------------------------------------------------- plan

def plan_md():
    return "# Scalp topography of the sampling-geometry contrast: analysis plan\n\n" + "\n\n".join(
        f"## {key}\n\n{value}" for key, value in PLAN.items()) + "\n"


def lock_plan():
    record = OUT / "PLAN_RECORD.json"
    if not record.exists():
        if (OUT / "ANALYSIS_PLAN.md").exists() or (OUT / "ANALYSIS_PLAN.json").exists():
            raise StopAnalysis("Incomplete plan exists; will not overwrite")
        write_json(OUT / "ANALYSIS_PLAN.json", PLAN)
        (OUT / "ANALYSIS_PLAN.md").write_text(plan_md(), encoding="utf-8")
        write_json(record, {
            "recorded_at_utc": stamp(),
            "channel_level_outcomes_calculated": False,
            "files": {name: {"sha256": sha(OUT / name)} for name in ("ANALYSIS_PLAN.md", "ANALYSIS_PLAN.json")},
        })
        log("plan recorded")
    saved = json.loads(record.read_text(encoding="utf-8"))
    if json.loads((OUT / "ANALYSIS_PLAN.json").read_text(encoding="utf-8")) != PLAN:
        raise StopAnalysis("Plan content changed")
    if (OUT / "ANALYSIS_PLAN.md").read_text(encoding="utf-8") != plan_md():
        raise StopAnalysis("MD and JSON plans differ")
    for name, data in saved["files"].items():
        if sha(OUT / name) != data["sha256"]:
            raise StopAnalysis("Plan hash changed")
    return saved


# ----------------------------------------------------------------------------- inputs

def load_schedule():
    if sha(SCHEDULE_FILE) != SCHEDULE_SHA256:
        raise StopAnalysis("registered schedule hash mismatch")
    with np.load(SCHEDULE_FILE, allow_pickle=False) as z:
        return z["participant_ids"].astype(str), z["eligible_masks"], z["schedule"]


def pre_rows():
    if sha(CENSUS_FILE) != CENSUS_SHA256:
        raise StopAnalysis("registered census hash mismatch")
    with CENSUS_FILE.open(encoding="utf-8", newline="") as f:
        census = [r for r in csv.DictReader(f) if r["final_eligible"].strip().lower() == "true"]
    ids, masks, _ = load_schedule()
    if [r["participant_id"] for r in census] != ids.tolist():
        raise StopAnalysis("census order differs from registered schedule")
    return [dict(id=r["participant_id"], path=RAW_PRE / r["target_recording"],
                 size=int(r["annex_bytes"]), digest=r["annex_digest"], fs=float(r["edf_eeg_sampling_hz"]),
                 mask=masks[i]) for i, r in enumerate(census)]


def post_rows():
    meta = {r["participant_id"]: r for r in json.loads(POST_METADATA.read_text(encoding="utf-8"))["rows"]}
    with np.load(POST_PREPARED, allow_pickle=False) as z:
        ids = z["participant_ids"].astype(str)
        masks = z["mask"]
    return [dict(id=p, path=RAW_POST / meta[p]["path"], size=int(meta[p]["expected_bytes"]),
                 digest=meta[p]["expected_sha256"], fs=None, mask=masks[i]) for i, p in enumerate(ids)]


# ----------------------------------------------------------------------------- extraction

def load_support(path):
    import mne
    raw = mne.io.read_raw_edf(path, preload=False, verbose="ERROR")
    eeg = tuple(c for c in raw.ch_names if c != "Status")
    # PLAN_DEVIATION.md: the same 64 labels in another file order are accepted;
    # channels are always selected by label below.
    if len(eeg) != 64 or set(eeg) != set(EEG_CHANNELS):
        raise StopAnalysis(f"channel set differs in {path.name}")
    fs = float(raw.info["sfreq"])
    if raw.n_times < int(round(180.0 * fs)):
        raise StopAnalysis(f"recording shorter than 180 s: {path.name}")
    data = raw.get_data(picks=list(EEG_CHANNELS), start=int(round(8.0 * fs)), stop=int(round(176.0 * fs)))
    if data.shape != (64, int(round(168.0 * fs))) or not np.isfinite(data).all():
        raise StopAnalysis(f"support shape or finiteness failed: {path.name}")
    return data, fs


def channel_band_powers(data, fs):
    """Return cells x channels x (alpha, total) band powers in V^2."""
    from preregistered_analysis import elementary_periodogram
    n = int(round(4.0 * fs))
    index = {c: i for i, c in enumerate(EEG_CHANNELS)}
    reference = (data[index["TP9"]] + data[index["TP10"]]) / 2.0
    signals = data[[index[c] for c in CHANNELS]] - reference
    frequency = np.fft.rfftfreq(n, 1.0 / fs)
    if not np.isclose(frequency[1] - frequency[0], 0.25):
        raise StopAnalysis("0.25-Hz grid not obtained")
    alpha = (frequency >= 8) & (frequency <= 13)
    total = (frequency >= 1) & (frequency <= 30)
    out = np.empty((42, len(CHANNELS), 2))
    for j in range(len(CHANNELS)):
        psd = elementary_periodogram(signals[j].reshape(42, n), fs)
        out[:, j, 0] = np.trapezoid(psd[:, alpha], frequency[alpha], axis=-1)
        out[:, j, 1] = np.trapezoid(psd[:, total], frequency[total], axis=-1)
    return out


def extract_one(task):
    cohort, row = task
    path = row["path"]
    if path.stat().st_size != row["size"] or sha(path) != row["digest"]:
        raise StopAnalysis(f"byte count or SHA-256 mismatch: {row['id']}")
    data, fs = load_support(path)
    if row["fs"] is not None and not np.isclose(fs, row["fs"]):
        raise StopAnalysis(f"sampling-rate mismatch: {row['id']}")
    powers = channel_band_powers(data, fs)
    used = powers[row["mask"]]
    if not np.isfinite(used).all() or np.any(used[..., 0] <= 0) or np.any(used[..., 1] <= used[..., 0]):
        raise StopAnalysis(f"invalid band power in an eligible cell: {row['id']}")
    target = CACHE / cohort / f"{row['id']}.npz"
    tmp = target.with_suffix(".tmp.npz")
    np.savez(tmp, band_powers=powers, sampling_rate=fs, raw_sha256=row["digest"])
    os.replace(tmp, target)
    return row["id"]


def extract(cohort, budget, workers):
    lock_plan()
    rows = pre_rows() if cohort == "pre" else post_rows()
    (CACHE / cohort).mkdir(parents=True, exist_ok=True)
    pending = [r for r in rows if not (CACHE / cohort / f"{r['id']}.npz").exists()]
    log(f"extract {cohort}: {len(rows) - len(pending)}/{len(rows)} already done")
    start = time.time()
    done = 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = []
        queue = list(pending)
        while queue or futures:
            while queue and len(futures) < 2 * workers and time.time() - start < budget:
                futures.append(pool.submit(extract_one, (cohort, queue.pop(0))))
            if not futures:
                break
            futures[0].result()
            futures.pop(0)
            done += 1
            if time.time() - start >= budget:
                queue = []
    remaining = sum(not (CACHE / cohort / f"{r['id']}.npz").exists() for r in rows)
    log(f"extract {cohort}: processed {done} in this run, {remaining} remaining")
    return remaining


def assemble():
    lock_plan()
    for cohort, rows in (("pre", pre_rows()), ("post", post_rows())):
        arrays, rates = [], []
        for r in rows:
            with np.load(CACHE / cohort / f"{r['id']}.npz", allow_pickle=False) as z:
                if str(z["raw_sha256"]) != r["digest"]:
                    raise StopAnalysis(f"cache digest mismatch: {r['id']}")
                arrays.append(z["band_powers"])
                rates.append(float(z["sampling_rate"]))
        powers = np.stack(arrays)
        np.savez_compressed(OUT / f"{cohort}_channel_band_powers.npz", participant_ids=np.array([r["id"] for r in rows]),
                            channels=np.array(CHANNELS), band_powers=powers, sampling_rate=np.array(rates),
                            masks=np.array([r["mask"] for r in rows]))
        log(f"assembled {cohort}: {powers.shape}")


# ----------------------------------------------------------------------------- statistics

def markers(powers, schedule):
    """powers: N x 42 x 2 (alpha, total); schedule: N x ... x 10 -> N x ...."""
    rows = np.arange(len(powers)).reshape((-1,) + (1,) * (schedule.ndim - 1))
    selected = powers[rows, schedule.astype(np.int64)].mean(axis=-2)
    return selected[..., 0] / selected[..., 1]


def point(mark):
    """mark: N x conditions x draws x arms -> mean Fisher z per condition, draw rho."""
    from preregistered_analysis import spearman_midrank_columns
    n, c, d, _ = mark.shape
    rho = spearman_midrank_columns(mark[..., 0].reshape(n, c * d), mark[..., 1].reshape(n, c * d)).reshape(c, d)
    if not np.all(np.abs(rho) < 1):
        raise StopAnalysis("undefined or infinite Fisher transform")
    return np.arctanh(rho).mean(axis=1), rho


def prepare_ranks(values):
    order = np.argsort(values, axis=0, kind="stable")
    ordered = np.take_along_axis(values, order, axis=0)
    n, m = values.shape
    index = np.broadcast_to(np.arange(n)[:, None], (n, m))
    first = np.r_[np.ones((1, m), bool), ordered[1:] != ordered[:-1]]
    last = np.r_[ordered[:-1] != ordered[1:], np.ones((1, m), bool)]
    start = np.maximum.accumulate(np.where(first, index, 0), axis=0)
    end = np.minimum.accumulate(np.where(last, index, n - 1)[::-1], axis=0)[::-1]
    inverse = np.argsort(order, axis=0)
    return order, inverse, start, end, bool(first.all())


def weighted_ranks(prepared, counts):
    order, inverse, start, end, unique = prepared
    w = counts[:, order]
    cumulative = np.cumsum(w, axis=1)
    if unique:
        ranks = cumulative - 0.5 * w + 0.5
    else:
        hi = np.take_along_axis(cumulative, end[None], axis=1)
        low = np.take_along_axis(cumulative - w, start[None], axis=1)
        ranks = (hi + low + 1.0) * 0.5
    return np.take_along_axis(ranks, inverse[None], axis=1)


def bootstrap_z_unique(mark, indices, batch=24):
    """Integer-exact version of bootstrap_z when every column has distinct values.

    A participant drawn w times gets twice-midrank 2*cumsum - w + 1 in each arm;
    sum(w * (x - y)^2) is accumulated in int64, so it equals the float formula exactly.
    """
    n, c, d, _ = mark.shape
    m = c * d
    left = mark[..., 0].reshape(n, m)
    right = mark[..., 1].reshape(n, m)
    order_left = np.argsort(left, axis=0, kind="stable")
    order_right = np.argsort(right, axis=0, kind="stable")
    position_right = np.argsort(order_right, axis=0)
    gather = (np.take_along_axis(position_right, order_left, axis=0) * m + np.arange(m)).ravel()
    output = np.empty((len(indices), c))
    for start in range(0, len(indices), batch):
        stop = min(start + batch, len(indices))
        k = stop - start
        counts = np.array([np.bincount(i, minlength=n) for i in indices[start:stop]], dtype=np.int64)
        w_left = counts[:, order_left]
        rank_left = 2 * np.cumsum(w_left, axis=1) - w_left + 1
        w_right = counts[:, order_right]
        rank_right = 2 * np.cumsum(w_right, axis=1) - w_right + 1
        diff = rank_left - rank_right.reshape(k, -1)[:, gather].reshape(k, n, m)
        squares = np.einsum("knm,knm->km", w_left, diff * diff)
        variance12 = n * (n * n - 1) - np.sum(counts ** 3 - counts, axis=1)
        rho = 1 - 6 * (squares / 4.0) / variance12[:, None]
        if not np.isfinite(rho).all() or np.any(np.abs(rho) >= 1):
            raise StopAnalysis("undefined bootstrap correlation")
        output[start:stop] = np.arctanh(rho).reshape(k, c, d).mean(axis=2)
    return output


def bootstrap_z(mark, indices, batch=12):
    """Exact participant bootstrap of mean Fisher z using multiplicities and midranks."""
    n, c, d, _ = mark.shape
    left = prepare_ranks(mark[..., 0].reshape(n, c * d))
    right = prepare_ranks(mark[..., 1].reshape(n, c * d))
    if left[-1] and right[-1]:
        return bootstrap_z_unique(mark, indices)
    output = np.empty((len(indices), c))
    center = (n + 1) / 2
    for start in range(0, len(indices), batch):
        stop = min(start + batch, len(indices))
        counts = np.array([np.bincount(i, minlength=n) for i in indices[start:stop]], dtype=float)
        x = weighted_ranks(left, counts)
        y = weighted_ranks(right, counts)
        w = counts[:, :, None]
        if left[-1] and right[-1]:
            variance12 = n * (n * n - 1) - np.sum(counts ** 3 - counts, axis=1)
            rho = 1 - 6 * np.sum(w * (x - y) ** 2, axis=1) / variance12[:, None]
        else:
            x -= center
            y -= center
            rho = np.sum(w * x * y, axis=1) / np.sqrt(np.sum(w * x * x, axis=1) * np.sum(w * y * y, axis=1))
        if not np.isfinite(rho).all() or np.any(np.abs(rho) >= 1):
            raise StopAnalysis("undefined bootstrap correlation")
        output[start:stop] = np.arctanh(rho).reshape(stop - start, c, d).mean(axis=2)
    return output


def interval(x):
    a = np.asarray(x)
    if not np.isfinite(a).all():
        raise StopAnalysis("nonfinite bootstrap values")
    return np.percentile(a, [2.5, 97.5], axis=0, method="linear")


# ----------------------------------------------------------------------------- comparison arrays

def load_powers(cohort):
    with np.load(OUT / f"{cohort}_channel_band_powers.npz", allow_pickle=False) as z:
        if tuple(z["channels"].astype(str)) != CHANNELS:
            raise StopAnalysis("channel order mismatch in assembled array")
        return z["participant_ids"].astype(str), z["band_powers"], z["masks"]


def comparison_markers():
    """Return {comparison: (ids, marks)} with marks N x channels x geometries x draws x arms."""
    pre_ids, pre_powers, _ = load_powers("pre")
    post_ids, post_powers, _ = load_powers("post")
    sched_ids, _, schedule = load_schedule()
    if not np.array_equal(pre_ids, sched_ids):
        raise StopAnalysis("pre participant order differs from registered schedule")
    with np.load(POST_PREPARED, allow_pickle=False) as z:
        if not np.array_equal(z["participant_ids"].astype(str), post_ids):
            raise StopAnalysis("post participant order mismatch")
        post_schedule = z["schedule"]
    g = [PRIMARY_REGIME_INDEX[k] for k in GEOMETRIES]
    within = np.stack([markers(pre_powers[:, :, j], schedule[:, g]) for j in range(len(CHANNELS))], axis=1)
    lookup = {p: i for i, p in enumerate(pre_ids)}
    rows = np.array([lookup[p] for p in post_ids])
    pre_a = within[rows][..., 0]
    post_a = np.stack([markers(post_powers[:, :, j], post_schedule[:, g]) for j in range(len(CHANNELS))], axis=1)[..., 0]
    between = np.stack([pre_a, post_a], axis=-1)
    return {"within": (pre_ids, within), "between": (post_ids, between)}


def bootstrap_indices(comparison):
    if comparison == "within":
        from preregistered_analysis import generate_bootstrap_indices
        return generate_bootstrap_indices(531, "primary_ds005385")
    with np.load(CROSS_INFERENCE, allow_pickle=False) as z:
        return z["bootstrap_indices"]


# ----------------------------------------------------------------------------- validation

def validate():
    lock_plan()
    checks = {"created_at_utc": stamp()}
    roi = [CHANNELS.index(c) for c in ROI]
    pre_ids, pre_powers, _ = load_powers("pre")
    post_ids, post_powers, _ = load_powers("post")
    with np.load(PRE_ROI_CELLS, allow_pickle=False) as z:
        ids516 = z["participant_ids"].astype(str)
        stored = z["cell_band_powers"]
    lookup = {p: i for i, p in enumerate(pre_ids)}
    mine = pre_powers[[lookup[p] for p in ids516]][:, :, roi].mean(axis=2)
    checks["a_pre_roi_cell_power_max_rel"] = float(np.max(np.abs(mine - stored) / np.abs(stored)))
    with np.load(POST_PREPARED, allow_pickle=False) as z:
        stored_post = z["power"]
        stored_post_markers = z["markers"]
        post_schedule = z["schedule"]
    mine_post = post_powers[:, :, roi].mean(axis=2)
    checks["a_post_roi_cell_power_max_rel"] = float(np.max(np.abs(mine_post - stored_post) / np.abs(stored_post)))
    _, _, schedule = load_schedule()
    with np.load(PRIMARY_RESULTS, allow_pickle=False) as z:
        primary_markers = z["primary_markers"]
        primary_fz = z["primary_draw_fisher_z"]
        primary_boot = z["primary_bootstrap_delta_z"]
    pre_roi = pre_powers[:, :, roi].mean(axis=2)
    checks["b_pre_marker_max_abs"] = float(np.max(np.abs(markers(pre_roi, schedule) - primary_markers)))
    checks["b_post_marker_max_abs"] = float(np.max(np.abs(markers(mine_post, post_schedule) - stored_post_markers)))
    g = [PRIMARY_REGIME_INDEX[k] for k in GEOMETRIES]
    within = markers(pre_roi, schedule[:, g])
    z_within, rho_within = point(within)
    checks["c_within_draw_z_max_abs"] = float(np.max(np.abs(np.arctanh(rho_within) - primary_fz[g])))
    boot = bootstrap_z(within, bootstrap_indices("within"))
    checks["c_within_bootstrap_delta_max_abs"] = float(np.max(np.abs((boot[:, 1] - boot[:, 0]) - primary_boot)))
    rows = np.array([lookup[p] for p in post_ids])
    between = np.stack([within[rows][..., 0], markers(mine_post, post_schedule[:, g])[..., 0]], axis=-1)
    z_between, _ = point(between)
    with np.load(CROSS_INFERENCE, allow_pickle=False) as z:
        cross_z = z["mean_z"][[0, 2]]
        cross_boot = z["bootstrap_z"][:, [0, 2]]
    checks["c_between_mean_z_max_abs"] = float(np.max(np.abs(z_between - cross_z)))
    boot_b = bootstrap_z(between, bootstrap_indices("between"))
    checks["c_between_bootstrap_z_max_abs"] = float(np.max(np.abs(boot_b - cross_boot)))
    limits = {"a_pre_roi_cell_power_max_rel": 1e-10, "a_post_roi_cell_power_max_rel": 1e-10,
              "b_pre_marker_max_abs": 1e-12, "b_post_marker_max_abs": 1e-12, "c_within_draw_z_max_abs": 1e-12,
              "c_within_bootstrap_delta_max_abs": 1e-10, "c_between_mean_z_max_abs": 1e-10,
              "c_between_bootstrap_z_max_abs": 1e-10}
    checks["limits"] = limits
    checks["posterior_reference"] = {
        "within": {"rho_C": float(np.tanh(z_within[0])), "rho_F10": float(np.tanh(z_within[1])),
                   "delta_z": float(z_within[1] - z_within[0]), "delta_z_ci95": interval(boot[:, 1] - boot[:, 0])},
        "between": {"rho_C": float(np.tanh(z_between[0])), "rho_F10": float(np.tanh(z_between[1])),
                    "delta_z": float(z_between[1] - z_between[0]), "delta_z_ci95": interval(boot_b[:, 1] - boot_b[:, 0])},
    }
    checks["passed"] = all(checks[k] <= v for k, v in limits.items())
    write_json(OUT / "VALIDATION.json", checks)
    log(f"validation passed={checks['passed']}")
    if not checks["passed"]:
        raise StopAnalysis("validation failed; see VALIDATION.json")


# ----------------------------------------------------------------------------- inference

_WORK = {}


def _init_worker(comparison):
    ids, marks = comparison_markers()[comparison]
    _WORK["marks"] = marks
    _WORK["indices"] = bootstrap_indices(comparison)


def _channel_job(j):
    mark = _WORK["marks"][:, j]
    z, rho = point(mark)
    boot = bootstrap_z(mark, _WORK["indices"])
    return j, z, rho, boot


def infer(workers):
    lock_plan()
    if not json.loads((OUT / "VALIDATION.json").read_text(encoding="utf-8"))["passed"]:
        raise StopAnalysis("validation has not passed")
    record = json.loads((OUT / "PLAN_RECORD.json").read_text(encoding="utf-8"))
    log(f"inference start; plan recorded at {record['recorded_at_utc']}")
    data = comparison_markers()
    payload = {}
    for comparison in COMPARISONS:
        ids, marks = data[comparison]
        indices = bootstrap_indices(comparison)
        if indices.shape != (5000, len(ids)):
            raise StopAnalysis("bootstrap matrix shape mismatch")
        k = len(CHANNELS)
        mean_z = np.empty((k, 2))
        draw_rho = np.empty((k, 2, 200))
        boot = np.empty((5000, k, 2))
        with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker, initargs=(comparison,)) as pool:
            for j, z, rho, b in pool.map(_channel_job, range(k)):
                mean_z[j], draw_rho[j], boot[:, j] = z, rho, b
                log(f"{comparison}: channel {CHANNELS[j]} done")
        payload[f"{comparison}_participant_ids"] = ids
        payload[f"{comparison}_mean_z"] = mean_z
        payload[f"{comparison}_draw_rho"] = draw_rho
        payload[f"{comparison}_bootstrap_z"] = boot
        payload[f"{comparison}_bootstrap_indices_sha256"] = np.array(array_sha(indices, "<i8"))
    np.savez_compressed(OUT / "topography_inference.npz", channels=np.array(CHANNELS),
                        geometries=np.array(GEOMETRIES), **payload)
    log("inference complete")


# ----------------------------------------------------------------------------- report

def report():
    from scipy.stats import spearmanr
    lock_plan()
    validation = json.loads((OUT / "VALIDATION.json").read_text(encoding="utf-8"))
    with np.load(OUT / "topography_inference.npz", allow_pickle=False) as z:
        res = {k: z[k] for k in z.files}
    summary = {"created_at_utc": stamp(), "status": PLAN["status"], "channels": list(CHANNELS), "comparisons": {}}
    rows = []
    for comparison in COMPARISONS:
        mz, rho, boot = res[f"{comparison}_mean_z"], res[f"{comparison}_draw_rho"], res[f"{comparison}_bootstrap_z"]
        delta = mz[:, 1] - mz[:, 0]
        delta_boot = boot[:, :, 1] - boot[:, :, 0]
        draw_delta = np.arctanh(rho[:, 1]) - np.arctanh(rho[:, 0])
        mcse = draw_delta.std(axis=1, ddof=1) / np.sqrt(200)
        ci = interval(delta_boot)
        rho_ci = np.tanh(interval(boot))
        per_channel = []
        for j, c in enumerate(CHANNELS):
            item = {"channel": c, "n": int(len(res[f"{comparison}_participant_ids"])),
                    "rho_C": float(np.tanh(mz[j, 0])), "rho_C_ci95": rho_ci[:, j, 0].tolist(),
                    "rho_F10": float(np.tanh(mz[j, 1])), "rho_F10_ci95": rho_ci[:, j, 1].tolist(),
                    "delta_z": float(delta[j]), "delta_z_ci95": ci[:, j].tolist(), "delta_z_mcse": float(mcse[j])}
            per_channel.append(item)
            rows.append({"comparison": comparison, **{k: (f"{v:.6f}" if isinstance(v, float) else v)
                                                       for k, v in item.items() if not isinstance(v, list)},
                         "rho_C_ci_low": f"{item['rho_C_ci95'][0]:.6f}", "rho_C_ci_high": f"{item['rho_C_ci95'][1]:.6f}",
                         "rho_F10_ci_low": f"{item['rho_F10_ci95'][0]:.6f}", "rho_F10_ci_high": f"{item['rho_F10_ci95'][1]:.6f}",
                         "delta_z_ci_low": f"{ci[0, j]:.6f}", "delta_z_ci_high": f"{ci[1, j]:.6f}"})
        channel_mean = float(delta.mean())
        channel_mean_ci = interval(delta_boot.mean(axis=1)).tolist()
        imin, imax = int(np.argmin(delta)), int(np.argmax(delta))
        roi = [CHANNELS.index(c) for c in ROI]
        summary["comparisons"][comparison] = {
            "n": int(len(res[f"{comparison}_participant_ids"])),
            "channel_mean_delta_z": channel_mean, "channel_mean_delta_z_ci95": channel_mean_ci,
            "channels_delta_z_positive": int((delta > 0).sum()),
            "channels_ci_wholly_above_zero": int((ci[0] > 0).sum()),
            "channels_ci_wholly_below_zero": int((ci[1] < 0).sum()),
            "min_delta_z": {"channel": CHANNELS[imin], "delta_z": float(delta[imin]), "ci95": ci[:, imin].tolist()},
            "max_delta_z": {"channel": CHANNELS[imax], "delta_z": float(delta[imax]), "ci95": ci[:, imax].tolist()},
            "spearman_rhoC_vs_delta_z_across_channels": float(spearmanr(np.tanh(mz[:, 0]), delta).statistic),
            "rho_C_range": [float(np.tanh(mz[:, 0]).min()), float(np.tanh(mz[:, 0]).max())],
            "rho_F10_range": [float(np.tanh(mz[:, 1]).min()), float(np.tanh(mz[:, 1]).max())],
            "single_channel_O1_Oz_O2_delta_z": {CHANNELS[j]: float(delta[j]) for j in roi},
            "posterior_roi_reference": validation["posterior_reference"][comparison],
            "per_channel": per_channel,
        }
    deviation = OUT / "DEVIATION_RECORD.json"
    if deviation.exists():
        saved = json.loads(deviation.read_text(encoding="utf-8"))
        if sha(OUT / "PLAN_DEVIATION.md") != saved["sha256"]:
            raise StopAnalysis("plan deviation file changed")
        summary["plan_deviations"] = saved
    import scipy
    summary["software"] = {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__}
    write_json(OUT / "RUN_RECORD.json", {
        "report_written_at_utc": summary["created_at_utc"],
        "plan_record_sha256": sha(OUT / "PLAN_RECORD.json"),
        "validation_sha256": sha(OUT / "VALIDATION.json"),
        "files_sha256": {name: sha(OUT / name) for name in
                         ("pre_channel_band_powers.npz", "post_channel_band_powers.npz", "topography_inference.npz")},
        "software": summary["software"],
        "platform": platform.platform(),
    })
    write_json(OUT / "results_summary.json", summary)
    with (OUT / "channel_results.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    log("report written")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["lock", "extract", "assemble", "validate", "infer", "report"])
    parser.add_argument("cohort", nargs="?", choices=["pre", "post"])
    parser.add_argument("--budget", type=float, default=float("inf"))
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    if args.stage == "lock":
        print(json.dumps(lock_plan(), indent=2))
    elif args.stage == "extract":
        extract(args.cohort, args.budget, args.workers)
    elif args.stage == "assemble":
        assemble()
    elif args.stage == "validate":
        validate()
    elif args.stage == "infer":
        infer(args.workers)
    else:
        report()


if __name__ == "__main__":
    main()
