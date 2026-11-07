"""Outcome-gated, reproducible exploratory EMG sensitivity analysis."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import re
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

sys.dont_write_bytecode = True
import numpy as np

DEFAULT_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DEFAULT_REPO / "scripts"))
COHORTS = ("ds005385", "ds004148")
PLAN = {
    "status": "Exploratory; fixed before any new relative-alpha or rank outcome. Local timestamp and SHA-256 are an audit record, not an external preregistration.",
    "question": "How much does distributed 10x4-s minus contiguous Fisher-z rank reproducibility change after an additional within-participant EMG screen?",
    "cohorts": "Start from exactly the published primary 531 and replication 59 participants, in published order. Require all 200 paired draws for C, F5 and F10 under the strict mask. Exclude a participant if any required draw fails; no partial schedules, repairs or cohort additions.",
    "estimator": "42 four-second cells in [8,176) s. O1, Oz, O2 referenced to mean(TP9,TP10). Reuse compute_cell_periodograms: linear detrend, periodic Hann, one-sided density periodogram at native 1000/500 Hz, 0.25-Hz grid. For alpha: mean of 10 periodograms within channel, then mean over channels; trapezoidal 8-13 / 1-30 Hz. Reuse cell_band_powers and arm_markers for algebraically equivalent linear power aggregation, never average cellwise ratios; verify against published original markers within absolute 1e-12 before inference.",
    "emg": "For each participant, posterior channel and four-second cell, x=natural log(trapezoidal integral of its 55-95 Hz density periodogram, endpoints included, power in V^2). No alpha or 1-30 Hz quantity enters this criterion. Within each participant and channel, calculate median and MAD only over originally eligible cells; z=(x-median)/(1.4826*MAD). Flag an originally eligible cell if ANY posterior channel has z>3; strict_mask=original_mask AND NOT flagged. Equality at z=3 passes. No amplitude thresholds are tightened.",
    "degenerate_values": "If MAD=0, define z=0 at the median and signed infinity for nonzero deviations, the limiting robust-z rule. Nonpositive or nonfinite eligible-cell EMG powers, undefined outcome correlations or infinite Fisher transforms stop analysis; no clipping, dropping or replacement. An undefined diagnostic due to a constant vector is reported as undefined.",
    "filter_audit": "Before EMG processing or alpha outcomes, inspect every included EDF/BrainVision header and available matching local BIDS sidecar. Record header hashes, literal prefilter entries and filter-related metadata. If a numeric low-pass cutoff is <=95 Hz, stop before outcome processing; changing band requires a separate PLAN_DEVIATION.md before outcomes. Missing/NaN hardware cutoffs are explicitly unknown, not evidence of no filtering; retain 55-95 Hz without assuming hardware passband validity and report this limitation.",
    "stage1": "Compute only EMG masks and deterministic sampling feasibility, with no relative-alpha or rank outcomes. Report flagged-count histogram/quantiles per participant, per-position flagged counts divided by original eligibility counts, adjacent flagged pairs and flagged-run lengths. Adjacent-pair random-placement reference is descriptive sum_i k_i(k_i-1)/42; no clustering test. Generate the exact 200-draw feasibility counts using the future strict-schedule seeds. If primary retained N<400, STOP here, save a stopped report and await user decision. No stage2/3 or optional outcome analysis after this stop.",
    "schedules": "Reuse rebuild_schedule/draw_pair without changing candidate enumeration, strata, A/B nonoverlap, rejection limit or failure rules. Change ONLY dataset seed argument to ds005385-artifact-emg-2026-11 or ds004148-artifact-emg-2026-11. Keep ASCII prefix AUX004 and all other seed tokens. Seed is unsigned big-endian first eight SHA-256 bytes of {prefix}|{dataset_label}|{participant}|{regime}|{draw}|{arm}; NumPy default_rng/PCG64. Save schedules only after stage1 stop gate passes. Save NPZ-file hashes and C-order uint8 array hashes.",
    "outcomes": "For each retained cohort and design, recompute across-participant Spearman midranks in each of 200 draws; Fisher-transform without clipping, average z and back-transform rho_tilde=tanh(mean z). Delta_z=mean_draw(z_F10-z_C). Comparator is the original marker array and original schedules on exactly the SAME retained participant subset, with correlations recomputed on that subset. Paired change=strict Delta_z-original Delta_z. MCSE is sample SD of draw-level contrast / sqrt(200), separately for strict, original and paired change.",
    "bootstrap": "5000 participant resamples using generate_bootstrap_indices(N, artifact_emg_ds005385) or artifact_emg_ds004148. Full seed strings OSF-nh4d8|participant_bootstrap|artifact_emg_ds005385 and OSF-nh4d8|participant_bootstrap|artifact_emg_ds004148; same first-eight-byte SHA-256 rule and PCG64. Generate one (5000,N) int64 matrix per cohort in retained order and use it jointly for strict, original, all designs, draws and arms. Recompute Spearman midranks in every resample, including duplication ties. Save index hashes as little-endian int64 C order. Unadjusted two-sided 95% percentile intervals, NumPy method=linear, for strict and original contrast and their paired change. No significance test or equivalence claim.",
    "diagnostic": "Spearman correlation of participant flagged-cell count against original relative alpha averaged across all three designs, 200 draws and both arms. Report separately for the full original cohort and the retained subset; this is diagnostic, with no p-value or selection threshold.",
    "interpretation": "If strict Delta_z interval is above zero AND paired-change interval includes zero: 'Daha sıkı bir kas-artefaktı ölçütü kontrastı belirgin biçimde değiştirmedi.' No equivalence claim. If paired-change interval is wholly below zero, report its magnitude and that part of the contrast may be associated with clustered muscle artifacts. If strict interval includes zero, state this directly without softening. Otherwise describe estimates and intervals without forcing a branch. EMG power is a proxy, not proof of muscle origin.",
    "outputs": "All new files go only to followup_artifact_2026-11. Do not alter original results, schedules, figures or manuscript. Deliver plan MD/JSON with identical content, PLAN_RECORD with SHA-256/timestamps, filter audit, stage1 JSON, participant and position CSVs, masks, schedules/hashes if gate passes, results_summary.json, small LaTeX table (no long dash; missing entries --), short Turkish SONUC_NOTU.txt, <=2 suggested manuscript sentences and <=1 suggested Online Resource paragraph, one-command script and short invariants test. If stopped, outcomes and schedule files are absent and table/report explicitly say not computed.",
    "optional": "The optional pre/post absolute-difference add-on is not selected for this run; it would need its own prior plan. No pre/post marker arrays will be opened.",
}


def stamp():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def array_sha(a, dtype=None):
    return hashlib.sha256(np.ascontiguousarray(a, dtype=dtype).tobytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def plan_md():
    return "# Artifact sensitivity analysis plan\n\n" + "\n\n".join(
        f"## {key}\n\n{value}" for key, value in PLAN.items()) + "\n"


def lock_plan(out):
    record = out / "PLAN_RECORD.json"
    if not record.exists():
        if (out / "ANALYSIS_PLAN.md").exists() or (out / "ANALYSIS_PLAN.json").exists():
            raise RuntimeError("Incomplete plan exists; will not overwrite")
        write_json(out / "ANALYSIS_PLAN.json", PLAN)
        (out / "ANALYSIS_PLAN.md").write_text(plan_md(), encoding="utf-8")
        write_json(record, {"recorded_at_utc": stamp(), "relative_alpha_or_rank_outcomes_calculated": False,
                            "files": {name: {"sha256": sha(out / name),
                                             "mtime_utc": datetime.fromtimestamp((out / name).stat().st_mtime, timezone.utc).isoformat()}
                                      for name in ("ANALYSIS_PLAN.md", "ANALYSIS_PLAN.json")}})
    saved = json.loads(record.read_text(encoding="utf-8"))
    if json.loads((out / "ANALYSIS_PLAN.json").read_text(encoding="utf-8")) != PLAN:
        raise RuntimeError("Plan content changed; explicit deviation required")
    if (out / "ANALYSIS_PLAN.md").read_text(encoding="utf-8") != plan_md():
        raise RuntimeError("MD/JSON plans differ")
    for name, data in saved["files"].items():
        if sha(out / name) != data["sha256"]:
            raise RuntimeError("Plan hash changed")
    return sha(record)


def read_edf_header(path):
    with path.open("rb") as f:
        fixed = f.read(256)
        total = int(fixed[184:192])
        channels = int(fixed[252:256])
        extra = f.read(total - 256)
    if len(extra) != total - 256:
        raise RuntimeError("Incomplete EDF header")
    fields = {}; offset = 0
    for name, width in (("label", 16), ("transducer", 80), ("unit", 8), ("pmin", 8), ("pmax", 8),
                        ("dmin", 8), ("dmax", 8), ("prefilter", 80), ("samples", 8), ("reserved", 32)):
        fields[name] = [extra[offset+i*width:offset+(i+1)*width].decode("ascii").strip() for i in range(channels)]
        offset += width * channels
    return {"header_sha256": hashlib.sha256(fixed+extra).hexdigest(),
            "prefilters": {label: fields["prefilter"][i] for i, label in enumerate(fields["label"])
                           if label in ("O1", "Oz", "O2", "TP9", "TP10")}}


def load_rosters(base, primary_root=None, replication_root=None):
    primary_root = primary_root or base / "raw_data/ds005385"
    replication_root = replication_root or base / "raw_data/ds004148"
    from run_preregistered_analysis import validate_registered_inputs, read_csv, REPLICATION_CENSUS
    from reproduce_inputs import load_replication_roster, parse_mask
    inputs = validate_registered_inputs()
    roster, census = load_replication_roster()
    rep = [row for row in roster if row["qc_schedule_eligible"].lower() == "true"]
    assert len(rep) == 59
    census_by_id = {row["participant_id"]: row for row in census}
    records = {"ds005385": [], "ds004148": []}
    for i, row in enumerate(inputs["primary_rows"]):
        records["ds005385"].append(dict(id=row["participant_id"], path=primary_root/row["target_recording"],
            mask=inputs["eligible_masks"][i], expected_size=int(row["annex_bytes"]), digest=row["annex_digest"], algorithm="sha256"))
    for row in rep:
        pid = row["participant_id"]; c = census_by_id[pid]
        records["ds004148"].append(dict(id=pid, path=replication_root/pid/"ses-session1/eeg"/f"{pid}_ses-session1_task-eyesclosed_eeg.vhdr",
            mask=parse_mask(row["eligible_mask"]), expected_size=int(c["annex_signal_bytes"]), digest=c["annex_digest"], algorithm="md5"))
    return records


def filter_audit(base, out, records, metadata_roots=None):
    metadata_roots = metadata_roots or {}
    audit = {"created_at_utc": stamp(), "before_emg_and_alpha": True, "cohorts": {}, "numeric_conflicts": []}
    for cohort, rows in records.items():
        snapshot = metadata_roots.get(cohort)
        entries = []
        for row in rows:
            p = row["path"]
            if not p.is_file():
                raise FileNotFoundError(p)
            if cohort == "ds005385":
                entry = read_edf_header(p)
                literals = list(entry["prefilters"].values())
            else:
                content = p.read_text(encoding="utf-8-sig")
                literals = [line for line in content.splitlines() if re.search(r"filter|pass|cutoff", line, re.I)]
                entry = {"header_sha256": sha(p), "filter_lines": literals}
            relative = Path(row["id"]) / ("ses-1/eeg" if cohort == "ds005385" else "ses-session1/eeg") / p.with_suffix(".json").name
            sidecar = snapshot / relative if snapshot is not None else p.with_suffix(".json")
            entry.update(participant_id=row["id"], header_file=p.name, sidecar_available=sidecar.exists())
            if sidecar.exists():
                metadata = json.loads(sidecar.read_text(encoding="utf-8-sig"))
                entry["sidecar_sha256"] = sha(sidecar)
                entry["filter_metadata"] = {k:v for k,v in metadata.items() if "filter" in k.lower()}
                literals += [json.dumps(entry["filter_metadata"])]
            for text in literals:
                for number in re.findall(r"(?:LP\s*:|low[ -]?pass\s*[:=]?)\s*([\d.]+)\s*(?:Hz)?", text, re.I):
                    if float(number) <= 95:
                        audit["numeric_conflicts"].append({"cohort": cohort, "participant_id": row["id"], "cutoff_hz": float(number), "literal": text})
            entry["hardware_lowpass_status"] = "unknown unless a numeric header/sidecar cutoff is explicitly recorded"
            entries.append(entry)
        audit["cohorts"][cohort] = {"N": len(entries), "entries": entries}
    audit["conclusion"] = ("STOP: numeric cutoff overlaps or precedes 55-95 Hz" if audit["numeric_conflicts"] else
        "No numeric low-pass cutoff established by the headers/sidecars. Hardware passband is UNKNOWN; NaN/missing/n/a does not establish absence of filtering. Retain planned 55-95 Hz proxy with this limitation; no band change.")
    write_json(out/"hardware_filter_audit.json", audit)
    print("FILTER_AUDIT:", audit["conclusion"], flush=True)
    if audit["numeric_conflicts"]:
        raise RuntimeError("Hardware-filter conflict; stop before EMG/alpha and record a deviation before any band change")
    return audit


def read_and_validate(row, cohort):
    from run_preregistered_analysis import _load_required_support, _brainvision_references, _validate_brainvision_continuity, time_domain_qc_mask
    raw = row["path"]
    if cohort == "ds004148":
        raw, marker = _brainvision_references(row["path"])
        _validate_brainvision_continuity(marker)
    if raw.stat().st_size != row["expected_size"]:
        raise RuntimeError("Raw size mismatch: " + row["id"])
    h = hashlib.new(row["algorithm"])
    with raw.open("rb") as f:
        for block in iter(lambda:f.read(1024*1024), b""):
            h.update(block)
    if h.hexdigest() != row["digest"]:
        raise RuntimeError("Raw checksum mismatch: " + row["id"])
    data, rate = _load_required_support(row["path"], "edf" if cohort == "ds005385" else "brainvision")
    if rate != (1000 if cohort == "ds005385" else 500):
        raise RuntimeError("Unexpected sampling rate")
    if not np.array_equal(time_domain_qc_mask(data*1e6, rate), row["mask"]):
        raise RuntimeError("Raw QC mask mismatch: " + row["id"])
    return data, rate


def emg_mask(data, rate, original_mask):
    from preregistered_analysis import compute_cell_periodograms
    frequency, psd = compute_cell_periodograms(data, rate)
    band = (frequency >= 55) & (frequency <= 95)
    powers = np.trapezoid(psd[..., band], frequency[band], axis=-1)
    if not np.isfinite(powers[:, original_mask]).all() or (powers[:, original_mask] <= 0).any():
        raise RuntimeError("Invalid eligible-cell EMG power; no repair")
    with np.errstate(divide="ignore", invalid="ignore"):
        x = np.log(powers)
    median = np.median(x[:, original_mask], axis=1)
    mad = np.median(np.abs(x[:, original_mask]-median[:, None]), axis=1)
    z = np.zeros_like(x)
    for channel in range(3):
        if mad[channel] == 0:
            z[channel] = np.where(x[channel] == median[channel], 0, np.where(x[channel] > median[channel], np.inf, -np.inf))
        else:
            z[channel] = (x[channel]-median[channel])/(1.4826*mad[channel])
    flags = original_mask & (z > 3).any(axis=0)
    strict = original_mask & ~flags
    return strict, flags, x, median, mad


def validate_schedule(strict, schedule):
    assert schedule.shape == (3, 200, 2, 10)
    assert np.asarray(strict)[schedule].all(), "Schedule selected rejected cell"
    assert (np.diff(np.sort(schedule, axis=-1), axis=-1) > 0).all(), "Repeated cell in arm"
    assert not (schedule[:, :, 0, :, None] == schedule[:, :, 1, None, :]).any(), "A/B overlap"


def distribution(values):
    a = np.asarray(values)
    return {"min": int(a.min()), "median": float(np.median(a)), "max": int(a.max()),
            "mean": float(a.mean()), "quantiles_0_25_50_75_100": np.percentile(a, [0,25,50,75,100], method="linear").tolist(),
            "histogram": {str(k): int(v) for k,v in sorted(Counter(a.tolist()).items())}}


def stage1_one(task):
    """Independent participant computation; no outcomes and no file writes."""
    from reproduce_inputs import rebuild_schedule, prepared_candidates
    row, cohort = task
    data, rate = read_and_validate(row, cohort)
    strict, flags, x, med, mad = emg_mask(data, rate, row["mask"])
    schedule, valid = rebuild_schedule(strict, cohort+"-artifact-emg-2026-11", row["id"])
    prepared_candidates.cache_clear()
    return strict, flags, x, med, mad, schedule, valid


def stage1(base, out, records):
    # map preserves roster order; participant streams depend only on fixed labels.
    with ProcessPoolExecutor(max_workers=6) as pool:
        return stage1_collect(base, out, records, pool)


def stage1_collect(base, out, records, pool):
    stage = {"started_at_utc": stamp(), "relative_alpha_or_rank_outcomes_calculated": False, "cohorts": {}}
    bundles = {}
    for cohort, rows in records.items():
        originals=[]; stricts=[]; flagged=[]; logs=[]; medians=[]; mads=[]; counts=[]; schedules=[]; included=[]; report=[]
        computed = pool.map(stage1_one, ((row, cohort) for row in rows))
        for i, (row, result) in enumerate(zip(rows, computed)):
            strict, flags, x, med, mad, schedule, valid = result
            assert not (strict & ~row["mask"]).any()
            if schedule is not None:
                validate_schedule(strict, schedule)
                schedules.append(schedule); included.append(i)
            originals.append(row["mask"]); stricts.append(strict); flagged.append(flags); logs.append(x); medians.append(med); mads.append(mad); counts.append(valid)
            report.append({"participant_id": row["id"], "original_eligible_cells": int(row["mask"].sum()),
                           "flagged_cells": int(flags.sum()), "strict_eligible_cells": int(strict.sum()),
                           "C_valid_draws": valid[0], "F5_valid_draws": valid[1], "F10_valid_draws": valid[2], "retained": schedule is not None,
                           "original_mask": "".join(map(str, row["mask"].astype(int))), "strict_mask": "".join(map(str, strict.astype(int)))})
            if (i+1)%10 == 0 or i+1 == len(rows):
                print(f"STAGE1 {cohort} {i+1}/{len(rows)} retained={len(included)}", flush=True)
        originals=np.asarray(originals); stricts=np.asarray(stricts); flagged=np.asarray(flagged); counts=np.asarray(counts)
        k=flagged.sum(axis=1); runs=Counter()
        for flags in flagged:
            boundaries=np.diff(np.r_[False,flags,False].astype(int))
            runs.update((np.flatnonzero(boundaries == -1)-np.flatnonzero(boundaries == 1)).tolist())
        positions=[]
        for j in range(42):
            den=int(originals[:,j].sum()); num=int(flagged[:,j].sum())
            positions.append({"cell_index":j, "start_seconds":8+4*j, "end_seconds":12+4*j,
                              "originally_eligible":den, "flagged":num, "flagged_fraction": num/den if den else None})
        stage["cohorts"][cohort]={"original_N":len(rows), "retained_N":len(included), "excluded_N":len(rows)-len(included),
            "flagged_cells_total":int(k.sum()), "original_eligible_cells_total":int(originals.sum()),
            "flagged_per_participant":distribution(k), "strict_cells_per_participant":distribution(stricts.sum(axis=1)),
            "participants_supporting_200_by_design":dict(zip(("C","F5","F10"), (counts==200).sum(axis=0).tolist())),
            "retained_ids":[rows[i]["id"] for i in included], "excluded_ids":[row["id"] for i,row in enumerate(rows) if i not in included],
            "adjacent_flagged_pairs_observed":int((flagged[:,:-1]&flagged[:,1:]).sum()),
            "adjacent_pairs_uniform_42_position_reference":float((k*(k-1)/42).sum()),
            "clustering_caution":"Descriptive only; uniform-placement reference ignores pre-existing ineligible cells and is not a test.",
            "flagged_run_length_histogram":{str(a):int(v) for a,v in sorted(runs.items())}, "cell_positions":positions,
            "zero_MAD_participant_channels":int((np.asarray(mads)==0).sum())}
        write_csv(out/f"stage1_participants_{cohort}.csv", report)
        write_csv(out/f"stage1_cell_positions_{cohort}.csv", positions)
        np.savez_compressed(out/f"masks_emg_{cohort}.npz", participant_ids=np.asarray([r["id"] for r in rows]),
            original_masks=originals, strict_masks=stricts, flagged_masks=flagged, log_emg_power=np.asarray(logs),
            log_emg_median=np.asarray(medians), log_emg_MAD=np.asarray(mads), valid_draw_counts=counts, retained_indices=np.asarray(included,dtype=np.int64))
        bundles[cohort]={"included":np.asarray(included,dtype=np.int64), "schedules":np.asarray(schedules,dtype=np.uint8), "strict_masks":stricts, "flags":flagged}
    stage["completed_at_utc"]=stamp()
    stage["stop_rule_triggered"]=stage["cohorts"]["ds005385"]["retained_N"] < 400
    stage["decision"]="STOP: primary N<400; user decision required" if stage["stop_rule_triggered"] else "PASS: proceed with unchanged plan"
    write_json(out/"stage1_feasibility.json", stage)
    print(stage["decision"], flush=True)
    return stage, bundles


def stopped_reports(out, stage):
    summary={"status":"stopped_after_stage1", "reason":"primary retained N < 400", "created_at_utc":stamp(),
             "relative_alpha_or_rank_outcomes_calculated":False, "plan_deviation":False,
             "stage1_summary":{c:{k:stage["cohorts"][c][k] for k in ("original_N","retained_N","excluded_N","flagged_cells_total")} for c in COHORTS},
             "stage2":"not run; no schedules saved", "stage3":"not run; no alpha, rank, bootstrap or diagnostic outcomes",
             "limitation":"Hardware low-pass cutoff unknown in the supplied headers and sidecars; EMG specificity cannot be established.",
             "manuscript_suggestion":"No numerical manuscript addition recommended before the user decision; no Delta-z outcome was calculated.",
             "online_resource_suggestion":"The planned exploratory 55-95 Hz EMG-proxy sensitivity analysis was stopped at the outcome-independent feasibility gate because fewer than 400 primary participants supported all 200 paired draws under all three designs. No relative-alpha, rank-reproducibility or bootstrap outcome was calculated; hardware low-pass settings were not recoverable from the available headers and sidecars."}
    write_json(out/"results_summary.json",summary)
    rows=[r"\begin{tabular}{lrrrr}", r"Cohort & Original $N$ & Strict $N$ & Strict $\Delta z$ & Paired change \\",r"\hline"]
    for c in COHORTS:
        s=stage["cohorts"][c]
        rows.append(f"{c} & {s['original_N']} & {s['retained_N']} & -- & --" + r" \\")
    rows += [r"\end{tabular}", "% Stopped before outcomes: primary strict N < 400. Missing estimates were not calculated."]
    (out/"table_artifact_sensitivity.tex").write_text("\n".join(rows)+"\n",encoding="utf-8")
    notes=["55-95 Hz log güç, katılımcı/kanal içi robust z > 3 kuralıyla tarandı; özgün QC maskesi daraltıldı."]
    for c in COHORTS:
        s=stage["cohorts"][c]; d=s["flagged_per_participant"]
        notes.append(f"{c}: {s['original_N']} katılımcıdan {s['retained_N']} kaldı; {s['flagged_cells_total']} hücre işaretlendi. Katılımcı başına işaretli hücre: medyan {d['median']:g}, aralık {d['min']}-{d['max']}.")
    notes += ["Birincil N < 400 olduğu için planın durma kuralı uygulandı. Göreli alfa, sıralama korelasyonu, Delta-z, bootstrap ve alfa-seçim tanısı hesaplanmadı; şema dosyaları üretilmedi.",
              "Ham başlıklar ve yerel BIDS yan dosyaları donanım alçak geçiren kesimini bildirmiyor (EDF NaN; BrainVision filtre alanı yok). Bu nedenle EMG bandının donanım geçiş bandıyla uyumu doğrulanamadı. Bant değiştirilmedi.",
              "Plan sapması yok. Makale değiştirilmedi. Devam kararı kullanıcıya ait; otomatik devam yok."]
    (out/"SONUC_NOTU.txt").write_text("\n\n".join(notes)+"\n",encoding="utf-8")
    (out/"MANUSCRIPT_SUGGESTIONS.txt").write_text(
        "Makale önerisi (2 cümle):\nAn exploratory EMG-proxy sensitivity analysis was attempted using within-participant outliers in 55-95 Hz posterior power. It stopped before relative-alpha or rank outcomes at the prespecified feasibility gate because fewer than 400 primary participants supported all required paired draws.\n\nOnline Resource 1 önerisi (1 paragraf):\n"+summary["online_resource_suggestion"]+"\n",encoding="utf-8")


def outcomes(base, release, out, records, stage, bundles):
    from preregistered_analysis import (compute_cell_periodograms, point_statistics, generate_bootstrap_indices,
        derive_bootstrap_seed, bootstrap_index_sha256, primary_bootstrap_distribution, percentile_interval)
    from exploratory_sampling import cell_band_powers, arm_markers
    from scipy.stats import spearmanr
    summaries={}; hashes={}
    with np.load(release/"results/preregistered/scientific_results.npz", allow_pickle=False) as old:
        for cohort in COHORTS:
            role="primary" if cohort=="ds005385" else "replication"
            bundle=bundles[cohort]; inc=bundle["included"]; schedule=bundle["schedules"]
            ids=np.asarray([records[cohort][i]["id"] for i in inc])
            fullids=np.asarray([r["id"] for r in records[cohort]])
            assert np.array_equal(fullids,old[role+"_participant_ids"])
            spath=out/f"schedules_{cohort}.npz"
            np.savez_compressed(spath,participant_ids=ids,regimes=np.asarray(["C","F5","F10"]),
                schedule=schedule,strict_masks=bundle["strict_masks"][inc],seed_dataset=np.asarray(cohort+"-artifact-emg-2026-11"))
            hashes[cohort]={"file_sha256":sha(spath),"schedule_uint8_C_order_sha256":array_sha(schedule,"u1")}
            original=old[role+"_markers"][inc]
            powers=[]
            for j,i in enumerate(inc):
                data,rate=read_and_validate(records[cohort][i],cohort)
                f,psd=compute_cell_periodograms(data,rate)
                powers.append(cell_band_powers(psd,f))
                if (j+1)%25==0 or j+1==len(inc):
                    print(f"STAGE3 powers {cohort} {j+1}/{len(inc)}",flush=True)
            powers=np.asarray(powers)
            def markers_for(s):
                result=np.empty((len(s),3,200,2))
                result[:,[0,2]]=arm_markers(powers,s[:,[0,2]])
                result[:,1]=arm_markers(powers,s[:,[1,1]])[:,0]
                return result
            strict=markers_for(schedule)
            if cohort=="ds005385":
                from run_preregistered_analysis import validate_registered_inputs
                original_schedule=validate_registered_inputs()["schedule"][inc]
            else:
                original_schedule=old["replication_schedule"][inc]
            regenerated=markers_for(original_schedule)
            error=float(np.max(np.abs(regenerated-original)))
            if error>1e-12:
                raise RuntimeError(f"Original marker reproduction failed: {error}")
            print(f"Original markers reproduced: {cohort} max_abs_error={error:.3g}",flush=True)
            strict_point=point_statistics(strict); orig_point=point_statistics(original)
            label="artifact_emg_"+cohort
            indices=generate_bootstrap_indices(len(inc),label)
            print(f"BOOTSTRAP strict {cohort}: 5000 resamples",flush=True)
            bs=primary_bootstrap_distribution(strict[:,[0,2]],indices)
            print(f"BOOTSTRAP original {cohort}: 5000 paired resamples",flush=True)
            bo=primary_bootstrap_distribution(original[:,[0,2]],indices)
            difference=bs-bo
            draw_change=strict_point["draw_differences"]-orig_point["draw_differences"]
            def compact(p,b):
                return {"rho_tilde":dict(zip(("C","F5","F10"),p["back_transformed_rho"].tolist())),
                        "mean_fisher_z":dict(zip(("C","F5","F10"),p["mean_z"].tolist())),
                        "delta_z":p["delta_z"], "ci95":list(percentile_interval(b)),"mcse":p["mcse"]}
            def diagnostic(select):
                flags=bundle["flags"].sum(axis=1)[select]
                alpha=old[role+"_markers"].mean(axis=(1,2,3))[select]
                if np.ptp(flags)==0 or np.ptp(alpha)==0:
                    return {"N":len(flags),"spearman_rho":None,"reason":"constant diagnostic vector"}
                return {"N":len(flags),"spearman_rho":float(spearmanr(flags,alpha).statistic)}
            ci=percentile_interval(bs); change_ci=percentile_interval(difference)
            interpretation=[]
            if ci[0]>0 and change_ci[0]<=0<=change_ci[1]:
                interpretation.append("Daha sıkı bir kas-artefaktı ölçütü kontrastı belirgin biçimde değiştirmedi.")
            if change_ci[1]<0:
                interpretation.append("Kontrastın bir kısmı kümelenmiş kas artefaktlarıyla ilişkili olabilir; eşli farkın aralığı sıfırın altında.")
            if ci[0]<=0<=ci[1]:
                interpretation.append("Sıkı Delta-z aralığı sıfırı içeriyor.")
            summaries[cohort]={"N":len(inc),"strict":compact(strict_point,bs),"original_same_subset":compact(orig_point,bo),
                "paired_change":{"delta_z":float(strict_point["delta_z"]-orig_point["delta_z"]),"ci95":list(change_ci),
                                 "mcse":float(np.std(draw_change,ddof=1)/np.sqrt(200))},
                "bootstrap":{"resamples":5000,"label":label,"seed":derive_bootstrap_seed(label),"indices_sha256":bootstrap_index_sha256(indices)},
                "diagnostic_full_cohort":diagnostic(np.arange(len(fullids))),"diagnostic_retained":diagnostic(inc),
                "original_marker_max_abs_error":error,"interpretation":interpretation,"schedule_hashes":hashes[cohort]}
            np.savez_compressed(out/f"inference_{cohort}.npz",participant_ids=ids,strict_markers=strict,original_markers=original,
                strict_draw_fisher_z=strict_point["draw_z"],original_draw_fisher_z=orig_point["draw_z"],
                bootstrap_indices=indices,strict_bootstrap_delta_z=bs,original_bootstrap_delta_z=bo,paired_bootstrap_change=difference)
            write_json(out/"results_summary.json",{"status":"in_progress","cohorts":summaries})
    write_json(out/"schedule_hashes.json",hashes)
    result={"status":"complete","created_at_utc":stamp(),"exploratory":True,"intervals":"unadjusted participant-bootstrap 95%; linear percentiles",
            "plan_deviation":False,"hardware_filter_limitation":"Hardware low-pass cutoffs unknown; 55-95 Hz is an EMG proxy, not confirmed muscle activity.","cohorts":summaries}
    write_json(out/"results_summary.json",result)
    rows=[r"\begin{tabular}{lrrrrrr}",r"Cohort & $N$ & $\tilde\rho_C$ & $\tilde\rho_{F5}$ & $\tilde\rho_{F10}$ & Strict $\Delta z$ [95\% CI] & Change [95\% CI] \\",r"\hline"]
    notes=[]
    for c,s in summaries.items():
        d=s["strict"];p=s["paired_change"];r=d["rho_tilde"]
        rows.append(f"{c} & {s['N']} & {r['C']:.3f} & {r['F5']:.3f} & {r['F10']:.3f} & {d['delta_z']:.3f} [{d['ci95'][0]:.3f}, {d['ci95'][1]:.3f}] & {p['delta_z']:.3f} [{p['ci95'][0]:.3f}, {p['ci95'][1]:.3f}]" + r" \\")
        notes.append(f"{c}: N={s['N']}; sıkı Delta-z={d['delta_z']:.4f}, %95 aralık {d['ci95']}; aynı alt kümede özgün Delta-z={s['original_same_subset']['delta_z']:.4f}; eşli fark={p['delta_z']:.4f}, %95 aralık {p['ci95']}; MCSE (sıkı/özgün/fark)={d['mcse']:.5f}/{s['original_same_subset']['mcse']:.5f}/{p['mcse']:.5f}. "+" ".join(s["interpretation"]))
    rows.append(r"\end{tabular}")
    (out/"table_artifact_sensitivity.tex").write_text("\n".join(rows)+"\n",encoding="utf-8")
    (out/"SONUC_NOTU.txt").write_text("55-95 Hz log güç için katılımcı/kanal içi robust z > 3 ölçütü uygulandı. Tüm tasarımlarda 200 çekilişi destekleyenler tutuldu.\n\n"+"\n\n".join(notes)+"\n\nAralıklar 5000 eşli katılımcı-bootstrap ile hesaplandı; keşifsel ve düzeltilmemiştir. Eşdeğerlik ya da anlamlılık testi yok. Donanım alçak geçiren kesimi başlıklardan doğrulanamadı. Plan sapması yok; makale değiştirilmedi.\n",encoding="utf-8")
    sentences=[]
    for c,s in summaries.items():
        d=s["strict"];p=s["paired_change"]
        sentences.append(f"In an exploratory 55-95 Hz EMG-proxy sensitivity analysis of {c} (N={s['N']}), the strict-mask contrast was {d['delta_z']:.3f} (unadjusted 95% participant-bootstrap interval {d['ci95'][0]:.3f} to {d['ci95'][1]:.3f}), with a paired change from the original schedules on the same subset of {p['delta_z']:.3f} ({p['ci95'][0]:.3f} to {p['ci95'][1]:.3f}).")
    paragraph="Within each participant and posterior channel, 55-95 Hz log power was robust-standardized across originally eligible four-second cells; cells exceeding z=3 in any channel were removed. All 200 disjoint paired draws were regenerated under the original sampling rules with independent fixed seed labels, retaining only participants supporting every design. Original-schedule contrasts were recomputed on the same retained subset; 5000 shared participant-bootstrap resamples recomputed Spearman midranks and provided unadjusted percentile intervals for both contrasts and their paired difference. Hardware low-pass cutoffs were unavailable in the headers and sidecars, so the metric is an EMG proxy with uncertain hardware passband validity; this is exploratory and does not establish equivalence."
    (out/"MANUSCRIPT_SUGGESTIONS.txt").write_text("Manuscript (two sentences):\n"+" ".join(sentences)+"\n\nOnline Resource 1 (one paragraph):\n"+paragraph+"\n",encoding="utf-8")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root",type=Path,default=DEFAULT_REPO)
    parser.add_argument("--primary-root",type=Path,required=True,help="ds005385 v1.0.3 raw BIDS root")
    parser.add_argument("--replication-root",type=Path,required=True,help="ds004148 v1.0.0 raw BIDS root")
    parser.add_argument("--primary-metadata-root",type=Path,help="Optional matching BIDS metadata root")
    parser.add_argument("--replication-metadata-root",type=Path,help="Optional matching BIDS metadata root")
    parser.add_argument("--output",type=Path,required=True,help="New output directory outside the supplied results")
    args=parser.parse_args()
    base=args.repo_root.resolve();out=args.output.resolve()
    if out == base or out == Path(__file__).resolve().parent:
        raise RuntimeError("Output must be a new dedicated directory")
    out.mkdir(parents=True,exist_ok=True)
    plan_record_sha=lock_plan(out)
    release=base
    sys.path.insert(0,str(release/"scripts"))
    records=load_rosters(base,args.primary_root.resolve(),args.replication_root.resolve())
    sources=[release/"scripts"/name for name in ("sampling_design.py","preregistered_analysis.py","reproduce_inputs.py","run_preregistered_analysis.py","exploratory_sampling.py")]
    sources += [release/"results/preregistered/scientific_results.npz", release/"docs/COMPUTATIONAL_CONVENTIONS.md", Path(__file__)]
    manifest={"started_at_utc":stamp(),"python":platform.python_version(),"numpy":np.__version__,
              "plan_record_sha256":plan_record_sha,"input_sha256":{p.resolve().relative_to(base).as_posix():sha(p) for p in sources}}
    write_json(out/"RUN_RECORD.json",manifest)
    filter_audit(base,out,records,{"ds005385":args.primary_metadata_root,"ds004148":args.replication_metadata_root})
    stage,bundles=stage1(base,out,records)
    if stage["stop_rule_triggered"]:
        stopped_reports(out,stage)
    else:
        lock_plan(out)
        outcomes(base,release,out,records,stage,bundles)
    for path,expected in manifest["input_sha256"].items():
        if sha(base/path)!=expected:
            raise RuntimeError("Input changed during run: "+path)
    manifest["completed_at_utc"]=stamp();manifest["status"]="stopped_after_stage1" if stage["stop_rule_triggered"] else "complete"
    write_json(out/"RUN_RECORD.json",manifest)
    write_json(out/"OUTPUT_MANIFEST.json",{"created_at_utc":stamp(),"files":{p.name:sha(p) for p in sorted(out.iterdir()) if p.is_file() and p.name!="OUTPUT_MANIFEST.json"}})
    print("FINISHED:",manifest["status"],flush=True)


if __name__ == "__main__":
    main()
