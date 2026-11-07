# Reproducing the study

Use the environment and commands in the root README. Checks based on the saved arrays do not reprocess raw EEG; raw reconstruction is described below.

## Saved-array computation

To recompute all 5000 original rank-bootstrap resamples per empirical cohort:

```text
python -B scripts/verify_results.py --full-bootstrap --output outputs/full_bootstrap_checks.json
```

To regenerate all 1000 temporal-order randomizations and 500 simulated cohorts per each of 12 scenarios:

```text
python -B scripts/run_exploratory_analysis.py --run --workers 2 --output outputs/exploratory_reproduction
```

To repeat the 526-participant boundary-annotation sensitivity:

```text
python -B scripts/run_boundary_sensitivity.py --run --output outputs/boundary_reproduction
```

Use new empty output directories. The additional analyses' full derived-input commands are in `additional/README.txt`; run those in a copy because they replace that copy's corresponding result files.

## Raw EEG and original input reconstruction

Obtain the exact dataset versions described in `DATA_ACCESS.md`, preserving BIDS paths and all BrainVision companion files. The primary QC check needs all 536 structurally eligible files, including the five later exclusions.

```text
python -B scripts/reproduce_inputs.py --schedule
python -B scripts/reproduce_inputs.py --replication-schedule
python -B scripts/reproduce_inputs.py --primary-root "raw_data/ds005385"
python -B scripts/reproduce_inputs.py --replication-root "raw_data/ds004148"
python -B scripts/run_preregistered_analysis.py --run --primary-root "raw_data/ds005385" --replication-root "raw_data/ds004148" --output outputs/raw_reproduction/summary_results.json
python -B scripts/compare_npz.py outputs/raw_reproduction/scientific_results.npz results/preregistered/scientific_results.npz
python -B scripts/run_exploratory_analysis.py --run --primary-root "raw_data/ds005385" --workers 2 --output outputs/raw_exploratory
```

The primary analysis uses the fixed census and schedule. `verify_structural_census.py` reconstructs the 608-to-536 structural selection from versioned metadata and EDF headers; see `COMPUTATIONAL_CONVENTIONS.md`. `release_manifest.json` lists the checksums of all files in this edition.

## Additional post-recording reconstruction

From this root, metadata alone can be checked with:

```text
python -B additional/src/prepare_post_metadata.py --snapshot-root "raw_metadata/ds005385"
```

Use the [pinned metadata snapshot](https://github.com/OpenNeuroDatasets/ds005385/tree/6a558cd5852503e66df9ac7fdac2f3a7f4ed5f12), retaining annex pointer files or symlinks. This checks 531 original participants against the pinned ds005385 commit and recovers the 527-recording metadata selection. Follow `additional/README.txt` to obtain the raw post EDF files, check their SHA-256 checksums, prepare the final 524-person common sample and run its inference. The original type-only boundary screen is preserved; event-count sensitivity is reported separately.

## Muscle-artifact sensitivity

Check the supplied results from the repository root:

```text
python -B artifact_sensitivity/test_artifact.py
python -B artifact_sensitivity/verify_saved.py
```

To reconstruct the analysis from EEG, obtain the 531 primary and 59 replication
recordings listed in the original results. Use ds005385 v1.0.3 pre-battery
session-1 eyes-closed EDF files and ds004148 v1.0.0 session-1 eyes-closed
BrainVision files with their data and marker companions. Preserve the BIDS
paths described in `DATA_ACCESS.md`.

```text
python -B artifact_sensitivity/run_artifact.py --primary-root raw_data/ds005385 --replication-root raw_data/ds004148 --output outputs/artifact_reproduction
python -B artifact_sensitivity/verify_saved.py --output outputs/artifact_reproduction
```

Matching BIDS sidecars can be supplied with `--primary-metadata-root` and
`--replication-metadata-root`. Otherwise, the script checks sidecars beside the
raw files when present. Use a new output directory. The script records the plan,
audits headers and checks feasibility before computing outcomes. Section 13 of
`SUPPLEMENTARY_DETAILS.md` gives the seeds, exclusions and full-precision results.

## Duration-by-geometry analysis

This analysis reads only supplied cell band powers; it needs no raw EEG. Check the supplied results from the repository root:

```text
python -B duration_geometry/test_duration_geometry.py
python -B duration_geometry/verify_saved.py
```

Add `--full` to the verifier to repeat all 5000 bootstrap resamples. To rerun the analysis and rebuild the table and figure in a new output directory:

```text
python -B duration_geometry/run_duration_geometry.py --output outputs/duration_geometry
python -B duration_geometry/verify_saved.py --results outputs/duration_geometry
python -B duration_geometry/make_displays.py --results outputs/duration_geometry/results_summary.json --output outputs/duration_geometry
```

Section 14 of `SUPPLEMENTARY_DETAILS.md` gives the seeds and full-precision results.

## Scalp topography

Check the supplied results from the repository root:

```text
python -B topography/test_topography.py
python -B topography/verify_saved.py
```

The verifier recomputes the validation and all channel-level point estimates. Add `--full --workers N` to repeat all 5000 bootstrap resamples for every channel. The stages of `topography/run_topography.py` write their outputs and `run_log.txt` into `topography/`, so run them in a copy of this directory. From the supplied band powers:

```text
python -B topography/run_topography.py validate
python -B topography/run_topography.py infer --workers 4
python -B topography/run_topography.py report
python -B topography/verify_saved.py
```

To recompute the band powers from EEG, obtain the session-1 eyes-closed pre-battery and post-battery EDF files of OpenNeuro ds005385 v1.0.3, with the BIDS paths `sub-XXX/ses-1/eeg/` described in `DATA_ACCESS.md`. The script reads them from `raw/ds005385` under the repository root. The environment variables `TOPOGRAPHY_RAW_PRE` and `TOPOGRAPHY_RAW_POST` can point to other folders. Every file is checked against its registered byte count and SHA-256 checksum. The extract stage keeps one file per participant in `topography/cache/` and can be resumed; `assemble` then replaces the two band-power arrays.

```text
python -B topography/run_topography.py extract pre --workers 4
python -B topography/run_topography.py extract post --workers 4
python -B topography/run_topography.py assemble
```

Then run `validate`, `infer` and `report` as above. Section 15 of `SUPPLEMENTARY_DETAILS.md` gives the method, the plan deviation and full-precision results.

## Figure scripts

- Article Figure 5: `duration_geometry/make_figure5_manuscript.py`. `duration_geometry/make_displays.py` draws a one-panel general plot and the table of the duration-by-geometry analysis.
- Article Figure 6 and Fig. S6: `topography/make_figure6_topography.py`.

```text
python -B duration_geometry/make_figure5_manuscript.py duration_geometry/results_summary.json outputs/figure5
python -B topography/make_figure6_topography.py topography/results_summary.json outputs/figure6
```

## Notes on record files

Plan and record files keep their original wording, including references to earlier editions of this archive. The status label in the boundary-sensitivity plan file was reworded before release; its selection rule, seed, estimand, bootstrap settings and planning date are unchanged, and its new checksum is recorded in the plan verifier and summary.
