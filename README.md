# Temporal Sampling Geometry and Participant-Level Reproducibility in Fixed-Duration Resting EEG

Software version **2026-11-07** accompanies the study of equal-duration temporal sampling in eyes-closed resting EEG. Each sample retains ten 4-s periodograms (40 s). The primary comparison concerns participant-rank reproducibility of posterior relative alpha under contiguous and distributed sampling geometries.

The package includes the original primary, complete-support and directional-replication results; temporal-order controls, simulations and boundary-annotation sensitivity; and the additional absolute-difference, pre/post recording, spectral-normalization and fixed-sample re-pairing analyses. The additional analyses were planned after the original results were known and are exploratory.

It also includes the exploratory muscle-artifact sensitivity analysis, the [duration-by-geometry analysis](duration_geometry/), the [scalp topography](topography/) of the geometry contrast and the [supplementary implementation details](docs/SUPPLEMENTARY_DETAILS.md). Plan and record files keep their original wording, including references to earlier editions of this archive.

## Study design and main findings

All geometries retain 40 s and use ten identically computed 4-s periodograms per sample, from a common 168-s candidate interval. A and B samples do not overlap within a draw; each participant contributes 200 paired draws.

| Geometry | Retained samples |
|---|---|
| Contiguous | One 40-s block |
| Distributed 5 × 8 s | Five 8-s blocks across five temporal sections |
| Distributed 10 × 4 s | Ten 4-s cells across ten temporal sections |

In 531 primary participants, the mean rank correlation was 0.931 with one contiguous 40-s block and 0.963 with ten distributed 4-s windows (Fisher-scale difference Δz = 0.31, 95% participant-bootstrap interval 0.25–0.37). A separate 59-participant dataset showed the same direction. Between separate pre- and post-battery recordings of 524 participants, distributed sampling still gave higher rank correlations (Δz = 0.08, 95% interval 0.05–0.10).

The results apply to posterior relative alpha in healthy adults at eyes-closed rest. They do not set a minimum recording duration.

## Data and study records

- Primary data: [OpenNeuro ds005385 v1.0.3](https://doi.org/10.18112/openneuro.ds005385.v1.0.3).
- Replication data: [OpenNeuro ds004148 v1.0.0](https://doi.org/10.18112/openneuro.ds004148.v1.0.0).
- Preregistration: [OSF nh4d8](https://osf.io/nh4d8/), with an embargo ending 30 August 2027.
- Supporting pre-analysis materials: [OSF g2apc](https://osf.io/g2apc/), deposited separately from the registration.
- Archived software: this version is archived on Zenodo; the DOI is given in the article.

The manuscript and its Online Resource 1 are supplied through the journal; this repository holds the code, derived results and implementation details.

## Install and verify

Use CPython 3.14.3 and the versions in `requirements.txt`. Run from this directory:

```text
python -m venv .venv
```

Activate the environment with `.venv\Scripts\Activate.ps1` in PowerShell, or `source .venv/bin/activate` in a POSIX shell. Then install and verify:

```text
python -m pip install -r requirements.txt
python -B scripts/verify_release.py
python -B -m unittest discover -s tests -v
python -B -m unittest discover -s additional/tests -v
python -B scripts/run_preregistered_analysis.py
python -B scripts/verify_results.py --output outputs/result_checks.json
python -B scripts/run_boundary_sensitivity.py --check
python -B additional/src/verify_manifest.py
python -B additional/src/verify_results.py
```

Without `--run`, the input validator checks the input files and schedules without reading EEG. The result verifier recomputes the stored correlations and summaries and repeats selected bootstrap and simulation steps; it does not reprocess raw EEG. The checks for the additional analyses are described in [additional/README.txt](additional/README.txt). The checksum check ignores local environments and caches: `.venv/`, `venv/`, `.git/`, `outputs/`, `__pycache__/` and `*.pyc`.

## Regenerate displays

```text
python -B scripts/make_figures.py --output outputs/displays
python -B scripts/make_tables.py --output outputs/tables
```

The figure command creates Figures 1–4 and S1–S5, Tables S3 and S5–S9, and the plotted data. The table command exports the numbers in Tables 1–2 and S1–S4. The reporting table is a fixed template; all numerical tables are built from the supplied results. Figures 5, 6 and S6 are drawn by the scripts listed in [the reproduction guide](docs/REPRODUCING.md#figure-scripts).

## Reproduction and interpretation

[The reproduction guide](docs/REPRODUCING.md) distinguishes saved-array checks, complete bootstrap computation, metadata reconstruction and raw-EEG processing. [Data access](docs/DATA_ACCESS.md) specifies versioned input layouts; [computational conventions](docs/COMPUTATIONAL_CONVENTIONS.md) gives sampling, quality-control and numerical details; [supplementary details](docs/SUPPLEMENTARY_DETAILS.md) lists the seeds, checksums, full-precision values and exclusion identifiers referred to in Online Resource 1.

## Contents

| Directory | Contents |
|---|---|
| `scripts/`, `tests/` | Original-analysis computation, verification and tests adapted to this directory layout |
| `inputs/preanalysis/` | Unchanged primary census, quality-control roster and sampling schedule |
| `structural/` | Versioned replication inventory |
| `results/` | Original analyses, exploratory controls and boundary sensitivity |
| `data/` | Source-attributed public demographic metadata |
| `additional/` | Additional-analysis code, metadata, derived inputs/results, templates and tests |
| `artifact_sensitivity/` | Exploratory muscle-artifact sensitivity: plan, code, schedules and results |
| `duration_geometry/` | Exploratory duration-by-geometry analysis: plan, code, draws, results, table, figure and the article Figure 5 script |
| `topography/` | Exploratory scalp topography of the geometry contrast: plan, deviation, code, channel band powers, results and Figures 6 and S6 |
| `release_manifest.json` | File checksums for this version |

Participant identifiers are those of the public datasets. No raw EEG is distributed.

## License and citation

Software and tests use the [MIT License](LICENSE-CODE.txt). Original documentation, figures, tables and derived research outputs use [CC BY 4.0](LICENSE-CONTENT.txt). Source OpenNeuro data retain their CC0 terms; factual source metadata are not relicensed. See [license scope](LICENSE.txt), [data attribution](DATA_LICENSE.md), and [citation](CITATION.md).
