# Scalp topography of the sampling-geometry contrast: analysis plan

## status

Exploratory. The local timestamp and SHA-256 values in PLAN_RECORD.json are an audit record. No multiplicity correction is applied; channel-level intervals are descriptive.

## question

Is the difference in rank reproducibility between distributed 10x4-s and contiguous 40-s sampling of relative alpha limited to the posterior channels of the main analysis (O1, Oz, O2), or does it extend across the scalp? Examined within the pre-battery recording (the main A/B comparison) and between the pre- and post-battery recordings (the comparison in which the timing of paired samples within a recording plays no role).

## scope

One feature (relative alpha, 8-13 Hz power over 1-30 Hz power), one contrast (distributed 10x4 s minus contiguous 40 s) and one cohort (ds005385). The 5x8-s geometry, other features and the ds004148 replication cohort are not part of this analysis.

## channels

The 62 EEG channels of ds005385 other than TP9 and TP10, each referenced to the mean of TP9 and TP10 as in the main analysis. The 64 EEG labels and their order are checked in every file; a missing or reordered channel stops the analysis.

## data

Pre-battery: the 531 primary EDF files in raw_cache/production/ds005385, verified against the registered byte counts and SHA-256 digests. Post-battery: the 524 post-battery EDF files of post_prepared.npz in followup_2026-11-01/raw_post, verified against the byte counts and SHA-256 digests in post_metadata.json.

## processing

Support [8,176) s and 42 non-overlapping 4-s cells. Per channel and cell, the registered elementary periodogram (linear detrend, periodic Hann, one-sided density, native sampling rate, 0.25-Hz grid). Band powers by trapezoidal integration over 8-13 Hz and 1-30 Hz, endpoints included. No filtering, resampling, ICA, channel interpolation or additional cleaning. The existing cell-eligibility masks and A/B schedules are reused unchanged; they were defined from the posterior channels and the reference, so other channels are not screened separately.

## marker

For each channel, sample and draw: mean 8-13 Hz power over the sample's ten cells divided by mean 1-30 Hz power over the same cells.

## comparisons

within: 531 participants; contiguous (C) and distributed 10x4-s (F10) original schedules; sample A against sample B within the pre-battery recording. between: 524 participants; pre-battery sample A (original schedule) against post-battery sample A (post-battery schedule of the 2026-11-01 follow-up), as in the published between-recording comparison.

## statistics

Per channel, comparison and geometry: Spearman midrank correlation across participants in each of 200 draws, Fisher z without clipping, mean over draws, rho_tilde = tanh(mean z). Delta_z = mean z(F10) - mean z(C). MCSE = sample SD of the draw-level contrast / sqrt(200).

## bootstrap

The existing 5000 x N participant index matrices, shared by all channels: within = generate_bootstrap_indices(531, 'primary_ds005385'), the primary-analysis matrix; between = bootstrap_indices stored in followup_2026-11-01/results/pre_post_cross_inference.npz. Exact weighted-midrank Spearman in every resample. Unadjusted 95% percentile intervals (NumPy method linear).

## summaries

All reported: (1) per-channel Delta_z scalp maps for both comparisons on one colour scale centred at zero, with O1, Oz and O2 marked; (2) the mean of the 62 channel Delta_z values with a bootstrap interval (mean over channels within each resample); (3) the number of channels with Delta_z > 0 and the numbers whose interval lies wholly above or wholly below zero; (4) the smallest and largest channel Delta_z and their channels; (5) across channels, the Spearman correlation between contiguous rho_tilde and Delta_z, descriptive only; (6) rho_tilde maps for C and F10 in both comparisons and a per-channel CSV.

## validation

Before any channel-level outcome: (a) per participant and cell, the mean of the O1, Oz and O2 band powers reproduces the stored posterior cell band powers (pre: exploratory_2026-09-20/cell_band_powers.npz for its 516 participants; post: post_prepared.npz power) within relative 1e-10; (b) posterior markers from the stored schedules reproduce primary_markers (531) and the post_prepared markers (524) within absolute 1e-12; (c) the inference code reproduces the stored primary draw-level Fisher z for C and F10 within 1e-12 and primary_bootstrap_delta_z within 1e-10, and the stored between-recording cross_A mean z and bootstrap z for C and F10 within 1e-10. Any failure stops the analysis.

## stop_rules

A missing or reordered channel, a hash or byte-count mismatch, a sampling-rate mismatch, a nonfinite sample, a nonfinite or nonpositive band power in an eligible cell, alpha power not below 1-30 Hz power, an undefined correlation or an infinite Fisher transform stops the analysis. No clipping, dropping, repair or replacement.

## interpretation

Descriptive. No channel is called significant and no region is selected after seeing the maps. The within-recording map carries the same timing confound as the main within-recording comparison; the between-recording map does not. Channels outside the posterior region were not separately screened for artifacts; this applies equally to both geometries but may lower reproducibility at those channels.

## reporting

All results are reported whatever their direction: one main-text figure with the two Delta_z maps and one Results paragraph; the rho_tilde maps and the summaries in the Online Resource; per-channel values in the archive.

## outputs

All new files go only to followup_topography_2026-11. Original results, schedules, figures and manuscript files are not changed by this script.
