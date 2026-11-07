# Supplementary implementation details

This file accompanies Online Resource 1 of the article "Temporal Sampling Geometry and Participant-Level Reproducibility in Fixed-Duration Resting EEG". Online Resource 1 describes the designs, results and analysis history in readable form; the implementation details needed to repeat the computations are collected here. Random-number conventions, quality-control implementation and numerical tolerances are also described in [COMPUTATIONAL_CONVENTIONS.md](COMPUTATIONAL_CONVENTIONS.md), and the commands are in [REPRODUCING.md](REPRODUCING.md).

## 1. Participant flow

- ds005385 version 1.0.3 contained 608 nominal participants. Structural assessment of the session-1, pre-task eyes-closed recording classified 536 recordings as eligible using channel, duration, continuity and event checks. Structural exclusions were 58 recordings with other task markers, 13 with USB-disconnection annotations and one shorter than 180 s.
- Five structurally eligible recordings could not complete all 200 paired draws under every geometry after time-domain quality control. For example, `sub-552` had 40 eligible cells but supported only 147 of the 200 required contiguous-sampling draws: a uniformly selected sample A could have no disjoint partner.
- ds004148 version 1.0.0 contained 60 structurally compatible session-1 eyes-closed recordings. `sub-43` had 27 eligible cells and could not construct the required 200 paired draws under any geometry. This exclusion preceded spectral estimation.
- Age and sex were obtained by exact participant-identifier matching to `participants.tsv` and its data dictionary in each versioned release. Both fields were complete for all analyzed participants.

## 2. Continuity screening

- ds005385 (EDF): the screen required a continuous-recording sidecar, rejected EDF+D headers, and accepted files whose event types were all `boundary`. Event counts and onsets did not enter this screen. The preregistration required continuous recordings with only an initial boundary event; this departure is addressed by the boundary-annotation sensitivity analysis (Section 10).
- ds004148 (BrainVision): one initial `New Segment` marker was permitted, no later segment break was permitted, and periodic synchronization triggers were not interpreted as discontinuities.

## 3. Paired-draw construction

- Temporal sections partition the full ordered 42-cell grid before the eligibility mask is applied. Ineligible cells are excluded from selection without shifting the section boundaries.
- For each dataset, participant, geometry, replicate and sample label, NumPy `default_rng` is initialized from the first 64 bits of a SHA-256 digest interpreted as an unsigned big-endian integer. The seed string combines a fixed label, dataset, participant, regime, replicate and sample label; COMPUTATIONAL_CONVENTIONS.md gives its exact construction.
- Contiguous sampling enumerates every legal ten-cell block; A is selected uniformly and B uniformly from blocks disjoint from A.
- Distributed 5 × 8-s sampling enumerates every legal five-block configuration satisfying the temporal sections and the spacing rule. A is selected uniformly; B candidates are drawn with replacement until a disjoint configuration is found, with a limit of 100,000 attempts.
- Distributed 10 × 4-s sampling draws one eligible cell within each of ten temporal sections for A and a different cell within the same section for B. No additional gap is required across section boundaries.
- A participant unable to construct any required draw is ineligible; partial schedules and repair draws are not used.
- The primary schedule has shape 531 × 3 × 200 × 2 × 10, dtype `uint8`, and SHA-256 `95c0da1f105ff9f92a64010d120b8ebecc760c75540b448f209dc391e91649c1`. The complete-support analysis subsets this schedule to 516 participant rows; it does not generate a new schedule.

## 4. Estimator validation

- Each cell is linearly detrended and multiplied by a periodic Hann taper (SciPy window setting `sym=False`) before a one-sided density periodogram is computed at the native sampling rate. The transform length is 4000 samples at 1000 Hz and 2000 samples at 500 Hz.
- Seven synthetic tests verify identical window length, taper, detrending, FFT length, frequency masks, aggregation, normalization and band integration across all three sampling designs at 500 and 1000 Hz. Repeated identical waveform cells produce equal estimates, and an independent white-noise ensemble shows no systematic shift in mean relative alpha. A nonstationary synthetic example demonstrates sensitivity to timestamp placement.
- These tests validate the specified software behavior; they do not establish geometry-invariant rank reproducibility for all stationary processes.

## 5. Bootstrap and numerical conventions

- The registration specified 5000 participant-bootstrap resamples and two-sided 95% percentile intervals. The bootstrap seed streams, reuse of the primary index matrix for the secondary signed shift, and the linear percentile-interpolation convention were documented as post-registration implementation details.
- Seeds are the first 64 bits of the SHA-256 digest of `OSF-nh4d8|participant_bootstrap|<analysis_label>`:
  - primary analysis: 15066513651140358874;
  - complete-support sensitivity: 6846413471351983530;
  - replication: 7728558626112585725.
- The same 5000 × 531 index matrix is used for the primary contrast and the participant-level signed shift in logit relative alpha. Within a resample, one participant index vector applies to all geometries, draws and samples.
- Intervals are the 2.5th and 97.5th percentiles with NumPy `method=linear`. Standard Spearman midranks handle ties. Correlations are transformed as z = atanh(ρ) without clipping, and relative alpha is logit-transformed without epsilon clipping. Any undefined or nonfinite quantity would have been reported rather than repaired; none occurred.
- Participant-bootstrap intervals represent participant-level sampling uncertainty conditional on the 200-draw schedule. The Monte Carlo standard error (MCSE) is the sample standard deviation of the 200 draw-specific Fisher-z contrasts divided by √200; the two are not combined.

## 6. Full-precision preregistered results

| Quantity | Primary | Complete support | Replication |
|---|---:|---:|---:|
| N | 531 | 516 | 59 |
| Mean Fisher z, contiguous | 1.6675487231448323 | 1.6754351068025801 | 1.1402036957241630 |
| Mean Fisher z, distributed 5 × 8 s | 1.9272565267319925 | – | 1.4767525005299382 |
| Mean Fisher z, distributed 10 × 4 s | 1.9806160050804218 | 1.9798925150863982 | 1.5833847521227518 |
| Back-transformed ρ, contiguous | 0.9312268566348489 | 0.9322666410908824 | 0.8144826727847853 |
| Back-transformed ρ, distributed 5 × 8 s | 0.9585110656428035 | – | 0.9008577667651775 |
| Back-transformed ρ, distributed 10 × 4 s | 0.9626321842587631 | 0.9625790870545489 | 0.9191288510480916 |
| Δz, 10 × 4 s minus contiguous | 0.3130672819355893 | 0.3044574082838180 | 0.4431810563985887 |
| 95% interval, lower | 0.2517213606865131 | 0.2435423229947954 | 0.2385572841994933 |
| 95% interval, upper | 0.3661860943222055 | 0.3546145733273337 | 0.5902983787001105 |
| MCSE | 0.0040212150775266 | 0.0040660371569642 | 0.0146146369755566 |

Secondary signed shift in logit relative alpha (10 × 4 s minus contiguous; each participant contributes 400 matched differences): mean 0.005843650550668098, 95% interval [0.0015860037194996778, 0.01012751613724536]. Full precision is given for checking only.

## 7. Result file and verification scope

- `results/preregistered/scientific_results.npz` contains participant marker arrays, draw-specific Spearman correlations and Fisher transformations, deterministic bootstrap distributions, participant ordering, the complete-support schedule subset and the replication schedule. SHA-256: `f0b0ec15935aa00d8120150e7eea01e59d50e3fb45a83ad63ce6d1aa5f66ae78`.
- The result verifier reconstructs participant ordering, draw-specific correlations and transformations, signed logit-shift values and complete-support subsetting. Its full-bootstrap option reconstructs all three empirical bootstrap distributions. Raw-data quality control and reconstruction of the replication schedule are checked by separate input-verification commands. Verification from saved arrays alone does not establish raw-data quality control.

## 8. Exploratory temporal-order randomization (specified 20 September 2026)

- Uses the 516-participant complete-support sample and its original contiguous and distributed 10 × 4-s schedules. For each participant and randomization, a uniformly random permutation reassigns all 42 cell PSD vectors to cell positions; channels and frequencies move together. The same permutation applies to both designs, both samples and all 200 draws; participants receive independent permutations and values never move between participants.
- For efficient computation, posterior ROI-averaged alpha-band and total-band integrals are cached per cell. Because channel averaging, PSD averaging and trapezoidal integration are linear, a sample estimate equals the sum of selected alpha-band powers divided by the sum of selected total-band powers. This is algebraically equivalent to the registered PSD aggregation and does not average cellwise ratios. Before randomization, the compact calculation reproduced all 412,800 original complete-support sample estimates and all 400 draw-specific correlations within an absolute tolerance of 1e-12.
- Cells are permuted without replacement, so this finite-set reference is not an independently generated stationary process. No confirmatory p value was calculated.
- Summary over 1000 randomizations: mean contrast −0.0009, SD 0.0214, central 95% range [−0.0438, 0.0405], Monte Carlo SE of the mean 0.000677. Mean back-transformed reproducibility 0.9594 for both designs.
- Randomization and simulation use NumPy PCG64. For zero-based repetition j, the seed is the unsigned big-endian integer of the first eight SHA-256 bytes of `temporal-sampling-exploratory-2026-09-20-v1|<analysis>|<j>`, with analysis label `order` or `simulation`. Participant permutations are generated in the retained participant order.

## 9. Exploratory cell-band-power simulation (specified 20 September 2026)

- Simulation draws occur in the order Z, B, initial e, the 516 × 41 innovation matrix and the 516 × 42 total-power matrix. The same variates are reused across scenarios within a repetition; only the model parameters change.
- Exact stationary initialization of e(i,0) avoids a startup transient. Powers are in arbitrary units.
- The Monte Carlo SE of a scenario mean is the SD across its 500 cohort contrasts divided by √500 (not the within-cohort 200-draw MCSE of the preregistered analysis).
- Undefined correlations or transformations trigger an error rather than clipping, omission or replacement.

## 10. Boundary-annotation sensitivity (rule fixed 5 October 2026)

- Event sidecars specify onset units in milliseconds; the common first onset of 1 therefore corresponds to 0.001 s.
- Additional onsets, in seconds: `sub-114` 1.481; `sub-206` 25.041 and 44.001; `sub-319` 17.201; `sub-493` 65.721; `sub-554` 121.281. The onset for `sub-114` fell before the analysis interval; all others fell in cells already ineligible under time-domain quality control. None of those cells was selected in any geometry, draw or sample.
- These facts establish which annotated locations were omitted; they do not establish uninterrupted acquisition or account for time removed before export. The available EDF headers and event documentation do not resolve the physical meaning of the additional annotations.
- Rule: exclude all five included participants with multiple boundary annotations (N = 526), using existing measurements and all 200 paired-draw schedules. 5000 participant-bootstrap resamples, linear percentile interpolation, seed 7932202023600012997. The rule and seed were recorded in the code package on 5 October 2026.

## 11. Additional measurement analyses (plan recorded 1 November 2026)

- All new intervals use 5000 participant-bootstrap resamples and linear percentile interpolation, from separate deterministic bootstrap streams. An interval recomputed for the complete-support reference can therefore differ slightly from the preregistered interval; the point estimate is unchanged. Spearman ranks are recomputed within each resample, including ties introduced by duplicated participants. Undefined correlations are not clipped or replaced. Intervals are exploratory and unadjusted for multiple comparisons.
- Post-battery metadata screen: posterior and mastoid channels present, continuous-recording metadata, at least 180 s of recording, boundary-only event types (counts and onsets not used, as in the implemented primary screen; the initial wording of the November plan allowed only an initial boundary event). Failures: `sub-149` (179-s recording); `sub-170`, `sub-317`, `sub-535` (other event types). The remaining 527 EDF files were obtained from the same versioned release with complete file-size and SHA-256 checks.
- Draw failures after quality screening: `sub-011` (distributed 5 × 8 s), `sub-481` and `sub-598` (contiguous). The common pre/post cohort comprised 524 participants.
- Post-battery schedules use the construction rules of Section 3 with the independent dataset seed label `ds005385-post-followup-2026-11-01`.
- Annotation-exclusion sensitivity for pre/post (N = 513) excluded the union of participants with additional boundary annotations in either recording: `sub-114`, `sub-206`, `sub-254`, `sub-314`, `sub-319`, `sub-471`, `sub-486`, `sub-493`, `sub-554`, `sub-569`, `sub-571`.
- Re-pairing: assignments depend only on temporal positions; deterministic secondary costs break ties without changing the primary minimum or maximum. Rank correlations are recomputed with the reassigned pairs, retaining the original draw positions of A samples.
- Spectral normalization: relative alpha is the ratio of aggregated powers, not an average of cell-level ratios.

## 12. Software

The scientific analysis ran under CPython 3.14.3, NumPy 2.3.4, SciPy 1.16.3 and MNE 1.10.2. `release_manifest.json` records file sizes and checksums for this edition.

## 13. Muscle-artifact sensitivity (plan recorded 6 November 2026)

The exploratory plan is in `artifact_sensitivity/ANALYSIS_PLAN.md` and its matching JSON file. `PLAN_RECORD.json` gives their checksums and local UTC timestamps.

For each four-second cell, 55-95 Hz log power was calculated in O1, Oz and O2 after reference to mean(TP9, TP10). Within each participant and channel, the median and 1.4826 times the median absolute deviation across originally eligible cells defined a robust z score. A cell was removed if any channel exceeded z = 3. The alpha and denominator bands did not enter this screen.

Schedules used the original construction rules and the dataset labels `ds005385-artifact-emg-2026-11` and `ds004148-artifact-emg-2026-11`. Each retained participant supported all 200 paired draws for C, F5 and F10. Bootstrap labels were `artifact_emg_ds005385` (seed 1734386567627607253) and `artifact_emg_ds004148` (seed 7312250668946274530). Each cohort used 5000 shared participant resamples for the strict and original schedules, with Spearman ranks recalculated in each resample. Intervals are unadjusted 95% linear-percentile intervals.

The screen removed 723 of 22,280 originally eligible cells in ds005385 and 92 of 2468 in ds004148. Retained samples were 477 of 531 and 51 of 59 participants.

A serial feasibility run was restarted with six independent participant processes. This changed only execution, before any new relative-alpha or rank-reproducibility outcome was calculated or inspected. The plan, metric, threshold, seeds, cohorts and inference rules stayed the same.

### ds005385

Excluded participants: `sub-011`, `sub-022`, `sub-029`, `sub-034`, `sub-068`, `sub-086`, `sub-095`, `sub-101`, `sub-110`, `sub-116`, `sub-122`, `sub-124`, `sub-133`, `sub-134`, `sub-139`, `sub-162`, `sub-170`, `sub-173`, `sub-178`, `sub-196`, `sub-201`, `sub-232`, `sub-235`, `sub-246`, `sub-247`, `sub-252`, `sub-256`, `sub-260`, `sub-264`, `sub-282`, `sub-319`, `sub-328`, `sub-353`, `sub-366`, `sub-374`, `sub-391`, `sub-400`, `sub-404`, `sub-413`, `sub-416`, `sub-417`, `sub-418`, `sub-421`, `sub-450`, `sub-464`, `sub-512`, `sub-520`, `sub-525`, `sub-529`, `sub-530`, `sub-536`, `sub-560`, `sub-569`, `sub-589`.

Flagged-cell count versus original participant-mean relative alpha: Spearman rho = 0.06839445372010756 in the original cohort (N = 531) and 0.042347933848137106 in the retained subset (N = 477).

Schedule NPZ checksum (SHA-256): `95278f760daf6d7189ce9a62c09123368e7ce22d68084a215e87a94d7df606f3`. Schedule array checksum (uint8, C order): `55690d357d28b6e7883442ac9d520555eec1e5610ed0887f0b67638d5428d03d`.

| Quantity | Strict mask | Original schedules, same participants |
|---|---:|---:|
| Back-transformed rho, C | 0.9400538189575224 | 0.9379535867945366 |
| Back-transformed rho, F5 | 0.9622676774805188 | 0.9611064758117842 |
| Back-transformed rho, F10 | 0.9661003863944428 | 0.9652040671037024 |
| Delta z | 0.2916899640251719 | 0.296172861315275 |
| MCSE | 0.004369794758102568 | 0.004095671298512163 |
| 95% interval, lower | 0.22478128795593577 | 0.2352915956008783 |
| 95% interval, upper | 0.34733548810886333 | 0.34870555553185234 |

Paired change (strict minus original): -0.004482897290103094; 95% interval [-0.03493783574233612, 0.022913750592715384]; MCSE 0.005908898388967398.

### ds004148

Excluded participants: `sub-03`, `sub-08`, `sub-20`, `sub-26`, `sub-29`, `sub-31`, `sub-49`, `sub-55`.

Flagged-cell count versus original participant-mean relative alpha: Spearman rho = -0.10413706918388042 in the original cohort (N = 59) and 0.09849869019578966 in the retained subset (N = 51).

Schedule NPZ checksum (SHA-256): `119f4aaa764b204a06a6e5232e386178c390dd1f1e2525c6fc1c77b9c64a7fb9`. Schedule array checksum (uint8, C order): `d3d2565bebf22182fdacceb95f43dc7889cc8578e374306af6157c69b2196139`.

| Quantity | Strict mask | Original schedules, same participants |
|---|---:|---:|
| Back-transformed rho, C | 0.7732677636506969 | 0.7779334184884411 |
| Back-transformed rho, F5 | 0.8913044379115235 | 0.8890117551738533 |
| Back-transformed rho, F10 | 0.9129551032774244 | 0.9098013937198537 |
| Delta z | 0.5165854440725451 | 0.48625542172527864 |
| MCSE | 0.014036303359950056 | 0.01479120038552127 |
| 95% interval, lower | 0.2719252645456574 | 0.27720294756139224 |
| 95% interval, upper | 0.6989891042610802 | 0.6318437817558614 |

Paired change (strict minus original): 0.030330022347266505; 95% interval [-0.05509917488862059, 0.10686130043793286]; MCSE 0.01997948707810524.

### Acquisition filters

The EDF prefilter fields contain NaN, and the BrainVision headers omit the hardware low-pass setting. These findings were recorded before the analysis. Du (2027), *Neurobiology of Aging* 170:26-38, section 3.2, reports acquisition with a 250 Hz online low-pass filter for ds005385 ([doi:10.1016/j.neurobiolaging.2026.10.003](https://doi.org/10.1016/j.neurobiolaging.2026.10.003)). This cutoff is above the 55-95 Hz screening band. The ds004148 hardware low-pass setting remains unreported. The screen uses high-frequency power as an EMG proxy.

The EDF headers and sidecars do not state the hardware low-pass cutoff. The ds005385 data descriptor (Getzmann et al. 2024, Scientific Data 11:988, doi:10.1038/s41597-024-03797-w) reports a 250-Hz online low-pass; the cutoff for ds004148 is not documented.

Both strict contrasts had intervals above zero. Both paired-change intervals included zero. The stricter screen did not show a clear change in the contrast. This exploratory analysis makes no equivalence claim.

## 14. Duration by geometry (exploratory, 7 November 2026)

The plan is in `duration_geometry/ANALYSIS_PLAN.md` and its matching JSON file. `PLAN_RECORD.json` gives their checksums and local UTC timestamp. The plan includes this provenance note: "A related 40-s comparison against an all-cell post-battery reference, using the original A/B schedules, was calculated during an independent audit before this plan. This analysis uses new single-sample draws, fixes all durations, designs and contrasts in advance, and reports every result."

The question is how well one pre-battery sample ranks participants by their level in the separate post-battery recording, and how this depends on sample duration and placement. No raw EEG was reprocessed. Pre-battery cell band powers come from `results/exploratory_2026-09-20/cell_band_powers.npz` (516 participants, all 42 cells eligible). Post-battery cell band powers and masks come from `additional/results/post_prepared.npz`. The cohort is the intersection of the two participant lists (N = 509).

- Reference: one post-battery relative alpha per participant, the sum of 8-13 Hz power over all eligible post cells divided by the sum of 1-30 Hz power over the same cells. It is the same for every design.
- Sample: one pre-battery sample per participant and draw, computed the same way from the selected cells. There is no A/B pair.
- Durations: 5, 10 and 20 four-second cells (20, 40 and 80 s).
- Contiguous: one block of n consecutive cells; the start is uniform over the 42 - n + 1 possible starts.
- Distributed: the 42 cells are split into n consecutive sections with the existing rule (divmod; the extra cells go to the first sections), and one cell is chosen uniformly from each section.
- Draws: 200 per participant and design. Seed: first eight bytes (big-endian) of SHA-256 of `duration-geometry-2026-11-07|<participant>|<design>|<draw>`, NumPy `default_rng` (PCG64). Design labels: `contiguous_5`, `contiguous_10`, `contiguous_20`, `distributed_5`, `distributed_10`, `distributed_20`.
- Outcome: Spearman correlation (average ranks) across participants between sample and reference in each draw. Fisher z values are averaged over draws and back-transformed to rho tilde. The mean raw correlation is also reported.
- Inference: 5000 participant-bootstrap resamples shared by all designs and contrasts, label `duration_geometry_bootstrap` (seed string `OSF-nh4d8|participant_bootstrap|duration_geometry_bootstrap`, seed 6784921211910029168). Ranks are recomputed in each resample, with average ranks for duplicated participants. Intervals are 95% linear-percentile intervals. No multiplicity correction is applied.
- MCSE: SD of the 200 draw-level z values divided by sqrt(200); for a contrast, the square root of the summed squared design MCSE values, because the design seed streams are independent.

Bootstrap index checksum (SHA-256, little-endian int64, C order): `c8e2086d98015126f032411440c648d1eae568fd5bcda63d5e8149b8e0e8b73e`.

| Design | Rho tilde | 95% interval, lower | 95% interval, upper | Mean raw rho | MCSE (z) |
|---|---:|---:|---:|---:|---:|
| Contiguous 20 s | 0.8252496333337777 | 0.793954102870571 | 0.8525784525951537 | 0.8250587998700556 | 0.0019110912577329502 |
| Contiguous 40 s | 0.8426724862579186 | 0.810942355087167 | 0.8691586752549569 | 0.8425650081761275 | 0.0014865238953359824 |
| Contiguous 80 s | 0.8552074192224749 | 0.8231682182862349 | 0.8809462905493901 | 0.8551566449576907 | 0.0010534940825213572 |
| Distributed 20 s | 0.8362796262323419 | 0.8047938183986874 | 0.8628731980987382 | 0.8361246365786387 | 0.0017595957428391785 |
| Distributed 40 s | 0.8524409643396987 | 0.8212624622047177 | 0.8779079610084393 | 0.8523786807360971 | 0.0011591111111594683 |
| Distributed 80 s | 0.8604720086743501 | 0.8288139716701275 | 0.8857199710036482 | 0.860449981845994 | 0.0007044033546281307 |

| Contrast | Delta z | 95% interval, lower | 95% interval, upper | Raw difference | MCSE |
|---|---:|---:|---:|---:|---:|
| Distributed 20 s minus contiguous 20 s | 0.035611682078386275 | 0.02292839142303179 | 0.04903635584725899 | 0.011065836708583077 | 0.00259777731409008 |
| Distributed 40 s minus contiguous 40 s | 0.034694311688885815 | 0.02104331092002084 | 0.05039834782875511 | 0.009813672559969677 | 0.001885017734510262 |
| Distributed 80 s minus contiguous 80 s | 0.019935361326290035 | 0.004019940874006767 | 0.03602064486305922 | 0.005293336888303246 | 0.0012672939153641038 |
| Contiguous 40 s minus contiguous 20 s | 0.057264482503370795 | 0.04337728437395873 | 0.07078998923779481 | 0.017506208306071858 | 0.0024211615160472403 |
| Contiguous 80 s minus contiguous 40 s | 0.0449035082278344 | 0.03184970660741184 | 0.05807903601919021 | 0.012591636781563276 | 0.0018219777916627796 |
| Distributed 40 s minus distributed 20 s | 0.056347112113870335 | 0.04531777658047288 | 0.06740264137529366 | 0.016254044157458458 | 0.002107063299056556 |
| Distributed 80 s minus distributed 40 s | 0.03014455786523862 | 0.02189629202697902 | 0.038609812102433534 | 0.008071301109896845 | 0.0013563637616895777 |
| Distributed 20 s minus contiguous 40 s | -0.02165280042498452 | -0.0361751684872402 | -0.0064790278555450645 | -0.006440371597488781 | 0.0023034605422326258 |
| Distributed 40 s minus contiguous 80 s | -0.010209196538948584 | -0.02601288422962045 | 0.0058746691807118915 | -0.0027779642215935985 | 0.0015663295789586729 |

`duration_geometry/results_summary.json` also gives the intervals of the raw differences. The outcome is agreement with a measurement from a separate recording. It is not a biological true level, and it includes any change in state after the cognitive battery. Overlapping intervals are not read as equality between designs; only the paired differences above are reported.

## 15. Scalp topography (exploratory)

The plan is in `topography/ANALYSIS_PLAN.md` and its matching JSON file. `PLAN_RECORD.json` gives their checksums and local UTC timestamp. The analysis is exploratory. It asks whether the difference between distributed 10 x 4 s and contiguous 40 s sampling of relative alpha is limited to O1, Oz and O2 or is also seen at other scalp channels. It uses ds005385 only. The 5 x 8 s geometry, other features and ds004148 are not part of it.

- Channels: the 62 EEG channels other than TP9 and TP10, in this order: Fp1, Fp2, F7, F3, Fz, F4, F8, FC5, FC1, FC2, FC6, T7, C3, Cz, C4, T8, CP5, CP1, CP2, CP6, P7, P3, Pz, P4, P8, PO9, O1, Oz, O2, PO10, AF7, AF3, AF4, AF8, F5, F1, F2, F6, FT9, FT7, FC3, FC4, FT8, FT10, C5, C1, C2, C6, TP7, CP3, CPz, CP4, TP8, P5, P1, P2, P6, PO7, PO3, POz, PO4, PO8. Each channel is referenced to the mean of TP9 and TP10, as in the main analysis. Channels are selected by label.
- Processing: unchanged from the main analysis. Support [8, 176) s; 42 non-overlapping 4-s cells; the registered periodogram (linear detrend, periodic Hann window, one-sided density, native sampling rate, 0.25-Hz grid); trapezoidal band power over 8-13 Hz and 1-30 Hz, endpoints included. There is no filtering, resampling, ICA or channel interpolation. The existing cell masks and A/B schedules are reused. They were defined from the posterior channels and the reference, so the other channels were not screened separately.
- Marker: per channel, sample and draw, the mean 8-13 Hz power over the ten cells divided by the mean 1-30 Hz power over the same cells.
- Within comparison: 531 participants. Sample A against sample B within the pre-battery recording, with the original contiguous (C) and distributed 10 x 4 s (F10) schedules.
- Between comparison: 524 participants. Pre-battery sample A (original schedule) against post-battery sample A (post-battery schedule of the additional analyses), as in the published between-recording comparison.
- Statistics: per channel, comparison and geometry, the Spearman correlation (average ranks) across participants in each of 200 draws. Fisher z values are averaged over draws, and rho tilde = tanh(mean z). Delta z = mean z(F10) - mean z(C). MCSE = SD of the 200 draw-level contrasts divided by sqrt(200).
- Bootstrap: the existing 5000 participant index matrices, shared by all channels. Within: the primary-analysis matrix, `generate_bootstrap_indices(531, 'primary_ds005385')` in `scripts/preregistered_analysis.py`. Between: `bootstrap_indices` in `additional/results/pre_post_cross_inference.npz`. Ranks are recomputed in every resample, with average ranks for duplicated participants. Intervals are 95% linear-percentile intervals without multiplicity correction. For the channel mean, the mean over the 62 channels is taken in each resample.

Bootstrap index checksums (SHA-256, little-endian int64, C order): within `aac06aa7fed9b0f0eb5b5d1bad1e6e87d43db823bf7026a10e3c1803707f546e`; between `b964eaa06eb1991e196e37a09de0cd740a25b4899881dd7b9f5678b7a496f565`.

Validation came before any channel-level outcome. The mean of the O1, Oz and O2 band powers reproduced the stored posterior cell band powers (largest relative difference 4.2e-14 pre-battery and 5.8e-14 post-battery). It also reproduced the stored posterior markers (largest absolute difference 1.9e-15 and 2.5e-15) and the stored Fisher z and bootstrap values of the main and between-recording analyses (at most 3.6e-15). `topography/VALIDATION.json` gives all values and limits.

One plan deviation was recorded before any channel-level outcome was calculated (`PLAN_DEVIATION.md`, checksum in `DEVIATION_RECORD.json`). In sub-075, both recordings list the same 64 EEG labels, but Fp1 and Fp2 are in swapped positions in the file. The plan said that a reordered channel stops the analysis. The change accepts the same 64 labels in another order, because channels are selected by label. A missing, duplicated or extra label still stops the analysis.

| Summary | Within (N = 531) | Between (N = 524) |
|---|---:|---:|
| Mean of the 62 channel Delta z | 0.20736682716666466 | 0.06095502841333361 |
| Mean, 95% interval | 0.17362400016253038 to 0.23825121438089134 | 0.045059206392608846 to 0.07657915538468278 |
| Channels with Delta z above zero | 62 | 62 |
| Channels with the interval wholly above zero | 62 | 62 |
| Channels with the interval wholly below zero | 0 | 0 |
| Smallest channel Delta z | FT10: 0.13028079319524077 | FT10: 0.03220619314493256 |
| Smallest, 95% interval | 0.07918283845453165 to 0.18116937073469222 | 0.009925661840129076 to 0.05382820838236971 |
| Largest channel Delta z | O2: 0.2856951613711969 | P2: 0.08381855141570993 |
| Largest, 95% interval | 0.22573774021251924 to 0.3389709969798926 | 0.06433442579881152 to 0.10359291930619348 |
| Channel rho tilde range, contiguous | 0.860656131108202 to 0.938151839245768 | 0.7134527033209874 to 0.8214709190764268 |
| Channel rho tilde range, distributed | 0.8982496522917921 to 0.9599800129951801 | 0.7296293104743569 to 0.8407456080623356 |
| Spearman correlation across channels of contiguous rho tilde with Delta z (descriptive) | 0.8562614892599029 | 0.6933343406109138 |
| Posterior O1/Oz/O2 marker of the main analysis, Delta z | 0.3130672819355895 | 0.07555700290023637 |

Per-channel values, with rho tilde and its interval for each geometry, are in `topography/channel_results.csv` and `topography/results_summary.json`. Channel intervals are descriptive, and no channel is called significant. The within-recording map carries the same timing confound as the main within-recording comparison; the between-recording map does not. The single-channel O1, Oz and O2 values differ from the posterior row above, because the main marker averages the band powers of the three channels before the ratio is taken.

Band powers were extracted on Linux with Python 3.10.12, NumPy 2.2.6, SciPy 1.15.3 and MNE 1.10.2. Validation, inference and the report ran on Linux with Python 3.13.18, NumPy 2.5.3 and SciPy 1.18.1. `run_log.txt` notes three script edits during the run; none changed the band-power code. On Windows with Python 3.14.3, NumPy 2.3.4, SciPy 1.16.3 and MNE 1.10.2, a rerun of validation, inference and the report from the saved band powers matched every saved inference array and summary value (largest absolute difference 8.9e-16). Band powers recomputed from the EDF files of 10 pre-battery and 10 post-battery participants, sub-075 included, matched the saved arrays (largest relative difference 1.3e-13).
