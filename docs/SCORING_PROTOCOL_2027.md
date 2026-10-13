# Pre-release registered scoring protocol for NSS 2027 (13 October 2026)

This document describes intended scoring before 2027 NSS results are available. It is not a formal registry entry until the timestamped public Zenodo version exists.

## Primary Model D forecasts
`predictions/2027_frozen/forecast_2027_sector_M4.csv` (868 2026-published UKPRN × theme cells). These are point and probabilistic forecasts for the *observed* 2027 OfS benchmark difference (`D_2027`, percentage points) in Engineering and Technology, all undergraduate students, themes 1–7.

## Matching
Join to the official 2027 OfS release by numeric UKPRN and theme number (not provider name), restricting to cells pre-listed in the forecast. Use the official 2027 observed `Difference (ppt)` for each matched cell, retaining published decimal precision. Do not score unpublished/suppressed cells. Tabulate missing cells by 2026 headcount and prior benchmark gap. Score the whole matched set and separately headcount <50, 50–99, 100–199, and >=200 based on archived 2026 responses; do not change these bands.

## Primary point scores
Compute RMSE and MAE of `pred27_mean` against 2027 observed `D`. Compare, on exactly the same matched rows, with baseline 1 = published observed 2026 `D_2026`; baseline 2 = simple 2023–2026 arithmetic mean of the provider×theme published gaps (compute using archived processed data, requiring >=1 of the four waves). Predefine incremental RMSE improvement = RMSE(baseline) − RMSE(Model D).

## Primary probabilistic scores
* 90% predictive interval coverage: share of observed 2027 `D` lying in `[pred27_lo90,pred27_hi90]`, endpoints inclusive.
* Mean predictive 90% interval width: mean(`pred27_hi90` − `pred27_lo90`).
* Below/above binary Brier scores for the events `D_2027 < -2.5` and `D_2027 > 2.5` using `P27_below` and `P27_above`, respectively. Combine as mean of two Brier scores when an overall number is desired; also report each separately. Treat exactly ±2.5 as in line. Apply the published one-decimal materiality convention consistently.
* Below/above Bernoulli log loss (clip archived probabilities to [0.001,0.999] for finite scoring; report the clipping rule).
* Do not claim CRPS if distribution quantile draws are not archived; 90% bounds and probabilities do not uniquely determine CRPS.

## Interpretation and reporting
Report provider-cluster bootstrap 90% intervals for paired RMSE differences where sample size allows. Report calibration separately for below and above with pre-defined probability bins [0,.2), [.2,.4), [.4,.6), [.6,.8), [.8,1]. Report the absolute performance and no superiority claims if uncertainty overlaps zero.

## Forecast-model identity
The similarly named `forecast_2027_sector_provider_theme.csv` was made using the earlier **Model I** and contains scenarios A/B. Its values do not coincide with the registered Model D forecast. It may be scored as an explicitly secondary, separately labelled model comparison but must not replace the preselected primary Model D forecast after seeing outcomes.

## Missingness and contextual limits
These are forecasts for post-redesign public NSS provider-theme gap estimates; the OfS benchmark itself can change, as can provider coding and publication eligibility. Do not retrospectively remap institutions without documenting all decisions. There is no claim that successful forecast scoring establishes structural intervention effects.
