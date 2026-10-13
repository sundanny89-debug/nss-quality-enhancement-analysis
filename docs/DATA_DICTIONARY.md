# Data dictionary

## Core modelling dataset: `data/processed/sector_cah10_provider_theme.csv`

Each row represents one published provider × NSS theme × year observation for the Engineering and Technology subject group.

Key variables:

- `Year` — NSS publication year.
- `UKPRN` — stable UK Provider Reference Number used as the provider key.
- `provider` — provider name as published for that year.
- `theme` — NSS theme number 1–7.
- `Question` — published NSS theme label.
- `P` — OfS positivity measure (%).
- `B` — provider-specific OfS benchmark (%).
- `D` — difference from benchmark in percentage points (`P - B`, subject to published rounding).
- `s` — OfS published standard deviation of the benchmark difference; treated as known measurement uncertainty in the hierarchical model.
- `R` — raw comparator gap, defined as provider positivity minus the corresponding UK Engineering and Technology positivity.
- `P_uk` — corresponding UK subject positivity.
- `n_resp` — published weighted response measure.
- `pop` — published population measure.
- `headcount` — publication response headcount.
- `rr` — publication response rate (%).
- `contrib` — provider contribution to its benchmark (%), where published.
- `p_below`, `p_inline`, `p_above` — published OfS materiality probabilities.

## QMUL/SEMS processed extracts

`sems_all_ug_long.csv`, `sems_all_levels_long.csv`, and `sems_theme_panel.csv` contain public OfS subject-category results used only for the illustrative QMUL/SEMS application. Public subject categories are proxies and should not be interpreted as governed internal programme populations.

## Q26–Q27

The principal question-level robustness analysis excludes Q26–Q27 because they are not constituent items of the seven common post-2023 NSS themes used in the primary framework.

## Missingness

Only publicly released aggregate observations are included. Publication availability is not random: small units are more likely to be absent under OfS publication thresholds.
