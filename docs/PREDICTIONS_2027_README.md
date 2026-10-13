# NSS 2027 forecast freeze: primary estimand and competing model outputs

**Primary registered forecast:** `forecast_2027_sector_M4.csv` — the final dependence-aware hierarchical model (Model D/M4). It contains 868 provider × theme cells observed for 2026, keyed by UKPRN and theme.

`forecast_2027_QMUL_M4.csv` is the seven-theme QMUL subset of that primary forecast.

**Distinct legacy forecast, NOT interchangeable with primary:** `forecast_2027_sector_provider_theme.csv` was generated earlier from the independent-error Model I/M1 under two different future-shock persistence assumptions (`pred_mean` scenario A and `predB_mean` scenario B). Do **not** silently pool, compare as though identical, or switch between these after seeing 2027.

`forecast_2027_sems_unit_theme.csv` is also Model I-based public-subject proxy forecasting; not the primary sector-wide validation.

The primary Model D file uses `pred27_mean`, `pred27_lo90`, `pred27_hi90`, `P27_below`, and `P27_above`. It predicts the next *published observed* benchmark gap using the published 2026 standard deviation as the proxy for 2027 survey uncertainty. It is not a student-level forecast.

The archived primary 2027 forecast must remain unchanged. Publish this release on GitHub/Zenodo before the 2027 NSS release and cite its immutable version DOI. Corrections or later scoring should always reference the original forecast version and be published separately.

See `docs/SCORING_PROTOCOL_2027.md` for the frozen prospective scoring rules.
