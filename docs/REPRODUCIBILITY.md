# Reproducibility guide

## Fastest route

The repository already includes the final harmonised datasets in `data/processed/`. To reproduce the Bayesian analyses, run the scripts from the `code/` directory in the same order used in the project.

The core final Bayesian workflow is:

1. `03_fit_holdout.py` / holdout scripts — retrospective temporal validation.
2. `09a_final_modelB.py` and associated final-model scripts — pre-specified independent-error model.
3. `10_model_v4.py`, `10a_model_v4_full.py`, `10b_model_v4_holdout.py` — final dependence-aware model.
4. `run_M4_clean_refit.py` / `run_clean.sh` — high-target-accept diagnostic refit.
5. `12_M4a_ablation.py` and `12b_M4a_modelA.py` — dependence-only ablation.
6. `11_m4_figures_forecast.py` — selected final outputs and 2027 predictions.

The original scripts assumed a project working directory; shell-script paths and the raw-data path in `01_build_datasets.py` have been made repository-relative for this archive. No substantive modelling code or priors were changed.

## Rebuilding processed data

The compact public extracts used by `01_build_datasets.py` are in `data/raw_public_extracts/`.

From `code/`:

```bash
python 01_build_datasets.py
```

This writes the harmonised files to the repository-root `data/` folder used by the model scripts.

## Exact software environment

Before the first public release, run the following inside the same Python environment used for the final clean Bayesian fit:

```bash
python --version > ../environment-python.txt
python -m pip freeze > ../requirements-lock.txt
```

Add both files to the repository before creating the immutable GitHub/Zenodo release.

## Computational note

The full NUTS fits are the computationally heavy component. They are not required merely to inspect or score the frozen posterior outputs. Final diagnostic summaries and posterior-derived tables are included under `results/key/`.

## Verification

`SHA256SUMS.txt` records file checksums. The 2027 forecasts should be scored in the future against exactly the files in `predictions/2027_frozen/` from the first public Zenodo release.

## Descriptive smoke tests

After installing pandas and numpy, run from `code/`:

```bash
python 01_build_datasets.py
python 02_stage1_reproduction.py
```

Outputs are written to `../data/` and `../out/`. These smoke tests do not replace full Bayesian re-fitting.
