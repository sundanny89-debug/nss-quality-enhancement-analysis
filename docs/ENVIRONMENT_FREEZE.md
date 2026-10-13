# Reproducible Python environment — practical instructions

The archived NumPyro model code is present at `code/bayes_model.py`, and runnable scripts at `code/run_M4_clean_refit.py` and `code/10_model_v4.py`. **Exact original package versions cannot be reconstructed from a later session**. To capture exact versions, run `python code/capture_environment.py` within the original modelling runtime. The script writes:

- `environment-python.txt`
- `requirements-lock.txt`
- `environment-details.json`

## If you no longer have the original runtime
Create a separate *reconstructed* environment and run smoke tests, but label the resulting lockfile `requirements-reconstructed.txt` and record that it is not the exact original setup. Do not claim bit-for-bit environmental identity.

## Windows PowerShell example
Open PowerShell in the extracted repository folder. Run `py -m venv .venv`, `.\.venv\Scripts\python.exe -m pip install --upgrade pip`, then `.\.venv\Scripts\python.exe -m pip install numpy pandas scipy matplotlib arviz jax numpyro`. Run `.\.venv\Scripts\python.exe code\capture_environment.py` to record that **reconstructed** environment. Rename the generated lock to `requirements-reconstructed.txt` unless this was the original Bayesian environment.

## Running the inexpensive public-data check
From repository root: `cd code`, `python 01_build_datasets.py`, `python 02_stage1_reproduction.py`. Check that reported comparator churn is 189/798 = 23.7%. The full NUTS clean refit is computationally much more expensive and is not a prerequisite to checking the frozen outputs.
