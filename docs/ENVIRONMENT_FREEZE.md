# Original Bayesian software environment — archived

The original Round-3 clean Model D NumPyro runtime is no longer available as an executing container, **but its contemporaneous environment capture has been recovered and archived**:

- `environment-python.txt` — original Python version (`Python 3.12.3`)
- `requirements-lock.txt` — full `python -m pip freeze`, 142 packages
- `PROVENANCE.md` — original environment provenance, fit timestamps, software and CPU/JAX execution settings

According to the accompanying provenance, these files were produced in the same runtime as the 18 August 2026 clean Model D fits, and package installations/updates had not occurred after the fits. The two text files were exported on 21 August 2026; they are not a reconstruction from the current session.

Core packages: `numpyro==0.21.0`, `jax==0.11.1`, `jaxlib==0.11.1`, `arviz==1.3.0`, `numpy==2.4.4`, `pandas==3.0.2`, `scipy==1.17.1`, `matplotlib==3.10.8`.

`environment-details.json` was **not** captured in August and is intentionally not represented as if original. The platform, CPU-only computation, four sequential chains, default JAX float32 and fixed seeds are documented in `PROVENANCE.md`.

`requirements-minimal.txt` is an unpinned convenience list. For the actual recorded software versions, use `requirements-lock.txt`.

No refit is required merely to preserve and disclose the existing original environment record. Bit-for-bit reproduction can also depend on the operating system and numerical backend, as described in `PROVENANCE.md`.

## Inexpensive descriptive reproduction

Run `python 01_build_datasets.py` and `python 02_stage1_reproduction.py` from `code/`. The expected comparator-classification result is 189/798 = 23.7% (final one-decimal threshold convention).

## Future reconstruction

If creating a new runtime, label it **reconstructed**, not original; never replace the dated original capture files. Model D code is in `code/bayes_model.py` and related scripts, with saved outputs under `results/key/`.
