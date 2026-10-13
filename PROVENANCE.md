# Environment provenance

## How these files were produced

Exported with the exact commands requested, run inside the same container image and filesystem in which
the final Model D ("clean M4") NumPyro fits were executed:

```
python --version > environment-python.txt
python -m pip freeze > requirements-lock.txt
```

## Evidence that this is the runtime in which the fits ran

| Check | Result |
|---|---|
| `numpyro` install timestamp (`dist-info` mtime) | 2026-08-18 08:53 UTC |
| Final Model D fits executed | 2026-08-18 15:45–16:06 UTC (`out/M4_clean_diagnostics.json`, `out/status_clean.txt`) |
| pip packages installed or updated after the fits finished | none (`find dist-packages -maxdepth 1 -name '*.dist-info' -newermt '2026-08-18 16:06'` returns nothing) |
| Fit outputs still present on this filesystem | `out/M4_clean_B_hyper.csv` (2026-08-18 15:56), `out/M4_clean_A_hyper.csv` and `out/M4_clean_diagnostics.json` (2026-08-18 16:05) |

The versions below are therefore read from the live interpreter that produced the posterior summaries in the
manuscript, not reconstructed from memory or from a different machine.

## Key versions

Python 3.12.3 (Ubuntu 24.04, x86_64, CPU-only; no GPU/TPU was used)

```
numpyro==0.21.0
jax==0.11.1
jaxlib==0.11.1
arviz==1.3.0
numpy==2.4.4
scipy==1.17.1
pandas==3.0.2
matplotlib==3.10.8
ml_dtypes==0.6.0
opt_einsum==3.4.0
multipledispatch==1.0.0
tqdm==4.70.0
```

The complete freeze (142 packages, including transitive dependencies of the notebook/OS layer that the analysis
does not import) is in `requirements-lock.txt`. `requirements-minimal.txt` lists only the packages the analysis
code actually imports, pinned to the same versions.

## Reproduction notes

* Sampling used `chain_method='sequential'` on a single CPU; four chains run one after another. Wall time for the
  final full-data fit was approximately 11 minutes and for the 2023–2025 fit approximately 9 minutes.
* JAX defaults to float32. Results in the manuscript were produced under that default; no
  `jax.config.update("jax_enable_x64", True)` call was made.
* Seeds are fixed in the scripts (9901 for the full-data fit, 9902 for the 2023–2025 fit). Bit-for-bit
  reproduction requires the same JAX/jaxlib version and the same CPU backend; different XLA versions can
  produce numerically negligible differences in the last decimal places.
