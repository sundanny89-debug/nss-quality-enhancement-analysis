# Beyond positivity — reproducibility archive

This repository supports the manuscript:

**Beyond positivity: benchmark-adjusted and reliability-aware use of the National Student Survey for quality enhancement**

The analysis develops a reproducible approach for annual National Student Survey (NSS) quality-enhancement analysis using public Office for Students (OfS) data from 2023–2026. The repository contains the processed public-data extracts, analysis code, final diagnostic outputs, and a timestamped freeze of the 2027 posterior predictive distributions.

## Status

**Pre-publication staging archive. Creator metadata is fixed as Yiwei Sun (sole author). Publish the versioned GitHub/Zenodo release only after the remaining checks in `STATUS_PRE_RELEASE.md` are complete.**

## Repository structure

- `data/raw_public_extracts/` – compact public-data extracts used by the project.
- `data/` – harmonised analysis datasets at paths used by the model scripts. `data/processed/` keeps an identical copy for provenance.
- `code/` – Python/NumPyro analysis scripts. Paths have been made repository-relative; substantive analysis/model code is unchanged.
- `results/key/` – selected final outputs supporting the paper.
- `predictions/2027_frozen/` – posterior predictive distributions generated before the 2027 NSS release.
- `docs/` – data dictionary, reproducibility notes, provenance and licensing.
- `SHA256SUMS.txt` – checksums for every archived file.

## Data source

The underlying NSS data are publicly released by the Office for Students. The official download and documentation pages should be treated as the authoritative source.

The files in `data/raw_public_extracts/` are compact extracts constructed from the public OfS releases for this analysis; they do not contain individual-level student records.

## Reproduction

See `docs/REPRODUCIBILITY.md`.

The Bayesian analyses require Python with NumPy, pandas, SciPy, Matplotlib, JAX and NumPyro. The **original final Bayesian runtime was captured on 21 August 2026** (after the 18 August clean fits), with no subsequent package changes reported in the provenance record. Its capture is preserved at repository root: `environment-python.txt` (Python 3.12.3), `requirements-lock.txt` (142 exact installed package versions), and `PROVENANCE.md` (runtime and timing evidence). The original container itself is no longer running. See `docs/ENVIRONMENT_FREEZE.md`.

## Frozen 2027 predictions

The files under `predictions/2027_frozen/` are intentionally frozen before the 2027 NSS release. They should not be overwritten after the first public Zenodo release. Later scoring of the 2027 predictions should use these exact files/checksums.

## Licensing

This is a mixed-license research compendium:

- analysis code: MIT License;
- OfS-owned public data and derived extracts: Open Government Licence v3.0, with OfS attribution;
- original documentation in this repository: CC BY 4.0.

See the licence files in the repository root and `docs/LICENSING.md`.

## Citation

The sole author has been confirmed as Yiwei Sun. After Zenodo creates the immutable DOI, cite the **version DOI** associated with this frozen release.
