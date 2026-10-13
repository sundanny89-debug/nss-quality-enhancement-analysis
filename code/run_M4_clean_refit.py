"""
Final clean M4 diagnostic refit.
Run from the code/ directory of NSS_bayes_analysis_v2_code_and_outputs in the
same NumPyro/JAX environment used for the original Bayesian rounds.

This is a diagnostic refit ONLY: same M4 structure, same priors, no tuning to results.
Outputs: M4_clean_B_hyper.csv, M4_clean_A_hyper.csv, M4_clean_diagnostics.json.
"""
import pandas as pd, json, time
from bayes_model import build_index, fit_v4, diagnostics

prov = pd.read_csv('../data/sector_cah10_provider_theme.csv')
results = {}

def run(label, df, seed):
    idx = build_index(df, 'UKPRN')
    t0 = time.time()
    mcmc, _ = fit_v4(
        idx,
        num_warmup=3000,
        num_samples=2000,
        num_chains=4,
        seed=seed,
        target_accept=0.99,
    )
    summ, diag, _ = diagnostics(
        mcmc,
        params=('mu','sd_gamma','sd_u','sd_c','rho','lam','kappa','sd_v_q')
    )
    summ.to_csv(f'../out/{label}_hyper.csv')
    results[label] = {'diagnostics': diag, 'elapsed_s': time.time()-t0}

run('M4_clean_B', prov, 9901)
run('M4_clean_A', prov[prov.Year <= 2025].copy(), 9902)
with open('../out/M4_clean_diagnostics.json','w') as f:
    json.dump(results, f, indent=2)
print(json.dumps(results, indent=2))
