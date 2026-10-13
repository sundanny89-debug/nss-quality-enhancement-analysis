"""Optional analysis 2 (validation memo): formal correlated-error ablation.
M4a = guide-specification latent structure (mu + alpha_q[sum-to-zero] + gamma_t + u_p + v_pq + c_pt + random-walk h_pqt, no eps)
      + equicorrelated same-provider-year observation error (rho, g_pt) exactly as in M4.  No theme loadings, no theme-specific sigma_v, no kappa.
Fits: Model B (2023-26) and Model A (2023-25), 4 chains x (2000 + 2000), target 0.95 (same budget as the original G/M4 runs).
Outputs: M4a_B_hyper.csv, M4a_A_hyper.csv, M4a_summary.json (PPC cross-theme stats, sigma_c, rho, QMUL, held-out 2026 metrics vs G and M4)."""
import pandas as pd, numpy as np, json, time, sys
from scipy import stats
from bayes_model import build_index, fit, diagnostics, flatten_samples, PRIORS
THR = 2.5
THEMES = {1: 'Teaching', 2: 'Learning opps', 3: 'Assessment & feedback', 4: 'Academic support', 5: 'Organisation & mgmt', 6: 'Learning resources', 7: 'Student voice'}
PRI = dict(PRIORS); PRI.update(sd_eta=3.0, alpha_fixed_sd=3.0)
prov = pd.read_csv('../data/sector_cah10_provider_theme.csv')
res = {}
def band(x): return np.where(x > THR, 'above', np.where(x < -THR, 'below', 'inline'))
def q95(x): return dict(mean=float(np.mean(x)), lo95=float(np.quantile(x, .025)), hi95=float(np.quantile(x, .975)))
def do_fit(df, seed):
    idx = build_index(df, 'UKPRN')
    t0 = time.time(); mcmc, _ = fit(idx, cohort=True, dynamic=True, eps=False, alpha_prior='fixed', priors=PRI, num_warmup=2000, num_samples=2000, num_chains=4, target_accept=0.95, seed=seed, rho_err=True)
    summ, diag, _ = diagnostics(mcmc, params=('mu', 'sd_gamma', 'sd_u', 'sd_v', 'sd_c', 'sd_eta', 'rho')); S = flatten_samples(mcmc); del mcmc
    print('M4a', seed, f'{time.time()-t0:.0f}s', diag, file=sys.stderr, flush=True)
    return idx, S, summ, diag

res = json.load(open('../out/M4a_summary.json'))
# ---------------- Model A + held-out 2026 ----------------
idxA, SA, summA, diagA = do_fit(prov[prov.Year <= 2025].copy(), 502); summA.to_csv('../out/M4a_A_hyper.csv'); res['A_diag'] = diagA; res['A_rho'] = q95(SA['rho']); res['A_sd_c'] = q95(SA['sd_c'])
te = prov[prov.Year == 2026].copy(); te['u_i'] = idxA['units'].get_indexer(te.UKPRN); te['q_i'] = idxA['themes'].get_indexer(te.theme)
te = te.merge(idxA['uq'][['u_i', 'q_i', 'v_i']], on=['u_i', 'q_i'], how='left'); seen = (te.u_i >= 0).values
Sn = SA['mu'].shape[0]; N = len(te); rng = np.random.default_rng(5)
vi = np.clip(te.v_i.fillna(0).astype(int).values, 0, None); ui = np.clip(te.u_i.values, 0, None); w_last = SA['w_dyn'][:, -1, :]
m_pred = SA['mu'][:, None] + SA['alpha'][:, te.q_i.values] + np.where(seen[None, :], SA['u'][:, ui] + SA['v'][:, vi] + w_last[:, vi], 0.0)
gam_new = SA['sd_gamma'][:, None] * rng.standard_normal((Sn, 1)); codes = te.UKPRN.astype('category').cat.codes.values
c_new = (SA['sd_c'][:, None] * rng.standard_normal((Sn, codes.max() + 1)))[:, codes]; eta_new = SA['sd_eta'][:, None] * rng.standard_normal((Sn, N))
g_new = rng.standard_normal((Sn, codes.max() + 1))[:, codes]; z = rng.standard_normal((Sn, N)); rho_a = SA['rho'][:, None]
D_pred = m_pred + gam_new + c_new + eta_new + te.s.values[None, :] * (np.sqrt(rho_a) * g_new + np.sqrt(1 - rho_a) * z)
te['pred_mean'] = m_pred.mean(0)
for lv in [50, 80, 90, 95]:
    te[f'lo{lv}'] = np.quantile(D_pred, (1 - lv / 100) / 2, 0); te[f'hi{lv}'] = np.quantile(D_pred, 1 - (1 - lv / 100) / 2, 0)
S_ = D_pred.shape[0]; xs = np.sort(D_pred, axis=0); ii = np.arange(1, S_ + 1)[:, None]
te['crps'] = np.abs(D_pred - te.D.values[None, :]).mean(0) - 0.5 * (2.0 / (S_ * S_)) * ((2 * ii - S_ - 1) * xs).sum(0)
te['p_above'] = (D_pred > THR).mean(0); te['p_below'] = (D_pred < -THR).mean(0); te['p_inline'] = 1 - te.p_above - te.p_below
te.to_csv('../out/M4a_A_predictions.csv', index=False)
b = pd.read_csv('../out/final_A_main2026_predictions.csv')[['UKPRN', 'theme', 'D', 'last', 'pw', 'eb', 'pred_mean']].rename(columns={'pred_mean': 'pred_G'})
m4c = pd.read_csv('../out/M4_clean_A_predictions.csv')[['UKPRN', 'theme', 'pred_mean', 'crps']].rename(columns={'pred_mean': 'pred_M4clean', 'crps': 'crps_M4clean'})
b = b.merge(m4c, on=['UKPRN', 'theme']).merge(te[['UKPRN', 'theme', 'pred_mean', 'crps', 'p_above', 'p_inline', 'p_below', 'lo50', 'hi50', 'lo80', 'hi80', 'lo90', 'hi90', 'lo95', 'hi95']], on=['UKPRN', 'theme'])
bm = b[b['last'].notna()].copy(); y = bm.D.values; ytrue = band(np.round(y, 1))
def score_probs(P):
    lab = np.array(['above', 'inline', 'below']); Y = (ytrue[:, None] == lab[None, :]).astype(float); P = np.clip(P, 1e-6, 1); P = P / P.sum(1, keepdims=True)
    return dict(brier=float(((P - Y) ** 2).sum(1).mean()), logloss=float(-np.log((P * Y).sum(1)).mean()))
ho = dict(n=int(len(bm)))
for col in ['last', 'pw', 'eb', 'pred_G', 'pred_M4clean', 'pred_mean']:
    e = bm[col] - bm.D; ho[col] = dict(RMSE=float(np.sqrt((e ** 2).mean())), MAE=float(e.abs().mean()))
ho['M4a_probabilistic'] = dict(**score_probs(bm[['p_above', 'p_inline', 'p_below']].values), coverage={lv: float(((y >= bm[f'lo{lv}']) & (y <= bm[f'hi{lv}'])).mean()) for lv in [50, 80, 90, 95]}, mean_width90=float((bm.hi90 - bm.lo90).mean()), mean_crps=float(bm.crps.mean()))
ho['M4clean_mean_crps'] = float(bm.crps_M4clean.mean())
res['holdout'] = ho
json.dump(res, open('../out/M4a_summary.json', 'w'), indent=2, default=str)
print(json.dumps({k: v for k, v in res.items() if k != 'QMUL_table'}, indent=2, default=str))
