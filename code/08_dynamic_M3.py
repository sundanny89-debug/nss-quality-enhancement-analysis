"""Step 8 (new): M3 = cohort model + random-walk unit x theme effects (theme-specific persistent shifts allowed).
Tests whether theme-specific latent change exists beyond the shared cohort shock; held-out 2026 prediction; full-data sd_eta."""
import pandas as pd, numpy as np, json, time, sys
from bayes_model import *
THR = 2.5; NW, NS, NC = 1000, 1000, 4
prov = pd.read_csv('../data/sector_cah10_provider_theme.csv')
res = {}
import os
if os.path.exists('../out/M3_holdout_partial.json') and (len(sys.argv) > 1 and sys.argv[1] == 'full'):
    res = json.load(open('../out/M3_holdout_partial.json'))
    RUN_HOLDOUT = False
else:
    RUN_HOLDOUT = True
tr = prov[prov.Year <= 2025]; te = prov[prov.Year == 2026]
idx = build_index(tr, 'UKPRN'); d = idx['df']
if RUN_HOLDOUT:
  t0 = time.time(); mcmc, kw = fit(idx, cohort=True, dynamic=True, num_warmup=NW, num_samples=NS, num_chains=NC, target_accept=0.95, seed=31)
  summ, diag, _ = diagnostics(mcmc); S = flatten_samples(mcmc); print('M3 holdout', f'{time.time()-t0:.0f}s', diag, file=sys.stderr)
  summ.to_csv('../out/holdout_main_M3_dynamic_hyper.csv'); res['holdout_hyper'] = summ[['mean', 'sd', 'eti90_lb', 'eti90_ub']].round(3).to_dict(); res['holdout_diag'] = diag
  te2 = te.copy(); te2['u_i'] = idx['units'].get_indexer(te2.UKPRN); te2['q_i'] = idx['themes'].get_indexer(te2.theme)
  te2 = te2.merge(idx['uq'][['u_i', 'q_i', 'v_i']], on=['u_i', 'q_i'], how='left'); seen = te2.u_i >= 0
  Sn = S['mu'].shape[0]; rng = np.random.default_rng(8)
  w_last = S['w_dyn'][:, -1, :]   # random-walk state at last training year (2025)
  eta_new = S['sd_eta'][:, None] * rng.standard_normal((Sn, len(te2)))
  m_pred = S['mu'][:, None] + S['alpha'][:, te2.q_i.values] + np.where(seen.values[None, :], S['u'][:, np.clip(te2.u_i.values, 0, None)] + S['v'][:, np.clip(te2.v_i.fillna(0).astype(int).values, 0, None)] + w_last[:, np.clip(te2.v_i.fillna(0).astype(int).values, 0, None)], 0.0)
  te2['pred_M3'] = m_pred.mean(0)   # point prediction: E[eta_new]=0
  te2.to_csv('../out/holdout_main_M3_dynamic_pred.csv', index=False)
  b = pd.read_csv('../out/holdout_main_all_predictions.csv').merge(te2[['UKPRN', 'theme', 'pred_M3']], on=['UKPRN', 'theme'])
  bm = b[b['last'].notna()]
  for col in ['last', 'pw', 'pred_M1_cohort', 'pred_M3']:
      e = bm[col] - bm.D; res[f'matched_{col}'] = dict(RMSE=float(np.sqrt((e ** 2).mean())), MAE=float(e.abs().mean()), n=int(len(e)))
  json.dump(res, open('../out/M3_holdout_partial.json', 'w'), indent=2, default=str)
# ---- full data ----
idx = build_index(prov, 'UKPRN'); d = idx['df']
t0 = time.time(); mcmc, kw = fit(idx, cohort=True, dynamic=True, num_warmup=700, num_samples=700, num_chains=NC, target_accept=0.97, seed=32)
summ, diag, _ = diagnostics(mcmc); S = flatten_samples(mcmc); print('M3 full', f'{time.time()-t0:.0f}s', diag, file=sys.stderr)
summ.to_csv('../out/full_M3_dynamic_hyper.csv'); res['full_hyper'] = summ[['mean', 'sd', 'eti90_lb', 'eti90_ub']].round(3).to_dict(); res['full_diag'] = diag
# theme-specific latent change 2025->2026 = eta_{2026} innovations (w_2026 - w_2025) per unit x theme; compare with cohort change
dw = S['w_dyn'][:, 3, :] - S['w_dyn'][:, 2, :]     # (S, n_v)
res['theme_specific_change_2025_26'] = dict(sd_of_posterior_means=float(dw.mean(0).std()), share_abs_mean_gt_2_5=float((np.abs(dw.mean(0)) > 2.5).mean()),
                                            share_P_gt_2_5=float(((np.abs(dw) > 2.5).mean(0) > 0.8).mean()), sd_eta_mean=float(S['sd_eta'].mean()),
                                            sd_c_mean=float(S['sd_c'].mean()))
# WAIC comparison
m = S['m']; sd_tot = np.sqrt(d.s.values[None, :] ** 2 + S['sd_eps'][:, None] ** 2)
ll = -0.5 * np.log(2 * np.pi) - np.log(sd_tot) - 0.5 * ((d.D.values[None, :] - m) / sd_tot) ** 2
lppd = np.log(np.exp(ll - ll.max(0)).mean(0)) + ll.max(0); pw_ = ll.var(0, ddof=1)
res['full_waic_M3'] = dict(waic=float(-2 * (lppd.sum() - pw_.sum())), p_waic=float(pw_.sum()))
# QMUL theme-specific drifts
uq = idx['uq']; qi = idx['units'].get_loc(10007775)
rows = []
for _, r in uq[uq.u_i == qi].iterrows():
    w = S['w_dyn'][:, :, int(r.v_i)]
    rows.append(dict(theme=int(idx['themes'][r.q_i]), w2023=0.0, w2024=float(w[:, 1].mean()), w2025=float(w[:, 2].mean()), w2026=float(w[:, 3].mean()), P_dw2526_gt_2_5=float((dw[:, int(r.v_i)] > 2.5).mean()), P_dw2526_lt_m2_5=float((dw[:, int(r.v_i)] < -2.5).mean())))
res['QMUL_theme_drift'] = rows
json.dump(res, open('../out/M3_dynamic_summary.json', 'w'), indent=2, default=str)
print(json.dumps(res, indent=2, default=str))
