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

# ---------------- Model B ----------------
idx, S, summ, diag = do_fit(prov, 501); summ.to_csv('../out/M4a_B_hyper.csv'); res['B_diag'] = diag
d = idx['df']; theta = np.asarray(S['m']).astype(np.float32); Sd = theta.shape[0]
res['B_rho'] = q95(S['rho']); res['B_sd_c'] = q95(S['sd_c']); res['B_sd_u'] = q95(S['sd_u']); res['B_sd_v'] = q95(S['sd_v']); res['B_sd_eta'] = q95(S['sd_eta'])
rng = np.random.default_rng(9); sub = rng.choice(Sd, 400, replace=False)
g_new = rng.standard_normal((400, idx['n_c'])); z = rng.standard_normal((400, len(d))); rho_s = S['rho'][sub][:, None]
Drep = theta[sub] + d.s.values[None, :] * (np.sqrt(rho_s) * g_new[:, d.c_i.values] + np.sqrt(1 - rho_s) * z); D = d.D.values
i25 = d.index[d.Year == 2025]; i26 = d.index[d.Year == 2026]
k25 = d.loc[i25].set_index(['u_i', 'q_i']).index; k26 = d.loc[i26].set_index(['u_i', 'q_i']).index; common = k25.intersection(k26)
pos25 = d.loc[i25].reset_index().set_index(['u_i', 'q_i']).loc[common, 'index'].values; pos26 = d.loc[i26].reset_index().set_index(['u_i', 'q_i']).loc[common, 'index'].values
units = np.array([c_[0] for c_ in common]); themes_ = np.array([c_[1] for c_ in common])
def concordance(dD): return pd.DataFrame({'u': units, 'dD': dD}).groupby('u')['dD'].agg(lambda x: max((x > 0).mean(), (x < 0).mean())).mean()
def pairwise_corr(dD):
    piv = pd.DataFrame({'u': units, 'q': themes_, 'dD': dD}).pivot(index='u', columns='q', values='dD'); cm = piv.corr().values; return cm[np.triu_indices(7, 1)].mean()
def pairwise_corr_level(x):
    dd = d.loc[i26]; piv = pd.DataFrame({'u': dd.u_i.values, 'q': dd.q_i.values, 'x': x}).pivot(index='u', columns='q', values='x'); cm = piv.corr().values; return cm[np.triu_indices(7, 1)].mean()
def stat_block(x1, xr): return dict(observed=float(x1), rep_mean=float(xr.mean()), rep_lo95=float(np.quantile(xr, .025)), rep_hi95=float(np.quantile(xr, .975)), p_value=float((xr >= x1).mean()))
res['PPC'] = {'crosstheme_sign_concordance_dD': stat_block(concordance(D[pos26] - D[pos25]), np.array([concordance(Drep[k, pos26] - Drep[k, pos25]) for k in range(400)])),
              'crosstheme_mean_pairwise_corr_dD': stat_block(pairwise_corr(D[pos26] - D[pos25]), np.array([pairwise_corr(Drep[k, pos26] - Drep[k, pos25]) for k in range(400)])),
              'crosstheme_mean_pairwise_corr_D2026': stat_block(pairwise_corr_level(D[i26]), np.array([pairwise_corr_level(Drep[k, i26]) for k in range(400)])),
              'mean': stat_block(D.mean(), Drep.mean(1)), 'sd': stat_block(D.std(), Drep.std(1))}
for q_ in range(7):
    mk = d.theme.values == q_ + 1; res['PPC'][f'sd_theme_{q_+1}'] = stat_block(D[mk].std(), Drep[:, mk].std(1))
# QMUL
qi = idx['units'].get_loc(10007775); res['QMUL_u'] = dict(**q95(S['u'][:, qi]), percentile=float((S['u'].mean(0) < S['u'][:, qi].mean()).mean()))
cq = idx['ut'].set_index(['u_i', 't_i'])['c_i']; res['QMUL_cohort_change_25_26'] = q95(S['c'][:, cq.loc[(qi, 3)]] - S['c'][:, cq.loc[(qi, 2)]]); res['QMUL_g_2026'] = q95(S['g'][:, cq.loc[(qi, 3)]])
rows = []
for t_ in range(7):
    j26 = d.index[(d.u_i == qi) & (d.q_i == t_) & (d.Year == 2026)][0]; j25 = d.index[(d.u_i == qi) & (d.q_i == t_) & (d.Year == 2025)][0]
    th26 = theta[:, j26]; dth = th26 - theta[:, j25]
    rows.append(dict(theme=THEMES[t_ + 1], D_2026=d.loc[j26, 'D'], theta26_mean=th26.mean(), theta26_lo95=np.quantile(th26, .025), theta26_hi95=np.quantile(th26, .975), P_below=(th26 < -THR).mean(), P_above=(th26 > THR).mean(),
                     dtheta_mean=dth.mean(), P_improving=(dth > THR).mean(), P_declining=(dth < -THR).mean()))
res['QMUL_table'] = pd.DataFrame(rows).round(3).to_dict('records')
# sector decision summary
d['P_below'] = (theta < -THR).mean(0); d['P_above'] = (theta > THR).mean(0); d['P_inline'] = 1 - d.P_below - d.P_above; d['obs_pos'] = band(d.D.round(1))
d['bayes_pos'] = np.array(['above', 'inline', 'below'])[np.argmax(d[['P_above', 'P_inline', 'P_below']].values, 1)]
loc26 = {(r.u_i, r.q_i): i for i, r in d[d.Year == 2026].iterrows()}; loc25 = {(r.u_i, r.q_i): i for i, r in d[d.Year == 2025].iterrows()}
rr = []
for key, i in loc26.items():
    if key not in loc25: continue
    j = loc25[key]; th26 = theta[:, i]; dth = th26 - theta[:, j]
    pl = np.where(th26 > THR, 'above', np.where(th26 < -THR, 'below', 'inline')); tl = np.where(dth > THR, 'improving', np.where(dth < -THR, 'declining', 'stable'))
    joint = pd.Series(pl + '/' + tl).value_counts(normalize=True); dD = round(d.loc[i, 'D'] - d.loc[j, 'D'], 1); obs_trj = 'improving' if dD > THR else ('declining' if dD < -THR else 'stable')
    rr.append(dict(UKPRN=d.loc[i, 'UKPRN'], theme=int(d.loc[i, 'theme']), obs_state=d.loc[i, 'obs_pos'] + '/' + obs_trj, obs_pos=d.loc[i, 'obs_pos'], obs_trj=obs_trj, bayes_pos=d.loc[i, 'bayes_pos'], bayes_state=joint.index[0],
                   P_below=float(d.loc[i, 'P_below']), P_declining=float((dth < -THR).mean()), P_improving=float((dth > THR).mean()), theta26=float(th26.mean())))
ut = pd.DataFrame(rr); ut.to_csv('../out/M4a_B_unit_theme_2026.csv', index=False)
res['B_churn_obs_vs_bayes'] = dict(n=int(len(ut)), full=float((ut.obs_state != ut.bayes_state).mean()), position=float((ut.obs_pos != ut.bayes_pos).mean()),
                                   obs_below_expected_precision=float(ut.loc[ut.obs_pos == 'below', 'P_below'].mean()), obs_declining_expected_precision=float(ut.loc[ut.obs_trj == 'declining', 'P_declining'].mean()))
m4 = pd.read_csv('../out/M4_clean_B_unit_theme_2026.csv')[['UKPRN', 'theme', 'bayes_pos', 'bayes_state', 'P_below', 'theta26']].rename(columns={'bayes_pos': 'bayes_pos_m4', 'bayes_state': 'bayes_state_m4', 'P_below': 'P_below_m4', 'theta26': 'theta26_m4'})
mm = ut.merge(m4, on=['UKPRN', 'theme'])
res['B_agreement_M4a_vs_M4clean'] = dict(pos=float((mm.bayes_pos == mm.bayes_pos_m4).mean()), state=float((mm.bayes_state == mm.bayes_state_m4).mean()), corr_P_below=float(np.corrcoef(mm.P_below, mm.P_below_m4)[0, 1]), corr_theta=float(np.corrcoef(mm.theta26, mm.theta26_m4)[0, 1]))
json.dump(res, open('../out/M4a_summary.json', 'w'), indent=2, default=str)
del theta, S, Drep

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
