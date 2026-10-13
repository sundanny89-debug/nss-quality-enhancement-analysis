"""Clean M4 diagnostic refit — extended runner.
The two fit calls are IDENTICAL to run_M4_clean_refit.py (same data, same fit_v4/M4 model and priors, 4 chains,
3,000 warm-up, 2,000 retained draws, target_accept 0.99, seeds 9901 [B] and 9902 [A]).  The ONLY additions are:
saving posterior draws and computing the comparison quantities the validation memo asks for (rho, sigma_c, variance
parameters, QMUL quantities, cross-theme PPC statistics, Model-A held-out 2026 predictive metrics).
Outputs: M4_clean_B_hyper.csv, M4_clean_A_hyper.csv, M4_clean_diagnostics.json (as in the original script) + extras."""
import pandas as pd, numpy as np, json, time, sys
from scipy import stats
from bayes_model import build_index, fit_v4, diagnostics, flatten_samples

THR = 2.5
THEMES = {1: 'Teaching', 2: 'Learning opps', 3: 'Assessment & feedback', 4: 'Academic support', 5: 'Organisation & mgmt', 6: 'Learning resources', 7: 'Student voice'}
prov = pd.read_csv('../data/sector_cah10_provider_theme.csv')
results = {}
def band(x): return np.where(x > THR, 'above', np.where(x < -THR, 'below', 'inline'))

def run(label, df, seed):
    idx = build_index(df, 'UKPRN')
    t0 = time.time()
    mcmc, _ = fit_v4(idx, num_warmup=3000, num_samples=2000, num_chains=4, seed=seed, target_accept=0.99)   # identical to run_M4_clean_refit.py
    summ, diag, _ = diagnostics(mcmc, params=('mu', 'sd_gamma', 'sd_u', 'sd_c', 'rho', 'lam', 'kappa', 'sd_v_q'))
    summ.to_csv(f'../out/{label}_hyper.csv')
    results[label] = {'diagnostics': diag, 'elapsed_s': time.time() - t0}
    print(label, f'{time.time()-t0:.0f}s', diag, file=sys.stderr, flush=True)
    S = flatten_samples(mcmc); del mcmc
    np.savez_compressed(f'../out/{label}_samples.npz', **{k: np.asarray(v).astype(np.float32) for k, v in S.items() if k != 'm'})
    np.save(f'../out/{label}_theta_draws.npy', np.asarray(S['m']).astype(np.float32))
    idx['df'].to_csv(f'../out/{label}_obs_index.csv', index=False)
    with open('../out/M4_clean_diagnostics.json', 'w') as f:
        json.dump(results, f, indent=2)
    return idx, S

# ================= Model B (2023-2026) =================
idxB, SB = run('M4_clean_B', prov, 9901)
d = idxB['df']; theta = np.asarray(SB['m']).astype(np.float32); Sd = theta.shape[0]
extra = {}
def q95(x): return dict(mean=float(np.mean(x)), lo95=float(np.quantile(x, .025)), hi95=float(np.quantile(x, .975)))
extra['B_rho'] = q95(SB['rho']); extra['B_sd_c'] = q95(SB['sd_c']); extra['B_sd_u'] = q95(SB['sd_u']); extra['B_sd_gamma'] = q95(SB['sd_gamma']); extra['B_mu'] = q95(SB['mu'])
extra['B_kappa'] = {int(idxB['years'][t]): q95(SB['kappa'][:, t]) for t in range(4)}
extra['B_lambda'] = {THEMES[q + 1]: q95(SB['lam'][:, q]) for q in range(7)}
extra['B_sd_v_q'] = {THEMES[q + 1]: q95(SB['sd_v_q'][:, q]) for q in range(7)}
extra['B_effective_independent_themes'] = q95(7 / (1 + 6 * SB['rho']))
# QMUL
qi = idxB['units'].get_loc(10007775); u_q = SB['u'][:, qi]
extra['QMUL_u'] = dict(**q95(u_q), percentile=float((SB['u'].mean(0) < u_q.mean()).mean()))
cq = idxB['ut'].set_index(['u_i', 't_i'])['c_i']
extra['QMUL_g_2026'] = q95(SB['g'][:, cq.loc[(qi, 3)]]); extra['QMUL_cohort_change_25_26'] = dict(**q95(SB['c'][:, cq.loc[(qi, 3)]] - SB['c'][:, cq.loc[(qi, 2)]]))
rows = []
for t_ in range(7):
    i26 = d.index[(d.u_i == qi) & (d.q_i == t_) & (d.Year == 2026)][0]; i25 = d.index[(d.u_i == qi) & (d.q_i == t_) & (d.Year == 2025)][0]
    th26 = theta[:, i26]; th25 = theta[:, i25]; dth = th26 - th25
    uv = SB['mu'] + SB['alpha'][:, t_] + SB['kappa'][:, 3] * SB['lam'][:, t_] * SB['u'][:, qi] + SB['v'][:, int(d.loc[i26, 'v_i'])]
    rows.append(dict(theme=THEMES[t_ + 1], D_2025=d.loc[i25, 'D'], D_2026=d.loc[i26, 'D'], theta26_mean=th26.mean(), theta26_lo95=np.quantile(th26, .025), theta26_hi95=np.quantile(th26, .975),
                     P_below=(th26 < -THR).mean(), P_inline=((th26 >= -THR) & (th26 <= THR)).mean(), P_above=(th26 > THR).mean(),
                     dtheta_mean=dth.mean(), P_improving=(dth > THR).mean(), P_declining=(dth < -THR).mean(), persist_mean=uv.mean(), P_persist_below=(uv < -THR).mean()))
qt = pd.DataFrame(rows); qt.to_csv('../out/M4_clean_B_QMUL.csv', index=False); extra['QMUL_table'] = qt.round(3).to_dict('records')
# cross-theme PPC statistics (400 replicates)
rng = np.random.default_rng(9); sub = rng.choice(Sd, 400, replace=False)
g_new = rng.standard_normal((400, idxB['n_c'])); z = rng.standard_normal((400, len(d))); rho_s = SB['rho'][sub][:, None]
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
extra['PPC'] = {'crosstheme_sign_concordance_dD': stat_block(concordance(D[pos26] - D[pos25]), np.array([concordance(Drep[k, pos26] - Drep[k, pos25]) for k in range(400)])),
                'crosstheme_mean_pairwise_corr_dD': stat_block(pairwise_corr(D[pos26] - D[pos25]), np.array([pairwise_corr(Drep[k, pos26] - Drep[k, pos25]) for k in range(400)])),
                'crosstheme_mean_pairwise_corr_D2026': stat_block(pairwise_corr_level(D[i26]), np.array([pairwise_corr_level(Drep[k, i26]) for k in range(400)])),
                'mean': stat_block(D.mean(), Drep.mean(1)), 'sd': stat_block(D.std(), Drep.std(1))}
# sector-wide decision summaries for comparison with old M4 (unit x theme 2026, cells with 2025)
d['P_below'] = (theta < -THR).mean(0); d['P_above'] = (theta > THR).mean(0); d['P_inline'] = 1 - d.P_below - d.P_above; d['theta_mean'] = theta.mean(0)
d['obs_pos'] = band(d.D.round(1)); d['bayes_pos'] = np.array(['above', 'inline', 'below'])[np.argmax(d[['P_above', 'P_inline', 'P_below']].values, 1)]
loc26 = {(r.u_i, r.q_i): i for i, r in d[d.Year == 2026].iterrows()}; loc25 = {(r.u_i, r.q_i): i for i, r in d[d.Year == 2025].iterrows()}
rr = []
for key, i in loc26.items():
    if key not in loc25: continue
    j = loc25[key]; th26 = theta[:, i]; dth = th26 - theta[:, j]
    pl = np.where(th26 > THR, 'above', np.where(th26 < -THR, 'below', 'inline')); tl = np.where(dth > THR, 'improving', np.where(dth < -THR, 'declining', 'stable'))
    joint = pd.Series(pl + '/' + tl).value_counts(normalize=True)
    dD = round(d.loc[i, 'D'] - d.loc[j, 'D'], 1); obs_trj = 'improving' if dD > THR else ('declining' if dD < -THR else 'stable')
    rr.append(dict(UKPRN=d.loc[i, 'UKPRN'], theme=int(d.loc[i, 'theme']), obs_state=d.loc[i, 'obs_pos'] + '/' + obs_trj, obs_pos=d.loc[i, 'obs_pos'], obs_trj=obs_trj, bayes_pos=d.loc[i, 'bayes_pos'],
                   bayes_state=joint.index[0], P_obs_state=float(joint.get(d.loc[i, 'obs_pos'] + '/' + obs_trj, 0.0)), P_below=float(d.loc[i, 'P_below']), P_declining=float((dth < -THR).mean()), P_improving=float((dth > THR).mean()),
                   theta26=float(th26.mean())))
ut = pd.DataFrame(rr); ut.to_csv('../out/M4_clean_B_unit_theme_2026.csv', index=False)
extra['B_churn_obs_vs_bayes'] = dict(n=int(len(ut)), full=float((ut.obs_state != ut.bayes_state).mean()), position=float((ut.obs_pos != ut.bayes_pos).mean()), mean_P_obs_state=float(ut.P_obs_state.mean()),
                                    obs_below_expected_precision=float(ut.loc[ut.obs_pos == 'below', 'P_below'].mean()), obs_declining_expected_precision=float(ut.loc[ut.obs_trj == 'declining', 'P_declining'].mean()))
old = pd.read_csv('../out/M4_full_unit_theme_2026.csv')[['UKPRN', 'theme', 'bayes_pos', 'bayes_state_modal', 'P_below', 'theta26_mean']].rename(columns={'bayes_pos': 'bayes_pos_old', 'bayes_state_modal': 'bayes_state_old', 'P_below': 'P_below_old', 'theta26_mean': 'theta26_old'})
m = ut.merge(old, on=['UKPRN', 'theme'])
extra['B_agreement_old_vs_clean'] = dict(n=int(len(m)), pos=float((m.bayes_pos == m.bayes_pos_old).mean()), state=float((m.bayes_state == m.bayes_state_old).mean()), corr_P_below=float(np.corrcoef(m.P_below, m.P_below_old)[0, 1]),
                                         mean_abs_dtheta=float((m.theta26 - m.theta26_old).abs().mean()), max_abs_dtheta=float((m.theta26 - m.theta26_old).abs().max()))
json.dump(extra, open('../out/M4_clean_B_extra.json', 'w'), indent=2, default=str)
del theta, SB, Drep

# ================= Model A (2023-2025) and held-out 2026 =================
idxA, SA = run('M4_clean_A', prov[prov.Year <= 2025].copy(), 9902)
te = prov[prov.Year == 2026].copy()
te['u_i'] = idxA['units'].get_indexer(te.UKPRN); te['q_i'] = idxA['themes'].get_indexer(te.theme)
te = te.merge(idxA['uq'][['u_i', 'q_i', 'v_i']], on=['u_i', 'q_i'], how='left'); seen = (te.u_i >= 0).values
Sn = SA['mu'].shape[0]; N = len(te); rng = np.random.default_rng(5)
vi = np.clip(te.v_i.fillna(0).astype(int).values, 0, None); ui = np.clip(te.u_i.values, 0, None)
m_pred = SA['mu'][:, None] + SA['alpha'][:, te.q_i.values] + np.where(seen[None, :], SA['kappa'][:, [-1]] * SA['lam'][:, te.q_i.values] * SA['u'][:, ui] + SA['v'][:, vi], 0.0)
gam_new = SA['sd_gamma'][:, None] * rng.standard_normal((Sn, 1)); codes = te.UKPRN.astype('category').cat.codes.values
c_new = (SA['sd_c'][:, None] * rng.standard_normal((Sn, codes.max() + 1)))[:, codes]; g_new = rng.standard_normal((Sn, codes.max() + 1))[:, codes]; z = rng.standard_normal((Sn, N)); rho_a = SA['rho'][:, None]
D_pred = m_pred + gam_new + c_new + te.s.values[None, :] * (np.sqrt(rho_a) * g_new + np.sqrt(1 - rho_a) * z)
te['pred_mean'] = m_pred.mean(0)
for lv in [50, 60, 70, 80, 90, 95]:
    te[f'lo{lv}'] = np.quantile(D_pred, (1 - lv / 100) / 2, 0); te[f'hi{lv}'] = np.quantile(D_pred, 1 - (1 - lv / 100) / 2, 0)
S_ = D_pred.shape[0]; xs = np.sort(D_pred, axis=0); ii = np.arange(1, S_ + 1)[:, None]
te['crps'] = np.abs(D_pred - te.D.values[None, :]).mean(0) - 0.5 * (2.0 / (S_ * S_)) * ((2 * ii - S_ - 1) * xs).sum(0)
te['lpd'] = stats.norm.logpdf(te.D.values, D_pred.mean(0), D_pred.std(0))
te['p_above'] = (D_pred > THR).mean(0); te['p_below'] = (D_pred < -THR).mean(0); te['p_inline'] = 1 - te.p_above - te.p_below
te.to_csv('../out/M4_clean_A_predictions.csv', index=False)
b = pd.read_csv('../out/final_A_main2026_predictions.csv')[['UKPRN', 'theme', 'D', 'last', 'pw', 'eb', 'pred_mean']].rename(columns={'pred_mean': 'pred_G'})
oldA = pd.read_csv('../out/M4_holdout_predictions.csv')[['UKPRN', 'theme', 'pred_mean', 'crps', 'p_above', 'p_inline', 'p_below', 'lo90', 'hi90']].rename(columns={c_: c_ + '_old' for c_ in ['pred_mean', 'crps', 'p_above', 'p_inline', 'p_below', 'lo90', 'hi90']})
b = b.merge(oldA, on=['UKPRN', 'theme']).merge(te[['UKPRN', 'theme', 'pred_mean', 'crps', 'lpd', 'p_above', 'p_inline', 'p_below'] + [f'{a}{lv}' for lv in [50, 60, 70, 80, 90, 95] for a in ['lo', 'hi']]], on=['UKPRN', 'theme'])
bm = b[b['last'].notna()].copy(); y = bm.D.values; ytrue = band(np.round(y, 1))
def score_probs(P):
    lab = np.array(['above', 'inline', 'below']); Y = (ytrue[:, None] == lab[None, :]).astype(float); P = np.clip(P, 1e-6, 1); P = P / P.sum(1, keepdims=True)
    return dict(brier=float(((P - Y) ** 2).sum(1).mean()), logloss=float(-np.log((P * Y).sum(1)).mean()))
ho = dict(n=int(len(bm)))
for col in ['last', 'pw', 'eb', 'pred_G', 'pred_mean_old', 'pred_mean']:
    e = bm[col] - bm.D; ho[col] = dict(RMSE=float(np.sqrt((e ** 2).mean())), MAE=float(e.abs().mean()))
ho['clean_probabilistic'] = dict(**score_probs(bm[['p_above', 'p_inline', 'p_below']].values), coverage={lv: float(((y >= bm[f'lo{lv}']) & (y <= bm[f'hi{lv}'])).mean()) for lv in [50, 60, 70, 80, 90, 95]},
                                 mean_width90=float((bm.hi90 - bm.lo90).mean()), mean_crps=float(bm.crps.mean()))
ho['old_probabilistic'] = dict(**score_probs(bm[['p_above_old', 'p_inline_old', 'p_below_old']].values), coverage90=float(((y >= bm.lo90_old) & (y <= bm.hi90_old)).mean()), mean_width90=float((bm.hi90_old - bm.lo90_old).mean()), mean_crps=float(bm.crps_old.mean()))
ho['corr_pred_old_vs_clean'] = float(np.corrcoef(bm.pred_mean_old, bm.pred_mean)[0, 1]); ho['mean_abs_diff_pred_old_vs_clean'] = float((bm.pred_mean_old - bm.pred_mean).abs().mean())
extraA = dict(A_rho=q95(SA['rho']), A_sd_c=q95(SA['sd_c']), A_sd_u=q95(SA['sd_u']), A_kappa={int(idxA['years'][t]): q95(SA['kappa'][:, t]) for t in range(3)}, holdout=ho)
json.dump(extraA, open('../out/M4_clean_A_extra.json', 'w'), indent=2, default=str)
print(json.dumps(results, indent=2)); print(json.dumps({k: v for k, v in extra.items() if k not in ('QMUL_table',)}, indent=2, default=str)); print(json.dumps(extraA, indent=2, default=str))
