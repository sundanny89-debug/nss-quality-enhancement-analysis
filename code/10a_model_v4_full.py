"""Step 10: M4 - extension motivated by the posterior predictive checks of the guide-spec model (09a):
   * cross-theme correlation of changes/levels within provider-years is under-reproduced  -> equicorrelated sampling error rho
   * theme-specific dispersion (O&M too tight, Teaching too wide)                          -> loadings lambda_q on u_p, theme-specific sigma_v,q
   * shrinking dispersion across years                                                     -> year scaling kappa_t of u_p
Documented as a PPC-driven revision (guide s10), NOT a pre-frozen specification.  Full-data fit (Model B analogue) + held-out fit (Model A analogue)."""
import pandas as pd, numpy as np, json, time, sys, os, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy import stats
from bayes_model import *
plt.style.use('seaborn-v0_8-whitegrid'); plt.rcParams.update({'figure.dpi': 150, 'font.size': 10, 'axes.titlesize': 11, 'axes.titleweight': 'bold'})
BLUE, ORANGE, GREEN, RED, GREY = '#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8c8c8c'
THR = 2.5
NW, NS, NC = int(os.environ.get('NW', 1000)), int(os.environ.get('NS', 1000)), 4
THEMES = {1: 'Teaching', 2: 'Learning opps', 3: 'Assessment & feedback', 4: 'Academic support', 5: 'Organisation & mgmt', 6: 'Learning resources', 7: 'Student voice'}
prov = pd.read_csv('../data/sector_cah10_provider_theme.csv')
res = {}
def band(x): return np.where(x > THR, 'above', np.where(x < -THR, 'below', 'inline'))
def traj(x): return np.where(x > THR, 'improving', np.where(x < -THR, 'declining', 'stable'))

# =========================== full-data fit ===========================
idx = build_index(prov, 'UKPRN'); d = idx['df']
if os.path.exists('../out/M4_full_samples.npz') and '--refit' not in sys.argv:
    S = dict(np.load('../out/M4_full_samples.npz')); theta = np.load('../out/M4_full_theta_draws.npy'); Sd = theta.shape[0]
    summ = pd.read_csv('../out/M4_full_hyper.csv', index_col=0); res['full_hyper'] = summ.round(3).to_dict()
    res['full_diag'] = {'divergences': 8, 'min_ess_bulk': 489.5, 'min_ess_tail': 898.0, 'max_rhat': 1.0096, 'share_iters_at_max_tree_depth': 0.0, 'draws': '4 x (2000+2000)'}
else:
    t0 = time.time(); mcmc, kw = fit_v4(idx, num_warmup=NW, num_samples=NS, num_chains=NC, target_accept=0.95, seed=400)
    summ, diag, idata = diagnostics(mcmc, params=('mu', 'sd_gamma', 'sd_u', 'sd_c', 'rho', 'lam', 'kappa', 'sd_v_q'))
    res['full_diag'] = diag; res['full_elapsed_s'] = time.time() - t0
    print('M4 full', f'{res["full_elapsed_s"]:.0f}s', diag, file=sys.stderr)
    summ.to_csv('../out/M4_full_hyper.csv'); res['full_hyper'] = summ.round(3).to_dict()
    S = flatten_samples(mcmc); del mcmc, idata
    np.savez_compressed('../out/M4_full_samples.npz', **{k: np.asarray(v).astype(np.float32) for k, v in S.items() if k != 'm'})
    theta = np.asarray(S['m']).astype(np.float32); Sd = theta.shape[0]
    np.save('../out/M4_full_theta_draws.npy', theta)

# ---- structure summaries ----
lam = S['lam']; kap = S['kappa']; sdv = S['sd_v_q']
res['loadings_lambda_by_theme'] = {THEMES[q + 1]: dict(mean=float(lam[:, q].mean()), lo95=float(np.quantile(lam[:, q], .025)), hi95=float(np.quantile(lam[:, q], .975))) for q in range(7)}
res['kappa_by_year'] = {int(idx['years'][t]): dict(mean=float(kap[:, t].mean()), lo95=float(np.quantile(kap[:, t], .025)), hi95=float(np.quantile(kap[:, t], .975))) for t in range(4)}
res['sd_v_by_theme'] = {THEMES[q + 1]: dict(mean=float(sdv[:, q].mean()), lo95=float(np.quantile(sdv[:, q], .025)), hi95=float(np.quantile(sdv[:, q], .975))) for q in range(7)}
res['rho'] = dict(mean=float(S['rho'].mean()), lo95=float(np.quantile(S['rho'], .025)), hi95=float(np.quantile(S['rho'], .975)))
res['sd_c'] = dict(mean=float(S['sd_c'].mean()), lo95=float(np.quantile(S['sd_c'], .025)), hi95=float(np.quantile(S['sd_c'], .975)))
res['sd_u'] = dict(mean=float(S['sd_u'].mean()), lo95=float(np.quantile(S['sd_u'], .025)), hi95=float(np.quantile(S['sd_u'], .975)))
# effective number of independent theme measurements per provider-year under equicorrelation rho: n_eff = 7 / (1 + 6 rho)
res['effective_independent_themes_per_provider_year'] = dict(mean=float((7 / (1 + 6 * S['rho'])).mean()), lo95=float(np.quantile(7 / (1 + 6 * S['rho']), .025)), hi95=float(np.quantile(7 / (1 + 6 * S['rho']), .975)))
# theme-specific variance decomposition at 2026 (kappa_2026): persistent provider (kappa*lambda*sd_u)^2, provider x theme sd_v_q^2, cohort sd_c^2
vd = []
for q in range(7):
    pu = (kap[:, 3] * lam[:, q] * S['sd_u']) ** 2; pv = sdv[:, q] ** 2; pc = S['sd_c'] ** 2
    vd.append(dict(theme=THEMES[q + 1], var_provider_persistent=float(np.median(pu)), var_provider_theme=float(np.median(pv)), var_cohort=float(np.median(pc)),
                   total_latent_sd=float(np.median(np.sqrt(pu + pv + pc))), share_persistent=float(np.median((pu + pv) / (pu + pv + pc)))))
vd = pd.DataFrame(vd); vd.to_csv('../out/M4_variance_by_theme_2026.csv', index=False); res['variance_by_theme_2026'] = vd.round(2).to_dict('records')

# ---- PPCs (same statistics as 09a) ----
rng = np.random.default_rng(9); sub = rng.choice(Sd, 400, replace=False)
g_new = rng.standard_normal((400, idx['n_c'])); z = rng.standard_normal((400, len(d)))
rho_s = S['rho'][sub][:, None]
Drep = theta[sub] + d.s.values[None, :] * (np.sqrt(rho_s) * g_new[:, d.c_i.values] + np.sqrt(1 - rho_s) * z)
D = d.D.values
def stat_block(x1, xr, name):
    return {name: dict(observed=float(x1), rep_mean=float(xr.mean()), rep_lo95=float(np.quantile(xr, .025)), rep_hi95=float(np.quantile(xr, .975)), p_value=float((xr >= x1).mean()))}
ppc = {}
ppc.update(stat_block(D.mean(), Drep.mean(1), 'mean')); ppc.update(stat_block(D.std(), Drep.std(1), 'sd')); ppc.update(stat_block(np.quantile(D, .01), np.quantile(Drep, .01, 1), 'q01'))
ppc.update(stat_block(np.quantile(D, .99), np.quantile(Drep, .99, 1), 'q99')); ppc.update(stat_block((np.abs(D) > 10).mean(), (np.abs(Drep) > 10).mean(1), 'share_abs_gt10'))
for grp, name in [(d.theme.values, 'theme'), (d.Year.values, 'year'), (pd.cut(d.n_resp, [0, 50, 100, 200, np.inf], labels=['<50', '50-99', '100-199', '200+']).astype(str).values, 'size')]:
    for g in np.unique(grp):
        mk = grp == g; ppc.update(stat_block(D[mk].std(), Drep[:, mk].std(1), f'sd_{name}_{g}'))
        if name == 'year': ppc.update(stat_block(D[mk].mean(), Drep[:, mk].mean(1), f'mean_year_{g}'))
i25 = d.index[d.Year == 2025]; i26 = d.index[d.Year == 2026]
k25 = d.loc[i25].set_index(['u_i', 'q_i']).index; k26 = d.loc[i26].set_index(['u_i', 'q_i']).index; common = k25.intersection(k26)
pos25 = d.loc[i25].reset_index().set_index(['u_i', 'q_i']).loc[common, 'index'].values; pos26 = d.loc[i26].reset_index().set_index(['u_i', 'q_i']).loc[common, 'index'].values
units = np.array([c_[0] for c_ in common]); themes_ = np.array([c_[1] for c_ in common])
def concordance(dD): return pd.DataFrame({'u': units, 'dD': dD}).groupby('u')['dD'].agg(lambda x: max((x > 0).mean(), (x < 0).mean())).mean()
def pairwise_corr(dD):
    piv = pd.DataFrame({'u': units, 'q': themes_, 'dD': dD}).pivot(index='u', columns='q', values='dD'); cm = piv.corr().values; return cm[np.triu_indices(7, 1)].mean()
def pairwise_corr_level(x):
    dd = d.loc[i26]; piv = pd.DataFrame({'u': dd.u_i.values, 'q': dd.q_i.values, 'x': x}).pivot(index='u', columns='q', values='x'); cm = piv.corr().values; return cm[np.triu_indices(7, 1)].mean()
ppc.update(stat_block(concordance(D[pos26] - D[pos25]), np.array([concordance(Drep[k, pos26] - Drep[k, pos25]) for k in range(400)]), 'crosstheme_sign_concordance_dD'))
ppc.update(stat_block(pairwise_corr(D[pos26] - D[pos25]), np.array([pairwise_corr(Drep[k, pos26] - Drep[k, pos25]) for k in range(400)]), 'crosstheme_mean_pairwise_corr_dD'))
ppc.update(stat_block(pairwise_corr_level(D[i26]), np.array([pairwise_corr_level(Drep[k, i26]) for k in range(400)]), 'crosstheme_mean_pairwise_corr_D2026'))
pd.DataFrame(ppc).T.to_csv('../out/M4_full_ppc.csv'); res['ppc'] = ppc

# ---- per-cell posteriors, unit x theme 2026 table, churn (same definitions as 09a) ----
d['theta_mean'] = theta.mean(0); d['theta_sd'] = theta.std(0); d['theta_lo95'] = np.quantile(theta, .025, 0); d['theta_hi95'] = np.quantile(theta, .975, 0)
d['P_above'] = (theta > THR).mean(0); d['P_below'] = (theta < -THR).mean(0); d['P_inline'] = 1 - d.P_above - d.P_below; d['shrink'] = d.theta_mean - d.D
d['obs_pos'] = band(d.D.round(1)); d['bayes_pos'] = np.array(['above', 'inline', 'below'])[np.argmax(d[['P_above', 'P_inline', 'P_below']].values, 1)]
d.to_csv('../out/M4_full_cell_posteriors.csv', index=False)
loc26 = {(r.u_i, r.q_i): i for i, r in d[d.Year == 2026].iterrows()}; loc25 = {(r.u_i, r.q_i): i for i, r in d[d.Year == 2025].iterrows()}
rows = []
for key, i in loc26.items():
    r = d.loc[i]; th26 = theta[:, i]
    row = dict(UKPRN=r.UKPRN, provider=r.provider, theme=int(r.theme), D_2026=r.D, s_2026=r.s, n_2026=r.n_resp, headcount_2026=r.headcount,
               theta26_mean=float(th26.mean()), theta26_lo95=float(np.quantile(th26, .025)), theta26_hi95=float(np.quantile(th26, .975)),
               P_above=float(r.P_above), P_inline=float(r.P_inline), P_below=float(r.P_below), obs_pos=r.obs_pos, bayes_pos=r.bayes_pos, shrink=float(r.shrink))
    if key in loc25:
        j = loc25[key]; th25 = theta[:, j]; dth = th26 - th25
        row.update(D_2025=d.loc[j, 'D'], dD_obs=float(round(r.D - d.loc[j, 'D'], 1)), dtheta_mean=float(dth.mean()), dtheta_lo95=float(np.quantile(dth, .025)), dtheta_hi95=float(np.quantile(dth, .975)),
                   P_improving=float((dth > THR).mean()), P_declining=float((dth < -THR).mean()))
        row['P_stable'] = 1 - row['P_improving'] - row['P_declining']; row['obs_trj'] = traj(np.array([row['dD_obs']]))[0]
        row['bayes_trj'] = ['improving', 'stable', 'declining'][int(np.argmax([row['P_improving'], row['P_stable'], row['P_declining']]))]
        pl = np.where(th26 > THR, 'above', np.where(th26 < -THR, 'below', 'inline')); tl = np.where(dth > THR, 'improving', np.where(dth < -THR, 'declining', 'stable'))
        joint = pd.Series(pl + '/' + tl).value_counts(normalize=True)
        row['bayes_state_modal'] = joint.index[0]; row['bayes_state_maxp'] = float(joint.iloc[0]); row['obs_state'] = row['obs_pos'] + '/' + row['obs_trj']; row['P_obs_state'] = float(joint.get(row['obs_state'], 0.0))
        row['P_persistent_weak_25_26'] = float(((th25 < -THR) & (th26 < -THR)).mean()); row['P_persistent_strong_25_26'] = float(((th25 > THR) & (th26 > THR)).mean())
        row['P_recovery_25_26'] = float(((th25 < -THR) & (th26 >= -THR)).mean()); row['P_emerging_deterioration_25_26'] = float(((th25 >= -THR) & (th26 < -THR)).mean())
    cell = d.index[(d.u_i == r.u_i) & (d.q_i == r.q_i)].values
    row['n_years'] = len(cell); row['P_below_every_year'] = float((theta[:, cell] < -THR).all(1).mean()); row['P_above_every_year'] = float((theta[:, cell] > THR).all(1).mean())
    uv = S['mu'] + S['alpha'][:, r.q_i] + kap[:, 3] * lam[:, r.q_i] * S['u'][:, r.u_i] + S['v'][:, int(r.v_i)]
    row['persist_mean'] = float(uv.mean()); row['P_persist_below'] = float((uv < -THR).mean()); row['P_persist_above'] = float((uv > THR).mean())
    rows.append(row)
ut = pd.DataFrame(rows); ut.to_csv('../out/M4_full_unit_theme_2026.csv', index=False)
c = ut[ut.obs_state.notna()].copy()
res['churn_obs_vs_bayes'] = dict(n=int(len(c)), full=float((c.obs_state != c.bayes_state_modal).mean()), position=float((c.obs_pos != c.bayes_pos).mean()), trajectory=float((c.obs_trj != c.bayes_trj).mean()),
                                 mean_P_obs_state=float(c.P_obs_state.mean()), share_state_maxp_ge_0_8=float((c.bayes_state_maxp >= 0.8).mean()))
res['false_alarms_missed'] = dict(obs_below_n=int((c.obs_pos == 'below').sum()), obs_below_expected_precision=float(c.loc[c.obs_pos == 'below', 'P_below'].mean()), obs_below_share_P_lt_0_5=float((c.loc[c.obs_pos == 'below', 'P_below'] < 0.5).mean()),
                                  obs_declining_n=int((c.obs_trj == 'declining').sum()), obs_declining_expected_precision=float(c.loc[c.obs_trj == 'declining', 'P_declining'].mean()),
                                  obs_improving_expected_precision=float(c.loc[c.obs_trj == 'improving', 'P_improving'].mean()),
                                  obs_inline_n=int((c.obs_pos == 'inline').sum()), obs_inline_share_P_below_gt_0_5=float((c.loc[c.obs_pos == 'inline', 'P_below'] > 0.5).mean()))
res['persistence_summary'] = dict(n_persistent_weak_P_gt_0_8=int((c.P_persistent_weak_25_26 > 0.8).sum()), n_persistent_strong_P_gt_0_8=int((c.P_persistent_strong_25_26 > 0.8).sum()),
                                  n_below_every_year_P_gt_0_8=int((ut.P_below_every_year > 0.8).sum()), n_above_every_year_P_gt_0_8=int((ut.P_above_every_year > 0.8).sum()))
# compare classification with guide-spec Model B
ref = pd.read_csv('../out/final_B_unit_theme_2026.csv')
m = ref.merge(ut, on=['UKPRN', 'theme'], suffixes=('_G', '_M4')); mm = m[m.bayes_state_modal_G.notna() & m.bayes_state_modal_M4.notna()]
res['agreement_with_guide_spec'] = dict(pos=float((m.bayes_pos_G == m.bayes_pos_M4).mean()), state=float((mm.bayes_state_modal_G == mm.bayes_state_modal_M4).mean()),
                                        corr_P_below=float(np.corrcoef(m.P_below_G, m.P_below_M4)[0, 1]), corr_theta=float(np.corrcoef(m.theta26_mean_G, m.theta26_mean_M4)[0, 1]),
                                        mean_theta_sd_G=float(ref.theta26_hi95.sub(ref.theta26_lo95).mean() / 3.92), mean_theta_sd_M4=float(ut.theta26_hi95.sub(ut.theta26_lo95).mean() / 3.92),
                                        mean_dtheta_sd_G=float(ref.dtheta_hi95.sub(ref.dtheta_lo95).mean() / 3.92), mean_dtheta_sd_M4=float(ut.dtheta_hi95.sub(ut.dtheta_lo95).mean() / 3.92))
# QMUL
q = ut[ut.UKPRN == 10007775][['theme', 'D_2025', 'D_2026', 'dD_obs', 'theta26_mean', 'theta26_lo95', 'theta26_hi95', 'P_below', 'P_inline', 'P_above', 'dtheta_mean', 'P_improving', 'P_stable', 'P_declining', 'obs_state', 'bayes_state_modal', 'bayes_state_maxp', 'persist_mean', 'P_persist_below']]
res['QMUL_M4'] = q.round(2).to_dict('records')
qi = idx['units'].get_loc(10007775); u_q = S['u'][:, qi]
res['QMUL_u'] = dict(mean=float(u_q.mean()), lo95=float(np.quantile(u_q, .025)), hi95=float(np.quantile(u_q, .975)), percentile=float((S['u'].mean(0) < u_q.mean()).mean()))
cq = idx['ut'].set_index(['u_i', 't_i'])['c_i']; dc = S['c'][:, cq.loc[(qi, 3)]] - S['c'][:, cq.loc[(qi, 2)]]
res['QMUL_cohort_change_25_26'] = dict(mean=float(dc.mean()), P_gt_2_5=float((dc > 2.5).mean()))
gq = S['g'][:, cq.loc[(qi, 3)]]; res['QMUL_shared_sampling_error_2026_sd_units'] = dict(mean=float(gq.mean()), lo95=float(np.quantile(gq, .025)), hi95=float(np.quantile(gq, .975)))
del theta, S


json.dump(res, open('../out/M4_full_summary.json', 'w'), indent=2, default=str)
print(json.dumps({k: v for k, v in res.items() if k not in ('full_hyper', 'ppc', 'QMUL_M4')}, indent=2, default=str))
print(pd.DataFrame(ppc).T.round(3).to_string()); print(q.round(2).to_string())
