"""Step 4: Full-data (2023-2026) cohort-aware hierarchical model on the CAH10 sector panel.
Outputs: variance decomposition, reliability curves, per-cell posterior 3x3 classification (RQ1 second churn),
persistence probabilities, regression-to-the-mean diagnostics on latent scale (RQ3), M1 vs M0 comparison, QMUL position."""
import pandas as pd, numpy as np, json, time, sys
from bayes_model import *

THR = 2.5
NW, NS, NC = 1000, 1000, 4
prov = pd.read_csv('../data/sector_cah10_provider_theme.csv')
idx = build_index(prov, 'UKPRN')
d = idx['df']
res = {}

def band(x): return np.where(x > THR, 'above', np.where(x < -THR, 'below', 'inline'))
def traj(x): return np.where(x > THR, 'improving', np.where(x < -THR, 'declining', 'stable'))

def waic(loglik):  # loglik (S,N)
    lppd = np.log(np.exp(loglik - loglik.max(0)).mean(0)) + loglik.max(0)
    p = loglik.var(0, ddof=1)
    return dict(waic=float(-2 * (lppd.sum() - p.sum())), p_waic=float(p.sum()), lppd=float(lppd.sum()))

fits = {}
for name, kwargs in [('M1_cohort', dict(cohort=True)), ('M0_nocohort', dict(cohort=False))]:
    t0 = time.time()
    mcmc, kw = fit(idx, num_warmup=NW, num_samples=NS, num_chains=NC, target_accept=0.95, seed=3, **kwargs)
    summ, diag, idata = diagnostics(mcmc)
    S = flatten_samples(mcmc)
    print(name, f'{time.time()-t0:.0f}s', diag, file=sys.stderr)
    summ.to_csv(f'../out/full_{name}_hyper.csv')
    m = S['m']; sd_tot = np.sqrt(d['s'].values[None, :] ** 2 + S['sd_eps'][:, None] ** 2)
    ll = -0.5 * np.log(2 * np.pi) - np.log(sd_tot) - 0.5 * ((d['D'].values[None, :] - m) / sd_tot) ** 2
    res[name] = dict(diag=diag, hyper=summ[['mean', 'sd', 'eti90_lb', 'eti90_ub']].round(3).to_dict(), waic=waic(ll))
    fits[name] = S
    np.savez_compressed(f'../out/full_{name}_samples.npz', **{k: np.asarray(v).astype(np.float32) for k, v in S.items() if k != 'm'})
    del mcmc

S = fits['M1_cohort']
theta = latent_draws(S, None, d).astype(np.float32)   # (Sdraws, N) latent gap incl. idiosyncratic component
np.save('../out/full_M1_theta_draws.npy', theta)
d.to_csv('../out/full_M1_obs_index.csv', index=False)
Sd = theta.shape[0]

# ---------- variance decomposition ----------
var = pd.DataFrame({'unit(persistent, all themes)': S['sd_u'] ** 2, 'unit x theme (persistent)': S['sd_v'] ** 2,
                    'unit x year cohort shock': S['sd_c'] ** 2, 'idiosyncratic theme-year': S['sd_eps'] ** 2,
                    'theme': S['sd_alpha'] ** 2, 'year': S['sd_gamma'] ** 2})
s2 = d['s'].values ** 2
var['sampling (mean s^2, all cells)'] = s2.mean()
var['sampling (median s^2)'] = np.median(s2)
vs = var.describe(percentiles=[.05, .5, .95]).T[['mean', '5%', '50%', '95%']]
vs['share_of_latent_between_unit_var'] = vs['mean'] / vs.loc[['unit(persistent, all themes)', 'unit x theme (persistent)', 'unit x year cohort shock', 'idiosyncratic theme-year'], 'mean'].sum()
vs.to_csv('../out/full_variance_decomposition.csv')
res['variance_decomposition'] = vs.round(3).to_dict()
# year-to-year latent correlation of a unit x theme value (persistent share)
pers = S['sd_u'] ** 2 + S['sd_v'] ** 2; trans = S['sd_c'] ** 2 + S['sd_eps'] ** 2
rho = pers / (pers + trans)
res['latent_year_to_year_correlation'] = dict(mean=float(rho.mean()), lo=float(np.quantile(rho, .05)), hi=float(np.quantile(rho, .95)))
# reliability curves as a function of responses n (p=0.8 -> s = 40/sqrt(n))
ns = np.array([10, 15, 20, 30, 50, 75, 100, 150, 200, 300, 500, 1000])
rows = []
for n in ns:
    s_n2 = (100 * np.sqrt(0.8 * 0.2 / n)) ** 2
    rel_current = (pers + trans) / (pers + trans + s_n2)
    rel_persist = pers / (pers + trans + s_n2)
    rel_theme_specific = S['sd_v'] ** 2 / (S['sd_v'] ** 2 + trans + s_n2)   # single-year theme deviation vs persistent theme-specific effect
    rows.append(dict(n=n, s=np.sqrt(s_n2), rel_current=rel_current.mean(), rel_persistent=rel_persist.mean(), rel_theme_specific=rel_theme_specific.mean(),
                     rel_persistent_lo=np.quantile(rel_persist, .05), rel_persistent_hi=np.quantile(rel_persist, .95)))
rel = pd.DataFrame(rows); rel.to_csv('../out/full_reliability_curve.csv', index=False)
res['reliability_curve'] = rel.round(3).to_dict('records')

# ---------- per-cell posterior classification (2026 position + 2025->2026 trajectory) ----------
d['theta_mean'] = theta.mean(0); d['theta_sd'] = theta.std(0)
d['theta_lo90'] = np.quantile(theta, .05, 0); d['theta_hi90'] = np.quantile(theta, .95, 0)
d['P_above'] = (theta > THR).mean(0); d['P_below'] = (theta < -THR).mean(0); d['P_inline'] = 1 - d['P_above'] - d['P_below']
d['obs_pos'] = band(d['D']); d['ofs_modal'] = np.array(['above', 'inline', 'below'])[np.argmax(d[['p_above', 'p_inline', 'p_below']].values, 1)]
d['bayes_modal_pos'] = np.array(['above', 'inline', 'below'])[np.argmax(d[['P_above', 'P_inline', 'P_below']].values, 1)]
d['bayes_maxp_pos'] = d[['P_above', 'P_inline', 'P_below']].max(1)
d.to_csv('../out/full_M1_cell_posteriors.csv', index=False)

# unit x theme table for 2026 with trajectory
key = d.set_index(['u_i', 'q_i', 'Year'])
rows = []
pos26 = d[d.Year == 2026]; pos25 = d[d.Year == 2025].set_index(['u_i', 'q_i'])
for i, r in pos26.iterrows():
    th26 = theta[:, i]
    row = dict(UKPRN=r.UKPRN, provider=r.provider, theme=r.theme, D_2026=r.D, s_2026=r.s, n_2026=r.n_resp,
               theta26_mean=th26.mean(), theta26_lo90=np.quantile(th26, .05), theta26_hi90=np.quantile(th26, .95),
               P_above=r.P_above, P_inline=r.P_inline, P_below=r.P_below, obs_pos=r.obs_pos, bayes_pos=r.bayes_modal_pos)
    if (r.u_i, r.q_i) in pos25.index:
        j = pos25.loc[(r.u_i, r.q_i)]
        j_i = int(d.index[(d.u_i == r.u_i) & (d.q_i == r.q_i) & (d.Year == 2025)][0])
        dth = th26 - theta[:, j_i]
        row.update(D_2025=j.D, dD_obs=r.D - j.D, dtheta_mean=dth.mean(), dtheta_lo90=np.quantile(dth, .05), dtheta_hi90=np.quantile(dth, .95),
                   P_improving=(dth > THR).mean(), P_declining=(dth < -THR).mean())
        row['P_stable'] = 1 - row['P_improving'] - row['P_declining']
        row['obs_trj'] = traj(np.array([r.D - j.D]))[0]
        row['bayes_trj'] = ['improving', 'stable', 'declining'][int(np.argmax([row['P_improving'], row['P_stable'], row['P_declining']]))]
        # joint 3x3 posterior: probability of each state (position, trajectory) jointly
        pos_lab = np.where(th26 > THR, 'above', np.where(th26 < -THR, 'below', 'inline'))
        trj_lab = np.where(dth > THR, 'improving', np.where(dth < -THR, 'declining', 'stable'))
        joint = pd.Series(pos_lab + '/' + trj_lab).value_counts(normalize=True)
        row['bayes_state_modal'] = joint.index[0]; row['bayes_state_maxp'] = float(joint.iloc[0])
        row['obs_state'] = row['obs_pos'] + '/' + row['obs_trj']
        row['P_obs_state'] = float(joint.get(row['obs_state'], 0.0))
    # persistence: persistent component u+v
    uv = S['u'][:, r.u_i] + S['v'][:, int(r.v_i)] + S['mu'] + S['alpha'][:, r.q_i]
    row.update(persist_mean=uv.mean(), P_persist_below=(uv < -THR).mean(), P_persist_above=(uv > THR).mean())
    # all-years-below probability
    cell_idx = d.index[(d.u_i == r.u_i) & (d.q_i == r.q_i)].values
    row['n_years'] = len(cell_idx)
    row['P_below_every_year'] = float((theta[:, cell_idx] < -THR).all(1).mean())
    row['P_above_every_year'] = float((theta[:, cell_idx] > THR).all(1).mean())
    rows.append(row)
ut = pd.DataFrame(rows)
ut.to_csv('../out/full_M1_unit_theme_2026.csv', index=False)

# ---------- churn: observed 3x3 vs Bayesian modal 3x3 (cells with 2025 & 2026) ----------
c = ut[ut['obs_state'].notna()].copy()
res['churn_obs_vs_bayes'] = dict(n=int(len(c)), full=float((c.obs_state != c.bayes_state_modal).mean()),
                                 position=float((c.obs_pos != c.bayes_pos).mean()), trajectory=float((c.obs_trj != c.bayes_trj).mean()),
                                 mean_P_of_observed_state=float(c.P_obs_state.mean()),
                                 share_obs_state_prob_below_0_5=float((c.P_obs_state < 0.5).mean()),
                                 share_bayes_state_maxp_gt_0_6=float((c.bayes_state_maxp > 0.6).mean()))
res['churn_ofs_modal_vs_obs_position_2026'] = float((pos26.ofs_modal != pos26.obs_pos).mean())
res['churn_bayes_vs_ofs_modal_position_2026'] = float((pos26.bayes_modal_pos != pos26.ofs_modal).mean())
pd.crosstab(c.obs_pos, c.bayes_pos).to_csv('../out/T_transition_position_obs_vs_bayes.csv')
pd.crosstab(c.obs_trj, c.bayes_trj).to_csv('../out/T_transition_trajectory_obs_vs_bayes.csv')
c['band'] = pd.cut(c['n_2026'], [0, 50, 100, 200, np.inf], labels=['<50', '50-99', '100-199', '200+'])
cb = c.groupby('band', observed=True).apply(lambda g: pd.Series({'n': len(g), 'full_churn': (g.obs_state != g.bayes_state_modal).mean(),
                                                                'position_churn': (g.obs_pos != g.bayes_pos).mean(), 'trajectory_churn': (g.obs_trj != g.bayes_trj).mean(),
                                                                'mean_P_obs_state': g.P_obs_state.mean()}))
cb.to_csv('../out/T_churn_obs_vs_bayes_by_size.csv'); res['churn_obs_vs_bayes_by_size'] = cb.round(3).to_dict()
# how many observed "materially declining/improving" cells survive as credible (P>0.8)?
res['trajectory_credibility'] = dict(
    obs_improving=int((c.obs_trj == 'improving').sum()), obs_improving_P_gt_0_8=int(((c.obs_trj == 'improving') & (c.P_improving > 0.8)).sum()),
    obs_declining=int((c.obs_trj == 'declining').sum()), obs_declining_P_gt_0_8=int(((c.obs_trj == 'declining') & (c.P_declining > 0.8)).sum()),
    obs_below=int((c.obs_pos == 'below').sum()), obs_below_P_gt_0_8=int(((c.obs_pos == 'below') & (c.P_below > 0.8)).sum()),
    obs_above=int((c.obs_pos == 'above').sum()), obs_above_P_gt_0_8=int(((c.obs_pos == 'above') & (c.P_above > 0.8)).sum()),
    persist_below_P_gt_0_8=int((ut.P_persist_below > 0.8).sum()), persist_above_P_gt_0_8=int((ut.P_persist_above > 0.8).sum()),
    below_every_year_P_gt_0_8=int((ut.P_below_every_year > 0.8).sum()), n_units_4yrs=int((ut.n_years == 4).sum()))

# ---------- RTM on latent vs observed scale ----------
obs = c[['D_2025', 'D_2026']].values
res['rtm_observed'] = dict(corr_D25_dD=float(np.corrcoef(obs[:, 0], obs[:, 1] - obs[:, 0])[0, 1]), slope=float(np.polyfit(obs[:, 0], obs[:, 1], 1)[0]),
                           var_dD=float(np.var(obs[:, 1] - obs[:, 0])))
# latent: compute per draw
i25 = [int(d.index[(d.u_i == r.u_i) & (d.q_i == r.q_i) & (d.Year == 2025)][0]) for r in pos26.itertuples() if (r.u_i, r.q_i) in pos25.index]
i26 = [int(i) for i, r in pos26.iterrows() if (r.u_i, r.q_i) in pos25.index]
T25 = theta[:, i25]; T26 = theta[:, i26]
corr_l = np.array([np.corrcoef(T25[k], T26[k] - T25[k])[0, 1] for k in range(0, Sd, 4)])
slope_l = np.array([np.polyfit(T25[k], T26[k], 1)[0] for k in range(0, Sd, 4)])
var_l = np.var(T26 - T25, axis=1)
res['rtm_latent'] = dict(corr_theta25_dtheta=float(corr_l.mean()), corr_lo=float(np.quantile(corr_l, .05)), corr_hi=float(np.quantile(corr_l, .95)),
                         slope=float(slope_l.mean()), var_dtheta=float(var_l.mean()), share_observed_change_variance_that_is_latent=float(var_l.mean() / np.var(obs[:, 1] - obs[:, 0])))
# cohort shocks
cs = S['c']  # (S, n_c)
res['cohort_shocks'] = dict(sd_c_mean=float(S['sd_c'].mean()), share_unit_years_abs_c_gt_2_5=float((np.abs(cs.mean(0)) > 2.5).mean()),
                            share_unit_years_P_abs_c_gt_2_5_gt_0_8=float((((np.abs(cs) > 2.5).mean(0)) > 0.8).mean()))
# corroboration: within provider-year, sign concordance of observed dD across the 7 themes
cc = c.groupby('UKPRN')['dD_obs'].agg(lambda x: max((x > 0).mean(), (x < 0).mean()))
res['within_provider_sign_concordance_dD_2025_26'] = dict(mean=float(cc.mean()), share_all7_same_sign=float((cc == 1).mean()), n_providers=int(len(cc)))
# permutation reference for concordance (shuffle dD across providers within theme)
rng = np.random.default_rng(5); perm = []
cc_df = c[['UKPRN', 'theme', 'dD_obs']].copy()
for _ in range(200):
    cc_df['sh'] = cc_df.groupby('theme')['dD_obs'].transform(lambda x: rng.permutation(x.values))
    perm.append(cc_df.groupby('UKPRN')['sh'].agg(lambda x: max((x > 0).mean(), (x < 0).mean())).mean())
res['within_provider_sign_concordance_dD_2025_26']['permutation_null_mean'] = float(np.mean(perm))

# ---------- M1 vs M0 uncertainty comparison ----------
S0 = fits['M0_nocohort']
uv1 = S['u'][:, d.u_i.values] + S['v'][:, d.v_i.values]; uv0 = S0['u'][:, d.u_i.values] + S0['v'][:, d.v_i.values]
res['M1_vs_M0'] = dict(mean_post_sd_persistent_effect_M1=float(uv1.std(0).mean()), mean_post_sd_persistent_effect_M0=float(uv0.std(0).mean()),
                       waic_M1=res['M1_cohort']['waic'], waic_M0=res['M0_nocohort']['waic'])

# ---------- QMUL position in sector ----------
q = ut[ut.provider.str.contains('Queen Mary')].copy()
for col in ['theta26_mean', 'dtheta_mean', 'persist_mean']:
    q[col + '_pct'] = [float((ut[col].dropna() < x).mean()) for x in q[col]]
q_u = S['u'][:, idx['units'].get_loc(10007775)]
res['QMUL'] = dict(u_mean=float(q_u.mean()), u_lo=float(np.quantile(q_u, .05)), u_hi=float(np.quantile(q_u, .95)),
                   u_percentile=float((S['u'].mean(0) < q_u.mean()).mean()),
                   table=q[['theme', 'D_2025', 'D_2026', 'dD_obs', 'theta26_mean', 'theta26_lo90', 'theta26_hi90', 'P_above', 'P_inline', 'P_below',
                            'dtheta_mean', 'P_improving', 'P_stable', 'P_declining', 'obs_state', 'bayes_state_modal', 'bayes_state_maxp', 'persist_mean',
                            'P_persist_below', 'theta26_mean_pct', 'dtheta_mean_pct']].round(3).to_dict('records'))
# unit-level trajectory: provider cohort effect 2026 vs 2025 (u+c) - QMUL vs sector
c_idx = idx['ut'].set_index(['u_i', 't_i'])['c_i']
prov_traj = []
for u_i_ in range(idx['n_u']):
    if (u_i_, 2) in c_idx.index and (u_i_, 3) in c_idx.index:
        dc = S['c'][:, c_idx.loc[(u_i_, 3)]] - S['c'][:, c_idx.loc[(u_i_, 2)]]
        prov_traj.append(dict(UKPRN=idx['units'][u_i_], dc_mean=dc.mean(), P_dc_gt_2_5=(dc > 2.5).mean()))
pt = pd.DataFrame(prov_traj); pt.to_csv('../out/full_M1_provider_cohort_change_2025_26.csv', index=False)
res['QMUL']['cohort_change_2025_26'] = pt[pt.UKPRN == 10007775].round(3).to_dict('records')
res['QMUL']['cohort_change_percentile'] = float((pt.dc_mean < pt[pt.UKPRN == 10007775].dc_mean.iloc[0]).mean())
json.dump(res, open('../out/full_summary.json', 'w'), indent=2, default=str)
print(json.dumps({k: v for k, v in res.items() if k not in ('reliability_curve',)}, indent=2, default=str))
