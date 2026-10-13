"""Step 9a: FINAL Model B (2023-2026) under the guide specification:
   theta_pqt = mu + alpha_q (sum-to-zero, N(0,3)) + gamma_t + u_p + v_pq + c_pt + h_pqt,  h random walk from 0 (2023), no idiosyncratic eps.
   4 chains x (2000 warm-up + 2000 draws), target accept 0.95.  Diagnostics: divergences, bulk/tail ESS, R-hat, tree depth, trace plots.
   Then: PPCs, variance table (median + 95% CI), latent 2026 status & 2025->26 trajectory probabilities, joint 3x3 posterior,
   observed-vs-Bayesian churn incl. 70/80/90% confidence sensitivity, persistence/recovery/deterioration, five action categories,
   missed problems, shrinkage vs headcount/SD, QMUL tables (2023-26; conventional vs benchmark vs Bayesian), figures."""
import pandas as pd, numpy as np, json, time, sys, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from bayes_model import *
plt.style.use('seaborn-v0_8-whitegrid'); plt.rcParams.update({'figure.dpi': 150, 'font.size': 10, 'axes.titlesize': 11, 'axes.titleweight': 'bold'})
BLUE, ORANGE, GREEN, RED, GREY = '#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8c8c8c'
THR = 2.5; NW, NS, NC = 2000, 2000, 4
THEMES = {1: 'Teaching', 2: 'Learning opps', 3: 'Assessment & feedback', 4: 'Academic support', 5: 'Organisation & mgmt', 6: 'Learning resources', 7: 'Student voice'}
PRI = dict(PRIORS); PRI.update(sd_eta=3.0, alpha_fixed_sd=3.0)   # guide: sigma_change ~ HalfNormal(3), alpha_q ~ N(0,3)
prov = pd.read_csv('../data/sector_cah10_provider_theme.csv')
idx = build_index(prov, 'UKPRN'); d = idx['df']
res = {'spec': 'mu + alpha_q[N(0,3), sum-to-zero] + gamma_t[HalfNormal(2)] + u_p[HN(5)] + v_pq[HN(5)] + c_pt[HN(5)] + h_pqt[RW, sigma_change HN(3)], no eps; D ~ N(theta, s^2)',
       'draws': f'{NC} chains x ({NW} warm-up + {NS} draws), target_accept 0.95, seed 100'}
def band(x): return np.where(x > THR, 'above', np.where(x < -THR, 'below', 'inline'))
def traj(x): return np.where(x > THR, 'improving', np.where(x < -THR, 'declining', 'stable'))

# ---------------- fit (or resume from saved draws) ----------------
import os
if os.path.exists('../out/final_B_samples.npz') and os.path.exists('../out/final_B_theta_draws.npy') and '--refit' not in sys.argv:
    S = dict(np.load('../out/final_B_samples.npz')); theta = np.load('../out/final_B_theta_draws.npy')
    summ = pd.read_csv('../out/final_B_hyper.csv', index_col=0); res['hyper'] = summ.round(3).to_dict()
    res['diag'] = json.load(open('../out/final_B_diag.json')) if os.path.exists('../out/final_B_diag.json') else 'see log'
    Sd = theta.shape[0]
else:
    t0 = time.time()
    mcmc, kw = fit(idx, cohort=True, dynamic=True, eps=False, alpha_prior='fixed', priors=PRI, num_warmup=NW, num_samples=NS, num_chains=NC, target_accept=0.95, seed=100)
    summ, diag, idata = diagnostics(mcmc, params=('mu', 'sd_gamma', 'sd_u', 'sd_v', 'sd_c', 'sd_eta'))
    res['elapsed_s'] = time.time() - t0; res['diag'] = diag; json.dump(diag, open('../out/final_B_diag.json', 'w'))
    print('ModelB_G', f'{res["elapsed_s"]:.0f}s', diag, file=sys.stderr)
    summ.to_csv('../out/final_B_hyper.csv'); res['hyper'] = summ.round(3).to_dict()
    S = flatten_samples(mcmc)
    np.savez_compressed('../out/final_B_samples.npz', **{k: np.asarray(v).astype(np.float32) for k, v in S.items() if k not in ('m',)})
    theta = np.asarray(S['m']).astype(np.float32)   # latent = m exactly (no eps)
    np.save('../out/final_B_theta_draws.npy', theta); d.to_csv('../out/final_B_obs_index.csv', index=False)
    Sd = theta.shape[0]
    fig, axes = plt.subplots(2, 3, figsize=(12, 5))
    for a, pn in zip(axes.flat, ['sd_u', 'sd_v', 'sd_c', 'sd_eta', 'sd_gamma', 'mu']):
        arr = np.asarray(idata.posterior[pn])
        for ch in range(arr.shape[0]): a.plot(arr[ch], lw=0.4, alpha=0.7)
        a.set_title(pn); a.set_xlabel('draw')
    plt.tight_layout(); plt.savefig('../fig/B0_traceplots.png', bbox_inches='tight'); plt.close()
    del mcmc, idata

# ---------------- variance table (median + 95% CI) ----------------
comp = {'provider (persistent)': S['sd_u'] ** 2, 'provider x theme (persistent)': S['sd_v'] ** 2, 'provider x year cohort shock': S['sd_c'] ** 2,
        'theme-specific change (random-walk innovation, per year)': S['sd_eta'] ** 2, 'year (sector-wide)': S['sd_gamma'] ** 2}
rows = []
tot = sum(comp[k] for k in ['provider (persistent)', 'provider x theme (persistent)', 'provider x year cohort shock', 'theme-specific change (random-walk innovation, per year)'])
for k, v in comp.items():
    rows.append(dict(component=k, sd_median=float(np.median(np.sqrt(v))), sd_lo95=float(np.quantile(np.sqrt(v), .025)), sd_hi95=float(np.quantile(np.sqrt(v), .975)),
                     var_median=float(np.median(v)), var_lo95=float(np.quantile(v, .025)), var_hi95=float(np.quantile(v, .975)),
                     share_median=float(np.median(v / tot)), share_lo95=float(np.quantile(v / tot, .025)), share_hi95=float(np.quantile(v / tot, .975))))
s2 = d.s.values ** 2
rows.append(dict(component='OfS sampling variance (median cell)', var_median=float(np.median(s2)), sd_median=float(np.sqrt(np.median(s2)))))
rows.append(dict(component='OfS sampling variance (mean cell)', var_median=float(np.mean(s2)), sd_median=float(np.sqrt(np.mean(s2)))))
vt = pd.DataFrame(rows); vt.to_csv('../out/final_B_variance_table.csv', index=False); res['variance_table'] = vt.round(3).to_dict('records')
pers = S['sd_u'] ** 2 + S['sd_v'] ** 2; trans = S['sd_c'] ** 2 + S['sd_eta'] ** 2
res['cohort_vs_theme_specific_change'] = dict(share_of_annual_systematic_movement_cohort_median=float(np.median(S['sd_c'] ** 2 / trans)),
                                             lo95=float(np.quantile(S['sd_c'] ** 2 / trans, .025)), hi95=float(np.quantile(S['sd_c'] ** 2 / trans, .975)),
                                             latent_year_to_year_corr_median=float(np.median(pers / (pers + trans))))

# ---------------- posterior predictive checks ----------------
rng = np.random.default_rng(9); sub = rng.choice(Sd, 400, replace=False)
Drep = theta[sub] + d.s.values[None, :] * rng.standard_normal((400, len(d)))
D = d.D.values
def stats_fn(x):
    o = dict(mean=x.mean(1), sd=x.std(1), q01=np.quantile(x, .01, 1), q99=np.quantile(x, .99, 1), share_abs_gt10=(np.abs(x) > 10).mean(1))
    return o
obs_st = {k: float(np.asarray(v).reshape(-1)[0]) for k, v in stats_fn(D[None, :]).items()}
rep_st = stats_fn(Drep)
ppc = {k: dict(observed=obs_st[k], rep_mean=float(rep_st[k].mean()), rep_lo95=float(np.quantile(rep_st[k], .025)), rep_hi95=float(np.quantile(rep_st[k], .975)),
                p_value=float((rep_st[k] >= obs_st[k]).mean())) for k in obs_st}
# by theme SD, by year SD/mean, by precision band SD
for grp, name in [(d.theme.values, 'theme'), (d.Year.values, 'year'), (pd.cut(d.n_resp, [0, 50, 100, 200, np.inf], labels=['<50', '50-99', '100-199', '200+']).astype(str).values, 'size')]:
    for g in np.unique(grp):
        mk = grp == g
        o = D[mk].std(); r = Drep[:, mk].std(1)
        ppc[f'sd_{name}_{g}'] = dict(observed=float(o), rep_mean=float(r.mean()), rep_lo95=float(np.quantile(r, .025)), rep_hi95=float(np.quantile(r, .975)), p_value=float((r >= o).mean()))
        if name == 'year':
            o = D[mk].mean(); r = Drep[:, mk].mean(1)
            ppc[f'mean_year_{g}'] = dict(observed=float(o), rep_mean=float(r.mean()), rep_lo95=float(np.quantile(r, .025)), rep_hi95=float(np.quantile(r, .975)), p_value=float((r >= o).mean()))
# cross-theme cohort dependence: within-provider sign concordance of dD 2025->26, and mean pairwise correlation of dD across themes
i25 = d.index[d.Year == 2025]; i26 = d.index[d.Year == 2026]
k25 = d.loc[i25].set_index(['u_i', 'q_i']).index; k26 = d.loc[i26].set_index(['u_i', 'q_i']).index
common = k25.intersection(k26)
pos25 = d.loc[i25].reset_index().set_index(['u_i', 'q_i']).loc[common, 'index'].values; pos26 = d.loc[i26].reset_index().set_index(['u_i', 'q_i']).loc[common, 'index'].values
units = np.array([c_[0] for c_ in common]); themes_ = np.array([c_[1] for c_ in common])
def concordance(dD):
    df_ = pd.DataFrame({'u': units, 'dD': dD})
    return df_.groupby('u')['dD'].agg(lambda x: max((x > 0).mean(), (x < 0).mean())).mean()
def pairwise_corr(dD):
    piv = pd.DataFrame({'u': units, 'q': themes_, 'dD': dD}).pivot(index='u', columns='q', values='dD')
    cm = piv.corr().values; return cm[np.triu_indices(7, 1)].mean()
o_c = concordance(D[pos26] - D[pos25]); o_r = pairwise_corr(D[pos26] - D[pos25])
r_c = np.array([concordance(Drep[k, pos26] - Drep[k, pos25]) for k in range(400)]); r_r = np.array([pairwise_corr(Drep[k, pos26] - Drep[k, pos25]) for k in range(400)])
ppc['crosstheme_sign_concordance_dD'] = dict(observed=float(o_c), rep_mean=float(r_c.mean()), rep_lo95=float(np.quantile(r_c, .025)), rep_hi95=float(np.quantile(r_c, .975)), p_value=float((r_c >= o_c).mean()))
ppc['crosstheme_mean_pairwise_corr_dD'] = dict(observed=float(o_r), rep_mean=float(r_r.mean()), rep_lo95=float(np.quantile(r_r, .025)), rep_hi95=float(np.quantile(r_r, .975)), p_value=float((r_r >= o_r).mean()))
# also same for level D within year 2026 (cross-theme correlation of D)
def pairwise_corr_level(x):
    dd = d.loc[i26]; piv = pd.DataFrame({'u': dd.u_i.values, 'q': dd.q_i.values, 'x': x}).pivot(index='u', columns='q', values='x'); cm = piv.corr().values; return cm[np.triu_indices(7, 1)].mean()
o_l = pairwise_corr_level(D[i26]); r_l = np.array([pairwise_corr_level(Drep[k, i26]) for k in range(400)])
ppc['crosstheme_mean_pairwise_corr_D2026'] = dict(observed=float(o_l), rep_mean=float(r_l.mean()), rep_lo95=float(np.quantile(r_l, .025)), rep_hi95=float(np.quantile(r_l, .975)), p_value=float((r_l >= o_l).mean()))
pd.DataFrame(ppc).T.to_csv('../out/final_B_ppc.csv'); res['ppc'] = ppc
# PPC figure
fig, ax = plt.subplots(1, 4, figsize=(14, 3.6))
for a, (k, ttl) in zip(ax, [('sd', 'SD of all gaps'), ('q01', '1st percentile'), ('crosstheme_sign_concordance_dD', 'within-provider sign concordance of ΔD'), ('crosstheme_mean_pairwise_corr_dD', 'mean cross-theme corr of ΔD')]):
    r = {'sd': rep_st['sd'], 'q01': rep_st['q01'], 'crosstheme_sign_concordance_dD': r_c, 'crosstheme_mean_pairwise_corr_dD': r_r}[k]
    a.hist(r, bins=30, color=GREY, alpha=0.7); a.axvline(ppc[k]['observed'], color=RED, lw=2); a.set_title(ttl, fontsize=9); a.set_xlabel('replicated (grey) vs observed (red)')
plt.tight_layout(); plt.savefig('../fig/B5_ppc.png', bbox_inches='tight'); plt.close()

# ---------------- per-cell posteriors and 2026 unit x theme table ----------------
d['theta_mean'] = theta.mean(0); d['theta_sd'] = theta.std(0); d['theta_lo95'] = np.quantile(theta, .025, 0); d['theta_hi95'] = np.quantile(theta, .975, 0)
d['P_above'] = (theta > THR).mean(0); d['P_below'] = (theta < -THR).mean(0); d['P_inline'] = 1 - d.P_above - d.P_below
d['shrink'] = d.theta_mean - d.D
d['obs_pos'] = band(d.D.round(1)); d['bayes_pos'] = np.array(['above', 'inline', 'below'])[np.argmax(d[['P_above', 'P_inline', 'P_below']].values, 1)]
d.to_csv('../out/final_B_cell_posteriors.csv', index=False)
loc26 = {(r.u_i, r.q_i): i for i, r in d[d.Year == 2026].iterrows()}; loc25 = {(r.u_i, r.q_i): i for i, r in d[d.Year == 2025].iterrows()}
rows = []
for key, i in loc26.items():
    r = d.loc[i]; th26 = theta[:, i]
    row = dict(UKPRN=r.UKPRN, provider=r.provider, theme=int(r.theme), D_2026=r.D, s_2026=r.s, n_2026=r.n_resp, headcount_2026=r.headcount, contrib_2026=r.contrib,
               theta26_mean=float(th26.mean()), theta26_lo95=float(np.quantile(th26, .025)), theta26_hi95=float(np.quantile(th26, .975)),
               P_above=float(r.P_above), P_inline=float(r.P_inline), P_below=float(r.P_below), obs_pos=r.obs_pos, bayes_pos=r.bayes_pos, shrink=float(r.shrink))
    if key in loc25:
        j = loc25[key]; th25 = theta[:, j]; dth = th26 - th25
        row.update(D_2025=d.loc[j, 'D'], dD_obs=float(round(r.D - d.loc[j, 'D'], 1)), dtheta_mean=float(dth.mean()), dtheta_lo95=float(np.quantile(dth, .025)), dtheta_hi95=float(np.quantile(dth, .975)),
                   P_improving=float((dth > THR).mean()), P_declining=float((dth < -THR).mean()))
        row['P_stable'] = 1 - row['P_improving'] - row['P_declining']
        row['obs_trj'] = traj(np.array([row['dD_obs']]))[0]
        row['bayes_trj'] = ['improving', 'stable', 'declining'][int(np.argmax([row['P_improving'], row['P_stable'], row['P_declining']]))]
        pl = np.where(th26 > THR, 'above', np.where(th26 < -THR, 'below', 'inline')); tl = np.where(dth > THR, 'improving', np.where(dth < -THR, 'declining', 'stable'))
        joint = pd.Series(pl + '/' + tl).value_counts(normalize=True)
        row['bayes_state_modal'] = joint.index[0]; row['bayes_state_maxp'] = float(joint.iloc[0]); row['obs_state'] = row['obs_pos'] + '/' + row['obs_trj']; row['P_obs_state'] = float(joint.get(row['obs_state'], 0.0))
        for st in ['below/declining', 'below/stable', 'below/improving', 'inline/declining', 'inline/stable', 'inline/improving', 'above/declining', 'above/stable', 'above/improving']:
            row['P_' + st] = float(joint.get(st, 0.0))
        # persistence / recovery / deterioration (guide s20)
        row['P_persistent_weak_25_26'] = float(((th25 < -THR) & (th26 < -THR)).mean()); row['P_persistent_strong_25_26'] = float(((th25 > THR) & (th26 > THR)).mean())
        row['P_recovery_25_26'] = float(((th25 < -THR) & (th26 >= -THR)).mean()); row['P_emerging_deterioration_25_26'] = float(((th25 >= -THR) & (th26 < -THR)).mean())
    cell = d.index[(d.u_i == r.u_i) & (d.q_i == r.q_i)].values
    row['n_years'] = len(cell); row['P_below_every_year'] = float((theta[:, cell] < -THR).all(1).mean()); row['P_above_every_year'] = float((theta[:, cell] > THR).all(1).mean())
    uv = S['mu'] + S['alpha'][:, r.q_i] + S['u'][:, r.u_i] + S['v'][:, int(r.v_i)]
    row['persist_mean'] = float(uv.mean()); row['P_persist_below'] = float((uv < -THR).mean()); row['P_persist_above'] = float((uv > THR).mean())
    rows.append(row)
ut = pd.DataFrame(rows)
# ---- five action categories (guide s21); thresholds are documented practical rules, with 0.7/0.8/0.9 sensitivity ----
def categorise(u, thr):
    c = u.copy(); cat = pd.Series('no action / in line', index=c.index)
    noisy = ((c.obs_pos != 'inline') | (c.obs_trj != 'stable')) & (c.bayes_state_maxp < 0.5)
    cat[noisy] = '5 uncertain / noisy signal'
    cat[(c.P_persist_above > thr) & (c.P_above > 0.5)] = '4 sustained high-confidence strength'
    cat[(c.P_above > 0.5) & (c.P_declining > thr)] = '3 strong but credibly deteriorating'
    cat[(c.P_below > 0.5) & (c.P_improving > thr)] = '2 weak but credibly improving'
    cat[(c.P_persistent_weak_25_26 > thr) & ~((c.P_below > 0.5) & (c.P_improving > thr))] = '1 persistent high-confidence weakness'
    return cat
for thr in [0.7, 0.8, 0.9]:
    ut[f'category_{int(thr*100)}'] = categorise(ut, thr)
ut.to_csv('../out/final_B_unit_theme_2026.csv', index=False)
res['action_categories'] = {f'{int(t*100)}%': ut[f'category_{int(t*100)}'].value_counts().to_dict() for t in [0.7, 0.8, 0.9]}

# ---------------- churn observed vs Bayesian (guide s14) with confidence sensitivity ----------------
c = ut[ut.obs_state.notna()].copy()
res['churn_obs_vs_bayes'] = dict(n=int(len(c)), full=float((c.obs_state != c.bayes_state_modal).mean()), position=float((c.obs_pos != c.bayes_pos).mean()),
                                 trajectory=float((c.obs_trj != c.bayes_trj).mean()), mean_P_obs_state=float(c.P_obs_state.mean()))
pd.crosstab(c.obs_state, c.bayes_state_modal).to_csv('../out/final_B_transition_3x3.csv')
pd.crosstab(c.obs_pos, c.bayes_pos).to_csv('../out/final_B_transition_position.csv'); pd.crosstab(c.obs_trj, c.bayes_trj).to_csv('../out/final_B_transition_trajectory.csv')
sens = {}
for thr in [0.7, 0.8, 0.9]:
    sens[f'{int(thr*100)}%'] = dict(
        obs_below_n=int((c.obs_pos == 'below').sum()), obs_below_share_losing_support=float((c.loc[c.obs_pos == 'below', 'P_below'] < thr).mean()),
        obs_above_n=int((c.obs_pos == 'above').sum()), obs_above_share_losing_support=float((c.loc[c.obs_pos == 'above', 'P_above'] < thr).mean()),
        obs_declining_n=int((c.obs_trj == 'declining').sum()), obs_declining_share_losing_support=float((c.loc[c.obs_trj == 'declining', 'P_declining'] < thr).mean()),
        obs_improving_n=int((c.obs_trj == 'improving').sum()), obs_improving_share_losing_support=float((c.loc[c.obs_trj == 'improving', 'P_improving'] < thr).mean()),
        obs_below_declining_n=int(((c.obs_pos == 'below') & (c.obs_trj == 'declining')).sum()),
        obs_below_declining_share_losing_support=float((c.loc[(c.obs_pos == 'below') & (c.obs_trj == 'declining'), 'P_below/declining'] < thr).mean()),
        share_all_cells_with_a_state_at_or_above_thr=float((c.bayes_state_maxp >= thr).mean()))
res['confidence_sensitivity'] = sens
# missed problems (guide s26D): observed in line but substantial posterior probability of latent underperformance
inl = c[c.obs_pos == 'inline']
res['missed_problems'] = dict(n_obs_inline=int(len(inl)), share_P_below_gt_0_5=float((inl.P_below > 0.5).mean()), share_P_below_gt_0_7=float((inl.P_below > 0.7).mean()),
                              share_P_persist_below_gt_0_8=float((inl.P_persist_below > 0.8).mean()), share_P_above_gt_0_5=float((inl.P_above > 0.5).mean()),
                              n_P_below_gt_0_7=int((inl.P_below > 0.7).sum()))
res['false_alarms'] = dict(obs_below_n=int((c.obs_pos == 'below').sum()), share_P_below_lt_0_5=float((c.loc[c.obs_pos == 'below', 'P_below'] < 0.5).mean()),
                           obs_declining_n=int((c.obs_trj == 'declining').sum()), share_P_declining_lt_0_5=float((c.loc[c.obs_trj == 'declining', 'P_declining'] < 0.5).mean()),
                           expected_precision_below=float(c.loc[c.obs_pos == 'below', 'P_below'].mean()), expected_precision_declining=float(c.loc[c.obs_trj == 'declining', 'P_declining'].mean()))
res['persistence_summary'] = dict(n=int(len(c)), n_persistent_weak_P_gt_0_8=int((c.P_persistent_weak_25_26 > 0.8).sum()), n_persistent_strong_P_gt_0_8=int((c.P_persistent_strong_25_26 > 0.8).sum()),
                                  n_recovery_P_gt_0_5=int((c.P_recovery_25_26 > 0.5).sum()), n_emerging_det_P_gt_0_5=int((c.P_emerging_deterioration_25_26 > 0.5).sum()),
                                  n_below_every_year_P_gt_0_8=int((ut.P_below_every_year > 0.8).sum()), n_above_every_year_P_gt_0_8=int((ut.P_above_every_year > 0.8).sum()))

# ---------------- shrinkage analysis (guide s19) ----------------
d['abs_shrink'] = d.shrink.abs()
d['hc_band'] = pd.cut(d.headcount, [0, 20, 50, 100, 200, 500, np.inf], labels=['<20', '20-49', '50-99', '100-199', '200-499', '500+'])
sh = d.groupby('hc_band', observed=True).agg(n=('abs_shrink', 'size'), median_abs_shrink=('abs_shrink', 'median'), mean_abs_shrink=('abs_shrink', 'mean'), median_s=('s', 'median'),
                                             median_ratio_shrink_to_absD=('shrink', lambda x: 0)).drop(columns='median_ratio_shrink_to_absD')
sh['median_abs_shrink_over_absD'] = d.groupby('hc_band', observed=True).apply(lambda g: (g.abs_shrink / g.D.abs().clip(lower=0.1)).median())
sh.to_csv('../out/final_B_shrinkage_by_headcount.csv'); res['shrinkage_by_headcount'] = sh.round(2).to_dict()
res['shrinkage_corr'] = dict(spearman_abs_shrink_vs_headcount=float(d[['abs_shrink', 'headcount']].corr(method='spearman').iloc[0, 1]), spearman_abs_shrink_vs_s=float(d[['abs_shrink', 's']].corr(method='spearman').iloc[0, 1]))
fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))
a = ax[0]; a.scatter(d.D, d.theta_mean, s=np.clip(d.headcount, 10, 600) / 8, c=np.log10(d.headcount), cmap='viridis', alpha=0.5, edgecolor='none')
a.plot([-40, 40], [-40, 40], color=GREY, ls='--', lw=0.8); a.set_xlim(-45, 45); a.set_ylim(-30, 30); a.set_xlabel('Observed benchmark gap D (pp)'); a.set_ylabel('Posterior mean latent gap θ (pp)'); a.set_title('Observed vs latent, all cells 2023–26 (colour = log₁₀ headcount)')
a = ax[1]; a.scatter(d.headcount, d.abs_shrink, s=6, alpha=0.35, color=BLUE); a.set_xscale('log'); a.set_xlabel('Publication response headcount (log)'); a.set_ylabel('|posterior mean − observed| (pp)'); a.set_title('Shrinkage vs cohort size')
med = d.groupby('hc_band', observed=True)['abs_shrink'].median(); a.plot([15, 35, 75, 150, 350, 800][:len(med)], med.values, color=RED, marker='o', label='band median'); a.legend()
a = ax[2]; a.scatter(d.s, d.abs_shrink, s=6, alpha=0.35, color=ORANGE); a.set_xlabel('OfS standard deviation (pp)'); a.set_ylabel('|posterior mean − observed| (pp)'); a.set_title('Shrinkage vs published uncertainty')
plt.tight_layout(); plt.savefig('../fig/B2_shrinkage.png', bbox_inches='tight'); plt.close()

# ---------------- QMUL (guide s22-23) ----------------
qi = idx['units'].get_loc(10007775)
q_all = d[d.u_i == qi][['Year', 'theme', 'P', 'B', 'D', 's', 'R', 'n_resp', 'headcount', 'rr', 'theta_mean', 'theta_lo95', 'theta_hi95', 'P_below', 'P_inline', 'P_above', 'shrink']].sort_values(['theme', 'Year'])
q_all.to_csv('../out/final_B_QMUL_all_years.csv', index=False)
q26 = ut[ut.UKPRN == 10007775].copy()
# three-way comparison: conventional (raw positivity change + raw UK comparator), benchmark-adjusted (D, dD), Bayesian (modal state + prob)
qw = prov[prov.UKPRN == 10007775].pivot_table(index='theme', columns='Year', values=['P', 'R', 'D'])
cmp_rows = []
for t_ in range(1, 8):
    r = q26[q26.theme == t_].iloc[0]
    dP = round(qw.loc[t_, ('P', 2026)] - qw.loc[t_, ('P', 2025)], 1); R26 = round(qw.loc[t_, ('R', 2026)], 1); dR = round(qw.loc[t_, ('R', 2026)] - qw.loc[t_, ('R', 2025)], 1)
    conv_pos = band(np.array([R26]))[0]; conv_trj = traj(np.array([dR]))[0]
    cmp_rows.append(dict(theme=THEMES[t_], raw_positivity_2026=qw.loc[t_, ('P', 2026)], dP_raw=dP, raw_UK_gap_2026=R26, dR=dR, conventional_state=f'{conv_pos}/{conv_trj}',
                         D_2026=r.D_2026, dD=r.dD_obs, benchmark_state=r.obs_state, bayes_state=r.bayes_state_modal, bayes_state_P=round(r.bayes_state_maxp, 2),
                         theta26=round(r.theta26_mean, 1), ci95=f'({r.theta26_lo95:.1f}, {r.theta26_hi95:.1f})', P_below=round(r.P_below, 2), P_inline=round(r.P_inline, 2), P_above=round(r.P_above, 2),
                         P_improving=round(r.P_improving, 2), P_stable=round(r.P_stable, 2), P_declining=round(r.P_declining, 2), P_persist_below=round(r.P_persist_below, 2), category_80=r.category_80,
                         changes_conv_to_bench=conv_pos + '/' + conv_trj != r.obs_state, changes_bench_to_bayes=r.obs_state != r.bayes_state_modal))
qc = pd.DataFrame(cmp_rows); qc.to_csv('../out/final_B_QMUL_three_way_comparison.csv', index=False); res['QMUL_three_way'] = qc.to_dict('records')
# QMUL probability figure (Bayesian Figure 4)
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
y = np.arange(7); labels = [THEMES[t] for t in q26.theme]
a = ax[0]
a.barh(y, q26.P_below, color=RED, label='materially below'); a.barh(y, q26.P_inline, left=q26.P_below, color=GREY, label='broadly in line'); a.barh(y, q26.P_above, left=q26.P_below + q26.P_inline, color=BLUE, label='materially above')
for i, r in enumerate(q26.itertuples()): a.text(1.01, i, f'obs {r.D_2026:+.1f} → θ {r.theta26_mean:+.1f}', va='center', fontsize=8)
a.set_yticks(y); a.set_yticklabels(labels); a.set_xlim(0, 1.35); a.set_xlabel('Posterior probability, latent 2026 position'); a.set_title('QMUL CAH10: 2026 position'); a.legend(fontsize=8, loc='lower right'); a.invert_yaxis()
a = ax[1]
a.barh(y, q26.P_declining, color=RED, label='materially declining'); a.barh(y, q26.P_stable, left=q26.P_declining, color=GREY, label='stable'); a.barh(y, q26.P_improving, left=q26.P_declining + q26.P_stable, color=GREEN, label='materially improving')
for i, r in enumerate(q26.itertuples()): a.text(1.01, i, f'ΔD {r.dD_obs:+.1f} → Δθ {r.dtheta_mean:+.1f}', va='center', fontsize=8)
a.set_xlim(0, 1.35); a.set_xlabel('Posterior probability, latent 2025→26 trajectory'); a.set_title('QMUL CAH10: trajectory'); a.legend(fontsize=8, loc='lower right')
plt.tight_layout(); plt.savefig('../fig/B4_qmul_probabilities.png', bbox_inches='tight'); plt.close()
res['QMUL_all_years'] = q_all.round(2).to_dict('records')
json.dump(res, open('../out/final_B_summary.json', 'w'), indent=2, default=str)
print(json.dumps({k: v for k, v in res.items() if k not in ('QMUL_all_years', 'hyper', 'ppc')}, indent=2, default=str))
print(pd.DataFrame(ppc).T.round(3).to_string())
