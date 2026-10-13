"""Step 5: QMUL/SEMS case study using public CAH3 subject proxies (All undergraduates).
Units: 8 SEMS-related CAH3 subjects (Materials technology dropped: identical to Materials science in 2025-26).
Fits: (a) theme-level M1 with weak priors [primary]; (b) M1 with sector-informed priors [sensitivity];
      (c) question-level M1 (25 questions) [robustness]; (d) held-out 2026 prediction for SEMS units.
Outputs: posterior 3x3 classification, churn vs observed, persistence, estimand comparison (RQ4)."""
import pandas as pd, numpy as np, json, time, sys
from bayes_model import *

THR = 2.5
NW, NS, NC = 1500, 1500, 4
au = pd.read_csv('../data/sems_all_ug_long.csv')
sec_hyper = pd.read_csv('../out/full_M1_cohort_hyper.csv', index_col=0)   # sector posterior for informed priors
res = {}
def band(x): return np.where(x > THR, 'above', np.where(x < -THR, 'below', 'inline'))
def traj(x): return np.where(x > THR, 'improving', np.where(x < -THR, 'declining', 'stable'))

units_keep = ['Aerospace', 'Biomedical', 'Chemical', 'Design studies (DICE proxy)', 'Eng non-specific', 'Materials science', 'Mechanical', 'Prod & manuf (Robotics proxy)']
th = au[au.is_theme & au.unit.isin(units_keep)].copy(); th['theme'] = th['theme'].astype(int)
th = th[['Year', 'unit', 'Subject code', 'theme', 'P', 'B', 'D', 's', 'n_resp', 'headcount', 'rr', 'contrib', 'p_above', 'p_inline', 'p_below']].reset_index(drop=True)
th.to_csv('../data/sems_theme_panel.csv', index=False)
res['n_obs_theme'] = int(len(th)); res['units'] = units_keep

def summarise(S, idx, d, tag):
    theta = latent_draws(S, None, d)
    d = d.copy(); d['theta_mean'] = theta.mean(0); d['theta_sd'] = theta.std(0)
    d['theta_lo90'] = np.quantile(theta, .05, 0); d['theta_hi90'] = np.quantile(theta, .95, 0)
    d['P_above'] = (theta > THR).mean(0); d['P_below'] = (theta < -THR).mean(0); d['P_inline'] = 1 - d.P_above - d.P_below
    d['obs_pos'] = band(d.D); d['bayes_pos'] = np.array(['above', 'inline', 'below'])[np.argmax(d[['P_above', 'P_inline', 'P_below']].values, 1)]
    d.to_csv(f'../out/sems_{tag}_cell_posteriors.csv', index=False)
    rows = []
    for i, r in d[d.Year == 2026].iterrows():
        row = dict(unit=r.unit, theme=r.theme, D_2026=r.D, n_2026=r.n_resp, theta26_mean=r.theta_mean, theta26_lo90=r.theta_lo90, theta26_hi90=r.theta_hi90,
                   P_above=r.P_above, P_inline=r.P_inline, P_below=r.P_below, obs_pos=r.obs_pos, bayes_pos=r.bayes_pos, ofs_p_below=r.p_below / 100, ofs_p_above=r.p_above / 100)
        prev = d[(d.unit == r.unit) & (d.theme == r.theme) & (d.Year == 2025)]
        if len(prev):
            j = prev.index[0]; dth = theta[:, i] - theta[:, j]
            row.update(D_2025=d.loc[j, 'D'], dD_obs=r.D - d.loc[j, 'D'], dtheta_mean=dth.mean(), dtheta_lo90=np.quantile(dth, .05), dtheta_hi90=np.quantile(dth, .95),
                       P_improving=(dth > THR).mean(), P_declining=(dth < -THR).mean())
            row['P_stable'] = 1 - row['P_improving'] - row['P_declining']
            row['obs_trj'] = traj(np.array([row['dD_obs']]))[0]
            row['bayes_trj'] = ['improving', 'stable', 'declining'][int(np.argmax([row['P_improving'], row['P_stable'], row['P_declining']]))]
            pos_lab = np.where(theta[:, i] > THR, 'above', np.where(theta[:, i] < -THR, 'below', 'inline'))
            trj_lab = np.where(dth > THR, 'improving', np.where(dth < -THR, 'declining', 'stable'))
            joint = pd.Series(pos_lab + '/' + trj_lab).value_counts(normalize=True)
            row['bayes_state_modal'] = joint.index[0]; row['bayes_state_maxp'] = float(joint.iloc[0])
            row['obs_state'] = row['obs_pos'] + '/' + row['obs_trj']; row['P_obs_state'] = float(joint.get(row['obs_state'], 0.0))
        uv = S['mu'] + S['alpha'][:, r.q_i] + S['u'][:, r.u_i] + S['v'][:, int(r.v_i)]
        row.update(persist_mean=uv.mean(), persist_lo90=np.quantile(uv, .05), persist_hi90=np.quantile(uv, .95), P_persist_below=(uv < -THR).mean(), P_persist_above=(uv > THR).mean())
        cell = d.index[(d.unit == r.unit) & (d.theme == r.theme)].values
        row['n_years'] = len(cell); row['P_below_every_year'] = float((theta[:, cell] < -THR).all(1).mean()); row['P_above_every_year'] = float((theta[:, cell] > THR).all(1).mean())
        rows.append(row)
    ut = pd.DataFrame(rows); ut.to_csv(f'../out/sems_{tag}_unit_theme_2026.csv', index=False)
    c = ut[ut.obs_state.notna()]
    churn = dict(n=int(len(c)), full=float((c.obs_state != c.bayes_state_modal).mean()), position=float((c.obs_pos != c.bayes_pos).mean()),
                 trajectory=float((c.obs_trj != c.bayes_trj).mean()), mean_P_obs_state=float(c.P_obs_state.mean()),
                 obs_below=int((c.obs_pos == 'below').sum()), obs_below_P_gt_0_8=int(((c.obs_pos == 'below') & (c.P_below > 0.8)).sum()),
                 obs_above=int((c.obs_pos == 'above').sum()), obs_above_P_gt_0_8=int(((c.obs_pos == 'above') & (c.P_above > 0.8)).sum()),
                 obs_improving=int((c.obs_trj == 'improving').sum()), obs_improving_P_gt_0_8=int(((c.obs_trj == 'improving') & (c.P_improving > 0.8)).sum()),
                 obs_declining=int((c.obs_trj == 'declining').sum()), obs_declining_P_gt_0_8=int(((c.obs_trj == 'declining') & (c.P_declining > 0.8)).sum()))
    return d, ut, theta, churn

# ---------- (a) primary theme-level fit ----------
idx = build_index(th, 'unit'); d = idx['df']
t0 = time.time(); mcmc, kw = fit(idx, cohort=True, num_warmup=NW, num_samples=NS, num_chains=NC, target_accept=0.95, seed=21)
summ, diag, _ = diagnostics(mcmc); S = flatten_samples(mcmc); print('SEMS theme', f'{time.time()-t0:.0f}s', diag, file=sys.stderr)
summ.to_csv('../out/sems_theme_M1_hyper.csv'); res['theme_M1'] = dict(diag=diag, hyper=summ[['mean', 'sd', 'eti90_lb', 'eti90_ub']].round(3).to_dict())
np.savez_compressed('../out/sems_theme_M1_samples.npz', **{k: np.asarray(v).astype(np.float32) for k, v in S.items()})
idx['df'].to_csv('../out/sems_theme_M1_obs_index.csv', index=False); pd.Series(idx['units']).to_csv('../out/sems_theme_M1_units.csv', index=False)
d1, ut1, theta1, churn1 = summarise(S, idx, d, 'theme_M1'); res['theme_M1']['churn_obs_vs_bayes'] = churn1
# unit-level summaries: persistent unit effect u, cohort shocks
uu = pd.DataFrame({'unit': idx['units'], 'u_mean': S['u'].mean(0), 'u_lo90': np.quantile(S['u'], .05, 0), 'u_hi90': np.quantile(S['u'], .95, 0)})
cmat = pd.DataFrame(S['c'].mean(0), index=idx['ut'].index)
ct = idx['ut'].copy(); ct['c_mean'] = S['c'].mean(0); ct['c_sd'] = S['c'].std(0); ct['P_abs_gt_2_5'] = (np.abs(S['c']) > 2.5).mean(0)
ct['unit'] = idx['units'][ct.u_i]; ct['Year'] = idx['years'][ct.t_i]; ct = ct[['unit', 'Year', 'c_mean', 'c_sd', 'P_abs_gt_2_5']]
uu.to_csv('../out/sems_theme_M1_unit_effects.csv', index=False); ct.to_csv('../out/sems_theme_M1_cohort_effects.csv', index=False)
res['theme_M1']['unit_effects'] = uu.round(2).to_dict('records'); res['theme_M1']['cohort_effects'] = ct.round(2).to_dict('records')

# ---------- (b) sector-informed priors sensitivity ----------
inf = {k: (float(sec_hyper.loc[k, 'mean']), 0.35) for k in ['sd_u', 'sd_v', 'sd_c'] if k in sec_hyper.index}
inf['sd_eps'] = (max(float(sec_hyper.loc['sd_eps', 'mean']), 0.3), 0.7)
t0 = time.time(); mcmc, kw = fit(idx, cohort=True, num_warmup=NW, num_samples=NS, num_chains=NC, target_accept=0.95, seed=22, informative=inf)
summ, diag, _ = diagnostics(mcmc); S2 = flatten_samples(mcmc); print('SEMS theme informed', f'{time.time()-t0:.0f}s', diag, file=sys.stderr)
summ.to_csv('../out/sems_theme_M1inf_hyper.csv'); res['theme_M1_sector_informed'] = dict(priors=inf, diag=diag, hyper=summ[['mean', 'sd', 'eti90_lb', 'eti90_ub']].round(3).to_dict())
d2, ut2, theta2, churn2 = summarise(S2, idx, d, 'theme_M1inf'); res['theme_M1_sector_informed']['churn_obs_vs_bayes'] = churn2
m = ut1.merge(ut2, on=['unit', 'theme'], suffixes=('_weak', '_inf'))
res['prior_sensitivity'] = dict(state_agreement=float((m.bayes_state_modal_weak == m.bayes_state_modal_inf).mean()),
                                pos_agreement=float((m.bayes_pos_weak == m.bayes_pos_inf).mean()),
                                mean_abs_diff_theta26=float((m.theta26_mean_weak - m.theta26_mean_inf).abs().mean()),
                                corr_P_below=float(np.corrcoef(m.P_below_weak, m.P_below_inf)[0, 1]))

# ---------- (c) question-level robustness ----------
qq = au[(~au.is_theme) & au.unit.isin(units_keep) & au.q_theme.notna()].copy()
qq['theme'] = qq['qnum'].astype(int)  # question number acts as the "theme" index in the model
qq['q_theme'] = qq['q_theme'].astype(int)
qq = qq[['Year', 'unit', 'theme', 'q_theme', 'P', 'B', 'D', 's', 'n_resp']].dropna(subset=['D', 's']).reset_index(drop=True)
res['n_obs_question'] = int(len(qq))
idxq = build_index(qq, 'unit'); dq = idxq['df']
t0 = time.time(); mcmc, kw = fit(idxq, cohort=True, num_warmup=NW, num_samples=NS, num_chains=NC, target_accept=0.95, seed=23)
summ, diag, _ = diagnostics(mcmc); Sq = flatten_samples(mcmc); print('SEMS question', f'{time.time()-t0:.0f}s', diag, file=sys.stderr)
summ.to_csv('../out/sems_question_M1_hyper.csv'); res['question_M1'] = dict(diag=diag, hyper=summ[['mean', 'sd', 'eti90_lb', 'eti90_ub']].round(3).to_dict())
thq = latent_draws(Sq, None, dq)
dq['theta_mean'] = thq.mean(0); dq['P_below'] = (thq < -THR).mean(0); dq['P_above'] = (thq > THR).mean(0)
dq['obs_pos'] = band(dq.D)
dq.to_csv('../out/sems_question_M1_cell_posteriors.csv', index=False)
# aggregate question posteriors to theme (mean over questions of latent theta) for 2026, compare with theme-level model
agg = []
for (u_, t_, y_), g in dq.groupby(['unit', 'q_theme', 'Year']):
    tm = thq[:, g.index.values].mean(1)
    agg.append(dict(unit=u_, theme=t_, Year=y_, thetaQ_mean=tm.mean(), PQ_below=(tm < -THR).mean(), PQ_above=(tm > THR).mean(), n_q=len(g),
                    n_q_obs_below=int((g.obs_pos == 'below').sum()), n_q_P_below_gt_0_8=int((g.P_below > 0.8).sum())))
agg = pd.DataFrame(agg); agg['PQ_inline'] = 1 - agg.PQ_below - agg.PQ_above
agg['bayesQ_pos'] = np.array(['above', 'inline', 'below'])[np.argmax(agg[['PQ_above', 'PQ_inline', 'PQ_below']].values, 1)]
agg.to_csv('../out/sems_question_aggregated_to_theme.csv', index=False)
cmp = ut1.merge(agg[agg.Year == 2026], on=['unit', 'theme'])
res['question_vs_theme_2026'] = dict(pos_agreement=float((cmp.bayes_pos == cmp.bayesQ_pos).mean()), corr_theta=float(np.corrcoef(cmp.theta26_mean, cmp.thetaQ_mean)[0, 1]),
                                     corr_P_below=float(np.corrcoef(cmp.P_below, cmp.PQ_below)[0, 1]), n=int(len(cmp)))
# question-level: which questions are credibly below in 2026 (P>0.8) per unit
qb = dq[(dq.Year == 2026) & (dq.P_below > 0.8)][['unit', 'theme', 'q_theme', 'D', 'theta_mean', 'P_below']].sort_values(['unit', 'theme'])
qb.to_csv('../out/sems_question_credibly_below_2026.csv', index=False)
res['question_credibly_below_2026'] = qb.round(2).to_dict('records')

# ---------- (d) estimand comparison (RQ4) 2026 and change 2025->26 ----------
d26 = d1[d1.Year == 2026]; d25 = d1[d1.Year == 2025]
i26 = {(r.unit, r.theme): i for i, r in d26.iterrows()}; i25 = {(r.unit, r.theme): i for i, r in d25.iterrows()}
est = []
for t_ in range(1, 8):
    g = d26[d26.theme == t_]; g5 = d25[d25.theme == t_]
    w_ = g.n_resp / g.n_resp.sum()
    ii = [i26[(u_, t_)] for u_ in g.unit]; jj = [i25[(u_, t_)] for u_ in g5.unit]
    lat_eq = theta1[:, ii].mean(1); lat_w = (theta1[:, ii] * w_.values[None, :]).sum(1)
    # change: units present in both years
    common = [u_ for u_ in g.unit if (u_, t_) in i25]
    ic = [i26[(u_, t_)] for u_ in common]; jc = [i25[(u_, t_)] for u_ in common]
    wc = g.set_index('unit').loc[common, 'n_resp']; wc = wc / wc.sum()
    dobs = g.set_index('unit').loc[common, 'D'].values - g5.set_index('unit').loc[common, 'D'].values
    dlat = theta1[:, ic] - theta1[:, jc]
    est.append(dict(theme=t_, typical_student_D=(g.D * w_).sum(), typical_unit_D=g.D.mean(), latent_equal_mean=lat_eq.mean(), latent_equal_lo90=np.quantile(lat_eq, .05), latent_equal_hi90=np.quantile(lat_eq, .95),
                    latent_student_weighted_mean=lat_w.mean(), P_latent_equal_below=(lat_eq < -THR).mean(), P_latent_equal_above=(lat_eq > THR).mean(),
                    typical_student_dD=(dobs * wc.values).sum(), typical_unit_dD=dobs.mean(), latent_equal_dD=dlat.mean(1).mean(), latent_equal_dD_lo90=np.quantile(dlat.mean(1), .05), latent_equal_dD_hi90=np.quantile(dlat.mean(1), .95),
                    P_latent_dD_gt_2_5=(dlat.mean(1) > THR).mean(), n_units=len(g), n_units_change=len(common)))
est = pd.DataFrame(est)
qm = au[(au.unit == 'CAH10 (all E&T)') & au.is_theme].copy(); qm['theme'] = qm.theme.astype(int)
est['QMUL_CAH10_provider_D_2026'] = est.theme.map(qm[qm.Year == 2026].set_index('theme')['D'])
est['QMUL_CAH10_provider_dD'] = est.theme.map(qm[qm.Year == 2026].set_index('theme')['D'] - qm[qm.Year == 2025].set_index('theme')['D'])
for col in ['typical_student_D', 'typical_unit_D', 'latent_equal_mean', 'QMUL_CAH10_provider_D_2026']:
    est['rank_' + col] = est[col].rank(ascending=False).astype(int)
est.to_csv('../out/sems_estimand_comparison_2026.csv', index=False)
res['estimand_comparison'] = est.round(2).to_dict('records')

# ---------- (e) held-out 2026 for SEMS units (theme level, weak priors) ----------
tr = th[th.Year <= 2025]; te = th[th.Year == 2026]
idxt = build_index(tr, 'unit'); dt = idxt['df']
mcmc, kw = fit(idxt, cohort=True, num_warmup=NW, num_samples=NS, num_chains=NC, target_accept=0.95, seed=24)
St = flatten_samples(mcmc)
te2 = te.copy(); te2['u_i'] = idxt['units'].get_indexer(te2.unit); te2['q_i'] = idxt['themes'].get_indexer(te2.theme)
te2 = te2.merge(idxt['uq'][['u_i', 'q_i', 'v_i']], on=['u_i', 'q_i'], how='left')
mp = St['mu'][:, None] + St['alpha'][:, te2.q_i.values] + St['u'][:, te2.u_i.values] + St['v'][:, te2.v_i.values]
te2['pred_M1'] = mp.mean(0)
w = th.pivot_table(index=['unit', 'theme'], columns='Year', values=['D', 's'])
te2 = te2.set_index(['unit', 'theme'])
te2['last'] = w[('D', 2025)]; te2['mean3'] = w[[('D', y) for y in [2023, 2024, 2025]]].mean(1)
vals = w[[('D', y) for y in [2023, 2024, 2025]]].values; wts = 1 / w[[('s', y) for y in [2023, 2024, 2025]]].values ** 2
te2['pw'] = pd.Series(np.nansum(np.where(np.isnan(vals), 0, vals * wts), 1) / np.nansum(np.where(np.isnan(vals), 0, wts), 1), index=w.index)
te2 = te2.reset_index(); te2.to_csv('../out/sems_holdout_2026_pred.csv', index=False)
ho = {}
for col in ['last', 'mean3', 'pw', 'pred_M1']:
    e = te2[col] - te2.D; ho[col] = dict(RMSE=float(np.sqrt((e ** 2).mean())), MAE=float(e.abs().mean()), n=int(e.notna().sum()))
res['sems_holdout_2026'] = ho
json.dump(res, open('../out/sems_summary.json', 'w'), indent=2, default=str)
print(json.dumps({k: v for k, v in res.items() if k not in ('estimand_comparison',)}, indent=2, default=str))
