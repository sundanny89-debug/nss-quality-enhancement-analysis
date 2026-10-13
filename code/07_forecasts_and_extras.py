"""Step 7 (new analyses):
(A) Pre-registrable 2027 predictive distributions for every sector provider x theme cell and every SEMS unit x theme
    (posterior predictive of the observed 2027 benchmark gap; falsifiable when NSS 2027 is published).
(B) Expected precision of naive threshold rules: among cells an analyst would flag from the observed 2026 gap / observed change,
    what is the posterior probability the flag is right?  (= mean posterior probability of the flagged state).
(C) Response-rate sensitivity: association of dD with change in response rate and response numbers, sector 2023-2026.
(D) Sector-level 'how much movement is real': share of provider-years whose cohort shock is credibly material."""
import pandas as pd, numpy as np, json
from scipy import stats
THR = 2.5
res = {}
rng = np.random.default_rng(2027)

def predictive_2027(S, mu, alpha_i, u_i, v_i, s_next, unit_codes, persist=None):
    """persist=None: Scenario A (model) - year & cohort effects are fresh draws (2026 shocks transient).
       persist=(gamma_2026, c_2026 per obs): Scenario B - the 2026 year and cohort effects carry into 2027 unchanged."""
    Sn = S['mu'].shape[0]
    m = S['mu'][:, None] + S['alpha'][:, alpha_i] + S['u'][:, u_i] + S['v'][:, v_i]
    if persist is None:
        gam = S['sd_gamma'][:, None] * rng.standard_normal((Sn, 1))
        cu = S['sd_c'][:, None] * rng.standard_normal((Sn, unit_codes.max() + 1)); c = cu[:, unit_codes]
    else:
        gam, c = persist
    eps = S['sd_eps'][:, None] * rng.standard_normal(m.shape)
    latent = m + gam + c + eps
    obs = latent + s_next[None, :] * rng.standard_normal(m.shape)
    return latent, obs

def scenarioB(S, d26, gamma_idx_last, c_index):
    gam = S['gamma'][:, [gamma_idx_last]]; c = S['c'][:, c_index]
    return gam, c

def summarise_pred(latent, obs):
    return dict(pred_mean=obs.mean(0), pred_lo90=np.quantile(obs, .05, 0), pred_hi90=np.quantile(obs, .95, 0), pred_lo50=np.quantile(obs, .25, 0), pred_hi50=np.quantile(obs, .75, 0),
                P27_above=(obs > THR).mean(0), P27_below=(obs < -THR).mean(0), latent27_mean=latent.mean(0), P27_latent_below=(latent < -THR).mean(0), P27_latent_above=(latent > THR).mean(0))

# ---------------- (A) sector 2027 forecasts ----------------
S = dict(np.load('../out/full_M1_cohort_samples.npz'))
d = pd.read_csv('../out/full_M1_obs_index.csv')
d26 = d[d.Year == 2026].copy()
codes = d26['u_i'].astype('category').cat.codes.values
lat, obs = predictive_2027(S, None, d26.q_i.values, d26.u_i.values, d26.v_i.values, d26.s.values.astype(float), codes)
for k, v in summarise_pred(lat, obs).items(): d26[k] = v
latB, obsB = predictive_2027(S, None, d26.q_i.values, d26.u_i.values, d26.v_i.values, d26.s.values.astype(float), codes, persist=scenarioB(S, d26, 3, d26.c_i.values))
d26['predB_mean'] = obsB.mean(0); d26['predB_lo90'] = np.quantile(obsB, .05, 0); d26['predB_hi90'] = np.quantile(obsB, .95, 0); d26['PB27_below'] = (obsB < -THR).mean(0); d26['PB27_above'] = (obsB > THR).mean(0)
d26['n_years_observed'] = d26.groupby(['u_i', 'q_i'])['Year'].transform('count')
cols = ['UKPRN', 'provider', 'theme', 'D', 's', 'n_resp', 'pred_mean', 'pred_lo50', 'pred_hi50', 'pred_lo90', 'pred_hi90', 'P27_above', 'P27_below', 'latent27_mean', 'P27_latent_below', 'P27_latent_above', 'predB_mean', 'predB_lo90', 'predB_hi90', 'PB27_below', 'PB27_above']
f27 = d26[cols].rename(columns={'D': 'D_2026', 's': 's_2026', 'n_resp': 'n_2026'})
f27.to_csv('../out/forecast_2027_sector_provider_theme.csv', index=False)
res['sector_2027'] = dict(n_cells=int(len(f27)), mean_width90=float((f27.pred_hi90 - f27.pred_lo90).mean()),
                          share_P27_below_gt_0_8=float((f27.P27_below > 0.8).mean()), share_P27_above_gt_0_8=float((f27.P27_above > 0.8).mean()),
                          note='Point forecasts are shrunk toward provider/theme means; predictive intervals include cohort shock, idiosyncratic and sampling variance using 2026 s as proxy for 2027 s.')
q = f27[f27.provider.str.contains('Queen Mary')]
res['QMUL_2027'] = q.round(2).to_dict('records')

# ---------------- (A) SEMS 2027 forecasts ----------------
Ss = dict(np.load('../out/sems_theme_M1_samples.npz'))
ds = pd.read_csv('../out/sems_theme_M1_obs_index.csv')
d26s = ds[ds.Year == 2026].copy()
codes = d26s['u_i'].astype('category').cat.codes.values
lat, obs = predictive_2027(Ss, None, d26s.q_i.values, d26s.u_i.values, d26s.v_i.values, d26s.s.values.astype(float), codes)
for k, v in summarise_pred(lat, obs).items(): d26s[k] = v
latB, obsB = predictive_2027(Ss, None, d26s.q_i.values, d26s.u_i.values, d26s.v_i.values, d26s.s.values.astype(float), codes, persist=scenarioB(Ss, d26s, 3, d26s.c_i.values))
d26s['predB_mean'] = obsB.mean(0); d26s['predB_lo90'] = np.quantile(obsB, .05, 0); d26s['predB_hi90'] = np.quantile(obsB, .95, 0); d26s['PB27_below'] = (obsB < -THR).mean(0); d26s['PB27_above'] = (obsB > THR).mean(0)
f27s = d26s[['unit', 'theme', 'D', 's', 'n_resp', 'pred_mean', 'pred_lo50', 'pred_hi50', 'pred_lo90', 'pred_hi90', 'P27_above', 'P27_below', 'latent27_mean', 'P27_latent_below', 'P27_latent_above', 'predB_mean', 'predB_lo90', 'predB_hi90', 'PB27_below', 'PB27_above']].rename(columns={'D': 'D_2026', 's': 's_2026', 'n_resp': 'n_2026'})
f27s.to_csv('../out/forecast_2027_sems_unit_theme.csv', index=False)
res['SEMS_2027'] = f27s.round(2).to_dict('records')

# ---------------- (B) expected precision of naive rules ----------------
ut = pd.read_csv('../out/full_M1_unit_theme_2026.csv')
c = ut[ut.obs_state.notna()]
def prec(mask, pcol): return dict(n_flagged=int(mask.sum()), expected_precision=float(c.loc[mask, pcol].mean()), share_P_gt_0_8=float((c.loc[mask, pcol] > 0.8).mean()), share_P_lt_0_5=float((c.loc[mask, pcol] < 0.5).mean()))
res['naive_rule_expected_precision_sector'] = {
    'flag materially below (obs D < -2.5)': prec(c.obs_pos == 'below', 'P_below'),
    'flag materially above (obs D > 2.5)': prec(c.obs_pos == 'above', 'P_above'),
    'flag materially declining (obs dD < -2.5)': prec(c.obs_trj == 'declining', 'P_declining'),
    'flag materially improving (obs dD > 2.5)': prec(c.obs_trj == 'improving', 'P_improving'),
    'flag below AND declining (highest priority quadrant)': prec((c.obs_pos == 'below') & (c.obs_trj == 'declining'), 'P_obs_state'),
    'flag above AND improving (sustain)': prec((c.obs_pos == 'above') & (c.obs_trj == 'improving'), 'P_obs_state')}
c2 = c.copy(); c2['band'] = pd.cut(c2['n_2026'], [0, 50, 100, 200, np.inf], labels=['<50', '50-99', '100-199', '200+'])
res['naive_rule_expected_precision_by_size'] = {str(b): dict(below=float(g.loc[g.obs_pos == 'below', 'P_below'].mean()), declining=float(g.loc[g.obs_trj == 'declining', 'P_declining'].mean()),
                                                              improving=float(g.loc[g.obs_trj == 'improving', 'P_improving'].mean())) for b, g in c2.groupby('band', observed=True)}
su = pd.read_csv('../out/sems_theme_M1_unit_theme_2026.csv'); cs = su[su.obs_state.notna()]
c = cs
res['naive_rule_expected_precision_sems'] = {
    'flag materially below (obs D < -2.5)': prec(c.obs_pos == 'below', 'P_below'),
    'flag materially above (obs D > 2.5)': prec(c.obs_pos == 'above', 'P_above'),
    'flag materially declining (obs dD < -2.5)': prec(c.obs_trj == 'declining', 'P_declining'),
    'flag materially improving (obs dD > 2.5)': prec(c.obs_trj == 'improving', 'P_improving')}

# ---------------- (C) response-rate sensitivity ----------------
prov = pd.read_csv('../data/sector_cah10_provider_theme.csv').sort_values(['UKPRN', 'theme', 'Year'])
for col in ['D', 'rr', 'n_resp', 'Year']:
    prov[col + '_prev'] = prov.groupby(['UKPRN', 'theme'])[col].shift(1)
adj = prov[prov.Year - prov.Year_prev == 1].copy()
adj['dD'] = adj.D - adj.D_prev; adj['drr'] = adj.rr - adj.rr_prev; adj['dlogn'] = np.log(adj.n_resp) - np.log(adj.n_resp_prev)
r1 = stats.pearsonr(adj.dD, adj.drr); r2 = stats.pearsonr(adj.dD, adj.dlogn)
# within-theme partial: regress dD on drr, dlogn with theme x year fixed effects (demeaned)
adj['key'] = adj.theme.astype(str) + '_' + adj.Year.astype(str)
dm = adj[['dD', 'drr', 'dlogn']] - adj.groupby('key')[['dD', 'drr', 'dlogn']].transform('mean')
X = np.column_stack([np.ones(len(dm)), dm.drr, dm.dlogn]); beta, *_ = np.linalg.lstsq(X, dm.dD.values, rcond=None)
resid = dm.dD.values - X @ beta; se = np.sqrt(np.diag(np.linalg.inv(X.T @ X)) * resid.var(ddof=3))
res['response_rate_sensitivity'] = dict(n_transitions=int(len(adj)), corr_dD_drr=float(r1[0]), p_drr=float(r1[1]), corr_dD_dlogn=float(r2[0]), p_dlogn=float(r2[1]),
                                        beta_drr_pp_per_pp=float(beta[1]), se_drr=float(se[1]), beta_dlogn=float(beta[2]), se_dlogn=float(se[2]),
                                        share_transitions_abs_drr_gt_10=float((adj.drr.abs() > 10).mean()),
                                        interpretation='Sign and size of association between change in benchmark gap and change in response rate / responses, net of theme-year effects.')
# same for SEMS units
th = pd.read_csv('../data/sems_theme_panel.csv').sort_values(['unit', 'theme', 'Year'])
for col in ['D', 'rr', 'Year']: th[col + '_prev'] = th.groupby(['unit', 'theme'])[col].shift(1)
adjs = th[th.Year - th.Year_prev == 1].copy(); adjs['dD'] = adjs.D - adjs.D_prev; adjs['drr'] = adjs.rr - adjs.rr_prev
res['response_rate_sensitivity']['sems_corr_dD_drr'] = float(stats.pearsonr(adjs.dD, adjs.drr)[0]); res['response_rate_sensitivity']['sems_n'] = int(len(adjs))
res['response_rate_sensitivity']['sems_transitions_abs_drr_gt_10'] = adjs[adjs.drr.abs() > 10][['unit', 'Year', 'theme', 'drr', 'dD']].groupby(['unit', 'Year']).agg(drr=('drr', 'first'), mean_dD=('dD', 'mean')).reset_index().round(1).to_dict('records')

# ---------------- (D) sector-level: how much movement is real ----------------
ct = pd.DataFrame({'c_mean': S['c'].mean(0), 'P_abs_gt_2_5': (np.abs(S['c']) > THR).mean(0)})
res['cohort_shocks_sector'] = dict(n_provider_years=int(len(ct)), share_abs_mean_gt_2_5=float((ct.c_mean.abs() > THR).mean()), share_credibly_material=float((ct.P_abs_gt_2_5 > 0.8).mean()),
                                   sd_of_posterior_mean_c=float(ct.c_mean.std()))
json.dump(res, open('../out/extras_summary.json', 'w'), indent=2, default=str)
print(json.dumps({k: v for k, v in res.items() if k not in ('SEMS_2027', 'QMUL_2027')}, indent=2, default=str))
print(pd.DataFrame(res['SEMS_2027'])[['unit', 'theme', 'D_2026', 'pred_mean', 'pred_lo90', 'pred_hi90', 'P27_below', 'P27_above', 'predB_mean', 'predB_lo90', 'predB_hi90']].to_string())
print(pd.DataFrame(res['QMUL_2027'])[['theme', 'D_2026', 'pred_mean', 'pred_lo90', 'pred_hi90', 'P27_below', 'P27_above', 'predB_mean', 'PB27_below']].to_string())
