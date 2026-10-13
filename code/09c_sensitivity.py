"""Step 9c: sensitivity analyses (guide s6, s24) on the final full-data guide-spec model, 4 chains x (800+800), target 0.95.
 S1 prior width HalfNormal(2.5) on all scale parameters; S2 HalfNormal(10);
 S3 exclude cells with headcount < 20; S4 exclude headcount < 50;
 S5 exclude providers with contribution to own benchmark > 10% in any year;
 S6 add standardised publication response rate as an associative covariate (beta ~ N(0,2)).
For each: hyperparameters, and agreement of 2026 modal position / modal 3x3 state / P(below) with the reference fit (final Model B)."""
import pandas as pd, numpy as np, json, time, sys
from bayes_model import *
THR = 2.5; NW, NS = 800, 800
prov = pd.read_csv('../data/sector_cah10_provider_theme.csv')
ref = pd.read_csv('../out/final_B_unit_theme_2026.csv')
res = {}
def band(x): return np.where(x > THR, 'above', np.where(x < -THR, 'below', 'inline'))

def derive(S, idx, d):
    theta = np.asarray(S['m'])
    d = d.copy(); d['P_above'] = (theta > THR).mean(0); d['P_below'] = (theta < -THR).mean(0); d['P_inline'] = 1 - d.P_above - d.P_below
    d['theta_mean'] = theta.mean(0); d['bayes_pos'] = np.array(['above', 'inline', 'below'])[np.argmax(d[['P_above', 'P_inline', 'P_below']].values, 1)]
    loc26 = {(r.u_i, r.q_i): i for i, r in d[d.Year == 2026].iterrows()}; loc25 = {(r.u_i, r.q_i): i for i, r in d[d.Year == 2025].iterrows()}
    rows = []
    for key, i in loc26.items():
        r = d.loc[i]; row = dict(UKPRN=r.UKPRN, theme=int(r.theme), theta26=r.theta_mean, P_below=r.P_below, P_above=r.P_above, bayes_pos=r.bayes_pos)
        if key in loc25:
            dth = theta[:, i] - theta[:, loc25[key]]; th26 = theta[:, i]
            pl = np.where(th26 > THR, 'above', np.where(th26 < -THR, 'below', 'inline')); tl = np.where(dth > THR, 'improving', np.where(dth < -THR, 'declining', 'stable'))
            joint = pd.Series(pl + '/' + tl).value_counts(normalize=True); row['bayes_state'] = joint.index[0]; row['P_improving'] = float((dth > THR).mean()); row['P_declining'] = float((dth < -THR).mean())
        rows.append(row)
    return pd.DataFrame(rows)

def compare(ut, tag):
    m = ref.merge(ut, on=['UKPRN', 'theme'], suffixes=('_ref', '_s'))
    mm = m[m.bayes_state_modal.notna() & m.bayes_state.notna()]
    return dict(n=int(len(m)), pos_agreement=float((m.bayes_pos_ref == m.bayes_pos_s).mean()), state_agreement=float((mm.bayes_state_modal == mm.bayes_state).mean()),
                corr_P_below=float(np.corrcoef(m.P_below_ref, m.P_below_s)[0, 1]), mean_abs_diff_theta26=float((m.theta26_mean - m.theta26).abs().mean()),
                share_obs_below_P_gt_0_8_ref=float((m.loc[m.obs_pos == 'below', 'P_below_ref'] > 0.8).mean()), share_obs_below_P_gt_0_8_s=float((m.loc[m.obs_pos == 'below', 'P_below_s'] > 0.8).mean()),
                churn_obs_vs_bayes_state_s=float((mm.obs_state != mm.bayes_state).mean()), churn_obs_vs_bayes_state_ref=float((mm.obs_state != mm.bayes_state_modal).mean()))

import os
def run(tag, data, priors, x=None):
    if os.path.exists(f'../out/sens_{tag}_summary.json'):
        return json.load(open(f'../out/sens_{tag}_summary.json'))
    idx = build_index(data, 'UKPRN'); d = idx['df']
    xx = None
    if x is not None:
        xx = d[x].values; xx = (xx - xx.mean()) / xx.std()
    t0 = time.time(); mcmc, kw = fit(idx, cohort=True, dynamic=True, eps=False, alpha_prior='fixed', priors=priors, num_warmup=NW, num_samples=NS, num_chains=4, target_accept=0.95, seed=300, x=xx)
    summ, diag, _ = diagnostics(mcmc, params=('mu', 'sd_gamma', 'sd_u', 'sd_v', 'sd_c', 'sd_eta', 'beta')); S = flatten_samples(mcmc); del mcmc
    print(tag, f'{time.time()-t0:.0f}s', diag, file=sys.stderr)
    ut = derive(S, idx, d); ut.to_csv(f'../out/sens_{tag}_unit_theme_2026.csv', index=False)
    out = dict(diag=diag, hyper=summ[['mean', 'eti95_lb', 'eti95_ub']].round(3).to_dict(), n_obs=int(len(d)), n_units=int(idx['n_u']))
    out['agreement_with_reference'] = compare(ut, tag)
    json.dump(out, open(f'../out/sens_{tag}_summary.json', 'w'), indent=2, default=str)
    return out

base = dict(PRIORS); base.update(sd_eta=3.0, alpha_fixed_sd=3.0)
narrow = {k: (2.5 if k in ('sd_u', 'sd_v', 'sd_c') else (1.5 if k == 'sd_eta' else v)) for k, v in base.items()}; narrow['sd_gamma'] = 1.0
wide = {k: (10.0 if k in ('sd_u', 'sd_v', 'sd_c') else (6.0 if k == 'sd_eta' else v)) for k, v in base.items()}; wide['sd_gamma'] = 4.0
res['S1_priors_halfnormal_2.5'] = run('S1_prior_narrow', prov, narrow)
res['S2_priors_halfnormal_10'] = run('S2_prior_wide', prov, wide)
res['S3_exclude_headcount_lt20'] = run('S3_hc20', prov[prov.headcount >= 20], base)
res['S4_exclude_headcount_lt50'] = run('S4_hc50', prov[prov.headcount >= 50], base)
hi = prov.groupby('UKPRN')['contrib'].max(); hi_prov = hi[hi > 10].index
res['S5_exclude_high_contribution'] = dict(excluded_providers=int(len(hi_prov)), **run('S5_contrib', prov[~prov.UKPRN.isin(hi_prov)], base))
res['S6_response_rate_covariate'] = run('S6_rr', prov, base, x='rr')
json.dump(res, open('../out/sensitivity_summary.json', 'w'), indent=2, default=str)
print(json.dumps(res, indent=2, default=str))
