"""Step 3: Held-out predictive validation (RQ2). Train 2023-2025, predict 2026 benchmark gaps.
Models: M0 (no cohort), M1 (cohort; primary), M2 (cohort + unit trend). Baselines: last-year, 3-yr mean, precision-weighted mean.
Also rolling-origin robustness: train 2023-2024, predict 2025 (M1 vs baselines)."""
import pandas as pd, numpy as np, json, time, sys
from bayes_model import *
from scipy import stats

THR = 2.5
prov = pd.read_csv('../data/sector_cah10_provider_theme.csv')
NW, NS, NC = 1000, 1000, 4

def band(x):
    return np.where(x > THR, 'above', np.where(x < -THR, 'below', 'inline'))

def crps_from_draws(draws, y):
    """CRPS via empirical formula: E|X-y| - 0.5 E|X-X'|. draws (S,N), y (N,)"""
    S = draws.shape[0]
    t1 = np.abs(draws - y[None, :]).mean(0)
    x = np.sort(draws, axis=0)
    # E|X-X'| = 2/S^2 * sum_i (2i - S - 1) x_(i)
    i = np.arange(1, S + 1)[:, None]
    t2 = (2.0 / (S * S)) * ((2 * i - S - 1) * x).sum(0)
    return t1 - 0.5 * t2

def crps_normal(m, sd, y):
    z = (y - m) / sd
    return sd * (z * (2 * stats.norm.cdf(z) - 1) + 2 * stats.norm.pdf(z) - 1 / np.sqrt(np.pi))

def band_probs_from_draws(draws):
    return np.stack([(draws > THR).mean(0), ((draws >= -THR) & (draws <= THR)).mean(0), (draws < -THR).mean(0)], 1)  # above, inline, below

def band_probs_normal(m, sd):
    return np.stack([1 - stats.norm.cdf(THR, m, sd), stats.norm.cdf(THR, m, sd) - stats.norm.cdf(-THR, m, sd), stats.norm.cdf(-THR, m, sd)], 1)

def score_probs(P, ytrue):
    """P (N,3) columns above/inline/below; ytrue array of labels. Returns Brier (multiclass), log loss, modal accuracy."""
    lab = np.array(['above', 'inline', 'below'])
    Y = (ytrue[:, None] == lab[None, :]).astype(float)
    P = np.clip(P, 1e-6, 1); P = P / P.sum(1, keepdims=True)
    brier = ((P - Y) ** 2).sum(1).mean()
    ll = -np.log((P * Y).sum(1)).mean()
    acc = (lab[P.argmax(1)] == ytrue).mean()
    return dict(brier=float(brier), logloss=float(ll), modal_acc=float(acc))

def run_holdout(train_years, test_year, tag):
    tr = prov[prov.Year.isin(train_years)].copy()
    te = prov[prov.Year == test_year].copy()
    idx = build_index(tr, 'UKPRN')
    d = idx['df']
    out = {}
    fits = {}
    for name, kwargs in [('M0_nocohort', dict(cohort=False)), ('M1_cohort', dict(cohort=True)), ('M2_cohort_trend', dict(cohort=True, trend=True))]:
        if tag == 'rolling' and name != 'M1_cohort':
            continue
        t0 = time.time()
        mcmc, kw = fit(idx, num_warmup=NW, num_samples=NS, num_chains=NC, target_accept=0.95, **kwargs)
        el = time.time() - t0
        summ, diag, idata = diagnostics(mcmc)
        S = flatten_samples(mcmc)
        print(name, tag, f'{el:.0f}s', diag, file=sys.stderr)
        summ.to_csv(f'../out/holdout_{tag}_{name}_hyper.csv')
        # ---- predictions for test year ----
        te2 = te.copy()
        te2['u_i'] = idx['units'].get_indexer(te2['UKPRN'])
        te2['q_i'] = idx['themes'].get_indexer(te2['theme'])
        te2 = te2.merge(idx['uq'][['u_i', 'q_i', 'v_i']], on=['u_i', 'q_i'], how='left')
        seen = te2['u_i'] >= 0
        mu = S['mu'][:, None]; alpha = S['alpha'][:, te2['q_i'].values]
        u = np.where(seen.values[None, :], S['u'][:, np.clip(te2['u_i'].values, 0, None)], 0.0)
        v = np.where(seen.values[None, :], S['v'][:, np.clip(te2['v_i'].fillna(0).astype(int).values, 0, None)], 0.0)
        m_pred = mu + alpha + u + v
        if kwargs.get('trend'):
            b = np.where(seen.values[None, :], S['b'][:, np.clip(te2['u_i'].values, 0, None)], 0.0)
            m_pred = m_pred + b * (test_year - 2024.5)
        rng = np.random.default_rng(7)
        Sn = m_pred.shape[0]; N = m_pred.shape[1]
        gam_new = S['sd_gamma'][:, None] * rng.standard_normal((Sn, 1))
        c_new = (S['sd_c'][:, None] * rng.standard_normal((Sn, N))) if kwargs.get('cohort') else 0.0
        # cohort shock is shared across themes of a unit-year: draw once per unit and broadcast
        if kwargs.get('cohort'):
            uid = te2['UKPRN'].astype('category').cat.codes.values
            cu = S['sd_c'][:, None] * rng.standard_normal((Sn, uid.max() + 1))
            c_new = cu[:, uid]
        eps_new = S['sd_eps'][:, None] * rng.standard_normal((Sn, N))
        meas = te2['s'].values[None, :] * rng.standard_normal((Sn, N))
        latent_pred = m_pred + gam_new + c_new + eps_new          # latent 2026 gap (no sampling error)
        D_pred = latent_pred + meas                                # observed 2026 gap
        te2['pred_mean'] = m_pred.mean(0)
        te2['pred_latent_mean'] = latent_pred.mean(0)
        te2['pred_lo90'] = np.quantile(D_pred, 0.05, 0); te2['pred_hi90'] = np.quantile(D_pred, 0.95, 0)
        te2['pred_lo50'] = np.quantile(D_pred, 0.25, 0); te2['pred_hi50'] = np.quantile(D_pred, 0.75, 0)
        te2['crps'] = crps_from_draws(D_pred, te2['D'].values)
        P = band_probs_from_draws(D_pred)
        te2['p_above_pred'], te2['p_inline_pred'], te2['p_below_pred'] = P[:, 0], P[:, 1], P[:, 2]
        # log predictive density of observed D under normal approx of predictive
        te2['lpd'] = stats.norm.logpdf(te2['D'].values, D_pred.mean(0), D_pred.std(0))
        te2['seen'] = seen.values
        te2.to_csv(f'../out/holdout_{tag}_{name}_pred.csv', index=False)
        fits[name] = te2
        out[name] = dict(diag=diag, hyper=summ[['mean', 'sd']].to_dict(), elapsed_s=el)
    # ---- baselines ----
    w = prov.pivot_table(index=['UKPRN', 'theme'], columns='Year', values=['D', 's', 'p_above', 'p_inline', 'p_below'])
    last_year = max(train_years)
    base = fits['M1_cohort'][['UKPRN', 'theme', 'D', 's', 'seen']].copy().set_index(['UKPRN', 'theme'])
    base['last'] = w[('D', last_year)]
    vals = w[[('D', y) for y in train_years]].values; wts = 1 / w[[('s', y) for y in train_years]].values ** 2
    mean3 = pd.Series(np.nanmean(vals, 1), index=w.index)
    with np.errstate(invalid='ignore', divide='ignore'):
        pw = pd.Series(np.nansum(np.where(np.isnan(vals), 0, vals * wts), 1) / np.nansum(np.where(np.isnan(vals), 0, wts), 1), index=w.index)
        pw_se = pd.Series(np.sqrt(1 / np.nansum(np.where(np.isnan(vals), 0, wts), 1)), index=w.index)
    base['mean3'] = mean3; base['pw'] = pw; base['pw_se'] = pw_se; base['s_last'] = w[('s', last_year)]
    for c_ in ['p_above', 'p_inline', 'p_below']:
        base['ofs_' + c_] = w[(c_, last_year)] / 100.0
    for name in fits:
        base['pred_' + name] = fits[name].set_index(['UKPRN', 'theme'])['pred_mean']
    base = base.reset_index()
    base.to_csv(f'../out/holdout_{tag}_all_predictions.csv', index=False)
    # ---- point metrics ----
    metrics = {}
    sets = {'seen_units': base['seen'], 'matched_last': base['last'].notna()}
    for sname, mask in sets.items():
        b = base[mask]
        met = {}
        for col in ['last', 'mean3', 'pw'] + ['pred_' + n for n in fits]:
            ok = b[col].notna(); e = b.loc[ok, col] - b.loc[ok, 'D']
            met[col] = dict(n=int(ok.sum()), RMSE=float(np.sqrt((e ** 2).mean())), MAE=float(e.abs().mean()),
                            corr=float(np.corrcoef(b.loc[ok, col], b.loc[ok, 'D'])[0, 1]))
        metrics[sname] = met
    # provider-cluster bootstrap for RMSE differences (M1 vs each baseline), matched set
    b = base[base['last'].notna()].copy()
    provs = b['UKPRN'].unique(); rng = np.random.default_rng(11)
    diffs = {k: [] for k in ['last', 'mean3', 'pw', 'pred_M0_nocohort', 'pred_M2_cohort_trend'] if k in b}
    grp = {p: b[b.UKPRN == p] for p in provs}
    for _ in range(1000):
        samp = rng.choice(provs, len(provs), replace=True)
        bb = pd.concat([grp[p] for p in samp])
        r1 = np.sqrt(((bb['pred_M1_cohort'] - bb['D']) ** 2).mean())
        for k in diffs:
            diffs[k].append(np.sqrt(((bb[k] - bb['D']) ** 2).mean()) - r1)
    metrics['rmse_gain_M1_vs_bootstrap90'] = {k: dict(mean=float(np.mean(v)), lo=float(np.quantile(v, .05)), hi=float(np.quantile(v, .95))) for k, v in diffs.items()}
    # ---- probabilistic metrics on 3-band classification of observed test-year D ----
    ytrue = band(b['D'].values)
    prob = {}
    for name in fits:
        f = fits[name].set_index(['UKPRN', 'theme']).loc[list(zip(b.UKPRN, b.theme))]
        P = f[['p_above_pred', 'p_inline_pred', 'p_below_pred']].values
        prob[name] = score_probs(P, ytrue)
        prob[name].update(dict(coverage90=float(((b['D'].values >= f['pred_lo90'].values) & (b['D'].values <= f['pred_hi90'].values)).mean()),
                               coverage50=float(((b['D'].values >= f['pred_lo50'].values) & (b['D'].values <= f['pred_hi50'].values)).mean()),
                               mean_crps=float(f['crps'].mean()), mean_lpd=float(f['lpd'].mean()),
                               mean_width90=float((f['pred_hi90'] - f['pred_lo90']).mean())))
    # baseline: OfS published materiality probabilities from the last training year
    P_ofs = b[['ofs_p_above', 'ofs_p_inline', 'ofs_p_below']].values
    prob['OfS_lastyear_probs'] = score_probs(P_ofs, ytrue)
    # baseline: persistence normal N(D_last, s_last^2 + s_test^2)
    sd_pers = np.sqrt(b['s_last'] ** 2 + b['s'] ** 2)
    P_pers = band_probs_normal(b['last'].values, sd_pers.values)
    prob['persistence_normal'] = score_probs(P_pers, ytrue)
    prob['persistence_normal']['coverage90'] = float((np.abs(b['D'] - b['last']) <= 1.645 * sd_pers).mean())
    prob['persistence_normal']['mean_crps'] = float(crps_normal(b['last'].values, sd_pers.values, b['D'].values).mean())
    # baseline: precision-weighted mean normal N(pw, pw_se^2 + s_test^2)
    sd_pw = np.sqrt(b['pw_se'] ** 2 + b['s'] ** 2)
    prob['pw_normal'] = score_probs(band_probs_normal(b['pw'].values, sd_pw.values), ytrue)
    prob['pw_normal']['coverage90'] = float((np.abs(b['D'] - b['pw']) <= 1.645 * sd_pw).mean())
    prob['pw_normal']['mean_crps'] = float(crps_normal(b['pw'].values, sd_pw.values, b['D'].values).mean())
    # climatological baseline (base rates)
    rates = pd.Series(ytrue).value_counts(normalize=True).reindex(['above', 'inline', 'below']).fillna(0).values
    prob['climatology'] = score_probs(np.tile(rates, (len(b), 1)), ytrue)
    metrics['probabilistic_matched'] = prob
    # calibration table for M1
    f = fits['M1_cohort'].set_index(['UKPRN', 'theme']).loc[list(zip(b.UKPRN, b.theme))]
    cal = []
    for col, lab in [('p_above_pred', 'above'), ('p_below_pred', 'below')]:
        p = f[col].values; y = (ytrue == lab).astype(float)
        bins = np.linspace(0, 1, 11)
        for i in range(10):
            mk = (p >= bins[i]) & (p < bins[i + 1] + (i == 9))
            if mk.sum() > 0:
                cal.append(dict(target=lab, bin_lo=bins[i], bin_hi=bins[i + 1], n=int(mk.sum()), mean_pred=float(p[mk].mean()), obs_freq=float(y[mk].mean())))
    pd.DataFrame(cal).to_csv(f'../out/holdout_{tag}_M1_calibration.csv', index=False)
    # OfS calibration too
    calo = []
    for col, lab in [('ofs_p_above', 'above'), ('ofs_p_below', 'below')]:
        p = b[col].values; y = (ytrue == lab).astype(float)
        bins = np.linspace(0, 1, 11)
        for i in range(10):
            mk = (p >= bins[i]) & (p < bins[i + 1] + (i == 9))
            if mk.sum() > 0:
                calo.append(dict(target=lab, bin_lo=bins[i], bin_hi=bins[i + 1], n=int(mk.sum()), mean_pred=float(p[mk].mean()), obs_freq=float(y[mk].mean())))
    pd.DataFrame(calo).to_csv(f'../out/holdout_{tag}_OfS_calibration.csv', index=False)
    # ---- metrics by size band (matched, M1 vs last) ----
    n_last = prov[prov.Year == last_year].set_index(['UKPRN', 'theme'])['n_resp']
    b['n_last'] = n_last.reindex(list(zip(b.UKPRN, b.theme))).values
    b['band'] = pd.cut(b['n_last'], [0, 50, 100, 200, np.inf], labels=['<50', '50-99', '100-199', '200+'])
    bysize = b.groupby('band', observed=True).apply(lambda g: pd.Series({'n': len(g),
        'RMSE_last': np.sqrt(((g['last'] - g['D']) ** 2).mean()), 'RMSE_pw': np.sqrt(((g['pw'] - g['D']) ** 2).mean()),
        'RMSE_M1': np.sqrt(((g['pred_M1_cohort'] - g['D']) ** 2).mean())}))
    bysize.to_csv(f'../out/holdout_{tag}_rmse_by_size.csv')
    metrics['by_size'] = bysize.to_dict()
    out['metrics'] = metrics
    json.dump(out, open(f'../out/holdout_{tag}_summary.json', 'w'), indent=2, default=str)
    return out

if __name__ == '__main__':
    which = sys.argv[1] if len(sys.argv) > 1 else 'main'
    if which == 'main':
        o = run_holdout([2023, 2024, 2025], 2026, 'main')
        print(json.dumps(o['metrics'], indent=2, default=str))
    else:
        o = run_holdout([2023, 2024], 2025, 'rolling')
        print(json.dumps(o['metrics'], indent=2, default=str))
