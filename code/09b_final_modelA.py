"""Step 9b: FINAL Model A (2023-2025 only) under the guide specification; held-out 2026 evaluation.
Baselines: (1) 2025 observed D; (2) 2023-25 mean; (3) precision-weighted mean; (4) transparent empirical-Bayes prototype
(per-theme DerSimonian-Laird tau^2, shrink each provider's precision-weighted 2023-25 mean toward the theme mean); (5) Bayesian posterior predictive.
Metrics: RMSE, MAE, coverage at 50/60/70/80/90/95%, mean interval widths, CRPS, log predictive density, band-probability calibration.
Cluster bootstrap by provider for RMSE differences. Rolling-origin robustness (train 2023-24 -> 2025) also under the guide spec."""
import pandas as pd, numpy as np, json, time, sys, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy import stats
from bayes_model import *
plt.style.use('seaborn-v0_8-whitegrid'); plt.rcParams.update({'figure.dpi': 150, 'font.size': 10, 'axes.titlesize': 11, 'axes.titleweight': 'bold'})
BLUE, ORANGE, GREEN, RED, GREY, PURPLE = '#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8c8c8c', '#8172B3'
THR = 2.5
PRI = dict(PRIORS); PRI.update(sd_eta=3.0, alpha_fixed_sd=3.0)
prov = pd.read_csv('../data/sector_cah10_provider_theme.csv')
def band(x): return np.where(x > THR, 'above', np.where(x < -THR, 'below', 'inline'))
def crps_draws(draws, y):
    S = draws.shape[0]; t1 = np.abs(draws - y[None, :]).mean(0); x = np.sort(draws, axis=0); i = np.arange(1, S + 1)[:, None]
    return t1 - 0.5 * (2.0 / (S * S)) * ((2 * i - S - 1) * x).sum(0)
def crps_normal(m, sd, y):
    z = (y - m) / sd; return sd * (z * (2 * stats.norm.cdf(z) - 1) + 2 * stats.norm.pdf(z) - 1 / np.sqrt(np.pi))
def score_probs(P, ytrue):
    lab = np.array(['above', 'inline', 'below']); Y = (ytrue[:, None] == lab[None, :]).astype(float); P = np.clip(P, 1e-6, 1); P = P / P.sum(1, keepdims=True)
    return dict(brier=float(((P - Y) ** 2).sum(1).mean()), logloss=float(-np.log((P * Y).sum(1)).mean()), modal_acc=float((lab[P.argmax(1)] == ytrue).mean()))
def band_probs_normal(m, sd):
    return np.stack([1 - stats.norm.cdf(THR, m, sd), stats.norm.cdf(THR, m, sd) - stats.norm.cdf(-THR, m, sd), stats.norm.cdf(-THR, m, sd)], 1)

def eb_prototype(w, train_years):
    """Per theme: precision-weighted provider means over training years; DL tau^2; shrink toward theme mean."""
    vals = w[[('D', y) for y in train_years]].values; wts = 1 / w[[('s', y) for y in train_years]].values ** 2
    with np.errstate(invalid='ignore', divide='ignore'):
        pw = np.nansum(np.where(np.isnan(vals), 0, vals * wts), 1) / np.nansum(np.where(np.isnan(vals), 0, wts), 1)
        se2 = 1 / np.nansum(np.where(np.isnan(vals), 0, wts), 1)
    out = pd.Series(np.nan, index=w.index)
    themes = w.index.get_level_values('theme')
    for q in np.unique(themes):
        mk = (themes == q) & ~np.isnan(pw)
        y = pw[mk]; v = se2[mk]; wq = 1 / v; ybar = (wq * y).sum() / wq.sum(); Q = (wq * (y - ybar) ** 2).sum(); k = len(y)
        tau2 = max(0.0, (Q - (k - 1)) / (wq.sum() - (wq ** 2).sum() / wq.sum()))
        shr = (y / v + ybar / tau2) / (1 / v + 1 / tau2) if tau2 > 0 else np.full(k, ybar)
        out[mk] = shr
    return out

def run(train_years, test_year, tag, nw, ns):
    tr = prov[prov.Year.isin(train_years)]; te = prov[prov.Year == test_year]
    idx = build_index(tr, 'UKPRN'); d = idx['df']
    t0 = time.time(); mcmc, kw = fit(idx, cohort=True, dynamic=True, eps=False, alpha_prior='fixed', priors=PRI, num_warmup=nw, num_samples=ns, num_chains=4, target_accept=0.95, seed=200 + test_year)
    summ, diag, _ = diagnostics(mcmc, params=('mu', 'sd_gamma', 'sd_u', 'sd_v', 'sd_c', 'sd_eta')); S = flatten_samples(mcmc); del mcmc
    print(tag, f'{time.time()-t0:.0f}s', diag, file=sys.stderr)
    summ.to_csv(f'../out/final_A_{tag}_hyper.csv')
    te2 = te.copy(); te2['u_i'] = idx['units'].get_indexer(te2.UKPRN); te2['q_i'] = idx['themes'].get_indexer(te2.theme)
    te2 = te2.merge(idx['uq'][['u_i', 'q_i', 'v_i']], on=['u_i', 'q_i'], how='left'); seen = (te2.u_i >= 0).values
    Sn = S['mu'].shape[0]; N = len(te2); rng = np.random.default_rng(3)
    vi = np.clip(te2.v_i.fillna(0).astype(int).values, 0, None); ui = np.clip(te2.u_i.values, 0, None)
    w_last = S['w_dyn'][:, -1, :]
    m_pred = S['mu'][:, None] + S['alpha'][:, te2.q_i.values] + np.where(seen[None, :], S['u'][:, ui] + S['v'][:, vi] + w_last[:, vi], 0.0)
    gam_new = S['sd_gamma'][:, None] * rng.standard_normal((Sn, 1))
    codes = te2.UKPRN.astype('category').cat.codes.values; cu = S['sd_c'][:, None] * rng.standard_normal((Sn, codes.max() + 1)); c_new = cu[:, codes]
    eta_new = S['sd_eta'][:, None] * rng.standard_normal((Sn, N))
    latent_pred = m_pred + gam_new + c_new + eta_new; D_pred = latent_pred + te2.s.values[None, :] * rng.standard_normal((Sn, N))
    te2['pred_mean'] = m_pred.mean(0); te2['pred_sd'] = D_pred.std(0)
    for lv in [50, 60, 70, 80, 90, 95]:
        te2[f'lo{lv}'] = np.quantile(D_pred, (1 - lv / 100) / 2, 0); te2[f'hi{lv}'] = np.quantile(D_pred, 1 - (1 - lv / 100) / 2, 0)
    te2['crps'] = crps_draws(D_pred, te2.D.values); te2['lpd'] = stats.norm.logpdf(te2.D.values, D_pred.mean(0), D_pred.std(0))
    te2['p_above'] = (D_pred > THR).mean(0); te2['p_below'] = (D_pred < -THR).mean(0); te2['p_inline'] = 1 - te2.p_above - te2.p_below
    te2['seen'] = seen
    # baselines
    w = prov.pivot_table(index=['UKPRN', 'theme'], columns='Year', values=['D', 's', 'p_above', 'p_inline', 'p_below'])
    last = max(train_years); b = te2.set_index(['UKPRN', 'theme'])
    b['last'] = w[('D', last)]; b['s_last'] = w[('s', last)]
    vals = w[[('D', y) for y in train_years]].values; wts = 1 / w[[('s', y) for y in train_years]].values ** 2
    with np.errstate(invalid='ignore', divide='ignore'):
        b['mean3'] = pd.Series(np.nanmean(vals, 1), index=w.index)
        b['pw'] = pd.Series(np.nansum(np.where(np.isnan(vals), 0, vals * wts), 1) / np.nansum(np.where(np.isnan(vals), 0, wts), 1), index=w.index)
        b['pw_se'] = pd.Series(np.sqrt(1 / np.nansum(np.where(np.isnan(vals), 0, wts), 1)), index=w.index)
    b['eb'] = eb_prototype(w, train_years)
    for c_ in ['p_above', 'p_inline', 'p_below']: b['ofs_' + c_] = w[(c_, last)] / 100
    b = b.reset_index(); b.to_csv(f'../out/final_A_{tag}_predictions.csv', index=False)
    bm = b[b['last'].notna()].copy(); res = dict(diag=diag, hyper=summ[['mean', 'sd']].round(3).to_dict(), n_matched=int(len(bm)), n_seen=int(b.seen.sum()))
    met = {}
    for col in ['last', 'mean3', 'pw', 'eb', 'pred_mean']:
        for nm, bb in [('matched', bm), ('seen', b[b.seen])]:
            ok = bb[col].notna(); e = bb.loc[ok, col] - bb.loc[ok, 'D']
            met[f'{col}_{nm}'] = dict(n=int(ok.sum()), RMSE=float(np.sqrt((e ** 2).mean())), MAE=float(e.abs().mean()))
    res['point_metrics'] = met
    provs = bm.UKPRN.unique(); rng2 = np.random.default_rng(11); grp = {p: bm[bm.UKPRN == p] for p in provs}
    diffs = {k: [] for k in ['last', 'mean3', 'pw', 'eb']}
    for _ in range(1000):
        bb = pd.concat([grp[p] for p in rng2.choice(provs, len(provs), replace=True)]); r1 = np.sqrt(((bb.pred_mean - bb.D) ** 2).mean())
        for k in diffs: diffs[k].append(np.sqrt(((bb[k] - bb.D) ** 2).mean()) - r1)
    res['rmse_gain_bayes_vs_bootstrap90'] = {k: dict(mean=float(np.mean(v)), lo=float(np.quantile(v, .05)), hi=float(np.quantile(v, .95))) for k, v in diffs.items()}
    # probabilistic
    y = bm.D.values; ytrue = band(np.round(y, 1))
    cov = {lv: float(((y >= bm[f'lo{lv}']) & (y <= bm[f'hi{lv}'])).mean()) for lv in [50, 60, 70, 80, 90, 95]}
    wid = {lv: float((bm[f'hi{lv}'] - bm[f'lo{lv}']).mean()) for lv in [50, 60, 70, 80, 90, 95]}
    prob = {'bayes': dict(**score_probs(bm[['p_above', 'p_inline', 'p_below']].values, ytrue), coverage=cov, mean_width=wid, mean_crps=float(bm.crps.mean()), mean_lpd=float(bm.lpd.mean()))}
    sd_pers = np.sqrt(bm.s_last ** 2 + bm.s ** 2); sd_pw = np.sqrt(bm.pw_se ** 2 + bm.s ** 2)
    for nm, m_, sd_ in [('persistence_normal', bm['last'].values, sd_pers.values), ('pw_normal', bm.pw.values, sd_pw.values), ('eb_normal', bm.eb.values, sd_pw.values)]:
        prob[nm] = dict(**score_probs(band_probs_normal(m_, sd_), ytrue), coverage={lv: float((np.abs(y - m_) <= stats.norm.ppf(0.5 + lv / 200) * sd_).mean()) for lv in [50, 60, 70, 80, 90, 95]},
                        mean_width={lv: float((2 * stats.norm.ppf(0.5 + lv / 200) * sd_).mean()) for lv in [50, 60, 70, 80, 90, 95]}, mean_crps=float(crps_normal(m_, sd_, y).mean()),
                        mean_lpd=float(stats.norm.logpdf(y, m_, sd_).mean()))
    prob['OfS_last_year_probs'] = score_probs(bm[['ofs_p_above', 'ofs_p_inline', 'ofs_p_below']].values, ytrue)
    rates = pd.Series(ytrue).value_counts(normalize=True).reindex(['above', 'inline', 'below']).fillna(0).values
    prob['climatology'] = score_probs(np.tile(rates, (len(bm), 1)), ytrue)
    res['probabilistic_matched'] = prob
    # size bands
    n_last = prov[prov.Year == last].set_index(['UKPRN', 'theme'])['n_resp']; bm['n_last'] = n_last.reindex(list(zip(bm.UKPRN, bm.theme))).values
    bm['band'] = pd.cut(bm.n_last, [0, 50, 100, 200, np.inf], labels=['<50', '50-99', '100-199', '200+'])
    res['rmse_by_size'] = bm.groupby('band', observed=True).apply(lambda g: pd.Series({'n': len(g), 'last': np.sqrt(((g['last'] - g.D) ** 2).mean()), 'pw': np.sqrt(((g.pw - g.D) ** 2).mean()),
                                                                                    'eb': np.sqrt(((g.eb - g.D) ** 2).mean()), 'bayes': np.sqrt(((g.pred_mean - g.D) ** 2).mean())})).round(2).to_dict()
    # calibration tables
    cal = []
    for col, lab in [('p_above', 'above'), ('p_below', 'below')]:
        p = bm[col].values; yy = (ytrue == lab).astype(float); bins = np.linspace(0, 1, 11)
        for i in range(10):
            mk = (p >= bins[i]) & (p < bins[i + 1] + (i == 9))
            if mk.sum(): cal.append(dict(target=lab, bin=f'{bins[i]:.1f}-{bins[i+1]:.1f}', n=int(mk.sum()), mean_pred=float(p[mk].mean()), obs_freq=float(yy[mk].mean())))
    pd.DataFrame(cal).to_csv(f'../out/final_A_{tag}_calibration_bands.csv', index=False)
    return res, bm, cal

if __name__ == '__main__':
    out = {}
    out['main_2026'], bm, cal = run([2023, 2024, 2025], 2026, 'main2026', 2000, 2000)
    out['rolling_2025'], _, _ = run([2023, 2024], 2025, 'rolling2025', 1000, 1000)
    json.dump(out, open('../out/final_A_summary.json', 'w'), indent=2, default=str)
    # figures: interval calibration + band calibration + RMSE
    r = out['main_2026']
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))
    a = ax[0]; lv = [50, 60, 70, 80, 90, 95]
    for nm, col, lab in [('bayes', BLUE, 'Bayesian posterior predictive'), ('persistence_normal', GREY, 'persistence N(D₂₀₂₅, s²₂₀₂₅+s²₂₀₂₆)'), ('pw_normal', ORANGE, 'precision-weighted N'), ('eb_normal', GREEN, 'EB prototype N')]:
        a.plot(lv, [100 * r['probabilistic_matched'][nm]['coverage'][l] for l in lv], marker='o', color=col, label=lab)
    a.plot([45, 100], [45, 100], color='k', ls='--', lw=0.8); a.set_xlabel('Nominal predictive interval (%)'); a.set_ylabel('Observed coverage of 2026 gaps (%)'); a.set_title('Predictive-interval calibration, held-out 2026'); a.legend(fontsize=7)
    a = ax[1]; caldf = pd.DataFrame(cal)
    for tgt, mk in [('below', 'o'), ('above', 's')]:
        g = caldf[caldf.target == tgt]; a.plot(g.mean_pred, g.obs_freq, marker=mk, color=BLUE, label=f'Bayes P({tgt})')
    ofs = pd.read_csv('../out/holdout_main_OfS_calibration.csv')
    for tgt, mk in [('below', 'o'), ('above', 's')]:
        g = ofs[ofs.target == tgt]; a.plot(g.mean_pred, g.obs_freq, marker=mk, color=ORANGE, label=f'OfS 2025 P({tgt})')
    a.plot([0, 1], [0, 1], color='k', ls='--', lw=0.8); a.set_xlabel('Forecast probability'); a.set_ylabel('Observed frequency in 2026'); a.set_title('Materiality-band probability calibration'); a.legend(fontsize=7)
    a = ax[2]; pm = r['point_metrics']
    keys = [('last_matched', '2025 observed'), ('mean3_matched', '2023–25 mean'), ('pw_matched', 'precision-weighted'), ('eb_matched', 'EB prototype'), ('pred_mean_matched', 'Bayesian (guide spec)')]
    a.barh([l for _, l in keys], [pm[k]['RMSE'] for k, _ in keys], color=[GREY, GREY, GREY, GREEN, BLUE])
    for i, (k, _) in enumerate(keys): a.text(pm[k]['RMSE'] + 0.05, i, f"{pm[k]['RMSE']:.2f}", va='center', fontsize=9)
    a.set_xlabel('RMSE, held-out 2026 (pp)'); a.set_title(f"Point prediction, n={pm['last_matched']['n']}"); a.invert_yaxis(); a.set_xlim(0, 7.5)
    plt.tight_layout(); plt.savefig('../fig/B3_heldout_final.png', bbox_inches='tight'); plt.close()
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != 'hyper'} for k, v in out.items()}, indent=2, default=str))
