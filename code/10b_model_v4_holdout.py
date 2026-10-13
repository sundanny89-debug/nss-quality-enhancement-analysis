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

res = {}
# =========================== held-out fit (train 2023-25, predict 2026) ===========================
tr = prov[prov.Year <= 2025]; te = prov[prov.Year == 2026]
idxa = build_index(tr, 'UKPRN'); da = idxa['df']
t0 = time.time(); mcmc, kw = fit_v4(idxa, num_warmup=NW, num_samples=NS, num_chains=NC, target_accept=0.95, seed=401)
summ, diag, _ = diagnostics(mcmc, params=('mu', 'sd_gamma', 'sd_u', 'sd_c', 'rho', 'lam', 'kappa', 'sd_v_q')); res['holdout_diag'] = diag
print('M4 holdout', f'{time.time()-t0:.0f}s', diag, file=sys.stderr)
summ.to_csv('../out/M4_holdout_hyper.csv'); Sa = flatten_samples(mcmc); del mcmc
te2 = te.copy(); te2['u_i'] = idxa['units'].get_indexer(te2.UKPRN); te2['q_i'] = idxa['themes'].get_indexer(te2.theme)
te2 = te2.merge(idxa['uq'][['u_i', 'q_i', 'v_i']], on=['u_i', 'q_i'], how='left'); seen = (te2.u_i >= 0).values
Sn = Sa['mu'].shape[0]; N = len(te2); rng = np.random.default_rng(5)
vi = np.clip(te2.v_i.fillna(0).astype(int).values, 0, None); ui = np.clip(te2.u_i.values, 0, None)
kap_last = Sa['kappa'][:, -1][:, None]   # carry the last observed year scaling forward
m_pred = Sa['mu'][:, None] + Sa['alpha'][:, te2.q_i.values] + np.where(seen[None, :], kap_last * Sa['lam'][:, te2.q_i.values] * Sa['u'][:, ui] + Sa['v'][:, vi], 0.0)
gam_new = Sa['sd_gamma'][:, None] * rng.standard_normal((Sn, 1))
codes = te2.UKPRN.astype('category').cat.codes.values
c_new = (Sa['sd_c'][:, None] * rng.standard_normal((Sn, codes.max() + 1)))[:, codes]
g_new = rng.standard_normal((Sn, codes.max() + 1))[:, codes]; z = rng.standard_normal((Sn, N)); rho_a = Sa['rho'][:, None]
D_pred = m_pred + gam_new + c_new + te2.s.values[None, :] * (np.sqrt(rho_a) * g_new + np.sqrt(1 - rho_a) * z)
te2['pred_mean'] = m_pred.mean(0)
for lv in [50, 60, 70, 80, 90, 95]:
    te2[f'lo{lv}'] = np.quantile(D_pred, (1 - lv / 100) / 2, 0); te2[f'hi{lv}'] = np.quantile(D_pred, 1 - (1 - lv / 100) / 2, 0)
S_ = D_pred.shape[0]; xs = np.sort(D_pred, axis=0); ii = np.arange(1, S_ + 1)[:, None]
te2['crps'] = np.abs(D_pred - te2.D.values[None, :]).mean(0) - 0.5 * (2.0 / (S_ * S_)) * ((2 * ii - S_ - 1) * xs).sum(0)
te2['lpd'] = stats.norm.logpdf(te2.D.values, D_pred.mean(0), D_pred.std(0))
te2['p_above'] = (D_pred > THR).mean(0); te2['p_below'] = (D_pred < -THR).mean(0); te2['p_inline'] = 1 - te2.p_above - te2.p_below; te2['seen'] = seen
te2.to_csv('../out/M4_holdout_predictions.csv', index=False)
b = pd.read_csv('../out/final_A_main2026_predictions.csv')[['UKPRN', 'theme', 'D', 's', 'last', 's_last', 'mean3', 'pw', 'pw_se', 'eb', 'pred_mean', 'crps', 'lpd', 'p_above', 'p_inline', 'p_below'] + [f'{a}{lv}' for lv in [50, 60, 70, 80, 90, 95] for a in ['lo', 'hi']]]
b = b.rename(columns={c_: c_ + '_G' for c_ in ['pred_mean', 'crps', 'lpd', 'p_above', 'p_inline', 'p_below'] + [f'{a}{lv}' for lv in [50, 60, 70, 80, 90, 95] for a in ['lo', 'hi']]})
b = b.merge(te2[['UKPRN', 'theme', 'pred_mean', 'crps', 'lpd', 'p_above', 'p_inline', 'p_below'] + [f'{a}{lv}' for lv in [50, 60, 70, 80, 90, 95] for a in ['lo', 'hi']]], on=['UKPRN', 'theme'])
bm = b[b['last'].notna()].copy(); y = bm.D.values; ytrue = band(np.round(y, 1))
met = {}
for col in ['last', 'mean3', 'pw', 'eb', 'pred_mean_G', 'pred_mean']:
    e = bm[col] - bm.D; met[col] = dict(RMSE=float(np.sqrt((e ** 2).mean())), MAE=float(e.abs().mean()))
res['holdout_point_matched'] = dict(n=int(len(bm)), **met)
provs = bm.UKPRN.unique(); rng2 = np.random.default_rng(11); grp = {p: bm[bm.UKPRN == p] for p in provs}; diffs = {k: [] for k in ['last', 'pw', 'eb', 'pred_mean_G']}
for _ in range(1000):
    bb = pd.concat([grp[p] for p in rng2.choice(provs, len(provs), replace=True)]); r1 = np.sqrt(((bb.pred_mean - bb.D) ** 2).mean())
    for k in diffs: diffs[k].append(np.sqrt(((bb[k] - bb.D) ** 2).mean()) - r1)
res['holdout_rmse_gain_M4_vs_bootstrap90'] = {k: dict(mean=float(np.mean(v)), lo=float(np.quantile(v, .05)), hi=float(np.quantile(v, .95))) for k, v in diffs.items()}
def score_probs(P, ytrue):
    lab = np.array(['above', 'inline', 'below']); Y = (ytrue[:, None] == lab[None, :]).astype(float); P = np.clip(P, 1e-6, 1); P = P / P.sum(1, keepdims=True)
    return dict(brier=float(((P - Y) ** 2).sum(1).mean()), logloss=float(-np.log((P * Y).sum(1)).mean()), modal_acc=float((lab[P.argmax(1)] == ytrue).mean()))
prob = {}
for tag_, sfx in [('guide_spec', '_G'), ('M4', '')]:
    prob[tag_] = dict(**score_probs(bm[[f'p_above{sfx}', f'p_inline{sfx}', f'p_below{sfx}']].values, ytrue), coverage={lv: float(((y >= bm[f'lo{lv}{sfx}']) & (y <= bm[f'hi{lv}{sfx}'])).mean()) for lv in [50, 60, 70, 80, 90, 95]},
                      mean_width90=float((bm[f'hi90{sfx}'] - bm[f'lo90{sfx}']).mean()), mean_crps=float(bm[f'crps{sfx}'].mean()), mean_lpd=float(bm[f'lpd{sfx}'].mean()))
res['holdout_probabilistic_matched'] = prob
n_last = prov[prov.Year == 2025].set_index(['UKPRN', 'theme'])['n_resp']; bm['n_last'] = n_last.reindex(list(zip(bm.UKPRN, bm.theme))).values
bm['band'] = pd.cut(bm.n_last, [0, 50, 100, 200, np.inf], labels=['<50', '50-99', '100-199', '200+'])
res['holdout_rmse_by_size'] = bm.groupby('band', observed=True).apply(lambda g: pd.Series({'n': len(g), 'last': np.sqrt(((g['last'] - g.D) ** 2).mean()), 'pw': np.sqrt(((g.pw - g.D) ** 2).mean()), 'G': np.sqrt(((g.pred_mean_G - g.D) ** 2).mean()), 'M4': np.sqrt(((g.pred_mean - g.D) ** 2).mean())})).round(2).to_dict()
json.dump(res, open('../out/M4_holdout_summary.json', 'w'), indent=2, default=str)
print(json.dumps(res, indent=2, default=str))
