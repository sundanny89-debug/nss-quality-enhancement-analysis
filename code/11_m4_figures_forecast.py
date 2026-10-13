"""Step 11: figures for the completed analysis and M4-based 2027 forecasts."""
import pandas as pd, numpy as np, json, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.style.use('seaborn-v0_8-whitegrid'); plt.rcParams.update({'figure.dpi': 150, 'font.size': 10, 'axes.titlesize': 11, 'axes.titleweight': 'bold'})
BLUE, ORANGE, GREEN, RED, GREY, PURPLE = '#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8c8c8c', '#8172B3'
THR = 2.5; OUT = '../out/'; FIG = '../fig/'
THEMES = {1: 'Teaching', 2: 'Learning opps', 3: 'Assessment & feedback', 4: 'Academic support', 5: 'Organisation & mgmt', 6: 'Learning resources', 7: 'Student voice'}
mf = json.load(open(OUT + 'M4_full_summary.json')); mh = json.load(open(OUT + 'M4_holdout_summary.json')); fa = json.load(open(OUT + 'final_A_summary.json'))['main_2026']

# ---------------- B6: M4 structure ----------------
fig, ax = plt.subplots(1, 4, figsize=(15, 3.8))
lam = pd.DataFrame(mf['loadings_lambda_by_theme']).T; sdv = pd.DataFrame(mf['sd_v_by_theme']).T; kap = pd.DataFrame(mf['kappa_by_year']).T
a = ax[0]; y = np.arange(7)
a.errorbar(lam['mean'], y, xerr=[lam['mean'] - lam['lo95'], lam['hi95'] - lam['mean']], fmt='o', color=BLUE, capsize=2); a.axvline(1, color=GREY, ls='--', lw=0.8)
a.set_yticks(y); a.set_yticklabels(lam.index, fontsize=8); a.invert_yaxis(); a.set_xlabel('loading λ_q on provider effect (Teaching = 1)'); a.set_title('How strongly each theme carries the provider effect')
a = ax[1]
a.errorbar(sdv['mean'], y, xerr=[sdv['mean'] - sdv['lo95'], sdv['hi95'] - sdv['mean']], fmt='o', color=ORANGE, capsize=2)
a.set_yticks(y); a.set_yticklabels(['' for _ in y]); a.invert_yaxis(); a.set_xlabel('σ_v,q (pp): persistent provider × theme SD'); a.set_title('Theme-specific persistent dispersion')
a = ax[2]
a.errorbar(kap.index.astype(int), kap['mean'], yerr=[kap['mean'] - kap['lo95'], kap['hi95'] - kap['mean']], fmt='o-', color=GREEN, capsize=2); a.axhline(1, color=GREY, ls='--', lw=0.8)
a.set_xticks([2023, 2024, 2025, 2026]); a.set_ylabel('κ_t (2023 = 1)'); a.set_title('Spread of persistent provider effects by year')
a = ax[3]
rho = mf['rho']; neff = mf['effective_independent_themes_per_provider_year']
a.bar(['ρ', 'σ_c | M4', 'σ_c | G'], [rho['mean'], mf['sd_c']['mean'], 2.42], color=[BLUE, ORANGE, GREY])
a.set_ylabel('ρ (corr.) / σ_c (pp)')
a.errorbar([0, 1], [rho['mean'], mf['sd_c']['mean']], yerr=[[rho['mean'] - rho['lo95'], mf['sd_c']['mean'] - mf['sd_c']['lo95']], [rho['hi95'] - rho['mean'], mf['sd_c']['hi95'] - mf['sd_c']['mean']]], fmt='none', ecolor='k', capsize=3)
a.set_title(f"ρ = {rho['mean']:.2f}: 7 themes ≈ {neff['mean']:.1f} independent measurements", fontsize=9.5); a.tick_params(axis='x', labelsize=9)
plt.tight_layout(); plt.savefig(FIG + 'B6_M4_structure.png', bbox_inches='tight'); plt.close()

# ---------------- B3 (updated): held-out with all models incl. EB and M4; interval calibration ----------------
pm = fa['point_metrics']; m4 = mh['holdout_point_matched']
fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
a = ax[0]
keys = [('2025 observed', pm['last_matched']['RMSE'], GREY), ('2023–25 mean', pm['mean3_matched']['RMSE'], GREY), ('precision-weighted', pm['pw_matched']['RMSE'], GREY),
        ('EB prototype (per theme)', pm['eb_matched']['RMSE'], GREEN), ('Bayes guide-spec (G)', pm['pred_mean_matched']['RMSE'], PURPLE), ('Bayes M4 (ρ, λ, κ)', m4['pred_mean']['RMSE'], BLUE)]
a.barh([k[0] for k in keys], [k[1] for k in keys], color=[k[2] for k in keys])
for i, k in enumerate(keys): a.text(k[1] + 0.05, i, f'{k[1]:.2f}', va='center', fontsize=9)
a.set_xlabel('RMSE, held-out 2026 benchmark gap (pp)'); a.set_title(f"Point prediction, n = {m4['n']} matched cells"); a.invert_yaxis(); a.set_xlim(0, 7.6)
a = ax[1]; lv = [50, 60, 70, 80, 90, 95]
covs = {'Bayes M4': (mh['holdout_probabilistic_matched']['M4']['coverage'], BLUE), 'Bayes guide-spec (G)': (fa['probabilistic_matched']['bayes']['coverage'], PURPLE),
        'EB prototype N': (fa['probabilistic_matched']['eb_normal']['coverage'], GREEN), 'precision-weighted N': (fa['probabilistic_matched']['pw_normal']['coverage'], ORANGE), 'persistence N(D₂₀₂₅)': (fa['probabilistic_matched']['persistence_normal']['coverage'], GREY)}
for nm, (cv, col) in covs.items(): a.plot(lv, [100 * cv[str(l)] for l in lv], marker='o', color=col, label=nm)
a.plot([45, 100], [45, 100], color='k', ls='--', lw=0.8); a.set_xlabel('Nominal predictive interval (%)'); a.set_ylabel('Observed coverage (%)'); a.set_title('Predictive-interval calibration'); a.legend(fontsize=7)
a = ax[2]
pr = fa['probabilistic_matched']; rows = [('OfS 2025 probs', pr['OfS_last_year_probs']['brier'], ORANGE), ('persistence N', pr['persistence_normal']['brier'], GREY), ('precision-wtd N', pr['pw_normal']['brier'], GREY),
                                          ('EB prototype N', pr['eb_normal']['brier'], GREEN), ('Bayes G', pr['bayes']['brier'], PURPLE), ('Bayes M4', mh['holdout_probabilistic_matched']['M4']['brier'], BLUE), ('base rates', pr['climatology']['brier'], GREY)]
a.barh([r[0] for r in rows], [r[1] for r in rows], color=[r[2] for r in rows])
for i, r in enumerate(rows): a.text(r[1] + 0.005, i, f'{r[1]:.3f}', va='center', fontsize=9)
a.set_xlabel('Multiclass Brier score for the 2026 materiality band'); a.set_title('Probabilistic band forecasts'); a.invert_yaxis(); a.set_xlim(0, 0.75)
plt.tight_layout(); plt.savefig(FIG + 'B3_heldout_final.png', bbox_inches='tight'); plt.close()

# ---------------- B7: QMUL under G vs M4 ----------------
g = pd.read_csv(OUT + 'final_B_unit_theme_2026.csv'); g = g[g.UKPRN == 10007775].sort_values('theme')
m = pd.read_csv(OUT + 'M4_full_unit_theme_2026.csv'); m = m[m.UKPRN == 10007775].sort_values('theme')
fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
a = ax[0]; y = np.arange(7)
a.scatter(g.D_2025, y + 0.22, marker='x', color=ORANGE, label='observed 2025'); a.scatter(g.D_2026, y + 0.22, marker='x', color='k', label='observed 2026')
a.errorbar(g.theta26_mean, y, xerr=[g.theta26_mean - g.theta26_lo95, g.theta26_hi95 - g.theta26_mean], fmt='s', color=PURPLE, capsize=2, label='latent 2026, guide-spec G (95%)')
a.errorbar(m.theta26_mean, y - 0.22, xerr=[m.theta26_mean - m.theta26_lo95, m.theta26_hi95 - m.theta26_mean], fmt='o', color=BLUE, capsize=2, label='latent 2026, M4 (95%)')
a.set_yticks(y); a.set_yticklabels([THEMES[t] for t in g.theme]); a.invert_yaxis(); a.axvline(0, color='k', lw=0.6); a.axvspan(-THR, THR, color=GREY, alpha=0.1); a.set_xlabel('Benchmark gap (pp)'); a.set_title('QMUL CAH10: observed vs latent 2026'); a.legend(fontsize=7, loc='upper left', bbox_to_anchor=(0, -0.12), ncol=2)
a = ax[1]
a.barh(y + 0.2, g.P_improving, height=0.38, color=PURPLE, label='P(materially improving) — G'); a.barh(y - 0.2, m.P_improving, height=0.38, color=BLUE, label='P(materially improving) — M4')
a.barh(y + 0.2, -g.P_below, height=0.38, color=PURPLE, alpha=0.45, label='P(materially below 2026) — G'); a.barh(y - 0.2, -m.P_below, height=0.38, color=BLUE, alpha=0.45, label='P(materially below 2026) — M4')
a.set_yticks(y); a.set_yticklabels(['' for _ in y]); a.invert_yaxis(); a.axvline(0, color='k', lw=0.6); a.set_xlim(-1.05, 1.05); a.set_xlabel('← P(below 2026)        P(improving 2025→26) →'); a.set_title('Same data, two observation models'); a.legend(fontsize=7, loc='lower right')
plt.tight_layout(); plt.savefig(FIG + 'B7_qmul_G_vs_M4.png', bbox_inches='tight'); plt.close()

# ---------------- 2027 forecasts under M4 ----------------
S = dict(np.load(OUT + 'M4_full_samples.npz')); d = pd.read_csv(OUT + 'M4_full_cell_posteriors.csv'); d26 = d[d.Year == 2026].copy()
rng = np.random.default_rng(2027); Sn = S['mu'].shape[0]; N = len(d26)
mp = S['mu'][:, None] + S['alpha'][:, d26.q_i.values] + S['kappa'][:, [3]] * S['lam'][:, d26.q_i.values] * S['u'][:, d26.u_i.values] + S['v'][:, d26.v_i.values]
codes = d26.u_i.astype('category').cat.codes.values
c_new = (S['sd_c'][:, None] * rng.standard_normal((Sn, codes.max() + 1)))[:, codes]; g_new = rng.standard_normal((Sn, codes.max() + 1))[:, codes]; z = rng.standard_normal((Sn, N))
gam = S['sd_gamma'][:, None] * rng.standard_normal((Sn, 1)); rho = S['rho'][:, None]
latent = mp + gam + c_new; obs = latent + d26.s.values[None, :] * (np.sqrt(rho) * g_new + np.sqrt(1 - rho) * z)
d26['pred27_mean'] = obs.mean(0); d26['pred27_lo90'] = np.quantile(obs, .05, 0); d26['pred27_hi90'] = np.quantile(obs, .95, 0); d26['P27_below'] = (obs < -THR).mean(0); d26['P27_above'] = (obs > THR).mean(0)
d26['latent27_mean'] = latent.mean(0); d26['P27_latent_below'] = (latent < -THR).mean(0); d26['P27_latent_above'] = (latent > THR).mean(0)
f = d26[['UKPRN', 'provider', 'theme', 'D', 's', 'n_resp', 'pred27_mean', 'pred27_lo90', 'pred27_hi90', 'P27_below', 'P27_above', 'latent27_mean', 'P27_latent_below', 'P27_latent_above']].rename(columns={'D': 'D_2026', 's': 's_2026', 'n_resp': 'n_2026'})
f.to_csv(OUT + 'forecast_2027_sector_M4.csv', index=False)
q = f[f.UKPRN == 10007775].round(2); q.to_csv(OUT + 'forecast_2027_QMUL_M4.csv', index=False)
print(q[['theme', 'D_2026', 'pred27_mean', 'pred27_lo90', 'pred27_hi90', 'P27_below', 'P27_above', 'latent27_mean']].to_string())
print('mean 90% width', (f.pred27_hi90 - f.pred27_lo90).mean().round(1), 'share P27_below>0.8', (f.P27_below > 0.8).mean().round(3))
