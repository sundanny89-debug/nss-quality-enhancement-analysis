"""Step 6: Figures (matplotlib) for the sector stress-test and SEMS case study."""
import pandas as pd, numpy as np, json, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt, matplotlib.ticker as mticker
from scipy import stats
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({'figure.dpi': 150, 'font.size': 10, 'axes.titlesize': 11, 'axes.titleweight': 'bold', 'axes.labelsize': 10, 'legend.fontsize': 9})
BLUE, ORANGE, GREEN, RED, PURPLE, GREY = '#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B3', '#8c8c8c'
THR = 2.5
OUT = '../out/'; FIG = '../fig/'
THEMES = {1: 'Teaching', 2: 'Learning opps', 3: 'Assessment & feedback', 4: 'Academic support', 5: 'Organisation & mgmt', 6: 'Learning resources', 7: 'Student voice'}
prov = pd.read_csv('../data/sector_cah10_provider_theme.csv')
w = prov.pivot_table(index=['UKPRN', 'theme'], columns='Year', values=['D', 'R', 'P', 'B', 's', 'n_resp'])
both = w[('D', 2025)].notna() & w[('D', 2026)].notna(); c = w[both].copy()
c['dP'] = c[('P', 2026)] - c[('P', 2025)]; c['dB'] = c[('B', 2026)] - c[('B', 2025)]; c['dD'] = c[('D', 2026)] - c[('D', 2025)]

# ---------------- F1: decomposition ----------------
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
parts = ax[0].violinplot([c['dP'], c['dB'], c['dD']], showmedians=True, widths=0.8)
for pc, col in zip(parts['bodies'], [GREY, ORANGE, BLUE]): pc.set_facecolor(col); pc.set_alpha(0.6)
ax[0].set_xticks([1, 2, 3]); ax[0].set_xticklabels(['ΔP (raw positivity)', 'ΔB (benchmark)', 'ΔD (benchmark gap)'])
ax[0].axhline(0, color='k', lw=0.8); ax[0].set_ylabel('Change 2025→2026 (pp)')
ax[0].set_title(f'ΔP = ΔB + ΔD across {len(c)} provider × theme cells')
ax[0].text(0.02, 0.97, f"medians: ΔP {c['dP'].median():+.1f}, ΔB {c['dB'].median():+.1f}, ΔD {c['dD'].median():+.1f}", transform=ax[0].transAxes, va='top', fontsize=9)
sc = ax[1].scatter(c['dP'], c['dD'], c=c['dB'], cmap='Oranges', s=12, alpha=0.8, edgecolor='none')
lim = max(abs(c['dP']).max(), abs(c['dD']).max()) * 1.05
ax[1].plot([-lim, lim], [-lim, lim], color=GREY, lw=0.8, ls='--', label='ΔD = ΔP (no benchmark movement)')
ax[1].axhline(0, color='k', lw=0.6); ax[1].axvline(0, color='k', lw=0.6)
ax[1].axhspan(-THR, THR, color=GREY, alpha=0.12)
ax[1].set_xlabel('ΔP: change in raw positivity (pp)'); ax[1].set_ylabel('ΔD: change in benchmark gap (pp)')
share = ((c['dP'] > 0) & (c['dD'] < 0)).mean()
ax[1].set_title(f'{share:.0%} of cells rose in raw positivity yet fell relative to benchmark')
cb = plt.colorbar(sc, ax=ax[1]); cb.set_label('ΔB (pp)'); ax[1].legend(loc='lower right')
plt.tight_layout(); plt.savefig(FIG + 'F1_decomposition.png', bbox_inches='tight'); plt.close()

# ---------------- F2: churn ----------------
t1 = pd.read_csv(OUT + 'T_transition_position_raw_vs_bench.csv', index_col=0)
t2 = pd.read_csv(OUT + 'T_transition_position_obs_vs_bayes.csv', index_col=0)
cb1 = pd.read_csv(OUT + 'T_churn_raw_vs_bench_by_size.csv', index_col=0); cb2 = pd.read_csv(OUT + 'T_churn_obs_vs_bayes_by_size.csv', index_col=0)
fig, ax = plt.subplots(1, 3, figsize=(13, 4))
order = ['below', 'inline', 'above']
for a, t, ttl, xl, yl in [(ax[0], t1, 'Position: raw UK comparator → OfS benchmark', 'OfS benchmark band', 'Raw UK comparator band'),
                          (ax[1], t2, 'Position: observed benchmark gap → posterior modal', 'Posterior modal band', 'Observed band')]:
    t = t.reindex(index=order, columns=order).fillna(0)
    im = a.imshow(t.values, cmap='Blues'); a.set_xticks(range(3)); a.set_yticks(range(3)); a.set_xticklabels(order); a.set_yticklabels(order)
    for i in range(3):
        for j in range(3):
            a.text(j, i, int(t.values[i, j]), ha='center', va='center', color='white' if t.values[i, j] > t.values.max() / 2 else 'black', fontsize=10)
    a.set_title(ttl, fontsize=10); a.set_xlabel(xl); a.set_ylabel(yl); a.grid(False)
x = np.arange(len(cb1)); wdt = 0.38
ax[2].bar(x - wdt / 2, cb1['full_churn'], wdt, color=ORANGE, label='raw → OfS benchmark')
ax[2].bar(x + wdt / 2, cb2['full_churn'], wdt, color=BLUE, label='observed → posterior (3×3 modal)')
ax[2].set_xticks(x); ax[2].set_xticklabels(cb1.index); ax[2].set_xlabel('Responses in 2026'); ax[2].set_ylabel('Share of cells whose 3×3 state changes')
ax[2].yaxis.set_major_formatter(mticker.PercentFormatter(1.0)); ax[2].set_title('Decision churn by unit size'); ax[2].legend()
plt.tight_layout(); plt.savefig(FIG + 'F2_churn.png', bbox_inches='tight'); plt.close()

# ---------------- F3: shrinkage ----------------
ut = pd.read_csv(OUT + 'full_M1_unit_theme_2026.csv')
fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
sz = np.clip(ut['n_2026'], 10, 600) / 6
a = ax[0]
a.scatter(ut['D_2026'], ut['theta26_mean'], s=sz, alpha=0.5, c=np.log10(ut['n_2026']), cmap='viridis', edgecolor='none')
lim = 30; a.plot([-lim, lim], [-lim, lim], color=GREY, ls='--', lw=0.8)
for v in [-THR, THR]: a.axhline(v, color=RED, lw=0.6, ls=':'); a.axvline(v, color=RED, lw=0.6, ls=':')
a.set_xlim(-lim, lim); a.set_ylim(-lim, lim); a.set_xlabel('Observed 2026 benchmark gap D (pp)'); a.set_ylabel('Posterior mean latent gap θ₂₀₂₆ (pp)')
a.set_title('Partial pooling: small units are shrunk most (colour = log₁₀ responses)')
a = ax[1]
a.scatter(ut['dD_obs'], ut['dtheta_mean'], s=sz, alpha=0.5, c=np.log10(ut['n_2026']), cmap='viridis', edgecolor='none')
lim = 30; a.plot([-lim, lim], [-lim, lim], color=GREY, ls='--', lw=0.8)
for v in [-THR, THR]: a.axhline(v, color=RED, lw=0.6, ls=':'); a.axvline(v, color=RED, lw=0.6, ls=':')
a.set_xlim(-lim, lim); a.set_ylim(-lim, lim); a.set_xlabel('Observed change in gap ΔD 2025→26 (pp)'); a.set_ylabel('Posterior mean latent change Δθ (pp)')
a.set_title('Observed vs latent year-on-year change')
plt.tight_layout(); plt.savefig(FIG + 'F3_shrinkage.png', bbox_inches='tight'); plt.close()

# ---------------- F4: predictive validation ----------------
ho = json.load(open(OUT + 'holdout_main_summary.json'))
met = ho['metrics']['matched_last']
labels = [('last', '2025 observed'), ('mean3', '2023–25 mean'), ('pw', 'precision-weighted'), ('pred_M0_nocohort', 'Bayes M0 (no cohort)'), ('pred_M1_cohort', 'Bayes M1 (cohort)'), ('pred_M2_cohort_trend', 'Bayes M2 (cohort+trend)')]
labels = [l for l in labels if l[0] in met]
fig, ax = plt.subplots(1, 3, figsize=(13.5, 4.2))
a = ax[0]
vals = [met[k]['RMSE'] for k, _ in labels]; cols = [GREY, GREY, GREY, PURPLE, BLUE, PURPLE][:len(labels)]
a.barh([l for _, l in labels], vals, color=cols)
for i, v_ in enumerate(vals): a.text(v_ + 0.05, i, f'{v_:.2f}', va='center', fontsize=9)
a.set_xlabel('RMSE of 2026 benchmark-gap prediction (pp)'); a.set_title(f"Held-out 2026, n={met['last']['n']} cells"); a.invert_yaxis(); a.set_xlim(0, max(vals) * 1.15)
cal = pd.read_csv(OUT + 'holdout_main_M1_calibration.csv'); calo = pd.read_csv(OUT + 'holdout_main_OfS_calibration.csv')
a = ax[1]
for df_, col, lab in [(cal, BLUE, 'Bayes M1 posterior predictive'), (calo, ORANGE, 'OfS 2025 published probabilities')]:
    for tgt, mk in [('below', 'o'), ('above', 's')]:
        g = df_[df_.target == tgt]
        a.plot(g.mean_pred, g.obs_freq, marker=mk, color=col, lw=1, ms=5, alpha=0.9, label=f'{lab} — P({tgt})')
a.plot([0, 1], [0, 1], color=GREY, ls='--', lw=0.8); a.set_xlabel('Forecast probability of 2026 band'); a.set_ylabel('Observed frequency'); a.set_title('Calibration of 2026 band forecasts'); a.legend(fontsize=7)
a = ax[2]
pm = ho['metrics']['probabilistic_matched']
keys = [('OfS_lastyear_probs', 'OfS 2025 probs'), ('persistence_normal', 'persistence N(D₂₀₂₅)'), ('pw_normal', 'precision-wtd N'), ('climatology', 'base rates'), ('M0_nocohort', 'Bayes M0'), ('M1_cohort', 'Bayes M1'), ('M2_cohort_trend', 'Bayes M2')]
keys = [k for k in keys if k[0] in pm]
a.barh([l for _, l in keys], [pm[k]['brier'] for k, _ in keys], color=[ORANGE, GREY, GREY, GREY, PURPLE, BLUE, PURPLE][:len(keys)])
for i, (k, _) in enumerate(keys): a.text(pm[k]['brier'] + 0.005, i, f"{pm[k]['brier']:.3f}", va='center', fontsize=9)
a.set_xlabel('Multiclass Brier score (lower = better)'); a.set_title('Probabilistic forecast of 2026 materiality band'); a.invert_yaxis()
plt.tight_layout(); plt.savefig(FIG + 'F4_predictive_validation.png', bbox_inches='tight'); plt.close()

# ---------------- F5: variance decomposition & reliability ----------------
vd = pd.read_csv(OUT + 'full_variance_decomposition.csv', index_col=0)
rel = pd.read_csv(OUT + 'full_reliability_curve.csv')
fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.2))
a = ax[0]
comps = ['unit(persistent, all themes)', 'unit x theme (persistent)', 'unit x year cohort shock', 'idiosyncratic theme-year']
labs = ['provider (persistent)', 'provider × theme (persistent)', 'cohort shock (provider × year)', 'idiosyncratic']
colsc = [BLUE, '#87a7d6', ORANGE, GREY]
bottom = 0
for comp, lab, col in zip(comps, labs, colsc):
    a.bar(['latent (between-unit)'], [vd.loc[comp, 'mean']], bottom=bottom, color=col, label=f"{lab}: {vd.loc[comp,'mean']:.1f}"); bottom += vd.loc[comp, 'mean']
for n, col in [(30, '#c44e52'), (100, '#d98a8c'), (300, '#efc4c5')]:
    s2 = (100 * np.sqrt(0.16 / n)) ** 2
    a.bar([f'sampling\nn={n}'], [s2], color=col, label=f'sampling variance, n={n}: {s2:.1f}')
a.set_ylabel('Variance (pp²)'); a.set_title('Where does variation in benchmark gaps come from?'); a.legend(fontsize=7.5, loc='upper right')
a = ax[1]
a.plot(rel.n, rel.rel_current, marker='o', color=BLUE, label='reliability for the current-year true gap')
a.plot(rel.n, rel.rel_persistent, marker='s', color=ORANGE, label='reliability for the persistent gap')
a.fill_between(rel.n, rel.rel_persistent_lo, rel.rel_persistent_hi, color=ORANGE, alpha=0.15)
a.plot(rel.n, rel.rel_theme_specific, marker='^', color=GREEN, label='reliability for a persistent theme-specific effect')
a.set_xscale('log'); a.set_xlabel('Responses in the unit (log scale)'); a.set_ylabel('Reliability = signal / (signal + noise)')
a.axhline(0.5, color=GREY, ls=':', lw=0.8); a.set_ylim(0, 1); a.set_title('How much can a single-year theme gap tell you?'); a.legend(fontsize=7.5)
plt.tight_layout(); plt.savefig(FIG + 'F5_variance_reliability.png', bbox_inches='tight'); plt.close()

# ---------------- F6: funnel plot 2026 with posterior colouring ----------------
d26 = ut.copy()
fig, ax = plt.subplots(figsize=(8, 5))
n = np.linspace(8, d26['n_2026'].max() * 1.05, 400); s = 100 * np.sqrt(0.16 / n)
for k, ls, lab in [(1.96, '--', '95% (binomial)'), (3.09, ':', '99.8% (binomial)')]:
    ax.plot(n, k * s, color=GREY, ls=ls, lw=0.9, label=lab); ax.plot(n, -k * s, color=GREY, ls=ls, lw=0.9)
pm = d26['P_below'] - d26['P_above']
sc = ax.scatter(d26['n_2026'], d26['D_2026'], c=pm, cmap='RdBu_r', vmin=-1, vmax=1, s=18, alpha=0.9, edgecolor='k', linewidth=0.2)
cb = plt.colorbar(sc); cb.set_label('P(materially below) − P(materially above), posterior')
ax.axhline(0, color='k', lw=0.7); ax.axhspan(-THR, THR, color=GREY, alpha=0.1)
ax.set_xscale('log'); ax.set_xlabel('Responses (log scale)'); ax.set_ylabel('Observed 2026 benchmark gap (pp)')
ax.set_title('Funnel plot, 2026: sampling limits vs posterior material deviation'); ax.legend(loc='upper right', fontsize=8)
plt.tight_layout(); plt.savefig(FIG + 'F6_funnel_2026.png', bbox_inches='tight'); plt.close()

# ---------------- F7: regression to the mean ----------------
fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
a = ax[0]
a.scatter(ut['D_2025'], ut['D_2026'], s=10, alpha=0.5, color=GREY, label='observed')
sl, ic = np.polyfit(ut['D_2025'].dropna(), ut.loc[ut['D_2025'].notna(), 'D_2026'], 1)
xx = np.linspace(-30, 30, 10); a.plot(xx, sl * xx + ic, color=RED, lw=1.5, label=f'observed slope {sl:.2f}')
a.plot(xx, xx, color='k', lw=0.8, ls='--', label='no reversion (slope 1)')
a.set_xlim(-30, 30); a.set_ylim(-30, 30); a.set_xlabel('2025 observed gap (pp)'); a.set_ylabel('2026 observed gap (pp)'); a.set_title('Regression to the mean, observed'); a.legend(fontsize=8)
a = ax[1]
fs = json.load(open(OUT + 'full_summary.json'))
a.scatter(ut['D_2025'], ut['dD_obs'], s=10, alpha=0.4, color=GREY, label=f"observed (r = {fs['rtm_observed']['corr_D25_dD']:.2f})")
a.scatter(ut['theta26_mean'] - ut['dtheta_mean'], ut['dtheta_mean'], s=10, alpha=0.6, color=BLUE, label=f"latent posterior means (r = {fs['rtm_latent']['corr_theta25_dtheta']:.2f})")
a.axhline(0, color='k', lw=0.6); a.set_xlabel('2025 gap (pp)'); a.set_ylabel('Change 2025→2026 (pp)'); a.set_title('Prior extreme → subsequent change'); a.legend(fontsize=8)
plt.tight_layout(); plt.savefig(FIG + 'F7_regression_to_mean.png', bbox_inches='tight'); plt.close()

# ---------------- F8: SEMS posterior 3x3 quadrant with uncertainty ----------------
su = pd.read_csv(OUT + 'sems_theme_M1_unit_theme_2026.csv')
units = sorted(su.unit.unique()); cmap = plt.get_cmap('tab10')
fig, ax = plt.subplots(1, 2, figsize=(13, 5.5))
a = ax[0]
for i, u_ in enumerate(units):
    g = su[su.unit == u_]
    a.errorbar(g.theta26_mean, g.dtheta_mean, xerr=[g.theta26_mean - g.theta26_lo90, g.theta26_hi90 - g.theta26_mean], yerr=[g.dtheta_mean - g.dtheta_lo90, g.dtheta_hi90 - g.dtheta_mean],
               fmt='o', color=cmap(i), ms=4, lw=0.6, alpha=0.8, label=u_, capsize=0)
    for _, r in g.iterrows(): a.annotate(str(int(r.theme)), (r.theta26_mean, r.dtheta_mean), fontsize=6, xytext=(2, 2), textcoords='offset points')
a.axhline(0, color='k', lw=0.6); a.axvline(0, color='k', lw=0.6)
for v in [-THR, THR]: a.axhline(v, color=RED, lw=0.6, ls=':'); a.axvline(v, color=RED, lw=0.6, ls=':')
a.set_xlabel('Latent 2026 benchmark gap θ (pp, 90% interval)'); a.set_ylabel('Latent change 2025→2026 Δθ (pp, 90% interval)')
a.set_title('SEMS units × themes: position × trajectory with uncertainty'); a.legend(fontsize=7, loc='lower right', ncol=2)
a = ax[1]
piv_obs = su.pivot(index='unit', columns='theme', values='D_2026'); piv_p = su.pivot(index='unit', columns='theme', values='P_below') - su.pivot(index='unit', columns='theme', values='P_above')
im = a.imshow(piv_p.values, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
a.set_xticks(range(7)); a.set_xticklabels([THEMES[t] for t in piv_p.columns], rotation=35, ha='right', fontsize=8); a.set_yticks(range(len(piv_p))); a.set_yticklabels(piv_p.index, fontsize=8)
for i in range(piv_p.shape[0]):
    for j in range(piv_p.shape[1]):
        a.text(j, i, f'{piv_obs.values[i, j]:+.0f}\n({piv_p.values[i, j]:+.2f})', ha='center', va='center', fontsize=6.5, color='black')
a.set_title('Observed 2026 gap (top) and P(below)−P(above) (bracket)'); a.grid(False)
cb = plt.colorbar(im, ax=a, fraction=0.03); cb.set_label('P(materially below) − P(materially above)')
plt.tight_layout(); plt.savefig(FIG + 'F8_sems_quadrant.png', bbox_inches='tight'); plt.close()

# ---------------- F9: SEMS estimand comparison ----------------
est = pd.read_csv(OUT + 'sems_estimand_comparison_2026.csv')
fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.5))
a = ax[0]; y = np.arange(7)
a.scatter(est.typical_student_D, y + 0.15, color=ORANGE, marker='s', label='typical student (response-weighted mean of unit gaps)')
a.scatter(est.typical_unit_D, y, color=GREEN, marker='^', label='typical unit (equal-weight mean)')
a.errorbar(est.latent_equal_mean, y - 0.15, xerr=[est.latent_equal_mean - est.latent_equal_lo90, est.latent_equal_hi90 - est.latent_equal_mean], fmt='o', color=BLUE, label='reliability-adjusted (latent equal-weight, 90%)', capsize=2)
a.scatter(est.QMUL_CAH10_provider_D_2026, y - 0.3, color=GREY, marker='D', label='QMUL CAH10 provider row (all E&T incl. non-SEMS)')
a.set_yticks(y); a.set_yticklabels([THEMES[t] for t in est.theme]); a.axvline(0, color='k', lw=0.6); a.axvspan(-THR, THR, color=GREY, alpha=0.1)
a.set_xlabel('2026 benchmark gap (pp)'); a.set_title('Same data, three estimands: 2026 position'); a.legend(fontsize=7, loc='lower right'); a.invert_yaxis()
a = ax[1]
a.scatter(est.typical_student_dD, y + 0.15, color=ORANGE, marker='s', label='typical student ΔD')
a.scatter(est.typical_unit_dD, y, color=GREEN, marker='^', label='typical unit ΔD')
a.errorbar(est.latent_equal_dD, y - 0.15, xerr=[est.latent_equal_dD - est.latent_equal_dD_lo90, est.latent_equal_dD_hi90 - est.latent_equal_dD], fmt='o', color=BLUE, label='latent Δθ (equal-weight, 90%)', capsize=2)
a.scatter(est.QMUL_CAH10_provider_dD, y - 0.3, color=GREY, marker='D', label='QMUL CAH10 provider row ΔD')
a.set_yticks(y); a.set_yticklabels(['' for _ in est.theme]); a.axvline(0, color='k', lw=0.6); a.axvspan(-THR, THR, color=GREY, alpha=0.1)
a.set_xlabel('Change 2025→2026 (pp)'); a.set_title('Trajectory under each estimand'); a.legend(fontsize=7, loc='lower right'); a.invert_yaxis()
plt.tight_layout(); plt.savefig(FIG + 'F9_sems_estimands.png', bbox_inches='tight'); plt.close()

# ---------------- F10: QMUL within sector ----------------
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
q = fs['QMUL']; qt = pd.DataFrame(q['table'])
a = ax[0]
a.hist(ut.groupby('provider')['persist_mean'].mean(), bins=30, color=GREY, alpha=0.7)
a.axvline(q['u_mean'], color=RED, lw=2, label=f"QMUL persistent effect ({q['u_mean']:+.1f} pp, {q['u_percentile']:.0%}ile)")
a.set_xlabel('Provider persistent benchmark gap (posterior mean, averaged over themes)'); a.set_ylabel('Providers'); a.set_title('QMUL CAH10 within the sector: persistent standing'); a.legend(fontsize=8)
a = ax[1]
a.errorbar(qt.theta26_mean, np.arange(7), xerr=[qt.theta26_mean - qt.theta26_lo90, qt.theta26_hi90 - qt.theta26_mean], fmt='o', color=BLUE, capsize=2, label='latent 2026 gap (90%)')
a.scatter(qt.D_2026, np.arange(7), color=GREY, marker='x', label='observed 2026 gap')
a.scatter(qt.D_2025, np.arange(7), color=ORANGE, marker='x', label='observed 2025 gap')
a.set_yticks(range(7)); a.set_yticklabels([THEMES[t] for t in qt.theme]); a.axvline(0, color='k', lw=0.6); a.axvspan(-THR, THR, color=GREY, alpha=0.1)
a.set_xlabel('Benchmark gap (pp)'); a.set_title('QMUL CAH10 (all E&T): 2025 → 2026'); a.legend(fontsize=8); a.invert_yaxis()
plt.tight_layout(); plt.savefig(FIG + 'F10_qmul_in_sector.png', bbox_inches='tight'); plt.close()
print('figures done')
