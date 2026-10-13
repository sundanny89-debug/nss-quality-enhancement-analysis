"""Step 2: Reproduce frozen Stage-1 sector findings (descriptive churn, baselines, RTM, volatility)."""
import pandas as pd, numpy as np, json
prov = pd.read_csv('../data/sector_cah10_provider_theme.csv')
# unit key = UKPRN (frozen rule 3). 19 providers changed name within 2023-2026; use latest name as canonical label.
canon = prov.sort_values('Year').groupby('UKPRN')['provider'].last()
prov['provider'] = prov['UKPRN'].map(canon)
prov.to_csv('../data/sector_cah10_provider_theme.csv', index=False)
THR = 2.5
def band(x):
    z = np.round(x, 1)  # freeze strict 1-decimal materiality convention
    return np.where(z > THR, 'above', np.where(z < -THR, 'below', 'inline'))
def traj(x):
    z = np.round(x, 1)  # annual change also rounded to 1 decimal
    return np.where(z > THR, 'improving', np.where(z < -THR, 'declining', 'stable'))
w = prov.pivot_table(index=['UKPRN','provider','theme'], columns='Year', values=['D','R','s','headcount','n_resp','contrib','P','B','rr'])
res = {}
both = w[('D',2025)].notna() & w[('D',2026)].notna()
c = w[both].copy()
for m in ['D','R']:
    c[(m,'pos')] = band(c[(m,2026)]); c[(m,'trj')] = traj(c[(m,2026)]-c[(m,2025)])
    c[(m,'state')] = c[(m,'pos')].astype(str)+'/'+c[(m,'trj')].astype(str)
res['n_cells_2025_2026'] = int(both.sum())
res['churn_raw_vs_bench_full'] = float((c[('D','state')]!=c[('R','state')]).mean())
res['churn_raw_vs_bench_pos'] = float((c[('D','pos')]!=c[('R','pos')]).mean())
res['churn_raw_vs_bench_trj'] = float((c[('D','trj')]!=c[('R','trj')]).mean())
# transition matrix (position)
tm = pd.crosstab(c[('R','pos')], c[('D','pos')]); tm.index.name='raw UK comparator'; tm.columns.name='OfS benchmark'
tm.to_csv('../out/T_transition_position_raw_vs_bench.csv')
tm2 = pd.crosstab(c[('R','trj')], c[('D','trj')]); tm2.to_csv('../out/T_transition_trajectory_raw_vs_bench.csv')
# churn by size band
c['band'] = pd.cut(c[('n_resp',2026)], [0,50,100,200,np.inf], labels=['<50','50-99','100-199','200+'])
cb = c.groupby('band', observed=True).apply(lambda g: pd.Series({'n':len(g),'full_churn':(g[('D','state')]!=g[('R','state')]).mean()}))
cb.to_csv('../out/T_churn_raw_vs_bench_by_size.csv'); print(cb)
# baseline predictive validation for 2026
tr = w[('D',2026)].notna(); d = w[tr]
pred = pd.DataFrame(index=d.index)
pred['obs'] = d[('D',2026)]; pred['last'] = d[('D',2025)]
pred['mean3'] = d[[('D',2023),('D',2024),('D',2025)]].mean(axis=1)
vals = d[[('D',2023),('D',2024),('D',2025)]].values; wts = 1/d[[('s',2023),('s',2024),('s',2025)]].values**2
pred['pw'] = np.nansum(np.where(np.isnan(vals),0,vals*wts),axis=1)/np.nansum(np.where(np.isnan(vals),0,wts),axis=1)
pred['n_train'] = (~np.isnan(vals)).sum(axis=1)
pred.to_csv('../out/pred_baselines_2026.csv')
bl = {}
for m in ['last','mean3','pw']:
    ok = pred[m].notna(); e = pred.loc[ok,m]-pred.loc[ok,'obs']
    bl[m] = {'n':int(ok.sum()),'RMSE':float(np.sqrt((e**2).mean())),'MAE':float(e.abs().mean())}
res['baselines_2026'] = bl
# matched set (cells with 2025 obs) for fair comparison
ok = pred['last'].notna()
for m in ['last','mean3','pw']:
    e = pred.loc[ok,m]-pred.loc[ok,'obs']
    bl[m+'_matched'] = {'n':int(ok.sum()),'RMSE':float(np.sqrt((e**2).mean())),'MAE':float(e.abs().mean())}
# RTM
d25 = w.loc[both,('D',2025)]; d26 = w.loc[both,('D',2026)]
res['rtm'] = {'corr_D25_dD':float(np.corrcoef(d25,d26-d25)[0,1]),'corr_D25_D26':float(np.corrcoef(d25,d26)[0,1]),'slope_D26_on_D25':float(np.polyfit(d25,d26,1)[0])}
# volatility by size band
long = prov.sort_values(['UKPRN','theme','Year']).copy()
long['Dprev'] = long.groupby(['UKPRN','theme'])['D'].shift(1); long['Yprev'] = long.groupby(['UKPRN','theme'])['Year'].shift(1)
adj = long[long['Year']-long['Yprev']==1].copy(); adj['absdD']=(adj['D']-adj['Dprev']).abs()
adj['band'] = pd.cut(adj['n_resp'], [0,50,100,200,np.inf], labels=['<50','50-99','100-199','200+'])
vb = adj.groupby('band', observed=True)['absdD'].agg(['median','mean','count']); vb.to_csv('../out/T_volatility_by_size.csv')
res['volatility_median_absdD_by_band'] = vb['median'].to_dict()
# Sector-level ΔP = ΔB + ΔD (2025->2026)
c['dP'] = c[('P',2026)]-c[('P',2025)]; c['dB'] = c[('B',2026)]-c[('B',2025)]; c['dD'] = c[('D',2026)]-c[('D',2025)]
res['decomp_2025_2026'] = {'median_dP':float(c['dP'].median()),'median_dB':float(c['dB'].median()),'median_dD':float(c['dD'].median()),
    'mean_dP':float(c['dP'].mean()),'mean_dB':float(c['dB'].mean()),'mean_dD':float(c['dD'].mean()),
    'share_cells_dP_pos':float((c['dP']>0).mean()),'share_cells_dD_pos':float((c['dD']>0).mean()),
    'share_dP_positive_but_dD_negative':float(((c['dP']>0)&(c['dD']<0)).mean()),
    'share_dP_positive_but_dD_below_-2.5':float(((c['dP']>0)&(c['dD']<-2.5)).mean()),
    'share_dP_gt2.5_but_dD_le0':float(((c['dP']>2.5)&(c['dD']<=0)).mean())}
# ΔB by theme
res['median_dB_by_theme'] = c.groupby(level='theme')['dB'].median().to_dict()
json.dump(res, open('../out/stage1_reproduction.json','w'), indent=2, default=str)
print(json.dumps(res, indent=2, default=str))
