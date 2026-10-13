"""Step 1: Harmonise public OfS extracts into analytical datasets.
Sector: CAH10 Engineering & technology, All undergraduates, provider x theme x year (2023-2026)
SEMS:   QMUL CAH3 subjects x theme/question x year, All undergraduates (primary level per plan)
"""
import pandas as pd, numpy as np
UP = '../data/raw_public_extracts/'
ren = {'Positivity measure (%)':'P','Benchmark (%)':'B','Difference (ppt)':'D','Standard deviation':'s',
       'Responses':'n_resp','Population':'pop','Publication response headcount':'headcount',
       'Publication response rate (%)':'rr','Contribution to benchmark (%)':'contrib',
       'Materially below benchmark (%)':'p_below','Broadly in line with benchmark (%)':'p_inline',
       'Materially above benchmark (%)':'p_above'}

# ---------- Sector ----------
raw = pd.read_csv(UP+'NSS2023_2026_CAH10_all_providers_themes.csv')
raw['Question'] = raw['Question'].str.strip()
raw['theme'] = raw['Question'].str.extract(r'Theme (\d)').astype(int)
uk = raw[raw['UKPRN']=='UK'][['Year','theme','Positivity measure (%)']].rename(columns={'Positivity measure (%)':'P_uk'})
prov = raw[raw['Benchmark (%)'].notna()].copy().rename(columns=ren)
prov = prov.merge(uk, on=['Year','theme'], how='left')
prov['R'] = prov['P'] - prov['P_uk']            # raw unadjusted comparator gap (frozen rule 4)
prov['provider'] = prov['Provider name']
keep = ['Year','UKPRN','provider','theme','Question','P','B','D','s','R','P_uk','n_resp','pop','headcount','rr','contrib','p_below','p_inline','p_above']
prov = prov[keep].sort_values(['UKPRN','theme','Year']).reset_index(drop=True)
assert (prov['D']-(prov['P']-prov['B'])).abs().max() < 0.11
prov.to_csv('../data/sector_cah10_provider_theme.csv', index=False)
print('sector rows', prov.shape, 'providers', prov.UKPRN.nunique())

# ---------- SEMS ----------
s = pd.read_csv(UP+'NSS2023_2026_QMUL_SEMS_public_extract_all_levels.csv')
s['Question'] = s['Question'].str.strip()
s = s.rename(columns=ren)
s['is_theme'] = s['Question'].str.startswith('Theme')
s['theme'] = np.where(s['is_theme'], s['Question'].str.extract(r'Theme (\d)')[0], np.nan)
s['qnum'] = np.where(~s['is_theme'], s['Question'].str.extract(r'Q(\d\d)')[0], np.nan)
# question -> theme map (post-2023 NSS structure; Q26 & Q27 not part of the 7 themes)
q2t = {**{f'{i:02d}':1 for i in range(1,5)}, **{f'{i:02d}':2 for i in range(5,10)}, **{f'{i:02d}':3 for i in range(10,15)},
       '15':4,'16':4,'17':5,'18':5,'19':6,'20':6,'21':6,'22':7,'23':7,'24':7,'25':7}
s['q_theme'] = s['qnum'].map(q2t)
short = {'Engineering and technology':'CAH10 (all E&T)','Engineering (non-specific)':'Eng non-specific',
         'Mechanical engineering':'Mechanical','Production and manufacturing engineering':'Prod & manuf (Robotics proxy)',
         'Aeronautical and aerospace engineering':'Aerospace','Bioengineering, medical and biomedical engineering':'Biomedical',
         'Chemical, process and energy engineering':'Chemical','Materials technology':'Materials technology',
         'Materials science':'Materials science','Design studies':'Design studies (DICE proxy)'}
s['unit'] = s['Subject'].map(short)
s.to_csv('../data/sems_all_levels_long.csv', index=False)
# primary level & duplication check
au = s[s['Level of study']=='All undergraduates'].copy()
ms = au[(au.unit=='Materials science')&au.is_theme].set_index(['Year','theme'])[['P','B','D','s','headcount']]
mt = au[(au.unit=='Materials technology')&au.is_theme].set_index(['Year','theme'])[['P','B','D','s','headcount']]
j = ms.join(mt, lsuffix='_sci', rsuffix='_tech', how='inner')
print('Materials sci vs tech identical rows (2025-26 overlap):', (j['P_sci']==j['P_tech']).mean(), (j['B_sci']==j['B_tech']).mean(), len(j))
au.to_csv('../data/sems_all_ug_long.csv', index=False)
print('SEMS AU rows', au.shape)
