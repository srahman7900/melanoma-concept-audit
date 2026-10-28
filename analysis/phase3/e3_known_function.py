"""E3. Controls with prediction functions known by construction (post hoc, offline).
Each function maps an image to a score through a fixed evaluator output, standardised on the 969 real test images:
f(x) = (z(x) - mean_test) / SD_test, so its 'logit SD' on the test images is 1. The audit then measures each knob's effect
on f exactly as for a classifier (mean change from level 0 to 1, bootstrap CI, Wilcoxon slope test, BH within 7 knobs).
Functions: CBM-B output of each concept (7), ISIC 2018 detector pigment network, lesion darkness, global darkness, mean b*.
For a single-concept function the audit should give the largest effect to that concept's knob, but only where the knob
moves that concept more than the other knobs do: this test therefore measures the knobs, not a classifier."""
import os, sys
import numpy as np, pandas as pd
from scipy import stats
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from load_local import *
OUT = os.environ.get('A3_OUT', os.path.join(ROOT, '..', 'results', 'phase3_rebuilt')); RNG = np.random.default_rng(2035)
P = os.path.join(ROOT, 'p2ainp', 'phase2', 'scores')


def ev(s):
    A, B = np.load(f'{P}/A_{s}.npz'), np.load(f'{P}/B_{s}.npz')
    F = {f'CBM-B {SHORT[k]}': B['cbmb'][:, k] for k in range(7)}
    F['ISIC 2018 PN'] = B['isic18'][:, 0]
    F['Lesion darkness'] = A['dark'][:, 0]; F['Global darkness'] = A['dark'][:, 3]; F['Mean b*'] = A['lab'][:, 2]
    return {k: v.astype(np.float64) for k, v in F.items()}


T = ev('cls_test'); mu = {k: np.nanmean(v) for k, v in T.items()}; sd = {k: np.nanstd(v) for k, v in T.items()}
rows = []
for s in SETS:
    m = meta(s); E = ev(s)
    for fn, v in E.items():
        z = (v - mu[fn]) / sd[fn]; fam = []
        for j in range(7):
            Ti, lev, _ = traj_index(m, j); Y = z[Ti]; ok = np.isfinite(Y).all(1); Y = Y[ok]
            d = Y[:, -1] - Y[:, 0]; x = lev - lev.mean(); sl = ((Y - Y.mean(1, keepdims=True)) * x).sum(1) / (x ** 2).sum()
            b = d[RNG.integers(0, len(d), (2000, len(d)))].mean(1)
            nz = sl[sl != 0]; p = stats.wilcoxon(nz).pvalue if len(nz) >= 5 else np.nan
            fam.append({'function': fn, 'base': s, 'knob': SHORT[j], 'effect_SD': d.mean(), 'lo': np.percentile(b, 2.5),
                        'hi': np.percentile(b, 97.5), 'p': p, 'median_slope': np.median(sl), 'n': int(ok.sum())})
        p = np.array([f['p'] for f in fam]); o = np.argsort(p); n = len(p); q = p[o] * n / (np.arange(n) + 1)
        q = np.minimum.accumulate(q[::-1])[::-1]; pb = np.empty(n); pb[o] = np.minimum(q, 1)
        for f, qq in zip(fam, pb): f['p_bh'] = qq
        rows += fam
df = pd.DataFrame(rows); df.to_csv(os.path.join(OUT, 'E3_known_function_all.csv'), index=False)
TGT = {f'CBM-B {s}': s for s in SHORT}; TGT['ISIC 2018 PN'] = 'PN'
summ = []
for fn, g in df.groupby('function', sort=False):
    r = {'function': fn}
    piv = g.pivot(index='knob', columns='base', values='effect_SD')
    for s in SETS:
        x = g[g.base == s].set_index('knob'); top = x.effect_SD.idxmax()
        r[f'top knob ({s})'] = top; r[f'top effect ({s})'] = x.effect_SD.max()
    if fn in TGT:
        t = TGT[fn]; r['target'] = t
        r['recovered on both realistic bases'] = all(
            (g[g.base == s].set_index('knob').effect_SD.idxmax() == t) and
            (g[(g.base == s) & (g.knob == t)].p_bh.iloc[0] < 0.05) for s in SETS)
    summ.append(r)
S = pd.DataFrame(summ); S.to_csv(os.path.join(OUT, 'E3_known_function_summary.csv'), index=False)
pd.set_option('display.width', 220); print(S.round(2).to_string(index=False))
