"""A. Off-target preservation: sweep effects on trajectories whose off-target CBM-B changes are small.

For knob j and trajectory i (one start noise, levels 0..1): dB_i = CBM-B(level 1) - CBM-B(level 0), in SD units of
the 969 real test images. Off-target size o_i = max_{k != j} |dB_ik|. Filters: o_i <= tau (absolute, tau = 1, 1.5, 2 SD)
and per-trajectory dominance (dB_ij > o_i). Effect = mean of logit(level 1) - logit(level 0) over kept trajectories,
bootstrap CI over trajectories (2,000). Covariate-adjusted effect: OLS of the per-trajectory logit change on the six
off-target changes and the change of mean L*, a*, b* (per 10 units); the intercept is the effect predicted at zero
measured off-target and colour change (an extrapolation when few trajectories are near zero).
"""
import os, sys, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from load_local import *

OUT = os.environ.get('A3_OUT', os.path.join(ROOT, '..', 'results', 'phase3_rebuilt')); os.makedirs(OUT, exist_ok=True)
RNG = np.random.default_rng(2031); NB = 2000
KNOBS = [0, 2, 3, 6]
sd = cbmb_sd(); C = classifiers()
MEL = [n for n in C if kind(n) == 'mel' and 'derm7pt' not in n and '_ham_' not in n and n.endswith(('_s1', '_s2'))]   # the six primary classifiers


def boot(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if len(x) < 2: return (np.nan, np.nan, np.nan)
    b = RNG.choice(x, (NB, len(x)), replace=True).mean(1)
    return (x.mean(), *np.percentile(b, [2.5, 97.5]))


rows, adj = [], []
cache = {s: (meta(s), evaluators(s)) for s in SETS}
for s in SETS:
    m, E = cache[s]
    for j in KNOBS:
        T, _, _ = traj_index(m, j)
        dB = (E['cbmb'][T[:, -1]] - E['cbmb'][T[:, 0]]) / sd
        dL = (E['lab'][T[:, -1]] - E['lab'][T[:, 0]]) / 10.0
        off = np.delete(np.abs(dB), j, 1).max(1)
        filters = {'all': np.ones(len(off), bool), 'off<=2SD': off <= 2, 'off<=1.5SD': off <= 1.5, 'off<=1SD': off <= 1,
                   'target>all off': dB[:, j] > off}
        # pooled over the six melanoma classifiers: mean of classifier-level effects on the same kept trajectories
        Dl = np.stack([logits(C[n])[s][T[:, -1]] - logits(C[n])[s][T[:, 0]] for n in MEL])   # (6, n_traj)
        for fn, keep in filters.items():
            k = keep & np.isfinite(Dl).all(0)
            r = {'base': s, 'knob': SHORT[j], 'filter': fn, 'n_traj': int(k.sum()),
                 'target_change_SD': float(dB[k, j].mean()) if k.any() else np.nan,
                 'max_offtarget_SD_median': float(np.median(off[k])) if k.any() else np.nan}
            mu, lo, hi = boot(Dl[:, k].mean(0)) if k.sum() >= 2 else (np.nan,) * 3
            r.update({'dlogit_mean6': mu, 'lo': lo, 'hi': hi})
            per = [float(Dl[i, k].mean()) if k.any() else np.nan for i in range(len(MEL))]
            r.update({f'dlogit_{n}': v for n, v in zip(MEL, per)})
            r['n_classifiers_positive'] = int(np.sum(np.array(per) > 0))
            rows.append(r)
        # covariate adjustment
        X = np.column_stack([np.delete(dB, j, 1), dL])
        for i, n in enumerate(MEL):
            y = Dl[i]; ok = np.isfinite(y)
            Xa = np.column_stack([np.ones(ok.sum()), X[ok]])
            beta = np.linalg.lstsq(Xa, y[ok], rcond=None)[0]
            bs = []
            for _ in range(500):
                ii = RNG.integers(0, ok.sum(), ok.sum())
                bs.append(np.linalg.lstsq(Xa[ii], y[ok][ii], rcond=None)[0][0])
            adj.append({'base': s, 'knob': SHORT[j], 'classifier': n, 'raw': float(y[ok].mean()), 'adjusted_intercept': float(beta[0]),
                        'lo': float(np.percentile(bs, 2.5)), 'hi': float(np.percentile(bs, 97.5)),
                        'median_offtarget_SD': float(np.median(off))})

df = pd.DataFrame(rows); df.to_csv(os.path.join(OUT, 'A_preservation_filter.csv'), index=False)
da = pd.DataFrame(adj); da.to_csv(os.path.join(OUT, 'A_covariate_adjusted.csv'), index=False)
pd.set_option('display.width', 250)
print(df[['base', 'knob', 'filter', 'n_traj', 'target_change_SD', 'max_offtarget_SD_median', 'dlogit_mean6', 'lo', 'hi', 'n_classifiers_positive']].round(2).to_string(index=False))
g = da.groupby(['base', 'knob'])[['raw', 'adjusted_intercept']].agg(['mean', 'min', 'max']).round(2)
print(g.to_string())
