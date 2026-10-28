"""B2. Which measured changes carry each knob's effect: Δlogit(knob j sweep, level 1) = sum_k b_k Δx_k + rest,
with b from the image-level decomposition (realistic bases). Contribution_k = b_k * mean Δx_k over the level-1 images of
the knob-j sweep (both realistic bases); rest = measured effect - sum of contributions."""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('A3_NB', '300')
from b_decomp import *

rows = []
for n in sorted(C):
    k = kind(n)
    if k not in ('mel', 'ctrlA', 'ctrlB', 'perm', 'quartile'): continue
    sets = BASESETS['realistic']; L = logits(C[n])
    g, X, Y = [], [], []
    for s in sets:
        gg, XX, i1, i0 = DES[s]; g.append(gg); X.append(XX); Y.append(L[s][i1] - L[s][i0])
    g, X, Y = np.concatenate(g), np.concatenate(X), np.concatenate(Y)
    mu, sdx = X.mean(0), X.std(0) + 1e-9
    f = Ridge(alpha=1.0).fit((X - mu) / sdx, Y); b = f.coef_ / sdx   # per raw unit
    ug, inv = np.unique(g, return_inverse=True); members = [np.where(inv == q)[0] for q in range(len(ug))]
    bb = []
    for _ in range(300):
        ii = np.concatenate([members[q] for q in RNG.integers(0, len(ug), len(ug))])
        Xi = X[ii]; m_, s_ = Xi.mean(0), Xi.std(0) + 1e-9
        bb.append(Ridge(alpha=1.0).fit((Xi - m_) / s_, Y[ii]).coef_ / s_)
    bb = np.array(bb)
    for j in [0, 2, 3, 6]:
        dx, dy = [], []
        for s in sets:
            m, E = cache[s]; T, _, _ = traj_index(m, j)
            dx.append(np.column_stack([E['cbmb'][T[:, -1]] - E['cbmb'][T[:, 0]], E['lab'][T[:, -1]] - E['lab'][T[:, 0]]]))
            dy.append(L[s][T[:, -1]] - L[s][T[:, 0]])
        dx, dy = np.concatenate(dx).mean(0), np.nanmean(np.concatenate(dy))
        con = b * dx; conb = bb * dx
        r = {'classifier': n, 'kind': k, 'knob': SHORT[j], 'measured': dy, 'R2': f.score((X - mu) / sdx, Y)}
        for q, nm in enumerate(NAMES):
            r[nm] = con[q]; r[nm + ' lo'] = np.percentile(conb[:, q], 2.5); r[nm + ' hi'] = np.percentile(conb[:, q], 97.5)
        r['colour'] = con[7:].sum(); r['concepts'] = con[:7].sum(); r['rest'] = dy - con.sum()
        r['target part'] = con[j]; r['target part lo'] = np.percentile(conb[:, j], 2.5); r['target part hi'] = np.percentile(conb[:, j], 97.5)
        rows.append(r)
df = pd.DataFrame(rows); df.to_csv(os.path.join(OUT, 'B2_knob_contributions.csv'), index=False)
pd.set_option('display.width', 250)
cols = ['classifier', 'knob', 'measured', 'R2'] + NAMES[:7] + ['colour', 'rest']
print(df[df.kind == 'mel'][cols].round(2).to_string(index=False))
print(df[df.kind.isin(['ctrlA'])][cols].round(2).to_string(index=False))
