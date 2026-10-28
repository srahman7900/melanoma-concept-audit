"""B. Image-level decomposition, tested on controls with known targets.

For every image of a sweep (levels > 0) against the level-0 image of its trajectory: y = change in classifier logit,
x = change in the seven CBM-B outputs and in mean L*, a*, b*. Ridge regression (alpha = 1) on standardised x, as in
the published decomposition (Methods). Coefficient = logit change per SD of the measured change. A control passes when
its target concept has the largest CBM-B coefficient and the bootstrap interval (trajectories resampled, 500) is above 0.
"""
import os, sys
import numpy as np, pandas as pd
from sklearn.linear_model import Ridge
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from load_local import *

OUT = os.environ.get('A3_OUT', os.path.join(ROOT, '..', 'results', 'phase3_rebuilt')); os.makedirs(OUT, exist_ok=True)
RNG = np.random.default_rng(2032); NB = int(os.environ.get('A3_NB', 500))
C = classifiers()
NAMES = [f'CBM-B {s}' for s in SHORT] + ['L*', 'a*', 'b*']
BASESETS = {'realistic': ['rsweep_ben', 'rsweep_mel'], 'all4': ['rsweep_ben', 'rsweep_mel', 'sweep_ben', 'sweep_mel']}
cache = {s: (meta(s), evaluators(s)) for s in BASESETS['all4']}


def design(sname):
    m, E = cache[sname]; g, X, idx1, idx0 = [], [], [], []
    for j in range(7):
        T, _, _ = traj_index(m, j)
        for t in range(T.shape[0]):
            for l in range(1, T.shape[1]):
                idx1.append(T[t, l]); idx0.append(T[t, 0]); g.append(f'{sname}|{j}|{t}')
    idx1, idx0 = np.array(idx1), np.array(idx0)
    X = np.column_stack([E['cbmb'][idx1] - E['cbmb'][idx0], E['lab'][idx1] - E['lab'][idx0]])
    return np.array(g), X, idx1, idx0


DES = {s: design(s) for s in BASESETS['all4']}


def fit(name, sets, nb=NB):
    L = logits(C[name]); g, X, Y = [], [], []
    for s in sets:
        gg, XX, i1, i0 = DES[s]; y = L[s][i1] - L[s][i0]; g.append(gg); X.append(XX); Y.append(y)
    g, X, Y = np.concatenate(g), np.concatenate(X), np.concatenate(Y)
    ok = np.isfinite(Y) & np.isfinite(X).all(1); g, X, Y = g[ok], X[ok], Y[ok]
    Z = (X - X.mean(0)) / (X.std(0) + 1e-9)
    f = Ridge(alpha=1.0).fit(Z, Y); coef = f.coef_
    ug, inv = np.unique(g, return_inverse=True); members = [np.where(inv == k)[0] for k in range(len(ug))]
    bs = []
    for _ in range(nb):
        ii = np.concatenate([members[k] for k in RNG.integers(0, len(ug), len(ug))])
        bs.append(Ridge(alpha=1.0).fit(Z[ii], Y[ii]).coef_)
    bs = np.array(bs)
    return coef, np.percentile(bs, 2.5, 0), np.percentile(bs, 97.5, 0), f.score(Z, Y), len(Y), bs


if __name__ == '__main__':
    rows = []
    names = sorted(C)
    for bset, sets in BASESETS.items():
        for n in names:
            k = kind(n)
            coef, lo, hi, r2, ny, bs = fit(n, sets)
            r = {'bases': bset, 'classifier': n, 'kind': k, 'R2': r2, 'n': ny}
            for i, nm in enumerate(NAMES): r[nm] = coef[i]; r[nm + ' lo'] = lo[i]; r[nm + ' hi'] = hi[i]
            top = int(np.argmax(coef[:7])); r['top'] = SHORT[top]
            # fraction of bootstrap samples in which the top concept stays top
            r['top_stable'] = float(np.mean(np.argmax(bs[:, :7], 1) == top))
            lab = n.split('_')[1]
            if lab in TARGET:
                j = TARGET[lab]; r['target'] = SHORT[j]
                r['pass'] = bool(top == j and lo[j] > 0)
            rows.append(r)
            print(bset, n, 'R2 %.2f' % r2, 'top', SHORT[top], r.get('target', ''), r.get('pass', ''), np.round(coef[:7], 2))
    df = pd.DataFrame(rows); df.to_csv(os.path.join(OUT, 'B_decomposition.csv'), index=False)
    for bset in BASESETS:
        d = df[(df.bases == bset) & df.kind.isin(['ctrlA', 'ctrlB'])]
        print(bset, d.groupby(['kind', 'target'])['pass'].agg(['sum', 'count']).to_string())
