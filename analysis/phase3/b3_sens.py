"""Sensitivity of the knob contributions to the regression penalty and to the colour terms."""
import os, sys
import numpy as np, pandas as pd
from sklearn.linear_model import Ridge, LinearRegression
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('A3_NB', '10')
from b_decomp import *
MEL = [n for n in C if kind(n) == 'mel' and 'derm' not in n]
CTRL = [n for n in C if kind(n) in ('ctrlA', 'ctrlB')]
rows = []
for variant in ['ridge1', 'ols', 'ridge10', 'ridge1_nocolour']:
    for n in MEL + CTRL:
        L = logits(C[n]); X, Y = [], []
        for s in BASESETS['realistic']:
            gg, XX, i1, i0 = DES[s]; X.append(XX); Y.append(L[s][i1] - L[s][i0])
        X, Y = np.concatenate(X), np.concatenate(Y)
        if variant.endswith('nocolour'): X = X[:, :7]
        mu, sdx = X.mean(0), X.std(0) + 1e-9; Z = (X - mu) / sdx
        mdl = {'ridge1': Ridge(1.0), 'ols': LinearRegression(), 'ridge10': Ridge(10.0), 'ridge1_nocolour': Ridge(1.0)}[variant]
        b = mdl.fit(Z, Y).coef_ / sdx
        top = SHORT[int(np.argmax(mdl.coef_[:7]))]
        r = {'variant': variant, 'classifier': n, 'top': top}
        lab = n.split('_')[1]
        if lab in TARGET: r['pass_top'] = top == SHORT[TARGET[lab]]
        for j in [3, 2]:
            m, E = cache['rsweep_ben']; dx = []
            for s in BASESETS['realistic']:
                m, E = cache[s]; T, _, _ = traj_index(m, j)
                d = np.column_stack([E['cbmb'][T[:, -1]] - E['cbmb'][T[:, 0]], E['lab'][T[:, -1]] - E['lab'][T[:, 0]]])
                dx.append(d[:, :X.shape[1]])
            con = b * np.concatenate(dx).mean(0)
            r[f'{SHORT[j]} sweep: own part'] = con[j]; r[f'{SHORT[j]} sweep: BWV part'] = con[1]
        rows.append(r)
df = pd.DataFrame(rows)
print(df[df.classifier.isin(CTRL)].groupby('variant').pass_top.agg(['sum', 'count']))
print(df[df.classifier.isin(MEL)].groupby('variant')[['Pig sweep: own part', 'Pig sweep: BWV part', 'VS sweep: own part']].agg(['min', 'max']).round(2).to_string())
df.to_csv(os.path.join(OUT, 'B3_sensitivity.csv'), index=False)
