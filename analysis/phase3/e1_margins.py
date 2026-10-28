"""E1. Standardised target-dominance margin of each control with a paired trajectory bootstrap (post hoc).
M = (effect of target knob - largest effect of the other six knobs) / SD of the classifier's logits on real test images.
Bootstrap: resample the 64 base indices (noise ids) with replacement; all seven knobs share them, so the margin is paired."""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from load_local import *
OUT = os.environ.get('A3_OUT', os.path.join(ROOT, '..', 'results', 'phase3_rebuilt'))
RNG = np.random.default_rng(2034); NB = 2000
C = classifiers(); metas = {s: meta(s) for s in SETS}
rows = []
for n in sorted(C):
    if kind(n) not in ('ctrlA', 'ctrlB', 'mel', 'quartile', 'perm'): continue
    L = logits(C[n]); sd = np.nanstd(L['cls_test'])
    lab = n.split('_')[1]; tgt = TARGET.get(lab, {'pig': 3, 'bwv': 1}.get(lab, 3))   # melanoma: margin of Pig
    for s in SETS:
        D = np.stack([(lambda T: L[s][T[:, -1]] - L[s][T[:, 0]])(traj_index(metas[s], j)[0]) for j in range(7)])  # (7, 64)
        def margin(idx):
            m = D[:, idx].mean(1); return (m[tgt] - np.delete(m, tgt).max()) / sd
        m0 = margin(np.arange(D.shape[1]))
        bs = np.array([margin(RNG.integers(0, D.shape[1], D.shape[1])) for _ in range(NB)])
        rows.append({'classifier': n, 'kind': kind(n), 'target': SHORT[tgt], 'base': s, 'logit_sd': sd,
                     'margin_logit': m0 * sd, 'margin_std': m0, 'lo': np.percentile(bs, 2.5), 'hi': np.percentile(bs, 97.5),
                     'P(margin>0)': float(np.mean(bs > 0))})
df = pd.DataFrame(rows); df.to_csv(os.path.join(OUT, 'E1_margins.csv'), index=False)
pd.set_option('display.width', 220)
print(df[df.kind.isin(['ctrlA', 'ctrlB'])].round(2).to_string(index=False))
g = df[df.kind.isin(['ctrlA', 'ctrlB'])].groupby(['kind', 'target']).apply(lambda x: pd.Series({'n cells': len(x), 'lo>0': int((x.lo > 0).sum()), 'median margin_std': x.margin_std.median()}))
print(g)
print(df[df.kind == 'mel'].round(2).to_string(index=False))
