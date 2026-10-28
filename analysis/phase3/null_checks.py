"""Post hoc checks with the Phase 3 null classifiers (offline; reads saved logits only).
1. The rank-based control rule (rules 2 to 4) applied to each permuted-label classifier on the four bases.
2. Standardised target margins of the matched controls and of the null classifiers (smaller value of the two realistic bases).
3. Architecture-matched groups of six null classifiers (2 ResNet-50, 2 EfficientNet-B0, 2 ViT-B/16) and an
   architecture-stratified comparison of the 15 ISIC 2020 melanoma classifiers.
Set A3_EXTRA_SCORES to the folder with the Phase 3 score files and A3_ROOT to the folder with p2ainp, p2cout, p2dout, p2eout."""
import os, sys, itertools
import numpy as np, pandas as pd
from scipy import stats
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from load_local import *
OUT = os.environ.get('A3_TABLES', os.path.join(ROOT, 'paper', 'tables'))
C = classifiers(); S4 = ['rsweep_ben', 'rsweep_mel', 'sweep_ben', 'sweep_mel']; M = {s: meta(s) for s in S4}


def bh(p):
    p = np.asarray(p); o = np.argsort(p); n = len(p); q = p[o] * n / (np.arange(n) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]; r = np.empty(n); r[o] = np.minimum(q, 1); return r


# 1. control rule on null classifiers
rows = []
for n in sorted(C):
    if kind(n) != 'perm': continue
    L = logits(C[n]); top = []
    for s in S4:
        eff, ps, ms = [], [], []
        for j in range(7):
            T, lev, _ = traj_index(M[s], j); Y = L[s][T]; d = Y[:, -1] - Y[:, 0]; xc = lev - lev.mean()
            sl = ((Y - Y.mean(1, keepdims=True)) * xc).sum(1) / (xc ** 2).sum(); nz = sl[sl != 0]
            ps.append(stats.wilcoxon(nz).pvalue); eff.append(d.mean()); ms.append(np.median(sl))
        q = bh(ps); k = int(np.argmax(eff)); top.append(SHORT[k] if (q[k] < 0.05 and ms[k] > 0) else None)
    r = {'classifier': n, **{SHORT[j]: sum(t == SHORT[j] for t in top) for j in [0, 2, 3, 6]}, 'tops': top}
    rows.append(r)
d = pd.DataFrame(rows); d.to_csv(os.path.join(OUT, 'T_null_control_rule.csv'), index=False)
print({SHORT[j]: int((d[SHORT[j]] >= 3).sum()) for j in [0, 2, 3, 6]}, 'of', len(d), 'null classifiers would pass')

# 2. standardised margins
rows = []
for n in sorted(C):
    k = kind(n)
    if k not in ('perm', 'ctrlA', 'ctrlB'): continue
    L = logits(C[n]); sd = np.nanstd(L['cls_test'])
    for tname, t in [('PN', 0), ('VS', 2), ('Pig', 3), ('Reg', 6)]:
        if k != 'perm' and TARGET[n.split('_')[1]] != t: continue
        ms = []
        for s in SETS:
            D = np.array([(lambda T: (L[s][T[:, -1]] - L[s][T[:, 0]]).mean())(traj_index(M[s], j)[0]) for j in range(7)])
            ms.append((D[t] - np.delete(D, t).max()) / sd)
        rows.append({'classifier': n, 'kind': k, 'target': tname, 'min_margin_std': min(ms), 'mean_margin_std': np.mean(ms)})
d = pd.DataFrame(rows); d.to_csv(os.path.join(OUT, 'T_margins_controls_vs_null.csv'), index=False)
for t in ['PN', 'VS', 'Pig', 'Reg']:
    q = d[(d.kind == 'perm') & (d.target == t)].min_margin_std.max(); c = d[(d.kind != 'perm') & (d.target == t)]
    print(t, 'largest null margin %.2f' % q, 'controls above:', int((c.min_margin_std > q).sum()), '/', len(c))

# 3. matched groups and architecture strata
E = pd.read_csv(os.path.join(OUT, 'P3_T2_effects_all.csv'))
per = E.groupby(['classifier', 'group', 'arch', 'knob']).scale_free.mean().reset_index()
out = []
for k in ['PN', 'VS', 'Pig', 'Reg']:
    dk = per[per.knob == k]; nul = dk[dk.group == 'null_trained']; T = dk[dk.group == 'mel_primary'].scale_free.mean()
    mel15 = dk[dk.group.isin(['mel_primary', 'mel_retrain'])]
    pools = {a: nul[nul.arch == a].scale_free.values for a in ['resnet50', 'effb0', 'vitb16']}
    means = np.array([np.mean(list(r) + list(e) + list(v)) for r in itertools.combinations(pools['resnet50'], 2)
                      for e in itertools.combinations(pools['effb0'], 2) for v in itertools.combinations(pools['vitb16'], 2)])
    r = {'knob': k, 'primary 6 mean': T, 'n matched null groups': len(means), 'null group 95th': np.percentile(means, 95),
         'null group max': means.max(), 'share of null groups >= observed': np.mean(means >= T)}
    for a in pools:
        m = mel15[mel15.arch == a].scale_free.values
        r[f'{a}: melanoma min'] = m.min(); r[f'{a}: null max (n={len(pools[a])})'] = pools[a].max(); r[f'{a}: melanoma above null max'] = f'{(m > pools[a].max()).sum()}/{len(m)}'
    out.append(r)
pd.DataFrame(out).round(4).to_csv(os.path.join(OUT, 'T_null_matched_groups_and_architecture.csv'), index=False)
print(pd.DataFrame(out).round(3).T)
