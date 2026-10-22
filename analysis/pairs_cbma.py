"""Rebuild the Phase 2c/2d matched pairs (CBM-A pseudo-labels) offline, with the same rule as worker2.matched_groups.
Used by integrate_2e.py to read the balance of these pairs in the independent CBM-B labels."""
import numpy as np

CN = ['pigment_network', 'blue_whitish_veil', 'vascular_structures', 'pigmentation', 'streaks', 'dots_and_globules',
      'regression_structures']
CFG_A = {'q': 0.25, 'caliper': 0.5, 'caliper_by_concept': {'regression_structures': 1.0},
         'abs_high': {'regression_structures': 0.5}, 'seed': 2029}


def greedy_pairs(Z, y, split, Hm, Lm, cal, seed):
    rng = np.random.default_rng(seed); out = {}
    for s in ['train', 'val', 'test']:
        Hh = np.where((split == s) & Hm)[0]; Ll = np.where((split == s) & Lm)[0]; pairs = []
        if len(Hh) and len(Ll):
            D = np.sqrt(((Z[Hh][:, None, :] - Z[Ll][None, :, :]) ** 2).sum(-1)); D[y[Hh][:, None] != y[Ll][None, :]] = np.inf
            used = np.zeros(len(Ll), bool)
            for i in rng.permutation(len(Hh)):
                d = np.where(used, np.inf, D[i]); k = int(np.argmin(d))
                if d[k] <= cal:
                    used[k] = True; pairs.append((Hh[i], Ll[k]))
        out[s] = np.array(pairs, dtype=int).reshape(-1, 2)
    return out


def pairs_cbma(Cc, y, split, concept, M=CFG_A):
    j = CN.index(concept); tr = split == 'train'
    lo, hi = np.quantile(Cc[tr, j], M['q']), np.quantile(Cc[tr, j], 1 - M['q'])
    if concept in M['abs_high']:
        hi = float(M['abs_high'][concept])
    oth = [k for k in range(7) if k != j]; mu = Cc[tr][:, oth].mean(0); sd = Cc[tr][:, oth].std(0) + 1e-9
    Z = (Cc[:, oth] - mu) / sd
    cal = float(M['caliper_by_concept'].get(concept, M['caliper']))
    return greedy_pairs(Z, y, split, Cc[:, j] >= hi, Cc[:, j] <= lo, cal, M['seed'] + j), (float(lo), float(hi))


def smd(C, p, tr):
    sd = C[tr].std(0) + 1e-9
    return (C[p[:, 0]].mean(0) - C[p[:, 1]].mean(0)) / sd


def quartile_groups(Cc, split, j, q=0.25):
    tr = split == 'train'; v = Cc[tr, j]; lo, hi = np.quantile(v, q), np.quantile(v, 1 - q)
    idx = np.where(tr)[0]; sel = (v <= lo) | (v >= hi)
    return idx[sel][v[sel] >= hi], idx[sel][v[sel] <= lo]


if __name__ == '__main__':
    import json
    z = np.load('/home/claude/p2ainp/phase2/cls/cls_labels.npz', allow_pickle=True)
    Cc = z['concepts'].astype(np.float64); y = z['target'].astype(int); split = z['split']; tr = split == 'train'
    bal = json.load(open('/home/claude/p2dout/audit/results/matched_controls.json'))['balance']
    for k, c in enumerate(bal['Concept']):
        pr, (lo, hi) = pairs_cbma(Cc, y, split, c); s = smd(Cc, pr['train'], tr); j = CN.index(c)
        print(f"{c:24s} pairs {len(pr['train'])}/{len(pr['val'])}/{len(pr['test'])} (paper {bal['Pairs train / val / test'][k]}), "
              f"thr {lo:.3f}/{hi:.3f} (paper {bal['Low / high threshold'][k]}), target SMD {s[j]:.2f} (paper {bal['Target SMD'][k]}), "
              f"max other {np.abs(np.delete(s, j)).max():.3f} (paper {bal['Max |SMD| other six'][k]})")
