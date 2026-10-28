"""ClinDiffXAI Phase 3 analysis: empirical null distribution and retraining variability of the concept audit.

Usage: python analysis3.py <cfg.json>
cfg keys: models (list of specs: name, arch, label, seed, data, group), scores_dir, results_dir, meta_dir, eval_dirs (list),
          cls_labels, generic_dir (optional), out, rule_text, primary (list of the six primary melanoma classifier names),
          n_boot.
Groups: 'mel_primary' (the six Phase 2 ISIC 2020 classifiers), 'mel_retrain' (new seeds), 'null_trained' (permuted
labels), 'other' (not used in the null or retraining summaries).
All effects use the realistic bases (rsweep_ben, rsweep_mel) and the four target-dominant knobs, as in the paper.
"""
import os, sys, json, glob, time, hashlib
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SHORT = ['PN', 'BWV', 'VS', 'Pig', 'Str', 'D/G', 'Reg']
KNOBS = [0, 2, 3, 6]   # target-dominant knobs (paper, Fig. 2c)
SETS = [('rsweep_ben', 'Real benign base'), ('rsweep_mel', 'Real melanoma base')]
ARCH = {'resnet50': 'ResNet-50', 'effb0': 'EfficientNet-B0', 'vitb16': 'ViT-B/16'}
CLIP = 15.0
REP = []


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


# ----------------------------------------------------------------------------- data
def load_meta(cfg, s):
    m = json.load(open(os.path.join(cfg['meta_dir'], f'{s}.json')))['meta']
    return {k: np.asarray(v) for k, v in m.items()}


def load_eval(cfg, s):
    out = {}
    for d in cfg['eval_dirs']:
        fa, fb = os.path.join(d, f'A_{s}.npz'), os.path.join(d, f'B_{s}.npz')
        if os.path.exists(fa) and os.path.exists(fb):
            A, B = np.load(fa), np.load(fb)
            return {'cbmb': B['cbmb'].astype(np.float64), 'lab': A['lab'].astype(np.float64)}
    raise FileNotFoundError(f'evaluator scores for {s} not found in {cfg["eval_dirs"]}')


def traj_index(m, j):
    sel = np.where(m['concept_idx'] == j)[0]
    lev = np.unique(m['level'][sel]); nid = np.unique(m['noise_id'][sel])
    T = np.full((len(nid), len(lev)), -1, int)
    T[np.searchsorted(nid, m['noise_id'][sel]), np.searchsorted(lev, m['level'][sel])] = sel
    if (T < 0).any(): raise ValueError('incomplete trajectories')
    return T, lev


def logits(path):
    z = np.load(path)
    return {k: np.clip(np.where(np.isfinite(z[k]), z[k], np.nan).astype(np.float64), -CLIP, CLIP) for k in z.files}


def slopes(Y, lev):
    x = np.asarray(lev, float); xc = x - x.mean()
    return ((Y - Y.mean(1, keepdims=True)) * xc).sum(1) / (xc ** 2).sum()


def wilcoxon_p(v):
    v = v[np.isfinite(v)]; nz = v[v != 0]
    if len(nz) < 5: return np.nan
    return float(stats.wilcoxon(nz).pvalue)


def bh(p):
    p = np.asarray(p, float); out = np.full_like(p, np.nan); ok = np.isfinite(p)
    if not ok.any(): return out
    q = p[ok]; n = len(q); o = np.argsort(q); r = q[o] * n / (np.arange(n) + 1)
    r = np.minimum.accumulate(r[::-1])[::-1]; res = np.empty(n); res[o] = np.minimum(r, 1); out[ok] = res
    return out


def boot_mean(x, B, rng):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if len(x) < 2: return np.nan, np.nan, np.nan
    b = x[rng.integers(0, len(x), (B, len(x)))].mean(1)
    return float(x.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def fci(m, lo, hi, d=2):
    return f"{m:+.{d}f} [{lo:+.{d}f}, {hi:+.{d}f}]" if np.isfinite(m) else 'n/a'


class Out:
    def __init__(self, root):
        self.root = root
        for d in ['tables', 'figures', 'results']: os.makedirs(os.path.join(root, d), exist_ok=True)

    def table(self, df, name, caption):
        df.to_csv(os.path.join(self.root, 'tables', f'{name}.csv'), index=False)
        try:
            tex = df.to_latex(index=False, escape=True, caption=caption, label=f'tab:{name}')
        except Exception:
            tex = df.to_string(index=False)
        open(os.path.join(self.root, 'tables', f'{name}.tex'), 'w').write(tex)
        REP.append(f'\n### {name}\n\n{caption}\n\n' + df.to_string(index=False))

    def fig(self, f, name):
        for ext in ['png', 'pdf']:
            f.savefig(os.path.join(self.root, 'figures', f'{name}.{ext}'), dpi=200, bbox_inches='tight')
        plt.close(f)


# ----------------------------------------------------------------------------- analyses
def classifier_table(cfg, out, models, L, rng):
    lab = np.load(cfg['cls_labels']); split = lab['split'].astype(str); yt = lab['target'][split == 'test']
    rows = []
    for m in models:
        lg = L[m['name']]; t = lg['cls_test']
        from sklearn.metrics import roc_auc_score
        ok = np.isfinite(t); auc = float(roc_auc_score(yt[ok], t[ok])) if len(np.unique(yt[ok])) == 2 else np.nan
        # bootstrap CI for AUC
        bs = []
        for _ in range(500):
            ii = rng.integers(0, ok.sum(), ok.sum()); yy = yt[ok][ii]
            if len(np.unique(yy)) == 2: bs.append(roc_auc_score(yy, t[ok][ii]))
        res = {}
        f = os.path.join(cfg['results_dir'], f"train_{m['name']}.json")
        if os.path.exists(f): res = json.load(open(f))
        rows.append({'classifier': m['name'], 'group': m['group'], 'arch': ARCH.get(m['arch'], m['arch']), 'seed': m['seed'],
                     'label': m['label'], 'test AUC (melanoma)': auc, 'AUC lo': np.percentile(bs, 2.5) if bs else np.nan,
                     'AUC hi': np.percentile(bs, 97.5) if bs else np.nan, 'test logit SD': float(np.nanstd(t)),
                     'best val AUC (own label)': res.get('best_val_auc', np.nan),
                     'own-split test AUC (melanoma)': res.get('test_auc_melanoma', np.nan), 'epochs': len(res.get('history', [])),
                     'minutes': res.get('minutes', np.nan)})
    df = pd.DataFrame(rows)
    out.table(df.round(3), 'T1_classifiers', 'Classifiers. Test AUC for the melanoma label on the 969 real ISIC 2020 test '
              'images (bootstrap 95% CI, 500 resamples; for HAM10000 classifiers this is a cross-dataset test); own-split test '
              'AUC is on the test split of the training dataset; test logit SD (ISIC 2020) is the scale of the scale-free effects.')
    # a classifier with (almost) constant output has no scale: its scale-free effects are not defined
    return {r['classifier']: (r['test logit SD'] if r['test logit SD'] > 1e-6 else np.nan) for r in rows}


def effects_table(cfg, out, models, L, sdlog, metas, rng):
    rows = []
    for m in models:
        for s, slab in SETS:
            fam = []
            for j in range(7):
                T, lev = traj_index(metas[s], j); Y = L[m['name']][s][T]
                ok = np.isfinite(Y).all(1); Y = Y[ok]
                dl = Y[:, -1] - Y[:, 0]; mu, lo, hi = boot_mean(dl, cfg['n_boot'], rng)
                fam.append({'classifier': m['name'], 'group': m['group'], 'arch': m['arch'], 'seed': m['seed'], 'base': s,
                            'knob': SHORT[j], 'n_traj': int(ok.sum()), 'dlogit': mu, 'lo': lo, 'hi': hi,
                            'p': wilcoxon_p(slopes(Y, lev)), 'median_slope': float(np.median(slopes(Y, lev))),
                            'scale_free': mu / sdlog[m['name']], 'sf_lo': lo / sdlog[m['name']], 'sf_hi': hi / sdlog[m['name']]})
            q = bh([f['p'] for f in fam])
            for f, qq in zip(fam, q): f['p_bh'] = float(qq); f['sig_pos'] = bool(qq < 0.05 and f['median_slope'] > 0)
            rows += fam
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out.root, 'tables', 'T2_effects_all.csv'), index=False)
    return df


def null_summary(cfg, out, E):
    """Rule 5: trained null (permuted labels), scale-free effects, pooled over the two realistic bases (mean per model)."""
    per = E.groupby(['classifier', 'group', 'knob'])[['scale_free', 'dlogit']].mean().reset_index()
    rows = []
    for j in KNOBS + [k for k in range(7) if k not in KNOBS]:
        k = SHORT[j]; d = per[per.knob == k]
        fin = lambda v: v[np.isfinite(v)]
        nul = fin(d[d.group == 'null_trained']['scale_free'].values); mel = fin(d[d.group == 'mel_primary']['scale_free'].values)
        mel_all = fin(d[d.group.isin(['mel_primary', 'mel_retrain'])]['scale_free'].values)
        q95 = float(np.percentile(nul, 95)) if len(nul) else np.nan
        mm = float(np.mean(mel)) if len(mel) else np.nan
        rows.append({'knob': k, 'target-dominant': j in KNOBS, 'n null models': len(nul),
                     'null mean': float(np.mean(nul)) if len(nul) else np.nan, 'null SD': float(np.std(nul, ddof=1)) if len(nul) > 1 else np.nan,
                     'null max': float(np.max(nul)) if len(nul) else np.nan, 'null 95th pct': q95,
                     'primary melanoma mean': mm, 'primary melanoma min': float(np.min(mel)) if len(mel) else np.nan,
                     'all melanoma mean': float(np.mean(mel_all)) if len(mel_all) else np.nan,
                     'empirical p (null >= melanoma mean)': (1 + np.sum(nul >= mm)) / (1 + len(nul)) if len(nul) else np.nan,
                     'n primary above null 95th': int(np.sum(mel > q95)) if len(nul) else np.nan,
                     'above null (rule 5)': bool(mm > q95) if len(nul) else None})
    df = pd.DataFrame(rows)
    out.table(df.round(3), 'T3_null_distribution', 'Rule 5. Scale-free sweep effect (logit change from level 0 to 1 divided '
              'by the SD of the classifier logits on the real test images), mean of the two realistic bases per classifier. '
              'Null: classifiers trained on permuted melanoma labels. A finding is above null when the mean of the six primary '
              'melanoma classifiers exceeds the 95th percentile of the null classifiers.')
    # per-classifier empirical upper-tail probabilities: one classifier against single null classifiers (comparable
    # statistics); resolution 1/(R+1), so with 18 trained nulls the smallest possible value is 1/19 = 0.053
    rows = []
    for g in ['mel_primary', 'mel_retrain', 'mel_ham']:
        for n in sorted(per[per.group == g].classifier.unique()):
            r = {'classifier': n, 'group': g}
            for j in KNOBS:
                k = SHORT[j]; d = per[per.knob == k]; T = d[d.classifier == n]['scale_free'].values[0]
                for ng, lab in [('null_trained', 'trained'), ('null_generic', 'random heads')]:
                    N = d[d.group == ng]['scale_free'].values; N = N[np.isfinite(N)]
                    r[f'{k} p ({lab})'] = (1 + np.sum(N >= T)) / (1 + len(N)) if len(N) and np.isfinite(T) else np.nan
                r[f'{k} scale-free'] = T
            rows.append(r)
    if rows:
        nt = int(np.isfinite(per[(per.group == 'null_trained') & (per.knob == 'Pig')]['scale_free']).sum())
        ng = int(np.isfinite(per[(per.group == 'null_generic') & (per.knob == 'Pig')]['scale_free']).sum())
        out.table(pd.DataFrame(rows).round(4), 'T3b_empirical_tail_probabilities',
                  f'Empirical upper-tail probability p = (1 + number of null classifiers with a scale-free effect at least as '
                  f'large) / (R + 1), for each melanoma classifier and knob, against the {nt} permuted-label classifiers (smallest '
                  f'possible value {1 / (nt + 1) if nt else float("nan"):.3f}) and the {ng} random heads (smallest possible '
                  f'value {1 / (ng + 1) if ng else float("nan"):.4f}). A robustness check, not a formal test at 0.05 when R = 18.')
    return df, per


def fpr_table(cfg, out, E):
    rows = []
    for g in ['null_trained', 'null_generic', 'mel_primary', 'mel_retrain', 'mel_ham']:
        d = E[(E.group == g) & E.knob.isin([SHORT[j] for j in KNOBS])]
        if not len(d): continue
        rows.append({'group': g, 'cells (model x base x knob)': len(d), 'BH p < 0.05 and rising': int(d.sig_pos.sum()),
                     'fraction': float(d.sig_pos.mean()),
                     'fraction, all 7 knobs': float(E[E.group == g].sig_pos.mean())})
    df = pd.DataFrame(rows)
    out.table(df.round(3), 'T4_test_false_positive_rate', 'Share of classifier x base x knob cells in which the per-classifier '
              'test of the paper (two-sided Wilcoxon signed-rank test of per-trajectory slopes, BH within the 7 knobs) is '
              'significant with a rising median slope. For null classifiers this is a false-positive rate of the per-model '
              'test, which treats trajectories, not models, as the unit.')
    return df


def retrain_table(cfg, out, E):
    """Rule 6: pigmentation-sweep finding under retraining (all ISIC 2020 melanoma classifiers)."""
    M = E[E.group.isin(['mel_primary', 'mel_retrain'])]
    rows = []
    for j in KNOBS:
        k = SHORT[j]; d = M[M.knob == k]
        per = d.groupby('classifier')['dlogit'].mean()
        pos_ci = d.groupby('classifier').apply(lambda x: bool((x['lo'] > 0).all()))
        rows.append({'knob': k, 'n classifiers': len(per), 'mean dlogit': per.mean(), 'SD between classifiers': per.std(ddof=1),
                     'min': per.min(), 'max': per.max(), 'n positive': int((per > 0).sum()),
                     'n CI > 0 on both bases': int(pos_ci.sum())})
    df = pd.DataFrame(rows)
    # pigmentation largest of the four target-dominant knobs
    piv = M[M.knob.isin([SHORT[j] for j in KNOBS])].groupby(['classifier', 'knob'])['dlogit'].mean().unstack()
    top = piv.idxmax(axis=1)
    n = len(piv); n_pig_top = int((top == 'Pig').sum())
    pig = df[df.knob == 'Pig'].iloc[0]
    # (a) CI above 0 on both bases for >= 80% of classifiers; (b) largest mean effect of the four knobs over all classifiers
    ok = bool(pig['n CI > 0 on both bases'] >= int(np.ceil(0.8 * n)) and df.loc[df['mean dlogit'].idxmax(), 'knob'] == 'Pig')
    df['Pig largest of 4 (n)'] = [int((top == k).sum()) for k in df.knob]
    out.table(df.round(3), 'T5_retraining', 'Rule 6. Sweep effects of all ISIC 2020 melanoma classifiers (six primary and the '
              'retrained seeds), mean of the two realistic bases. The pigmentation finding is stable if its bootstrap CI is '
              'above 0 on both bases for at least 80% of the classifiers and pigmentation has the largest mean effect of the '
              'four target-dominant knobs over all classifiers. Last column: number of classifiers in which each knob has the '
              'largest effect (descriptive).')
    piv.round(3).to_csv(os.path.join(out.root, 'tables', 'T5b_retraining_per_classifier.csv'))
    # sources of variation, kept apart: trajectories (bootstrap within a classifier), seeds (within an architecture),
    # architectures (between architecture means); raw logits and scale-free units
    vc = []
    for j in KNOBS:
        k = SHORT[j]; d = M[M.knob == k]
        for unit, col, lo, hi in [('logit', 'dlogit', 'lo', 'hi'), ('scale-free', 'scale_free', 'sf_lo', 'sf_hi')]:
            pc = d.groupby(['classifier', 'arch'])[col].mean().reset_index()
            am = pc.groupby('arch')[col].mean(); ws = pc.groupby('arch')[col].std(ddof=1)
            vc.append({'knob': k, 'unit': unit, 'mean over classifiers': pc[col].mean(),
                       'SD between architectures (of architecture means)': am.std(ddof=1),
                       'mean SD between seeds within architecture': ws.mean(),
                       'mean half-width of trajectory 95% CI': float(((d[hi] - d[lo]) / 2).mean()),
                       **{f'mean {ARCH.get(x, x)}': v for x, v in am.items()},
                       **{f'n seeds {ARCH.get(x, x)}': int((pc.arch == x).sum()) for x in am.index}})
    out.table(pd.DataFrame(vc).round(3), 'T5c_sources_of_variation', 'Sources of variation of the sweep effects of the ISIC '
              '2020 melanoma classifiers (mean of the two realistic bases), reported separately: between architectures, between '
              'seeds of one architecture (retraining) and between generated trajectories for a fixed classifier (bootstrap).')
    REP.append('\nPer classifier (mean of two bases):\n\n' + piv.round(2).to_string())
    return df, bool(ok), n, n_pig_top


def profile_table(cfg, out, per):
    rows = []
    for g in ['null_trained', 'null_generic', 'mel_primary', 'mel_retrain', 'mel_ham']:
        d = per[per.group == g]
        if not len(d): continue
        piv = d.pivot(index='classifier', columns='knob', values='scale_free')[[SHORT[j] for j in range(7)]].dropna(how='any')
        if not len(piv): continue
        top = piv.idxmax(axis=1)
        r = {'group': g, 'n': len(piv)}
        for k in SHORT: r[f'{k} largest'] = float((top == k).mean())
        r['mean |effect| over 7 knobs'] = float(piv.abs().mean(axis=1).mean())
        rows.append(r)
    df = pd.DataFrame(rows)
    out.table(df.round(3), 'T6_profile', 'Share of classifiers in which each knob has the largest scale-free effect (mean of '
              'the two realistic bases). A sweep that moves any image-sensitive model would be largest also for the null '
              'classifiers.')
    return df


def observational_null(cfg, out, models, L, sdlog):
    """observational attribution (ridge, alpha 1, standardised CBM-B and CIELAB on the real test images), scale-free"""
    from sklearn.linear_model import Ridge
    ev = load_eval(cfg, 'cls_test'); X = np.column_stack([ev['cbmb'], ev['lab']]); Z = (X - X.mean(0)) / (X.std(0) + 1e-9)
    rows = []
    for m in models:
        y = L[m['name']]['cls_test']; ok = np.isfinite(y)
        c = Ridge(alpha=1.0).fit(Z[ok], y[ok]).coef_
        rows.append({'classifier': m['name'], 'group': m['group'], **{SHORT[i]: c[i] / sdlog[m['name']] for i in range(7)}})
    O = pd.DataFrame(rows); O.to_csv(os.path.join(out.root, 'tables', 'T7b_observational_all.csv'), index=False)
    res = []
    for j in KNOBS:
        k = SHORT[j]; nul = O[O.group == 'null_trained'][k].values; mel = O[O.group == 'mel_primary'][k].values
        nul, mel = nul[np.isfinite(nul)], mel[np.isfinite(mel)]
        q95 = np.percentile(nul, 95) if len(nul) else np.nan; mm = mel.mean() if len(mel) else np.nan
        res.append({'concept': k, 'null mean': nul.mean() if len(nul) else np.nan, 'null 95th pct': q95,
                    'primary melanoma mean': mm, 'empirical p': (1 + np.sum(nul >= mm)) / (1 + len(nul)) if len(nul) else np.nan,
                    'above null': bool(mm > q95) if len(nul) else None})
    df = pd.DataFrame(res)
    out.table(df.round(3), 'T7_observational_null', 'Observational attribution against the same null: ridge coefficient per '
              'SD of each predictor, divided by the SD of the classifier logits on the real test images.')
    return df


def preservation_table(cfg, out, models, L, metas, evs, rng):
    """off-target preservation filters (as in the offline analysis A) for every melanoma classifier"""
    sdB = load_eval(cfg, 'cls_test')['cbmb'].std(0) + 1e-9
    rows = []
    for s, slab in SETS:
        for j in KNOBS:
            T, lev = traj_index(metas[s], j)
            dB = (evs[s]['cbmb'][T[:, -1]] - evs[s]['cbmb'][T[:, 0]]) / sdB
            off = np.delete(np.abs(dB), j, 1).max(1)
            filt = {'all': np.ones(len(off), bool), 'off-target <= 2 SD': off <= 2, 'target > every off-target': dB[:, j] > off}
            for g in ['mel_primary', 'mel_retrain', 'mel_ham']:
                mm = [m for m in models if m['group'] == g]
                if not mm: continue
                D = np.stack([L[m['name']][s][T[:, -1]] - L[m['name']][s][T[:, 0]] for m in mm])
                for fn, keep in filt.items():
                    k = keep & np.isfinite(D).all(0)
                    mu, lo, hi = boot_mean(D[:, k].mean(0), cfg['n_boot'], rng) if k.sum() >= 2 else (np.nan,) * 3
                    rows.append({'group': g, 'base': s, 'knob': SHORT[j], 'filter': fn, 'n trajectories': int(k.sum()),
                                 'n trajectories before filter': int(len(k)),
                                 'target change (SD)': float(dB[k, j].mean()) if k.any() else np.nan,
                                 'median largest off-target change (SD)': float(np.median(off[k])) if k.any() else np.nan,
                                 'mean base value of target (kept)': float(np.mean(metas[s]['base_value'][T[k, 0]])) if k.any() else np.nan,
                                 'mean dlogit': mu, 'lo': lo, 'hi': hi,
                                 'n classifiers > 0': int(np.sum(D[:, k].mean(1) > 0)) if k.any() else 0, 'n classifiers': len(mm)})
    df = pd.DataFrame(rows)
    out.table(df.round(3), 'T8_preservation', 'Sweep effects on trajectories with small off-target change. Off-target change: '
              'largest absolute change of the six other CBM-B outputs (level 1 minus level 0, SD units of real test images).')
    return df


def ham_table(cfg, out, E):
    """Rule 7: classifiers trained on HAM10000"""
    M = E[E.group == 'mel_ham']
    if not len(M): return None, None
    rows = []
    for j in KNOBS:
        k = SHORT[j]; d = M[M.knob == k]; per = d.groupby('classifier')['dlogit'].mean()
        pos_ci = d.groupby('classifier').apply(lambda x: bool((x['lo'] > 0).all()))
        rows.append({'knob': k, 'n classifiers': len(per), 'mean dlogit': per.mean(), 'min': per.min(), 'max': per.max(),
                     'n CI > 0 on both bases': int(pos_ci.sum()),
                     'ISIC 2020 primary mean': E[(E.group == 'mel_primary') & (E.knob == k)].groupby('classifier')['dlogit'].mean().mean()})
    df = pd.DataFrame(rows); n = df['n classifiers'].max()
    pig = df[df.knob == 'Pig'].iloc[0]
    ok = bool(pig['n CI > 0 on both bases'] >= n - 1 and df.loc[df['mean dlogit'].idxmax(), 'knob'] == 'Pig') if n else None
    out.table(df.round(3), 'T9_ham10000', 'Rule 7. Cross-dataset classifier transfer under a fixed counterfactual instrument: '
              'sweep effects of the melanoma classifiers trained on HAM10000 (mean of the two realistic bases), with the same '
              'ISIC 2020-trained generator and sweeps. The pigmentation finding transfers if its CI is above 0 on both bases in all but at '
              'most one HAM10000 classifier and pigmentation has the largest mean effect of the four knobs.')
    piv = M[M.knob.isin([SHORT[j] for j in range(7)])].groupby(['classifier', 'knob'])['dlogit'].mean().unstack()
    piv.round(3).to_csv(os.path.join(out.root, 'tables', 'T9b_ham10000_per_classifier.csv'))
    REP.append('\nHAM10000 classifiers, all seven knobs (mean of two bases):\n\n' + piv.round(2).to_string())
    return df, ok


def ham_reports(cfg, out):
    rd = cfg['results_dir']
    for f, title in [('ham_data.json', 'HAM10000 data and overlap protocol'), ('ham_near_duplicates.json', 'HAM10000 near-duplicate screen (report only)')]:
        p = os.path.join(rd, f)
        if os.path.exists(p):
            d = json.load(open(p)); d = {k: v for k, v in d.items() if not isinstance(v, list)}
            REP.append(f'\n### {title}\n\n' + '\n'.join(f'- {k}: {v}' for k, v in d.items()))
            json.dump(d, open(os.path.join(out.root, 'results', f), 'w'), indent=2)


def ph2_tables(cfg, out):
    rd = cfg['results_dir']; f = os.path.join(rd, 'ph2_results.json')
    if not os.path.exists(f): return None
    st = json.load(open(f))
    if st.get('status') != 'ok':
        REP.append('\n### PH2\n\nPH2 step: ' + str(st.get('status')) + '\n\nFirst lines of the annotation file:\n\n' + '\n'.join(st.get('first_lines', [])))
        return st
    fm = os.path.join(rd, 'ph2_mapping.csv')
    if os.path.exists(fm):
        out.table(pd.read_csv(fm), 'T10a_ph2_mapping', 'PH2 criteria and the CBM-B outputs they are compared with (frozen before the run).')
    ev = pd.read_csv(os.path.join(rd, 'ph2_evaluators.csv'))
    if 'n positive' in ev: ev['n negative'] = ev['n'] - ev['n positive']
    ev['confirmed (CI > 0.5)'] = ev['lo'] > 0.5 if 'lo' in ev else False
    out.table(ev.round(3), 'T10_ph2_evaluators', 'Rule 8 (descriptive). Concept evaluators against the expert dermoscopic '
              'criteria of PH2 (200 images from a different centre, not used in any training). AUC with 95% bootstrap CI '
              '(2,000 resamples); a criterion is confirmed when the lower bound is above 0.5.')
    fc = os.path.join(rd, 'ph2_classifiers.csv')
    if os.path.exists(fc) and os.path.getsize(fc) > 5:
        try:
            out.table(pd.read_csv(fc).round(3), 'T10b_ph2_classifiers', 'Melanoma classifiers on PH2 (melanoma against common and atypical nevi).')
        except Exception:
            pass
    return st


# ----------------------------------------------------------------------------- figures
COL = {'null_trained': '#8a8a8a', 'null_generic': '#c9c9c9', 'mel_primary': '#c2410c', 'mel_retrain': '#2563eb', 'mel_ham': '#15803d'}
LAB = {'null_trained': 'Permuted-label classifiers', 'null_generic': 'Random heads on ImageNet features',
       'mel_primary': 'Melanoma classifiers (primary six)', 'mel_retrain': 'Melanoma classifiers (new seeds)',
       'mel_ham': 'Melanoma classifiers (HAM10000)'}


def fig_null(out, per, nul_tab):
    f, ax = plt.subplots(figsize=(7.2, 3.4)); rng = np.random.default_rng(0)
    ks = [SHORT[j] for j in KNOBS]
    for xi, k in enumerate(ks):
        for gi, g in enumerate(['null_generic', 'null_trained', 'mel_retrain', 'mel_primary', 'mel_ham']):
            v = per[(per.knob == k) & (per.group == g)]['scale_free'].values
            if not len(v): continue
            off = {'null_generic': -0.32, 'null_trained': -0.16, 'mel_retrain': 0.0, 'mel_primary': 0.16, 'mel_ham': 0.32}[g]
            ax.scatter(xi + off + rng.uniform(-0.05, 0.05, len(v)), v, s=9 if g == 'null_generic' else 16, c=COL[g],
                       alpha=0.5 if g == 'null_generic' else 0.9, lw=0, label=LAB[g] if xi == 0 else None, zorder=2)
        q = nul_tab[nul_tab.knob == k]['null 95th pct'].values
        if len(q) and np.isfinite(q[0]):
            ax.plot([xi - 0.4, xi + 0.4], [q[0], q[0]], color='k', lw=0.8, ls='--', zorder=3)
    ax.axhline(0, color='k', lw=0.5)
    ax.set_xticks(range(len(ks))); ax.set_xticklabels(['Pigment network', 'Vascular structures', 'Pigmentation', 'Regression'])
    ax.set_ylabel('Scale-free sweep effect\n(Δ logit / SD of test logits)')
    ax.legend(frameon=False, fontsize=7, loc='upper left'); ax.spines[['top', 'right']].set_visible(False)
    ax.text(0.99, 0.02, 'dashed: 95th percentile of permuted-label classifiers', transform=ax.transAxes, ha='right', fontsize=7)
    out.fig(f, 'F1_null_distribution')


def fig_retrain(out, E):
    M = E[E.group.isin(['mel_primary', 'mel_retrain', 'mel_ham'])]
    ks = [SHORT[j] for j in KNOBS]; names = sorted(M.classifier.unique(), key=lambda n: (n.split('_')[1], int(n.split('_s')[-1]) if n.split('_s')[-1].isdigit() else 0))
    f, axs = plt.subplots(1, 4, figsize=(9, 3.6), sharey=True)
    for ax, k in zip(axs, ks):
        for yi, n in enumerate(names):
            for bi, (s, _) in enumerate(SETS):
                r = M[(M.classifier == n) & (M.knob == k) & (M.base == s)]
                if not len(r): continue
                r = r.iloc[0]; y = yi + (0.15 if bi else -0.15)
                ax.plot([r['lo'], r['hi']], [y, y], color=COL[r['group']], lw=1)
                ax.plot(r['dlogit'], y, 'o' if bi else 's', ms=3, color=COL[r['group']])
        ax.axvline(0, color='k', lw=0.5); ax.set_title({'PN': 'Pigment network', 'VS': 'Vascular structures', 'Pig': 'Pigmentation', 'Reg': 'Regression'}[k], fontsize=8)
        ax.set_xlabel('Δ logit, level 0 to 1', fontsize=7); ax.spines[['top', 'right']].set_visible(False)
    axs[0].set_yticks(range(len(names))); axs[0].set_yticklabels([n.replace('mel_', '') for n in names], fontsize=6)
    axs[0].set_yticklabels([n.replace('mel_', '') for n in names], fontsize=6)
    axs[0].text(0, 1.08, 'squares: real benign base; circles: real melanoma base; orange: primary six; blue: new seeds; green: HAM10000',
                transform=axs[0].transAxes, fontsize=6.5)
    out.fig(f, 'F2_retraining')


# ----------------------------------------------------------------------------- generic null
def add_generic(cfg, models, L, sdlog_extra):
    """random linear heads on ImageNet features, saved by worker3 job 'generic' (one npz per arch: key set -> (H, N))"""
    gd = cfg.get('generic_dir')
    if not gd or not os.path.isdir(gd): return []
    new = []
    for f in sorted(glob.glob(os.path.join(gd, 'generic_*.npz'))):
        arch = os.path.basename(f)[8:-4]; z = np.load(f); H = z['cls_test'].shape[0]
        for h in range(H):
            name = f'generic_{arch}_h{h:03d}'
            L[name] = {k: z[k][h].astype(np.float64) for k in z.files}
            new.append({'name': name, 'arch': arch, 'label': 'generic', 'seed': h, 'data': 'imagenet', 'group': 'null_generic'})
    return new


def main(cfg_path):
    cfg = json.load(open(cfg_path)); out = Out(cfg['out']); rng = np.random.default_rng(2033)
    REP.append('# ClinDiffXAI Phase 3 report: empirical null distribution and retraining variability\n')
    REP.append('Rules fixed before the run (SHA-256 of the text: %s):\n\n%s\n' % (
        hashlib.sha256(cfg['rule_text'].encode()).hexdigest(), cfg['rule_text']))
    models = []; L = {}
    for m in cfg['models']:
        p = os.path.join(cfg['scores_dir'], f"{m['name']}.npz")
        if os.path.exists(p): models.append(m); L[m['name']] = logits(p)
        else: log(f"WARNING: no scores for {m['name']}")
    metas = {s: load_meta(cfg, s) for s, _ in SETS}; evs = {s: load_eval(cfg, s) for s, _ in SETS}
    sdlog = classifier_table(cfg, out, models, L, rng)
    gen = add_generic(cfg, models, L, sdlog)
    for g in gen: sdlog[g['name']] = float(np.nanstd(L[g['name']]['cls_test']))
    log(f'{len(models)} classifiers, {len(gen)} random heads')
    E = effects_table(cfg, out, models + gen, L, sdlog, metas, rng)
    nul, per = null_summary(cfg, out, E)
    fpr = fpr_table(cfg, out, E)
    rt, ok6, n6, npig = retrain_table(cfg, out, E)
    prof = profile_table(cfg, out, per)
    obs = observational_null(cfg, out, models, L, sdlog)
    pres = preservation_table(cfg, out, models, L, metas, evs, rng)
    ham, ok7 = ham_table(cfg, out, E)
    ph2 = ph2_tables(cfg, out); ham_reports(cfg, out)
    fig_null(out, per, nul); fig_retrain(out, E)
    verdict5 = {r['knob']: r['above null (rule 5)'] for _, r in nul.iterrows() if r['target-dominant']}
    res = {'rule_text': cfg['rule_text'], 'rule_sha256': hashlib.sha256(cfg['rule_text'].encode()).hexdigest(),
           'rule5_above_null': verdict5, 'rule6_stable': ok6, 'rule6_n_classifiers': n6, 'rule6_n_pig_largest': npig,
           'n_null_trained': int((per.group == 'null_trained').sum() / 7), 'n_null_generic': len(gen),
           'n_mel': int(per.group.isin(['mel_primary', 'mel_retrain']).sum() / 7),
           'rule7_transfers_to_ham10000': ok7, 'n_ham': int((per.group == 'mel_ham').sum() / 7),
           'ph2_status': (ph2 or {}).get('status', 'not attached')}
    json.dump(res, open(os.path.join(out.root, 'results', 'phase3_results.json'), 'w'), indent=2, default=str)
    REP.insert(2, '\n## Verdicts\n\n' + '\n'.join([f'- Rule 5, {k}: ' + ('above null' if v else 'NOT above null') for k, v in verdict5.items()]
                                                + [f"- Rule 6 (retraining): {'stable' if ok6 else 'NOT stable'} "
                                                   f"(pigmentation largest of 4 knobs in {npig} of {n6} classifiers, descriptive)",
                                                   '- Rule 7 (HAM10000): ' + ('not run' if ok7 is None else ('transfers' if ok7 else 'does NOT transfer')),
                                                   '- Rule 8 (PH2, descriptive): ' + str((ph2 or {}).get('status', 'not attached'))]))
    open(os.path.join(out.root, 'phase3_report.md'), 'w').write('\n'.join(REP))
    log('analysis3 finished')


if __name__ == '__main__':
    main(sys.argv[1])
