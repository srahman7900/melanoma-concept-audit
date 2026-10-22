"""Integrate the Phase 2e outputs (matched controls labelled by CBM-B) into the paper.
Usage: python integrate_2e.py /path/to/phase2e_outputs.zip      (or without argument when p2eout/ is already extracted)
Writes numbers/numbers_2e.json, numbers/numbers_2e.md and tables/T2e_*.csv, and prints the decision per concept."""
import os, sys, json, zipfile, shutil, hashlib
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pairs_cbma import CN, pairs_cbma, smd, quartile_groups

H = '/home/claude'; OUT = os.path.join(H, 'paper'); P2D = os.path.join(H, 'p2dout'); AI = os.path.join(H, 'p2ainp')
P2E = os.environ.get('P2E_DIR', os.path.join(H, 'p2eout'))
SH = ['PN', 'BWV', 'VS', 'Pig', 'Str', 'D/G', 'Reg']
MB = {'bpn': 'pigment_network', 'bvs': 'vascular_structures', 'bpig': 'pigmentation', 'breg': 'regression_structures'}
CONC = list(MB.values())
EXPECTED_SHA = '919a511cf55ae6f5b249ba11c465ea25a460529fb775e49913c4d5a360ccf880'
SRC = 'phase2e_outputs.zip'
N = {}


def put(key, value, source, note=''):
    N[key] = {'value': value, 'source': source, 'note': note}


def f(*p):
    return os.path.join(P2E, *p)


def r3(x):
    return None if x is None or not np.isfinite(x) else round(float(x), 3)


if len(sys.argv) > 1:
    shutil.rmtree(P2E, ignore_errors=True); os.makedirs(P2E)
    with zipfile.ZipFile(sys.argv[1]) as z:
        z.extractall(P2E)
os.makedirs(os.path.join(OUT, 'tables'), exist_ok=True); os.makedirs(os.path.join(OUT, 'numbers'), exist_ok=True)

# ------------------------------------------------------------------ 1. rule and its fingerprint
txt = open(f('results', 'rule_2e.txt')).read()
rule = txt.split('\n\nSHA-256:')[0]; sha_file = txt.split('SHA-256:')[1].split()[0]
sha_now = hashlib.sha256(rule.encode('utf8')).hexdigest()
printed = txt.split('Printed at (UTC):')[1].strip() if 'Printed at (UTC):' in txt else ''
assert sha_now == sha_file, 'rule text does not match its printed SHA-256'
put('p2e_rule_sha256', sha_now, f'{SRC}: results/rule_2e.txt', 'equal to the fingerprint of the delivered cell' if sha_now == EXPECTED_SHA
    else 'DIFFERENT from the fingerprint of the delivered cell: check which cell was run')
put('p2e_rule_printed_utc', printed, f'{SRC}: results/rule_2e.txt')

# ------------------------------------------------------------------ 2. design (fixed before training)
D = {r['concept']: r for r in json.load(open(f('results', 'design_cbmb.json')))['design']}
rows = []
for c in CONC:
    r = D[c]; j = CN.index(c)
    rows.append({'Concept': SH[j], 'Low / high threshold (CBM-B)': f"{r['low_threshold']:.3f} / {r['high_threshold']:.3f}",
                 'Caliper': r['caliper'], 'Pairs train / val / test': f"{r['pairs_train']} / {r['pairs_val']} / {r['pairs_test']}",
                 'Melanoma pairs (train)': r['melanoma_pairs_train'], 'Testable': r['testable'],
                 'Target SMD (CBM-B)': r3(r['target_smd_cbmb']), 'Max |SMD| other six (CBM-B)': r3(r['max_abs_other_smd_cbmb']),
                 'Target SMD (CBM-A)': r3(r['target_smd_cbma']), 'Max |SMD| other six (CBM-A)': r3(r['max_abs_other_smd_cbma'])})
design = pd.DataFrame(rows)
put('p2e_design', design.to_dict('list'), f'{SRC}: results/design_cbmb.json')

# ------------------------------------------------------------------ 3. verdict per control and per concept
J = json.load(open(f('audit', 'results', 'matched_controls_cbmb.json')))
ctr = J['controls']; con = J['concepts']
tp10 = pd.read_csv(f('audit', 'tables', 'TP10_matched_controls_cbmb.csv'))
e2 = pd.read_csv(f('audit', 'tables', 'TP2_counterfactual_effects_all.csv'))
real2 = e2[e2.base.isin(['real_ben', 'real_mel'])]
crow = []
for k, v in sorted(ctr.items(), key=lambda kv: (CONC.index(kv[1]['concept']), kv[1]['arch'] != 'resnet50', kv[1]['seed'])):
    j = CN.index(v['concept']); t = real2[(real2.classifier == k) & (real2.concept == v['concept'])]
    tj = os.path.join(P2E, 'results', f'train_{k}.json'); tr = json.load(open(tj)) if os.path.exists(tj) else {}
    crow.append({'Control': k, 'Concept': SH[j], 'Architecture': {'resnet50': 'ResNet-50', 'vitb16': 'ViT-B/16'}.get(v['arch'], v['arch']),
                 'Seed': v['seed'], 'Trained': 'note' not in v, 'Training images': tr.get('n_train'),
                 'Own-label test AUC': r3(tr.get('test_auc_own_label', np.nan)) if tr else None,
                 'Bases passed (of 4)': v['bases_passed'], 'Pass': v['pass'],
                 'Target Δlogit, realistic bases': r3(t.dlogit.mean()) if len(t) else None,
                 'Observational largest': v.get('observational_largest'), 'Note': v.get('note', '')})
controls = pd.DataFrame(crow)
gng = []
for c in CONC:
    v = con.get(c, {'controls': 0, 'passed': 0, 'needed': 3, 'go': False, 'mean_rho_between_controls': None})
    gng.append({'Concept': SH[CN.index(c)], 'Testable': bool(D[c]['testable']), 'Controls trained': int(controls[(controls.Concept == SH[CN.index(c)]) & controls.Trained].shape[0]),
                'Controls passed': v['passed'], 'Needed': v['needed'], 'Decision': 'GO' if v['go'] else 'NO-GO',
                'Mean ρ between controls': r3(v['mean_rho_between_controls']) if v['mean_rho_between_controls'] is not None else None})
gng = pd.DataFrame(gng)
put('p2e_verdict', J.get('verdict'), f'{SRC}: audit/results/matched_controls_cbmb.json')
put('p2e_go_no_go', gng.to_dict('list'), 'same')
put('p2e_controls', controls.to_dict('list'), f'{SRC}: matched_controls_cbmb.json, TP2_counterfactual_effects_all.csv, results/train_*.json')
tr_ = controls[controls.Trained]
if len(tr_):
    put('p2e_own_auc_range', [r3(tr_['Own-label test AUC'].min()), r3(tr_['Own-label test AUC'].max())], 'same')
    put('p2e_target_dlogit_range', [r3(tr_['Target Δlogit, realistic bases'].min()), r3(tr_['Target Δlogit, realistic bases'].max())], 'same')
    put('p2e_n_pass', int(tr_.Pass.sum()), 'same', f'of {len(controls)} planned controls ({len(tr_)} trained)')

# ------------------------------------------------------------------ 4. the 10 earlier classifiers: same results as Phase 2d?
e1 = pd.read_csv(os.path.join(P2D, 'audit', 'tables', 'TP2_counterfactual_effects_all.csv'))
m = e1.merge(e2, on=['classifier', 'base', 'concept'], suffixes=('_2d', '_2e'))
if len(m):
    put('p2e_repro_vs_2d', {'classifiers': int(m.classifier.nunique()), 'effects': int(len(m)),
                            'max_abs_dlogit_diff': float((m.dlogit_2d - m.dlogit_2e).abs().max()),
                            'max_abs_p_bh_diff': float((m.p_bh_2d - m.p_bh_2e).abs().max())},
        f'{SRC} and phase2d_outputs.zip: TP2_counterfactual_effects_all.csv')

# ------------------------------------------------------------------ 5. balance of every design, read with both label sets
z = np.load(os.path.join(AI, 'phase2', 'cls', 'cls_labels.npz'), allow_pickle=True)
Ca = z['concepts'].astype(np.float64); y = z['target'].astype(int); split = z['split']; trm = split == 'train'
zb = np.load(f('results', 'cbmb_labels.npz'), allow_pickle=True)
same_order = len(zb['image_name']) == len(z['image_name']) and bool(np.all(zb['image_name'].astype(str) == z['image_name'].astype(str)))
xrows = []
if same_order:
    Cb = zb['cbmb'].astype(np.float64)
    put('p2e_cbmb_label_means', [r3(v) for v in Cb.mean(0)], f'{SRC}: results/cbmb_labels.npz')
    put('p2e_cbma_cbmb_spearman', [r3(pd.Series(Ca[:, k]).corr(pd.Series(Cb[:, k]), method='spearman')) for k in range(7)],
        'same, with phase2_analysis_inputs.zip: phase2/cls/cls_labels.npz', 'per concept, all 6,424 classifier images')

    def add(design_name, c, hi_idx, lo_idx):
        j = CN.index(c)
        for lab, C in [('CBM-A', Ca), ('CBM-B', Cb)]:
            sd = C[trm].std(0) + 1e-9; s = (C[hi_idx].mean(0) - C[lo_idx].mean(0)) / sd
            xrows.append({'Design': design_name, 'Concept': SH[j], 'Read with': lab, 'Pairs or images': f'{len(hi_idx)} / {len(lo_idx)}',
                          'Target SMD': r3(s[j]), 'Max |SMD| other six': r3(np.abs(np.delete(s, j)).max()),
                          'Worst other concept': [SH[x] for x in range(7) if x != j][int(np.argmax(np.abs(np.delete(s, j))))],
                          **{f'SMD {SH[x]}': r3(s[x]) for x in range(7)}})
    for c in ['pigmentation', 'blue_whitish_veil']:
        hi_i, lo_i = quartile_groups(Ca, split, CN.index(c)); add('Quartile groups (CBM-A)', c, hi_i, lo_i)
    for c in CONC:
        pr, _ = pairs_cbma(Ca, y, split, c); p = pr['train']; add('Matched pairs (CBM-A)', c, p[:, 0], p[:, 1])
    # the Phase 2e pairs, rebuilt with the rule of worker2.matched_groups(source='cbmb')
    from pairs_cbma import greedy_pairs
    MBc = {'q': 0.25, 'min_high': 0.5, 'caliper': 0.5, 'fallback_caliper': 1.0, 'min_pairs': 100, 'seed': 3029}
    for c in CONC:
        j = CN.index(c); lo, hi = np.quantile(Cb[trm, j], MBc['q']), np.quantile(Cb[trm, j], 1 - MBc['q']); hi = max(float(hi), MBc['min_high'])
        oth = [k for k in range(7) if k != j]; mu = Cb[trm][:, oth].mean(0); sd = Cb[trm][:, oth].std(0) + 1e-9; Z = (Cb[:, oth] - mu) / sd
        cal = MBc['caliper']; pr = greedy_pairs(Z, y, split, Cb[:, j] >= hi, Cb[:, j] <= lo, cal, MBc['seed'] + j)
        if len(pr['train']) < MBc['min_pairs']:
            cal = MBc['fallback_caliper']; pr = greedy_pairs(Z, y, split, Cb[:, j] >= hi, Cb[:, j] <= lo, cal, MBc['seed'] + j)
        p = pr['train']
        assert len(p) == D[c]['pairs_train'], f'{c}: rebuilt {len(p)} pairs, the run had {D[c]["pairs_train"]}'
        if len(p):
            add('Matched pairs (CBM-B)', c, p[:, 0], p[:, 1])
    xb = pd.DataFrame(xrows)
    put('p2e_cross_label_balance', xb[['Design', 'Concept', 'Read with', 'Target SMD', 'Max |SMD| other six', 'Worst other concept']].to_dict('list'),
        f'{SRC}: results/cbmb_labels.npz; pairs rebuilt offline (pairs_cbma.py), identical to the run')
else:
    xb = pd.DataFrame()
    print('WARNING: cbmb_labels.npz is not in the order of cls_labels.npz; cross-label balance skipped')

# ------------------------------------------------------------------ 6. calibration of observation with the CBM-B controls (exploratory check)
o5 = pd.read_csv(f('audit', 'tables', 'TP5_observational_attribution.csv'))
o5d = pd.read_csv(os.path.join(P2D, 'audit', 'tables', 'TP5_observational_attribution.csv'))
mm = o5.merge(o5d, on='Classifier', suffixes=('_2e', '_2d'))
put('p2e_obs_repro_vs_2d', float(max((mm[c + '_2e'] - mm[c + '_2d']).abs().max() for c in o5.columns if c != 'Classifier')),
    'TP5_observational_attribution.csv of Phase 2d and 2e', f'{len(mm)} classifiers in both')
AR = {'resnet50': 'ResNet-50', 'vitb16': 'ViT-B/16', 'effb0': 'EfficientNet-B0'}
NM = {'pigment_network': 'pigment network', 'vascular_structures': 'vascular structures', 'pigmentation': 'pigmentation'}
cls2 = e2.drop_duplicates('classifier')[['classifier', 'label', 'arch', 'seed', 'data']]
iv2 = real2.groupby(['classifier', 'concept']).dlogit.mean()
Kb, Kb_all = {}, {}
for lab, c in [('bpn', 'pigment_network'), ('bvs', 'vascular_structures'), ('bpig', 'pigmentation')]:
    s_ = SH[CN.index(c)]; ks = []
    for _, r in cls2[cls2.label == lab].iterrows():
        nm = f"Control+ CBM-B: {NM[c]} ({AR[r['arch']]} s{r['seed']})"
        ks.append(iv2[(r['classifier'], c)] / o5.loc[o5.Classifier == nm, f'CBM-B {s_}'].values[0])
    Kb[s_] = float(np.median(ks)); Kb_all[s_] = [r3(x) for x in ks]
rows6 = []
for _, r in cls2[(cls2.label == 'mel') & (cls2.data == 'isic2020')].iterrows():
    nm = f"{AR[r['arch']]} s{r['seed']}"; row = {'Classifier': nm}
    for s_, c in [('PN', 'pigment_network'), ('VS', 'vascular_structures'), ('Pig', 'pigmentation')]:
        ob = o5.loc[o5.Classifier == nm, f'CBM-B {s_}'].values[0]
        row[f'{s_} expected (CBM-B ratio)'] = r3(Kb[s_] * ob); row[f'{s_} measured'] = r3(iv2[(r['classifier'], c)])
    rows6.append(row)
cal_b = pd.DataFrame(rows6)
summ = {}
for s_ in ['PN', 'VS', 'Pig']:
    d_ = cal_b[f'{s_} expected (CBM-B ratio)'] - cal_b[f'{s_} measured']
    summ[s_] = {'median_ratio': r3(Kb[s_]), 'over_predicted': int((d_ > 0).sum()), 'under_predicted': int((d_ < 0).sum()),
                'expected_range': [r3(cal_b[f'{s_} expected (CBM-B ratio)'].min()), r3(cal_b[f'{s_} expected (CBM-B ratio)'].max())]}
put('p2e_calibration_cbmb', {'summary': summ, 'per_control_ratios': Kb_all}, 'computed from TP2 and TP5 (Phase 2e); exploratory check of the CBM-A calibration')
cal_b.to_csv(os.path.join(OUT, 'tables', 'T2e_calibration_cbmb.csv'), index=False)

# ------------------------------------------------------------------ write
design.to_csv(os.path.join(OUT, 'tables', 'T2e_design_cbmb.csv'), index=False)
controls.to_csv(os.path.join(OUT, 'tables', 'T2e_matched_controls_cbmb.csv'), index=False)
gng.to_csv(os.path.join(OUT, 'tables', 'T2e_go_no_go.csv'), index=False)
if len(xb): xb.to_csv(os.path.join(OUT, 'tables', 'T2e_cross_label_balance.csv'), index=False)
json.dump(N, open(os.path.join(OUT, 'numbers', 'numbers_2e.json'), 'w'), indent=1, default=str)
with open(os.path.join(OUT, 'numbers', 'numbers_2e.md'), 'w') as fh:
    fh.write('| Key | Value | Source | Note |\n|---|---|---|---|\n')
    for k, v in N.items():
        fh.write(f"| {k} | {json.dumps(v['value'], default=str, ensure_ascii=False)[:400]} | {v['source']} | {v['note']} |\n")

pd.set_option('display.width', 220); pd.set_option('display.max_columns', 30)
print('Rule SHA-256:', sha_now, '(matches delivered cell)' if sha_now == EXPECTED_SHA else '(DOES NOT match delivered cell)')
print('\nDesign (fixed before training):\n', design.to_string(index=False))
print('\nControls:\n', controls.drop(columns=['Note']).to_string(index=False))
print('\nDecision per concept:\n', gng.to_string(index=False))
if 'p2e_repro_vs_2d' in N: print('\nPhase 2d classifiers re-scored in 2e:', N['p2e_repro_vs_2d']['value'])
print('Observational coefficients, largest difference to 2d:', N['p2e_obs_repro_vs_2d']['value'])
print('\nCalibration with the CBM-B controls (exploratory):', json.dumps(N['p2e_calibration_cbmb']['value']['summary']))
print(cal_b.to_string(index=False))
if len(xb): print('\nBalance of each design, read with both label sets:\n',
                  xb[['Design', 'Concept', 'Read with', 'Pairs or images', 'Target SMD', 'Max |SMD| other six', 'Worst other concept']].to_string(index=False))
go = gng[gng.Decision == 'GO'].Concept.tolist()
print(f"\nSCENARIO: {'ALL GO' if len(go) == 4 else ('NONE GO' if not go else 'PARTIAL GO: ' + ', '.join(go))}")
