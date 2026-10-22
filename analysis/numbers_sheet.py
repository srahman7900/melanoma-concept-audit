"""ClinDiffXAI numbers sheet: every value used in the manuscript, read from the run outputs, with its source.
Writes numbers/numbers.json, numbers/numbers.md and the new offline tables in tables/.
Inputs (unpacked run outputs): p1bout (Phase 1b), p1cout (1c), p1dout (1d), p2dout (2d: audit and TCAV of all 26
classifiers), p2cout (2c), p2ainp (phase2_analysis_inputs.zip)."""
import os, json
import numpy as np
import pandas as pd
from scipy import stats

H = '/home/claude'
P1B, P1C, P1D, P2C, P2D, AI = [os.path.join(H, d) for d in ['p1bout', 'p1cout', 'p1dout', 'p2cout', 'p2dout', 'p2ainp']]
OUT = os.path.join(H, 'paper')
CN = ['pigment_network', 'blue_whitish_veil', 'vascular_structures', 'pigmentation', 'streaks', 'dots_and_globules', 'regression_structures']
SH = ['PN', 'BWV', 'VS', 'Pig', 'Str', 'D/G', 'Reg']
N = {}


def put(key, value, source, note=''):
    if isinstance(value, (np.floating, np.integer)): value = value.item()
    N[key] = {'value': value, 'source': source, 'note': note}


def rd(x, k=3):
    return None if x is None or (isinstance(x, float) and not np.isfinite(x)) else round(float(x), k)


# ------------------------------------------------------------------ data
pl = pd.read_csv(os.path.join(AI, 'phase2/pseudo_labels_slim.csv'))
put('isic2020_images', int(len(pl)), 'phase2_analysis_inputs.zip: phase2/pseudo_labels_slim.csv')
put('isic2020_melanoma', int(pl.target.sum()), 'same')
cd = json.load(open(os.path.join(P2D, 'audit/results/cls_data.json')))
put('cls_images', cd['n'], 'phase2d_outputs.zip: audit/results/cls_data.json')
put('cls_melanoma', cd['n_mel'], 'same')
put('cls_split_train_val_test', cd['split_counts'], 'same')
put('cls_split_melanoma', cd['split_mel'], 'same')
put('cls_split_by', 'patient' if cd['have_patient_ids'] else 'image', 'same')
put('isic2020_association_std_coef', {SH[CN.index(k)]: rd(v) for k, v in cd['association_std_coef'].items()}, 'same',
    'balanced logistic regression of melanoma on standardised pseudo-labels, all ISIC 2020 images')
plan = json.load(open(os.path.join(P2D, 'results/plan.json')))
put('tcav_targets', {'melanoma': len(plan['pos']['tgt_mel']), 'benign': len(plan['pos']['tgt_ben'])}, 'phase2d_outputs.zip: results/plan.json')
put('derm7pt_images_tcav', len(plan['pos'].get('derm', [])), 'same')

# ------------------------------------------------------------------ pseudo-label diagnostics (offline)
rows = []
for c, s in zip(CN, SH):
    v = pl[f'concept_{c}'].values; m = pl.target.values == 1
    rows.append({'Concept': s, 'Mean': rd(v.mean(), 4), 'Median': rd(np.median(v), 4), 'P75': rd(np.quantile(v, .75), 4),
                 'P90': rd(np.quantile(v, .9), 4), 'P99': rd(np.quantile(v, .99), 4), 'n >= 0.5': int((v >= 0.5).sum()),
                 'Melanoma P95': rd(np.quantile(v[m], .95), 4)})
lab_diag = pd.DataFrame(rows); lab_diag.to_csv(os.path.join(OUT, 'tables/T_pseudo_label_distribution.csv'), index=False)
put('pseudo_label_distribution', lab_diag.to_dict(orient='records'), 'computed from pseudo_labels_slim.csv (tables/T_pseudo_label_distribution.csv)')
b = pl.concept_blue_whitish_veil.values
put('bwv_prototype_value', 0.7781, 'common.py MEL_VEC_LIST (melanoma prototype vector)')
put('bwv_share_below_prototype', rd((b < 0.7781).mean(), 5), 'computed from pseudo_labels_slim.csv')
put('bwv_n_at_or_above_prototype', int((b >= 0.7781).sum()), 'same')
z = np.load(os.path.join(AI, 'phase2/cls/cls_labels.npz'))
split, Cc, yy = z['split'], z['concepts'].astype(float), z['target'].astype(int)
tr = split == 'train'; sd_tr = Cc[tr].std(0)
ctrl = {}
for lab, j in [('bwv', 1), ('pig', 3)]:
    v = Cc[tr, j]; lo, hi = np.quantile(v, .25), np.quantile(v, .75); sel = (v <= lo) | (v >= hi); g = (v[sel] >= hi)
    X = Cc[tr][sel]; smd = (X[g].mean(0) - X[~g].mean(0)) / sd_tr
    ctrl[lab] = {'n': int(sel.sum()), 'mean_target_high': rd(X[g, j].mean(), 4), 'mean_target_low': rd(X[~g, j].mean(), 4),
                 'smd': {s: rd(x, 2) for s, x in zip(SH, smd)}}
put('original_controls_group_smd', ctrl, 'computed from cls_labels.npz (training split; quartile groups as in Phase 2)')

# ------------------------------------------------------------------ evaluators and reproduction (Phase 1, 1b)
t1 = pd.read_csv(os.path.join(P1D, 'tables/T1_evaluator_validity.csv'))
put('evaluator_auc_128', t1[['Concept', 'CBM-A 128', 'CBM-B 128', 'ISIC18 128 (attr)']].to_dict(orient='records'),
    'phase1d_outputs.zip: tables/T1_evaluator_validity.csv (copied from Phase 1b)')
put('cbma_derm7pt_native', {'mean_concept_auc': 0.8224, 'diagnosis_auc': 0.5425}, 'phase1_report.md section 1 (Phase 1b)')
put('reproduction', {'wilcoxon_p': 0.0009194, 'r_rb': 0.254, 'max_abs_delta_pearson_r': 0.000}, 'phase1_report.md section 2 (Phase 1b)')
put('isic18_overlap_with_isic2020', 0, 'phase1_report.md section 1 (Phase 1b)')

# ------------------------------------------------------------------ colour (1b, 1c, 1d)
put('colour_outliers_V0', {'real_mel': 2.4, 'real_ben': 0.6, 'main_mel': 77.8, 'main_ben': 83.7, 'base_mel': 78.5, 'base_ben': 80.7},
    'Phase 1b phase1_report.md section 9', '% of images outside the 99% colour region of real images')
put('noise_mean_to_colour_cv_r2', 0.90, 'phase1c_report.md Part A')
put('colour_outliers_V1', [74.2, 80.9, 75.0, 75.8], 'phase1c_report.md Part B (main mel, main ben, base mel, base ben)')
put('colour_outliers_V2_phase1c', [6.6, 7.4, 13.3, 7.0], 'same')
c1d = pd.read_csv(os.path.join(P1D, 'tables/TD1_colour_fidelity_all_sets.csv'))
ev = c1d[c1d['Set'].str.startswith('eval_')]
put('colour_outliers_V2_eval_sets', dict(zip(ev['Set'], ev['Colour outliers %'])), 'phase1d_outputs.zip: tables/TD1_colour_fidelity_all_sets.csv')
tg = pd.read_csv(os.path.join(P1D, 'tables/TD2_colour_targeting.csv'))
put('colour_targeting_r2_range', {'R': [rd(tg['R2 R'].min(), 2), rd(tg['R2 R'].max(), 2)], 'G': [rd(tg['R2 G'].min(), 2), rd(tg['R2 G'].max(), 2)],
                                  'B': [rd(tg['R2 B'].min(), 2), rd(tg['R2 B'].max(), 2)]}, 'tables/TD2_colour_targeting.csv (Phase 1d)')
cs = c1d[c1d['Set'].str.startswith('cfg_sweep')]
put('cfg_sweep_mel_outliers', {r['Set'].split('=')[1]: r['Colour outliers %'] for _, r in cs[cs['Class'] == 'mel'].iterrows()}, 'TD1 (Phase 1d)')

# ------------------------------------------------------------------ Phase 1d generator results
r1d = json.load(open(os.path.join(P1D, 'results/phase1_results.json')))
put('gate1_V2', 'PASS', 'phase1d_full_analysis_report.md')
dose = pd.read_csv(os.path.join(P1D, 'tables/T3_dose_response_mel_base.csv'))
dben = pd.read_csv(os.path.join(P1D, 'tables/T3_dose_response_ben_base.csv'))


def dr(df, ev, sw, out_):
    r = df[(df['Evaluator'] == ev) & (df['Swept concept'] == sw) & (df['Output'] == out_)]
    return None if not len(r) else {'delta_sd': r['Δ 0→1 in real-SD units [95% CI]'].values[0], 'rising': r['Traj. with slope > 0'].values[0], 'p_bh': r['p (BH)'].values[0]}


put('dose_mel_base', {'CBM-B Pig': dr(dose, 'CBM-B', 'pigmentation', 'pigmentation'), 'CBM-B PN': dr(dose, 'CBM-B', 'pigment_network', 'pigment_network'),
                      'Darkness Pig': dr(dose, 'Lesion darkness', 'pigmentation', 'lesion_darkness'),
                      'ISIC18 PN': dr(dose, 'ISIC-2018 detector', 'pigment_network', 'pigment_network')}, 'tables/T3_dose_response_mel_base.csv (Phase 1d)')
put('dose_ben_base', {'CBM-B PN': dr(dben, 'CBM-B', 'pigment_network', 'pigment_network'), 'ISIC18 PN': dr(dben, 'ISIC-2018 detector', 'pigment_network', 'pigment_network'),
                      'CBM-B Pig': dr(dben, 'CBM-B', 'pigmentation', 'pigmentation'), 'CBM-B BWV': dr(dben, 'CBM-B', 'blue_whitish_veil', 'blue_whitish_veil'),
                      'CBM-A BWV': dr(dben, 'CBM-A', 'blue_whitish_veil', 'blue_whitish_veil'), 'CBM-B VS': dr(dben, 'CBM-B', 'vascular_structures', 'vascular_structures'),
                      'CBM-B Reg': dr(dben, 'CBM-B', 'regression_structures', 'regression_structures')}, 'tables/T3_dose_response_ben_base.csv (Phase 1d)')
dist = pd.read_csv(os.path.join(P1D, 'tables/T2d_distribution_distances.csv'))
put('distribution_distances_V2', dist.to_dict(orient='records'), 'tables/T2d_distribution_distances.csv (Phase 1d)')
put('class_signal_colour_control', pd.read_csv(os.path.join(P1D, 'tables/TD3_class_signal_colour_control.csv')).to_dict(orient='records'),
    'tables/TD3_class_signal_colour_control.csv (Phase 1d)')
put('diversity_V2', pd.read_csv(os.path.join(P1D, 'tables/TD4_diversity.csv')).to_dict(orient='records'), 'tables/TD4_diversity.csv (Phase 1d)')
put('memorization_screen', r1d['memo'], 'phase1d_outputs.zip: results/phase1_results.json (memo)')

# ------------------------------------------------------------------ knob specificity and cross-talk (offline, from the archive)
ref = np.load(os.path.join(AI, 'phase2/scores/B_cls_test.npz')); sdB, sdI = ref['cbmb'].std(0), ref['isic18'].std(0)
bases = [('Real benign case', 'phase2', 'rsweep_ben'), ('Real melanoma case', 'phase2', 'rsweep_mel'), ('Benign prototype', 'phase1d', 'sweep_ben'), ('Melanoma prototype', 'phase1d', 'sweep_mel')]
D = {}
for blab, ph, s in bases:
    m = json.load(open(os.path.join(AI, f'{ph}/sets_meta/{s}.json')))['meta']; ci, lv, nid = (np.array(m[k]) for k in ['concept_idx', 'level', 'noise_id'])
    sc = np.load(os.path.join(AI, f'{ph}/scores/B_{s}.npz')); out = {'CBM-B': sc['cbmb'] / sdB, 'ISIC18': sc['isic18'] / sdI}
    for evn, X in out.items():
        M = np.zeros((7, X.shape[1]))
        for j in range(7):
            a_, b_ = (ci == j) & (lv == lv.min()), (ci == j) & (lv == lv.max())
            M[j] = (X[b_][np.argsort(nid[b_])] - X[a_][np.argsort(nid[a_])]).mean(0)
        D[(blab, evn)] = M
spec_rows = []
for evn, o, k, oname in [('CBM-B', j, j, SH[j]) for j in range(7)] + [('ISIC18', 0, 0, 'PN (ISIC 2018)'), ('ISIC18', 2, 4, 'Streaks (ISIC 2018)'), ('ISIC18', 4, 5, 'Globules (ISIC 2018)')]:
    r = {'Evaluator output': f'{evn}: {oname}', 'Knob': SH[k]}; npass = 0
    for blab, _, _ in bases:
        col = D[(blab, evn)][:, o]; t = col[k]; oth = np.delete(col, k); jo = [x for x in range(7) if x != k][int(np.argmax(oth))]
        ok = bool(t > oth.max() and t > 0); npass += ok
        r[blab] = f"{t:+.2f} ({'largest' if ok else 'largest: ' + SH[jo] + f' {oth.max():+.2f}'})"
    r['Bases where the knob is largest'] = f'{npass}/4'; spec_rows.append(r)
spec = pd.DataFrame(spec_rows); spec.to_csv(os.path.join(OUT, 'tables/T_knob_specificity.csv'), index=False)
put('knob_specificity', spec[['Evaluator output', 'Bases where the knob is largest']].to_dict(orient='records'),
    'computed from phase2_analysis_inputs.zip (B_*.npz, sets_meta) -> tables/T_knob_specificity.csv', 'mean change level 1 minus 0, in SD of real test images')
ct = []
for blab in ['Real benign case', 'Real melanoma case']:
    M = D[(blab, 'CBM-B')]
    for j in range(7):
        ct.append({'Base': blab, 'Knob': SH[j], **{f'CBM-B {s}': rd(M[j, k], 2) for k, s in enumerate(SH)}})
ctab = pd.DataFrame(ct); ctab.to_csv(os.path.join(OUT, 'tables/T_crosstalk_realistic_bases.csv'), index=False)
put('crosstalk_realistic', 'tables/T_crosstalk_realistic_bases.csv', 'computed from phase2_analysis_inputs.zip')
np.save(os.path.join(OUT, 'numbers/knob_matrices.npy'), {f'{a}|{b}': v for (a, b), v in D.items()}, allow_pickle=True)

# ------------------------------------------------------------------ classifiers (Phase 2d audit tables cover all 26)
t1c = pd.read_csv(os.path.join(P2D, 'audit/tables/TP1_classifiers.csv'))
put('classifiers', t1c.to_dict(orient='records'), 'phase2d_outputs.zip: audit/tables/TP1_classifiers.csv')
mel = t1c[t1c['Label'] == 'mel']
auc = mel[mel['Data'] == 'isic2020']['Test AUC, melanoma [95% CI]'].str.split().str[0].astype(float)
put('melanoma_test_auc_range', [auc.min(), auc.max()], 'TP1 (ISIC 2020 melanoma classifiers)')
t0 = pd.read_csv(os.path.join(P2D, 'audit/tables/TP0_numerical_check.csv'))
put('numerical_check', t0.to_dict(orient='records'), 'phase2d_outputs.zip: audit/tables/TP0_numerical_check.csv')

# ------------------------------------------------------------------ audit effects
e = pd.read_csv(os.path.join(P2D, 'audit/tables/TP2_counterfactual_effects_all.csv'))
real = e[e.base.isin(['real_ben', 'real_mel'])]
mm = real[real.label == 'mel']
summ = []
for c, s in zip(CN, SH):
    x = mm[mm.concept == c]; xn = x[x.arch != 'effb0']
    pc = x.groupby('classifier').dlogit.mean()
    summ.append({'Concept': s, 'Real benign: mean dlogit': rd(x[x.base == 'real_ben'].dlogit.mean(), 2), 'Real melanoma: mean dlogit': rd(x[x.base == 'real_mel'].dlogit.mean(), 2),
                 'Without EfficientNet (benign / melanoma)': f"{xn[xn.base == 'real_ben'].dlogit.mean():+.2f} / {xn[xn.base == 'real_mel'].dlogit.mean():+.2f}",
                 'Per-classifier mean over the two bases: min / max': f"{pc.min():+.2f} / {pc.max():+.2f}",
                 'Mean dp(melanoma)': rd(x.dprob.mean(), 3), 'Significant tests': f"{int(((x.p_bh < 0.05) & (x.median_slope > 0)).sum())}/{len(x)}"})
es = pd.DataFrame(summ); es.to_csv(os.path.join(OUT, 'tables/T_melanoma_effects_realistic.csv'), index=False)
put('melanoma_effects_realistic', es.to_dict(orient='records'), 'computed from audit/tables/TP2_counterfactual_effects_all.csv (Phase 2d) -> tables/T_melanoma_effects_realistic.csv')
rel = pd.read_csv(os.path.join(P2D, 'audit/tables/TP8_reliability.csv')).groupby('Pair')['Spearman ρ of effects'].mean()
put('audit_reliability', {k: rd(v) for k, v in rel.items()}, 'audit/tables/TP8_reliability.csv (Phase 2d)')
col = pd.read_csv(os.path.join(P2D, 'audit/tables/TP7_colour_sensitivity.csv'))
cm = col[col['Classifier'].str.match(r'^(ResNet|Efficient|ViT)') & ~col['Classifier'].str.contains('Derm7pt')]
put('colour_sensitivity_melanoma', {'mean_abs_dlogit_colour': [rd(cm['mean |Δ logit| from colour change'].min(), 2), rd(cm['mean |Δ logit| from colour change'].max(), 2)],
                                    'logit_per_10_Lstar': [rd(cm['logit per +10 L*'].min(), 2), rd(cm['logit per +10 L*'].max(), 2)]}, 'audit/tables/TP7_colour_sensitivity.csv (Phase 2d)')
dec = pd.read_csv(os.path.join(P2D, 'audit/tables/TP4_decomposition_concept_colour.csv'))
dm = dec[dec['Classifier'].str.match(r'^(ResNet|Efficient|ViT)') & ~dec['Classifier'].str.contains('Derm7pt')]
put('decomposition_r2_melanoma', [rd(dm['R2'].min(), 3), rd(dm['R2'].max(), 3)], 'audit/tables/TP4_decomposition_concept_colour.csv')
fl = pd.read_csv(os.path.join(P2D, 'audit/tables/TP3_decision_flips.csv'))
put('flips_base_rate_resnet_benign', fl[(fl['Classifier'].str.startswith('ResNet-50 s')) & ~fl['Classifier'].str.contains('Derm7pt') & (fl['Base'] == 'Real benign case')]['p(mel) call at level 0'].describe()[['min', 'max']].round(3).to_dict(),
    'audit/tables/TP3_decision_flips.csv', 'share of starting images already called melanoma at tau90')
prof = pd.read_csv(os.path.join(P2D, 'audit/tables/TP6_profile_agreement.csv')).groupby('Profile').mean(numeric_only=True).round(3)
put('profile_agreement', prof.to_dict(orient='index'), 'audit/tables/TP6_profile_agreement.csv')

# ------------------------------------------------------------------ controls
av = json.load(open(os.path.join(P2D, 'audit/results/audit_validity.json')))
put('original_rule_verdict', av, 'phase2d_outputs.zip: audit/results/audit_validity.json (identical to Phase 2)')
mc = json.load(open(os.path.join(P2D, 'audit/results/matched_controls.json')))
put('matched_concepts', mc['concepts'], 'audit/results/matched_controls.json')
put('matched_balance', mc['balance'], 'same')
put('matched_controls_audit', {k: {kk: v[kk] for kk in ['label', 'arch', 'seed', 'bases_passed', 'pass', 'observational_largest', 'decomposition_largest']} for k, v in mc['controls'].items()}, 'same')
mt = []
for c, s in [('pigment_network', 'PN'), ('vascular_structures', 'VS'), ('pigmentation', 'Pig'), ('regression_structures', 'Reg')]:
    lab = {'PN': 'mpn', 'VS': 'mvs', 'Pig': 'mpig', 'Reg': 'mreg'}[s]
    x = real[(real.label == lab) & (real.concept == c)]
    for arch in ['resnet50', 'vitb16']:
        y = x[x.arch == arch]
        mt.append({'Concept': s, 'Architecture': arch, 'Target dlogit min': rd(y.dlogit.min(), 2), 'Target dlogit max': rd(y.dlogit.max(), 2)})
put('matched_target_effects_realistic', mt, 'computed from TP2 (Phase 2d)')
trn = {}
for f in sorted(os.listdir(os.path.join(P2D, 'results'))):
    if f.startswith('train_ctrl_m'): d = json.load(open(os.path.join(P2D, 'results', f))); trn[f[6:-5]] = {'n_train': d['n_train'], 'own_test_auc': rd(d.get('test_auc_own_label'))}
for f in sorted(os.listdir(os.path.join(P2C, 'results'))):
    if f.startswith('train_ctrl_m'): d = json.load(open(os.path.join(P2C, 'results', f))); trn[f[6:-5]] = {'n_train': d['n_train'], 'own_test_auc': rd(d.get('test_auc_own_label'))}
put('matched_training', trn, 'results/train_ctrl_m*.json (Phase 2c and 2d)')

# ------------------------------------------------------------------ TCAV (ours) and Captum
tv = json.load(open(os.path.join(P2D, 'tcav/results/tcav_validity.json')))
put('tcav_rule_verdict', tv, 'phase2d_outputs.zip: tcav/results/tcav_validity.json (identical to Phase 2b)')
mct = json.load(open(os.path.join(P2D, 'tcav/results/matched_controls_tcav.json')))
put('tcav_matched', {k: (v['settings_passed'], v['pass']) for k, v in mct.items() if not k.startswith('concept:')}, 'tcav/results/matched_controls_tcav.json')
put('tcav_matched_concepts', {k: v for k, v in mct.items() if k.startswith('concept:')}, 'same')
tr6 = pd.read_csv(os.path.join(P2D, 'tcav/tables/TT6_reliability.csv')).groupby('Pair')['ρ, all settings'].mean()
put('tcav_reliability', {k: rd(v) for k, v in tr6.items()}, 'tcav/tables/TT6_reliability.csv')
cap = json.load(open(os.path.join(P2D, 'tcav/results/captum_results.json')))
put('captum', {'verdict': cap['verdict'], 'negative': cap['rule'].get('negative'), 'positive_pig': cap['rule'].get('positive_pig'),
               'positive_bwv': cap['rule'].get('positive_bwv'), 'matched': {k: (v['settings_passed'], v['pass']) for k, v in cap['matched'].items()},
               'rho_seeds': rd(np.nanmean(cap['rho_seeds'])), 'rho_cross': rd(np.nanmean(cap['rho_cross'])),
               'agreement_with_our_tcav': {k: rd(v, 2) for k, v in cap['agreement_with_our_tcav'].items()}}, 'tcav/results/captum_results.json')
cr = pd.read_csv(os.path.join(P2D, 'results/captum_runs.csv'))
put('captum_share_scores_0_or_1', rd(((cr.tcav_mel == 0) | (cr.tcav_mel == 1)).mean(), 3), 'results/captum_runs.csv')
tc = pd.read_csv(os.path.join(P2D, 'results/tcav_runs.csv')); tc2 = pd.read_csv(os.path.join(P2C, 'results/tcav_runs.csv'))
mg = tc.merge(tc2, on=['classifier', 'layer', 'variant', 'concept', 'run'], suffixes=('_d', '_c'))
put('tcav_reproduction', {'runs_compared': int(len(mg)), 'max_abs_diff_tcav_mel': float((mg.tcav_mel_d - mg.tcav_mel_c).abs().max())},
    'results/tcav_runs.csv of Phase 2d versus Phase 2c')
put('tcav_cav_accuracy', {'concept_mean': rd(tc[tc.concept != '_random'].cav_acc.mean()), 'random_mean': rd(tc[tc.concept == '_random'].cav_acc.mean())}, 'results/tcav_runs.csv (Phase 2d)')
o5 = pd.read_csv(os.path.join(P2D, 'audit/tables/TP5_observational_attribution.csv'))
neg_obs = o5[o5.Classifier == 'Control-: permuted labels'][[f'CBM-B {s}' for s in SH]].abs().max(axis=1).values[0]
mel_obs = o5[o5.Classifier.str.match(r'^(ResNet|Efficient|ViT)')][[f'CBM-B {s}' for s in SH]].abs().max(axis=1)
put('observational_negative_ratio', rd(neg_obs / np.median(mel_obs), 3), 'computed from TP5 (Phase 2d)')
obs_rel = {}
mlabs = [l for l in o5.Classifier if l.startswith(('ResNet', 'Efficient', 'ViT'))]
s_, c_ = [], []
for i in range(len(mlabs)):
    for k in range(i + 1, len(mlabs)):
        a = o5[o5.Classifier == mlabs[i]][[f'CBM-B {s}' for s in SH]].values[0]; bb = o5[o5.Classifier == mlabs[k]][[f'CBM-B {s}' for s in SH]].values[0]
        same = mlabs[i].rsplit(' s', 1)[0] == mlabs[k].rsplit(' s', 1)[0] and 'Derm7pt' not in mlabs[i] + mlabs[k]
        (s_ if same else c_).append(stats.spearmanr(a, bb)[0])
put('observational_reliability', {'same architecture, other seed': rd(np.mean(s_)), 'different classifier': rd(np.mean(c_))}, 'computed from TP5 (Phase 2d)')

# ------------------------------------------------------------------ calibrated observational versus interventional comparison (offline, exploratory)
ARCH = {'resnet50': 'ResNet-50', 'vitb16': 'ViT-B/16', 'effb0': 'EfficientNet-B0'}
MT = {'mpn': ('pigment_network', 'PN'), 'mvs': ('vascular_structures', 'VS'), 'mpig': ('pigmentation', 'Pig'), 'mreg': ('regression_structures', 'Reg')}


def mlabel(r):
    if r['label'] == 'mel': return f"{ARCH[r['arch']]} s{r['seed']}" + (' (Derm7pt)' if r['data'] == 'derm7pt' else '')
    base = {'mpn': 'Control+ matched: pigment network', 'mvs': 'Control+ matched: vascular structures', 'mpig': 'Control+ matched: pigmentation', 'mreg': 'Control+ matched: regression'}[r['label']]
    return base + (f" ({ARCH[r['arch']]} s{r['seed']})" if (r['arch'], r['seed']) != ('resnet50', 1) else '')


cls = e.drop_duplicates('classifier')[['classifier', 'label', 'arch', 'seed', 'data']]
iv = real.groupby(['classifier', 'concept']).dlogit.mean()
K, Kall = {}, {}
for l, (c, s) in MT.items():
    ks = [iv[(r['classifier'], c)] / o5.loc[o5.Classifier == mlabel(r), f'CBM-B {s}'].values[0] for _, r in cls[cls.label == l].iterrows()]
    K[s] = float(np.median(ks)); Kall[s] = [rd(x, 2) for x in ks]
rows = []
for _, r in cls[cls.label == 'mel'].iterrows():
    row = {'Classifier': mlabel(r)}
    for l, (c, s) in MT.items():
        ob = o5.loc[o5.Classifier == mlabel(r), f'CBM-B {s}'].values[0]
        row[f'{s} observational'] = rd(ob, 2); row[f'{s} expected'] = rd(K[s] * ob, 2); row[f'{s} measured'] = rd(iv[(r['classifier'], c)], 2)
    rows.append(row)
cal = pd.DataFrame(rows); cal.to_csv(os.path.join(OUT, 'tables/T_calibrated_obs_vs_intervention.csv'), index=False)
put('calibration_ratios', {'median': {k: rd(v, 2) for k, v in K.items()}, 'per_control': Kall}, 'computed from TP2 and TP5 (Phase 2d); exploratory')
put('calibrated_comparison', 'tables/T_calibrated_obs_vs_intervention.csv', 'computed; exploratory (defined after Phase 2d)')

# ------------------------------------------------------------------ method comparison (final)
mcmp = [{'Method': 'Counterfactual audit', 'Matched controls passed': '16/16',
         'Per-concept rule (PN, VS, Pig, Reg)': 'valid, valid, valid, valid',
         'Negative control ratio (limit 0.25)': rd(av['negative']['max_abs_dlogit'] / av['negative']['median_max_abs_dlogit_melanoma_classifiers'], 3),
         'rho other seed': N['audit_reliability']['value']['same architecture, other seed'], 'rho other classifier': N['audit_reliability']['value']['different classifier']}]
ours = [v for k, v in mct.items() if not k.startswith('concept:')]
mcmp.append({'Method': 'TCAV, our implementation', 'Matched controls passed': f"{sum(v['pass'] for v in ours)}/{len(ours)}",
             'Per-concept rule (PN, VS, Pig, Reg)': ', '.join('valid' if mct[f'concept:{c}']['valid'] else 'not valid' for c in ['pigment_network', 'vascular_structures', 'pigmentation', 'regression_structures']),
             'Negative control ratio (limit 0.25)': rd(tv['negative']['mean_max_abs_effect'] / tv['negative']['median_mean_max_abs_effect_melanoma_classifiers'], 3),
             'rho other seed': N['tcav_reliability']['value']['same architecture, other seed'], 'rho other classifier': N['tcav_reliability']['value']['different classifier']})
mcmp.append({'Method': 'TCAV, Captum', 'Matched controls passed': f"{sum(v['pass'] for v in cap['matched'].values())}/{len(cap['matched'])}",
             'Per-concept rule (PN, VS, Pig, Reg)': 'not tested (ResNet-50 seed-1 controls only)',
             'Negative control ratio (limit 0.25)': rd(cap['rule']['negative']['mean_max_abs_effect'] / cap['rule']['negative']['median_mean_max_abs_effect_melanoma_classifiers'], 3),
             'rho other seed': N['captum']['value']['rho_seeds'], 'rho other classifier': N['captum']['value']['rho_cross']})
nobs = sum(1 for v in mc['controls'].values() if v['observational_largest'] == {'pigment_network': 'PN', 'vascular_structures': 'VS', 'pigmentation': 'Pig', 'regression_structures': 'Reg'}[v['concept']])
mcmp.append({'Method': 'Observational (ridge on CBM-B outputs, real images)', 'Matched controls passed': f'{nobs}/{len(mc["controls"])} (largest coefficient)',
             'Per-concept rule (PN, VS, Pig, Reg)': 'one setting only', 'Negative control ratio (limit 0.25)': N['observational_negative_ratio']['value'],
             'rho other seed': N['observational_reliability']['value']['same architecture, other seed'], 'rho other classifier': N['observational_reliability']['value']['different classifier']})
mdf = pd.DataFrame(mcmp); mdf.to_csv(os.path.join(OUT, 'tables/T_method_comparison_final.csv'), index=False)
put('method_comparison', mdf.to_dict(orient='records'), 'computed -> tables/T_method_comparison_final.csv')

# ------------------------------------------------------------------ six ISIC 2020 melanoma classifiers only (primary in the text)
m6 = real[(real.label == 'mel') & (real.data == 'isic2020')]; rows6 = []
for c, s in zip(CN, SH):
    x = m6[m6.concept == c]; xn = x[x.arch != 'effb0']; pc = x.groupby('classifier').dlogit.mean()
    rows6.append({'Concept': s, 'Knob specific': s in ('PN', 'VS', 'Pig', 'Reg'), 'Real benign base': rd(x[x.base == 'real_ben'].dlogit.mean(), 2),
                  'Real melanoma base': rd(x[x.base == 'real_mel'].dlogit.mean(), 2), 'Both bases': rd(x.dlogit.mean(), 2),
                  'Without EfficientNet-B0': rd(xn.dlogit.mean(), 2), 'Per-classifier min': rd(pc.min(), 2), 'Per-classifier max': rd(pc.max(), 2),
                  'Mean change in p(melanoma)': rd(x.dprob.mean(), 3), 'Positive and significant (of 12)': int(((x.p_bh < 0.05) & (x.median_slope > 0)).sum())})
t6 = pd.DataFrame(rows6); t6.to_csv(os.path.join(OUT, 'tables/T_melanoma_effects_isic6.csv'), index=False)
put('melanoma_effects_isic6', t6.to_dict(orient='records'), 'computed from TP2 (Phase 2d) -> tables/T_melanoma_effects_isic6.csv',
    'six ISIC 2020 classifiers, two realistic bases; 12 tests = 6 classifiers x 2 bases')
dd = real[(real.data == 'derm7pt')].groupby('concept').dlogit.mean()
put('derm7pt_classifier_effects', {SH[CN.index(k)]: rd(v, 2) for k, v in dd.items()}, 'computed from TP2 (Phase 2d)', 'mean of the two realistic bases')

# ------------------------------------------------------------------ further values used in the manuscript text
isic6 = [c for c in o5.Classifier if c.startswith(('ResNet', 'Efficient', 'ViT')) and 'Derm7pt' not in c]
oc = o5[o5.Classifier.isin(isic6)]
put('observational_coef_isic6', {s: [rd(oc[f'CBM-B {s}'].min(), 2), rd(oc[f'CBM-B {s}'].max(), 2)] for s in SH},
    'audit/tables/TP5_observational_attribution.csv (Phase 2d)', 'min and max over the six ISIC 2020 classifiers, logit per SD')
put('observational_top_concept_isic6', {c: oc[oc.Classifier == c][[f'CBM-B {s}' for s in SH]].astype(float).idxmax(axis=1).values[0][6:] for c in isic6}, 'same')
a1 = t1c[(t1c['Label'] == 'mel') & (t1c['Data'] == 'isic2020')]['AUC on V2 images (real vectors)']
put('auc_generated_real_vectors_isic6', [rd(a1.min(), 3), rd(a1.max(), 3)], 'audit/tables/TP1_classifiers.csv')
fx = fl[fl['Classifier'].str.match(r'^(ResNet|Efficient|ViT)') & ~fl['Classifier'].str.contains('Derm7pt') & (fl['Base'] == 'Real benign case')]
put('melanoma_call_at_level0_real_benign_isic6', [rd(fx['p(mel) call at level 0'].min(), 3), rd(fx['p(mel) call at level 0'].max(), 3)],
    'audit/tables/TP3_decision_flips.csv', 'share of real-benign-base images above the 90%-sensitivity threshold at knob level 0, over classifiers and knobs')
mtr = {}
for l, (c, s) in MT.items():
    v = [iv[(r['classifier'], c)] for _, r in cls[cls.label == l].iterrows()]; mtr[s] = [rd(min(v), 2), rd(max(v), 2)]
put('matched_target_effect_mean_two_bases', mtr, 'computed from TP2 (Phase 2d)', 'target knob, mean of the two realistic bases, min and max over the 4 controls (Fig. 2b)')
put('captum_cav_accuracy_reported', {'mean': rd(cr.cav_acc.mean(), 3), 'min': rd(cr.cav_acc.min(), 2), 'max': rd(cr.cav_acc.max(), 2), 'n': int(len(cr))},
    'results/captum_runs.csv (Phase 2d)')
put('captum_classifiers', int(cr.classifier.nunique()), 'results/captum_runs.csv (Phase 2d)')
put('logit_clipping_max_share', t0.sort_values('% |logit| > 15', ascending=False).head(3)[['classifier', 'group', '% |logit| > 15', 'min logit']].to_dict(orient='records'),
    'audit/tables/TP0_numerical_check.csv')
put('isic6_top_specific_knob', {c: max(['pigment_network', 'vascular_structures', 'pigmentation', 'regression_structures'], key=lambda k: iv[(c, k)])
                                for c in cls[(cls.label == 'mel') & (cls.data == 'isic2020')].classifier}, 'computed from TP2 (Phase 2d)')

# ------------------------------------------------------------------ Phase 1c paired colour test (Fig. 1b)
tc2 = pd.read_csv(os.path.join(P1C, 'tables/TC2_variants_colour_realism.csv'))
put('colour_outliers_phase1c_paired', tc2[['Generator', 'Class', 'Variant', 'n', 'Colour outliers %', 'KID x1e3 (Inception)']].to_dict(orient='records'),
    'phase1c_outputs: tables/TC2_variants_colour_realism.csv', 'same start noises for V0, V1 and V2; source of Fig. 1b')

# ------------------------------------------------------------------ write
json.dump(N, open(os.path.join(OUT, 'numbers/numbers.json'), 'w'), indent=1, default=str)
with open(os.path.join(OUT, 'numbers/numbers.md'), 'w') as f:
    f.write('| Key | Value | Source | Note |\n|---|---|---|---|\n')
    for k, v in N.items():
        val = json.dumps(v['value'], default=str)
        f.write(f"| {k} | {val[:300] + (' ...' if len(val) > 300 else '')} | {v['source']} | {v['note']} |\n")
print(len(N), 'entries')
