"""Post hoc check of the negative-control criterion (rule 1) on scale-free units (Methods, "Output scale of the
negative control"; Supplementary Table S22). Rule 1 compares raw logit changes; here each classifier's effects are
also divided by a measure of its own logit spread. TCAV scores do not depend on output scale and are not changed.
Inputs: figure_inputs/ (audit effects, observational coefficients, per-classifier logits). Output: results/tables/."""
import os, numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.environ.get('DATA_ROOT', os.path.join(ROOT, 'figure_inputs'))
d = pd.read_csv(f'{D}/p2dout/audit/tables/TP2_counterfactual_effects_all.csv')
o = pd.read_csv(f'{D}/p2dout/audit/tables/TP5_observational_attribution.csv').set_index('Classifier')
MEL7 = ['mel_resnet50_s1', 'mel_resnet50_s2', 'mel_effb0_s1', 'mel_effb0_s2', 'mel_vitb16_s1', 'mel_vitb16_s2', 'mel_resnet50_derm7pt']  # as in rule 1
NEG = 'ctrl_perm_resnet50'
ONAME = {'mel_resnet50_s1': 'ResNet-50 s1', 'mel_resnet50_s2': 'ResNet-50 s2', 'mel_effb0_s1': 'EfficientNet-B0 s1', 'mel_effb0_s2': 'EfficientNet-B0 s2',
         'mel_vitb16_s1': 'ViT-B/16 s1', 'mel_vitb16_s2': 'ViT-B/16 s2', 'mel_resnet50_derm7pt': 'ResNet-50 s1 (Derm7pt)', NEG: 'Control-: permuted labels'}
CB = [c for c in o.columns if c.startswith('CBM-B')]
scale = {}
for n in MEL7 + [NEG]:
    z = np.load(f'{D}/p2ainp/phase2/scores2_fp32/{n}.npz'); t = np.clip(z['cls_test'], -15, 15)
    r = np.clip(np.concatenate([z['rsweep_ben'], z['rsweep_mel']]), -15, 15); q75, q25 = np.percentile(t, [75, 25])
    scale[n] = {'raw': 1.0, 'sd_test': t.std(), 'iqr_test': (q75 - q25) / 1.349, 'sd_rsweep': r.std()}
audit = lambda n, k: d[d.classifier == n].dlogit.abs().max() / scale[n][k]
obs = lambda n, k: o.loc[ONAME[n], CB].abs().max() / scale[n][k]
rows = [dict(scale=k, audit=round(audit(NEG, k) / np.median([audit(m, k) for m in MEL7]), 3),
             observational=round(obs(NEG, k) / np.median([obs(m, k) for m in MEL7]), 3)) for k in ['raw', 'sd_test', 'iqr_test', 'sd_rsweep']]
out = pd.DataFrame(rows); print(out.to_string(index=False))
os.makedirs(f'{ROOT}/results/tables', exist_ok=True); out.to_csv(f'{ROOT}/results/tables/T_negctrl_scale_recomputed.csv', index=False)
