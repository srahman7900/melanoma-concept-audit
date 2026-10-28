"""Supplementary files for submission:
 - Supplementary_Information.pdf: Supplementary Figures S1 to S15 with legends
 - Supplementary_Data_1.xlsx: Supplementary Tables S1 to S22, one sheet each, with an index sheet (title and source file)"""
import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image
from openpyxl import load_workbook
from openpyxl.styles import Font, Alignment
from openpyxl.utils import get_column_letter

H = '/home/claude'; OUT = os.path.join(H, 'paper'); SUB = os.path.join(OUT, 'submission'); os.makedirs(SUB, exist_ok=True)
FS = os.path.join(OUT, 'figures', 'supplementary')
FIGS = [('FigS01_evaluator_auc', 'Evaluator AUC at native and 128-pixel input.\nPer-concept test AUC on the official Derm7pt test split (n = 394) for CBM-A at its native resolution, CBM-A with the 128-pixel pipeline and CBM-B with the 128-pixel pipeline. Dotted line, chance.'),
        ('FigS02_dose_response_melanoma_prototype', 'Dose responses on the melanoma prototype.\nEvaluator outputs (SD units from the mean of real benign images) as each knob moves from 0 to 1 in steps of 0.1 on the melanoma prototype; mean and 95% interval over 32 trajectories. CBM-A made the pseudo-labels; CBM-B, the ISIC 2018 detector and lesion darkness are the other readings.'),
        ('FigS03_dose_response_benign_prototype', 'Dose responses on the benign prototype.\nAs Supplementary Figure S2, for the benign prototype.'),
        ('FigS04_crosstalk_melanoma_prototype', 'Knob cross-talk on the melanoma prototype.\nMean per-trajectory Spearman correlation between the level of the swept knob (rows) and each evaluator output (columns) for CBM-A, CBM-B and the ISIC 2018 detector. Off-diagonal values show changes in concepts other than the swept one.'),
        ('FigS05_colour_targeting', 'Colour targeting with the colour-calibrated start noise.\nAchieved against targeted mean value per RGB channel (model scale) for generated melanoma and benign images. Dotted line, achieved equals target. The fit is weakest for the red channel.'),
        ('FigS06_guidance_scale_sweep', 'Guidance-scale sweep.\nConcept agreement with CBM-A for the melanoma and benign prototypes (left), mean per-concept AUC of CBM-A and CBM-B on the generated images (middle) and Inception-feature diversity of generated melanoma and benign images (right), against the classifier-free guidance scale s. Dotted line, s = 3.5, the value used in all audits.'),
        ('FigS07_class_signal_colour_control', 'Class signal with class-specific and class-agnostic colour targets.\nMelanoma-against-benign AUC of CBM-B, CBM-A and a colour-only classifier (mean L*, a*, b*) on generated images; error bars, 95% bootstrap intervals. Concept, concept-conditioned model; Baseline, class-conditioned model; prototype, class prototype vector; real vectors, pseudo-label vectors of real images.'),
        ('FigS08_memorisation_screen', 'Memorisation screen.\nLeft, cosine similarity (Inception features) of each generated melanoma image to its nearest real training melanoma, with the real leave-one-out distribution as reference; dotted line, 99th percentile of the real distribution. Right, the four most similar generated images and their nearest real images. Concept, concept-conditioned model; Baseline, class-conditioned model; real vectors, pseudo-label vectors of real images.'),
        ('FigS09_qualitative_paired', 'Paired qualitative examples.\nGenerated melanoma and benign images from the concept-conditioned (Concept) and class-conditioned (Baseline) models; images in one column share the start noise.'),
        ('FigS10_audit_effects_all_bases', 'Audit effects on all four bases.\nChange in the melanoma logit from knob level 0 to 1 for the classifiers named on the left, on the two realistic bases and the two prototypes (mean over trajectories).'),
        ('FigS11_dose_response_real_benign', 'Dose responses on the real benign base.\nMean p(melanoma) over trajectories against the knob level for each architecture (realistic benign base, 6 levels). The knobs change other concepts as well (Fig. 2c).'),
        ('FigS12_decision_flips', 'Decision flips.\nShare of trajectories whose decision changes from benign to melanoma between knob level 0 and 1, at the threshold that gives 90% sensitivity on the validation set, for each knob and base.'),
        ('FigS13_tcav_heatmap', 'TCAV scores of all classifiers (our implementation).\nMean TCAV score on melanoma target images for the middle and last layers and the three concept-set types; colour, score minus the random-CAV mean; asterisk and bold, different from random CAVs (Benjamini-Hochberg p < 0.05) and above them.'),
        ('FigS14_tcav_controls', 'TCAV results for the controls.\nTCAV score per concept (mean and SD over runs) for each control at the middle and last layers, with ISIC 2020 concept sets from all training images and melanoma inputs; red, the target concept; grey band and line, random CAVs (mean and SD).'),
        ('FigS15_captum_heatmap', 'Captum TCAV scores.\nCaptum 0.9.0 TCAV scores at the last layer for the two ISIC concept-set types; colour, score minus the random mean; asterisk and bold, different from random sets.')]
TITLE = 'Testing concept-level audits of melanoma classifiers with diffusion counterfactuals and matched positive controls'

with PdfPages(os.path.join(SUB, 'Supplementary_Information.pdf')) as pdf:
    fig = plt.figure(figsize=(8.27, 11.69)); fig.text(0.5, 0.62, 'Supplementary Information', ha='center', fontsize=18, weight='bold')
    fig.text(0.5, 0.56, TITLE, ha='center', fontsize=10, wrap=True)
    fig.text(0.5, 0.50, 'Supplementary Figures S1 to S15. Supplementary Tables S1 to S22 are in Supplementary Data 1.', ha='center', fontsize=9)
    pdf.savefig(fig); plt.close(fig)
    for i, (fn, cap) in enumerate(FIGS, 1):
        im = Image.open(os.path.join(FS, fn + '.png')).convert('RGB'); im.thumbnail((2400, 2400))
        fig = plt.figure(figsize=(8.27, 11.69))
        ax = fig.add_axes([0.07, 0.12, 0.86, 0.76]); ax.imshow(im); ax.axis('off')
        head, *rest = cap.split('\n')
        fig.text(0.07, 0.93, f'Supplementary Figure S{i} | {head}', fontsize=10, weight='bold', ha='left')
        if rest:
            import textwrap
            fig.text(0.07, 0.915, textwrap.fill(' '.join(rest), 112), fontsize=8.5, ha='left', va='top')
        pdf.savefig(fig, dpi=200); plt.close(fig)

P2D = os.path.join(H, 'p2dout'); P2E = os.path.join(H, 'p2eout'); T = os.path.join(OUT, 'tables'); P1D = os.path.join(H, 'p1dout', 'tables')
TABLES = [('S1', 'Classifiers and their test performance', os.path.join(P2D, 'audit/tables/TP1_classifiers.csv')),
          ('S2', 'Numerical checks of the logits', os.path.join(P2D, 'audit/tables/TP0_numerical_check.csv')),
          ('S3', 'Pseudo-label distribution', os.path.join(T, 'T_pseudo_label_distribution.csv')),
          ('S4', 'Target dominance of the knobs on four bases (CBM-B; largest = the knob raised its own output more than any other knob)', os.path.join(T, 'T_knob_specificity.csv')),
          ('S5', 'Cross-talk on the realistic bases', os.path.join(T, 'T_crosstalk_realistic_bases.csv')),
          ('S6', 'Matched-control balance (CBM-A labels)', os.path.join(P2D, 'audit/tables/TP9b_matched_balance.csv')),
          ('S7', 'All audit effects', os.path.join(P2D, 'audit/tables/TP2_counterfactual_effects_all.csv')),
          ('S8', 'Effects of the six ISIC 2020 classifiers', os.path.join(T, 'T_melanoma_effects_isic6.csv')),
          ('S9', 'Decomposition into concept and colour', os.path.join(P2D, 'audit/tables/TP4_decomposition_concept_colour.csv')),
          ('S10', 'Colour sensitivity', os.path.join(P2D, 'audit/tables/TP7_colour_sensitivity.csv')),
          ('S11', 'Observational attribution', os.path.join(P2D, 'audit/tables/TP5_observational_attribution.csv')),
          ('S12', 'Calibrated comparison of observation and intervention (CBM-A controls)', os.path.join(T, 'T_calibrated_obs_vs_intervention.csv')),
          ('S13', 'Reliability', os.path.join(P2D, 'audit/tables/TP8_reliability.csv')),
          ('S14', 'TCAV concept sets', os.path.join(P2D, 'tcav/tables/TT1_concept_sets.csv')),
          ('S15', 'TCAV scores', os.path.join(P2D, 'tcav/tables/TT2_tcav_all.csv')),
          ('S16', 'Captum TCAV scores', os.path.join(P2D, 'tcav/tables/TT10_captum_tcav_all.csv')),
          ('S17', 'Evaluator validity', os.path.join(P1D, 'T1_evaluator_validity.csv')),
          ('S18a', 'Distribution distances', os.path.join(P1D, 'TD5_T2d_distribution_distances_V0_vs_V2.csv')),
          ('S18b', 'Diversity', os.path.join(P1D, 'TD4_diversity.csv')),
          ('S19a', 'Design of the CBM-B matched controls (fixed before training)', os.path.join(T, 'T2e_design_cbmb.csv')),
          ('S19b', 'Balance of every control design, read with both label sets', os.path.join(T, 'T2e_cross_label_balance.csv')),
          ('S20a', 'CBM-B matched controls: results per control', os.path.join(T, 'T2e_matched_controls_cbmb.csv')),
          ('S20b', 'CBM-B matched controls: results per base', os.path.join(P2E, 'audit/tables/TP10_matched_controls_cbmb.csv')),
          ('S20c', 'CBM-B matched controls: decision per concept', os.path.join(T, 'T2e_go_no_go.csv')),
          ('S21', 'Calibration of observation with the CBM-B controls (exploratory)', os.path.join(T, 'T2e_calibration_cbmb.csv')),
          ('S22a', 'Negative-control ratio on the logit scale and on three scale-free scales (post hoc); TCAV scores do not depend on output scale', os.path.join(T, 'T_S22a_negative_control_scale.csv')),
          ('S22b', 'Logit spread of the melanoma classifiers and the original controls on the real test images', os.path.join(T, 'T_S22b_logit_spread.csv'))]
xl = os.path.join(SUB, 'Supplementary_Data_1.xlsx')
# internal run names are replaced by the names used in the paper
RENAME_VALUES = {'V0 original (Phase 1b)': 'Original start noise',
                 'V2 colour-calibrated (Phase 1d)': 'Colour-calibrated start noise',
                 'ClinDiffXAI': 'Concept-conditioned'}
def clean(df):
    df = df.rename(columns=lambda c: c.replace('V2 images', 'generated images'))
    for c in df.columns:
        if df[c].dtype == object or str(df[c].dtype).startswith('str'):
            df[c] = df[c].map(lambda v: RENAME_VALUES.get(v, v.replace('ClinDiffXAI', 'Concept-conditioned')) if isinstance(v, str) else v)
    return df
with pd.ExcelWriter(xl, engine='openpyxl') as w:
    pd.DataFrame([{'Table': f'Supplementary Table {s}', 'Title': t} for s, t, p in TABLES]).to_excel(w, sheet_name='Index', index=False)
    for s, t, p in TABLES:
        clean(pd.read_csv(p)).to_excel(w, sheet_name=s, index=False, startrow=2)
wb = load_workbook(xl)
for s, t, p in TABLES:
    ws = wb[s]; ws['A1'] = f'Supplementary Table {s} | {t}'; ws['A1'].font = Font(name='Arial', bold=True, size=11)
for ws in wb.worksheets:
    for row in ws.iter_rows():
        for c in row:
            if c.coordinate != 'A1' or ws.title == 'Index':
                c.font = Font(name='Arial', size=10, bold=(c.row == (1 if ws.title == 'Index' else 3)))
    for k, col in enumerate(ws.columns, 1):
        w_ = max(len(str(c.value)) if c.value is not None and c.row > 1 else 0 for c in col)
        ws.column_dimensions[get_column_letter(k)].width = min(max(10, w_ + 2), 60)
    ws.freeze_panes = 'A2' if ws.title == 'Index' else 'A4'
wb.save(xl)
print('wrote', os.listdir(SUB))
