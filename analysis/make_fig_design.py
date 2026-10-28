"""Study-design figure (Fig. 1 of the manuscript; the other main figures are Fig. 2 to 6). Vector PDF and 600-dpi PNG, 183 mm wide, same style as make_figures.py."""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

FIG = os.environ.get('FIG_DIR', os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'figures_rebuilt')); os.makedirs(FIG, exist_ok=True)
MM = 1 / 25.4; W, H = 183, 96
C = dict(blue='#0072B2', orange='#E69F00', otext='#A36A00', green='#009E73', grey='#7F7F7F', ink='#222222', pale='#F4F4F4')
plt.rcParams.update({'font.family': 'DejaVu Sans', 'pdf.fonttype': 42, 'ps.fonttype': 42})
fig = plt.figure(figsize=(W * MM, H * MM)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis('off')

def box(x, y, w, h, title, body, ec=C['ink'], fc='white', tc=None, fs=6.3, lw=0.7, out=None, outc=None):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0,rounding_size=1.2', ec=ec, fc=fc, lw=lw))
    ax.text(x + 2, y + h - 2.2, title, fontsize=6.8 if fs > 6 else 5.9, fontweight='bold', color=tc or ec, va='top', ha='left')
    ax.text(x + 2, y + h - 6.6, body, fontsize=fs, color=C['ink'], va='top', ha='left', linespacing=1.25)
    if out:
        ax.text(x + 2, y + 2.0, out, fontsize=fs, color=outc or ec, fontweight='bold', va='bottom', ha='left', linespacing=1.25)

def arrow(x0, y0, x1, y1, c=C['ink'], text=None, dy=1.6):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle='-|>', mutation_scale=7, lw=0.8, color=c, shrinkA=0, shrinkB=0))
    if text:
        ax.text((x0 + x1) / 2, (y0 + y1) / 2 + dy, text, fontsize=5.6, color=c, ha='center', va='bottom', style='italic')

def letter(x, y, s):
    ax.text(x, y, s, fontsize=8.5, fontweight='bold', va='top', ha='left')

# ---------------------------------------------------------------- a, instrument and audit
letter(0.5, 95.5, 'a')
x0, wb, gap, ya, hb = 5, 30.4, 6.0, 73, 22
A = [('Real images', 'ISIC 2020,\n33,126 dermoscopy\nimages at 128 px'),
     ('Concept labels', 'CBM-A pseudo-labels\nfor the seven\n7-point concepts'),
     ('Generator', 'Concept-conditioned\nDDPM with colour-\ncalibrated start noise'),
     ('Knob sweeps', 'One knob from 0 to 1;\nthe other six concepts,\nnoise and colour fixed;\nfour bases'),
     ('Audit', 'Classifier logit change\nfrom level 0 to 1;\nWilcoxon test on slopes,\nBH correction')]
for i, (t, b) in enumerate(A):
    x = x0 + i * (wb + gap)
    box(x, ya, wb, hb, t, b, ec=C['ink'])
    if i < len(A) - 1:
        arrow(x + wb + 0.6, ya + hb / 2, x + wb + gap - 0.6, ya + hb / 2)
# knob check under the generator and sweeps
xk = x0 + 2 * (wb + gap); wk = 3 * wb + 2 * gap
ax.add_patch(FancyBboxPatch((xk, 62), wk, 7.2, boxstyle='round,pad=0,rounding_size=1.0', ec=C['grey'], fc=C['pale'], lw=0.6))
ax.text(xk + 2, 65.6, 'Knob check with CBM-B and the ISIC 2018 detector: PN, VS, Pig and Reg are\ntarget-dominant but also change other concepts; BWV, Str and D/G are not', fontsize=5.8, color=C['ink'], va='center', ha='left', linespacing=1.2)
xs = x0 + 3 * (wb + gap) + wb / 2
arrow(xs, ya - 0.4, xs, 69.6, c=C['grey'])

# ---------------------------------------------------------------- b, validity rules in the order they were fixed
letter(0.5, 58.5, 'b')
yb, hr = 27, 30; gb = 5; wr = (176 - 3 * gb) / 4
R = [('Rule 1: before the first run', 'Quartile positive controls\n(pigmentation, blue-whitish veil)\nPermuted-label negative control',
      'Not valid: veil control\nconfounded (0 of 4 bases)', C['orange'], C['otext']),
     ('Rules 2, 3: after rule 1 failed', 'Controls matched on the other\nsix CBM-A labels; 4 controls\nper concept, 16 in total',
      'Recovered for PN, VS, Pig, Reg\n(16 of 16 controls passed)', C['blue'], C['blue']),
     ('Rule 4: fixed with SHA-256', 'Controls matched on the other\nsix CBM-B outputs (a second\nevaluator, trained on Derm7pt)',
      'Recovered for PN, VS, Pig\n(12 of 12); Reg not testable', C['green'], C['green']),
     ('Rules 5-8: fixed with SHA-256', '18 permuted-label classifiers\n(null); 9 retrained and 6\nHAM10000 melanoma classifiers',
      'Pig above null; stable (15 of 15);\ntransfers to HAM10000 (6 of 6)', '#5B2A86', '#5B2A86')]
for i, (t, b, o, c, ct) in enumerate(R):
    x = x0 + i * (wr + gb)
    box(x, yb, wr, hr, t, b, ec=c, tc=ct, lw=1.0, out=o, outc=ct, fs=5.7)
    if i < len(R) - 1:
        arrow(x + wr + 0.5, yb + hr / 2, x + wr + gb - 0.5, yb + hr / 2)

# ---------------------------------------------------------------- c, comparison and application
letter(0.5, 23.5, 'c')
yc, hc = 2, 20.5; wc1 = 92; wc2 = 176 - wc1 - 9
box(x0, yc, wc1, hc, 'Three other methods, same controls',
    'TCAV, our implementation: target found in 11 of 16 matched controls\n'
    'TCAV, Captum 0.9.0: target found in 0 of 3 matched controls\n'
    'Observational attribution: target found in all matched controls\n'
    'Permuted-label controls: used as a null distribution, not to rank methods', ec=C['ink'])
box(x0 + wc1 + 9, yc, wc2, hc, 'Melanoma classifiers',
    '15 ISIC 2020 classifiers (3 architectures,\n5 seeds), 6 HAM10000 classifiers, 1 Derm7pt\nclassifier: knob-sweep responses, observational\nattribution and decomposition', ec=C['ink'])

for ext in ('pdf', 'png'):
    fig.savefig(os.path.join(FIG, 'Fig1_study_design.' + ext), dpi=600 if ext == 'png' else None)
print('wrote', os.path.join(FIG, 'Fig1_study_design.pdf'))
