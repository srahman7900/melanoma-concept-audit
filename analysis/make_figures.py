"""Main figures 2 to 6 (one style). Reads the run outputs and the tables of numbers_sheet.py.
Output: figures/Fig2..Fig6 as PDF (Fig1, the study design, comes from make_fig_design.py) (vector) and PNG (600 dpi). Width: 183 mm (double column)."""
import os, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from matplotlib.colors import TwoSlopeNorm

# Input root: DATA_ROOT if set; otherwise the repository folder figure_inputs/ (next to analysis/) if it exists;
# otherwise the authors' working folder. Output: FIG_DIR if set, otherwise <input root>/paper/figures.
_REPO_INPUTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'figure_inputs')
H = os.environ.get('DATA_ROOT') or (_REPO_INPUTS if os.path.isdir(_REPO_INPUTS) else '/home/claude')
OUT = os.path.join(H, 'paper'); FIG = os.environ.get('FIG_DIR', os.path.join(OUT, 'figures')); os.makedirs(FIG, exist_ok=True)
AI = os.path.join(H, 'p2ainp'); P1C = os.path.join(H, 'p1cout'); P2D = os.path.join(H, 'p2dout')
CN = ['pigment_network', 'blue_whitish_veil', 'vascular_structures', 'pigmentation', 'streaks', 'dots_and_globules', 'regression_structures']
SH = ['PN', 'BWV', 'VS', 'Pig', 'Str', 'D/G', 'Reg']
LONG = {'PN': 'Pigment network', 'BWV': 'Blue-whitish veil', 'VS': 'Vascular structures', 'Pig': 'Pigmentation',
        'Str': 'Streaks', 'D/G': 'Dots and globules', 'Reg': 'Regression structures'}
VALID_KNOB = {'PN', 'VS', 'Pig', 'Reg'}
MM = 1 / 25.4; W2 = 183 * MM
C = dict(blue='#0072B2', orange='#E69F00', verm='#D55E00', green='#009E73', purple='#CC79A7', sky='#56B4E9', grey='#7F7F7F', lgrey='#C8C8C8', black='#000000')
METH = [('Counterfactual audit', C['blue']), ('TCAV (ours)', C['orange']), ('TCAV (Captum)', C['verm']), ('Observational', C['green'])]
ARCH = {'resnet50': ('ResNet-50', C['blue'], 'o'), 'effb0': ('EfficientNet-B0', C['orange'], 's'), 'vitb16': ('ViT-B/16', C['purple'], '^')}
ANAME = {'resnet50': 'ResNet-50', 'effb0': 'EfficientNet-B0', 'vitb16': 'ViT-B/16'}

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 7, 'axes.labelsize': 7, 'axes.titlesize': 7, 'xtick.labelsize': 6.5,
                     'ytick.labelsize': 6.5, 'legend.fontsize': 6.5, 'axes.linewidth': 0.6, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
                     'xtick.major.size': 2.5, 'ytick.major.size': 2.5, 'axes.spines.top': False, 'axes.spines.right': False,
                     'pdf.fonttype': 42, 'ps.fonttype': 42, 'legend.frameon': False, 'axes.titleweight': 'normal'})
CMAP = plt.get_cmap('RdBu_r')


def letter(ax, s, x=-0.02, y=1.0):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=8, fontweight='bold', va='bottom', ha='right')


def save(fig, name):
    fig.savefig(os.path.join(FIG, name + '.pdf'), bbox_inches='tight', pad_inches=0.02)
    fig.savefig(os.path.join(FIG, name + '.png'), dpi=600, bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)


def heat(ax, M, vmax, rows, cols, fmt='{:+.1f}', outline=None, fs=5.6):
    ax.imshow(M, cmap=CMAP, norm=TwoSlopeNorm(0, -vmax, vmax), aspect='auto', interpolation='nearest')
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            if np.isfinite(v):
                ax.text(j, i, fmt.format(v), ha='center', va='center', fontsize=fs, color='white' if abs(v) > 0.6 * vmax else 'black')
    if outline is not None:
        for i, j in outline:
            ax.add_patch(Rectangle((j - .5, i - .5), 1, 1, fill=False, lw=1.0, ec='black'))
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols); ax.set_yticks(range(len(rows))); ax.set_yticklabels(rows)
    ax.tick_params(length=0)
    for s in ax.spines.values(): s.set_visible(False)


def cbar(fig, ax, vmax, label, ticks=None):
    sm = plt.cm.ScalarMappable(cmap=CMAP, norm=TwoSlopeNorm(0, -vmax, vmax)); sm.set_array([])
    cb = fig.colorbar(sm, cax=ax, orientation='horizontal', extend='both')
    cb.set_label(label, fontsize=6.5); cb.ax.tick_params(labelsize=6, length=2); cb.outline.set_linewidth(0.5)
    if ticks is not None: cb.set_ticks(ticks)
    return cb


# ---------------------------------------------------------------- shared data
e = pd.read_csv(os.path.join(P2D, 'audit/tables/TP2_counterfactual_effects_all.csv'))
real = e[e.base.isin(['real_ben', 'real_mel'])]
iv = real.groupby(['classifier', 'concept']).dlogit.mean()
mc = json.load(open(os.path.join(P2D, 'audit/results/matched_controls.json')))
av = json.load(open(os.path.join(P2D, 'audit/results/audit_validity.json')))
NUM = json.load(open(os.path.join(OUT, 'numbers/numbers.json')))
KM = np.load(os.path.join(OUT, 'numbers/knob_matrices.npy'), allow_pickle=True).item()
spec = pd.read_csv(os.path.join(OUT, 'tables/T_knob_specificity.csv'))

# Phase 2e (matched controls labelled by CBM-B), used when integrate_2e.py has run
P2E = os.environ.get('P2E_DIR', os.path.join(H, 'p2eout'))
HAVE2E = os.path.exists(os.path.join(P2E, 'audit/results/matched_controls_cbmb.json')) and \
    os.path.exists(os.path.join(OUT, 'tables/T2e_cross_label_balance.csv'))
if HAVE2E:
    e2 = pd.read_csv(os.path.join(P2E, 'audit/tables/TP2_counterfactual_effects_all.csv'))
    iv2 = e2[e2.base.isin(['real_ben', 'real_mel'])].groupby(['classifier', 'concept']).dlogit.mean()
    mc2 = json.load(open(os.path.join(P2E, 'audit/results/matched_controls_cbmb.json')))
    xb = pd.read_csv(os.path.join(OUT, 'tables/T2e_cross_label_balance.csv'))
    gng = pd.read_csv(os.path.join(OUT, 'tables/T2e_go_no_go.csv'))


def clsname(r):
    a = ANAME[r['arch']]
    return f"{a} s{r['seed']}" + (' (Derm7pt)' if r['data'] == 'derm7pt' else '')


# ================================================================= Fig 1: the instrument
def fig1():
    fig = plt.figure(figsize=(W2, 112 * MM))
    gs = GridSpec(2, 2, figure=fig, width_ratios=[1.0, 1.08], height_ratios=[0.8, 1.2], wspace=0.28, hspace=0.55)
    # (a) one realistic benign base, all seven knobs, six levels
    s = 'rsweep_ben'; NB = 5
    z = np.load(os.path.join(AI, f'phase2/fig_images/{s}.npz')); m = json.load(open(os.path.join(AI, f'phase2/sets_meta/{s}.json')))['meta']
    idx = z['index']; ci = np.array(m['concept_idx'])[idx]; lv = np.array(m['level'])[idx]; nid = np.array(m['noise_id'])[idx]
    levels = sorted(set(lv.tolist()))
    sub = GridSpecFromSubplotSpec(7, 6, subplot_spec=gs[:, 0], wspace=0.04, hspace=0.06); top = []
    for j in range(7):
        for k, l in enumerate(levels):
            a = fig.add_subplot(sub[j, k]); w = np.where((nid == NB) & (ci == j) & np.isclose(lv, l))[0][0]
            a.imshow(z['imgs'][w].transpose(1, 2, 0), interpolation='lanczos'); a.set_xticks([]); a.set_yticks([])
            for sp in a.spines.values(): sp.set_visible(False)
            if j == 0: a.set_title(f'{l:.1f}', fontsize=6.5, pad=2); top.append(a)
            if k == 0:
                ok = SH[j] in VALID_KNOB
                a.set_ylabel(SH[j], fontsize=7, rotation=0, ha='right', va='center', labelpad=3, color='black' if ok else C['grey'],
                             fontweight='bold' if ok else 'normal')
    for a_ in top: a_.apply_aspect()
    p0, p5 = top[0].get_position(), top[-1].get_position(); yt = p0.y1 + 0.045
    fig.text(p0.x0 - 0.06, yt, 'a', fontsize=8, fontweight='bold', va='bottom')
    fig.text((p0.x0 + p5.x1) / 2, yt, 'Knob level (concept value in the conditioning vector)', ha='center', va='bottom', fontsize=6.5)
    # (b) colour outliers, paired test of the start-noise variants (Phase 1c)
    tc = pd.read_csv(os.path.join(P1C, 'tables/TC2_variants_colour_realism.csv'))
    ax = fig.add_subplot(gs[0, 1]); letter(ax, 'b', x=-0.13)
    groups = [('ClinDiffXAI, prototype', 'mel', 'Model\nmelanoma'), ('ClinDiffXAI, prototype', 'ben', 'Model\nbenign'),
              ('Baseline', 'mel', 'Baseline\nmelanoma'), ('Baseline', 'ben', 'Baseline\nbenign')]
    var = [('V0 original sampling', 'Original start noise', C['lgrey']), ('V1 trailing start', 'Trailing timestep', C['grey']),
           ('V2 colour-calibrated noise', 'Colour-calibrated start', C['blue'])]
    wb = 0.26
    for gi, (g, c, lab) in enumerate(groups):
        for vi, (v, vl, col) in enumerate(var):
            val = tc[(tc.Generator == g) & (tc.Class == c) & (tc.Variant == v)]['Colour outliers %'].values[0]
            ax.bar(gi + (vi - 1) * wb, val, wb * 0.95, color=col, label=vl if gi == 0 else None, lw=0)
            if vi == 2: ax.text(gi + wb, val + 2, f'{val:.1f}', ha='center', fontsize=5.8, color=C['blue'])
        ref = 2.4 if c == 'mel' else 0.6
        ax.plot([gi - 1.5 * wb, gi + 1.5 * wb], [ref, ref], color='black', lw=0.8, ls=':')
    ax.axhline(20, color=C['verm'], lw=0.7, ls='--'); ax.text(1.01, 20, 'limit\n20%', transform=ax.get_yaxis_transform(), fontsize=5.8, color=C['verm'], ha='left', va='center', clip_on=False)
    ax.set_xticks(range(4)); ax.set_xticklabels([g[2] for g in groups], fontsize=6.2); ax.set_ylim(0, 100)
    ax.set_ylabel('Colour outliers (%)')
    h, l = ax.get_legend_handles_labels(); h.append(Line2D([], [], color='black', ls=':', lw=0.8)); l.append('Real images')
    ax.legend(h, l, loc='upper right', bbox_to_anchor=(1.02, 1.12), ncol=2, fontsize=5.8, handlelength=1.2, columnspacing=0.8)
    # (c) knob specificity: CBM-B response to each knob, mean over the two realistic bases
    ax = fig.add_subplot(gs[1, 1]); letter(ax, 'c', x=-0.19, y=1.12)
    M = (KM['Real benign case|CBM-B'] + KM['Real melanoma case|CBM-B']) / 2
    heat(ax, M, 4, [f'{s} knob' for s in SH], SH, outline=[(i, i) for i in range(7)])
    ax.set_xlabel('CBM-B concept output (change in SD of real images)', labelpad=2); ax.xaxis.set_label_position('top'); ax.xaxis.tick_top()
    for i, s_ in enumerate(SH):
        r = spec[spec['Evaluator output'] == f'CBM-B: {s_}']['Bases where the knob is largest'].values[0]
        ax.text(7.0, i, r, va='center', ha='left', fontsize=6, color='black' if s_ in VALID_KNOB else C['grey'],
                fontweight='bold' if s_ in VALID_KNOB else 'normal')
    ax.text(7.0, -0.95, 'Target-dominant\n(of 4 bases)', fontsize=5.6, ha='left', va='bottom')
    cax = ax.inset_axes([0.15, -0.13, 0.7, 0.04]); cbar(fig, cax, 4, 'Change from knob level 0 to 1 (SD)', [-4, -2, 0, 2, 4])
    save(fig, 'Fig2_instrument')


# ================================================================= Fig 2: matched positive controls and the audit
def fig2():
    z = np.load(os.path.join(AI, 'phase2/cls/cls_labels.npz')); split, Cc = z['split'], z['concepts'].astype(float)
    tr = split == 'train'; sd = Cc[tr].std(0)
    rows = []
    for lab, j in [('Quartile groups: Pig', 3), ('Quartile groups: BWV', 1)]:
        v = Cc[tr, j]; lo, hi = np.quantile(v, .25), np.quantile(v, .75); sel = (v <= lo) | (v >= hi); g = v[sel] >= hi
        X = Cc[tr][sel]; rows.append((lab, j, (X[g].mean(0) - X[~g].mean(0)) / sd))
    bal = mc['balance']
    for k, c in enumerate(bal['Concept']):
        smd = np.array([bal[f'SMD {s}'][k] for s in SH]); rows.append((f'Matched pairs: {SH[CN.index(c)]}', CN.index(c), smd))
    fig = plt.figure(figsize=(W2, 98 * MM))
    gs = GridSpec(1, 2, figure=fig, width_ratios=[0.8, 1.2], wspace=0.62)
    ax = fig.add_subplot(gs[0]); letter(ax, 'a', x=-0.72, y=1.01)
    ax.axvspan(-0.1, 0.1, color=C['lgrey'], alpha=0.5, lw=0)
    for i, (lab, j, smd) in enumerate(rows):
        y = len(rows) - 1 - i
        oth = np.delete(smd, j)
        ax.scatter(oth, np.full(6, y), s=11, facecolor='white', edgecolor=C['grey'], lw=0.7, zorder=3)
        ax.scatter([smd[j]], [y], s=22, marker='D', color=C['blue'] if lab.startswith('Matched') else C['verm'], zorder=4)
        worst = np.argmax(np.abs(oth)); wn = [s for x, s in enumerate(SH) if x != j][worst]
        if abs(oth[worst]) > 0.3: ax.text(oth[worst], y + 0.3, wn, fontsize=5.6, ha='center', color=C['grey'])
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows][::-1], fontsize=6.3)
    ax.axvline(0, color='black', lw=0.5); ax.set_xlabel('Standardised mean difference, high minus low group')
    ax.legend([Line2D([], [], marker='D', ls='', color=C['verm'], ms=4), Line2D([], [], marker='D', ls='', color=C['blue'], ms=4),
               Line2D([], [], marker='o', ls='', mfc='white', mec=C['grey'], ms=3.5)],
              ['Target concept (quartile design)', 'Target concept (matched design)', 'Other six concepts'], loc='upper center', fontsize=5.8,
              bbox_to_anchor=(0.45, -0.13), ncol=1)
    ax.set_xlim(-1.6, 4.5); ax.spines['left'].set_visible(False); ax.tick_params(axis='y', length=0)
    # (b) audit of the control classifiers
    order = [('ctrl_pig_resnet50', 'pigmentation', 'Quartile Pig', f"{av['positive_pig']['bases_with_target_largest_and_significant']}/4"),
             ('ctrl_bwv_resnet50', 'blue_whitish_veil', 'Quartile BWV', f"{av['positive_bwv']['bases_with_target_largest_and_significant']}/4"),
             ('ctrl_perm_resnet50', None, 'Permuted labels', 'neg.')]
    ctr = mc['controls']
    for c in ['pigment_network', 'vascular_structures', 'pigmentation', 'regression_structures']:
        ks = sorted([k for k, v in ctr.items() if v['concept'] == c], key=lambda k: (ctr[k]['arch'] != 'resnet50', ctr[k]['seed']))
        for k in ks:
            v = ctr[k]; order.append((k, c, f"{ANAME[v['arch']]} s{v['seed']}", f"{v['bases_passed']}/4"))
    M = np.array([[iv[(k, cc)] for cc in CN] for k, _, _, _ in order])
    ax = fig.add_subplot(gs[1]); letter(ax, 'b', x=-0.5, y=1.06)
    outl = [(i, CN.index(c)) for i, (_, c, _, _) in enumerate(order) if c is not None]
    heat(ax, M, 8, [o[2] for o in order], [f'{s}\nknob' for s in SH], outline=outl, fs=5.3)
    ax.xaxis.tick_top()
    for i, o in enumerate(order):
        ax.text(7.0, i, o[3], va='center', ha='left', fontsize=5.8, fontweight='bold' if o[3] in ('4/4', '3/4') else 'normal',
                color=C['verm'] if o[3] in ('0/4', '1/4', '2/4') else 'black')
    ax.text(7.0, -1.05, 'Bases\npassed', fontsize=5.6, ha='left', va='bottom')
    for yb in [2.5, 6.5, 10.5, 14.5]: ax.axhline(yb, color='white', lw=1.6)
    for y0, y1, g in [(0, 2, 'Original'), (3, 6, 'Matched PN'), (7, 10, 'Matched VS'), (11, 14, 'Matched Pig'), (15, 18, 'Matched Reg')]:
        ax.plot([-0.335, -0.335], [y0 - 0.35, y1 + 0.35], transform=ax.get_yaxis_transform(), color='black', lw=0.6, clip_on=False)
        ax.text(-0.36, (y0 + y1) / 2, g, transform=ax.get_yaxis_transform(), rotation=90, ha='right', va='center', fontsize=6.2, clip_on=False)
    for t, lab in zip(ax.get_xticklabels(), SH): t.set_color('black' if lab in VALID_KNOB else C['grey'])
    cax = ax.inset_axes([0.15, -0.06, 0.7, 0.022]); cbar(fig, cax, 8, 'Change in classifier logit, knob level 0 to 1 (mean of the two realistic bases)', [-8, -4, 0, 4, 8])
    save(fig, 'Fig3_controls')


# ================================================================= Fig 2 (with Phase 2e): balance read with both label sets, audit of all controls
CONC4 = ['pigment_network', 'vascular_structures', 'pigmentation', 'regression_structures']


def _ctrl_order(ctr):
    out = []
    for c in CONC4:
        ks = sorted([k for k, v in ctr.items() if v['concept'] == c], key=lambda k: (ctr[k]['arch'] != 'resnet50', ctr[k]['seed']))
        out += [(k, c) for k in ks]
    return out


def _groups(ax, spans, x=-0.40):
    for y0, y1, g in spans:
        ax.plot([x + 0.025, x + 0.025], [y0 - 0.35, y1 + 0.35], transform=ax.get_yaxis_transform(), color='black', lw=0.6, clip_on=False)
        ax.text(x, (y0 + y1) / 2, g, transform=ax.get_yaxis_transform(), rotation=90, ha='right', va='center', fontsize=6.2, clip_on=False)


def fig2_2e():
    fig = plt.figure(figsize=(W2, 178 * MM))
    gs = GridSpec(2, 1, figure=fig, height_ratios=[0.60, 1.40], hspace=0.40, top=0.95, bottom=0.08)
    top = GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[0], wspace=0.06)
    DES = [('Quartile groups (CBM-A)', 'Quartile, CBM-A', C['verm']), ('Matched pairs (CBM-A)', 'Matched, CBM-A', C['blue']),
           ('Matched pairs (CBM-B)', 'Matched, CBM-B', C['green'])]
    rows = []
    for d, dl, col in DES:
        for s_ in (['Pig', 'BWV'] if d.startswith('Quartile') else ['PN', 'VS', 'Pig', 'Reg']):
            if ((xb.Design == d) & (xb.Concept == s_)).any(): rows.append((d, dl, col, s_))
    nR = len(rows); bounds = [i for i in range(1, nR) if rows[i][0] != rows[i - 1][0]]
    for pi, lab in enumerate(['CBM-A', 'CBM-B']):
        ax = fig.add_subplot(top[pi]); letter(ax, 'ab'[pi], x=-0.30 if pi == 0 else -0.03, y=1.02)
        ax.axvspan(-0.1, 0.1, color=C['lgrey'], alpha=0.5, lw=0)
        for i, (d, dl, col, s_) in enumerate(rows):
            y = nR - 1 - i; r = xb[(xb.Design == d) & (xb.Concept == s_) & (xb['Read with'] == lab)].iloc[0]
            j = SH.index(s_); v = np.array([r[f'SMD {x}'] for x in SH], float); oth = np.delete(v, j)
            ax.scatter(oth, np.full(6, y), s=10, facecolor='white', edgecolor=C['grey'], lw=0.7, zorder=3)
            ax.scatter([v[j]], [y], s=20, marker='D', color=col, zorder=4)
            w = int(np.argmax(np.abs(oth)))
            if abs(oth[w]) > 0.3: ax.text(oth[w], y + 0.28, [x for k, x in enumerate(SH) if k != j][w], fontsize=5.4, ha='center', color=C['grey'])
        for b in bounds: ax.axhline(nR - b - 0.5, color=C['lgrey'], lw=0.6)
        ax.axvline(0, color='black', lw=0.5); ax.set_xlim(-1.6, 5.4); ax.set_ylim(-0.6, nR - 0.4)
        ax.set_title(f'Balance read with {lab} labels', fontsize=6.8, pad=3)
        ax.set_yticks(range(nR))
        NT = set(gng[~gng.Testable.astype(bool)].Concept)
        ax.set_yticklabels([f'{dl}: {s_}' + (' (n.t.)' if ('CBM-B' in d and s_ in NT) else '') for d, dl, _, s_ in rows][::-1] if pi == 0 else [], fontsize=6.1)
        ax.spines['left'].set_visible(False); ax.tick_params(axis='y', length=0)
        ax.set_xlabel('Standardised mean difference, high minus low group', fontsize=6.5)
    fig.legend([Line2D([], [], marker='D', ls='', color=c_, ms=4) for _, _, c_ in DES] + [Line2D([], [], marker='o', ls='', mfc='white', mec=C['grey'], ms=3.5)],
               ['Target: quartile groups, CBM-A labels', 'Target: matched pairs, CBM-A labels', 'Target: matched pairs, CBM-B labels', 'Other six concepts'],
               loc='lower center', ncol=4, fontsize=5.8, bbox_to_anchor=(0.55, 0.975), handlelength=1.0, columnspacing=0.9)
    # (c) audit of the original controls and the CBM-A matched controls; (d) audit of the CBM-B matched controls
    bot = GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[1], wspace=0.62)
    n1 = 3 + len(_ctrl_order(mc['controls'])); n2 = len(_ctrl_order(mc2['controls']))
    botd = GridSpecFromSubplotSpec(2, 1, subplot_spec=bot[1], height_ratios=[max(n1 - n2, 0.01), n2], hspace=0.0)
    order = [('ctrl_pig_resnet50', 'pigmentation', 'Quartile Pig', f"{av['positive_pig']['bases_with_target_largest_and_significant']}/4"),
             ('ctrl_bwv_resnet50', 'blue_whitish_veil', 'Quartile BWV', f"{av['positive_bwv']['bases_with_target_largest_and_significant']}/4"),
             ('ctrl_perm_resnet50', None, 'Permuted labels', 'neg.')]
    ctr = mc['controls']
    order += [(k, c, f"{ANAME[ctr[k]['arch']]} s{ctr[k]['seed']}", f"{ctr[k]['bases_passed']}/4") for k, c in _ctrl_order(ctr)]
    ctr2 = mc2['controls']
    order2 = [(k, c, f"{ANAME[ctr2[k]['arch']]} s{ctr2[k]['seed']}",
               'n.t.' if 'note' in ctr2[k] else f"{ctr2[k]['bases_passed']}/4") for k, c in _ctrl_order(ctr2)]
    for pi, (od, ivx, title) in enumerate([(order, iv, 'Original and CBM-A matched controls'), (order2, iv2, 'CBM-B matched controls')]):
        M = np.array([[ivx.get((k, cc), np.nan) for cc in CN] for k, _, _, _ in od])
        ax = fig.add_subplot(bot[0] if pi == 0 else botd[1]); letter(ax, 'cd'[pi], x=-0.62 if pi == 0 else -0.36, y=1.0 + (0.085 if pi == 0 else 0.105))
        outl = [(i, CN.index(c)) for i, (_, c, _, _) in enumerate(od) if c is not None and np.isfinite(M[i]).any()]
        heat(ax, M, 8, [o[2] for o in od], [f'{s_}\nknob' for s_ in SH], outline=outl, fs=5.0)
        ax.xaxis.tick_top(); ax.set_title(title, fontsize=6.8, pad=3)
        for i, o in enumerate(od):
            if not np.isfinite(M[i]).any():
                ax.text(3, i, 'not testable: too few pairs', ha='center', va='center', fontsize=5.4, color=C['grey'])
            ax.text(7.0, i, o[3], va='center', ha='left', fontsize=5.6, fontweight='bold' if o[3] in ('4/4', '3/4') else 'normal',
                    color=C['verm'] if o[3] in ('0/4', '1/4', '2/4', 'n.t.') else 'black')
        ax.text(7.0, -1.0, 'Bases\npassed', fontsize=5.4, ha='left', va='bottom')
        for t, lab in zip(ax.get_xticklabels(), SH): t.set_color('black' if lab in VALID_KNOB else C['grey'])
        spans = []; i0 = 0
        names = {'pigment_network': 'PN', 'vascular_structures': 'VS', 'pigmentation': 'Pig', 'regression_structures': 'Reg'}
        if pi == 0: spans.append((0, 2, 'Original')); i0 = 3
        for c in CONC4:
            n = sum(1 for o in od[i0:] if o[1] == c)
            if n: spans.append((i0, i0 + n - 1, ('Matched ' if pi == 0 else '') + names[c])); i0 += n
        for y0, y1, _ in spans[:-1]: ax.axhline(y1 + 0.5, color='white', lw=1.6)
        _groups(ax, spans, x=-0.40)
        if pi == 1:
            for y0, y1, g in spans:
                r = gng[gng.Concept == g]
                if len(r):
                    dec = r.Decision.values[0]
                    ax.text(8.15, (y0 + y1) / 2, 'valid' if dec == 'GO' else 'not valid', va='center', ha='left', fontsize=6.0, fontweight='bold', color='black' if dec == 'GO' else C['verm'])
    cax = fig.add_axes([0.30, 0.035, 0.42, 0.008]); cbar(fig, cax, 8, 'Change in classifier logit, knob level 0 to 1 (mean of the two realistic bases)', [-8, -4, 0, 4, 8])
    save(fig, 'Fig3_controls')


# ================================================================= Fig 3: method comparison
def fig3():
    mct = json.load(open(os.path.join(P2D, 'tcav/results/matched_controls_tcav.json')))
    cap = json.load(open(os.path.join(P2D, 'tcav/results/captum_results.json')))
    ctr = mc['controls']; MAP = {'pigment_network': 'PN', 'vascular_structures': 'VS', 'pigmentation': 'Pig', 'regression_structures': 'Reg'}
    cc = list(MAP)
    res = {m: {} for m, _ in METH}
    for c in cc:
        ks = [k for k, v in ctr.items() if v['concept'] == c]
        res['Counterfactual audit'][c] = (sum(ctr[k]['pass'] for k in ks), len(ks))
        res['TCAV (ours)'][c] = (sum(mct[k]['pass'] for k in ks), len(ks))
        kc = [k for k in cap['matched'] if k in ks]
        res['TCAV (Captum)'][c] = (sum(cap['matched'][k]['pass'] for k in kc), len(kc))
        res['Observational'][c] = (sum(ctr[k]['observational_largest'] == MAP[c] for k in ks), len(ks))
    meth = list(METH)
    if HAVE2E:
        meth[0] = ('Counterfactual audit,\nCBM-A controls', C['blue']); res[meth[0][0]] = res['Counterfactual audit']
        meth.append(('Counterfactual audit,\nCBM-B controls', C['sky'])); res[meth[-1][0]] = {}
        c2 = mc2['controls']
        for c in cc:
            ks = [k for k, v in c2.items() if v['concept'] == c and 'note' not in v]
            res[meth[-1][0]][c] = (sum(c2[k]['pass'] for k in ks), len(ks))
    fig = plt.figure(figsize=(W2, 58 * MM))
    gs = GridSpec(1, 3, figure=fig, width_ratios=[1.35, 0.8, 0.9], wspace=0.42)
    ax = fig.add_subplot(gs[0]); letter(ax, 'a', x=-0.1)
    wb = 0.8 / len(meth)
    for mi, (m, col) in enumerate(meth):
        for ci_, c in enumerate(cc):
            p, n = res[m][c]; x = ci_ + (mi - (len(meth) - 1) / 2) * wb
            if n == 0:
                ax.text(x, 0.02, 'n.t.', fontsize=4.9, ha='center', va='bottom', color=col, rotation=90); continue
            ax.bar(x, p / n, wb * 0.92, color=col, lw=0, label=m if ci_ == 0 else None)
            ax.text(x, p / n + 0.02, f'{p}/{n}', fontsize=4.9, ha='center', va='bottom', rotation=90)
    ax.axhline(0.75, color='black', lw=0.6, ls='--'); ax.text(1.01, 0.75, 'rule:\n3 of 4', transform=ax.get_yaxis_transform(), fontsize=5.6, ha='left', va='center', clip_on=False)
    ax.set_xticks(range(4)); ax.set_xticklabels([MAP[c] for c in cc]); ax.set_ylim(0, 1.15); ax.set_yticks([0, .25, .5, .75, 1])
    ax.set_ylabel('Share of matched positive\ncontrols with the target found')
    ax.set_xlabel('Target concept of the control classifier')
    fig.legend([Rectangle((0, 0), 1, 1, color=c_, lw=0) for _, c_ in meth], [m.replace('\n', ' ') for m, _ in meth], loc='lower center',
               ncol=len(meth), bbox_to_anchor=(0.5, 0.97), fontsize=6.0 if len(meth) > 4 else 6.3, handlelength=1.2, columnspacing=0.9)
    # (b) negative control
    tv = json.load(open(os.path.join(P2D, 'tcav/results/tcav_validity.json')))
    neg = {'Counterfactual audit': av['negative']['max_abs_dlogit'] / av['negative']['median_max_abs_dlogit_melanoma_classifiers'],
           'TCAV (ours)': tv['negative']['mean_max_abs_effect'] / tv['negative']['median_mean_max_abs_effect_melanoma_classifiers'],
           'TCAV (Captum)': cap['rule']['negative']['mean_max_abs_effect'] / cap['rule']['negative']['median_mean_max_abs_effect_melanoma_classifiers'],
           'Observational': NUM['observational_negative_ratio']['value']}
    # scale-free version (post hoc): effects divided by the SD of each classifier's logits on the real test images;
    # TCAV scores are signs of directional derivatives and do not depend on output scale
    sct = pd.read_csv(os.path.join(OUT, 'tables/T_negctrl_scale.csv')).set_index('scale')
    sf = {'Counterfactual audit': float(sct.loc['sd_test', 'audit']), 'Observational': float(sct.loc['sd_test', 'observational']),
          'TCAV (ours)': neg['TCAV (ours)'], 'TCAV (Captum)': neg['TCAV (Captum)']}
    ax = fig.add_subplot(gs[1]); letter(ax, 'b', x=-0.1)
    for i, (m, col) in enumerate(METH):
        y = 3 - i
        ax.barh(y + 0.18, neg[m], 0.34, color=col, lw=0); ax.text(neg[m] + 0.02, y + 0.18, f'{neg[m]:.2f}', va='center', fontsize=5.4)
        ax.barh(y - 0.18, sf[m], 0.34, facecolor='white', edgecolor=col, hatch='//////', lw=0.6); ax.text(sf[m] + 0.02, y - 0.18, f'{sf[m]:.2f}', va='center', fontsize=5.4)
    ax.axvline(0.25, color='black', lw=0.6, ls='--'); ax.text(0.235, 3.62, 'limit 0.25', fontsize=5.6, va='bottom', ha='right')
    ax.legend([Rectangle((0, 0), 1, 1, color=C['grey'], lw=0), Rectangle((0, 0), 1, 1, facecolor='white', edgecolor=C['grey'], hatch='//////', lw=0.6)],
              ['logit scale (rule 1)', 'scale-free (post hoc)'], loc='upper right', bbox_to_anchor=(1.04, 1.02), fontsize=5.2, handlelength=1.4, borderaxespad=0.1)
    ax.set_yticks([]); ax.spines['left'].set_visible(False); ax.set_xlim(0, 1.05); ax.set_ylim(-0.6, 4.35)
    ax.set_xlabel('Permuted-label control:\nlargest effect / median of\nmelanoma classifiers')
    # (c) reliability
    rel = {'Counterfactual audit': NUM['audit_reliability']['value'], 'TCAV (ours)': NUM['tcav_reliability']['value'],
           'TCAV (Captum)': {'same architecture, other seed': NUM['captum']['value']['rho_seeds'], 'different classifier': NUM['captum']['value']['rho_cross']},
           'Observational': NUM['observational_reliability']['value']}
    ax = fig.add_subplot(gs[2]); letter(ax, 'c', x=-0.1)
    for i, (m, col) in enumerate(METH):
        a_, b_ = rel[m]['same architecture, other seed'], rel[m]['different classifier']
        ax.plot([0, 1], [a_, b_], color=col, lw=1.0, marker='o', ms=3.5)
    ax.set_xticks([0, 1]); ax.set_xticklabels(['Same architecture,\nother seed', 'Different\nclassifier']); ax.set_xlim(-0.3, 1.3)
    ax.set_ylim(0, 1.02); ax.set_ylabel('Mean Spearman ρ of concept profiles')
    save(fig, 'Fig4_methods')


# ================================================================= Fig 4: audit of the melanoma classifiers
def fig4():
    cls = e.drop_duplicates('classifier')[['classifier', 'label', 'arch', 'seed', 'data']]
    mel = cls[cls.label == 'mel'].copy(); mel['o'] = mel.apply(lambda r: (r['data'] == 'derm7pt', ['resnet50', 'effb0', 'vitb16'].index(r['arch']), r['seed']), axis=1)
    mel = mel.sort_values('o')
    M = np.array([[iv[(k, c)] for c in CN] for k in mel.classifier])
    fig = plt.figure(figsize=(W2, 72 * MM))
    gs = GridSpec(1, 2, figure=fig, width_ratios=[1.0, 1.05], wspace=0.42)
    ax = fig.add_subplot(gs[0]); letter(ax, 'a', x=-0.5, y=1.08)
    REGM = HAVE2E and not gng[gng.Concept == 'Reg'].Decision.eq('GO').any()
    heat(ax, M, 4, [clsname(r) for _, r in mel.iterrows()], [(s + '‡' if (s == 'Reg' and REGM) else s) if s in VALID_KNOB else s + '†' for s in SH], fs=5.8)
    ax.xaxis.tick_top()
    for t, lab in zip(ax.get_xticklabels(), SH): t.set_color('black' if lab in VALID_KNOB else C['grey'])
    ax.axhline(5.5, color='white', lw=1.6)
    ax.text(1.0, -0.035, '† knob not target-dominant (Fig. 2c)', transform=ax.transAxes, fontsize=5.6, color=C['grey'], ha='right', va='top')
    if REGM: ax.text(1.0, -0.33, '‡ audit validated with CBM-A-labelled controls only', transform=ax.transAxes, fontsize=5.6, color=C['grey'], ha='right', va='top')
    cax = ax.inset_axes([0.15, -0.12, 0.7, 0.04]); cbar(fig, cax, 4, 'Change in melanoma logit, level 0 to 1', [-4, -2, 0, 2, 4])
    # (b) valid knobs: melanoma classifiers versus classifiers trained to use the concept
    ax = fig.add_subplot(gs[1]); letter(ax, 'b', x=-0.2, y=1.08)
    ctr = mc['controls']; vk = [('pigment_network', 'PN'), ('vascular_structures', 'VS'), ('pigmentation', 'Pig'), ('regression_structures', 'Reg')]
    for xi, (c, s) in enumerate(vk):
        tv_ = [iv[(k, c)] for k, v in ctr.items() if v['concept'] == c]
        if HAVE2E:
            ax.add_patch(Rectangle((xi - 0.34, min(tv_)), 0.32, max(tv_) - min(tv_), color=C['lgrey'], lw=0, zorder=1))
            tb_ = [iv2[(k, c)] for k, v in mc2['controls'].items() if v['concept'] == c and 'note' not in v]
            if tb_: ax.add_patch(Rectangle((xi + 0.02, min(tb_)), 0.32, max(tb_) - min(tb_), color='#BFE3F5', lw=0, zorder=1))
            else: ax.text(xi + 0.18, 0.2, 'not\ntestable', fontsize=5.0, ha='center', va='bottom', color=C['grey'], zorder=2)
        else:
            ax.add_patch(Rectangle((xi - 0.32, min(tv_)), 0.64, max(tv_) - min(tv_), color=C['lgrey'], lw=0, zorder=1))
        rng = np.random.default_rng(7)
        for _, r in mel.iterrows():
            nm, col, mk = ARCH[r['arch']]
            if r['data'] == 'derm7pt': col = C['grey']
            ax.scatter(xi + rng.uniform(-0.18, 0.18), iv[(r['classifier'], c)], s=14, color=col, marker=mk, lw=0, zorder=3)
        mm = np.mean([iv[(k, c)] for k in mel[mel.data == 'isic2020'].classifier])
        ax.plot([xi - 0.25, xi + 0.25], [mm, mm], color='black', lw=1.0, zorder=4)
    ax.axhline(0, color='black', lw=0.5)
    ax.set_xticks(range(4)); ax.set_xticklabels([LONG[s] for _, s in vk], fontsize=6.2, rotation=0)
    ax.set_xticklabels(['Pigment\nnetwork', 'Vascular\nstructures', 'Pigmentation', 'Regression\nstructures' + ('‡' if REGM else '')])
    ax.set_ylabel('Change in logit, knob level 0 to 1\n(mean of the two realistic bases)'); ax.set_ylim(-1.5, 10)
    hs = [Line2D([], [], marker=ARCH[a][2], ls='', color=ARCH[a][1], ms=4) for a in ['resnet50', 'effb0', 'vitb16']] + \
         [Line2D([], [], marker='o', ls='', color=C['grey'], ms=4), Line2D([], [], color='black', lw=1.0),
          Rectangle((0, 0), 1, 1, color=C['lgrey'], lw=0)] + ([Rectangle((0, 0), 1, 1, color='#BFE3F5', lw=0)] if HAVE2E else [])
    ax.legend(hs, ['ResNet-50 (ISIC 2020)', 'EfficientNet-B0', 'ViT-B/16', 'ResNet-50 (Derm7pt)', 'Mean of the six ISIC 2020 classifiers',
                   'Range of matched controls,\nCBM-A labels (n = 4)' if HAVE2E else 'Range of matched positive controls\n(target concept, n = 4)']
              + (['Range of matched controls,\nCBM-B labels (n = 4)'] if HAVE2E else []), loc='lower center', fontsize=5.6, ncol=2, bbox_to_anchor=(0.45, 1.0),
              columnspacing=0.8, handlelength=1.2)
    save(fig, 'Fig5_melanoma_audit')


# ================================================================= Fig 5: observational versus interventional
def fig5():
    cal = pd.read_csv(os.path.join(OUT, 'tables/T_calibrated_obs_vs_intervention.csv'))
    K = NUM['calibration_ratios']['value']['median']
    fig, axs = plt.subplots(1, 4, figsize=(W2, 52 * MM), sharex=True, sharey=True)
    plt.subplots_adjust(wspace=0.12)
    for i, s in enumerate(['PN', 'VS', 'Pig', 'Reg']):
        ax = axs[i]; letter(ax, 'abcd'[i], x=-0.06, y=1.0)
        ax.plot([-2.5, 5], [-2.5, 5], color=C['grey'], lw=0.6, ls='--', zorder=1)
        ax.axhline(0, color='black', lw=0.4, zorder=1); ax.axvline(0, color='black', lw=0.4, zorder=1)
        for _, r in cal.iterrows():
            nm = r['Classifier']; a = 'resnet50' if nm.startswith('ResNet') else 'effb0' if nm.startswith('Eff') else 'vitb16'
            col = C['grey'] if 'Derm7pt' in nm else ARCH[a][1]
            ax.scatter(r[f'{s} expected'], r[f'{s} measured'], s=16, marker=ARCH[a][2], color=col, lw=0, zorder=3)
        ax.set_title(f"{LONG[s]}{'‡' if (s == 'Reg' and HAVE2E and not gng[gng.Concept == 'Reg'].Decision.eq('GO').any()) else ''}\nratio {K[s]:.2f}", fontsize=6.6, pad=3)
        ax.set_xlim(-2.0, 4.6); ax.set_ylim(-2.0, 4.6); ax.set_aspect('equal')
        if i == 0: ax.set_ylabel('Measured by intervention\n(change in logit)')
    fig.text(0.5, 0.02, 'Expected from the observational coefficient (median ratio of the matched controls × coefficient)', fontsize=6.6, ha='center', va='top')
    hs = [Line2D([], [], marker=ARCH[a][2], ls='', color=ARCH[a][1], ms=4) for a in ['resnet50', 'effb0', 'vitb16']] + \
         [Line2D([], [], marker='o', ls='', color=C['grey'], ms=4), Line2D([], [], color=C['grey'], ls='--', lw=0.6)]
    fig.legend(hs, ['ResNet-50', 'EfficientNet-B0', 'ViT-B/16', 'ResNet-50 (Derm7pt)', 'Expected = measured'], loc='upper center', ncol=5,
               bbox_to_anchor=(0.5, 1.08), fontsize=6)
    save(fig, 'Fig6_observational_vs_intervention')


if __name__ == '__main__':
    os.makedirs(FIG, exist_ok=True)
    for f in [fig1, fig2_2e if HAVE2E else fig2, fig3, fig4, fig5]:
        f(); print(f.__name__, 'ok')
