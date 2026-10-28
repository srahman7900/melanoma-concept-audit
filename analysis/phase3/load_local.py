"""Load the saved audit data (no GPU, no images): sweep metadata, evaluator outputs, classifier logits."""
import os, json, glob
import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ROOT = os.environ.get('A3_ROOT', os.path.join(_REPO, 'figure_inputs'))
SHORT = ['PN', 'BWV', 'VS', 'Pig', 'Str', 'D/G', 'Reg']
CONCEPTS = ['pigment_network', 'blue_whitish_veil', 'vascular_structures', 'pigmentation', 'streaks',
            'dots_and_globules', 'regression_structures']
TARGET = {'mpn': 0, 'mvs': 2, 'mpig': 3, 'mreg': 6, 'bpn': 0, 'bvs': 2, 'bpig': 3}
CLIP = 15.0
SETS = ['rsweep_ben', 'rsweep_mel']


def score_dirs():
    return [os.path.join(ROOT, 'p2ainp/phase2/scores2_fp32'), os.path.join(ROOT, 'p2cout/results'),
            os.path.join(ROOT, 'p2dout/results'), os.path.join(ROOT, 'p2eout/results')] + \
           [d for d in os.environ.get('A3_EXTRA_SCORES', os.path.join(ROOT, 'p3out', 'results')).split(':') if d]


def classifiers():
    out = {}
    for d in score_dirs():
        for f in sorted(glob.glob(os.path.join(d, '*.npz'))):
            n = os.path.basename(f)[:-4]
            if n.startswith('pred_') or n == 'cbmb_labels': continue
            if not (n.startswith('mel_') or n.startswith('ctrl_')): continue
            out[n] = f
    return out


def kind(name):
    if name.startswith('mel_'): return 'mel'
    lab = name.split('_')[1]
    if lab == 'perm': return 'perm'
    if lab in ('pig', 'bwv'): return 'quartile'
    if lab.startswith('m'): return 'ctrlA'
    if lab.startswith('b'): return 'ctrlB'
    return 'other'


def logits(path):
    z = np.load(path)
    return {k: np.clip(np.where(np.isfinite(z[k]), z[k], np.nan).astype(np.float64), -CLIP, CLIP) for k in z.files}


def _phase(sname):
    return 'phase2' if sname.startswith('rsweep') or sname == 'cls_test' else 'phase1d'


def meta(sname):
    m = json.load(open(os.path.join(ROOT, 'p2ainp', _phase(sname), 'sets_meta', f'{sname}.json')))['meta']
    return {k: np.asarray(v) for k, v in m.items()}


def evaluators(sname):
    A = np.load(os.path.join(ROOT, 'p2ainp', _phase(sname), 'scores', f'A_{sname}.npz'))
    B = np.load(os.path.join(ROOT, 'p2ainp', _phase(sname), 'scores', f'B_{sname}.npz'))
    return {'cbmb': B['cbmb'].astype(np.float64), 'lab': A['lab'].astype(np.float64), 'cbma': A['cbma'].astype(np.float64)}


def cbmb_sd():
    B = np.load(os.path.join(ROOT, 'p2ainp/phase2/scores', 'B_cls_test.npz'))['cbmb']
    return B.std(0) + 1e-9


def lab_test():
    A = np.load(os.path.join(ROOT, 'p2ainp/phase2/scores', 'A_cls_test.npz'))
    return A['lab'].astype(np.float64)


def traj_index(m, j):
    """(n_noise, n_levels) index array for concept j"""
    sel = np.where(m['concept_idx'] == j)[0]
    lev = np.unique(m['level'][sel]); nid = np.unique(m['noise_id'][sel])
    T = np.full((len(nid), len(lev)), -1, int)
    T[np.searchsorted(nid, m['noise_id'][sel]), np.searchsorted(lev, m['level'][sel])] = sel
    assert (T >= 0).all()
    return T, lev, nid
