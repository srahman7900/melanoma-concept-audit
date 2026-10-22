# =============================================================================
# ClinDiffXAI  PHASE 1b                              (one Kaggle cell, GPU T4 x2)
# =============================================================================
# WHAT THIS CELL DOES
#   Phase 1b completes Phase 1 and adds three analyses that the Phase 1 results
#   showed are needed.
#   1. ISIC 2018 Task 2 attribute detector (ResNet-50, independent data). This
#      gives Gate 1 check G1d, as the rule fixed before Phase 1 requires.
#   2. EXPLORATORY (decided after seeing Phase 1): the main model conditioned on
#      concept vectors of REAL training images instead of the p90/p10 prototypes,
#      1,000 + 1,000 images with the same noises as Phase 1. Tests whether the
#      realism loss comes from the extreme prototype.
#   3. Colour fidelity (CIELAB) of real and generated images.
#   4. Cross-talk of the dose-response versus real-image concept co-occurrence.
#   The complete analysis (all Phase 1 tables and figures plus the new ones) is
#   then run again, and everything is saved in phase1b_outputs.zip.
#
# REUSE OF PHASE 1
#   Attach the output of your committed Phase 1 notebook (Add Input -> Notebooks).
#   The cell then copies the saved images, scores and CBM-B, and runs only the new
#   jobs (my estimate: 45 to 75 minutes). If no Phase 1 output is attached, it
#   makes every image again from the same fixed noise seeds (my estimate: 2.5 to
#   3.5 hours). Both are estimates, not measurements.
#
# INPUTS TO ATTACH
#   Phase 1 notebook output (recommended), the two epoch-150 checkpoints, the two
#   paper checkpoints, CBM-A v3, the pseudo-label CSV, ISIC 2020 images, Derm7pt,
#   ISIC 2018 Task 2 masks (rohanawan/isic-2018-task-2) and ISIC 2018 images
#   (tschandl/isic2018-challenge-task1-data-segmentation).
#   Settings: Accelerator GPU T4 x2, Internet ON. Run with "Save Version ->
#   Save & Run All (Commit)".
#
# GATE 1: same rule as Phase 1, applied again to the complete data:
#   PASS if CBM-B shows a positive, significant dose-response for BOTH
#   pigmentation and pigment network from the melanoma prototype (two-sided
#   Wilcoxon signed-rank test on per-trajectory slopes, BH-corrected p < 0.05,
#   median slope > 0) AND the non-CBM checks agree (lesion darkness for
#   pigmentation, ISIC 2018 detector for pigment network). INCONCLUSIVE if the
#   CBM-B test AUC (128 pipeline) is below 0.70 for either concept. PARTIAL if
#   CBM-B passes but a non-CBM check fails. FAIL otherwise.
# =============================================================================

DRY_RUN     = False
FORCE_RERUN = False

KNOWN_PATHS = {   # from your NB1 / NB4 runs; the cell searches by file name if these move
    'cbma':         '/kaggle/input/datasets/shohanurrahman7/best-cbm-model-128/best_cbm_model.pth',
    'main_best':    '/kaggle/input/datasets/shohanoor7900/best-clindiffxai-v3-tpu/best_clindiffxai_v3_tpu.pth',
    'base_best':    '/kaggle/input/datasets/shohanoor7900/best-clindiffxai-baseline-tpu/best_clindiffxai_baseline_tpu.pth',
    'pseudo_csv':   '/kaggle/input/datasets/shohanurrahman7/isic2020-pseudo-labels-v3/isic2020_pseudo_labels_v3.csv',
    'isic2020_dir': '/kaggle/input/datasets/sumaiyabinteshahid/isic-challenge-dataset-2020/ISIC_2020_Dataset/train',
    'derm7pt_csv':  '/kaggle/input/datasets/shohanoor7900/derm7points/upload/derm7pt/derm7pt.zip/meta/meta.csv',
    'derm7pt_img_dir':   '/kaggle/input/datasets/shohanoor7900/derm7points/upload/derm7pt/derm7pt.zip/images',
    'derm7pt_train_idx': '/kaggle/input/datasets/shohanoor7900/derm7points/upload/derm7pt/derm7pt.zip/meta/train_indexes.csv',
    'derm7pt_valid_idx': '/kaggle/input/datasets/shohanoor7900/derm7points/upload/derm7pt/derm7pt.zip/meta/valid_indexes.csv',
    'derm7pt_test_idx':  '/kaggle/input/datasets/shohanoor7900/derm7points/upload/derm7pt/derm7pt.zip/meta/test_indexes.csv',
}

import os, sys, json, time, glob, shutil, subprocess, threading, re
import numpy as np, pandas as pd

# testing overrides for an offline dry run on another machine; leave unset on Kaggle
INPUT_ROOT = os.environ.get('P1_INPUT_ROOT', '/kaggle/input')
WORK_ROOT = os.environ.get('P1_WORK_ROOT', '/kaggle/working')
PRETRAINED = os.environ.get('P1_NO_PRETRAINED') is None
IMAGE_SIZE = int(os.environ.get('P1_IMAGE_SIZE', '128'))
EVAL_SIZE = int(os.environ.get('P1_EVAL_SIZE', '224'))
INCEP_SIZE = int(os.environ.get('P1_INCEP_SIZE', '299'))
if os.environ.get('P1_INPUT_ROOT'):
    KNOWN_PATHS = {k: v.replace('/kaggle/input', INPUT_ROOT) for k, v in KNOWN_PATHS.items()}

WORK = os.path.join(WORK_ROOT, 'phase1b_dryrun' if DRY_RUN else 'phase1b')
CODE = os.path.join(WORK, 'code'); OUTD = os.path.join(WORK, 'outputs')
for d in [WORK, CODE, OUTD, os.path.join(WORK, 'logs'), os.path.join(WORK, 'done')]:
    os.makedirs(d, exist_ok=True)
if FORCE_RERUN:
    for d in ['done', 'sets', 'scores', 'evaluators', 'results']:
        shutil.rmtree(os.path.join(WORK, d), ignore_errors=True)
    os.makedirs(os.path.join(WORK, 'done'), exist_ok=True)


def say(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------- 0. packages
say('installing diffusers==0.33.0 and accelerate==1.6.0 (same pins as NB4)')
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'diffusers==0.33.0', 'accelerate==1.6.0'], check=False)

# ---------------------------------------------------------------- 1. find inputs
say(f'indexing {INPUT_ROOT} ...')
NAME_INDEX = {}
for dp, dn, fn in os.walk(INPUT_ROOT):
    for f in fn:
        NAME_INDEX.setdefault(f, []).append(os.path.join(dp, f))


def by_name(name, prefer=(), avoid=()):
    hits = [p for p in NAME_INDEX.get(name, []) if not any(a in p for a in avoid)]
    for pref in prefer:
        h = [p for p in hits if pref in p]
        if h: return h[0], len(hits)
    return (hits[0] if hits else None), len(hits)


paths, notes = {}, []
for key, fname, prefer, avoid in [
        ('cbma', 'best_cbm_model.pth', ('cbma-v3',), ('best-cbm-model-new',)),
        ('main_best', 'best_clindiffxai_v3_tpu.pth', (), ()),
        ('base_best', 'best_clindiffxai_baseline_tpu.pth', (), ()),
        ('pseudo_csv', 'isic2020_pseudo_labels_v3.csv', (), ())]:
    if os.path.exists(KNOWN_PATHS[key]):
        paths[key] = KNOWN_PATHS[key]
    else:
        p, n = by_name(fname, prefer, avoid); paths[key] = p
        if p: notes.append(f"{key}: known path missing, using {p} ({n} candidate(s))")
for key, fname in [('main_e150', 'checkpoint_tpu_epoch150.pth'), ('base_e150', 'checkpoint_baseline_tpu_epoch150.pth')]:
    paths[key], n = by_name(fname)
    if n > 1: notes.append(f"{key}: {n} files named {fname}; using {paths[key]}. Detach the others if this is wrong.")
# ISIC 2020 image folder
if os.path.isdir(KNOWN_PATHS['isic2020_dir']):
    paths['isic2020_dir'] = KNOWN_PATHS['isic2020_dir']
else:
    paths['isic2020_dir'] = None
    if paths['pseudo_csv']:
        probe = pd.read_csv(paths['pseudo_csv'], nrows=5)['image_name'].iloc[0] + '.jpg'
        p, _ = by_name(probe)
        if p: paths['isic2020_dir'] = os.path.dirname(p); notes.append(f"isic2020_dir found at {paths['isic2020_dir']}")
# Derm7pt
if os.path.exists(KNOWN_PATHS['derm7pt_csv']):
    for k in ['derm7pt_csv', 'derm7pt_img_dir', 'derm7pt_train_idx', 'derm7pt_valid_idx', 'derm7pt_test_idx']:
        paths[k] = KNOWN_PATHS[k]
else:
    cands = [p for p in NAME_INDEX.get('meta.csv', []) if os.path.exists(os.path.join(os.path.dirname(p), 'train_indexes.csv'))]
    if cands:
        md = os.path.dirname(cands[0]); paths['derm7pt_csv'] = cands[0]
        paths['derm7pt_img_dir'] = os.path.join(os.path.dirname(md), 'images')
        for k in ['train', 'valid', 'test']:
            paths[f'derm7pt_{k}_idx'] = os.path.join(md, f'{k}_indexes.csv')
        notes.append(f"Derm7pt found at {md}")
# ISIC 2018 Task 2: attribute masks + training images
ATTRS = ['pigment_network', 'negative_network', 'streaks', 'milia_like_cyst', 'globules']
masks = {}
for f, plist in NAME_INDEX.items():
    if '_attribute_' in f and f.endswith('.png'):
        iid, a = f[:-4].split('_attribute_', 1)
        if a in ATTRS:
            masks.setdefault(iid, {})[a] = plist[0]
rows = []
for iid, mdict in masks.items():
    if len(mdict) < 5: continue
    img = None
    for ext in ['.jpg', '.jpeg', '.png', '.JPG']:
        cand = [p for p in NAME_INDEX.get(iid + ext, []) if '_attribute_' not in p and '_segmentation' not in p
                and not (paths.get('isic2020_dir') and p.startswith(paths['isic2020_dir']))]
        if cand:
            pref = [p for p in cand if ('Task1-2' in p or 'Training_Input' in p or '2018' in p)]
            img = (pref or cand)[0]; break
    if img:
        rows.append({'id': iid, 'img': img, **{f'm_{a}': mdict[a] for a in ATTRS}})
min_isic = 30 if DRY_RUN else 300
if len(rows) >= min_isic:
    paths['isic18_index'] = os.path.join(WORK, 'isic18_index.csv'); pd.DataFrame(rows).to_csv(paths['isic18_index'], index=False)
else:
    paths['isic18_index'] = None
    if rows: notes.append(f"ISIC 2018 Task 2: only {len(rows)} complete image+mask sets found; detector skipped")

print('\n' + '=' * 78 + '\nINPUTS\n' + '=' * 78)
critical = ['cbma', 'main_best', 'base_best', 'pseudo_csv', 'isic2020_dir', 'derm7pt_csv']
for k in critical + ['main_e150', 'base_e150', 'isic18_index']:
    v = paths.get(k)
    print(f"  {'OK ' if v else '-- '} {k:<13} {v if v else 'NOT FOUND'}")
if paths.get('isic18_index'):
    print(f"      ISIC 2018 Task 2 images with all 5 masks: {len(rows)}")
for n in notes: print('  note:', n)
missing = [k for k in critical if not paths.get(k)]
if missing:
    raise FileNotFoundError(f"Required inputs not found: {missing}. Attach the datasets or fix KNOWN_PATHS.")
if not paths.get('main_e150') or not paths.get('base_e150'):
    print("  WARNING: an epoch-150 checkpoint is missing. The new evaluation then uses the paper checkpoints\n"
          "           (main epoch 149, baseline epoch 120) for that model, and the report says so.")

# ---------------------------------------------------------------- 2. code files
FILES = {
    'common.py': r'''"""ClinDiffXAI Phase 1: shared code (architectures, loaders, generation, scoring).

The U-Net, the CBM-A class and the supplement generation function are copied
verbatim from NB4 / NB4_Stats_Supplement so that the saved checkpoints load
with strict key matching and the reproduction run follows the original code.
"""
import os, sys, json, time, glob, math, warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models
from torchvision import transforms
import torchvision.transforms.functional as TF
warnings.filterwarnings('ignore')

CONCEPT_NAMES = ['pigment_network', 'blue_whitish_veil', 'vascular_structures',
                 'pigmentation', 'streaks', 'dots_and_globules', 'regression_structures']
SHORT = ['PN', 'BWV', 'VS', 'Pig', 'Str', 'D/G', 'Reg']
MEL_VEC_LIST = [0.8292, 0.7781, 0.9926, 0.8918, 0.4585, 0.9256, 0.8769]
BEN_VEC_LIST = [0.0800, 0.0002, 0.0047, 0.0055, 0.0018, 0.1501, 0.0003]
ISIC18_ATTRS = ['pigment_network', 'negative_network', 'streaks', 'milia_like_cyst', 'globules']
# ISIC 2018 attribute -> Derm7pt concept it measures (only 3 overlap)
ISIC18_TO_DERM = {'pigment_network': 'pigment_network', 'streaks': 'streaks',
                  'globules': 'dots_and_globules'}
IMNET_MEAN = [0.485, 0.456, 0.406]
IMNET_STD = [0.229, 0.224, 0.225]


# ----------------------------------------------------------------------------
# small utilities
# ----------------------------------------------------------------------------
def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_cfg(path):
    with open(path) as f:
        return json.load(f)


def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def pretrained_ok(cfg):
    return bool(cfg.get('pretrained', True))


def save_json(obj, path):
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(obj, f, indent=2, default=_json_default)
    os.replace(tmp, path)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def save_set(path, imgs_u8, meta, info):
    """imgs_u8: uint8 tensor (N,3,H,W). meta: dict of equal-length lists."""
    tmp = path + '.tmp'
    torch.save({'imgs': imgs_u8.contiguous(), 'meta': meta, 'info': info}, tmp)
    os.replace(tmp, path)


def load_set(path):
    return torch.load(path, map_location='cpu', weights_only=False)


def float_to_u8(x01):
    return (x01.clamp(0, 1) * 255.0).round().to(torch.uint8)


# ----------------------------------------------------------------------------
# U-Net (verbatim from NB4 v3; identical to NB3 / NB3b / TPU scripts)
# ----------------------------------------------------------------------------
class ConceptEmbedding(nn.Module):
    def __init__(self, n, d=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n, d), nn.SiLU(), nn.Linear(d, d), nn.SiLU())

    def forward(self, c):
        return self.net(c)


class ResBlock(nn.Module):
    def __init__(self, ic, oc, td, cd):
        super().__init__()
        self.n1 = nn.GroupNorm(8, ic); self.c1 = nn.Conv2d(ic, oc, 3, padding=1)
        self.n2 = nn.GroupNorm(8, oc); self.c2 = nn.Conv2d(oc, oc, 3, padding=1)
        self.tp = nn.Linear(td, oc); self.cp = nn.Linear(cd, oc)
        self.sk = nn.Conv2d(ic, oc, 1) if ic != oc else nn.Identity(); self.a = nn.SiLU()

    def forward(self, x, t, c):
        h = self.c1(self.a(self.n1(x)))
        h = h + self.tp(self.a(t))[:, :, None, None]
        h = h + self.cp(self.a(c))[:, :, None, None]
        return self.c2(self.a(self.n2(h))) + self.sk(x)


class SelfAttention(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.n = nn.GroupNorm(8, ch)
        self.a = nn.MultiheadAttention(ch, num_heads=4, batch_first=True)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.n(x).reshape(B, C, H * W).permute(0, 2, 1)
        h, _ = self.a(h, h, h); return x + h.permute(0, 2, 1).reshape(B, C, H, W)


class ConceptConditionedUNet128(nn.Module):
    def __init__(self, image_size=128, num_concepts=7, base_channels=64):
        super().__init__()
        self.image_size = image_size; C = [64, 64, 128, 256, 256]
        td = base_channels * 4; cd = base_channels * 2
        self.time_mlp = nn.Sequential(nn.Linear(base_channels, td), nn.SiLU(), nn.Linear(td, td))
        self.concept_emb = ConceptEmbedding(num_concepts, cd)
        self.enc_in = nn.Conv2d(3, C[0], 3, padding=1)
        self.enc = nn.ModuleList([
            nn.ModuleList([ResBlock(C[0], C[0], td, cd), ResBlock(C[0], C[0], td, cd)]),
            nn.ModuleList([ResBlock(C[0], C[1], td, cd), ResBlock(C[1], C[1], td, cd)]),
            nn.ModuleList([ResBlock(C[1], C[2], td, cd), ResBlock(C[2], C[2], td, cd), SelfAttention(C[2])]),
            nn.ModuleList([ResBlock(C[2], C[3], td, cd), ResBlock(C[3], C[3], td, cd), SelfAttention(C[3])]),
            nn.ModuleList([ResBlock(C[3], C[4], td, cd), ResBlock(C[4], C[4], td, cd), SelfAttention(C[4])]),
        ])
        self.down = nn.ModuleList([nn.Conv2d(C[i], C[i], 3, stride=2, padding=1) for i in range(4)])
        self.mid = nn.ModuleList([ResBlock(C[4], C[4], td, cd), SelfAttention(C[4]), ResBlock(C[4], C[4], td, cd)])
        self.dec = nn.ModuleList([
            nn.ModuleList([ResBlock(C[4] + C[3], C[3], td, cd), ResBlock(C[3], C[3], td, cd), SelfAttention(C[3])]),
            nn.ModuleList([ResBlock(C[3] + C[2], C[2], td, cd), ResBlock(C[2], C[2], td, cd), SelfAttention(C[2])]),
            nn.ModuleList([ResBlock(C[2] + C[1], C[1], td, cd), ResBlock(C[1], C[1], td, cd)]),
            nn.ModuleList([ResBlock(C[1] + C[0], C[0], td, cd), ResBlock(C[0], C[0], td, cd)]),
        ])
        self.up = nn.ModuleList([nn.ConvTranspose2d(C[4], C[4], 2, stride=2), nn.ConvTranspose2d(C[3], C[3], 2, stride=2),
                                 nn.ConvTranspose2d(C[2], C[2], 2, stride=2), nn.ConvTranspose2d(C[1], C[1], 2, stride=2)])
        self.out = nn.Sequential(nn.GroupNorm(8, C[0]), nn.SiLU(), nn.Conv2d(C[0], 3, 3, padding=1))

    def sin_emb(self, t, d):
        half = d // 2; f = torch.exp(-torch.arange(half, device=t.device) * (np.log(10000) / (half - 1)))
        a = t[:, None].float() * f[None]; return torch.cat([torch.cos(a), torch.sin(a)], dim=-1)

    def forward(self, x, ts, concepts):
        te = self.time_mlp(self.sin_emb(ts, self.time_mlp[0].in_features))
        ce = self.concept_emb(concepts); h = self.enc_in(x); sk = []
        for i, blks in enumerate(self.enc):
            for b in blks: h = b(h) if isinstance(b, SelfAttention) else b(h, te, ce)
            sk.append(h)
            if i < 4: h = self.down[i](h)
        for b in self.mid: h = b(h) if isinstance(b, SelfAttention) else b(h, te, ce)
        usable_sk = sk[:-1]
        for i, (blks, up) in enumerate(zip(self.dec, self.up)):
            h = up(h); s = usable_sk[-(i + 1)]
            if h.shape != s.shape: h = F.interpolate(h, size=s.shape[-2:])
            h = torch.cat([h, s], dim=1)
            for b in blks: h = b(h) if isinstance(b, SelfAttention) else b(h, te, ce)
        o = self.out(h)
        if o.shape[-1] != self.image_size: o = F.interpolate(o, size=(self.image_size, self.image_size))
        return o


def _strip_module(sd):
    return {(k[7:] if k.startswith('module.') else k): v for k, v in sd.items()}


def load_diffusion(path, num_concepts, device, image_size=128):
    """Loads EMA weights when present (same rule as NB4 / supplement)."""
    ck = torch.load(path, map_location='cpu', weights_only=False)
    m = ConceptConditionedUNet128(image_size, num_concepts).to(device).eval()
    used = 'ema' if 'ema_state_dict' in ck else 'raw'
    sd = ck['ema_state_dict'] if used == 'ema' else ck['model_state_dict']
    m.load_state_dict(_strip_module(sd), strict=True)
    for p in m.parameters():
        p.requires_grad_(False)
    info = {'path': path, 'weights': used, 'epoch_1based': int(ck.get('epoch', -2)) + 1,
            'num_concepts_in_ckpt': ck.get('num_concepts'), 'model_type': ck.get('model_type', 'concept'),
            'val_loss': ck.get('val_loss')}
    log(f"  loaded diffusion ({used}) epoch {info['epoch_1based']} from {path}")
    return m, info


# ----------------------------------------------------------------------------
# CBM-A (verbatim structure from NB1 / NB4)
# ----------------------------------------------------------------------------
class CBM_A(nn.Module):
    def __init__(self):
        super().__init__()
        bb = tv_models.efficientnet_v2_s(weights=None)
        in_f = bb.classifier[1].in_features; bb.classifier = nn.Identity(); self.backbone = bb
        self.concept_feature = nn.Sequential(nn.Linear(in_f, 512), nn.LayerNorm(512), nn.GELU(), nn.Dropout(0.3))
        self.concept_heads = nn.ModuleList([nn.Sequential(nn.Linear(512, 128), nn.GELU(), nn.Dropout(0.2), nn.Linear(128, 1)) for _ in range(7)])
        self.diagnosis_head = nn.Sequential(nn.Linear(7, 64), nn.GELU(), nn.Dropout(0.3), nn.Linear(64, 2))

    def forward(self, x):
        f = self.backbone(x)
        if f.dim() > 2: f = f.mean([-2, -1])
        cf = self.concept_feature(f)
        cl = torch.cat([h(cf) for h in self.concept_heads], dim=1)
        return cl, torch.sigmoid(cl), self.diagnosis_head(torch.sigmoid(cl))

    def forward_all(self, x):
        """concept probs, 512-d concept_feature (the NB4 MedFID feature), diagnosis prob"""
        f = self.backbone(x)
        if f.dim() > 2: f = f.mean([-2, -1])
        cf = self.concept_feature(f)
        cl = torch.cat([h(cf) for h in self.concept_heads], dim=1)
        cp = torch.sigmoid(cl)
        dp = torch.softmax(self.diagnosis_head(cp), dim=1)[:, 1]
        return cp, cf, dp


def load_cbm_a(path, device):
    m = CBM_A(); ck = torch.load(path, map_location='cpu', weights_only=False)
    m.load_state_dict(_strip_module(ck['model_state_dict']), strict=True)
    log(f"  loaded CBM-A (stored epoch index {ck.get('epoch', '?')}) from {path}")
    return m.to(device).eval(), {'path': path, 'epoch_index': ck.get('epoch')}


# ----------------------------------------------------------------------------
# Independent evaluators
# ----------------------------------------------------------------------------
class ConceptNetB(nn.Module):
    """CBM-B: ConvNeXt-Tiny with 7 sigmoid concept outputs (no bottleneck needed)."""
    def __init__(self, pretrained=True, n_out=7):
        super().__init__()
        w = tv_models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
        m = tv_models.convnext_tiny(weights=w)
        self.features = m.features; self.avgpool = m.avgpool; self.norm = m.classifier[0]
        self.drop = nn.Dropout(0.2); self.head = nn.Linear(768, n_out)

    def embed(self, x):
        f = self.avgpool(self.features(x)); f = self.norm(f); return torch.flatten(f, 1)

    def forward(self, x):
        return self.head(self.drop(self.embed(x)))


class AttrNet(nn.Module):
    """ISIC 2018 Task 2 attribute-presence detector: ResNet-50, 5 sigmoid outputs."""
    def __init__(self, pretrained=True, n_out=5):
        super().__init__()
        w = tv_models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        m = tv_models.resnet50(weights=w); m.fc = nn.Identity(); self.body = m
        self.drop = nn.Dropout(0.2); self.head = nn.Linear(2048, n_out)

    def forward(self, x):
        return self.head(self.drop(self.body(x)))


def build_inception(cfg, device):
    w = tv_models.Inception_V3_Weights.DEFAULT if pretrained_ok(cfg) else None
    if w is None:
        net = tv_models.inception_v3(weights=None, aux_logits=False, transform_input=False, init_weights=True)
    else:
        net = tv_models.inception_v3(weights=w, transform_input=False)
    net.fc = nn.Identity()
    return net.eval().to(device)


# ----------------------------------------------------------------------------
# Generation
# ----------------------------------------------------------------------------
def make_ddim(steps):
    from diffusers import DDIMScheduler
    ddim = DDIMScheduler(num_train_timesteps=1000, beta_schedule='linear', clip_sample=True)
    ddim.set_timesteps(steps)
    return ddim


@torch.no_grad()
def generate_batch_supplement(model, cvec, n, image_size, device, cfg_scale=3.5, steps=50, bs=16):
    """VERBATIM logic of NB4_Stats_Supplement.generate_batch (global CUDA RNG,
    separate cond / uncond passes, batch 16). Used only for the reproduction run."""
    from diffusers import DDIMScheduler
    ddim = DDIMScheduler(num_train_timesteps=1000, beta_schedule='linear', clip_sample=True)
    ddim.set_timesteps(steps); all_imgs = []
    for start in range(0, n, bs):
        b = min(bs, n - start); imgs = torch.randn(b, 3, image_size, image_size, device=device)
        cv = cvec.unsqueeze(0).repeat(b, 1).to(device)
        for t in ddim.timesteps:
            ts = torch.full((b,), t, device=device, dtype=torch.long)
            eps_c = model(imgs, ts, cv); eps_u = model(imgs, ts, torch.zeros_like(cv))
            imgs = ddim.step(eps_u + cfg_scale * (eps_c - eps_u), t, imgs).prev_sample
        all_imgs.append(((imgs.clamp(-1, 1) + 1) / 2).cpu())
    return torch.cat(all_imgs, 0)


def cpu_noise(n, image_size, seed):
    """Initial noise drawn on CPU with its own generator: identical on any GPU/TPU."""
    g = torch.Generator().manual_seed(int(seed))
    return torch.randn(n, 3, image_size, image_size, generator=g)


@torch.no_grad()
def generate_crn(model, cond, noise, device, cfg_scale=3.5, steps=50, bs=32):
    """Classifier-free-guided DDIM (eta=0, deterministic) from given noise.
    cond: (N,d) float; noise: (N,3,H,W) float. Returns uint8 (N,3,H,W) on CPU."""
    ddim = make_ddim(steps); out = []
    N = noise.shape[0]
    for s in range(0, N, bs):
        x = noise[s:s + bs].to(device); c = cond[s:s + bs].to(device); b = x.shape[0]
        z = torch.zeros_like(c)
        for t in ddim.timesteps:
            ts = torch.full((b,), int(t), device=device, dtype=torch.long)
            if cfg_scale == 0.0:
                eps = model(x, ts, z)
            elif cfg_scale == 1.0:
                eps = model(x, ts, c)
            else:
                e = model(torch.cat([x, x]), torch.cat([ts, ts]), torch.cat([c, z]))
                e_c, e_u = e.chunk(2)
                eps = e_u + cfg_scale * (e_c - e_u)
            x = ddim.step(eps, t, x).prev_sample
        out.append(float_to_u8((x.clamp(-1, 1) + 1) / 2).cpu())
    return torch.cat(out, 0)


# ----------------------------------------------------------------------------
# Evaluator input pipeline. Every evaluator sees images that passed through the
# diffusion resolution (IMAGE_SIZE) and are then resized to EVAL_SIZE, so real
# anchors and generated images share one pipeline.
# ----------------------------------------------------------------------------
def u8_to_eval(u8, device, eval_size=224):
    x = u8.to(device, non_blocking=True).float() / 255.0
    x = TF.resize(x, [eval_size, eval_size], antialias=True)
    return TF.normalize(x, IMNET_MEAN, IMNET_STD)


def u8_to_inception(u8, device, incep_size=299):
    x = u8.to(device, non_blocking=True).float() / 255.0
    x = TF.resize(x, [incep_size, incep_size], antialias=True)
    return TF.normalize(x, [0.5] * 3, [0.5] * 3)   # same as NB4


# ----------------------------------------------------------------------------
# Non-learned colour measurement (pigmentation check)
# ----------------------------------------------------------------------------
def srgb_u8_to_Lstar(u8):
    """(N,3,H,W) uint8 -> (N,H,W) CIE L* (D65). numpy only."""
    c = u8.astype(np.float32) / 255.0
    lin = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    Y = 0.2126 * lin[:, 0] + 0.7152 * lin[:, 1] + 0.0722 * lin[:, 2]
    f = np.where(Y > 0.008856, np.cbrt(Y), 7.787 * Y + 16.0 / 116.0)
    return 116.0 * f - 16.0


def _otsu(v, nbins=128):
    hist, edges = np.histogram(v, bins=nbins)
    hist = hist.astype(np.float64); mids = (edges[:-1] + edges[1:]) / 2
    w0 = np.cumsum(hist); w1 = w0[-1] - w0
    m0 = np.cumsum(hist * mids) / np.maximum(w0, 1e-9)
    m1 = (np.sum(hist * mids) - np.cumsum(hist * mids)) / np.maximum(w1, 1e-9)
    between = w0 * w1 * (m0 - m1) ** 2
    return mids[int(np.argmax(between))]


DARK_COLS = ['lesion_darkness', 'lesion_contrast', 'lesion_area_frac', 'global_darkness']


def darkness_metrics(u8):
    """Per image: lesion darkness = 100 - mean L* inside the lesion (Otsu on L*
    inside a central disk, largest connected component); contrast = skin L* -
    lesion L*; area fraction of the disk; global darkness = 100 - mean L*."""
    from scipy import ndimage
    L = srgb_u8_to_Lstar(u8)
    N, H, W = L.shape
    yy, xx = np.mgrid[0:H, 0:W]
    disk = ((yy - (H - 1) / 2) ** 2 + (xx - (W - 1) / 2) ** 2) <= (0.46 * H) ** 2
    out = np.full((N, 4), np.nan, dtype=np.float32)
    for i in range(N):
        li = L[i]; v = li[disk]
        out[i, 3] = 100.0 - float(v.mean())
        if v.max() - v.min() < 1.0:
            continue
        thr = _otsu(v)
        les = (li < thr) & disk
        lab, nlab = ndimage.label(les)
        if nlab == 0:
            continue
        sizes = ndimage.sum(les, lab, index=np.arange(1, nlab + 1))
        les = lab == (int(np.argmax(sizes)) + 1)
        frac = les.sum() / disk.sum()
        if frac < 0.01:
            continue
        skin = disk & ~les
        lesL = float(li[les].mean()); skinL = float(li[skin].mean()) if skin.sum() > 0 else np.nan
        out[i, 0] = 100.0 - lesL; out[i, 1] = skinL - lesL; out[i, 2] = frac
    return out


# ----------------------------------------------------------------------------
# Derm7pt loading: data logic copied from NB1 v1.1 (scorers, label mapping,
# official split). Prints are reduced; checks are kept.
# ----------------------------------------------------------------------------
def score_pigment_network(v):
    v = str(v).lower()
    if 'atypical' in v: return 1.0
    if 'typical' in v: return 0.5
    return 0.0

def score_blue_veil(v):
    v = str(v).lower(); return 1.0 if 'present' in v else 0.0

def score_vascular(v):
    v = str(v).lower()
    if 'within regression' in v or 'absent' in v: return 0.0
    return 1.0

def score_binary_irregular(v):
    v = str(v).lower()
    if 'irregular' in v: return 1.0
    if 'regular' in v: return 0.5
    return 0.0

def score_regression(v):
    v = str(v).lower(); return 0.0 if 'absent' in v else 1.0

SCORERS = {'pigment_network': score_pigment_network, 'blue_whitish_veil': score_blue_veil,
           'vascular_structures': score_vascular, 'pigmentation': score_binary_irregular,
           'streaks': score_binary_irregular, 'dots_and_globules': score_binary_irregular,
           'regression_structures': score_regression}


def load_derm7pt(paths):
    import pandas as pd
    raw_df = pd.read_csv(paths['derm7pt_csv'])
    raw_df['_orig_idx'] = raw_df.index
    df = raw_df.copy()
    for c in CONCEPT_NAMES:
        df[f'concept_{c}'] = df[c].apply(SCORERS[c])
    if 'seven_point_score' in df.columns:
        W = {'pigment_network': 2, 'blue_whitish_veil': 2, 'vascular_structures': 2,
             'pigmentation': 1, 'streaks': 1, 'dots_and_globules': 1, 'regression_structures': 1}
        der = sum(df[f'concept_{c}'] * w for c, w in W.items())
        corr = der.corr(df['seven_point_score'])
        log(f"  Derm7pt label check: corr(derived, seven_point_score) = {corr:.4f}")
        if corr < 0.70:
            raise ValueError('Derm7pt concept scoring check failed (same check as NB1)')
    df['target'] = df['diagnosis'].astype(str).str.lower().str.contains('melanoma').astype(int)

    def resolve(name):
        name = str(name)
        for ext in ['', '.jpg', '.jpeg', '.png']:
            p = os.path.join(paths['derm7pt_img_dir'], name + ext)
            if os.path.exists(p): return p
        return None
    df['image_path'] = df['derm'].apply(resolve)
    df = df[df['image_path'].notna()].reset_index(drop=True)

    idx = {}
    for k in ['train', 'valid', 'test']:
        t = pd.read_csv(paths[f'derm7pt_{k}_idx'])
        idx[k] = t[t.columns[0]].astype(int).values
    tr_a = df[df['case_num'].isin(idx['train'])]; vl_a = df[df['case_num'].isin(idx['valid'])]
    te_a = df[df['case_num'].isin(idx['test'])]
    if len(tr_a) + len(vl_a) + len(te_a) >= 0.8 * len(df):
        tr, vl, te = tr_a, vl_a, te_a; how = 'case_num'
    else:
        tr = df[df['_orig_idx'].isin(idx['train'])]; vl = df[df['_orig_idx'].isin(idx['valid'])]
        te = df[df['_orig_idx'].isin(idx['test'])]; how = 'positional'
        if len(tr) + len(vl) + len(te) < 0.8 * len(df):
            raise ValueError('Official Derm7pt split matching failed (same rule as NB1)')
    if set(tr['case_num']) & set(te['case_num']) or set(vl['case_num']) & set(te['case_num']):
        raise ValueError('Derm7pt split overlap detected')
    log(f"  Derm7pt official split via {how}: train={len(tr)} valid={len(vl)} test={len(te)}")
    return tr.reset_index(drop=True), vl.reset_index(drop=True), te.reset_index(drop=True)


def pil_resize_u8(path, size):
    from PIL import Image
    img = Image.open(path).convert('RGB')
    img = transforms.Resize((size, size))(img)   # bilinear + antialias (same as NB3 val pipeline)
    return np.asarray(img, dtype=np.uint8).transpose(2, 0, 1).copy()


def concept_aucs(y_true_soft, probs):
    """NB1 rule: a concept counts as present when its label is > 0."""
    from sklearn.metrics import roc_auc_score
    yb = (np.asarray(y_true_soft) > 0).astype(int); out = []
    for k in range(probs.shape[1]):
        try:
            out.append(float(roc_auc_score(yb[:, k], probs[:, k])))
        except ValueError:
            out.append(float('nan'))
    return out


# ----------------------------------------------------------------------------
# Colour statistics (Phase 1b): per-image mean CIELAB (D65) over the whole image
# ----------------------------------------------------------------------------
def lab_means(u8, chunk=256):
    """(N,3,H,W) uint8 sRGB -> (N,3) mean L*, a*, b* per image. numpy only."""
    M = np.array([[0.4124564, 0.3575761, 0.1804375],
                  [0.2126729, 0.7151522, 0.0721750],
                  [0.0193339, 0.1191920, 0.9503041]], dtype=np.float32)
    white = np.array([0.95047, 1.0, 1.08883], dtype=np.float32)
    out = []
    for s in range(0, len(u8), chunk):
        c = u8[s:s + chunk].astype(np.float32) / 255.0
        lin = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
        xyz = np.einsum('ij,njhw->nihw', M, lin) / white[None, :, None, None]
        f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16.0 / 116.0)
        L = 116.0 * f[:, 1] - 16.0; a = 500.0 * (f[:, 0] - f[:, 1]); b = 200.0 * (f[:, 1] - f[:, 2])
        out.append(np.stack([L.mean((1, 2)), a.mean((1, 2)), b.mean((1, 2))], 1))
    return np.concatenate(out, 0)
''',
    'worker.py': r'''"""ClinDiffXAI Phase 1 worker. Usage: python worker.py <job> <cfg.json>
Each job writes its outputs under cfg['work'] and a marker file done/<job>.ok."""
import os, sys, json, time, glob, math, traceback
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *  # noqa


def P(cfg, *parts):
    p = os.path.join(cfg['work'], *parts)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    return p


# =============================================================================
# JOB repro: re-run NB4_Stats_Supplement exactly (paper checkpoints, same seeds,
# same order of random draws, single GPU) and compare with the paper numbers.
# =============================================================================
def job_repro(cfg):
    from scipy.stats import wilcoxon, spearmanr, pearsonr
    dev = get_device(); torch.backends.cudnn.benchmark = True
    S = cfg['sizes']; IS = cfg['image_size']; N_EVAL = S['repro_n_eval']; N_P = S['repro_n_pearson']
    cbm, _ = load_cbm_a(cfg['paths']['cbma'], dev)
    main, minfo = load_diffusion(cfg['paths']['main_best'], 7, dev, IS)
    base, binfo = load_diffusion(cfg['paths']['base_best'], 2, dev, IS)
    MEL = torch.FloatTensor(MEL_VEC_LIST); BEN = torch.FloatTensor(BEN_VEC_LIST)
    rsz_norm = lambda imgs: torch.stack([TF.normalize(TF.resize(i, [cfg['eval_size']] * 2, antialias=True), IMNET_MEAN, IMNET_STD) for i in imgs])

    @torch.no_grad()
    def preds(imgs):
        out = []
        for s in range(0, len(imgs), 64):
            _, cp, _ = cbm(rsz_norm(imgs[s:s + 64]).to(dev)); out.append(cp.cpu().numpy())
        return np.concatenate(out, 0)

    def cos(P_, c):
        c = c.numpy(); return np.array([np.dot(c, p) / (np.linalg.norm(c) * np.linalg.norm(p) + 1e-8) for p in P_])

    gen = lambda m, cv, n, steps: generate_batch_supplement(m, cv, n, IS, dev, cfg_scale=3.5, steps=steps, bs=16)
    # identical draw order to the supplement
    torch.manual_seed(42); _ = [torch.randn(1, 3, IS, IS) for _ in range(N_EVAL)]
    torch.manual_seed(42); _ = [torch.randn(1, 3, IS, IS) for _ in range(N_EVAL)]
    log('repro: generating 4 paired evaluation sets')
    torch.manual_seed(42); main_mel = gen(main, MEL, N_EVAL, S['ddim_steps'])
    torch.manual_seed(42); base_mel = gen(base, torch.FloatTensor([0., 1.]), N_EVAL, S['ddim_steps'])
    torch.manual_seed(123); main_ben = gen(main, BEN, N_EVAL, S['ddim_steps'])
    torch.manual_seed(123); base_ben = gen(base, torch.FloatTensor([1., 0.]), N_EVAL, S['ddim_steps'])
    pr = {k: preds(v) for k, v in [('main_mel', main_mel), ('base_mel', base_mel), ('main_ben', main_ben), ('base_ben', base_ben)]}
    mm, bm = cos(pr['main_mel'], MEL), cos(pr['base_mel'], MEL)
    mb, bb = cos(pr['main_ben'], BEN), cos(pr['base_ben'], BEN)

    def rank_biserial_supp(x, y):   # formula exactly as in the supplement
        d = x - y; d = d[d != 0]; n = len(d)
        ranks = np.argsort(np.argsort(np.abs(d))) + 1
        W_neg = ranks[d < 0].sum(); return 1 - (2 * W_neg) / (n * (n + 1) / 2)

    res = {'main_ckpt': minfo, 'base_ckpt': binfo, 'torch': torch.__version__,
           'gpu': torch.cuda.get_device_name(0) if dev.type == 'cuda' else 'cpu',
           'n_eval': N_EVAL, 'n_pearson': N_P, 'ccfs_mean': {
               'main_mel': float(mm.mean()), 'base_mel': float(bm.mean()),
               'main_ben': float(mb.mean()), 'base_ben': float(bb.mean())}}
    for name, x, y in [('mel', mm, bm), ('ben', mb, bb), ('all', np.concatenate([mm, mb]), np.concatenate([bm, bb]))]:
        try:
            W, p = wilcoxon(x, y, alternative='greater')
        except ValueError:
            W, p = float('nan'), float('nan')
        res[f'wilcoxon_{name}'] = {'W': float(W), 'p': float(p), 'r_rb': float(rank_biserial_supp(x, y))}
    for nm, arr in [('repro_eval_main_mel', main_mel), ('repro_eval_base_mel', base_mel),
                    ('repro_eval_main_ben', main_ben), ('repro_eval_base_ben', base_ben)]:
        save_set(P(cfg, 'sets', nm + '.pt'), float_to_u8(arr), {'noise_id': list(range(len(arr)))},
                 {'kind': 'repro_eval', 'model': nm.split('_')[2], 'cls': nm.split('_')[3]})

    log('repro: per-concept sweep (supplement order, 30 DDIM steps)')
    steps = np.linspace(0, 1, S['repro_levels'])
    sweep_imgs, meta = [], {'concept_idx': [], 'level': [], 'noise_id': []}
    r_p, p_p, r_s, p_s = {}, {}, {}, {}
    for ci, cname in enumerate(CONCEPT_NAMES):
        iv, pv = [], []
        for val in steps:
            cvec = MEL.clone(); cvec[ci] = float(val)
            imgs = gen(main, cvec, N_P, S['repro_sweep_steps'])
            pp = preds(list(imgs))[:, ci]
            iv.extend([float(val)] * N_P); pv.extend(pp.tolist())
            sweep_imgs.append(float_to_u8(imgs))
            meta['concept_idx'] += [ci] * N_P; meta['level'] += [float(val)] * N_P; meta['noise_id'] += list(range(N_P))
        a, b = pearsonr(iv, pv); c, d = spearmanr(iv, pv)
        r_p[cname], p_p[cname], r_s[cname], p_s[cname] = float(a), float(b), float(c), float(d)
        log(f"  {cname:<24} r={a:+.3f} rho={c:+.3f}")
    res['pearson_r'] = r_p; res['pearson_p'] = p_p; res['spearman_rho'] = r_s; res['spearman_p'] = p_s
    save_set(P(cfg, 'sets', 'repro_sweep.pt'), torch.cat(sweep_imgs, 0), meta, {'kind': 'repro_sweep', 'base': 'mel'})
    save_json(res, P(cfg, 'results', 'repro_results.json'))


# =============================================================================
# JOB prep_real: real anchor images (all 584 training melanomas, held-out benign)
# =============================================================================
def _load_one(args):
    path, size = args
    try:
        return pil_resize_u8(path, size)
    except Exception:
        return None


def _load_multi(args):
    """decode once, return one uint8 array per requested size"""
    from PIL import Image
    path, sizes = args
    try:
        img = Image.open(path).convert('RGB')
        return [np.asarray(transforms.Resize((s, s))(img), dtype=np.uint8).transpose(2, 0, 1).copy() for s in sizes]
    except Exception:
        return None


def job_prep_real(cfg):
    import pandas as pd
    from multiprocessing import Pool
    IS = cfg['image_size']; S = cfg['sizes']
    df = pd.read_csv(cfg['paths']['pseudo_csv'])
    mel = df[df['target'] == 1].copy(); ben = df[df['target'] == 0].copy()
    mel_all = mel
    # exact NB3 balancing: 5x melanoma, benign sample of 3x that size, seed 42
    n_ben_train = len(mel) * 5 * 3
    ben_train = ben.sample(n=min(n_ben_train, len(ben)), random_state=42)
    held = ben[~ben['image_name'].isin(set(ben_train['image_name']))]
    held = held.sample(n=min(S['n_real_ben'], len(held)), random_state=2026)
    if S.get('n_real_mel_max'):
        mel = mel.head(S['n_real_mel_max'])   # dry run only
    log(f"prep_real: melanoma={len(mel)} (all in diffusion training), benign train={len(ben_train)}, "
        f"held-out benign sample={len(held)}")
    ccols = [f'concept_{c}' for c in CONCEPT_NAMES]
    for nm, sub in [('real_mel', mel), ('real_ben', held)]:
        paths = [os.path.join(cfg['paths']['isic2020_dir'], f"{n}.jpg") for n in sub['image_name']]
        with Pool(cfg['cpu_workers']) as pool:
            arrs = pool.map(_load_one, [(p, IS) for p in paths], chunksize=8)
        keep = [i for i, a in enumerate(arrs) if a is not None]
        imgs = torch.from_numpy(np.stack([arrs[i] for i in keep]))
        meta = {'image_name': [sub['image_name'].iloc[i] for i in keep]}
        for c in ccols:
            meta[c] = [float(sub[c].iloc[i]) for i in keep]
        save_set(P(cfg, 'sets', nm + '.pt'), imgs, meta,
                 {'kind': 'real', 'cls': nm.split('_')[1], 'in_diffusion_training': nm == 'real_mel'})
        log(f"  {nm}: {len(keep)} images saved ({len(paths) - len(keep)} missing)")
    save_json({'n_mel': int(len(mel)), 'n_ben_train': int(len(ben_train)), 'n_ben_heldout_sample': int(len(held))},
              P(cfg, 'results', 'real_sets.json'))


# =============================================================================
# Evaluator training helpers
# =============================================================================
def _aug_batch(u8_big, out_size, train):
    """u8_big: (B,3,s,s) uint8 CPU. train: random crop to out_size, flips, rot,
    light colour jitter. eval: centre resize to out_size."""
    x = u8_big.float() / 255.0
    if train:
        B, _, s, _ = x.shape
        crops = []
        for i in range(B):   # per-sample crop, flips and 90-degree rotation
            i0 = np.random.randint(0, s - out_size + 1); j0 = np.random.randint(0, s - out_size + 1)
            xi = x[i, :, i0:i0 + out_size, j0:j0 + out_size]
            if np.random.rand() < 0.5: xi = xi.flip(2)
            if np.random.rand() < 0.5: xi = xi.flip(1)
            crops.append(torch.rot90(xi, np.random.randint(4), dims=(1, 2)))
        x = torch.stack(crops)
        b = 1 + (torch.rand(B, 1, 1, 1) - 0.5) * 0.2; c = 1 + (torch.rand(B, 1, 1, 1) - 0.5) * 0.2
        m = x.mean(dim=(1, 2, 3), keepdim=True)
        x = ((x - m) * c + m) * b
        x = x.clamp(0, 1)
    else:
        if x.shape[-1] != out_size:
            x = TF.resize(x, [out_size, out_size], antialias=True)
    return (x * 255).round().to(torch.uint8)


def _train_multilabel(cfg, net, X_tr, Y_tr, X_vl, Y_vl, sel_fn, pos_weight=None, tag='eval'):
    """Generic trainer on cached uint8 arrays. X_*: uint8 (N,3,s,s) where training
    arrays carry a margin for random crops. Returns best state_dict and history."""
    dev = get_device(); net = net.to(dev)
    E = cfg['train']['epochs']; bs = cfg['train']['batch']; IS = cfg['image_size']
    opt = torch.optim.AdamW(net.parameters(), lr=cfg['train']['lr'], weight_decay=0.05)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg['train']['lr'], total_steps=E * max(1, len(X_tr) // bs),
                                                pct_start=0.1, anneal_strategy='cos')
    use_amp = dev.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    pw = None if pos_weight is None else torch.tensor(pos_weight, dtype=torch.float32, device=dev)
    best, best_state, hist, bad = -1, None, [], 0
    Ytr = torch.tensor(Y_tr, dtype=torch.float32)
    for ep in range(E):
        net.train(); perm = np.random.permutation(len(X_tr)); losses = []
        for s in range(0, len(perm) - bs + 1, bs):
            idx = perm[s:s + bs]
            xb = u8_to_eval(_aug_batch(X_tr[idx], IS, True), dev, cfg['eval_size'])
            yb = Ytr[idx].to(dev)
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp):
                logits = net(xb)
            loss = F.binary_cross_entropy_with_logits(logits.float(), yb, pos_weight=pw)
            opt.zero_grad(set_to_none=True); scaler.scale(loss).backward()
            scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step(); losses.append(loss.item())
        pv = predict_probs(net, X_vl, cfg, resize_from=None)
        score = sel_fn(Y_vl, pv)
        hist.append({'epoch': ep + 1, 'train_loss': float(np.mean(losses)) if losses else None, 'val_score': score})
        log(f"  [{tag}] ep {ep + 1}/{E} loss={hist[-1]['train_loss']} val_meanAUC={score:.4f}")
        if score > best + 1e-4:
            best, bad = score, 0
            best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= cfg['train']['patience']:
                log(f"  [{tag}] early stop at epoch {ep + 1}"); break
    net.load_state_dict(best_state)
    return net, hist, best


@torch.no_grad()
def predict_probs(net, X_u8, cfg, resize_from=None, bs=64):
    dev = get_device(); net.eval(); out = []
    for s in range(0, len(X_u8), bs):
        out.append(torch.sigmoid(net(u8_to_eval(X_u8[s:s + bs], dev, cfg['eval_size'])).float()).cpu().numpy())
    return np.concatenate(out, 0) if out else np.zeros((0, 1))


@torch.no_grad()
def cbma_probs(cbm, X_u8, cfg, bs=64):
    dev = get_device(); out, dg = [], []
    for s in range(0, len(X_u8), bs):
        cp, _, dp = cbm.forward_all(u8_to_eval(X_u8[s:s + bs], dev, cfg['eval_size']))
        out.append(cp.cpu().numpy()); dg.append(dp.cpu().numpy())
    return np.concatenate(out, 0), np.concatenate(dg, 0)


def _load_multi_draft(args):
    """as _load_multi, but lets libjpeg decode large JPEGs at a reduced scale
    (never below 2x the largest requested size) before the antialiased resize"""
    from PIL import Image
    path, sizes = args
    try:
        img = Image.open(path)
        if img.format == 'JPEG':
            img.draft('RGB', (2 * max(sizes), 2 * max(sizes)))
        img = img.convert('RGB')
        return [np.asarray(transforms.Resize((s, s))(img), dtype=np.uint8).transpose(2, 0, 1).copy() for s in sizes]
    except Exception:
        return None


def _cache(paths_list, sizes, workers, draft=False):
    """returns a list of uint8 tensors, one per size, each image decoded once"""
    from multiprocessing import Pool
    with Pool(workers) as pool:
        arrs = pool.map(_load_multi_draft if draft else _load_multi, [(p, sizes) for p in paths_list], chunksize=8)
    bad = [i for i, a in enumerate(arrs) if a is None]
    if bad:
        raise RuntimeError(f"{len(bad)} images failed to load, e.g. {paths_list[bad[0]]}")
    return [torch.from_numpy(np.stack([a[k] for a in arrs])) for k in range(len(sizes))]


# =============================================================================
# JOB train_cbmb: independent concept evaluator on Derm7pt (official split)
# =============================================================================
def job_train_cbmb(cfg):
    from sklearn.metrics import roc_auc_score
    torch.manual_seed(7); np.random.seed(7)
    IS = cfg['image_size']; ES = cfg['eval_size']; W = cfg['cpu_workers']
    tr, vl, te = load_derm7pt(cfg['paths'])
    ccols = [f'concept_{c}' for c in CONCEPT_NAMES]
    margin = IS + IS // 8   # 144 for 128, like the NB3 Resize(+16) / RandomCrop
    log('train_cbmb: caching Derm7pt images')
    Xtr, = _cache(list(tr['image_path']), [margin], W); Xvl, = _cache(list(vl['image_path']), [IS], W)
    Xte, Xte_nat = _cache(list(te['image_path']), [IS, ES], W)
    Ytr, Yvl, Yte = tr[ccols].values, vl[ccols].values, te[ccols].values
    sel = lambda Y, Pp: float(np.nanmean(concept_aucs(Y, Pp)))
    net = ConceptNetB(pretrained=pretrained_ok(cfg))
    net, hist, best = _train_multilabel(cfg, net, Xtr, Ytr, Xvl, Yvl, sel, tag='CBM-B')
    torch.save({'state_dict': net.state_dict(), 'arch': 'convnext_tiny', 'hist': hist}, P(cfg, 'evaluators', 'cbmb.pth'))

    # evaluate CBM-B and CBM-A on the held-out Derm7pt test set
    dev = get_device()
    cbm, _ = load_cbm_a(cfg['paths']['cbma'], dev)
    b_bneck = predict_probs(net, Xte, cfg)
    b_nat = predict_probs(net, Xte_nat, cfg)
    a_bneck, a_bneck_dg = cbma_probs(cbm, Xte, cfg)
    a_nat, a_nat_dg = cbma_probs(cbm, Xte_nat, cfg)
    yd = te['target'].values

    def dauc(p):
        try: return float(roc_auc_score(yd, p))
        except ValueError: return float('nan')
    res = {'n_train': len(tr), 'n_valid': len(vl), 'n_test': len(te), 'best_val_meanAUC': best, 'history': hist,
           'positives_test': {c: int((Yte[:, i] > 0).sum()) for i, c in enumerate(CONCEPT_NAMES)},
           'auc': {'CBM-A_native': concept_aucs(Yte, a_nat), 'CBM-A_bottleneck': concept_aucs(Yte, a_bneck),
                   'CBM-B_native': concept_aucs(Yte, b_nat), 'CBM-B_bottleneck': concept_aucs(Yte, b_bneck)},
           'diag_auc': {'CBM-A_native': dauc(a_nat_dg), 'CBM-A_bottleneck': dauc(a_bneck_dg)}}
    for k, v in res['auc'].items():
        log(f"  test {k:<18} mean AUC={np.nanmean(v):.4f}  " + ' '.join(f"{s}={a:.3f}" for s, a in zip(SHORT, v)))
    log(f"  CBM-A native diagnosis AUC = {res['diag_auc']['CBM-A_native']:.4f} (paper: 0.5425)")
    save_json(res, P(cfg, 'results', 'derm7pt_evaluators.json'))
    np.savez(P(cfg, 'results', 'derm7pt_test_preds.npz'), y=Yte, a_nat=a_nat, a_bneck=a_bneck, b_nat=b_nat, b_bneck=b_bneck)


# =============================================================================
# JOB train_isic18: ISIC 2018 Task 2 attribute detector (different dataset)
# =============================================================================
def _mask_present(path):
    from PIL import Image
    try:
        im = Image.open(path)
        im.draft('L', (im.size[0] // 8, im.size[1] // 8))
        return int(im.convert('L').getextrema()[1] > 0)
    except Exception:
        return -1


def job_train_isic18(cfg):
    import pandas as pd
    from multiprocessing import Pool
    idx_csv = cfg['paths'].get('isic18_index')
    if not idx_csv or not os.path.exists(idx_csv):
        log('train_isic18: ISIC 2018 Task 2 data not found, job skipped'); return 'skipped'
    torch.manual_seed(11); np.random.seed(11)
    IS = cfg['image_size']; ES = cfg['eval_size']; W = cfg.get('cpu_workers_io', cfg['cpu_workers'])
    d = pd.read_csv(idx_csv)
    log(f"train_isic18: {len(d)} images; reading attribute masks")
    Y = np.zeros((len(d), len(ISIC18_ATTRS)), dtype=np.float32)
    with Pool(W) as pool:
        for j, a in enumerate(ISIC18_ATTRS):
            v = np.array(pool.map(_mask_present, list(d[f'm_{a}']), chunksize=16))
            if (v < 0).any():
                raise RuntimeError(f"unreadable masks for {a}: {int((v < 0).sum())}")
            Y[:, j] = v
    rng = np.random.RandomState(42); perm = rng.permutation(len(d))
    n_te = int(round(0.15 * len(d))); n_vl = int(round(0.15 * len(d)))
    te_i, vl_i, tr_i = perm[:n_te], perm[n_te:n_te + n_vl], perm[n_te + n_vl:]
    log('  caching images')
    margin = IS + IS // 8
    paths = list(d['img'])
    Xtr, = _cache([paths[i] for i in tr_i], [margin], W, draft=True); Xvl, = _cache([paths[i] for i in vl_i], [IS], W, draft=True)
    Xte, Xte_nat = _cache([paths[i] for i in te_i], [IS, ES], W, draft=True)
    pos = Y[tr_i].sum(0); neg = len(tr_i) - pos
    pw = np.clip(neg / np.maximum(pos, 1), 1, 10).tolist()
    sel = lambda Yt, Pp: float(np.nanmean(concept_aucs(Yt, Pp)))
    net = AttrNet(pretrained=pretrained_ok(cfg))
    net, hist, best = _train_multilabel(cfg, net, Xtr, Y[tr_i], Xvl, Y[vl_i], sel, pos_weight=pw, tag='ISIC18')
    torch.save({'state_dict': net.state_dict(), 'arch': 'resnet50', 'attrs': ISIC18_ATTRS, 'hist': hist},
               P(cfg, 'evaluators', 'isic18_detector.pth'))
    p_b = predict_probs(net, Xte, cfg); p_n = predict_probs(net, Xte_nat, cfg)
    overlap = None
    try:
        names = set(pd.read_csv(cfg['paths']['pseudo_csv'])['image_name'])
        overlap = int(len(set(d['id']) & names))
    except Exception:
        pass
    res = {'n_total': len(d), 'n_train': len(tr_i), 'n_valid': len(vl_i), 'n_test': len(te_i),
           'prevalence_all': {a: float(Y[:, j].mean()) for j, a in enumerate(ISIC18_ATTRS)},
           'positives_test': {a: int(Y[te_i, j].sum()) for j, a in enumerate(ISIC18_ATTRS)},
           'auc': {'ISIC18_native': concept_aucs(Y[te_i], p_n), 'ISIC18_bottleneck': concept_aucs(Y[te_i], p_b)},
           'best_val_meanAUC': best, 'history': hist, 'id_overlap_with_isic2020_csv': overlap,
           'split_seed': 42}
    for k, v in res['auc'].items():
        log(f"  test {k:<18} " + ' '.join(f"{a}={x:.3f}" for a, x in zip(ISIC18_ATTRS, v)))
    log(f"  ISIC 2018 IDs that also appear in the ISIC 2020 CSV: {overlap}")
    save_json(res, P(cfg, 'results', 'isic18_evaluator.json'))


# =============================================================================
# Generation jobs (new design: CPU-drawn noise, common random numbers)
# =============================================================================
def _ckpt(cfg, which):
    p = cfg['paths'].get(f'{which}_e150') or cfg['paths'][f'{which}_best']
    return p


def job_gen_eval(cfg, which):
    dev = get_device(); torch.backends.cudnn.benchmark = True
    S = cfg['sizes']; IS = cfg['image_size']; n = S['n_eval']
    model, info = load_diffusion(_ckpt(cfg, which), 7 if which == 'main' else 2, dev, IS)
    for cls, seed in [('mel', cfg['seeds']['eval_mel']), ('ben', cfg['seeds']['eval_ben'])]:
        out = P(cfg, 'sets', f'eval_{which}_{cls}.pt')
        if os.path.exists(out):
            log(f'  {out} exists, skipped'); continue
        noise = cpu_noise(n, IS, seed)
        if which == 'main':
            v = torch.FloatTensor(MEL_VEC_LIST if cls == 'mel' else BEN_VEC_LIST)
        else:
            v = torch.FloatTensor([0., 1.] if cls == 'mel' else [1., 0.])
        cond = v.unsqueeze(0).repeat(n, 1)
        t0 = time.time()
        imgs = generate_crn(model, cond, noise, dev, cfg_scale=S['cfg_scale'], steps=S['ddim_steps'], bs=S['gen_bs'])
        log(f"  eval_{which}_{cls}: {n} images in {time.time() - t0:.0f}s")
        save_set(out, imgs, {'noise_id': list(range(n))},
                 {'kind': 'eval', 'model': which, 'cls': cls, 'ckpt': info, 'noise_seed': seed,
                  'cfg_scale': S['cfg_scale'], 'ddim_steps': S['ddim_steps']})


def job_gen_eval_realc(cfg):
    """Exploratory (added after Phase 1): the main model conditioned on the
    pseudo-label vectors of REAL training images instead of the p90/p10
    prototypes. Same noises as eval_main / eval_base, so all three are paired."""
    import pandas as pd
    dev = get_device(); torch.backends.cudnn.benchmark = True
    S = cfg['sizes']; IS = cfg['image_size']; n = S['n_eval']
    df = pd.read_csv(cfg['paths']['pseudo_csv']); ccols = [f'concept_{c}' for c in CONCEPT_NAMES]
    mel = df[df['target'] == 1]; ben = df[df['target'] == 0]
    ben_train = ben.sample(n=min(len(mel) * 15, len(ben)), random_state=42)   # NB3 balancing
    pools = {'mel': (mel, cfg['seeds']['realc_mel'], True), 'ben': (ben_train, cfg['seeds']['realc_ben'], False)}
    model = None
    for cls, nseed in [('mel', cfg['seeds']['eval_mel']), ('ben', cfg['seeds']['eval_ben'])]:
        out = P(cfg, 'sets', f'eval_realc_{cls}.pt')
        if os.path.exists(out):
            log(f'  {out} exists, skipped'); continue
        if model is None:
            model, info = load_diffusion(_ckpt(cfg, 'main'), 7, dev, IS)
        pool, vseed, repl = pools[cls]
        rows = pool.sample(n=n, replace=repl or n > len(pool), random_state=vseed)
        cond = torch.FloatTensor(rows[ccols].values.astype(np.float32)).clamp(0, 1)
        noise = cpu_noise(n, IS, nseed)
        t0 = time.time()
        imgs = generate_crn(model, cond, noise, dev, cfg_scale=S['cfg_scale'], steps=S['ddim_steps'], bs=S['gen_bs'])
        log(f"  eval_realc_{cls}: {n} images in {time.time() - t0:.0f}s")
        meta = {'noise_id': list(range(n)), 'source_image': list(rows['image_name'])}
        for c in ccols:
            meta[c] = [float(v) for v in rows[c]]
        save_set(out, imgs, meta, {'kind': 'eval_realc', 'model': 'main', 'cls': cls, 'ckpt': info,
                                   'noise_seed': nseed, 'vector_seed': vseed, 'cfg_scale': S['cfg_scale'],
                                   'ddim_steps': S['ddim_steps']})


def job_gen_sweep(cfg, base):
    dev = get_device(); torch.backends.cudnn.benchmark = True
    S = cfg['sizes']; IS = cfg['image_size']; n = S['sweep_n']
    out = P(cfg, 'sets', f'sweep_{base}.pt')
    if os.path.exists(out):
        log(f'  {out} exists, skipped'); return
    model, info = load_diffusion(_ckpt(cfg, 'main'), 7, dev, IS)
    levels = [round(float(x), 4) for x in np.linspace(0, 1, S['sweep_levels'])]
    noise = cpu_noise(n, IS, cfg['seeds']['sweep'])   # the same noises for every level/concept/base
    bvec = torch.FloatTensor(MEL_VEC_LIST if base == 'mel' else BEN_VEC_LIST)
    imgs, meta = [], {'concept_idx': [], 'level': [], 'noise_id': []}
    t0 = time.time()
    for j in range(7):
        for l in levels:
            v = bvec.clone(); v[j] = l
            imgs.append(generate_crn(model, v.unsqueeze(0).repeat(n, 1), noise, dev,
                                     cfg_scale=S['cfg_scale'], steps=S['ddim_steps'], bs=S['gen_bs']))
            meta['concept_idx'] += [j] * n; meta['level'] += [l] * n; meta['noise_id'] += list(range(n))
        log(f"  sweep_{base}: concept {CONCEPT_NAMES[j]} done ({time.time() - t0:.0f}s)")
    save_set(out, torch.cat(imgs, 0), meta, {'kind': 'sweep', 'base': base, 'ckpt': info, 'levels': levels,
                                             'noise_seed': cfg['seeds']['sweep'], 'cfg_scale': S['cfg_scale']})


def job_gen_cfg(cfg):
    dev = get_device(); torch.backends.cudnn.benchmark = True
    S = cfg['sizes']; IS = cfg['image_size']; n = S['cfg_n']
    out = P(cfg, 'sets', 'cfg_sweep.pt')
    if os.path.exists(out):
        log(f'  {out} exists, skipped'); return
    model, info = load_diffusion(_ckpt(cfg, 'main'), 7, dev, IS)
    noise = cpu_noise(n, IS, cfg['seeds']['cfg'])
    imgs, meta = [], {'scale': [], 'cls': [], 'noise_id': []}
    for s in S['cfg_scales']:
        for cls, vec in [('mel', MEL_VEC_LIST), ('ben', BEN_VEC_LIST)]:
            cond = torch.FloatTensor(vec).unsqueeze(0).repeat(n, 1)
            imgs.append(generate_crn(model, cond, noise, dev, cfg_scale=float(s), steps=S['ddim_steps'], bs=S['gen_bs']))
            meta['scale'] += [float(s)] * n; meta['cls'] += [cls] * n; meta['noise_id'] += list(range(n))
        log(f"  cfg sweep: s={s} done")
    save_set(out, torch.cat(imgs, 0), meta, {'kind': 'cfg', 'ckpt': info})


# =============================================================================
# Scoring jobs
# =============================================================================
NEED_INCEP = ('eval_', 'real_', 'cfg_', 'repro_eval_')


def _sets(cfg):
    return sorted(glob.glob(os.path.join(cfg['work'], 'sets', '*.pt')))


def job_score_A(cfg):
    """CBM-A probs + 512-d CBM-A feature (MedFID), Inception 2048-d, darkness."""
    dev = get_device()
    cbm, _ = load_cbm_a(cfg['paths']['cbma'], dev)
    incep = build_inception(cfg, dev)
    for sp in _sets(cfg):
        name = os.path.basename(sp)[:-3]; out = P(cfg, 'scores', f'A_{name}.npz')
        if os.path.exists(out): continue
        X = load_set(sp)['imgs']; t0 = time.time()
        cp, cf, dg = [], [], []
        with torch.no_grad():
            for s in range(0, len(X), 64):
                a, b, c = cbm.forward_all(u8_to_eval(X[s:s + 64], dev, cfg['eval_size']))
                cp.append(a.cpu().numpy()); cf.append(b.cpu().numpy()); dg.append(c.cpu().numpy())
            feats = {}
            if name.startswith(NEED_INCEP):
                fi = []
                for s in range(0, len(X), 64):
                    f = incep(u8_to_inception(X[s:s + 64], dev, cfg['incep_size']))
                    if isinstance(f, tuple): f = f[0]
                    fi.append(f.float().cpu().numpy())
                feats['incep'] = np.concatenate(fi, 0)
        dk = darkness_metrics(X.numpy())
        with open(out + '.tmp', 'wb') as fh:   # atomic: a killed job never leaves a broken file
            np.savez(fh, cbma=np.concatenate(cp, 0), cbma_feat=np.concatenate(cf, 0), cbma_diag=np.concatenate(dg, 0),
                     dark=dk, **feats)
        os.replace(out + '.tmp', out)
        log(f"  score_A {name}: n={len(X)} ({time.time() - t0:.0f}s)")


def job_score_B(cfg):
    """CBM-B probs + 768-d CBM-B embedding; ISIC 2018 attribute probs."""
    dev = get_device()
    nets = {}
    pb = os.path.join(cfg['work'], 'evaluators', 'cbmb.pth')
    if os.path.exists(pb):
        n = ConceptNetB(pretrained=False); n.load_state_dict(torch.load(pb, map_location='cpu', weights_only=False)['state_dict'])
        nets['cbmb'] = n.to(dev).eval()
    pi = os.path.join(cfg['work'], 'evaluators', 'isic18_detector.pth')
    if os.path.exists(pi):
        n = AttrNet(pretrained=False); n.load_state_dict(torch.load(pi, map_location='cpu', weights_only=False)['state_dict'])
        nets['isic18'] = n.to(dev).eval()
    if not nets:
        log('score_B: no independent evaluator was trained; job skipped'); return
    for sp in _sets(cfg):
        name = os.path.basename(sp)[:-3]; out = P(cfg, 'scores', f'B_{name}.npz')
        have = {}
        if os.path.exists(out):
            z = np.load(out); have = {k: z[k] for k in z.files}
        todo = [k for k in nets if k not in have]
        if not todo: continue
        X = load_set(sp)['imgs']; res = {k: [] for k in ['cbmb', 'cbmb_feat', 'isic18']}
        with torch.no_grad():
            for s in range(0, len(X), 64):
                x = u8_to_eval(X[s:s + 64], dev, cfg['eval_size'])
                if 'cbmb' in todo:
                    e = nets['cbmb'].embed(x); res['cbmb_feat'].append(e.cpu().numpy())
                    res['cbmb'].append(torch.sigmoid(nets['cbmb'].head(e)).cpu().numpy())
                if 'isic18' in todo:
                    res['isic18'].append(torch.sigmoid(nets['isic18'](x)).cpu().numpy())
        have.update({k: np.concatenate(v, 0) for k, v in res.items() if v})
        with open(out + '.tmp', 'wb') as fh:
            np.savez(fh, **have)
        os.replace(out + '.tmp', out)
        log(f"  score_B {name}: n={len(X)} added {todo}")


JOBS = {
    'repro': job_repro, 'prep_real': job_prep_real, 'train_cbmb': job_train_cbmb,
    'train_isic18': job_train_isic18,
    'gen_eval_main': lambda c: job_gen_eval(c, 'main'), 'gen_eval_base': lambda c: job_gen_eval(c, 'base'),
    'gen_sweep_mel': lambda c: job_gen_sweep(c, 'mel'), 'gen_sweep_ben': lambda c: job_gen_sweep(c, 'ben'),
    'gen_cfg': job_gen_cfg, 'gen_eval_realc': job_gen_eval_realc, 'score_A': job_score_A, 'score_B': job_score_B,
}

if __name__ == '__main__':
    job, cfg_path = sys.argv[1], sys.argv[2]
    cfg = load_cfg(cfg_path)
    log(f"job {job} start | device={get_device()} | torch {torch.__version__} | "
        f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")
    t0 = time.time()
    try:
        ret = JOBS[job](cfg)
    except Exception:
        traceback.print_exc(); sys.exit(1)
    if ret == 'skipped':
        log(f"job {job} skipped (no done marker written)"); sys.exit(0)
    with open(P(cfg, 'done', f'{job}.ok'), 'w') as f:
        f.write(f"{time.time() - t0:.0f}s\n")
    log(f"job {job} done in {(time.time() - t0) / 60:.1f} min")
''',
    'analysis.py': r'''"""ClinDiffXAI Phase 1 analysis (CPU). Reads image-set metadata and evaluator
scores, computes every metric, writes tables (CSV + LaTeX), figures (PNG + PDF),
gate1.json and phase1_report.md. Usage: python analysis.py <cfg.json>"""
import os, sys, json, glob, time, traceback, math
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (CONCEPT_NAMES, SHORT, MEL_VEC_LIST, BEN_VEC_LIST, ISIC18_ATTRS, ISIC18_TO_DERM,
                    DARK_COLS, load_cfg, load_set, save_json, log)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats
from scipy.linalg import sqrtm

# ----------------------------------------------------------------------------
# Paper values (manuscript Tables 2-5 and Results text; NB4_Stats_Supplement run)
# ----------------------------------------------------------------------------
PAPER = {
    'wilcoxon_mel': {'W': 12604.0, 'p': 0.0009, 'r_rb': 0.254},
    'wilcoxon_all': {'W': 47242.0, 'p': 0.001, 'r_rb': 0.178},
    'wilcoxon_ben': {'p': 0.110},
    'pearson_r': {'pigmentation': 0.717, 'pigment_network': 0.296, 'streaks': 0.200,
                  'vascular_structures': 0.163, 'regression_structures': 0.090,
                  'dots_and_globules': 0.078, 'blue_whitish_veil': 0.019},
    'spearman_rho': {'pigmentation': 0.730, 'pigment_network': 0.331, 'streaks': 0.079,
                     'vascular_structures': 0.239, 'regression_structures': 0.080,
                     'dots_and_globules': 0.075, 'blue_whitish_veil': 0.033},
    'cbma_test_mean_auc': 0.8224, 'cbma_test_diag_auc': 0.5425,
}

EVS = ['cbma', 'cbmb', 'isic18', 'dark']
EV_LABEL = {'cbma': 'CBM-A', 'cbmb': 'CBM-B', 'isic18': 'ISIC-2018 detector', 'dark': 'Lesion darkness'}
EV_LONG = {'cbma': 'CBM-A (EfficientNetV2-S, Derm7pt; produced the pseudo-labels)',
           'cbmb': 'CBM-B (ConvNeXt-Tiny, Derm7pt; independent architecture and seed)',
           'isic18': 'ISIC 2018 Task 2 attribute detector (ResNet-50; independent data)',
           'dark': 'Lesion darkness, 100 - mean lesion L* (non-learned)'}
EV_COLS = {'cbma': CONCEPT_NAMES, 'cbmb': CONCEPT_NAMES, 'isic18': ISIC18_ATTRS, 'dark': DARK_COLS}
# colour by job: fixed categorical order (validated reference palette, light mode)
C_EV = {'cbma': '#2a78d6', 'cbmb': '#eb6834', 'isic18': '#1baf7a', 'dark': '#eda100'}
M_EV = {'cbma': 'o', 'cbmb': 's', 'isic18': '^', 'dark': 'D'}
C_MODEL = {'main': '#2a78d6', 'base': '#eb6834', 'real': '#52514e', 'realc': '#1baf7a'}
RC_LAB = 'ClinDiffXAI, real vectors'
INK, INK2, GRID = '#0b0b0b', '#52514e', '#e6e5e1'
DIV = LinearSegmentedColormap.from_list('div', ['#e34948', '#f0efec', '#2a78d6'])

plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 8, 'axes.titlesize': 8, 'axes.labelsize': 8,
    'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 7, 'legend.frameon': False,
    'axes.spines.top': False, 'axes.spines.right': False, 'axes.edgecolor': '#8a8984',
    'axes.linewidth': 0.6, 'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.5,
    'axes.axisbelow': True, 'lines.linewidth': 1.5, 'lines.markersize': 4, 'text.color': INK,
    'axes.labelcolor': INK, 'xtick.color': INK2, 'ytick.color': INK2,
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'savefig.dpi': 300, 'figure.dpi': 100,
})

RNG = np.random.default_rng(20261017)
REPORT, ERRORS, WARN = [], [], []


# ----------------------------------------------------------------------------
# I/O helpers
# ----------------------------------------------------------------------------
class Out:
    def __init__(self, root):
        self.root = root
        for d in ['tables', 'figures', 'results']:
            os.makedirs(os.path.join(root, d), exist_ok=True)

    def fig(self, fig, name):
        fig.savefig(os.path.join(self.root, 'figures', name + '.png'), dpi=300, bbox_inches='tight', facecolor='white')
        fig.savefig(os.path.join(self.root, 'figures', name + '.pdf'), bbox_inches='tight', facecolor='white')
        plt.close(fig); log(f"  figure {name}")

    def table(self, df, name, caption='', index=False):
        df.to_csv(os.path.join(self.root, 'tables', name + '.csv'), index=index)
        with open(os.path.join(self.root, 'tables', name + '.tex'), 'w') as f:
            f.write(to_latex(df, caption, name, index))
        log(f"  table {name}")


def _tex_escape(s):
    s = str(s)
    for a, b in [('\\', r'\textbackslash{}'), ('&', r'\&'), ('%', r'\%'), ('_', r'\_'), ('#', r'\#'),
                 ('$', r'\$'), ('^', r'\^{}'), ('~', r'\~{}')]:
        s = s.replace(a, b)
    return s.replace('<', r'$<$').replace('>', r'$>$').replace('±', r'$\pm$').replace('ρ', r'$\rho$').replace('Δ', r'$\Delta$')


def to_latex(df, caption, label, index=False):
    d = df.reset_index() if index else df
    cols = list(d.columns)
    lines = [r'\begin{table}[t]', r'\centering', r'\small', f'\\caption{{{_tex_escape(caption)}}}',
             f'\\label{{tab:{label}}}', r'\begin{tabular}{' + 'l' * 1 + 'c' * (len(cols) - 1) + '}', r'\toprule',
             ' & '.join(_tex_escape(c) for c in cols) + r' \\', r'\midrule']
    for _, r in d.iterrows():
        lines.append(' & '.join(_tex_escape(r[c]) for c in cols) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}', '']
    return '\n'.join(lines)


def f3(x):
    return '' if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.3f}"


def fci(m, lo, hi, d=3):
    if m is None or not np.isfinite(m):
        return 'n/a'
    if lo is None or not np.isfinite(lo):
        return f"{m:.{d}f}"
    return f"{m:.{d}f} [{lo:.{d}f}, {hi:.{d}f}]"


def fp(p):
    if p is None or not np.isfinite(p):
        return 'n/a'
    return '<0.001' if p < 0.001 else f"{p:.3f}"


def section(name):
    def deco(fn):
        def run(*a, **k):
            log(f"== {name}")
            try:
                return fn(*a, **k)
            except Exception as e:
                tb = traceback.format_exc(); ERRORS.append(f"{name}: {e}"); print(tb, flush=True)
                REPORT.append(f"\n**{name}: FAILED** ({e}). See the analysis log.\n")
                return None
        return run
    return deco


# ----------------------------------------------------------------------------
# statistics
# ----------------------------------------------------------------------------
def boot_mean(x, B):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if len(x) == 0: return np.nan, np.nan, np.nan
    if len(x) < 3: return float(x.mean()), np.nan, np.nan
    idx = RNG.integers(0, len(x), (B, len(x))); m = x[idx].mean(1)
    return float(x.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def auc_fast(pos, neg):
    pos = np.asarray(pos, float); neg = np.asarray(neg, float)
    if len(pos) == 0 or len(neg) == 0: return np.nan
    r = stats.rankdata(np.concatenate([pos, neg]))
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def auc_ci(pos, neg, B):
    a = auc_fast(pos, neg)
    if len(pos) < 3 or len(neg) < 3: return a, np.nan, np.nan
    bs = [auc_fast(pos[RNG.integers(0, len(pos), len(pos))], neg[RNG.integers(0, len(neg), len(neg))]) for _ in range(B)]
    return a, float(np.nanpercentile(bs, 2.5)), float(np.nanpercentile(bs, 97.5))


def paired_test(x, y):
    """two-sided Wilcoxon signed-rank (x vs y) + matched-pairs rank-biserial r."""
    x = np.asarray(x, float); y = np.asarray(y, float); ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]; d = x - y; nz = d[d != 0]
    if len(nz) < 5: return {'W': np.nan, 'p': np.nan, 'r_rb': np.nan, 'n': int(len(d))}
    W, p = stats.wilcoxon(x, y, alternative='two-sided')
    r = stats.rankdata(np.abs(nz)); rb = (r[nz > 0].sum() - r[nz < 0].sum()) / r.sum()
    return {'W': float(W), 'p': float(p), 'r_rb': float(rb), 'n': int(len(d))}


def one_sample_test(v):
    v = np.asarray(v, float); v = v[np.isfinite(v)]; nz = v[v != 0]
    if len(nz) < 5: return np.nan, np.nan
    W, p = stats.wilcoxon(nz)
    r = stats.rankdata(np.abs(nz)); rb = (r[nz > 0].sum() - r[nz < 0].sum()) / r.sum()
    return float(p), float(rb)


def bh(p):
    p = np.asarray(p, float); out = np.full_like(p, np.nan); ok = np.isfinite(p)
    if ok.sum() == 0: return out
    q = p[ok]; n = len(q); order = np.argsort(q); ranked = q[order] * n / (np.arange(n) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]; res = np.empty(n); res[order] = np.minimum(ranked, 1)
    out[ok] = res; return out


def fid(X, Y):
    X = np.asarray(X, np.float64); Y = np.asarray(Y, np.float64)
    mu1, mu2 = X.mean(0), Y.mean(0)
    c1 = np.cov(X, rowvar=False) + np.eye(X.shape[1]) * 1e-6
    c2 = np.cov(Y, rowvar=False) + np.eye(Y.shape[1]) * 1e-6
    try:
        cm = sqrtm(c1 @ c2, disp=False)[0]   # SciPy < 1.16
    except TypeError:
        cm = sqrtm(c1 @ c2)                  # SciPy >= 1.16 removed `disp`
    cm = np.real(cm)
    return float(((mu1 - mu2) ** 2).sum() + np.trace(c1 + c2 - 2 * cm))


def kid(X, Y, m, n_sub):
    X = np.asarray(X, np.float64); Y = np.asarray(Y, np.float64); d = X.shape[1]
    m = int(min(m, len(X), len(Y)))
    if m < 4: return np.nan, np.nan, m
    vals = []
    for _ in range(n_sub):
        x = X[RNG.choice(len(X), m, replace=False)]; y = Y[RNG.choice(len(Y), m, replace=False)]
        kxx = (x @ x.T / d + 1) ** 3; kyy = (y @ y.T / d + 1) ** 3; kxy = (x @ y.T / d + 1) ** 3
        vals.append((kxx.sum() - np.trace(kxx)) / (m * (m - 1)) + (kyy.sum() - np.trace(kyy)) / (m * (m - 1)) - 2 * kxy.mean())
    return float(np.mean(vals)), float(np.std(vals)), m


def rank_rows(a):
    return stats.rankdata(a, axis=-1)


def traj_stats(T, levels):
    """T: (n_traj, n_levels). Per-trajectory OLS slope and Spearman rho."""
    x = np.asarray(levels, float); xc = x - x.mean()
    slope = ((T - T.mean(1, keepdims=True)) * xc).sum(1) / (xc ** 2).sum()
    rx = stats.rankdata(x); rx = rx - rx.mean()
    ry = rank_rows(T); ry = ry - ry.mean(1, keepdims=True)
    den = np.sqrt((ry ** 2).sum(1) * (rx ** 2).sum())
    rho = np.where(den > 0, (ry * rx).sum(1) / np.where(den > 0, den, 1), np.nan)
    return slope, rho


# ----------------------------------------------------------------------------
# data
# ----------------------------------------------------------------------------
def load_all(work):
    D = {}
    for sp in sorted(glob.glob(os.path.join(work, 'sets', '*.pt'))):
        name = os.path.basename(sp)[:-3]
        s = load_set(sp)
        rec = {'meta': pd.DataFrame(s['meta']), 'info': s['info'], 'n': int(len(s['imgs'])), 'path': sp, 'A': {}, 'B': {}}
        for tag in ['A', 'B']:
            f = os.path.join(work, 'scores', f'{tag}_{name}.npz')
            if os.path.exists(f):
                z = np.load(f); rec[tag] = {k: z[k] for k in z.files}
        D[name] = rec
        del s
    return D


def ev_out(rec, ev):
    if ev == 'cbma': return rec['A'].get('cbma')
    if ev == 'cbmb': return rec['B'].get('cbmb')
    if ev == 'isic18': return rec['B'].get('isic18')
    if ev == 'dark': return rec['A'].get('dark')


def have(D, *names):
    return all(n in D for n in names)


def imgs_of(rec):
    return load_set(rec['path'])['imgs']


# ----------------------------------------------------------------------------
# sections
# ----------------------------------------------------------------------------
@section('Evaluator validity (Derm7pt test and ISIC 2018 test)')
def s_evaluators(cfg, out, R):
    work = cfg['work']; rows = []
    j1 = os.path.join(work, 'results', 'derm7pt_evaluators.json')
    j2 = os.path.join(work, 'results', 'isic18_evaluator.json')
    dz = os.path.join(work, 'results', 'derm7pt_test_preds.npz')
    E1 = json.load(open(j1)) if os.path.exists(j1) else None
    E2 = json.load(open(j2)) if os.path.exists(j2) else None
    R['evaluators'] = {'derm7pt': E1, 'isic18': E2}
    if E1 is None:
        WARN.append('CBM-B was not trained (train_cbmb failed or did not run); independent replication is unavailable.')
        return
    agree = {}
    if os.path.exists(dz):
        z = np.load(dz)
        for i, c in enumerate(CONCEPT_NAMES):
            agree[c] = float(stats.spearmanr(z['a_bneck'][:, i], z['b_bneck'][:, i])[0])
    inv = {v: k for k, v in ISIC18_TO_DERM.items()}
    for i, c in enumerate(CONCEPT_NAMES):
        row = {'Concept': c, 'Test positives': E1['positives_test'][c],
               'CBM-A native': f3(E1['auc']['CBM-A_native'][i]), 'CBM-A 128': f3(E1['auc']['CBM-A_bottleneck'][i]),
               'CBM-B native': f3(E1['auc']['CBM-B_native'][i]), 'CBM-B 128': f3(E1['auc']['CBM-B_bottleneck'][i]),
               'ρ(A,B) test': f3(agree.get(c, np.nan))}
        if E2 is not None and c in inv:
            a = inv[c]; j = ISIC18_ATTRS.index(a)
            row['ISIC18 128 (attr)'] = f"{f3(E2['auc']['ISIC18_bottleneck'][j])} ({a})"
        rows.append(row)
    mean_row = {'Concept': 'mean', 'Test positives': '',
                'CBM-A native': f3(np.nanmean(E1['auc']['CBM-A_native'])), 'CBM-A 128': f3(np.nanmean(E1['auc']['CBM-A_bottleneck'])),
                'CBM-B native': f3(np.nanmean(E1['auc']['CBM-B_native'])), 'CBM-B 128': f3(np.nanmean(E1['auc']['CBM-B_bottleneck'])),
                'ρ(A,B) test': f3(np.nanmean(list(agree.values())) if agree else np.nan)}
    rows.append(mean_row)
    df = pd.DataFrame(rows).fillna('')
    out.table(df, 'T1_evaluator_validity',
              'Per-concept test AUC of the three evaluators. Derm7pt held-out test set (official split). '
              '"128" means the image was first resized to 128x128 (the diffusion resolution), the same pipeline used '
              'for generated images. rho(A,B): Spearman correlation between CBM-A and CBM-B outputs on the test images.')
    m_nat = float(np.nanmean(E1['auc']['CBM-A_native']))
    R['cbma_native_check'] = {'mean_auc': m_nat, 'paper': PAPER['cbma_test_mean_auc'],
                              'diag_auc': E1['diag_auc']['CBM-A_native'], 'paper_diag': PAPER['cbma_test_diag_auc'],
                              'match': abs(m_nat - PAPER['cbma_test_mean_auc']) < 0.005}
    REPORT.append(f"\n## 1. Evaluators\n\n- CBM-A on the Derm7pt test set at native input: mean concept AUC {m_nat:.4f} "
                  f"(paper 0.8224), diagnosis AUC {E1['diag_auc']['CBM-A_native']:.4f} (paper 0.5425). "
                  f"{'MATCH: this is the paper checkpoint.' if R['cbma_native_check']['match'] else 'MISMATCH: check that the CBM-A path is the v3 checkpoint.'}")
    REPORT.append(f"- At the 128x128 pipeline: CBM-A mean AUC {np.nanmean(E1['auc']['CBM-A_bottleneck']):.3f}; "
                  f"CBM-B mean AUC {np.nanmean(E1['auc']['CBM-B_bottleneck']):.3f}.")
    if E2 is not None:
        REPORT.append(f"- ISIC 2018 detector (128 pipeline) test AUC: " + ', '.join(
            f"{a} {x:.3f} (pos={E2['positives_test'][a]})" for a, x in zip(ISIC18_ATTRS, E2['auc']['ISIC18_bottleneck'])) +
            f". ISIC 2018 IDs also found in the ISIC 2020 CSV: {E2.get('id_overlap_with_isic2020_csv')}.")
    else:
        WARN.append('ISIC 2018 Task 2 detector unavailable: pigment-network support check (G1d) cannot run.')
    # figure
    fig, ax = plt.subplots(figsize=(6.0, 2.6))
    x = np.arange(7); w = 0.27
    for k, (lab, key, col) in enumerate([('CBM-A native', 'CBM-A_native', '#2a78d6'), ('CBM-A 128', 'CBM-A_bottleneck', '#86b6ef'),
                                         ('CBM-B 128', 'CBM-B_bottleneck', '#eb6834')]):
        ax.bar(x + (k - 1) * w, E1['auc'][key], w * 0.92, color=col, label=lab, edgecolor='white', linewidth=0.8)
    ax.axhline(0.5, color=INK2, lw=0.8, ls=':'); ax.set_ylim(0.3, 1.0)
    ax.set_xticks(x); ax.set_xticklabels(SHORT); ax.set_ylabel('Test AUC (Derm7pt, n=%d)' % E1['n_test'])
    ax.legend(ncol=3, loc='upper left', bbox_to_anchor=(0, 1.15))
    out.fig(fig, 'S_fig_evaluator_auc')


@section('Reproduction of the paper statistics (NB4_Stats_Supplement)')
def s_repro(cfg, out, R, D):
    f = os.path.join(cfg['work'], 'results', 'repro_results.json')
    if not os.path.exists(f):
        WARN.append('Reproduction run missing.'); return
    rr = json.load(open(f)); rows = []
    for k in ['mel', 'all']:
        for s in ['W', 'p', 'r_rb']:
            rows.append({'Statistic': f'Wilcoxon {k} {s}', 'Paper': PAPER[f'wilcoxon_{k}'][s], 'Reproduced': rr[f'wilcoxon_{k}'][s]})
    rows.append({'Statistic': 'Wilcoxon ben p', 'Paper': PAPER['wilcoxon_ben']['p'], 'Reproduced': rr['wilcoxon_ben']['p']})
    cbmb_r = {}
    if 'repro_sweep' in D and ev_out(D['repro_sweep'], 'cbmb') is not None:
        rec = D['repro_sweep']; Yb = ev_out(rec, 'cbmb'); m = rec['meta']
        for i, c in enumerate(CONCEPT_NAMES):
            sel = m['concept_idx'].values == i
            cbmb_r[c] = float(stats.pearsonr(m['level'].values[sel], Yb[sel, i])[0])
    dmax = 0.0
    for c in CONCEPT_NAMES:
        d = rr['pearson_r'][c] - PAPER['pearson_r'][c]; dmax = max(dmax, abs(d))
        rows.append({'Statistic': f'Pearson r {c}', 'Paper': PAPER['pearson_r'][c], 'Reproduced': rr['pearson_r'][c],
                     'CBM-B on same images': cbmb_r.get(c, np.nan)})
        rows.append({'Statistic': f'Spearman rho {c}', 'Paper': PAPER['spearman_rho'][c], 'Reproduced': rr['spearman_rho'][c]})
    df = pd.DataFrame(rows)
    df['Δ'] = df['Reproduced'] - df['Paper']
    for col in ['Paper', 'Reproduced', 'Δ', 'CBM-B on same images']:
        if col in df: df[col] = df[col].map(lambda v: '' if not np.isfinite(v) else (f"{v:.0f}" if abs(v) >= 100 else f"{v:.4f}"))
    out.table(df.fillna(''), 'S_T_reproduction', 'Paper statistics versus a re-run of the original supplement code '
              '(same checkpoints, seeds and draw order). Exact equality needs the same GPU model and library versions.')
    verdict = 'DRY RUN: sizes and DDIM steps differ from the paper, so this comparison is not meaningful' if cfg.get('dry_run') else 'REPRODUCED (max |delta r| <= 0.02)' if dmax <= 0.02 else (
        'APPROXIMATELY REPRODUCED (max |delta r| <= 0.10)' if dmax <= 0.10 else 'NOT REPRODUCED (max |delta r| > 0.10)')
    R['repro'] = {'max_abs_delta_r': dmax, 'verdict': verdict, 'torch': rr.get('torch'), 'gpu': rr.get('gpu'),
                  'cbmb_r_on_paper_sweep': cbmb_r, 'ckpt_main_epoch': rr['main_ckpt']['epoch_1based'],
                  'ckpt_base_epoch': rr['base_ckpt']['epoch_1based']}
    REPORT.append(f"\n## 2. Reproduction of the paper's statistics\n\n- Verdict: **{verdict}** (max |delta Pearson r| = {dmax:.3f}; "
                  f"torch {rr.get('torch')}, {rr.get('gpu')}). Checkpoints: main epoch {rr['main_ckpt']['epoch_1based']}, "
                  f"baseline epoch {rr['base_ckpt']['epoch_1based']}.")
    REPORT.append(f"- Wilcoxon melanoma: paper p=0.0009, r=0.254; re-run p={rr['wilcoxon_mel']['p']:.4g}, r={rr['wilcoxon_mel']['r_rb']:.3f}.")
    if cbmb_r:
        REPORT.append("- The paper's own sweep images re-scored by CBM-B (Pearson r): " +
                      ', '.join(f"{s} {cbmb_r[c]:+.3f}" for s, c in zip(SHORT, CONCEPT_NAMES)))
    # figure
    fig, ax = plt.subplots(figsize=(4.2, 2.6)); y = np.arange(7)[::-1]
    ax.scatter([PAPER['pearson_r'][c] for c in CONCEPT_NAMES], y, marker='o', facecolor='white', edgecolor=INK, s=24, label='Paper (CBM-A)', zorder=3)
    ax.scatter([rr['pearson_r'][c] for c in CONCEPT_NAMES], y, marker='o', color=C_EV['cbma'], s=14, label='Re-run (CBM-A)', zorder=4)
    if cbmb_r:
        ax.scatter([cbmb_r[c] for c in CONCEPT_NAMES], y, marker='s', color=C_EV['cbmb'], s=14, label='Same images, CBM-B', zorder=4)
    ax.axvline(0, color=INK2, lw=0.8); ax.set_yticks(y); ax.set_yticklabels(SHORT); ax.set_xlabel('Pearson r (commanded level vs evaluator output)')
    ax.legend(loc='lower right'); out.fig(fig, 'S_fig_reproduction')


def ca_scores(P_, c):
    return 1.0 - np.abs(P_ - np.asarray(c)[None, :]).mean(1)


def cos_scores(P_, c):
    c = np.asarray(c); return (P_ @ c) / (np.linalg.norm(c) * np.linalg.norm(P_, axis=1) + 1e-8)


@section('Concept agreement (CBM-A units) with real anchors and chance levels')
def s_agreement(cfg, out, R, D):
    B = cfg['analysis']['n_boot']; cm, cb = np.array(MEL_VEC_LIST), np.array(BEN_VEC_LIST)
    need = ['eval_main_mel', 'eval_main_ben', 'eval_base_mel', 'eval_base_ben', 'real_mel', 'real_ben']
    if not have(D, *need):
        WARN.append('Agreement section skipped: missing sets ' + str([n for n in need if n not in D])); return
    A = {n: ev_out(D[n], 'cbma') for n in need}
    nat = {n: D[n]['meta'][[f'concept_{c}' for c in CONCEPT_NAMES]].values for n in ['real_mel', 'real_ben']}
    rows, store = [], {}
    for cls, c in [('mel', cm), ('ben', cb)]:
        opp = 'ben' if cls == 'mel' else 'mel'
        entries = [('Real ISIC 2020, 128 pipeline', ca_scores(A[f'real_{cls}'], c)),
                   ('Real ISIC 2020, native (pseudo-label CSV)', ca_scores(nat[f'real_{cls}'], c)),
                   ('ClinDiffXAI (concept-conditioned)', ca_scores(A[f'eval_main_{cls}'], c)),
                   ('Baseline (class-conditioned)', ca_scores(A[f'eval_base_{cls}'], c)),
                   ('ClinDiffXAI, opposite-class images', ca_scores(A[f'eval_main_{opp}'], c))]
        for lab, s in entries:
            m, lo, hi = boot_mean(s, B); store[(cls, lab)] = (m, lo, hi)
            rows.append({'Class target': cls, 'Images': lab, 'n': len(s), 'Agreement [95% CI]': fci(m, lo, hi)})
        chance = {'Uniform random predictions (analytic)': 1 - np.mean((c ** 2 + (1 - c) ** 2) / 2),
                  'Constant 0.5 predictions': 1 - np.mean(np.abs(c - 0.5)),
                  'All-zero predictions': 1 - np.mean(c)}
        for lab, v in chance.items():
            store[(cls, lab)] = (v, np.nan, np.nan)
            rows.append({'Class target': cls, 'Images': lab, 'n': '', 'Agreement [95% CI]': f"{v:.3f}"})
    df = pd.DataFrame(rows)
    out.table(df, 'T2a_concept_agreement', 'Concept agreement = 1 - mean_k |c_k - c_hat_k| (CBM-A probability units), '
              'against the conditioning prototype of the same class. Real-image and chance rows give the scale.')
    R['agreement'] = {f"{k[0]}|{k[1]}": v for k, v in store.items()}
    # per-concept MAE (supplement)
    mae_rows = []
    for cls, c in [('mel', cm), ('ben', cb)]:
        for lab, key in [('Real 128', f'real_{cls}'), ('ClinDiffXAI', f'eval_main_{cls}'), ('Baseline', f'eval_base_{cls}')]:
            e = np.abs(A[key] - c[None, :]).mean(0)
            mae_rows.append({'Class': cls, 'Images': lab, **{s: f"{v:.3f}" for s, v in zip(SHORT, e)}})
    out.table(pd.DataFrame(mae_rows), 'S_T_per_concept_MAE', 'Per-concept mean absolute error |c_k - c_hat_k| (CBM-A units).')
    # legacy cosine CCFS with its floor (supplement)
    U = RNG.random((200000, 7)); cos_rows = []
    for cls, c in [('mel', cm), ('ben', cb)]:
        r = {'Class': cls, 'Uniform-random floor': f"{cos_scores(U, c).mean():.3f}"}
        for lab, key in [('Real 128', f'real_{cls}'), ('ClinDiffXAI', f'eval_main_{cls}'), ('Baseline', f'eval_base_{cls}')]:
            r[lab] = f"{cos_scores(A[key], c).mean():.3f}"
        cos_rows.append(r)
    out.table(pd.DataFrame(cos_rows), 'S_T_legacy_cosine_CCFS', 'Legacy cosine CCFS of the first submission, shown with the '
              'score of uniform random predictions. Cosine between non-negative vectors has a high floor, so it is not used as a primary metric.')
    # paired main vs baseline (same starting noise)
    prow = []
    for cls, c in [('mel', cm), ('ben', cb)]:
        x, y = ca_scores(A[f'eval_main_{cls}'], c), ca_scores(A[f'eval_base_{cls}'], c)
        t = paired_test(x, y); m, lo, hi = boot_mean(x - y, B)
        prow.append({'Class': cls, 'n pairs': t['n'], 'Mean Δ (main - base) [95% CI]': fci(m, lo, hi, 4),
                     'Wilcoxon W': f"{t['W']:.0f}" if np.isfinite(t['W']) else 'n/a', 'p (two-sided)': fp(t['p']),
                     'rank-biserial r': f3(t['r_rb'])})
        R.setdefault('paired_ca', {})[cls] = {**t, 'mean_diff': m, 'lo': lo, 'hi': hi}
    out.table(pd.DataFrame(prow), 'T2b_paired_agreement', 'Paired comparison of per-image concept agreement. Each '
              'ClinDiffXAI image and its baseline partner start from the same noise tensor.')
    REPORT.append("\n## 3. Concept agreement (CBM-A units)\n")
    for cls in ['mel', 'ben']:
        g = lambda lab: store[(cls, lab)][0]
        REPORT.append(f"- {cls}: real {g('Real ISIC 2020, 128 pipeline'):.3f} | ClinDiffXAI {g('ClinDiffXAI (concept-conditioned)'):.3f} | "
                      f"baseline {g('Baseline (class-conditioned)'):.3f} | uniform chance {g('Uniform random predictions (analytic)'):.3f} | "
                      f"all-zero {g('All-zero predictions'):.3f}. Paired main-base p={fp(R['paired_ca'][cls]['p'])}, "
                      f"r={R['paired_ca'][cls]['r_rb']:.3f}.")
    # figure
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.5), sharey=False)
    for ax, cls in zip(axes, ['mel', 'ben']):
        labs = [('Real', 'Real ISIC 2020, 128 pipeline', C_MODEL['real']), ('ClinDiffXAI', 'ClinDiffXAI (concept-conditioned)', C_MODEL['main']),
                ('Baseline', 'Baseline (class-conditioned)', C_MODEL['base'])]
        for i, (short, lab, col) in enumerate(labs):
            m, lo, hi = store[(cls, lab)]
            ax.errorbar(i, m, yerr=[[m - lo], [hi - m]] if np.isfinite(lo) else None, fmt='o', color=col, ms=6, capsize=3, lw=1.2)
        # chance-level values are in Table T2a; they are not drawn here because the
        # prototypes are extreme (p90 / p10), so real images can sit below "uniform chance"
        ax.set_xticks(range(3)); ax.set_xticklabels([l[0] for l in labs]); ax.set_xlim(-0.5, 2.5)
        ax.set_title('Melanoma prototype' if cls == 'mel' else 'Benign prototype')
        ax.set_ylabel('Concept agreement (CBM-A)')
    fig.tight_layout(); out.fig(fig, 'fig_concept_agreement')


@section('Condition discriminability (per-concept AUC and real-trained classifier)')
def s_discrim(cfg, out, R, D):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    B = cfg['analysis']['n_boot_auc']
    need = ['eval_main_mel', 'eval_main_ben', 'eval_base_mel', 'eval_base_ben', 'real_mel', 'real_ben']
    if not have(D, *need): return
    rows, comp_rows, R['discrim'] = [], [], {}
    RC = have(D, 'eval_realc_mel', 'eval_realc_ben')
    for ev in ['cbma', 'cbmb', 'isic18']:
        if ev_out(D['eval_main_mel'], ev) is None: continue
        O = {n: ev_out(D[n], ev) for n in need}
        if RC:
            O['eval_realc_mel'] = ev_out(D['eval_realc_mel'], ev); O['eval_realc_ben'] = ev_out(D['eval_realc_ben'], ev)
        pairs = [('Real', 'real_mel', 'real_ben'), ('ClinDiffXAI', 'eval_main_mel', 'eval_main_ben'),
                 ('Baseline', 'eval_base_mel', 'eval_base_ben')]
        if RC: pairs.append((RC_LAB, 'eval_realc_mel', 'eval_realc_ben'))
        for k, cname in enumerate(EV_COLS[ev]):
            r = {'Evaluator': EV_LABEL[ev], 'Output': cname}
            for lab, a, b in pairs:
                r[lab] = fci(*auc_ci(O[a][:, k], O[b][:, k], B))
            rows.append(r)
        # composite: logistic regression trained on REAL melanoma vs benign evaluator outputs
        Xr = np.concatenate([O['real_mel'], O['real_ben']]); yr = np.r_[np.ones(len(O['real_mel'])), np.zeros(len(O['real_ben']))]
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, class_weight='balanced', max_iter=2000))
        nf = int(min(5, yr.sum(), (1 - yr).sum()))
        if nf < 2:
            WARN.append(f'discriminability for {ev}: too few real images'); continue
        cv = cross_val_predict(clf, Xr, yr, cv=StratifiedKFold(nf, shuffle=True, random_state=0), method='predict_proba')[:, 1]
        a_real = auc_ci(cv[yr == 1], cv[yr == 0], B)
        clf.fit(Xr, yr)
        sc = {n: clf.predict_proba(O[n])[:, 1] for n in need[:4]}
        a_main = auc_ci(sc['eval_main_mel'], sc['eval_main_ben'], B); a_base = auc_ci(sc['eval_base_mel'], sc['eval_base_ben'], B)
        # paired bootstrap of the AUC difference (same noise indices for both models)
        nm, nb_ = len(sc['eval_main_mel']), len(sc['eval_main_ben']); diffs = []
        for _ in range(B):
            i = RNG.integers(0, nm, nm); j = RNG.integers(0, nb_, nb_)
            diffs.append(auc_fast(sc['eval_main_mel'][i], sc['eval_main_ben'][j]) - auc_fast(sc['eval_base_mel'][i], sc['eval_base_ben'][j]))
        dlo, dhi = np.percentile(diffs, [2.5, 97.5])
        # class-fidelity paired test on the melanoma-likeness score
        tm = paired_test(sc['eval_main_mel'], sc['eval_base_mel']); tb = paired_test(-sc['eval_main_ben'], -sc['eval_base_ben'])
        comp_rows.append({'Evaluator': EV_LABEL[ev], 'Real (5-fold CV)': fci(*a_real), 'ClinDiffXAI': fci(*a_main),
                          'Baseline': fci(*a_base), 'Δ AUC [95% CI]': fci(a_main[0] - a_base[0], dlo, dhi),
                          'mel-likeness, mel images: p, r': f"{fp(tm['p'])}, {f3(tm['r_rb'])}",
                          'ben-likeness, ben images: p, r': f"{fp(tb['p'])}, {f3(tb['r_rb'])}"})
        R['discrim'][ev] = {'real': a_real, 'main': a_main, 'base': a_base, 'delta': (a_main[0] - a_base[0], dlo, dhi),
                            'paired_mel': tm, 'paired_ben': tb}
        if RC:
            sr = {n: clf.predict_proba(O[n])[:, 1] for n in ['eval_realc_mel', 'eval_realc_ben']}
            a_rc = auc_ci(sr['eval_realc_mel'], sr['eval_realc_ben'], B)
            trm = paired_test(sr['eval_realc_mel'], sc['eval_base_mel']); trb = paired_test(-sr['eval_realc_ben'], -sc['eval_base_ben'])
            comp_rows[-1][RC_LAB] = fci(*a_rc)
            comp_rows[-1][f'{RC_LAB} vs baseline, mel images: p, r'] = f"{fp(trm['p'])}, {f3(trm['r_rb'])}"
            comp_rows[-1][f'{RC_LAB} vs baseline, ben images: p, r'] = f"{fp(trb['p'])}, {f3(trb['r_rb'])}"
            R['discrim'][ev].update({'realc': a_rc, 'paired_realc_mel': trm, 'paired_realc_ben': trb})
    out.table(pd.DataFrame(rows), 'S_T_per_output_discriminability', 'AUC for separating melanoma-conditioned from '
              'benign-conditioned images using one evaluator output at a time; "Real" separates real melanoma from real benign images.')
    out.table(pd.DataFrame(comp_rows), 'T2c_condition_discriminability', 'Condition discriminability. A logistic regression '
              'is trained on evaluator outputs of REAL images only (melanoma vs held-out benign) and applied unchanged to generated '
              'images. Positive r: ClinDiffXAI images are more class-typical than their baseline partners.')
    REPORT.append("\n## 4. Condition discriminability (classifier trained on real images only)\n")
    for ev, v in R['discrim'].items():
        REPORT.append(f"- {EV_LABEL[ev]}: real {v['real'][0]:.3f} | ClinDiffXAI {v['main'][0]:.3f} | baseline {v['base'][0]:.3f} | "
                      f"delta {v['delta'][0]:+.3f} [{v['delta'][1]:+.3f}, {v['delta'][2]:+.3f}]"
                      + (f" | {RC_LAB} {v['realc'][0]:.3f}" if 'realc' in v else ''))
    evs = list(R['discrim'].keys()); x = np.arange(len(evs))
    bars = [('Real (CV)', 'real'), ('ClinDiffXAI', 'main'), ('Baseline', 'base')]
    if RC: bars.append((RC_LAB, 'realc'))
    w = 0.8 / len(bars); fig, ax = plt.subplots(figsize=(4.4 + 0.8 * (len(bars) - 3), 2.5))
    for i, (lab, key) in enumerate(bars):
        col = C_MODEL[key]
        i = i - (len(bars) - 1) / 2 + 1
        m = [R['discrim'][e][key][0] for e in evs]; lo = [R['discrim'][e][key][1] for e in evs]; hi = [R['discrim'][e][key][2] for e in evs]
        ax.bar(x + (i - 1) * w, m, w * 0.92, color=col, label=lab, edgecolor='white', linewidth=0.8)
        ax.errorbar(x + (i - 1) * w, m, yerr=[np.array(m) - np.array(lo), np.array(hi) - np.array(m)], fmt='none', ecolor=INK, lw=0.8, capsize=2)
    lows = [R['discrim'][e][k][1] for e in evs for _, k in bars if np.isfinite(R['discrim'][e][k][1])]
    ax.axhline(0.5, color=INK2, lw=0.8, ls=':'); ax.set_ylim(min([0.4] + [v - 0.05 for v in lows]), 1.02)
    ax.set_xticks(x); ax.set_xticklabels([EV_LABEL[e] for e in evs]); ax.set_ylabel('AUC, melanoma vs benign')
    ax.legend(ncol=len(bars), loc='upper left', bbox_to_anchor=(0, 1.16)); out.fig(fig, 'fig_condition_discriminability')


@section('Distribution distances (FID and KID in three feature spaces)')
def s_distribution(cfg, out, R, D):
    need = ['eval_main_mel', 'eval_main_ben', 'eval_base_mel', 'eval_base_ben', 'real_mel', 'real_ben']
    if not have(D, *need): return
    spaces = [('Inception', 'A', 'incep'), ('CBM-A (MedFID)', 'A', 'cbma_feat'), ('CBM-B', 'B', 'cbmb_feat')]
    m_sub, n_sub = cfg['analysis']['kid_subset'], cfg['analysis']['kid_n_sub']
    rows, R['dist'] = [], {}
    for sname, tag, key in spaces:
        if key not in D['eval_main_mel'][tag]: continue
        F_ = {n: D[n][tag][key] for n in need}
        nm = len(F_['real_mel']); perm_m = RNG.permutation(nm); perm_b = RNG.permutation(len(F_['real_ben']))
        real_all = np.concatenate([F_['real_mel'], F_['real_ben'][perm_b[:nm]]])
        hm, hb = nm // 2, min(nm // 2, len(perm_b) // 2)
        comps = {
            ('all', 'ClinDiffXAI'): (np.concatenate([F_['eval_main_mel'], F_['eval_main_ben']]), real_all),
            ('all', 'Baseline'): (np.concatenate([F_['eval_base_mel'], F_['eval_base_ben']]), real_all),
            ('all', 'Real vs real (floor)'): (np.concatenate([F_['real_mel'][perm_m[:hm]], F_['real_ben'][perm_b[:hb]]]),
                                              np.concatenate([F_['real_mel'][perm_m[hm:2 * hm]], F_['real_ben'][perm_b[hb:2 * hb]]])),
            ('mel', 'ClinDiffXAI'): (F_['eval_main_mel'], F_['real_mel']), ('mel', 'Baseline'): (F_['eval_base_mel'], F_['real_mel']),
            ('mel', 'Real vs real (floor)'): (F_['real_mel'][perm_m[:hm]], F_['real_mel'][perm_m[hm:2 * hm]]),
            ('ben', 'ClinDiffXAI'): (F_['eval_main_ben'], F_['real_ben']), ('ben', 'Baseline'): (F_['eval_base_ben'], F_['real_ben']),
            ('ben', 'Real vs real (floor)'): (F_['real_ben'][perm_b[:len(perm_b) // 2]], F_['real_ben'][perm_b[len(perm_b) // 2:]]),
        }
        if have(D, 'eval_realc_mel', 'eval_realc_ben') and key in D['eval_realc_mel'][tag]:
            rm_, rb_ = D['eval_realc_mel'][tag][key], D['eval_realc_ben'][tag][key]
            comps[('all', RC_LAB)] = (np.concatenate([rm_, rb_]), real_all)
            comps[('mel', RC_LAB)] = (rm_, F_['real_mel']); comps[('ben', RC_LAB)] = (rb_, F_['real_ben'])
        for (split, lab), (X, Y) in comps.items():
            fd = fid(X, Y) if min(len(X), len(Y)) > 2 else np.nan
            km, ks, mm = kid(X, Y, m_sub, n_sub)
            rows.append({'Feature space': sname, 'Split': split, 'Images': lab, 'n gen/ref': f"{len(X)}/{len(Y)}",
                         'FID': f"{fd:.1f}", 'KID x1e3 (mean ± sd)': f"{km * 1e3:.2f} ± {ks * 1e3:.2f}", 'KID subset': mm})
            R['dist'][f"{sname}|{split}|{lab}"] = {'fid': fd, 'kid': km, 'kid_sd': ks, 'n': [len(X), len(Y)], 'm': mm}
        log(f"  {sname} done")
    out.table(pd.DataFrame(rows), 'T2d_distribution_distances', 'FID and KID (unbiased, cubic polynomial kernel) between '
              'generated and real images. Real melanoma = all 584 ISIC 2020 melanomas (all were in diffusion training); '
              'real benign = benign images NOT used in diffusion training. "Real vs real" gives the sampling floor. '
              'Inception features follow NB4 (torchvision weights), so values are not comparable to literature FIDs.')
    REPORT.append("\n## 5. Distribution distances (lower is better; KID is unbiased at these n)\n")
    for k, v in R['dist'].items():
        if k.split('|')[1] in ('all', 'mel'):
            REPORT.append(f"- {k}: FID {v['fid']:.1f}, KID x1e3 {v['kid'] * 1e3:.2f}")


def build_traj(rec, Y, j):
    m = rec['meta']; sel = m['concept_idx'].values == j
    levels = np.sort(np.unique(m['level'].values[sel])); nid = np.sort(np.unique(m['noise_id'].values[sel]))
    li = np.searchsorted(levels, m['level'].values[sel]); ni = np.searchsorted(nid, m['noise_id'].values[sel])
    T = np.full((len(nid), len(levels), Y.shape[1]), np.nan)
    T[ni, li] = Y[sel]
    return T, levels


@section('Dose-response (per-concept control) with independent evaluators')
def s_dose(cfg, out, R, D):
    B = cfg['analysis']['n_boot']
    if not have(D, 'real_mel', 'real_ben'): return
    inv = {v: k for k, v in ISIC18_TO_DERM.items()}
    R['dose'] = {}; diag_rows = {}; xt = {}
    for base in ['mel', 'ben']:
        nm = f'sweep_{base}'
        if nm not in D: WARN.append(f'{nm} missing'); continue
        rec = D[nm]; R['dose'][base] = {}; rows = []
        for ev in EVS:
            Y = ev_out(rec, ev)
            if Y is None: continue
            rm, rb = ev_out(D['real_mel'], ev), ev_out(D['real_ben'], ev)
            sd = np.sqrt((np.nanvar(rm, 0) + np.nanvar(rb, 0)) / 2) + 1e-9; mu_b = np.nanmean(rb, 0)
            K = Y.shape[1]; MR = np.full((7, K), np.nan)
            fam = []
            for j in range(7):
                T, levels = build_traj(rec, Y, j)
                for k in range(K):
                    Tk = T[:, :, k]
                    ok = np.isfinite(Tk).all(1)
                    if ok.sum() < 3: continue
                    slope, rho = traj_stats(Tk[ok], levels); MR[j, k] = np.nanmean(rho)
                    if ev in ('cbma', 'cbmb'): is_diag = (k == j)
                    elif ev == 'isic18': is_diag = (CONCEPT_NAMES[j] in inv and ISIC18_ATTRS[k] == inv[CONCEPT_NAMES[j]])
                    else: is_diag = (CONCEPT_NAMES[j] == 'pigmentation')
                    if not is_diag: continue
                    p, rbis = one_sample_test(slope)
                    ms, mlo, mhi = boot_mean(slope, B)
                    dz = (Tk[ok][:, -1] - Tk[ok][:, 0]) / sd[k]; dm, dlo, dhi = boot_mean(dz, B)
                    # pooled Spearman over all images (comparable to the paper) with cluster bootstrap
                    xs = np.tile(levels, (ok.sum(), 1)); pr = stats.spearmanr(xs.ravel(), Tk[ok].ravel())[0]
                    bsr = []
                    for _ in range(min(B, 1000)):
                        ii = RNG.integers(0, ok.sum(), ok.sum())
                        bsr.append(stats.spearmanr(xs[ii].ravel(), Tk[ok][ii].ravel())[0])
                    Z = (Tk[ok] - mu_b[k]) / sd[k]
                    cb_ = np.array([Z[RNG.integers(0, len(Z), len(Z))].mean(0) for _ in range(300)])
                    fam.append({'base': base, 'ev': ev, 'concept': CONCEPT_NAMES[j], 'output': EV_COLS[ev][k],
                                'n_traj': int(ok.sum()), 'levels': len(levels), 'median_slope': float(np.median(slope)),
                                'mean_slope': ms, 'slope_lo': mlo, 'slope_hi': mhi, 'mean_traj_rho': float(np.nanmean(rho)),
                                'frac_pos_slope': float(np.mean(slope > 0)), 'p': p, 'r_rb': rbis,
                                'pooled_rho': float(pr), 'pooled_lo': float(np.nanpercentile(bsr, 2.5)),
                                'pooled_hi': float(np.nanpercentile(bsr, 97.5)), 'delta_sd': dm, 'delta_lo': dlo, 'delta_hi': dhi,
                                'curve_mean': Z.mean(0).tolist(),
                                'curve_lo': np.percentile(cb_, 2.5, axis=0).tolist(),
                                'curve_hi': np.percentile(cb_, 97.5, axis=0).tolist(),
                                'levels_list': list(map(float, levels))})
            q = bh([f['p'] for f in fam])
            for f, qq in zip(fam, q): f['p_bh'] = float(qq)
            rows += fam; xt[(base, ev)] = MR
        R['dose'][base] = rows; diag_rows[base] = rows
        tab = [{'Evaluator': EV_LABEL[f['ev']], 'Swept concept': f['concept'], 'Output': f['output'],
                'Median slope': f"{f['median_slope']:+.3f}", 'Mean slope [95% CI]': fci(f['mean_slope'], f['slope_lo'], f['slope_hi']),
                'Traj. with slope > 0': f"{f['frac_pos_slope'] * 100:.0f}%", 'p (BH)': fp(f['p_bh']), 'r_rb': f3(f['r_rb']),
                'Δ 0→1 in real-SD units [95% CI]': fci(f['delta_sd'], f['delta_lo'], f['delta_hi'], 2),
                'Pooled ρ [95% CI]': fci(f['pooled_rho'], f['pooled_lo'], f['pooled_hi'])} for f in rows]
        out.table(pd.DataFrame(tab), f"T3_dose_response_{base}_base",
                  f"Per-concept dose-response, {'melanoma' if base == 'mel' else 'benign'} prototype as the base vector. One "
                  f"concept is set to each level in [0, 1] while the other six stay at the prototype value. All levels share the "
                  f"same starting noises, so each noise gives one trajectory (the unit of analysis). p: two-sided Wilcoxon "
                  f"signed-rank test of per-trajectory OLS slopes against 0, BH-corrected within each evaluator. Delta: change "
                  f"from level 0 to 1 divided by the pooled SD of the evaluator output on real images.")
    for (base, ev), MR in xt.items():
        cols = EV_COLS[ev]
        pd.DataFrame(MR, index=CONCEPT_NAMES, columns=cols).to_csv(os.path.join(out.root, 'tables', f'S_crosstalk_{base}_{ev}.csv'))
    R['crosstalk'] = {f"{b}|{e}": v.tolist() for (b, e), v in xt.items()}
    # --- figure: dose-response curves (one panel per concept) ---
    for base in diag_rows:
        rows = diag_rows[base]
        if not rows: continue
        fig, axes = plt.subplots(2, 4, figsize=(7.2, 3.8), sharex=True)
        axes = axes.ravel()
        for j, c in enumerate(CONCEPT_NAMES):
            ax = axes[j]
            for f in [r for r in rows if r['concept'] == c and (r['ev'] != 'dark' or r['output'] == 'lesion_darkness')]:
                L = np.array(f['levels_list']); m = np.array(f['curve_mean'])
                ax.fill_between(L, f['curve_lo'], f['curve_hi'], color=C_EV[f['ev']], alpha=0.15, lw=0)
                ax.plot(L, m, color=C_EV[f['ev']], marker=M_EV[f['ev']], ms=3, lw=1.5, label=EV_LABEL[f['ev']])
            ax.axhline(0, color=INK2, lw=0.6, ls=':')
            ax.set_title(f"{SHORT[j]} ({c.replace('_', ' ')})", fontsize=7)
            if j >= 3: ax.tick_params(labelbottom=True)
        axes[7].axis('off')
        fig.supylabel('Evaluator output (SD units from the real benign mean)', fontsize=7)
        fig.supxlabel('Commanded concept level', fontsize=7)
        h, l = [], []
        for ev in EVS:
            if any(r['ev'] == ev for r in rows):
                h.append(plt.Line2D([], [], color=C_EV[ev], marker=M_EV[ev], ms=4))
                l.append({'cbma': 'CBM-A (made the\npseudo-labels)', 'cbmb': 'CBM-B (independent\narchitecture)',
                          'isic18': 'ISIC 2018 detector\n(independent data)', 'dark': 'Lesion darkness\n(non-learned)'}[ev])
        axes[7].legend(h, l, loc='center', fontsize=6.5, handlelength=1.5, labelspacing=0.9)
        fig.suptitle(f"Dose-response from the {'melanoma' if base == 'mel' else 'benign'} prototype (mean and 95% CI over trajectories)", fontsize=8)
        fig.tight_layout(); out.fig(fig, f"fig_dose_response_{base}_base" if base == 'mel' else f"S_fig_dose_response_{base}_base")
    # --- figure: cross-talk heat maps (melanoma base) ---
    avail = [ev for ev in ['cbma', 'cbmb', 'isic18'] if ('mel', ev) in xt]
    if avail:
        fig, axes = plt.subplots(1, len(avail), figsize=(2.6 * len(avail) + 0.6, 2.9), squeeze=False); axes = axes[0]
        for ax, ev in zip(axes, avail):
            MR = xt[('mel', ev)]; im = ax.imshow(MR, cmap=DIV, vmin=-1, vmax=1, aspect='auto')
            for a in range(MR.shape[0]):
                for b in range(MR.shape[1]):
                    if np.isfinite(MR[a, b]): ax.text(b, a, f"{MR[a, b]:.2f}", ha='center', va='center', fontsize=5.5, color=INK)
            ax.set_xticks(range(MR.shape[1])); ax.set_xticklabels(SHORT if ev != 'isic18' else ['PN', 'NegN', 'Str', 'Milia', 'Glob'], fontsize=6)
            ax.set_yticks(range(7)); ax.set_yticklabels(SHORT, fontsize=6); ax.grid(False)
            ax.set_xlabel(f'{EV_LABEL[ev]} output'); ax.set_title(EV_LABEL[ev])
            if ev == avail[0]: ax.set_ylabel('Swept concept')
        cb = fig.colorbar(im, ax=list(axes), fraction=0.025, pad=0.02); cb.set_label('Mean per-trajectory Spearman ρ', fontsize=7)
        out.fig(fig, 'fig_concept_crosstalk_mel_base')
    REPORT.append("\n## 6. Dose-response (melanoma base; one trajectory = one starting noise)\n")
    for f in diag_rows.get('mel', []):
        if f['ev'] == 'dark' and f['output'] != 'lesion_darkness': continue
        REPORT.append(f"- {EV_LABEL[f['ev']]:<18} {f['concept']:<22} median slope {f['median_slope']:+.3f}, "
                      f"{f['frac_pos_slope'] * 100:.0f}% of trajectories rise, p_BH={fp(f['p_bh'])}, "
                      f"delta {f['delta_sd']:+.2f} SD, pooled rho {f['pooled_rho']:+.3f}")


@section('Gate 1 (pre-registered decision)')
def s_gate(cfg, out, R):
    G = cfg['gate1']; rows = {(f['ev'], f['concept'], f['output']): f for f in R.get('dose', {}).get('mel', [])}
    E1 = (R.get('evaluators') or {}).get('derm7pt')
    def crit(ev, concept, output):
        f = rows.get((ev, concept, output))
        if f is None: return {'available': False}
        ok = (f['p_bh'] < G['alpha']) and (f['median_slope'] > 0)
        return {'available': True, 'pass': bool(ok), 'p_bh': f['p_bh'], 'median_slope': f['median_slope'],
                'delta_sd': f['delta_sd'], 'frac_pos': f['frac_pos_slope']}
    res = {'rule': G['text'], 'criteria': {}}
    res['criteria']['G1a CBM-B pigmentation'] = crit('cbmb', 'pigmentation', 'pigmentation')
    res['criteria']['G1b CBM-B pigment network'] = crit('cbmb', 'pigment_network', 'pigment_network')
    res['criteria']['G1c darkness for pigmentation'] = crit('dark', 'pigmentation', 'lesion_darkness')
    res['criteria']['G1d ISIC-2018 pigment network'] = crit('isic18', 'pigment_network', 'pigment_network')
    valid = {}
    if E1 is not None:
        for c in ['pigmentation', 'pigment_network']:
            valid[c] = E1['auc']['CBM-B_bottleneck'][CONCEPT_NAMES.index(c)]
    res['cbmb_validity_auc_128'] = valid
    a, b = res['criteria']['G1a CBM-B pigmentation'], res['criteria']['G1b CBM-B pigment network']
    sup = [res['criteria'][k] for k in ['G1c darkness for pigmentation', 'G1d ISIC-2018 pigment network']]
    if not (a.get('available') and b.get('available')):
        decision = 'NOT EVALUATED (CBM-B results missing)'
    elif any(v < G['min_evaluator_auc'] for v in valid.values()):
        decision = (f"INCONCLUSIVE: CBM-B test AUC below {G['min_evaluator_auc']} for "
                    + ', '.join(k for k, v in valid.items() if v < G['min_evaluator_auc']) +
                    '. A failure to replicate would not be informative.')
    elif a['pass'] and b['pass']:
        av = [s for s in sup if s.get('available')]
        if all(s['pass'] for s in av):
            decision = 'PASS' + ('' if len(av) == 2 else ' (support check G1d not available)')
        else:
            decision = 'PARTIAL: CBM-B replicates, but a non-CBM support check failed'
    else:
        decision = 'FAIL: CBM-B does not replicate ' + ' and '.join(
            n for n, c in [('pigmentation', a), ('pigment network', b)] if not c['pass'])
    if cfg.get('dry_run'):
        decision = 'DRY RUN ONLY (tiny sizes, 1 training epoch): not a decision. Would read: ' + decision
    res['decision'] = decision
    save_json(res, os.path.join(out.root, 'results', 'gate1.json')); R['gate1'] = res
    block = [f"# ClinDiffXAI Phase 1 report (with Phase 1b additions)\n\n**Gate 1 decision: {decision}**\n\nRule (fixed before the run): {G['text']}\n"]
    for k, v in res['criteria'].items():
        if v.get('available'):
            block.append(f"- {k}: {'pass' if v['pass'] else 'fail'} (p_BH={fp(v['p_bh'])}, median slope {v['median_slope']:+.3f}, "
                         f"delta {v['delta_sd']:+.2f} SD, {v['frac_pos'] * 100:.0f}% rising)")
        else:
            block.append(f"- {k}: not available")
    if valid:
        block.append('- CBM-B validity (test AUC, 128 pipeline): ' + ', '.join(f"{k} {v:.3f}" for k, v in valid.items()))
    REPORT.insert(0, '\n'.join(block) + '\n')


@section('Guidance-scale sweep')
def s_cfg(cfg, out, R, D):
    if 'cfg_sweep' not in D or not have(D, 'real_mel', 'real_ben'): return
    rec = D['cfg_sweep']; m = rec['meta']; cm, cb = np.array(MEL_VEC_LIST), np.array(BEN_VEC_LIST)
    A = ev_out(rec, 'cbma'); Bb = ev_out(rec, 'cbmb'); I = rec['A'].get('incep')
    Ir = {c: D[f'real_{c}']['A'].get('incep') for c in ['mel', 'ben']}
    rows = []; scales = sorted(m['scale'].unique())
    for s in scales:
        sm = (m['scale'].values == s) & (m['cls'].values == 'mel'); sb = (m['scale'].values == s) & (m['cls'].values == 'ben')
        r = {'s': s, 'CA mel': ca_scores(A[sm], cm).mean(), 'CA ben': ca_scores(A[sb], cb).mean(),
             'AUC CBM-A (mean over concepts)': np.nanmean([auc_fast(A[sm][:, k], A[sb][:, k]) for k in range(7)])}
        if Bb is not None:
            r['AUC CBM-B (mean over concepts)'] = np.nanmean([auc_fast(Bb[sm][:, k], Bb[sb][:, k]) for k in range(7)])
        if I is not None:
            for cls, sel in [('mel', sm), ('ben', sb)]:
                f = I[sel] / (np.linalg.norm(I[sel], axis=1, keepdims=True) + 1e-9)
                r[f'Diversity {cls} (1 - mean cos)'] = 1 - (f @ f.T)[np.triu_indices(len(f), 1)].mean()
                if Ir[cls] is not None:
                    r[f'KID {cls} x1e3'] = kid(I[sel], Ir[cls], cfg['analysis']['kid_subset_small'], cfg['analysis']['kid_n_sub'])[0] * 1e3
        rows.append(r)
    df = pd.DataFrame(rows); R['cfg'] = df.to_dict(orient='list')
    out.table(df.round(4), 'T4_guidance_scale_sweep', 'Effect of the classifier-free guidance scale s (same noises at every s). '
              'CA: concept agreement (CBM-A units). AUC: mean per-concept separation of melanoma- vs benign-conditioned images. '
              'Diversity: 1 - mean pairwise cosine similarity of Inception features within a class.')
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.3))
    axes[0].plot(df['s'], df['CA mel'], marker='o', color='#2a78d6', label='Melanoma prototype')
    axes[0].plot(df['s'], df['CA ben'], marker='s', color='#eb6834', label='Benign prototype')
    axes[0].set_ylabel('Concept agreement (CBM-A)'); axes[0].legend()
    axes[1].plot(df['s'], df['AUC CBM-A (mean over concepts)'], marker='o', color=C_EV['cbma'], label='CBM-A')
    if 'AUC CBM-B (mean over concepts)' in df:
        axes[1].plot(df['s'], df['AUC CBM-B (mean over concepts)'], marker='s', color=C_EV['cbmb'], label='CBM-B')
    axes[1].set_ylabel('Mean per-concept AUC'); axes[1].legend()
    if 'Diversity mel (1 - mean cos)' in df:
        axes[2].plot(df['s'], df['Diversity mel (1 - mean cos)'], marker='o', color='#2a78d6', label='Melanoma')
        axes[2].plot(df['s'], df['Diversity ben (1 - mean cos)'], marker='s', color='#eb6834', label='Benign')
        axes[2].set_ylabel('Diversity (Inception)'); axes[2].legend()
    for ax in axes:
        ax.axvline(cfg['sizes']['cfg_scale'], color=INK2, lw=0.8, ls=':'); ax.set_xlabel('Guidance scale s')
    fig.tight_layout(); out.fig(fig, 'fig_guidance_scale_sweep')
    REPORT.append("\n## 7. Guidance-scale sweep\n\n" + df.round(3).to_string(index=False) + "\n")


@section('Memorization screen (nearest real training melanoma)')
def s_memo(cfg, out, R, D):
    if not have(D, 'real_mel', 'eval_main_mel', 'eval_base_mel') or 'incep' not in D['real_mel']['A']: return
    nrm = lambda X: X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    Rm = nrm(D['real_mel']['A']['incep']); S_rr = Rm @ Rm.T; np.fill_diagonal(S_rr, -np.inf)
    loo = S_rr.max(1); thr99 = np.percentile(loo, 99); thrmax = loo.max()
    rows, nn = [], {}
    cands = [('Real melanoma (leave-one-out)', None), ('ClinDiffXAI', 'eval_main_mel'), ('Baseline', 'eval_base_mel')]
    if 'eval_realc_mel' in D and 'incep' in D['eval_realc_mel']['A']: cands.append((RC_LAB, 'eval_realc_mel'))
    for lab, key in cands:
        if key is None: s = loo
        else:
            S = nrm(D[key]['A']['incep']) @ Rm.T; s = S.max(1); nn[key] = (s, S.argmax(1))
        rows.append({'Images': lab, 'n': len(s), 'median NN cosine': f"{np.median(s):.4f}", '95th pct': f"{np.percentile(s, 95):.4f}",
                     'max': f"{s.max():.4f}", '% above real 99th pct': f"{np.mean(s > thr99) * 100:.1f}" if key else '1.0 (by definition)',
                     '% above real max': f"{np.mean(s > thrmax) * 100:.1f}" if key else ''})
    out.table(pd.DataFrame(rows), 'S_T_memorization_screen', 'Nearest-neighbour cosine similarity (Inception features) between '
              'each melanoma-conditioned image and the 584 real training melanomas. Reference: each real melanoma to its nearest other '
              'real melanoma. A high share above the real reference suggests copying; inspect the image pairs in the figure.')
    R['memo'] = {r['Images']: r for r in rows}
    fig = plt.figure(figsize=(7.2, 3.2)); gs = fig.add_gridspec(2, 7, height_ratios=[1, 1])
    ax = fig.add_subplot(gs[:, :3])
    bins = np.linspace(min(loo.min(), min(v[0].min() for v in nn.values())), 1, 40)
    ax.hist(loo, bins=bins, density=True, histtype='step', lw=1.5, color=C_MODEL['real'], label='Real (leave-one-out)')
    ax.hist(nn['eval_main_mel'][0], bins=bins, density=True, histtype='step', lw=1.5, color=C_MODEL['main'], label='ClinDiffXAI')
    ax.hist(nn['eval_base_mel'][0], bins=bins, density=True, histtype='step', lw=1.5, color=C_MODEL['base'], label='Baseline')
    if 'eval_realc_mel' in nn:
        ax.hist(nn['eval_realc_mel'][0], bins=bins, density=True, histtype='step', lw=1.5, color=C_MODEL['realc'], label=RC_LAB)
    ax.axvline(thr99, color=INK2, ls=':', lw=0.8); ax.set_xlabel('Cosine similarity to nearest real melanoma'); ax.set_ylabel('Density'); ax.legend()
    G = imgs_of(D['eval_main_mel']); RI = imgs_of(D['real_mel'])
    s, idx = nn['eval_main_mel']; top = np.argsort(-s)[:4]
    for c, t in enumerate(top):
        a1 = fig.add_subplot(gs[0, 3 + c]); a1.imshow(G[t].permute(1, 2, 0).numpy()); a1.axis('off'); a1.set_title(f"gen, {s[t]:.3f}", fontsize=6)
        a2 = fig.add_subplot(gs[1, 3 + c]); a2.imshow(RI[idx[t]].permute(1, 2, 0).numpy()); a2.axis('off'); a2.set_title('nearest real', fontsize=6)
    out.fig(fig, 'S_fig_memorization_screen')


@section('Qualitative figures')
def s_qual(cfg, out, R, D):
    if have(D, 'eval_main_mel', 'eval_main_ben', 'eval_base_mel', 'eval_base_ben'):
        k = min(8, D['eval_main_mel']['n'])
        rows = [('ClinDiffXAI\nmelanoma', 'eval_main_mel'), ('Baseline\nmelanoma', 'eval_base_mel'),
                ('ClinDiffXAI\nbenign', 'eval_main_ben'), ('Baseline\nbenign', 'eval_base_ben')]
        fig, axes = plt.subplots(4, k, figsize=(0.9 * k + 0.9, 4.0))
        for r, (lab, key) in enumerate(rows):
            X = imgs_of(D[key])
            for c in range(k):
                ax = axes[r, c]; ax.imshow(X[c].permute(1, 2, 0).numpy()); ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
                for sp in ax.spines.values(): sp.set_visible(False)
            axes[r, 0].set_ylabel(lab, fontsize=6.5)
        fig.suptitle('Same column = same starting noise for both models', fontsize=7)
        fig.tight_layout(); out.fig(fig, 'fig_qualitative_paired')
    if 'sweep_mel' in D:
        rec = D['sweep_mel']; X = imgs_of(rec); m = rec['meta']
        levels = np.sort(m['level'].unique()); nz = min(3, m['noise_id'].nunique())
        for cname in ['pigmentation', 'pigment_network', 'blue_whitish_veil']:
            j = CONCEPT_NAMES.index(cname)
            fig, axes = plt.subplots(nz, len(levels), figsize=(0.62 * len(levels) + 0.5, 0.66 * nz + 0.5), squeeze=False)
            for r in range(nz):
                for c, l in enumerate(levels):
                    i = np.where((m['concept_idx'].values == j) & (m['level'].values == l) & (m['noise_id'].values == r))[0][0]
                    ax = axes[r, c]; ax.imshow(X[i].permute(1, 2, 0).numpy()); ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
                    for sp in ax.spines.values(): sp.set_visible(False)
                    if r == nz - 1: ax.set_xlabel(f"{l:.1f}", fontsize=6)
            fig.suptitle(f"{cname.replace('_', ' ')}: commanded level 0 to 1, other concepts at the melanoma prototype", fontsize=7)
            fig.tight_layout(); out.fig(fig, f"fig_sweep_strip_{cname}" if cname != 'blue_whitish_veil' else f"S_fig_sweep_strip_{cname}")


@section('Colour fidelity (CIELAB means, Phase 1b)')
def s_colour(cfg, out, R, D):
    from common import lab_means
    if not have(D, 'real_mel', 'real_ben'): return
    groups = [('Real', 'real'), ('ClinDiffXAI', 'eval_main'), ('Baseline', 'eval_base'), (RC_LAB, 'eval_realc')]
    L = {}
    for g, pre in groups:
        for cls in ['mel', 'ben']:
            if f'{pre}_{cls}' in D:
                L[(g, cls)] = lab_means(imgs_of(D[f'{pre}_{cls}']).numpy())
    Rl = np.concatenate([L[('Real', 'mel')], L[('Real', 'ben')]])
    mu = Rl.mean(0); icov = np.linalg.inv(np.cov(Rl, rowvar=False))
    md = lambda X: np.sqrt(np.einsum('ij,jk,ik->i', X - mu, icov, X - mu))
    thr = float(np.percentile(md(Rl), 99))
    rows, R['colour'] = [], {}
    for (g, cls), X in L.items():
        ref = L[('Real', cls)]; o = float(np.mean(md(X) > thr) * 100)
        rows.append({'Images': g, 'Class': cls, 'n': len(X), 'L* mean': f"{X[:, 0].mean():.1f}", 'a* mean': f"{X[:, 1].mean():.1f}",
                     'b* mean': f"{X[:, 2].mean():.1f}", 'Chroma mean': f"{np.hypot(X[:, 1], X[:, 2]).mean():.1f}",
                     'W1 L*': f"{stats.wasserstein_distance(X[:, 0], ref[:, 0]):.2f}",
                     'W1 a*': f"{stats.wasserstein_distance(X[:, 1], ref[:, 1]):.2f}",
                     'W1 b*': f"{stats.wasserstein_distance(X[:, 2], ref[:, 2]):.2f}", 'Colour outliers %': f"{o:.1f}"})
        R['colour'][f"{g}|{cls}"] = o
    out.table(pd.DataFrame(rows), 'T5_colour_fidelity', 'Colour fidelity. Per-image mean CIELAB (D65) over the whole image. '
              'W1: Wasserstein-1 distance to real images of the same class. Colour outlier: Mahalanobis distance (L*, a*, b*) '
              'above the 99th percentile of real images (so about 1% of real images by construction).')
    cfg_rows = []
    if 'cfg_sweep' in D:
        rec = D['cfg_sweep']; Lc = lab_means(imgs_of(rec).numpy()); m = rec['meta']
        for sc_ in sorted(m['scale'].unique()):
            for cls in ['mel', 'ben']:
                sel = (m['scale'].values == sc_) & (m['cls'].values == cls)
                cfg_rows.append({'s': sc_, 'Class': cls, 'Colour outliers %': float(np.mean(md(Lc[sel]) > thr) * 100),
                                 'Chroma mean': float(np.hypot(Lc[sel][:, 1], Lc[sel][:, 2]).mean())})
        out.table(pd.DataFrame(cfg_rows).round(2), 'S_T_colour_vs_guidance', 'Colour outlier rate of ClinDiffXAI images at each guidance scale.')
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5))
    for ax, cls in zip(axes[:2], ['mel', 'ben']):
        for g, col, msz in [('Real', C_MODEL['real'], 3), ('Baseline', C_MODEL['base'], 2), ('ClinDiffXAI', C_MODEL['main'], 2), (RC_LAB, C_MODEL['realc'], 2)]:
            if (g, cls) in L:
                X = L[(g, cls)]; ax.scatter(X[:, 1], X[:, 2], s=msz, color=col, alpha=0.35, lw=0, label=g, rasterized=True)
        ax.set_xlabel('mean a* (green to red)'); ax.set_ylabel('mean b* (blue to yellow)')
        ax.set_title('Melanoma' if cls == 'mel' else 'Benign')
    axes[0].legend(markerscale=4, fontsize=6, loc='best')
    if cfg_rows:
        df = pd.DataFrame(cfg_rows)
        for cls, col, mk in [('mel', '#2a78d6', 'o'), ('ben', '#eb6834', 's')]:
            d = df[df['Class'] == cls]; axes[2].plot(d['s'], d['Colour outliers %'], marker=mk, color=col, label='Melanoma' if cls == 'mel' else 'Benign')
        axes[2].set_xlabel('Guidance scale s'); axes[2].set_ylabel('Colour outliers (%)'); axes[2].legend()
        axes[2].axhline(1, color=INK2, lw=0.8, ls=':')
    else:
        axes[2].axis('off')
    fig.tight_layout(); out.fig(fig, 'fig_colour_fidelity')
    REPORT.append("\n## 9. Colour fidelity (Phase 1b)\n\n- Colour outliers (% of images outside the 99% region of real images): " +
                  ', '.join(f"{k} {v:.1f}%" for k, v in R['colour'].items()))


@section('Concept agreement with real concept vectors (Phase 1b, exploratory)')
def s_realc_agree(cfg, out, R, D):
    if not have(D, 'eval_realc_mel', 'eval_realc_ben', 'real_mel', 'real_ben'): return
    B = cfg['analysis']['n_boot']; ccols = [f'concept_{c}' for c in CONCEPT_NAMES]; rows, rho_rows = [], []
    for cls in ['mel', 'ben']:
        g, r = D[f'eval_realc_{cls}'], D[f'real_{cls}']
        Cg = g['meta'][ccols].values; Cr = r['meta'][ccols].values
        Pg, Pr = ev_out(g, 'cbma'), ev_out(r, 'cbma')
        perm = RNG.permutation(len(Cg))
        for lab, v in [('Real images: native pseudo-label vs 128-pipeline CBM-A', 1 - np.abs(Cr - Pr).mean(1)),
                       (f'{RC_LAB}: commanded vector vs CBM-A', 1 - np.abs(Cg - Pg).mean(1)),
                       ('Shuffled pairing (chance for this metric)', 1 - np.abs(Cg[perm] - Pg).mean(1))]:
            rows.append({'Class': cls, 'Pairing': lab, 'n': len(v), 'Agreement [95% CI]': fci(*boot_mean(v, B))})
        Bg, Br = ev_out(g, 'cbmb'), ev_out(r, 'cbmb')
        for k, c in enumerate(CONCEPT_NAMES):
            rr = {'Class': cls, 'Concept': c,
                  'Real: rho(native, CBM-A 128)': stats.spearmanr(Cr[:, k], Pr[:, k])[0],
                  'Generated: rho(commanded, CBM-A)': stats.spearmanr(Cg[:, k], Pg[:, k])[0]}
            if Bg is not None:
                rr['Real: rho(native, CBM-B 128)'] = stats.spearmanr(Cr[:, k], Br[:, k])[0]
                rr['Generated: rho(commanded, CBM-B)'] = stats.spearmanr(Cg[:, k], Bg[:, k])[0]
            rho_rows.append(rr)
    out.table(pd.DataFrame(rows), 'T6_realistic_vector_agreement', 'Exploratory analysis added after Phase 1. The main model is '
              'conditioned on the pseudo-label vectors of real training images (same noises as the main evaluation). Agreement = '
              '1 - mean_k |c_k - c_hat_k| between each image\'s own vector and its CBM-A output. The real-image row is the same '
              'quantity for real images (native-resolution pseudo-label vs 128-pipeline prediction).')
    out.table(pd.DataFrame(rho_rows).round(3), 'T6b_realistic_vector_tracking', 'Across-image Spearman correlation between the '
              'commanded concept value and the evaluator output (generated), and between the native pseudo-label and the evaluator '
              'output at the 128 pipeline (real). CBM-B columns are independent of the pseudo-labelling model.')
    d = pd.DataFrame(rho_rows); R['realc_tracking'] = d.to_dict(orient='list')
    REPORT.append("\n## 10. Real concept vectors (exploratory, Phase 1b)\n")
    for cls in ['mel', 'ben']:
        dd = d[d['Class'] == cls]
        txt = f"- {cls}: mean rho(commanded, CBM-A) {dd['Generated: rho(commanded, CBM-A)'].mean():.3f} vs real {dd['Real: rho(native, CBM-A 128)'].mean():.3f}"
        if 'Generated: rho(commanded, CBM-B)' in dd:
            txt += f"; with CBM-B {dd['Generated: rho(commanded, CBM-B)'].mean():.3f} vs real {dd['Real: rho(native, CBM-B 128)'].mean():.3f}"
        REPORT.append(txt)


@section('Cross-talk versus real-image co-occurrence (Phase 1b)')
def s_cooccur(cfg, out, R, D):
    if not have(D, 'real_mel', 'real_ben') or 'crosstalk' not in R: return
    off = ~np.eye(7, dtype=bool); rows, mats = [], {}
    for ev in ['cbma', 'cbmb']:
        key = f'mel|{ev}'
        if key not in R['crosstalk'] or ev_out(D['real_mel'], ev) is None: continue
        MR = np.array(R['crosstalk'][key])
        Y = np.concatenate([ev_out(D['real_mel'], ev), ev_out(D['real_ben'], ev)])
        Cp = stats.spearmanr(Y).correlation; Cm = stats.spearmanr(ev_out(D['real_mel'], ev)).correlation
        mats[ev] = (MR, Cp)
        rows.append({'Evaluator': EV_LABEL[ev],
                     'rho(cross-talk, real co-occurrence, all real)': stats.spearmanr(MR[off], Cp[off])[0],
                     'rho(cross-talk, real co-occurrence, real melanoma)': stats.spearmanr(MR[off], Cm[off])[0],
                     'mean |off-diagonal| cross-talk': np.abs(MR[off]).mean(),
                     'mean |off-diagonal| real co-occurrence': np.abs(Cp[off]).mean()})
    dz = os.path.join(cfg['work'], 'results', 'derm7pt_test_preds.npz')
    if os.path.exists(dz):
        z = np.load(dz); Cy = stats.spearmanr(z['y']).correlation
        for lab, k in [('CBM-A', 'a_bneck'), ('CBM-B', 'b_bneck')]:
            Co = stats.spearmanr(z[k]).correlation
            rows.append({'Evaluator': f'{lab} on Derm7pt test (expert labels)',
                         'mean |off-diagonal| evaluator outputs': np.abs(Co[off]).mean(),
                         'mean |off-diagonal| expert labels': np.abs(Cy[off]).mean(),
                         'rho(output correlations, label correlations)': stats.spearmanr(Co[off], Cy[off])[0]})
    df = pd.DataFrame(rows); R['cooccur'] = df.to_dict(orient='list')
    out.table(df.round(3), 'T7_crosstalk_vs_cooccurrence', 'Does the cross-talk of the dose-response (melanoma base) follow how '
              'concepts co-occur in real images? Compared over the 42 off-diagonal cells. Derm7pt rows: whether an evaluator '
              'couples concepts more strongly than the expert labels do (a coupling evaluator creates apparent cross-talk). '
              'Descriptive: the cells are not independent, so no p-values are given.')
    if 'cbmb' in mats:
        MR, Cp = mats['cbmb']
        fig, axes = plt.subplots(1, 3, figsize=(7.4, 3.0), gridspec_kw={'width_ratios': [1, 1, 1.25], 'wspace': 0.4})
        for ax, M, t in [(axes[0], Cp, 'Real co-occurrence (CBM-B)'), (axes[1], MR, 'Cross-talk (CBM-B)')]:
            im = ax.imshow(M, cmap=DIV, vmin=-1, vmax=1); ax.grid(False)
            ax.set_xticks(range(7)); ax.set_xticklabels(SHORT, fontsize=6); ax.set_yticks(range(7)); ax.set_yticklabels(SHORT, fontsize=6); ax.set_title(t)
        for ev, mk in [('cbma', 'o'), ('cbmb', 's')]:
            if ev in mats:
                M1, C1 = mats[ev]; axes[2].scatter(C1[off], M1[off], s=10, marker=mk, color=C_EV[ev], label=EV_LABEL[ev])
        axes[2].axhline(0, color=INK2, lw=0.6); axes[2].axvline(0, color=INK2, lw=0.6)
        axes[2].set_xlabel('Real co-occurrence (Spearman)'); axes[2].set_ylabel('Cross-talk (mean ρ)'); axes[2].legend(fontsize=6)
        cb = fig.colorbar(im, ax=list(axes[:2]), orientation='horizontal', fraction=0.05, pad=0.16, aspect=40)
        cb.ax.tick_params(labelsize=6); cb.set_label('Spearman ρ', fontsize=7); out.fig(fig, 'fig_crosstalk_vs_cooccurrence')
    REPORT.append("\n## 11. Cross-talk versus real co-occurrence (Phase 1b)\n\n" + df.round(3).to_string(index=False) + "\n")


@section('Qualitative: prototype versus real concept vectors (Phase 1b)')
def s_qual_realc(cfg, out, R, D):
    need = ['eval_main_mel', 'eval_realc_mel', 'eval_base_mel', 'eval_main_ben', 'eval_realc_ben', 'eval_base_ben']
    if not have(D, *need): return
    k = min(8, D['eval_main_mel']['n'])
    labs = ['ClinDiffXAI\nprototype', 'ClinDiffXAI\nreal vectors', 'Baseline'] * 2
    fig, axes = plt.subplots(6, k, figsize=(0.9 * k + 1.0, 5.8))
    for r, (key, lab) in enumerate(zip(need, labs)):
        X = imgs_of(D[key])
        for c in range(k):
            ax = axes[r, c]; ax.imshow(X[c].permute(1, 2, 0).numpy()); ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
            for sp in ax.spines.values(): sp.set_visible(False)
        axes[r, 0].set_ylabel(lab + ('\n(melanoma)' if r < 3 else '\n(benign)'), fontsize=6)
    fig.suptitle('Same column = same starting noise', fontsize=7); fig.tight_layout(); out.fig(fig, 'fig_qualitative_realistic_vectors')


def main():
    cfg = load_cfg(sys.argv[1]); out = Out(cfg['outputs']); t0 = time.time()
    log('analysis: loading sets and scores'); D = load_all(cfg['work'])
    log('  sets: ' + ', '.join(f"{k}({v['n']})" for k, v in D.items()))
    R = {'checkpoints': {k: v['info'].get('ckpt') for k, v in D.items() if 'ckpt' in v['info']}}
    s_evaluators(cfg, out, R); s_repro(cfg, out, R, D); s_agreement(cfg, out, R, D)
    s_discrim(cfg, out, R, D); s_dose(cfg, out, R, D); s_gate(cfg, out, R)
    s_cfg(cfg, out, R, D); s_distribution(cfg, out, R, D); s_memo(cfg, out, R, D); s_qual(cfg, out, R, D)
    s_colour(cfg, out, R, D); s_realc_agree(cfg, out, R, D); s_cooccur(cfg, out, R, D); s_qual_realc(cfg, out, R, D)
    ck = {}
    for k, v in R['checkpoints'].items():
        if v: ck[k] = f"{os.path.basename(v['path'])} epoch {v['epoch_1based']} ({v['weights']})"
    REPORT.append("\n## Checkpoints used for the new evaluation\n\n" + '\n'.join(f"- {k}: {v}" for k, v in ck.items()))
    if WARN: REPORT.append("\n## Warnings\n\n" + '\n'.join(f"- {w}" for w in WARN))
    if ERRORS: REPORT.append("\n## Errors\n\n" + '\n'.join(f"- {e}" for e in ERRORS))
    REPORT.append(f"\n_Analysis time {(time.time() - t0) / 60:.1f} min._\n")
    with open(os.path.join(out.root, 'phase1_report.md'), 'w') as f:
        f.write('\n'.join(REPORT))
    save_json(R, os.path.join(out.root, 'results', 'phase1_results.json'))
    log('analysis finished'); print('\n'.join(REPORT))


if __name__ == '__main__':
    main()
''',
}
for fn, src in FILES.items():
    with open(os.path.join(CODE, fn), 'w') as f:
        f.write(src)
say(f'wrote {len(FILES)} code files to {CODE}')

# ---------------------------------------------------------------- 2b. reuse Phase 1
# stop worker processes left over from an interrupted run of this cell (they would hold the GPUs)
subprocess.run(['pkill', '-f', os.path.join(CODE, 'worker.py')], check=False)
REUSED = None
if not DRY_RUN or os.environ.get('P1_FORCE_REUSE'):   # the env var is for offline testing only
    cands = [os.path.dirname(p) for p in NAME_INDEX.get('cfg.json', [])]
    cands.append(os.path.join(WORK_ROOT, 'phase1'))   # Phase 1 run earlier in this same session
    for d in cands:
        if os.path.basename(d) == 'phase1' and len(glob.glob(os.path.join(d, 'sets', '*.pt'))) >= 14:
            REUSED = d; break
    if REUSED:
        n_sets = len(glob.glob(os.path.join(REUSED, 'sets', '*.pt')))
        say(f'reusing Phase 1 output at {REUSED} ({n_sets} image sets); copying ...')
        for sub in ['sets', 'scores', 'evaluators', 'results', 'done']:
            src = os.path.join(REUSED, sub)
            if os.path.isdir(src):
                shutil.copytree(src, os.path.join(WORK, sub), dirs_exist_ok=True)
        if not os.path.exists(os.path.join(WORK, 'evaluators', 'isic18_detector.pth')):
            for j in ['train_isic18']:
                p = os.path.join(WORK, 'done', f'{j}.ok')
                if os.path.exists(p): os.remove(p)
        for j in ['score_A', 'score_B']:   # incremental: they only add what is missing
            p = os.path.join(WORK, 'done', f'{j}.ok')
            if os.path.exists(p): os.remove(p)
    else:
        print('  NOTE: no Phase 1 output found (attached or in /kaggle/working/phase1). All images will be generated again (same seeds).')

# ---------------------------------------------------------------- 3. config
full = dict(n_eval=1000, sweep_n=32, sweep_levels=11, cfg_n=128, cfg_scales=[0.0, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.5, 6.0],
            ddim_steps=50, cfg_scale=3.5, gen_bs=32, repro_n_eval=200, repro_n_pearson=20, repro_levels=10,
            repro_sweep_steps=30, n_real_ben=2000, n_real_mel_max=None)
dry = dict(full, n_eval=24, sweep_n=6, sweep_levels=3, cfg_n=8, cfg_scales=[0.0, 3.5], ddim_steps=4, repro_n_eval=16,
           repro_n_pearson=4, repro_levels=3, repro_sweep_steps=3, n_real_ben=48, n_real_mel_max=48)
ncpu = os.cpu_count() or 4
cfg = {
    'work': WORK, 'outputs': OUTD, 'paths': paths, 'image_size': IMAGE_SIZE, 'eval_size': EVAL_SIZE,
    'incep_size': INCEP_SIZE, 'pretrained': PRETRAINED, 'cpu_workers': max(1, ncpu // 2), 'cpu_workers_io': max(2, ncpu),
    'dry_run': DRY_RUN, 'reused_phase1': REUSED,
    'sizes': dry if DRY_RUN else full,
    'seeds': {'eval_mel': 1001, 'eval_ben': 1002, 'sweep': 2001, 'cfg': 3001, 'realc_mel': 4001, 'realc_ben': 4002},
    'train': {'epochs': 1 if DRY_RUN else 40, 'patience': 10, 'batch': 16 if DRY_RUN else 32, 'lr': 1e-4},
    'analysis': {'n_boot': 200 if DRY_RUN else 2000, 'n_boot_auc': 100 if DRY_RUN else 1000,
                 'kid_subset': 500, 'kid_subset_small': 100, 'kid_n_sub': 10 if DRY_RUN else 100},
    'gate1': {'alpha': 0.05, 'min_evaluator_auc': 0.70, 'text':
              'PASS if CBM-B shows a positive, significant dose-response (two-sided Wilcoxon signed-rank test on '
              'per-trajectory slopes, BH-corrected p < 0.05, median slope > 0) for BOTH pigmentation and pigment network '
              'from the melanoma prototype, and the available non-CBM checks agree (lesion darkness for pigmentation; '
              'ISIC 2018 detector for pigment network). INCONCLUSIVE if CBM-B test AUC (128 pipeline) < 0.70 for either '
              'concept. PARTIAL if CBM-B passes but a non-CBM check fails. FAIL otherwise.'},
}
CFG = os.path.join(WORK, 'cfg.json')
with open(CFG, 'w') as f:
    json.dump(cfg, f, indent=2)

# ---------------------------------------------------------------- 4. weights + GPUs
if PRETRAINED:
    say('downloading torchvision weights once (ConvNeXt-Tiny, ResNet-50, Inception-v3)')
    import torchvision.models as tvm
    try:
        tvm.convnext_tiny(weights=tvm.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V2)
        tvm.inception_v3(weights=tvm.Inception_V3_Weights.DEFAULT)
    except Exception as e:
        raise RuntimeError(f"Could not download torchvision weights ({e}). Turn Internet ON in the notebook settings.")
try:
    n_gpu = len([l for l in subprocess.run(['nvidia-smi', '-L'], capture_output=True, text=True).stdout.splitlines() if l.startswith('GPU')])
except Exception:
    n_gpu = 0
say(f'GPUs visible: {n_gpu}')
if n_gpu == 0 and not os.environ.get('P1_INPUT_ROOT'):
    raise RuntimeError('No GPU found. Set Accelerator = GPU T4 x2.')

# ---------------------------------------------------------------- 5. job runner
WORKER = os.path.join(CODE, 'worker.py'); STATUS = {}


def run_job(job, gpu):
    if os.path.exists(os.path.join(WORK, 'done', f'{job}.ok')):
        STATUS[job] = 'done earlier'; return
    STATUS[job] = 'running'
    env = os.environ.copy(); env['CUDA_VISIBLE_DEVICES'] = '' if gpu is None else str(gpu)
    env['PYTHONUNBUFFERED'] = '1'
    with open(os.path.join(WORK, 'logs', f'{job}.log'), 'a') as lf:
        rc = subprocess.run([sys.executable, '-u', WORKER, job, CFG], stdout=lf, stderr=subprocess.STDOUT, env=env, cwd=CODE).returncode
    STATUS[job] = 'ok' if rc == 0 else f'FAILED (exit {rc})'


def run_queue(jobs, gpu):
    for j in jobs: run_job(j, gpu)


def last_line(job):
    p = os.path.join(WORK, 'logs', f'{job}.log')
    if not os.path.exists(p): return ''
    with open(p, 'rb') as f:
        f.seek(0, 2); n = f.tell(); f.seek(max(0, n - 4000))
        lines = [l for l in f.read().decode('utf8', 'ignore').splitlines() if l.strip()]
    return lines[-1][-110:] if lines else ''


def run_stage(title, queues):
    say(f'===== {title} =====')
    th = [threading.Thread(target=run_queue, args=(q, g), daemon=True) for q, g in queues]
    for t in th: t.start()
    t0 = time.time(); tick = 0
    while any(t.is_alive() for t in th):
        time.sleep(5); tick += 5
        if tick % 60 == 0:
            print(f"--- {title}: {(time.time() - t0) / 60:.1f} min elapsed", flush=True)
            for q, _ in queues:
                for j in q:
                    st = STATUS.get(j, 'waiting')
                    if st == 'running': print(f"    {j:<14} running | {last_line(j)}", flush=True)
    for q, _ in queues:
        for j in q: print(f"    {j:<14} {STATUS.get(j, '?')}", flush=True)


# estimated minutes on one T4 (from the Phase 1 run); used only to balance the two GPUs
EST = {'repro': 28, 'gen_eval_main': 33, 'gen_eval_base': 33, 'gen_sweep_mel': 35, 'gen_sweep_ben': 35,
       'gen_cfg': 33, 'gen_eval_realc': 33, 'train_cbmb': 3, 'train_isic18': 30}
pending = [j for j in EST if not os.path.exists(os.path.join(WORK, 'done', f'{j}.ok'))]
pending.sort(key=lambda j: -EST[j])
if 'train_cbmb' in pending:   # CBM-B first in its queue, so a crash shows early
    pending.remove('train_cbmb'); pending.insert(0, 'train_cbmb')
G0, G1, load = [], [], [0, 0]
for j in pending:
    q = 0 if load[0] <= load[1] else 1
    (G0 if q == 0 else G1).append(j); load[q] += EST[j]
say(f'pending GPU jobs: GPU0 {G0} (~{load[0]} min) | GPU1 {G1} (~{load[1]} min)')
if n_gpu >= 2:
    stage1 = [(G0, 0), (G1, 1), (['prep_real'], None)]; stage2 = [(['score_A'], 0), (['score_B'], 1)]
elif n_gpu == 1:
    print('  WARNING: only one GPU; the GPU jobs run one after another (slower).')
    stage1 = [(G1 + G0, 0), (['prep_real'], None)]; stage2 = [(['score_B', 'score_A'], 0)]
else:   # offline test only
    stage1 = [(G0, None), (G1, None), (['prep_real'], None)]; stage2 = [(['score_A'], None), (['score_B'], None)]

T0 = time.time()
run_stage('STAGE 1: generation, evaluator training, real anchors', stage1)
run_stage('STAGE 2: scoring every image set', stage2)
say('===== STAGE 3: analysis (CPU) =====')
with open(os.path.join(WORK, 'logs', 'analysis.log'), 'w') as lf:
    rc = subprocess.run([sys.executable, '-u', os.path.join(CODE, 'analysis.py'), CFG], stdout=lf, stderr=subprocess.STDOUT,
                        cwd=CODE, env={**os.environ, 'CUDA_VISIBLE_DEVICES': ''}).returncode
STATUS['analysis'] = 'ok' if rc == 0 else f'FAILED (exit {rc})'

# ---------------------------------------------------------------- 6. package
for fn in FILES: shutil.copy(os.path.join(CODE, fn), os.path.join(OUTD, 'code_' + fn))
shutil.copy(CFG, os.path.join(OUTD, 'cfg.json'))
os.makedirs(os.path.join(OUTD, 'logs'), exist_ok=True)
for lg in glob.glob(os.path.join(WORK, 'logs', '*.log')): shutil.copy(lg, os.path.join(OUTD, 'logs'))
for fn in glob.glob(os.path.join(WORK, 'results', '*')): shutil.copy(fn, os.path.join(OUTD, 'results'))
zip_base = os.path.join(WORK_ROOT, 'phase1b_dryrun_outputs' if DRY_RUN else 'phase1b_outputs')
shutil.make_archive(zip_base, 'zip', OUTD)
print('\n' + '=' * 78)
print(f"PHASE 1b {'DRY RUN ' if DRY_RUN else ''}FINISHED in {(time.time() - T0) / 60:.1f} min"
      f" ({'reused Phase 1 output' if REUSED else 'all images generated in this run'})")
for j, s in STATUS.items(): print(f"  {j:<14} {s}")
rep = os.path.join(OUTD, 'phase1_report.md')
if os.path.exists(rep):
    print('=' * 78); print(open(rep).read())
else:
    print('No report: see', os.path.join(WORK, 'logs', 'analysis.log'))
print(f"\nDownload: {zip_base}.zip  (tables, figures, report, logs, code)")
print(f"Generated images stay in {os.path.join(WORK, 'sets')} for Phase 2.")
