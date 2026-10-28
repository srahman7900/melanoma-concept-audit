# =============================================================================
# ClinDiffXAI  PHASE 3b: PH2 (corrected parser) and HAM10000 overlap checks
#                        (one Kaggle cell, GPU T4 x2; my estimate 15 to 30 minutes)
# =============================================================================
# WHERE TO RUN
#   A NEW notebook session (GPU T4 x2, Internet on). Attach the SAME inputs as
#   Phase 3 (archives, ISIC 2020, pseudo-label CSV, Phase 1b evaluators, HAM10000,
#   PH2). Optional, for the classifiers on PH2: phase2_models.zip and the output of
#   the Phase 3 notebook (Add Input > Notebook output), which holds the new models.
#   Download phase3b_outputs.zip when it finishes.
#
# WHAT THIS CELL DOES (no classifier is trained; rules 5 to 8 are unchanged,
#   SHA-256 287492fb0ca91d0dbaee3f839345472d28fec594ac208c02e6cdf87dad20e318)
#   1. Builds the ISIC 2020 classifier data again and checks it equals Phase 2.
#   2. HAM10000: builds the data again with the Phase 3 order and split, and adds
#      the overlap checks of the protocol frozen on 2026-10-28 (SHA-256 of that
#      file 35eeefd1e0ad476d2486e27d73508b0e67132bcb1fab0b20e84fa674b26518c1):
#      identical 128-pixel images against the ISIC 2020 classifier data, identical
#      images with different lesion ids, lesions in more than one split, and a
#      near-duplicate screen with ImageNet ResNet-50 features (report only). It
#      checks that the split equals the one the Phase 3 HAM10000 classifiers used.
#   3. PH2 (rule 8): CBM-B and the ISIC 2018 detector against the expert criteria,
#      with the parser corrected for the real file layout (the six criteria are
#      separated by single bars); classifiers found in the inputs on PH2.
# =============================================================================
DRY_RUN     = False
FORCE_RERUN = False

import os, sys, json, time, glob, shutil, subprocess, uuid, signal, copy, zipfile, hashlib

WORK_ROOT = os.environ.get('P1_WORK_ROOT', '/kaggle/working')
INPUT_ROOT = os.environ.get('P1_INPUT_ROOT', '/kaggle/input')
RULE_3 = 'Phase 3 rules, fixed before the run. Data: the two realistic bases (64 benign and 64 melanoma base vectors), the four target-dominant knobs (pigment network, vascular structures, pigmentation, regression structures), logits clipped at +/-15, effect = mean logit change from knob level 0 to 1 over trajectories. Scale-free effect = effect divided by the SD of the classifier logits on the 969 real test images, mean of the two bases per classifier. Rule 5 (trained null): null classifiers are the Phase 2 permuted-label ResNet-50 and 17 new permuted-label classifiers (ResNet-50 seeds 2 to 10, EfficientNet-B0 seeds 1 to 5, ViT-B/16 seeds 1 to 3), trained with the Phase 2 recipe. For each knob, a melanoma finding is above null when the mean scale-free effect of the six primary melanoma classifiers is above the 95th percentile of the null classifiers. Reported for all four knobs; the same comparison is reported for the observational coefficients. Rule 6 (retraining): nine new melanoma classifiers (ResNet-50, EfficientNet-B0 and ViT-B/16, seeds 3 to 5) are added to the six primary ones. The pigmentation-sweep finding is stable if (a) its bootstrap 95% CI is above 0 on both realistic bases for at least 12 of the 15 classifiers and (b) pigmentation has the largest mean effect of the four knobs averaged over the 15 classifiers. Rule 7 (second training dataset): six melanoma classifiers trained on HAM10000 (melanoma against all other diagnoses; split by lesion; images whose names occur in ISIC 2020 removed; ResNet-50, EfficientNet-B0 and ViT-B/16, seeds 1 and 2; Phase 2 recipe). The pigmentation-sweep finding transfers if its bootstrap 95% CI is above 0 on both realistic bases in at least 5 of the 6 and pigmentation has the largest mean effect of the four knobs averaged over the 6. Rule 8 (independent expert criteria, descriptive): CBM-B and the ISIC 2018 detector are compared with the PH2 expert criteria (pigment network atypical against typical; dots and globules; streaks; regression areas; blue-whitish veil) by AUC with a 95% bootstrap CI; a criterion is confirmed on PH2 when the lower bound is above 0.5. If PH2 is not attached, rule 8 is not run. Descriptive only: the false-positive rate of the per-classifier Wilcoxon test in the null classifiers; 100 random linear heads per architecture on ImageNet features; off-target preservation filters for all melanoma classifiers. Earlier verdicts are not changed.'
RULE_3_SHA = '287492fb0ca91d0dbaee3f839345472d28fec594ac208c02e6cdf87dad20e318'
assert hashlib.sha256(RULE_3.encode()).hexdigest() == RULE_3_SHA, 'the rule text was changed after it was fixed'


def say(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def walk(root, match, max_depth=7):
    hits = []
    if not os.path.isdir(root): return hits
    for dp, dn, fn in os.walk(root, followlinks=True):
        if dp.count(os.sep) - root.count(os.sep) >= max_depth: dn[:] = []
        if len(fn) > 3000: dn[:] = []; continue
        for f in fn:
            if match(dp, f): hits.append(os.path.join(dp, f))
    return hits


WORK = os.path.join(WORK_ROOT, 'phase3b_dryrun' if DRY_RUN else 'phase3b')
WA, W1D = os.path.join(WORK, 'audit'), os.path.join(WORK, 'phase1d')
CODE = os.path.join(WORK, 'code'); UNZ = os.path.join(WORK, 'inputs_unzipped')
os.makedirs(WORK, exist_ok=True)
RUN_ID = uuid.uuid4().hex
for _ in range(2):   # stop worker processes of an interrupted earlier run (they would hold the GPUs)
    for w in ['worker2.py', 'worker2d.py', 'worker3.py', 'analysis3.py']:   # noqa
        try:
            if subprocess.run(['pkill', '-f', os.path.join(CODE, w)], check=False).returncode == 0:
                say('stopped worker processes of an earlier run of this cell')
        except FileNotFoundError:
            pass
    time.sleep(2)
if FORCE_RERUN:
    for d in [WA, W1D, os.path.join(WORK, 'logs'), os.path.join(WORK, 'package')]:
        shutil.rmtree(d, ignore_errors=True)
for d in [CODE, UNZ, WA, W1D, os.path.join(WORK, 'logs'), os.path.join(WA, 'done'), os.path.join(WA, 'results'),
          os.path.join(WA, 'models')]:
    os.makedirs(d, exist_ok=True)
with open(os.path.join(WA, 'run_id.txt'), 'w') as f:
    f.write(RUN_ID)

# ---------------------------------------------------------------- 0. archives
NEED = {'gen': lambda d, f: f == 'rsweep_ben.npz' and os.path.exists(os.path.join(d, 'rsweep_ben.json')),   # not the figure subset
        'ainp': lambda d, f: f == 'B_rsweep_ben.npz' and os.path.basename(d) == 'scores'}
ZIPS = {'gen': 'clindiffxai_generated_images.zip', 'ainp': 'phase2_analysis_inputs.zip'}
roots = [INPUT_ROOT, UNZ]
found = {}
for k, fn in NEED.items():
    h = [p for r in roots for p in walk(r, fn)]
    if not h:
        z = [p for p in walk(INPUT_ROOT, lambda d, f: f == ZIPS[k])]
        if z:
            dst = os.path.join(UNZ, k); say(f'unzipping {os.path.basename(z[0])} ...')
            with zipfile.ZipFile(z[0]) as zz: zz.extractall(dst)
            h = walk(dst, fn)
    if not h:
        raise FileNotFoundError(f'the contents of {ZIPS[k]} were not found: attach the dataset with this archive')
    found[k] = h[0]
GEN = os.path.dirname(found['gen'])
AINP = os.path.dirname(os.path.dirname(os.path.dirname(found['ainp'])))   # .../phase2/scores/B_rsweep_ben.npz
say(f'archives: images {GEN}; analysis inputs {AINP}')
m2 = [p for r in roots for p in walk(r, lambda d, f: f == 'mel_resnet50_s1.pt')]
if not m2:
    z = walk(INPUT_ROOT, lambda d, f: f == 'phase2_models.zip')
    if z:
        say('unzipping phase2_models.zip ...')
        with zipfile.ZipFile(z[0]) as zz: zz.extractall(os.path.join(UNZ, 'm2'))
        m2 = walk(os.path.join(UNZ, 'm2'), lambda d, f: f == 'mel_resnet50_s1.pt')
M2 = os.path.dirname(m2[0]) if m2 else None
say(f"Phase 2 models: {M2 or 'not attached (PH2 will score only the new classifiers)'}")
cfg2 = json.load(open(os.path.join(AINP, 'phase2', 'cfg.json')))
if cfg2.get('dry_run') and not DRY_RUN:
    raise RuntimeError('The archive holds a dry run. Use the full Phase 2 archives, or set DRY_RUN = True.')

# ---------------------------------------------------------------- 1. input datasets
paths = dict(cfg2['paths'])


def first(match, root=INPUT_ROOT):
    h = walk(root, match); return h[0] if h else None


fix = {'pseudo_csv': lambda d, f: f == os.path.basename(paths.get('pseudo_csv') or 'isic2020_pseudo_labels_v3.csv'),
       'isic2020_meta': lambda d, f: f == os.path.basename(paths.get('isic2020_meta') or 'ISIC_2020_Train_Metadata.csv')}
for k, m in fix.items():
    if not paths.get(k) or not os.path.exists(paths[k]):
        p = first(m); paths[k] = p
        if p: say(f'{k}: found at {p}')
if not paths.get('isic2020_dir') or not os.path.isdir(paths['isic2020_dir']):
    if paths.get('isic2020_meta') and os.path.isdir(os.path.join(os.path.dirname(paths['isic2020_meta']), 'train')):
        paths['isic2020_dir'] = os.path.join(os.path.dirname(paths['isic2020_meta']), 'train')
miss = [k for k in ['pseudo_csv', 'isic2020_dir'] if not paths.get(k) or not os.path.exists(paths[k])]
if miss:
    raise FileNotFoundError(f'inputs missing: {miss}. Attach the same ISIC 2020 and pseudo-label datasets as Phase 2.')
for k, fn in [('cbmb', 'cbmb.pth'), ('isic18', 'isic18_detector.pth')]:
    if not paths.get(k) or not os.path.exists(paths[k]):
        paths[k] = first(lambda d, f, fn=fn: f == fn)
# HAM10000 (required): metadata csv and the folders with its ISIC_*.jpg images
ham_meta = first(lambda d, f: f == 'HAM10000_metadata.csv')
if not ham_meta:
    raise FileNotFoundError('HAM10000_metadata.csv not found: attach the HAM10000 dataset (Skin Cancer MNIST: HAM10000).')
HAM_ROOT = os.path.dirname(ham_meta); ham_dirs = []
for dp, dn, fn in os.walk(HAM_ROOT, followlinks=True):
    if any(f.startswith('ISIC_') and f.lower().endswith('.jpg') for f in fn[:200]): ham_dirs.append(dp)
if not ham_dirs:
    raise FileNotFoundError(f'no HAM10000 images (ISIC_*.jpg) found under {HAM_ROOT}')
# PH2 (optional)
ph2_txt = first(lambda d, f: f.lower() == 'ph2_dataset.txt')
PH2 = {'txt': ph2_txt, 'root': os.path.dirname(ph2_txt)} if ph2_txt else None
if PH2 and not paths.get('cbmb'):
    say('PH2 found, but cbmb.pth was not found: attach the Phase 1b evaluators to run rule 8'); PH2 = None
print('\n' + '=' * 78 + '\nINPUTS\n' + '=' * 78)
print(f"  OK  HAM10000       {ham_meta} ({len(ham_dirs)} image folders)")
print(f"  {'OK ' if PH2 else '-- '} PH2            {ph2_txt or 'not attached (rule 8 is not run)'}")
for k in ['cbmb', 'isic18']:
    print(f"  {'OK ' if paths.get(k) and os.path.exists(paths[k]) else '-- '} {k:<14} {paths.get(k)}")
for k in ['pseudo_csv', 'isic2020_dir', 'isic2020_meta']:
    print(f"  {'OK ' if paths.get(k) and os.path.exists(paths[k]) else '-- '} {k:<14} {paths.get(k)}")

# ---------------------------------------------------------------- 2. GPUs, code, weights
try:
    gl = [l for l in subprocess.run(['nvidia-smi', '-L'], capture_output=True, text=True).stdout.splitlines() if l.startswith('GPU')]
except Exception:
    gl = []
N_GPU = len(gl)
if N_GPU == 0 and not os.environ.get('P1_INPUT_ROOT'):
    raise RuntimeError('No GPU found. Set Accelerator = GPU T4 x2.')
HW = (gl[0].split(':')[1].split('(')[0].strip() + f' x{N_GPU}') if gl else 'CPU (offline test)'
SHARDS = [(str(g), g) for g in range(N_GPU)] if N_GPU else [('', 0), ('', 1)]
ncpu = os.cpu_count() or 4
say(f'hardware: {HW}; {len(SHARDS)} GPU worker(s); {ncpu} CPUs')
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
def make_ddim(steps, spacing='leading'):
    from diffusers import DDIMScheduler
    ddim = DDIMScheduler(num_train_timesteps=1000, beta_schedule='linear', clip_sample=True, timestep_spacing=spacing)
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
def generate_crn(model, cond, noise, device, cfg_scale=3.5, steps=50, bs=32, spacing='leading'):
    """Classifier-free-guided DDIM (eta=0, deterministic) from given noise.
    cond: (N,d) float; noise: (N,3,H,W) float. Returns uint8 (N,3,H,W) on CPU."""
    ddim = make_ddim(steps, spacing); out = []
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


def shades_of_gray(u8, p=6):
    """Shades-of-Gray colour constancy (Finlayson and Trezzi, 2004), as used for
    dermoscopy by Barata et al. (2015). (N,3,H,W) uint8 -> uint8."""
    x = u8.astype(np.float32) / 255.0
    e = np.power(np.mean(np.power(x, p), axis=(2, 3)), 1.0 / p)          # (N,3) illuminant estimate
    e = e / (np.linalg.norm(e, axis=1, keepdims=True) + 1e-8)
    y = x / (e[:, :, None, None] * np.sqrt(3.0) + 1e-8)
    return (np.clip(y, 0, 1) * 255.0 + 0.5).astype(np.uint8)


# ----------------------------------------------------------------------------
# DDIM without diffusers (Phase 1d). Same numbers as
# diffusers.DDIMScheduler(num_train_timesteps=1000, beta_schedule='linear',
# clip_sample=True, set_alpha_to_one=True, steps_offset=0), eta = 0.
# ----------------------------------------------------------------------------
def ddim_table(steps=50, spacing='leading', T=1000, beta_start=1e-4, beta_end=0.02):
    """returns timesteps (list of int) and coefficients (steps, 4) float32:
    sqrt(a_t), sqrt(1 - a_t), sqrt(a_prev), sqrt(1 - a_prev)"""
    betas = torch.linspace(beta_start, beta_end, T, dtype=torch.float32)
    ac = torch.cumprod(1.0 - betas, dim=0)
    if spacing == 'leading':
        r = T // steps
        ts = (np.arange(0, steps) * r).round()[::-1].copy().astype(np.int64)
    elif spacing == 'trailing':
        r = T / steps
        ts = (np.round(np.arange(T, 0, -r)).astype(np.int64) - 1)
    else:
        raise ValueError(spacing)
    rows = []
    for t in ts:
        prev = int(t) - T // steps
        a_t = ac[int(t)]
        a_p = ac[prev] if prev >= 0 else torch.tensor(1.0)
        rows.append(torch.stack([a_t ** 0.5, (1 - a_t) ** 0.5, a_p ** 0.5, (1 - a_p) ** 0.5]))
    return [int(t) for t in ts], torch.stack(rows).float()


def ddim_update(x, eps, co):
    """one DDIM step (eta = 0, clip_sample). co = (sqrt a_t, sqrt(1-a_t), sqrt a_prev, sqrt(1-a_prev))"""
    x0 = ((x - co[1] * eps) / co[0]).clamp(-1.0, 1.0)
    return co[2] * x0 + co[3] * eps


@torch.no_grad()
def generate_v2_eager(model, cond, noise, scale, device, steps=50, bs=32):
    """reference implementation (any device, eager): per-item guidance scale. Returns uint8."""
    ts_list, tab = ddim_table(steps); out = []
    for s in range(0, noise.shape[0], bs):
        x = noise[s:s + bs].to(device); c = cond[s:s + bs].to(device); g = scale[s:s + bs].to(device).view(-1, 1, 1, 1)
        b = x.shape[0]; z = torch.zeros_like(c)
        for k, t in enumerate(ts_list):
            ts = torch.full((2 * b,), t, dtype=torch.long, device=device)
            e_c, e_u = model(torch.cat([x, x]), ts, torch.cat([c, z])).chunk(2)
            x = ddim_update(x, e_u + g * (e_c - e_u), tab[k].to(device))
        out.append(float_to_u8((x.clamp(-1, 1) + 1) / 2).cpu())
    return torch.cat(out, 0)
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
    'worker2.py': r'''"""ClinDiffXAI Phase 2 worker: counterfactual concept audit of melanoma classifiers.
Usage: python worker2.py <job> <cfg.json> [rank world]
Jobs: plan (CPU), prep_cls (CPU), gen (GPU shard), assemble (CPU), train (GPU shard), score (GPU shard)."""
import os, sys, json, time, glob, math, traceback
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *  # noqa
from worker import _aug_batch, _cache   # Phase 1b helpers (augmentation, image cache)


def P(cfg, *parts):
    p = os.path.join(cfg['work'], *parts)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    return p


def W1D(cfg, *parts):
    return os.path.join(cfg['work1d'], *parts)


def atomic_torch_save(obj, path):
    torch.save(obj, path + '.tmp'); os.replace(path + '.tmp', path)


def atomic_npz(path, **arrays):
    with open(path + '.tmp', 'wb') as fh:
        np.savez(fh, **arrays)
    os.replace(path + '.tmp', path)


# =============================================================================
# plan (CPU): realistic-base sweeps. The base vectors, start noises and colour
# targets are those of the Phase 1d eval_realc images (cases 0..n_bases-1);
# one concept is set to each level, the other six keep the real case's values.
# =============================================================================
def job_plan(cfg):
    R = cfg['rsweep']; out = P(cfg, 'plan', 'rsweep_plan.pt')
    sets, chunks = {}, []
    if R['enabled']:
        plan1d = torch.load(W1D(cfg, 'plan', 'plan.pt'), map_location='cpu', weights_only=False)
        levels = [round(float(x), 4) for x in np.linspace(0, 1, R['n_levels'])]
        for cls in ['mel', 'ben']:
            S = plan1d['sets'][f'eval_realc_{cls}']; nb = min(R['n_bases'], S['n'])
            src = list(S['meta'].get('source_image', [''] * S['n']))[:nb]
            conds, idx, ms, meta = [], [], [], {'concept_idx': [], 'level': [], 'noise_id': [], 'base_value': [], 'source_image': []}
            for j in range(7):
                for l in levels:
                    c = S['cond'][:nb].clone(); bv = c[:, j].clone(); c[:, j] = l
                    conds.append(c); idx.append(S['noise_idx'][:nb]); ms.append(S['mstar'][:nb])
                    meta['concept_idx'] += [j] * nb; meta['level'] += [l] * nb; meta['noise_id'] += list(range(nb))
                    meta['base_value'] += [float(v) for v in bv]; meta['source_image'] += src
            n = 7 * len(levels) * nb
            sets[f'rsweep_{cls}'] = {'model': 'main', 'ckpt': S['ckpt'], 'nc': 7, 'ckpt_info': S['ckpt_info'],
                                     'cond': torch.cat(conds), 'noise_seed': S['noise_seed'], 'noise_n': S['noise_n'],
                                     'noise_idx': torch.cat(idx), 'scale': torch.full((n,), float(cfg['cfg_scale'])),
                                     'mstar': torch.cat(ms), 'meta': meta, 'n': n,
                                     'info': {'kind': 'realistic_sweep', 'base_class': cls, 'levels': levels, 'n_bases': nb,
                                              'sampler': 'V2 colour-calibrated start (Phase 1d plan of eval_realc)'}}
        B = cfg['gpu']['gen_batch']
        for name in sorted(sets):
            for lo in range(0, sets[name]['n'], B):
                chunks.append({'set': name, 'lo': lo, 'hi': min(lo + B, sets[name]['n'])})
    atomic_torch_save({'sets': sets, 'chunks': chunks}, out)
    log(f"plan: {len(sets)} realistic sweep sets, {sum(s['n'] for s in sets.values())} images, {len(chunks)} chunks")


# =============================================================================
# prep_cls (CPU): classifier data from ISIC 2020 (all melanomas + a benign sample),
# split by patient when patient ids are available
# =============================================================================
def job_prep_cls(cfg):
    import pandas as pd
    from sklearn.model_selection import StratifiedGroupKFold
    C = cfg['cls']; IS = cfg['image_size']
    df = pd.read_csv(cfg['paths']['pseudo_csv'])
    meta = cfg['paths'].get('isic2020_meta')
    if 'patient_id' not in df.columns and meta:
        m = pd.read_csv(meta)
        m = m[[c for c in ['image_name', 'patient_id'] if c in m.columns]].drop_duplicates('image_name')
        df = df.merge(m, on='image_name', how='left')
    have_pid = 'patient_id' in df.columns and df['patient_id'].notna().mean() > 0.95
    mel = df[df['target'] == 1]
    if C.get('n_mel_max'):
        mel = mel.sample(n=min(len(mel), C['n_mel_max']), random_state=2027)
    ben_all = df[df['target'] == 0]
    ben = ben_all.sample(n=min(len(ben_all), int(len(mel) * C['ben_ratio'])), random_state=2027)
    d = pd.concat([mel, ben]).reset_index(drop=True)
    groups = (d['patient_id'].fillna(d['image_name']) if have_pid else d['image_name']).astype(str).values
    fold = np.zeros(len(d), int)
    sgkf = StratifiedGroupKFold(n_splits=20, shuffle=True, random_state=2027)
    for k, (_, te) in enumerate(sgkf.split(np.zeros(len(d)), d['target'].values, groups)):
        fold[te] = k
    split = np.where(fold < 3, 'test', np.where(fold < 6, 'val', 'train'))
    paths = [os.path.join(cfg['paths']['isic2020_dir'], f'{n}.jpg') for n in d['image_name']]
    log(f"prep_cls: {len(d)} images ({int(d['target'].sum())} melanoma); split by {'patient' if have_pid else 'IMAGE (no patient ids)'}; "
        f"train/val/test = {(split == 'train').sum()}/{(split == 'val').sum()}/{(split == 'test').sum()}")
    X144, X128 = _cache(paths, [IS + IS // 8, IS], cfg['cpu_workers_io'], draft=True)
    ccols = [f'concept_{c}' for c in CONCEPT_NAMES]
    tr = split == 'train'
    atomic_torch_save({'X144_train': X144[torch.from_numpy(tr)], 'X128': X128, 'split': split.tolist(),
                       'target': d['target'].values.astype(np.float32), 'concepts': d[ccols].values.astype(np.float32),
                       'image_name': list(d['image_name']), 'group': list(groups), 'have_pid': bool(have_pid)},
                      P(cfg, 'cls', 'data.pt'))
    te = np.where(split == 'test')[0]
    save_set(P(cfg, 'sets', 'cls_test.pt'), X128[torch.from_numpy(te)],
             {'image_name': [d['image_name'].iloc[i] for i in te], 'target': [int(d['target'].iloc[i]) for i in te],
              **{c: [float(d[c].iloc[i]) for i in te] for c in ccols}},
             {'kind': 'classifier_test', 'split_by': 'patient' if have_pid else 'image'})
    # data-level association: melanoma label ~ pseudo-label concepts (all ISIC 2020 images in the CSV)
    from sklearn.linear_model import LogisticRegression
    Xa = df[ccols].values; Xa = (Xa - Xa.mean(0)) / (Xa.std(0) + 1e-9)
    lr = LogisticRegression(max_iter=2000, class_weight='balanced').fit(Xa, df['target'].values)
    save_json({'have_patient_ids': bool(have_pid), 'n': int(len(d)), 'n_mel': int(d['target'].sum()),
               'split_counts': {k: int((split == k).sum()) for k in ['train', 'val', 'test']},
               'split_mel': {k: int(d['target'].values[split == k].sum()) for k in ['train', 'val', 'test']},
               'association_std_coef': dict(zip(CONCEPT_NAMES, lr.coef_[0].tolist())), 'n_association': int(len(df))},
              P(cfg, 'results', 'cls_data.json'))


# =============================================================================
# gen (one process per GPU): the Phase 1c/1d V2 sampler
# =============================================================================
def job_gen(cfg, rank=0, world=1):
    dev = get_device(); torch.backends.cudnn.benchmark = True
    plan = torch.load(P(cfg, 'plan', 'rsweep_plan.pt'), map_location='cpu', weights_only=False)
    IS = cfg['image_size']; B = cfg['gpu']['gen_batch']
    out_dir = os.path.join(cfg['work'], 'chunks_gen'); os.makedirs(out_dir, exist_ok=True)
    todo = [(i, c) for i, c in enumerate(plan['chunks']) if i % world == rank and not os.path.exists(os.path.join(out_dir, f'{i:05d}.pt'))]
    log(f"[gen {rank}/{world}] {dev}: {len(todo)} chunks to make")
    models, noise_cache = {}, {}; t0 = time.time(); n_img = 0
    for k, (ci, ch) in enumerate(todo):
        S = plan['sets'][ch['set']]; lo, hi = ch['lo'], ch['hi']
        if S['model'] not in models:
            models[S['model']] = load_diffusion(S['ckpt'], S['nc'], dev, IS)[0]
        key = (S['noise_seed'], S['noise_n'])
        if key not in noise_cache:
            noise_cache[key] = cpu_noise(S['noise_n'], IS, S['noise_seed'])
        z = noise_cache[key][S['noise_idx'][lo:hi]]
        z = z - z.mean((2, 3), keepdim=True) + S['mstar'][lo:hi][:, :, None, None]
        imgs = generate_crn(models[S['model']], S['cond'][lo:hi], z, dev, cfg_scale=float(S['scale'][lo]),
                            steps=cfg['ddim_steps'], bs=B)
        atomic_torch_save(imgs, os.path.join(out_dir, f'{ci:05d}.pt'))
        n_img += hi - lo
        if k == 0 or (k + 1) % 10 == 0:
            el = time.time() - t0
            log(f"[gen {rank}] {k + 1}/{len(todo)} chunks, {n_img / el:.2f} images/s, about {(len(todo) - k - 1) * el / (k + 1) / 60:.0f} min left")
    log(f"[gen {rank}] finished: {n_img} images in {(time.time() - t0) / 60:.1f} min")


def job_assemble(cfg):
    plan = torch.load(P(cfg, 'plan', 'rsweep_plan.pt'), map_location='cpu', weights_only=False)
    miss = [i for i in range(len(plan['chunks'])) if not os.path.exists(P(cfg, 'chunks_gen', f'{i:05d}.pt'))]
    if miss:
        raise RuntimeError(f'{len(miss)} generation chunks are missing; run the cell again to finish them')
    by = {}
    for i, ch in enumerate(plan['chunks']):
        by.setdefault(ch['set'], []).append((ch['lo'], i))
    for name, S in plan['sets'].items():
        out = P(cfg, 'sets', f'{name}.pt')
        if os.path.exists(out): continue
        imgs = torch.cat([torch.load(P(cfg, 'chunks_gen', f'{i:05d}.pt'), weights_only=False) for _, i in sorted(by[name])])
        assert len(imgs) == S['n']
        save_set(out, imgs, S['meta'], {**S['info'], 'ckpt': S['ckpt_info']})
        log(f"  assembled {name}: {len(imgs)} images")


# =============================================================================
# classifiers under audit
# =============================================================================
class Clf(nn.Module):
    def __init__(self, arch, pretrained, img_size):
        super().__init__()
        import torchvision.models as tvm
        if arch == 'resnet50':
            m = tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None); d = m.fc.in_features; m.fc = nn.Identity()
        elif arch == 'effb0':
            m = tvm.efficientnet_b0(weights=tvm.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None)
            d = m.classifier[1].in_features; m.classifier = nn.Identity()
        elif arch == 'vitb16':
            m = tvm.vit_b_16(weights=tvm.ViT_B_16_Weights.IMAGENET1K_V1) if pretrained else tvm.vit_b_16(weights=None, image_size=img_size)
            d = m.heads.head.in_features; m.heads = nn.Identity()
        else:
            raise ValueError(arch)
        self.body = m; self.drop = nn.Dropout(0.2); self.head = nn.Linear(d, 1)

    def forward(self, x):
        return self.head(self.drop(self.body(x))).squeeze(1)


@torch.no_grad()
def clf_logits(net, X_u8, cfg, dev, bs=128, amp=True):
    """amp=True (float16) only for validation during training; every audit score uses float32 (amp=False)"""
    net.eval(); out = []
    for s in range(0, len(X_u8), bs):
        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=amp and dev.type == 'cuda'):
            o = net(u8_to_eval(X_u8[s:s + bs], dev, cfg['eval_size']))
        out.append(o.float().cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0, np.float32)


def _auc(y, s):
    from sklearn.metrics import roc_auc_score
    y = np.asarray(y); ok = np.isfinite(s)
    if len(np.unique(y[ok])) < 2: return float('nan')
    return float(roc_auc_score(y[ok], s[ok]))


MATCHED = {'mpn': 'pigment_network', 'mvs': 'vascular_structures', 'mpig': 'pigmentation',
           'mreg': 'regression_structures'}   # Phase 2c; regression added in Phase 2d


def matched_groups(data, cfg, concept):
    """Phase 2c matched control groups: high versus low quartile of one pseudo-label (thresholds from the training
    split), 1:1 pairs matched on the other six pseudo-labels (standardised on the training split, Euclidean caliper)
    and exactly on the melanoma label; greedy nearest neighbour without replacement in a seeded random order;
    separately within each split. Returns {split: (n_pairs, 2) array of (high, low) indices}, (low, high) thresholds."""
    M = cfg['matched']; split = np.array(data['split']); y = np.asarray(data['target']).astype(int)
    Cc = np.asarray(data['concepts'], np.float64); j = CONCEPT_NAMES.index(concept)
    tr = split == 'train'; lo, hi = np.quantile(Cc[tr, j], M['q']), np.quantile(Cc[tr, j], 1 - M['q'])
    if concept in M.get('abs_high', {}):   # sparse concept (Phase 2d): high = pseudo-label at or above a fixed value
        hi = float(M['abs_high'][concept])
    cal = float(M.get('caliper_by_concept', {}).get(concept, M['caliper']))
    oth = [k for k in range(7) if k != j]; mu = Cc[tr][:, oth].mean(0); sd = Cc[tr][:, oth].std(0) + 1e-9
    Z = (Cc[:, oth] - mu) / sd; rng = np.random.default_rng(M['seed'] + j); out = {}
    for s in ['train', 'val', 'test']:
        H = np.where((split == s) & (Cc[:, j] >= hi))[0]; L = np.where((split == s) & (Cc[:, j] <= lo))[0]; pairs = []
        if len(H) and len(L):
            Dm = np.sqrt(((Z[H][:, None, :] - Z[L][None, :, :]) ** 2).sum(-1)); Dm[y[H][:, None] != y[L][None, :]] = np.inf
            used = np.zeros(len(L), bool)
            for i in rng.permutation(len(H)):
                d = np.where(used, np.inf, Dm[i]); k = int(np.argmin(d))
                if d[k] <= cal:
                    used[k] = True; pairs.append((H[i], L[k]))
        out[s] = np.array(pairs, dtype=int).reshape(-1, 2)
    return out, (float(lo), float(hi))


def _labels(spec, data, cfg):
    """returns per-split (index array, label array) for the spec's label"""
    split = np.array(data['split']); y = data['target']; Cc = data['concepts']; out = {}
    if spec['label'] in ('mel', 'perm'):
        for k in ['train', 'val', 'test']:
            ii = np.where(split == k)[0]; out[k] = (ii, y[ii].copy())
        if spec['label'] == 'perm':   # negative control: labels permuted within train and within val
            rng = np.random.default_rng(spec['seed'] + 99)
            for k in ['train', 'val']:
                ii, yy = out[k]; out[k] = (ii, rng.permutation(yy))
    elif spec['label'] in MATCHED:    # Phase 2c matched positive control
        pr, _ = matched_groups(data, cfg, MATCHED[spec['label']])
        for k in ['train', 'val', 'test']:
            p = pr[k]; out[k] = (np.concatenate([p[:, 0], p[:, 1]]), np.r_[np.ones(len(p)), np.zeros(len(p))].astype(np.float32))
    else:                              # positive control: high versus low pseudo-label of one concept
        j = CONCEPT_NAMES.index({'pig': 'pigmentation', 'bwv': 'blue_whitish_veil'}[spec['label']])
        trv = Cc[split == 'train', j]
        lo, hi = np.quantile(trv, cfg['cls']['ctrl_q']), np.quantile(trv, 1 - cfg['cls']['ctrl_q'])
        for k in ['train', 'val', 'test']:
            ii = np.where((split == k) & ((Cc[:, j] <= lo) | (Cc[:, j] >= hi)))[0]
            out[k] = (ii, (Cc[ii, j] >= hi).astype(np.float32))
    return out


def train_one(cfg, spec, data, dev):
    torch.manual_seed(spec['seed']); np.random.seed(spec['seed'])
    T = cfg['train']; IS = cfg['image_size']
    if spec['data'] == 'derm7pt':
        tr, vl, te = load_derm7pt(cfg['paths'])
        Xtr, = _cache(list(tr['image_path']), [IS + IS // 8], cfg['cpu_workers_io']); Xvl, = _cache(list(vl['image_path']), [IS], cfg['cpu_workers_io'])
        Xte, = _cache(list(te['image_path']), [IS], cfg['cpu_workers_io'])
        L = {'train': (None, tr['target'].values.astype(np.float32)), 'val': (None, vl['target'].values.astype(np.float32)),
             'test': (None, te['target'].values.astype(np.float32))}
    else:
        L = _labels(spec, data, cfg); split = np.array(data['split'])
        pos_in_train = {g: i for i, g in enumerate(np.where(split == 'train')[0])}
        Xtr = data['X144_train'][torch.as_tensor([pos_in_train[i] for i in L['train'][0]], dtype=torch.long)]
        Xvl = data['X128'][torch.as_tensor(L['val'][0], dtype=torch.long)]; Xte = None
    ytr = torch.tensor(L['train'][1]); yvl = L['val'][1]
    net = Clf(spec['arch'], cfg['pretrained'], cfg['eval_size']).to(dev)
    E = T['epochs']; bs = T['batch']; lr = T['lr_vit'] if spec['arch'] == 'vitb16' else T['lr_cnn']
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=0.05)
    steps = E * max(1, len(Xtr) // bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1, anneal_strategy='cos')
    use_amp = dev.type == 'cuda'; scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    npos = float(ytr.sum()); pw = torch.tensor(min(20.0, max(1.0, (len(ytr) - npos) / max(npos, 1.0))), device=dev)
    best, best_state, bad, hist = -1.0, None, 0, []
    t0 = time.time()
    for ep in range(E):
        net.train(); perm = np.random.permutation(len(Xtr)); losses = []
        for s in range(0, len(perm) - bs + 1, bs):
            idx = perm[s:s + bs]
            xb = u8_to_eval(_aug_batch(Xtr[idx], IS, True), dev, cfg['eval_size']); yb = ytr[idx].to(dev)
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp):
                lo = net(xb)
            loss = F.binary_cross_entropy_with_logits(lo.float(), yb, pos_weight=pw)
            opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); scaler.step(opt); scaler.update(); sched.step()
            losses.append(loss.item())
        auc = _auc(yvl, clf_logits(net, Xvl, cfg, dev))
        hist.append({'epoch': ep + 1, 'loss': float(np.mean(losses)) if losses else None, 'val_auc': auc})
        log(f"  [{spec['name']}] epoch {ep + 1}/{E} loss {hist[-1]['loss']} val AUC {auc:.4f} ({(time.time() - t0) / 60:.1f} min)")
        if np.isfinite(auc) and auc > best + 1e-4:
            best, bad = auc, 0; best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= T['patience']: break
    if best_state is not None: net.load_state_dict(best_state)
    res = {'spec': spec, 'history': hist, 'best_val_auc': best, 'n_train': int(len(ytr)), 'n_train_pos': int(npos),
           'minutes': (time.time() - t0) / 60}
    # threshold at 90% sensitivity on the validation set (for the melanoma label)
    split = np.array(data['split']); yv_true = data['target'][split == 'val']; v_log = clf_logits(net, data['X128'][torch.from_numpy(split == 'val')], cfg, dev)
    t_log = clf_logits(net, data['X128'][torch.from_numpy(split == 'test')], cfg, dev); yt = data['target'][split == 'test']
    pos = np.sort(v_log[yv_true == 1])
    tau = float(pos[int(np.floor(0.10 * len(pos)))]) if len(pos) else float('nan')
    res.update({'tau90_logit': tau, 'test_auc_melanoma': _auc(yt, t_log), 'val_auc_melanoma': _auc(yv_true, v_log),
                'test_sens_at_tau': float(np.mean(t_log[yt == 1] >= tau)) if (yt == 1).any() else float('nan'),
                'test_spec_at_tau': float(np.mean(t_log[yt == 0] < tau)) if (yt == 0).any() else float('nan')})
    if spec['label'] in ('pig', 'bwv') or spec['label'] in MATCHED:
        ii, yy = L['test']; res['test_auc_own_label'] = _auc(yy, clf_logits(net, data['X128'][torch.as_tensor(ii, dtype=torch.long)], cfg, dev))
    if spec['data'] == 'derm7pt':
        res['derm7pt_test_auc'] = _auc(L['test'][1], clf_logits(net, Xte, cfg, dev))
    atomic_npz(P(cfg, 'models', f"pred_{spec['name']}.npz"), val_logit=v_log, test_logit=t_log)
    atomic_torch_save({'state_dict': net.state_dict(), 'spec': spec}, P(cfg, 'models', f"{spec['name']}.pt"))
    save_json(res, P(cfg, 'results', f"train_{spec['name']}.json"))
    log(f"  [{spec['name']}] done: val AUC {best:.4f}, test AUC (melanoma) {res['test_auc_melanoma']:.4f}, tau90 {tau:.3f}")


def job_train(cfg, rank=0, world=1):
    dev = get_device(); torch.backends.cudnn.benchmark = True
    mine = [m for m in cfg['models'] if m['gpu'] % world == rank]
    data = torch.load(P(cfg, 'cls', 'data.pt'), map_location='cpu', weights_only=False)
    log(f"[train {rank}/{world}] {dev}: {[m['name'] for m in mine]}")
    for spec in mine:
        if os.path.exists(P(cfg, 'models', f"{spec['name']}.pt")):
            log(f"  {spec['name']}: done earlier"); continue
        train_one(cfg, spec, data, dev)


# =============================================================================
# score (one process per GPU): classifier logits on every audit set; CBM-A,
# CBM-B, ISIC 2018 detector, darkness and CIELAB for the new sets
# =============================================================================
AUDIT_1D = ['sweep_mel', 'sweep_ben']   # Phase 3: prototype sweeps only (the evaluation sets are not needed)
AUDIT_2 = ['rsweep_mel', 'rsweep_ben', 'cls_test']


def audit_sets(cfg):
    out = []
    for n in AUDIT_1D:
        p = W1D(cfg, 'sets', f'{n}.pt')
        if os.path.exists(p): out.append((n, p))
    for n in AUDIT_2:
        p = P(cfg, 'sets', f'{n}.pt')
        if os.path.exists(p): out.append((n, p))
    return out


def score_tasks(cfg):
    cost = {'vitb16': 3, 'resnet50': 1, 'effb0': 1}
    tasks = [(m['name'], cost[m['arch']]) for m in cfg['models']] + [('aux', 2)]
    return sorted(tasks, key=lambda t: (-t[1], t[0]))


def job_score(cfg, rank=0, world=1):
    dev = get_device(); load = [0] * world; assign = {}
    for name, c in score_tasks(cfg):
        g = int(np.argmin(load)); assign[name] = g; load[g] += c
    mine = [n for n, g in assign.items() if g == rank]
    sets = audit_sets(cfg); log(f"[score {rank}/{world}] {dev}: {mine}; {len(sets)} sets")
    cache = {}
    dat = torch.load(P(cfg, 'cls', 'data.pt'), map_location='cpu', weights_only=False)
    Xval = dat['X128'][torch.from_numpy(np.array(dat['split']) == 'val')]; del dat

    def imgs(name, path):
        if name not in cache: cache[name] = load_set(path)['imgs']
        return cache[name]
    for name in mine:
        if name == 'aux':
            _score_aux(cfg, dev, [(n, p) for n, p in sets if n in AUDIT_2], imgs); continue
        out = P(cfg, cfg['scores_dir'], f'{name}.npz')
        if os.path.exists(out): continue
        ck = torch.load(P(cfg, 'models', f'{name}.pt'), map_location='cpu', weights_only=False)
        net = Clf(ck['spec']['arch'], False, cfg['eval_size']); net.load_state_dict(ck['state_dict']); net = net.to(dev).eval()
        t0 = time.time(); res = {n: clf_logits(net, imgs(n, p), cfg, dev, amp=False) for n, p in sets}
        res['_val'] = clf_logits(net, Xval, cfg, dev, amp=False)   # for the 90%-sensitivity threshold in float32
        nf = sum(int((~np.isfinite(v)).sum()) for v in res.values())
        atomic_npz(out, **res); log(f"  scored {name} (float32) on {sum(len(v) for v in res.values())} images in "
                                    f"{(time.time() - t0) / 60:.1f} min; non-finite outputs: {nf}")
        del net; torch.cuda.empty_cache() if dev.type == 'cuda' else None


def _score_aux(cfg, dev, sets, imgs):
    if all(os.path.exists(P(cfg, 'scores', f'{t}_{n}.npz')) for n, _ in sets for t in 'AB'):
        log('  aux scores: all present, skipped'); return
    cbm, _ = load_cbm_a(cfg['paths']['cbma'], dev)
    nb = ConceptNetB(pretrained=False)
    nb.load_state_dict(torch.load(cfg['paths']['cbmb'], map_location='cpu', weights_only=False)['state_dict']); nb = nb.to(dev).eval()
    ni = None
    if cfg['paths'].get('isic18'):
        ni = AttrNet(pretrained=False)
        ni.load_state_dict(torch.load(cfg['paths']['isic18'], map_location='cpu', weights_only=False)['state_dict']); ni = ni.to(dev).eval()
    for name, path in sets:
        fa, fb = P(cfg, 'scores', f'A_{name}.npz'), P(cfg, 'scores', f'B_{name}.npz')
        if os.path.exists(fa) and os.path.exists(fb): continue
        X = imgs(name, path); A, B = {'cbma': [], 'cbma_feat': [], 'cbma_diag': []}, {'cbmb': [], 'cbmb_feat': [], 'isic18': []}
        with torch.no_grad():
            for s in range(0, len(X), 64):
                xe = u8_to_eval(X[s:s + 64], dev, cfg['eval_size'])
                cp, cf, dp = cbm.forward_all(xe); e = nb.embed(xe)
                A['cbma'].append(cp.cpu().numpy()); A['cbma_feat'].append(cf.cpu().numpy()); A['cbma_diag'].append(dp.cpu().numpy())
                B['cbmb'].append(torch.sigmoid(nb.head(e)).cpu().numpy()); B['cbmb_feat'].append(e.cpu().numpy())
                if ni is not None: B['isic18'].append(torch.sigmoid(ni(xe)).cpu().numpy())
        u8 = X.numpy()
        atomic_npz(fa, **{k: np.concatenate(v) for k, v in A.items()}, dark=darkness_metrics(u8), lab=lab_means(u8))
        atomic_npz(fb, **{k: np.concatenate(v) for k, v in B.items() if v})
        log(f"  aux scores {name}: n={len(X)}")


JOBS = {'plan': job_plan, 'prep_cls': job_prep_cls, 'gen': job_gen, 'assemble': job_assemble, 'train': job_train,
        'score': job_score, 'score32': job_score}
SHARDED = ('gen', 'train', 'score', 'score32')

if __name__ == '__main__':
    job, cfg_path = sys.argv[1], sys.argv[2]
    rank = int(sys.argv[3]) if len(sys.argv) > 3 else 0; world = int(sys.argv[4]) if len(sys.argv) > 4 else 1
    cfg = load_cfg(cfg_path)
    rid = os.environ.get('P2_RUN_ID')
    if rid:
        try:
            cur = open(os.path.join(cfg['work'], 'run_id.txt')).read().strip()
        except Exception:
            cur = rid
        if cur != rid:
            print(f'job {job}: started by an older run of the cell; exit without work', flush=True); sys.exit(3)
    tag = f'{job}_{rank}' if job in SHARDED else job
    log(f"job {tag} start | device={get_device()} | torch {torch.__version__} | CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")
    t0 = time.time()
    try:
        JOBS[job](cfg, rank, world) if job in SHARDED else JOBS[job](cfg)
    except Exception:
        traceback.print_exc(); sys.exit(1)
    with open(P(cfg, 'done', f'{tag}.ok'), 'w') as f:
        f.write(f"{time.time() - t0:.0f}s\n")
    log(f"job {tag} done in {(time.time() - t0) / 60:.1f} min")
''',
    'worker2d.py': r'''"""ClinDiffXAI Phase 2d worker.
Usage: python worker2d.py <job> <cfg.json> [rank world]
Jobs: rebuild (CPU): image sets and scores from the saved archives; verify_cls (CPU): the rebuilt classifier data must equal
the Phase 2 data; captum (GPU shard): TCAV with the Captum library (default classifier) on the Phase 2c classifiers."""
import os, sys, json, glob, time, shutil, traceback
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *  # noqa
from worker2 import Clf, atomic_npz

CAPTUM_LAYER = {'resnet50': 'net.body.layer4', 'effb0': 'net.body.features.8', 'vitb16': 'net.body.encoder.layers.encoder_layer_11'}


def P(cfg, *parts):
    p = os.path.join(cfg['work'], *parts)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    return p


# =============================================================================
# rebuild (CPU): Phase 1d and Phase 2 image sets from clindiffxai_generated_images.zip; evaluator and classifier
# scores, training records and set metadata from phase2_analysis_inputs.zip and phase2c_outputs.zip
# =============================================================================
def job_rebuild(cfg):
    I = cfg['inputs']; W1D = cfg['work1d']; WA = cfg['work']
    for d in [os.path.join(W1D, 'sets'), os.path.join(W1D, 'scores'), os.path.join(WA, 'sets'), os.path.join(WA, 'scores'),
              os.path.join(WA, 'scores2_fp32'), os.path.join(WA, 'results')]:
        os.makedirs(d, exist_ok=True)
    n = 0
    for js in sorted(glob.glob(os.path.join(I['gen_dir'], '*.json'))):
        name = os.path.basename(js)[:-5]; npz = js[:-5] + '.npz'
        if not os.path.exists(npz): continue
        dst = os.path.join(WA if name.startswith('rsweep_') else W1D, 'sets', f'{name}.pt')
        if os.path.exists(dst): continue
        d = json.load(open(js)); x = np.load(npz)['imgs']
        if len(x) != d['n']: raise ValueError(f'{name}: {len(x)} images but the metadata says {d["n"]}')
        save_set(dst, torch.from_numpy(x), d['meta'], d.get('info', {})); n += 1
        log(f"  set {name}: {len(x)} images")
    log(f"rebuild: {n} image sets written")
    A = I['ainp_dir']
    for src, dst in [(os.path.join(A, 'phase1d', 'scores'), os.path.join(W1D, 'scores')), (os.path.join(A, 'phase2', 'scores'), os.path.join(WA, 'scores')),
                     (os.path.join(A, 'phase2', 'scores2_fp32'), os.path.join(WA, 'scores2_fp32'))]:
        for f in glob.glob(os.path.join(src, '*.npz')):
            if not os.path.exists(os.path.join(dst, os.path.basename(f))): shutil.copy(f, dst)
    for f in glob.glob(os.path.join(A, 'phase2', 'results', '*.json')):
        shutil.copy(f, os.path.join(WA, 'results'))
    C = I.get('p2c_dir') or os.path.join(WA, '_none')   # Phase 3: no Phase 2c archive is needed
    for f in glob.glob(os.path.join(C, 'train_ctrl_m*.json')): shutil.copy(f, os.path.join(WA, 'results'))
    for f in glob.glob(os.path.join(C, 'ctrl_m*.npz')): shutil.copy(f, os.path.join(WA, 'scores2_fp32'))
    for f in glob.glob(os.path.join(C, 'pred_ctrl_m*.npz')): shutil.copy(f, os.path.join(WA, 'results'))
    have = sorted(os.path.basename(f)[:-4] for f in glob.glob(os.path.join(WA, 'scores2_fp32', '*.npz')))
    log(f"rebuild: classifier scores present for {have}")


# =============================================================================
# verify_cls (CPU): the classifier data built again must be the Phase 2 data (same images, same split)
# =============================================================================
def job_verify_cls(cfg):
    ref = np.load(os.path.join(cfg['inputs']['ainp_dir'], 'phase2', 'cls', 'cls_labels.npz'))
    dat = torch.load(os.path.join(cfg['work'], 'cls', 'data.pt'), map_location='cpu', weights_only=False)
    names = np.asarray(dat['image_name']).astype(str); split = np.asarray(dat['split']).astype(str)
    ok = (len(names) == len(ref['image_name']) and np.array_equal(names, ref['image_name'].astype(str)) and
          np.array_equal(split, ref['split'].astype(str)) and np.allclose(np.asarray(dat['concepts']), ref['concepts'], atol=1e-6))
    save_json({'identical_to_phase2': bool(ok), 'n': int(len(names))}, P(cfg, 'results', 'cls_rebuild_check.json'))
    if not ok:
        raise RuntimeError('The classifier data built again differ from Phase 2 (images, split or labels). Stop: the saved '
                           'classifiers would then be tested on other images. Check that the same ISIC 2020 files are attached.')
    log(f"verify_cls: identical to Phase 2 ({len(names)} images, same order, split and pseudo-labels)")


# =============================================================================
# captum (GPU shard): TCAV with the Captum library on the Phase 2c classifiers
# =============================================================================
class _Wrap(nn.Module):
    def __init__(self, net):
        super().__init__(); self.net = net

    def forward(self, x):
        return self.net(x).unsqueeze(1)


class _Batches:
    """re-iterable concept data: preprocessed image batches on the device"""
    def __init__(self, X, idx, dev, size, bs):
        self.X, self.idx, self.dev, self.size, self.bs = X, list(idx), dev, size, bs

    def __iter__(self):
        for s in range(0, len(self.idx), self.bs):
            yield u8_to_eval(self.X[torch.as_tensor(self.idx[s:s + self.bs], dtype=torch.long)], self.dev, self.size)


def job_captum(cfg, rank=0, world=1):
    import warnings
    warnings.filterwarnings('ignore')
    from captum.concept import TCAV, Concept
    from captum.concept._utils.common import concepts_to_str
    import pandas as pd
    import captum
    dev = get_device(); C = cfg['captum']
    plan = json.load(open(P(cfg, 'plan.json'))); X = torch.load(P(cfg, 'bank.pt'), weights_only=False)['X']
    mine = [m for i, m in enumerate(cfg['captum_models']) if i % world == rank]
    tg = {'mel': plan['pos']['tgt_mel'], 'ben': plan['pos']['tgt_ben']}
    log(f"[captum {rank}/{world}] {dev}, captum {captum.__version__}: {[m['name'] for m in mine]}")
    for spec in mine:
        out = P(cfg, 'captum', f"{spec['name']}.csv")
        if os.path.exists(out): log(f"  {spec['name']}: done earlier"); continue
        t0 = time.time(); path = None
        for d in cfg['models_dirs']:
            if os.path.exists(os.path.join(d, f"{spec['name']}.pt")): path = os.path.join(d, f"{spec['name']}.pt"); break
        if path is None: raise FileNotFoundError(f"model {spec['name']}.pt not found")
        ck = torch.load(path, map_location='cpu', weights_only=False)
        net = Clf(spec['arch'], False, cfg['eval_size']); net.load_state_dict(ck['state_dict'])
        wrap = _Wrap(net).to(dev).eval()
        for p_ in wrap.parameters(): p_.requires_grad_(False)
        layer = CAPTUM_LAYER[spec['arch']]; rows = []
        for vi, v in enumerate(C['variants']):
            if v not in plan['variants']: continue
            V = plan['variants'][v]; save = os.path.join(C['tmp'], f"{spec['name']}__{v}")
            shutil.rmtree(save, ignore_errors=True); os.makedirs(save, exist_ok=True)
            tc = TCAV(wrap, layers=[layer], model_id=f"{spec['name']}__{v}", save_path=save)
            bs = C['batch']
            cons = {c: Concept(vi * 1000 + j, f'{v}:{c}', _Batches(X, V['concepts'][c][0][0], dev, cfg['eval_size'], bs))
                    for j, c in enumerate(CONCEPT_NAMES) if c in V['concepts']}
            r0 = Concept(vi * 1000 + 500, f'{v}:random0', _Batches(X, V['null'][0][0], dev, cfg['eval_size'], bs))
            rk = [Concept(vi * 1000 + 501 + k, f'{v}:random{k + 1}', _Batches(X, V['null'][k][1], dev, cfg['eval_size'], bs))
                  for k in range(min(C['runs'], len(V['null'])))]
            exp = [[cons[c], r] for c in cons for r in rk] + [[r0, r] for r in rk]
            tc.compute_cavs(exp)
            sums = {}
            for inp, idx in tg.items():
                n = 0
                for s in range(0, len(idx), C['input_batch']):
                    xb = u8_to_eval(X[torch.as_tensor(idx[s:s + C['input_batch']], dtype=torch.long)], dev, cfg['eval_size'])
                    sc = tc.interpret(inputs=xb, experimental_sets=exp, target=0)
                    for e in exp:
                        key = concepts_to_str(e)
                        sums[(inp, key)] = sums.get((inp, key), 0.0) + float(sc[key][layer]['sign_count'][0]) * len(xb)
                    n += len(xb)
                for e in exp: sums[(inp, concepts_to_str(e))] /= max(n, 1)
            for e in exp:
                key = concepts_to_str(e); first = e[0]
                concept = '_random' if first is r0 else first.name.split(':', 1)[1]
                n_set = len(V['null'][0][0]) if first is r0 else len(V['concepts'][concept][0][0])
                st = tc.cavs[key][layer].stats if tc.cavs.get(key, {}).get(layer) is not None else {}
                rows.append({'classifier': spec['name'], 'layer': 'late', 'variant': v, 'concept': concept, 'run': rk.index(e[1]),
                             'cav_acc': float(st['accs']) if 'accs' in st else np.nan, 'n_set': n_set,
                             'tcav_mel': sums[('mel', key)], 'tcav_ben': sums[('ben', key)]})
            shutil.rmtree(save, ignore_errors=True)
            log(f"  {spec['name']} {v}: {len(exp)} CAVs ({(time.time() - t0) / 60:.1f} min)")
        pd.DataFrame(rows).to_csv(out + '.tmp', index=False); os.replace(out + '.tmp', out)
        log(f"  {spec['name']}: done in {(time.time() - t0) / 60:.1f} min")
        del net, wrap; torch.cuda.empty_cache() if dev.type == 'cuda' else None


def job_captum_merge(cfg):
    import pandas as pd
    fs = sorted(glob.glob(P(cfg, 'captum', '*.csv')))
    if fs:
        pd.concat([pd.read_csv(f) for f in fs]).to_csv(P(cfg, 'results', 'captum_runs.csv'), index=False)
    log(f"captum_merge: {len(fs)} classifiers")


JOBS = {'rebuild': job_rebuild, 'verify_cls': job_verify_cls, 'captum': job_captum, 'captum_merge': job_captum_merge}
SHARDED = ('captum',)

if __name__ == '__main__':
    job, cfg_path = sys.argv[1], sys.argv[2]
    rank = int(sys.argv[3]) if len(sys.argv) > 3 else 0; world = int(sys.argv[4]) if len(sys.argv) > 4 else 1
    cfg = load_cfg(cfg_path)
    rid = os.environ.get('P2D_RUN_ID')
    if rid:
        try:
            cur = open(os.path.join(cfg['work'], 'run_id.txt')).read().strip()
        except Exception:
            cur = rid
        if cur != rid:
            print(f'job {job}: started by an older run of the cell; exit without work', flush=True); sys.exit(3)
    tag = f'{job}_{rank}' if job in SHARDED else job
    log(f"job {tag} start | device={get_device()} | torch {torch.__version__} | CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")
    t0 = time.time()
    try:
        JOBS[job](cfg, rank, world) if job in SHARDED else JOBS[job](cfg)
    except Exception:
        traceback.print_exc(); sys.exit(1)
    with open(P(cfg, 'done', f'{tag}.ok'), 'w') as f:
        f.write(f"{time.time() - t0:.0f}s\n")
    log(f"job {tag} done in {(time.time() - t0) / 60:.1f} min")
''',
    'worker3.py': r'''"""ClinDiffXAI Phase 3 worker.
Usage: python worker3.py <job> <cfg.json> [rank world]
generic (GPU shard): ImageNet-pretrained backbones (no training) and fixed random linear heads, scored on the realistic
    sweeps and the real test images. Each head is a generic image-sensitive function with no melanoma training: a second,
    untrained null for the scale-free audit effects.
prep_ham (CPU): HAM10000 classifier data (melanoma against all other diagnoses), split by lesion, images that also occur
    in ISIC 2020 removed.
train_ham (GPU shard): melanoma classifiers trained on HAM10000 with the Phase 2 recipe (worker2.train_one).
ph2 (GPU): CBM-B and the ISIC 2018 detector on the PH2 images, against the PH2 expert dermoscopic criteria; melanoma
    classifiers on PH2 (melanoma against nevi)."""
import os, sys, json, time, traceback
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *  # noqa
from worker2 import Clf, atomic_npz, train_one, clf_logits, atomic_torch_save
from worker import _cache

GEN_SETS = ['rsweep_ben', 'rsweep_mel', 'cls_test']


def P(cfg, *parts):
    p = os.path.join(cfg['work'], *parts)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    return p


@torch.no_grad()
def feats(net, X, cfg, dev, bs=128):
    out = []
    for s in range(0, len(X), bs):
        out.append(net.body(u8_to_eval(X[s:s + bs], dev, cfg['eval_size'])).float().cpu().numpy())
    return np.concatenate(out)


def job_generic(cfg, rank=0, world=1):
    dev = get_device(); G = cfg['generic']
    archs = [a for i, a in enumerate(G['archs']) if i % world == rank]
    log(f"[generic {rank}/{world}] {dev}: {archs}")
    for ai, arch in enumerate(G['archs']):
        if arch not in archs: continue
        outp = P(cfg, 'generic', f'generic_{arch}.npz')
        if os.path.exists(outp): log(f'  {arch}: done earlier'); continue
        t0 = time.time()
        net = Clf(arch, cfg['pretrained'], cfg['eval_size']).to(dev).eval()
        F_ = {}
        for s in GEN_SETS:
            X = load_set(P(cfg, 'sets', f'{s}.pt'))['imgs']; F_[s] = feats(net, X, cfg, dev)
        mu = F_['cls_test'].mean(0); d = mu.shape[0]
        rng = np.random.default_rng(G['seed'] + ai)
        W = rng.standard_normal((d, G['n_heads'])) / np.sqrt(d)
        atomic_npz(outp, **{s: ((F_[s] - mu) @ W).T.astype(np.float32) for s in GEN_SETS})
        log(f'  {arch}: {G["n_heads"]} random heads on {d} features in {(time.time() - t0) / 60:.1f} min')
        del net; torch.cuda.empty_cache() if dev.type == 'cuda' else None


# =============================================================================
# HAM10000: a second training dataset for melanoma classifiers
# =============================================================================
def job_prep_ham(cfg):
    """HAM10000 data. Overlap protocol (fixed before the run): (1) remove images whose names occur in ISIC 2020;
    (2) remove images whose decoded 128-pixel arrays are identical (MD5) to an image of the ISIC 2020 classifier data;
    (3) split by lesion_id (no lesion in two splits); (4) count exact duplicates inside HAM10000 that belong to different
    lesions and whether they cross splits. Near-duplicates are screened (reported, not removed) by job ham_dupes."""
    import pandas as pd, hashlib
    from sklearn.model_selection import StratifiedGroupKFold
    H = cfg['ham']; IS = cfg['image_size']
    md = pd.read_csv(H['meta'])
    idx = {}
    for d in H['image_dirs']:
        for f in os.listdir(d):
            if f.lower().endswith('.jpg'): idx.setdefault(f[:-4], os.path.join(d, f))
    n_meta = int(len(md))
    md = md[md['image_id'].isin(idx)].drop_duplicates('image_id').reset_index(drop=True)
    n_found = int(len(md))
    isic = set(pd.read_csv(cfg['paths']['pseudo_csv'])['image_name'].astype(str))
    overlap = md['image_id'].isin(isic)
    n_over = int(overlap.sum()); removed = list(md.loc[overlap, 'image_id']); md = md[~overlap].reset_index(drop=True)
    if H.get('max_n'): md = md.sample(n=min(len(md), H['max_n']), random_state=2027).reset_index(drop=True)
    X144, X128 = _cache([idx[i] for i in md['image_id']], [IS + IS // 8, IS], cfg['cpu_workers_io'], draft=True)
    hsh = np.array([hashlib.md5(X128[i].numpy().tobytes()).hexdigest() for i in range(len(md))])
    cls = torch.load(P(cfg, 'cls', 'data.pt'), map_location='cpu', weights_only=False)
    ih = set(hashlib.md5(cls['X128'][i].numpy().tobytes()).hexdigest() for i in range(len(cls['X128']))); del cls
    same = np.array([h in ih for h in hsh]); n_hash = int(same.sum())
    removed += list(md.loc[same, 'image_id']); keep = ~same
    md = md[keep].reset_index(drop=True); X144, X128, hsh = X144[torch.from_numpy(keep)], X128[torch.from_numpy(keep)], hsh[keep]
    y = (md['dx'] == 'mel').astype(np.float32).values
    groups = md['lesion_id'].astype(str).values
    fold = np.zeros(len(md), int)
    for k, (_, te) in enumerate(StratifiedGroupKFold(n_splits=20, shuffle=True, random_state=2027).split(np.zeros(len(md)), y, groups)):
        fold[te] = k
    split = np.where(fold < 3, 'test', np.where(fold < 6, 'val', 'train'))
    lesion_split = pd.DataFrame({'l': groups, 's': split}).groupby('l')['s'].nunique()
    dd = pd.DataFrame({'h': hsh, 'l': groups, 's': split, 'id': md['image_id']})
    multi = dd.groupby('h').filter(lambda g: g['l'].nunique() > 1)
    cross = multi.groupby('h').filter(lambda g: g['s'].nunique() > 1)
    tr = split == 'train'
    atomic_torch_save({'X144_train': X144[torch.from_numpy(tr)], 'X128': X128, 'split': split.tolist(), 'target': y,
                       'concepts': np.zeros((len(md), 7), np.float32), 'image_name': list(md['image_id']),
                       'group': list(groups), 'have_pid': False}, P(cfg, 'ham', 'data.pt'))
    info = {'n_metadata_rows': n_meta, 'n_images_found': n_found, 'n_removed_name_in_isic2020': n_over,
            'n_removed_identical_pixels_isic2020_classifier_data': n_hash, 'n': int(len(md)), 'n_mel': int(y.sum()),
            'lesions': int(len(set(groups))), 'lesions_in_more_than_one_split': int((lesion_split > 1).sum()),
            'identical_images_with_different_lesion_ids': int(len(multi)), 'of_these_crossing_splits': int(len(cross)),
            'split_counts': {k: int((split == k).sum()) for k in ['train', 'val', 'test']},
            'split_mel': {k: int(y[split == k].sum()) for k in ['train', 'val', 'test']},
            'split_lesions': {k: int(len(set(groups[split == k]))) for k in ['train', 'val', 'test']},
            'dx_counts': md['dx'].value_counts().to_dict(), 'removed_image_ids': removed,
            'identical_different_lesion_ids': multi.sort_values('h')[['id', 'l', 's']].values.tolist()}
    save_json(info, P(cfg, 'results', 'ham_data.json'))
    log(f"prep_ham: {({k: v for k, v in info.items() if not isinstance(v, list)})}")


@torch.no_grad()
def job_ham_dupes(cfg, rank=0, world=1):
    """Near-duplicate screen (report only, fixed before the run): cosine similarity of ImageNet ResNet-50 features
    (128-pixel images) from every HAM10000 image to its nearest ISIC 2020 classifier image, and to its nearest HAM10000
    image of a different lesion in a different split. Counts at 0.95, 0.98, 0.99; the 50 closest pairs are saved and the
    12 closest HAM-ISIC pairs are drawn for visual review."""
    if rank != 0: return
    import pandas as pd
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    dev = get_device()
    net = Clf('resnet50', cfg['pretrained'], cfg['eval_size']).to(dev).eval()
    def feats(X):
        out = []
        for s in range(0, len(X), 256):
            f = net.body(u8_to_eval(X[s:s + 256], dev, cfg['eval_size'])).float(); out.append(torch.nn.functional.normalize(f, dim=1).cpu())
        return torch.cat(out)
    hd = torch.load(P(cfg, 'ham', 'data.pt'), map_location='cpu', weights_only=False)
    cd = torch.load(P(cfg, 'cls', 'data.pt'), map_location='cpu', weights_only=False)
    Fh, Fc = feats(hd['X128']), feats(cd['X128'])
    best, arg = [], []
    for s in range(0, len(Fh), 1024):
        S = Fh[s:s + 1024].to(dev) @ Fc.to(dev).T; v, i = S.max(1); best.append(v.cpu()); arg.append(i.cpu())
    best, arg = torch.cat(best).numpy(), torch.cat(arg).numpy()
    les = np.array(hd['group']); spl = np.array(hd['split'])
    wb, wa = [], []
    for s in range(0, len(Fh), 1024):
        S = Fh[s:s + 1024].to(dev) @ Fh.to(dev).T
        rows = np.arange(s, min(s + 1024, len(Fh)))
        bad = torch.from_numpy((les[rows][:, None] == les[None, :]) | (spl[rows][:, None] == spl[None, :])).to(dev)
        S[bad] = -1.0; v, i = S.max(1); wb.append(v.cpu()); wa.append(i.cpu())
    wb, wa = torch.cat(wb).numpy(), torch.cat(wa).numpy()
    summ = {f'HAM-ISIC >= {t}': int((best >= t).sum()) for t in [0.95, 0.98, 0.99]}
    summ.update({f'HAM-HAM (other lesion, other split) >= {t}': int((wb >= t).sum()) for t in [0.95, 0.98, 0.99]})
    summ['n_ham'] = int(len(Fh)); summ['n_isic'] = int(len(Fc)); summ['note'] = 'report only; no image was removed by this screen'
    save_json(summ, P(cfg, 'results', 'ham_near_duplicates.json'))
    o = np.argsort(-best)[:50]
    pd.DataFrame({'ham_image': np.array(hd['image_name'])[o], 'ham_split': spl[o], 'isic_image': np.array(cd['image_name'])[arg[o]],
                  'cosine': best[o]}).to_csv(P(cfg, 'results', 'ham_near_duplicates_isic_top50.csv'), index=False)
    o2 = np.argsort(-wb)[:50]
    pd.DataFrame({'ham_image': np.array(hd['image_name'])[o2], 'split': spl[o2], 'other_image': np.array(hd['image_name'])[wa[o2]],
                  'other_split': spl[wa[o2]], 'cosine': wb[o2]}).to_csv(P(cfg, 'results', 'ham_near_duplicates_within_top50.csv'), index=False)
    f, axs = plt.subplots(2, 12, figsize=(14, 2.8))
    for c, k in enumerate(o[:12]):
        axs[0, c].imshow(hd['X128'][k].permute(1, 2, 0).numpy()); axs[1, c].imshow(cd['X128'][arg[k]].permute(1, 2, 0).numpy())
        axs[0, c].set_title(f'{best[k]:.3f}', fontsize=7)
        for a in axs[:, c]: a.axis('off')
    axs[0, 0].text(-0.3, 0.5, 'HAM10000', transform=axs[0, 0].transAxes, rotation=90, va='center', fontsize=7)
    axs[1, 0].text(-0.3, 0.5, 'ISIC 2020', transform=axs[1, 0].transAxes, rotation=90, va='center', fontsize=7)
    f.savefig(P(cfg, 'results', 'ham_near_duplicates_top12.png'), dpi=150, bbox_inches='tight'); plt.close(f)
    log(f'ham_dupes: {summ}')


def job_train_ham(cfg, rank=0, world=1):
    dev = get_device(); torch.backends.cudnn.benchmark = True
    mine = [m for m in cfg['ham_models'] if m['gpu'] % world == rank]
    data = torch.load(P(cfg, 'ham', 'data.pt'), map_location='cpu', weights_only=False)
    log(f"[train_ham {rank}/{world}] {dev}: {[m['name'] for m in mine]}")
    for spec in mine:
        if os.path.exists(P(cfg, 'models', f"{spec['name']}.pt")):
            log(f"  {spec['name']}: done earlier"); continue
        train_one(cfg, {**spec, 'data': 'isic2020'}, data, dev)   # same code path; the data object is HAM10000
        r = json.load(open(P(cfg, 'results', f"train_{spec['name']}.json")))
        r['spec']['data'] = 'ham10000'; r['note'] = 'test_auc_melanoma, val_auc_melanoma and tau90 refer to the HAM10000 split'
        save_json(r, P(cfg, 'results', f"train_{spec['name']}.json"))


# =============================================================================
# PH2: independent expert dermoscopic criteria for the concept evaluators
# =============================================================================
PH2_MAP = [   # frozen before the run: PH2 criterion -> Derm7pt-scored CBM-B output
    ('Pigment network', 'AT vs T', 'CBM-B PN (Derm7pt: typical 0.5, atypical 1)', 'mapped; only typical vs atypical (PH2 has no absent class)'),
    ('Dots/globules', 'T or AT vs A; AT vs A or T', 'CBM-B D/G (Derm7pt: regular 0.5, irregular 1)', 'mapped; PH2 typical/atypical taken as Derm7pt regular/irregular'),
    ('Streaks', 'P vs A', 'CBM-B Str', 'mapped (presence only; PH2 does not grade regular/irregular)'),
    ('Regression areas', 'P vs A', 'CBM-B Reg', 'mapped'),
    ('Blue-whitish veil', 'P vs A', 'CBM-B BWV', 'mapped'),
    ('Vascular structures', '-', 'CBM-B VS', 'not testable: not a PH2 criterion'),
    ('Pigmentation (blotches)', '-', 'CBM-B Pig', 'not testable: not a PH2 criterion (PH2 colours are not blotches)'),
    ('Colours', '-', '-', 'not mapped: no Derm7pt concept')]
PH2_COLS = {'pigment network': 'PN', 'dots/globules': 'DG', 'streaks': 'Str', 'regression areas': 'Reg',
            'blue-whitish veil': 'BWV', 'clinical diagnosis': 'Dx'}


def _ph2_table(path):
    """PH2_dataset.txt: rows of '||'-separated fields; the header row names the criteria. Returns a DataFrame or raises."""
    import pandas as pd
    rows = []
    for line in open(path, encoding='latin-1'):
        if '||' not in line: continue
        # PH2_dataset.txt separates most fields by '||' but the six dermoscopic criteria by single '|'
        f = [x.strip() for x in line.strip().strip('|').replace('||', '|').split('|')]
        rows.append(f)
    hi = next(i for i, r in enumerate(rows) if any('pigment network' in x.lower() for x in r))
    head = [x.lower() for x in rows[hi]]
    col = {}
    for i, h in enumerate(head):
        if h.startswith('name') or 'image name' in h: col['name'] = i
        for k, v in PH2_COLS.items():
            if h.startswith(k): col[v] = i
    need = ['name', 'PN', 'DG', 'Str', 'Reg', 'BWV']
    if any(k not in col for k in need): raise ValueError(f'PH2 header not recognised: {rows[hi]}')
    out = []
    for r in rows[hi + 1:]:
        if len(r) <= max(col.values()) or not r[col['name']].upper().startswith('IMD'): continue
        out.append({k: r[i] for k, i in col.items()})
    df = pd.DataFrame(out)
    if len(df) < 20: raise ValueError(f'only {len(df)} PH2 rows parsed')
    return df


def job_ph2(cfg):
    import pandas as pd
    from sklearn.metrics import roc_auc_score
    dev = get_device(); Q = cfg['ph2']; res = {'status': 'not run'}
    try:
        df = _ph2_table(Q['txt'])
    except Exception as e:
        lines = open(Q['txt'], encoding='latin-1').read().splitlines()[:25]
        save_json({'status': f'parse failed: {e}', 'first_lines': lines}, P(cfg, 'results', 'ph2_results.json'))
        log(f'ph2: could not parse {Q["txt"]}: {e}'); return
    imgs = {}
    for dp, dn, fn in os.walk(Q['root']):
        for f in fn:
            if f.lower().endswith('.bmp') and f[:-4].upper().startswith('IMD') and '_' not in f: imgs[f[:-4].upper()] = os.path.join(dp, f)
    df['name'] = df['name'].str.upper(); df = df[df['name'].isin(imgs)].reset_index(drop=True)
    X128, = _cache([imgs[n] for n in df['name']], [cfg['image_size']], cfg['cpu_workers_io'])
    nb = ConceptNetB(pretrained=False)
    nb.load_state_dict(torch.load(cfg['paths']['cbmb'], map_location='cpu', weights_only=False)['state_dict']); nb = nb.to(dev).eval()
    ni = None
    if cfg['paths'].get('isic18') and os.path.exists(cfg['paths']['isic18']):
        ni = AttrNet(pretrained=False)
        ni.load_state_dict(torch.load(cfg['paths']['isic18'], map_location='cpu', weights_only=False)['state_dict']); ni = ni.to(dev).eval()
    B, I = [], []
    with torch.no_grad():
        for s in range(0, len(X128), 32):
            xe = u8_to_eval(X128[s:s + 32], dev, cfg['eval_size'])
            B.append(torch.sigmoid(nb(xe)).cpu().numpy())
            if ni is not None: I.append(torch.sigmoid(ni(xe)).cpu().numpy())
    B = np.concatenate(B); I = np.concatenate(I) if I else None
    up = lambda c: df[c].astype(str).str.upper().str.strip()
    tests = [('Pigment network atypical (AT vs T)', up('PN') == 'AT', up('PN').isin(['AT', 'T']), B[:, 0], 'CBM-B PN'),
             ('Dots/globules present (T or AT vs A)', up('DG').isin(['T', 'AT']), up('DG').isin(['A', 'T', 'AT']), B[:, 5], 'CBM-B D/G'),
             ('Dots/globules atypical (AT vs A or T)', up('DG') == 'AT', up('DG').isin(['A', 'T', 'AT']), B[:, 5], 'CBM-B D/G'),
             ('Streaks present', up('Str') == 'P', up('Str').isin(['A', 'P']), B[:, 4], 'CBM-B Str'),
             ('Regression areas present', up('Reg') == 'P', up('Reg').isin(['A', 'P']), B[:, 6], 'CBM-B Reg'),
             ('Blue-whitish veil present', up('BWV') == 'P', up('BWV').isin(['A', 'P']), B[:, 1], 'CBM-B BWV')]
    if I is not None:
        tests += [('Dots/globules present (T or AT vs A)', up('DG').isin(['T', 'AT']), up('DG').isin(['A', 'T', 'AT']), I[:, 4], 'ISIC 2018 globules'),
                  ('Streaks present', up('Str') == 'P', up('Str').isin(['A', 'P']), I[:, 2], 'ISIC 2018 streaks')]
    rng = np.random.default_rng(2036); rows = []
    for name, pos, ok, score, ev in tests:
        y = pos[ok].values.astype(int); sc = score[ok.values]
        if len(np.unique(y)) < 2: rows.append({'criterion': name, 'evaluator': ev, 'n': int(ok.sum()), 'n positive': int(y.sum()), 'AUC': np.nan}); continue
        auc = roc_auc_score(y, sc); bs = []
        for _ in range(2000):
            ii = rng.integers(0, len(y), len(y))
            if len(np.unique(y[ii])) == 2: bs.append(roc_auc_score(y[ii], sc[ii]))
        rows.append({'criterion': name, 'evaluator': ev, 'n': int(ok.sum()), 'n positive': int(y.sum()), 'AUC': auc,
                     'lo': np.percentile(bs, 2.5), 'hi': np.percentile(bs, 97.5)})
    pd.DataFrame(rows).to_csv(P(cfg, 'results', 'ph2_evaluators.csv'), index=False)
    pd.DataFrame(PH2_MAP, columns=['PH2 criterion', 'comparison', 'evaluator output', 'mapping']).to_csv(P(cfg, 'results', 'ph2_mapping.csv'), index=False)
    # melanoma classifiers on PH2: melanoma (clinical diagnosis 2) against common and atypical nevi
    dx = pd.to_numeric(df['Dx'], errors='coerce').values if 'Dx' in df else np.full(len(df), np.nan); ymel = (dx == 2).astype(int); okd = np.isfinite(dx)
    crow = []
    for spec in cfg['ph2_models']:
        path = next((os.path.join(d, f"{spec['name']}.pt") for d in cfg['models_dirs'] if os.path.exists(os.path.join(d, f"{spec['name']}.pt"))), None)
        if path is None: continue
        ck = torch.load(path, map_location='cpu', weights_only=False)
        net = Clf(ck['spec']['arch'], False, cfg['eval_size']); net.load_state_dict(ck['state_dict']); net = net.to(dev).eval()
        lg = clf_logits(net, X128, cfg, dev, amp=False)
        if len(np.unique(ymel[okd])) == 2:
            crow.append({'classifier': spec['name'], 'PH2 AUC (melanoma vs nevi)': roc_auc_score(ymel[okd], lg[okd])})
        del net; torch.cuda.empty_cache() if dev.type == 'cuda' else None
    pd.DataFrame(crow).to_csv(P(cfg, 'results', 'ph2_classifiers.csv'), index=False)
    np.savez(P(cfg, 'results', 'ph2_scores.npz'), cbmb=B, isic18=I if I is not None else np.zeros((len(B), 0)), names=np.array(df['name']))
    save_json({'status': 'ok', 'n_images': int(len(df)), 'n_melanoma': int(ymel.sum()), 'n_classifiers': len(crow)},
              P(cfg, 'results', 'ph2_results.json'))
    log(f'ph2: {len(df)} images, {len(rows)} criterion tests, {len(crow)} classifiers')


JOBS = {'generic': job_generic, 'prep_ham': job_prep_ham, 'ham_dupes': job_ham_dupes, 'train_ham': job_train_ham, 'ph2': job_ph2}
SHARDED = ('generic', 'train_ham', 'ham_dupes')

if __name__ == '__main__':
    job, cfg_path = sys.argv[1], sys.argv[2]
    rank = int(sys.argv[3]) if len(sys.argv) > 3 else 0; world = int(sys.argv[4]) if len(sys.argv) > 4 else 1
    cfg = load_cfg(cfg_path)
    rid = os.environ.get('P3_RUN_ID')
    if rid:
        try:
            cur = open(os.path.join(cfg['work'], 'run_id.txt')).read().strip()
        except Exception:
            cur = rid
        if cur != rid:
            print(f'job {job}: started by an older run of the cell; exit without work', flush=True); sys.exit(3)
    tag = f'{job}_{rank}' if job in SHARDED else job
    log(f"job {tag} start | device={get_device()} | torch {torch.__version__} | CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")
    t0 = time.time()
    try:
        JOBS[job](cfg, rank, world) if job in SHARDED else JOBS[job](cfg)
    except Exception:
        traceback.print_exc(); sys.exit(1)
    with open(P(cfg, 'done', f'{tag}.ok'), 'w') as f:
        f.write(f"{time.time() - t0:.0f}s\n")
    log(f"job {tag} done in {(time.time() - t0) / 60:.1f} min")
''',
    'analysis3.py': r'''"""ClinDiffXAI Phase 3 analysis: empirical null distribution and retraining variability of the concept audit.

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
''',
}
for fn, src in FILES.items():
    with open(os.path.join(CODE, fn), 'w') as f:
        f.write(src)
if cfg2.get('pretrained') and not DRY_RUN:
    import torchvision.models as tvm
    tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V2); tvm.vit_b_16(weights=tvm.ViT_B_16_Weights.IMAGENET1K_V1)
    tvm.efficientnet_b0(weights=tvm.EfficientNet_B0_Weights.IMAGENET1K_V1)

# ---------------------------------------------------------------- 3. config
OLD = copy.deepcopy(cfg2['models'])
NEW = []
NEW += [{'name': f'ctrl_perm_resnet50_s{s}', 'arch': 'resnet50', 'label': 'perm', 'seed': s, 'data': 'isic2020'} for s in range(2, 11)]
NEW += [{'name': f'ctrl_perm_effb0_s{s}', 'arch': 'effb0', 'label': 'perm', 'seed': s, 'data': 'isic2020'} for s in range(1, 6)]
NEW += [{'name': f'ctrl_perm_vitb16_s{s}', 'arch': 'vitb16', 'label': 'perm', 'seed': s, 'data': 'isic2020'} for s in range(1, 4)]
for a in ['resnet50', 'effb0', 'vitb16']:
    NEW += [{'name': f'mel_{a}_s{s}', 'arch': a, 'label': 'mel', 'seed': s, 'data': 'isic2020'} for s in range(3, 6)]
if DRY_RUN:
    NEW = [m for m in NEW if m['name'] in ('ctrl_perm_resnet50_s2', 'ctrl_perm_effb0_s1', 'ctrl_perm_vitb16_s1', 'mel_resnet50_s3', 'mel_effb0_s3')]
HAMM = [{'name': f'mel_ham_{a}_s{s}', 'arch': a, 'label': 'mel', 'seed': s, 'data': 'ham10000'} for a in ['resnet50', 'effb0', 'vitb16'] for s in [1, 2]]
if DRY_RUN:
    HAMM = [m for m in HAMM if m['name'] in ('mel_ham_resnet50_s1', 'mel_ham_effb0_s1')]
load = [0.0] * len(SHARDS)
for m in sorted(HAMM, key=lambda m: -{'resnet50': 2.0, 'effb0': 1.2, 'vitb16': 3.0}[m['arch']]):
    g = load.index(min(load)); m['gpu'] = g; load[g] += {'resnet50': 2.0, 'effb0': 1.2, 'vitb16': 3.0}[m['arch']]
names = [m['name'] for m in OLD + NEW + HAMM]
assert len(names) == len(set(names)), 'duplicate classifier names'
COSTN = {'resnet50': 2.0, 'effb0': 1.2, 'vitb16': 3.0}; load = [0.0] * len(SHARDS)
for m in sorted(NEW, key=lambda m: -COSTN[m['arch']] * (2.5 if m['label'] == 'mel' else 1.0)):
    g = load.index(min(load)); m['gpu'] = g; load[g] += COSTN[m['arch']] * (2.5 if m['label'] == 'mel' else 1.0)
cfg = copy.deepcopy(cfg2)
cfg.update({'work': WA, 'work1d': W1D, 'outputs': os.path.join(WA, 'outputs'), 'paths': paths, 'models': OLD + NEW,
            'hardware': HW, 'cpu_workers': max(1, ncpu), 'cpu_workers_io': max(2, ncpu), 'scores_dir': 'scores2_fp32',
            'dry_run': DRY_RUN, 'inputs': {'gen_dir': GEN, 'ainp_dir': AINP, 'p2c_dir': None},
            'generic': {'archs': ['resnet50', 'effb0', 'vitb16'], 'n_heads': 20 if DRY_RUN else 100, 'seed': 4040},
            'ham': {'meta': ham_meta, 'image_dirs': ham_dirs, 'max_n': 300 if DRY_RUN else None}, 'ham_models': HAMM,
            'ph2': PH2, 'ph2_models': [m for m in OLD + NEW + HAMM if m['label'] in ('mel', 'perm')],
            'models_dirs': [os.path.join(WA, 'models')] + ([M2] if M2 else []),
            'rule_3': {'text': RULE_3, 'sha256': RULE_3_SHA}})
CFG = os.path.join(WA, 'cfg.json')
with open(CFG, 'w') as f: json.dump(cfg, f, indent=2)
cfg_new = copy.deepcopy(cfg); cfg_new['models'] = NEW
CFG_NEW = os.path.join(WA, 'cfg_new.json')
with open(CFG_NEW, 'w') as f: json.dump(cfg_new, f, indent=2)
cfg_score = copy.deepcopy(cfg); cfg_score['models'] = NEW + HAMM   # scoring: ISIC 2020 and HAM10000 classifiers
CFG_SCORE = os.path.join(WA, 'cfg_score.json')
with open(CFG_SCORE, 'w') as f: json.dump(cfg_score, f, indent=2)


def group(m):
    if m['label'] == 'perm': return 'null_trained'
    if m['data'] == 'ham10000': return 'mel_ham'
    if m['label'] == 'mel' and m['data'] == 'isic2020':
        return 'mel_retrain' if m in NEW else 'mel_primary'
    return 'other'


PRIMARY = [m['name'] for m in OLD if m['label'] == 'mel' and m['data'] == 'isic2020']
cfg3 = {'models': [{**m, 'group': group(m)} for m in OLD + NEW + HAMM], 'scores_dir': os.path.join(WA, 'scores2_fp32'),
        'results_dir': os.path.join(WA, 'results'), 'meta_dir': GEN, 'eval_dirs': [os.path.join(WA, 'scores'), os.path.join(W1D, 'scores')],
        'cls_labels': os.path.join(AINP, 'phase2', 'cls', 'cls_labels.npz'), 'generic_dir': os.path.join(WA, 'generic'),
        'out': os.path.join(WA, 'outputs3'), 'rule_text': RULE_3, 'primary': PRIMARY, 'n_boot': 2000}
CFG3 = os.path.join(WA, 'cfg_analysis3.json')
with open(CFG3, 'w') as f: json.dump(cfg3, f, indent=2)
print('  primary melanoma classifiers: ' + ', '.join(PRIMARY))
print('  new classifiers: ' + '; '.join(f"GPU{m['gpu']}: {m['name']}" for m in sorted(NEW, key=lambda m: m['gpu'])))

# ---------------------------------------------------------------- 4. stages
STATUS = {}


def last_line(log):
    if not os.path.exists(log): return ''
    with open(log, 'rb') as f:
        f.seek(0, 2); n = f.tell(); f.seek(max(0, n - 6000))
        lines = [l for l in f.read().decode('utf8', 'ignore').splitlines() if l.strip() and 'WARNING' not in l and 'Warning' not in l]
    return lines[-1][-120:] if lines else ''


def start(jobs, script, cfg_path, work, extra_env=None, analysis=False):
    procs = []
    for job, shards in jobs:
        tags = [job] if shards is None else [f'{job}_{r}' for _, r in shards]
        key = f'{os.path.basename(work)}:{job}'
        if not analysis and all(os.path.exists(os.path.join(work, 'done', f'{t}.ok')) for t in tags):
            STATUS[key] = 'done earlier'; say(f'{key}: done earlier, skipped'); continue
        for (vis, rank), tag in zip(shards or [(None, 0)], tags):
            env = {**os.environ, 'PYTHONUNBUFFERED': '1', 'P2_RUN_ID': RUN_ID, 'P2D_RUN_ID': RUN_ID, 'P3_RUN_ID': RUN_ID, **(extra_env or {})}
            if shards is not None: env['CUDA_VISIBLE_DEVICES'] = vis
            args = [sys.executable, '-u', os.path.join(CODE, script)] + \
                   ([cfg_path] if analysis else [job, cfg_path] + ([str(rank), str(len(shards))] if shards else []))
            lp = os.path.join(WORK, 'logs', f'{os.path.basename(work)}_{tag}.log'); lf = open(lp, 'a')
            procs.append((key, lp, subprocess.Popen(args, stdout=lf, stderr=subprocess.STDOUT, env=env, cwd=CODE, start_new_session=True), lf))
    return procs


def wait(title, procs):
    if not procs: return
    say(f'===== {title} ====='); t0 = time.time()
    try:
        tick = 0
        while any(p.poll() is None for _, _, p, _ in procs):
            time.sleep(5); tick += 5
            if tick % 60 == 0:
                print(f"    {title}: {(time.time() - t0) / 60:.1f} min", flush=True)
                for _, lp, p, _ in procs:
                    if p.poll() is None: print(f"      {os.path.basename(lp)}: {last_line(lp)}", flush=True)
    except KeyboardInterrupt:
        for _, _, p, _ in procs:
            try: os.killpg(p.pid, signal.SIGTERM)
            except Exception: pass
        time.sleep(3)
        for _, _, p, _ in procs:
            try: os.killpg(p.pid, signal.SIGKILL)
            except Exception: pass
        raise
    for _, _, _, lf in procs: lf.close()
    bad = [(lp, p.returncode) for _, lp, p, _ in procs if p.returncode != 0]
    for key in {k for k, _, _, _ in procs}:
        STATUS[key] = 'ok' if not [1 for k, lp, p, _ in procs if k == key and p.returncode != 0] else 'FAILED'
    say(f"{title}: {'ok' if not bad else 'FAILED ' + str([os.path.basename(b[0]) for b in bad])} after {(time.time() - t0) / 60:.1f} min")
    if bad:
        for lp, _ in bad:
            print(f'---- last lines of {os.path.basename(lp)} ----'); os.system(f'tail -n 40 {lp}')
        raise RuntimeError(f'{title} failed; see the logs in {os.path.join(WORK, "logs")}')


def run(title, jobs, script, cfg_path, work, **kw):
    wait(title, start(jobs, script, cfg_path, work, **kw))


FAILED_OPTIONAL = []


def run_optional(title, *a, **kw):
    """a failed optional stage is recorded and kept in the outputs; the run continues"""
    try:
        run(title, *a, **kw)
    except RuntimeError as e:
        FAILED_OPTIONAL.append(f'{title}: {e}'); say(f'OPTIONAL STAGE FAILED (recorded, run continues): {title}')



# Phase 3 classifiers in the inputs (the Phase 3 notebook output), for PH2 only
p3m = walk(INPUT_ROOT, lambda d, f: f == 'mel_ham_resnet50_s1.pt')
if not p3m:
    z = walk(INPUT_ROOT, lambda d, f: f == 'phase3_models.zip')
    if z:
        say('unzipping phase3_models.zip ...')
        with zipfile.ZipFile(z[0]) as zz: zz.extractall(os.path.join(UNZ, 'm3'))
        p3m = walk(os.path.join(UNZ, 'm3'), lambda d, f: f == 'mel_ham_resnet50_s1.pt')
if p3m:
    cfg['models_dirs'] = cfg['models_dirs'] + [os.path.dirname(p3m[0])]
    with open(CFG, 'w') as f: json.dump(cfg, f, indent=2)
say(f"classifier folders for PH2: {cfg['models_dirs']}")
T0 = time.time()
run('classifier data again from ISIC 2020 (CPU)', [('prep_cls', None)], 'worker2.py', CFG, WA)
run('check: classifier data equal to Phase 2', [('verify_cls', None)], 'worker2d.py', CFG, WA)
run('HAM10000 data again, with the overlap checks (CPU)', [('prep_ham', None)], 'worker3.py', CFG, WA)
H = json.load(open(os.path.join(WA, 'results', 'ham_data.json')))
P3 = {'n': 10015, 'split_counts': {'train': 7040, 'val': 1530, 'test': 1445}}   # Phase 3 ham_data.json
same_split = (H['n'] == P3['n'] and H['split_counts'] == P3['split_counts'] and H['n_removed_identical_pixels_isic2020_classifier_data'] == 0)
say(f"HAM10000 split equal to Phase 3: {same_split}; identical to ISIC 2020 classifier images: "
    f"{H['n_removed_identical_pixels_isic2020_classifier_data']}; identical images with different lesion ids: "
    f"{H['identical_images_with_different_lesion_ids']} (crossing splits: {H['of_these_crossing_splits']})")
run_optional('HAM10000 near-duplicate screen (GPU)', [('ham_dupes', SHARDS)], 'worker3.py', CFG, WA)
if PH2:
    run_optional('PH2: evaluators against expert criteria; classifiers on PH2 (GPU)', [('ph2', None)], 'worker3.py', CFG, WA)
else:
    say('PH2 not attached: rule 8 is not run')

PKG = os.path.join(WORK, 'package'); shutil.rmtree(PKG, ignore_errors=True); os.makedirs(os.path.join(PKG, 'logs'))
for f in glob.glob(os.path.join(WA, 'results', 'ham_*')) + glob.glob(os.path.join(WA, 'results', 'ph2_*')) + \
        [os.path.join(WA, 'results', 'cls_rebuild_check.json')]:
    if os.path.exists(f): shutil.copy(f, PKG)
for lg in glob.glob(os.path.join(WORK, 'logs', '*.log')): shutil.copy(lg, os.path.join(PKG, 'logs'))
for fn in FILES: shutil.copy(os.path.join(CODE, fn), os.path.join(PKG, 'code_' + fn))
with open(os.path.join(PKG, 'phase3b_summary.json'), 'w') as f:
    json.dump({'ham_split_equal_to_phase3': bool(same_split), 'stages': STATUS, 'failed_optional_stages': FAILED_OPTIONAL,
               'minutes': (time.time() - T0) / 60, 'hardware': HW}, f, indent=2)
zip_base = os.path.join(WORK_ROOT, 'phase3b_dryrun_outputs' if DRY_RUN else 'phase3b_outputs')
shutil.make_archive(zip_base, 'zip', PKG)
print('\n' + '=' * 78); print(f"PHASE 3b {'DRY RUN ' if DRY_RUN else ''}FINISHED in {(time.time() - T0) / 60:.1f} min on {HW}")
for j, s in STATUS.items(): print(f"  {j:<28} {s}")
for x in FAILED_OPTIONAL: print(f"  OPTIONAL STAGE FAILED: {x}")
for f in ['ham_near_duplicates.json', 'ph2_results.json', 'ph2_evaluators.csv', 'ph2_classifiers.csv']:
    p = os.path.join(WA, 'results', f)
    if os.path.exists(p): print(f'---- {f}'); print(open(p).read()[:3000])
print(f"\nDownload: {zip_base}.zip")
