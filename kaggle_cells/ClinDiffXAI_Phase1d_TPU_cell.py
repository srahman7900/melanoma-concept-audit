# =============================================================================
# ClinDiffXAI  PHASE 1d: final evaluation with colour-calibrated sampling (V2)
#                                        (one Kaggle cell, Accelerator = TPU)
# =============================================================================
# WHAT THIS CELL DOES
#   Phase 1c chose V2 with the rule fixed before that run. Phase 1d makes the final
#   paper numbers with V2. Every image and every score is made in this run on
#   the TPU, so all final numbers come from one hardware pipeline.
#   1. Real anchor images (same code and seeds as Phase 1b).
#   2. V2 images (CFG s = 3.5, DDIM 50 steps, same noise seeds as Phase 1):
#      - evaluation sets: 1,000 per class for the main model (prototypes), the
#        baseline, and the main model with real concept vectors;
#      - control: 500 per class with CLASS-AGNOSTIC colour targets, to measure how
#        much class signal the colour targets add;
#      - dose-response sweeps: 7 concepts x 11 levels x 32 trajectories, from the
#        melanoma and the benign prototype;
#      - guidance-scale sweep: 9 scales x 2 classes x 128.
#      Colour targets come from a 3-D Gaussian fitted to the mean colours of the
#      real images of each class. The noise-to-colour maps are the Phase 1c maps.
#   3. Scores: CBM-A, CBM-B, ISIC 2018 detector, Inception-v3 (all on the TPU).
#   4. The complete Phase 1 analysis on the V2 sets (tables T1 to T7, Gate 1,
#      all figures) plus the Phase 1d tables (TD1 to TD6).
#
# INPUTS TO ATTACH
#   The two epoch-150 checkpoints, CBM-A (best_cbm_model.pth), the pseudo-label
#   CSV, the ISIC 2020 images, and your private dataset with phase1b_evaluators
#   (cbmb.pth, isic18_detector.pth). Derm7pt is not needed: the evaluator
#   validity results from Phase 1b are built into this cell.
#   Settings: Accelerator TPU, Internet ON.
#
# HOW TO RUN
#   1. Set DRY_RUN = True and run the cell (a few images; my estimate 10 to 20
#      minutes, mostly compilation). It must end with "PHASE 1d DRY RUN FINISHED".
#   2. Set DRY_RUN = False and use "Save Version -> Save & Run All (Commit)", so
#      that the output stays after the session ends.
#   TIME: I cannot estimate the TPU speed well. The log prints the measured
#   images per second after the first chunks.
#   If you stop the cell, run it again: finished chunks are kept and skipped.
# =============================================================================

DRY_RUN       = False
FORCE_RERUN   = False
GEN_PRECISION = 'default'   # XLA matrix precision for generation: 'default', 'high' or 'highest'
GEN_BATCH     = 32          # images per TPU core per step (lower it only if the log shows an out-of-memory error)

KNOWN_PATHS = {   # from your Phase 1b / 1c runs; the cell searches by file name if these move
    'cbma':         '/kaggle/input/datasets/shohanurrahman7/best-cbm-model-128/best_cbm_model.pth',
    'main_e150':    '/kaggle/input/datasets/shohanoor7900/checkpoint-tpu-epoch150/checkpoint_tpu_epoch150.pth',
    'base_e150':    '/kaggle/input/datasets/shohanoor7900/checkpoint-baseline-tpu-epoch150/checkpoint_baseline_tpu_epoch150.pth',
    'pseudo_csv':   '/kaggle/input/datasets/shohanurrahman7/isic2020-pseudo-labels-v3/isic2020_pseudo_labels_v3.csv',
    'isic2020_dir': '/kaggle/input/datasets/sumaiyabinteshahid/isic-challenge-dataset-2020/ISIC_2020_Dataset/train',
}

import os, sys, json, time, glob, shutil, subprocess, uuid, base64, gzip, zipfile, signal, importlib

INPUT_ROOT = os.environ.get('P1_INPUT_ROOT', '/kaggle/input')
WORK_ROOT = os.environ.get('P1_WORK_ROOT', '/kaggle/working')
XLA_CPU_TEST = os.environ.get('P1D_XLA_CPU')            # offline test only: XLA on CPU devices
IMAGE_SIZE = int(os.environ.get('P1_IMAGE_SIZE', '128'))
EVAL_SIZE = int(os.environ.get('P1_EVAL_SIZE', '224'))
INCEP_SIZE = int(os.environ.get('P1_INCEP_SIZE', '299'))
PRETRAINED = os.environ.get('P1_NO_PRETRAINED') is None
DDIM_STEPS = int(os.environ.get('P1_DDIM_STEPS', '50'))
if os.environ.get('P1_INPUT_ROOT'):
    KNOWN_PATHS = {k: v.replace('/kaggle/input', INPUT_ROOT) for k, v in KNOWN_PATHS.items()}


def say(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


WORK = os.path.join(WORK_ROOT, 'phase1d_dryrun' if DRY_RUN else 'phase1d')
CODE = os.path.join(WORK, 'code'); OUTD = os.path.join(WORK, 'outputs'); WORKER = os.path.join(CODE, 'worker1d.py')
os.makedirs(WORK, exist_ok=True)
RUN_ID = uuid.uuid4().hex
with open(os.path.join(WORK, 'run_id.txt'), 'w') as f:
    f.write(RUN_ID)
for _ in range(2):   # stop worker processes of an interrupted earlier run (they would hold the TPU)
    try:
        if subprocess.run(['pkill', '-f', WORKER], check=False).returncode == 0:
            say('stopped worker processes of an earlier run of this cell')
    except FileNotFoundError:
        pass
    time.sleep(3)
if FORCE_RERUN:
    for d in ['done', 'sets', 'scores', 'results', 'outputs', 'logs', 'plan', 'chunks_gen', 'chunks_score', 'ref']:
        shutil.rmtree(os.path.join(WORK, d), ignore_errors=True)
for d in [CODE, OUTD, 'logs', 'done', 'results', 'ref', 'plan']:
    os.makedirs(d if os.path.isabs(d) else os.path.join(WORK, d), exist_ok=True)

# ---------------------------------------------------------------- 0. Python packages (CPU side)
for mod, pipname in [('torchvision', None), ('pandas', 'pandas'), ('scipy', 'scipy'), ('sklearn', 'scikit-learn'),
                     ('matplotlib', 'matplotlib'), ('PIL', 'pillow')]:
    try:
        importlib.import_module(mod)
    except Exception:
        if mod == 'torchvision':
            import torch
            mj, mn = torch.__version__.split('+')[0].split('.')[:2]
            pipname = f'torchvision==0.{int(mn) + 15}.*' if mj == '2' else 'torchvision'
        say(f'installing {pipname}')
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', pipname], check=False)

# ---------------------------------------------------------------- 1. inputs
say(f'indexing {INPUT_ROOT} ...')
NAME_INDEX = {}
for dp, dn, fn in os.walk(INPUT_ROOT):
    if len(fn) > 2000 and not any(f.endswith(('.pth', '.zip', '.csv')) for f in fn[:50]):
        continue   # skip big image folders (ISIC 2020)
    for f in fn:
        NAME_INDEX.setdefault(f, []).append(os.path.join(dp, f))


def by_name(name):
    hits = NAME_INDEX.get(name, [])
    return (hits[0] if hits else None), len(hits)


paths, notes = {}, []
for key, fname in [('cbma', 'best_cbm_model.pth'), ('main_e150', 'checkpoint_tpu_epoch150.pth'),
                   ('base_e150', 'checkpoint_baseline_tpu_epoch150.pth'), ('pseudo_csv', 'isic2020_pseudo_labels_v3.csv')]:
    if os.path.exists(KNOWN_PATHS[key]):
        paths[key] = KNOWN_PATHS[key]
    else:
        paths[key], n = by_name(fname)
        if paths[key]: notes.append(f"{key}: known path missing; using {paths[key]} ({n} candidate(s))")
paths['isic2020_dir'] = KNOWN_PATHS['isic2020_dir'] if os.path.isdir(KNOWN_PATHS['isic2020_dir']) else None
if not paths['isic2020_dir'] and paths.get('pseudo_csv'):
    import pandas as pd
    probe = pd.read_csv(paths['pseudo_csv'], nrows=3)['image_name'].iloc[0] + '.jpg'
    for dp, dn, fn in os.walk(INPUT_ROOT):
        if probe in fn:
            paths['isic2020_dir'] = dp; notes.append(f'isic2020_dir found at {dp}'); break
# Phase 1b evaluators: files, or the zip if Kaggle did not unpack it
EVD = os.path.join(WORK, 'evaluators'); os.makedirs(EVD, exist_ok=True)
for key, fname in [('cbmb', 'cbmb.pth'), ('isic18', 'isic18_detector.pth')]:
    p, n = by_name(fname)
    if p: paths[key] = p
if not paths.get('cbmb'):
    z, _ = by_name('phase1b_evaluators.zip')
    if z:
        with zipfile.ZipFile(z) as zf:
            zf.extractall(EVD)
        for key, fname in [('cbmb', 'cbmb.pth'), ('isic18', 'isic18_detector.pth')]:
            hits = glob.glob(os.path.join(EVD, '**', fname), recursive=True)
            if hits: paths[key] = hits[0]
print('\n' + '=' * 78 + '\nINPUTS\n' + '=' * 78)
critical = ['cbma', 'main_e150', 'base_e150', 'pseudo_csv', 'isic2020_dir', 'cbmb']
for k in critical + ['isic18']:
    print(f"  {'OK ' if paths.get(k) else '-- '} {k:<13} {paths.get(k) or 'NOT FOUND'}")
for n in notes: print('  note:', n)
missing = [k for k in critical if not paths.get(k)]
if missing:
    raise FileNotFoundError(f'Required inputs not found: {missing}. Attach the datasets (see the header of this cell).')
if not paths.get('isic18'):
    print('  WARNING: isic18_detector.pth not found. The ISIC 2018 check of Gate 1 (G1d) cannot run.')

# ---------------------------------------------------------------- 2. TPU check (in a subprocess: the notebook never holds the TPU)
XENV = os.environ.copy(); XENV['PYTHONUNBUFFERED'] = '1'; XENV['TF_CPP_MIN_LOG_LEVEL'] = '2'
if XLA_CPU_TEST:
    XENV['PJRT_DEVICE'] = 'CPU'; XENV['CPU_NUM_DEVICES'] = XLA_CPU_TEST
else:
    XENV['PJRT_DEVICE'] = 'TPU'
probe_code = (
    "import torch, torch_xla\n"
    "try:\n    import torch_xla.runtime as xr; n = xr.global_runtime_device_count(); t = xr.device_type()\n"
    "except Exception as e:\n    import torch_xla.core.xla_model as xm; n = len(xm.get_xla_supported_devices()); t = 'TPU?'\n"
    "print('XLAINFO', torch.__version__, torch_xla.__version__, t, n)\n")
pr = subprocess.run([sys.executable, '-c', probe_code], env=XENV, capture_output=True, text=True, timeout=900)
info = [l for l in pr.stdout.splitlines() if l.startswith('XLAINFO')]
if not info:
    print(pr.stdout[-3000:], pr.stderr[-3000:])
    raise RuntimeError('No TPU found through torch_xla. Set Accelerator = TPU. If a TPU is set, restart the session: '
                       'another process may hold the TPU.')
_, TORCH_V, XLA_V, DEV_TYPE, N_DEV = info[0].split()
HW = f"{DEV_TYPE} x{N_DEV} ({os.environ.get('TPU_ACCELERATOR_TYPE', 'type not reported')})"
say(f'torch {TORCH_V}, torch_xla {XLA_V}: {HW}')

# ---------------------------------------------------------------- 3. code and built-in reference files
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
    'worker1d.py': r'''"""ClinDiffXAI Phase 1d worker: final evaluation with colour-calibrated sampling (V2)
on TPU (PyTorch/XLA). Usage: python worker1d.py <job> <cfg.json>
Jobs: prep (CPU), gen (TPU), assemble (CPU), score (TPU), finalize (CPU)."""
import os, sys, json, time, glob, math, traceback
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *  # noqa

NEED_INCEP = ('eval_', 'real_', 'cfg_', 'pool_')


def P(cfg, *parts):
    p = os.path.join(cfg['work'], *parts)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    return p


def atomic_torch_save(obj, path):
    torch.save(obj, path + '.tmp'); os.replace(path + '.tmp', path)


def atomic_npz(path, **arrays):
    with open(path + '.tmp', 'wb') as fh:
        np.savez(fh, **arrays)
    os.replace(path + '.tmp', path)


# =============================================================================
# XLA helpers (imported only inside the TPU jobs)
# =============================================================================
class XLA:
    def __init__(self):
        import torch_xla
        import torch_xla.core.xla_model as xm
        self.tx, self.xm = torch_xla, xm
        try:
            import torch_xla.runtime as xr
        except Exception:
            xr = None
        self.xr = xr

    def device(self):
        return self.tx.device() if hasattr(self.tx, 'device') else self.xm.xla_device()

    def sync(self):
        if hasattr(self.tx, 'sync'):
            self.tx.sync()
        else:
            self.xm.mark_step()

    def rank(self):
        if self.xr is not None and hasattr(self.xr, 'global_ordinal'):
            return int(self.xr.global_ordinal())
        return int(self.xm.get_ordinal())

    def world(self):
        if self.xr is not None and hasattr(self.xr, 'world_size'):
            return int(self.xr.world_size())
        return int(self.xm.xrt_world_size())

    def set_precision(self, mode):
        if mode in (None, '', 'default'):
            return 'default'
        tries = []
        try:
            import torch_xla.backends as xb
            tries.append(lambda: xb.set_mat_mul_precision(mode))
        except Exception:
            pass
        tries.append(lambda: self.tx._XLAC._xla_set_mat_mul_precision(mode))
        if mode == 'highest':
            tries.append(lambda: self.tx._XLAC._xla_set_use_full_mat_mul_precision(use_full_mat_mul_precision=True))
        for f in tries:
            try:
                f(); return mode
            except Exception:
                continue
        return f'default (could not set {mode})'


def run_spawned(fn, cfg_path):
    """one replica per TPU device (xmp.spawn); single device if spawning is not possible"""
    import torch_xla.distributed.xla_multiprocessing as xmp
    try:
        xmp.spawn(fn, args=(cfg_path,))
    except Exception as e:
        log(f'xmp.spawn failed ({type(e).__name__}: {e}); running on one device instead')
        traceback.print_exc()
        fn(0, cfg_path, True)


def my_items(items, rank, world):
    return [(i, it) for i, it in enumerate(items) if i % world == rank]


# =============================================================================
# prep (CPU): real anchor sets, colour priors, generation plan
# =============================================================================
def job_prep(cfg):
    import pandas as pd
    from worker import job_prep_real            # Phase 1b code, unchanged
    if not (os.path.exists(P(cfg, 'sets', 'real_mel.pt')) and os.path.exists(P(cfg, 'sets', 'real_ben.pt'))):
        job_prep_real(cfg)
    IS = cfg['image_size']; S = cfg['sizes']; sd = cfg['seeds']
    real = {c: load_set(P(cfg, 'sets', f'real_{c}.pt'))['imgs'] for c in ['mel', 'ben']}
    rgb = {c: real[c].float().div(127.5).sub(1.0).mean((2, 3)).numpy().astype(np.float64) for c in real}
    lo = np.minimum(rgb['mel'].min(0), rgb['ben'].min(0)); hi = np.maximum(rgb['mel'].max(0), rgb['ben'].max(0))
    prior = {c: {'mean': rgb[c].mean(0).tolist(), 'cov': np.cov(rgb[c], rowvar=False).tolist(), 'n': len(rgb[c])} for c in rgb}
    save_json({'class_priors': prior, 'clip_lo': lo.tolist(), 'clip_hi': hi.tolist(),
               'note': 'per-image mean RGB in the model scale [-1, 1] of the real anchor images'},
              P(cfg, 'results', 'colour_prior.json'))

    def draw(cls, n, seed):
        rng = np.random.default_rng(seed)
        if cls == 'pooled':      # class-agnostic: 50/50 mixture of the two class Gaussians
            pick = rng.random(n) < 0.5
            a = rng.multivariate_normal(prior['mel']['mean'], prior['mel']['cov'], n)
            b = rng.multivariate_normal(prior['ben']['mean'], prior['ben']['cov'], n)
            t = np.where(pick[:, None], a, b)
        else:
            t = rng.multivariate_normal(prior[cls]['mean'], prior[cls]['cov'], n)
        return np.clip(t, lo, hi)

    CAL = cfg['calibration']                     # Phase 1c maps: image mean = W m + b

    def mstar(key, t):
        W, b = np.array(CAL[key]['W']), np.array(CAL[key]['b'])
        return np.linalg.lstsq(W, (t - b).T, rcond=None)[0].T

    ck = {}
    for g, nc in [('main', 7), ('base', 2)]:
        p = cfg['paths'][f'{g}_e150']
        c = torch.load(p, map_location='cpu', weights_only=False)
        ck[g] = {'path': p, 'weights': 'ema' if 'ema_state_dict' in c else 'raw', 'epoch_1based': int(c.get('epoch', -2)) + 1,
                 'nc': nc}
        del c
    N, NP = S['n_eval'], S['n_pool']
    T = {'mel': draw('mel', N, sd['target_mel']), 'ben': draw('ben', N, sd['target_ben']), 'pooled': draw('pooled', NP, sd['target_pooled'])}
    MEL, BEN = np.array(MEL_VEC_LIST, np.float32), np.array(BEN_VEC_LIST, np.float32)
    # real concept vectors: identical draw to Phase 1b (gen_eval_realc)
    df = pd.read_csv(cfg['paths']['pseudo_csv']); ccols = [f'concept_{c}' for c in CONCEPT_NAMES]
    mel_df = df[df['target'] == 1]; ben_df = df[df['target'] == 0]
    ben_train = ben_df.sample(n=min(len(mel_df) * 15, len(ben_df)), random_state=42)
    RC = {}
    for cls, pool, vseed, repl in [('mel', mel_df, sd['realc_mel'], True), ('ben', ben_train, sd['realc_ben'], False)]:
        rows = pool.sample(n=N, replace=repl or N > len(pool), random_state=vseed)
        RC[cls] = (np.clip(rows[ccols].values.astype(np.float32), 0, 1), list(rows['image_name']))
    sets = {}

    def add(name, gen, cond, seed, noise_n, idx, scale, tgt, mkey, meta, kind):
        n = len(idx)
        ms = mstar(mkey, tgt)
        meta = dict(meta); meta['noise_id'] = [int(i) for i in idx]
        for k, ch in enumerate('rgb'):
            meta[f'target_{ch}'] = [float(v) for v in tgt[:, k]]
        sets[name] = {'model': gen, 'ckpt': ck[gen]['path'], 'nc': ck[gen]['nc'], 'ckpt_info': {k: v for k, v in ck[gen].items() if k != 'nc'},
                      'cond': torch.tensor(np.asarray(cond, np.float32)), 'noise_seed': int(seed), 'noise_n': int(noise_n),
                      'noise_idx': torch.tensor(np.asarray(idx, np.int64)), 'scale': torch.tensor(np.asarray(scale, np.float32)),
                      'mstar': torch.tensor(ms.astype(np.float32)), 'meta': meta, 'n': n,
                      'info': {'kind': kind, 'model': gen, 'sampler': 'V2 colour-calibrated start', 'map': mkey,
                               'noise_seed': int(seed), 'cfg_scale_default': S['cfg_scale'], 'ddim_steps': S['ddim_steps']}}

    seeds = {'mel': sd['eval_mel'], 'ben': sd['eval_ben']}
    for cls in ['mel', 'ben']:
        idx = np.arange(N); sc = np.full(N, S['cfg_scale'])
        add(f'eval_main_{cls}', 'main', np.tile(MEL if cls == 'mel' else BEN, (N, 1)), seeds[cls], S['n_noise_eval'], idx, sc, T[cls], f'main|{cls}', {}, 'eval')
        add(f'eval_base_{cls}', 'base', np.tile([0., 1.] if cls == 'mel' else [1., 0.], (N, 1)), seeds[cls], S['n_noise_eval'], idx, sc, T[cls], f'base|{cls}', {}, 'eval')
        v, names = RC[cls]
        add(f'eval_realc_{cls}', 'main', v, seeds[cls], S['n_noise_eval'], idx, sc, T[cls], f'realc|{cls}',
            {'source_image': names, **{c: [float(x) for x in v[:, j]] for j, c in enumerate(ccols)}}, 'eval_realc')
        # control: class-agnostic colour targets (the same targets for both classes)
        ip = np.arange(NP); scp = np.full(NP, S['cfg_scale'])
        add(f'pool_main_{cls}', 'main', np.tile(MEL if cls == 'mel' else BEN, (NP, 1)), seeds[cls], S['n_noise_eval'], ip, scp, T['pooled'], f'main|{cls}', {}, 'pooled_control')
        add(f'pool_base_{cls}', 'base', np.tile([0., 1.] if cls == 'mel' else [1., 0.], (NP, 1)), seeds[cls], S['n_noise_eval'], ip, scp, T['pooled'], f'base|{cls}', {}, 'pooled_control')
        add(f'pool_realc_{cls}', 'main', v[:NP], seeds[cls], S['n_noise_eval'], ip, scp, T['pooled'], f'realc|{cls}',
            {'source_image': names[:NP], **{c: [float(x) for x in v[:NP, j]] for j, c in enumerate(ccols)}}, 'pooled_control')
    # dose-response sweeps (one colour target per trajectory, the same at every concept and level)
    levels = [round(float(x), 4) for x in np.linspace(0, 1, S['sweep_levels'])]; n = S['sweep_n']
    for base, vec, ts_seed in [('mel', MEL, sd['target_sweep_mel']), ('ben', BEN, sd['target_sweep_ben'])]:
        tt = draw(base, n, ts_seed); conds, idx, tg, meta = [], [], [], {'concept_idx': [], 'level': []}
        for j in range(7):
            for l in levels:
                v = vec.copy(); v[j] = l
                conds.append(np.tile(v, (n, 1))); idx += list(range(n)); tg.append(tt)
                meta['concept_idx'] += [j] * n; meta['level'] += [l] * n
        add(f'sweep_{base}', 'main', np.concatenate(conds), sd['sweep'], S['sweep_n'], np.array(idx), np.full(len(idx), S['cfg_scale']),
            np.concatenate(tg), f'main|{base}', meta, 'sweep')
        sets[f'sweep_{base}']['info'].update({'base': base, 'levels': levels})
    # guidance-scale sweep
    n = S['cfg_n']; conds, idx, tg, sc, meta = [], [], [], [], {'scale': [], 'cls': []}
    tcfg = {'mel': draw('mel', n, sd['target_cfg_mel']), 'ben': draw('ben', n, sd['target_cfg_ben'])}
    for s in S['cfg_scales']:
        for cls, vec in [('mel', MEL), ('ben', BEN)]:
            conds.append(np.tile(vec, (n, 1))); idx += list(range(n)); tg.append(tcfg[cls]); sc += [float(s)] * n
            meta['scale'] += [float(s)] * n; meta['cls'] += [cls] * n
    # the map fitted at s = 3.5 is used at every scale (reported in the colour-vs-guidance table)
    tg = np.concatenate(tg); ms_keys = np.array(meta['cls'])
    add('cfg_sweep', 'main', np.concatenate(conds), sd['cfg'], S['cfg_n'], np.array(idx), np.array(sc), tg, 'main|mel', meta, 'cfg')
    msb = mstar('main|ben', tg)
    sets['cfg_sweep']['mstar'] = torch.tensor(np.where((ms_keys == 'ben')[:, None], msb, sets['cfg_sweep']['mstar'].numpy()).astype(np.float32))
    sets['cfg_sweep']['info']['map'] = 'main|mel or main|ben by class'
    # chunks: never mix sets; main-model chunks first so each replica compiles one graph at a time
    B = cfg['xla']['gen_batch']; chunks = []
    for name in sorted(sets, key=lambda k: (sets[k]['model'] != 'main', k)):
        for lo_ in range(0, sets[name]['n'], B):
            chunks.append({'set': name, 'lo': lo_, 'hi': min(lo_ + B, sets[name]['n'])})
    atomic_torch_save({'sets': sets, 'chunks': chunks}, P(cfg, 'plan', 'plan.pt'))
    tot = sum(s['n'] for s in sets.values())
    log(f"prep: {len(sets)} image sets, {tot} images, {len(chunks)} chunks of up to {B}")
    for k, s in sets.items():
        log(f"  {k:<16} n={s['n']:<5} model={s['model']} map={s['info']['map']}")


# =============================================================================
# gen (TPU): CFG + DDIM (eta 0, 50 steps), V2 colour-calibrated start
# =============================================================================
def _gen_fn(index, cfg_path, single=False):
    cfg = load_cfg(cfg_path); X = XLA(); dev = X.device()
    rank, world = (0, 1) if single else (X.rank(), X.world())
    prec = X.set_precision(cfg['xla']['gen_precision'])
    plan = torch.load(os.path.join(cfg['work'], 'plan', 'plan.pt'), map_location='cpu', weights_only=False)
    IS = cfg['image_size']; B = cfg['xla']['gen_batch']; steps = cfg['sizes']['ddim_steps']
    ts_list, tab = ddim_table(steps)
    out_dir = os.path.join(cfg['work'], 'chunks_gen'); os.makedirs(out_dir, exist_ok=True)
    todo = [(i, c) for i, c in my_items(plan['chunks'], rank, world) if not os.path.exists(os.path.join(out_dir, f'{i:05d}.pt'))]
    log(f"[gen {rank}/{world}] {dev} precision={prec}: {len(todo)} chunks to make")
    models, noise_cache = {}, {}
    t0 = time.time(); n_img = 0; t_first = None
    with torch.no_grad():
        for ci, ch in todo:
            S = plan['sets'][ch['set']]; lo, hi = ch['lo'], ch['hi']; n = hi - lo
            if S['model'] not in models:
                models[S['model']] = load_diffusion(S['ckpt'], S['nc'], dev, IS)[0]
            key = (S['noise_seed'], S['noise_n'])
            if key not in noise_cache:
                noise_cache[key] = cpu_noise(S['noise_n'], IS, S['noise_seed'])
            z = noise_cache[key][S['noise_idx'][lo:hi]]
            z = z - z.mean((2, 3), keepdim=True) + S['mstar'][lo:hi][:, :, None, None]      # V2
            c = S['cond'][lo:hi]; g = S['scale'][lo:hi]
            if n < B:   # fixed shape: one compiled graph per model
                rep = B - n
                z = torch.cat([z, z[-1:].repeat(rep, 1, 1, 1)]); c = torch.cat([c, c[-1:].repeat(rep, 1)]); g = torch.cat([g, g[-1:].repeat(rep)])
            x = z.to(dev); cd = c.to(dev); gd = g.to(dev).view(-1, 1, 1, 1); zc = torch.zeros_like(cd)
            m = models[S['model']]
            for k, t in enumerate(ts_list):
                tt = torch.full((2 * B,), t, dtype=torch.long).to(dev); co = tab[k].to(dev)
                e_c, e_u = m(torch.cat([x, x]), tt, torch.cat([cd, zc])).chunk(2)
                x = ddim_update(x, e_u + gd * (e_c - e_u), co)
                X.sync()
            img = float_to_u8(((x.clamp(-1, 1) + 1) / 2).cpu())[:n]
            atomic_torch_save(img, os.path.join(out_dir, f'{ci:05d}.pt'))
            n_img += n
            if t_first is None:
                t_first = time.time() - t0; log(f"[gen {rank}] first chunk done in {t_first:.0f}s (includes compilation)")
            elif n_img % (B * 8) < B:
                el = time.time() - t0
                log(f"[gen {rank}] {n_img} images, {n_img / el:.2f} images/s on this device")
    log(f"[gen {rank}] finished: {n_img} images in {(time.time() - t0) / 60:.1f} min")


def job_gen(cfg):
    run_spawned(_gen_fn, cfg['_cfg_path'])
    plan = torch.load(P(cfg, 'plan', 'plan.pt'), map_location='cpu', weights_only=False)
    miss = [i for i in range(len(plan['chunks'])) if not os.path.exists(P(cfg, 'chunks_gen', f'{i:05d}.pt'))]
    if miss:
        raise RuntimeError(f'{len(miss)} generation chunks are missing (first: {miss[0]}); run the cell again to finish them')


# =============================================================================
# assemble (CPU): chunks -> image sets; hardware check on 4 images
# =============================================================================
def job_assemble(cfg):
    plan = torch.load(P(cfg, 'plan', 'plan.pt'), map_location='cpu', weights_only=False)
    by_set = {}
    for i, ch in enumerate(plan['chunks']):
        by_set.setdefault(ch['set'], []).append((ch['lo'], i))
    for name, S in plan['sets'].items():
        out = P(cfg, 'sets', f'{name}.pt')
        if os.path.exists(out): continue
        imgs = torch.cat([torch.load(P(cfg, 'chunks_gen', f'{i:05d}.pt'), weights_only=False) for _, i in sorted(by_set[name])])
        assert len(imgs) == S['n'], (name, len(imgs), S['n'])
        info = dict(S['info']); info['ckpt'] = S['ckpt_info']; info['hardware'] = cfg.get('hardware', 'TPU')
        info['gen_precision'] = cfg['xla']['gen_precision']
        save_set(out, imgs, S['meta'], info)
        log(f"  assembled {name}: {len(imgs)} images")
    # hardware check: the first images of eval_main_mel again in fp32 on the CPU (eager PyTorch)
    k = cfg['xla']['hw_check_n']
    if k and not os.path.exists(P(cfg, 'results', 'hardware_check.json')):
        S = plan['sets']['eval_main_mel']; IS = cfg['image_size']
        m, _ = load_diffusion(S['ckpt'], S['nc'], 'cpu', IS)
        z = cpu_noise(S['noise_n'], IS, S['noise_seed'])[S['noise_idx'][:k]]
        z = z - z.mean((2, 3), keepdim=True) + S['mstar'][:k][:, :, None, None]
        t0 = time.time()
        ref = generate_v2_eager(m, S['cond'][:k], z, S['scale'][:k], 'cpu', steps=cfg['sizes']['ddim_steps'], bs=k)
        tpu = load_set(P(cfg, 'sets', 'eval_main_mel.pt'))['imgs'][:k]
        d = (ref.float() - tpu.float()).abs()
        mse = ((ref.float() - tpu.float()) ** 2).mean((1, 2, 3))
        res = {'n': k, 'mean_abs_diff_0_255': float(d.mean()), 'max_abs_diff': float(d.max()),
               'psnr_db_per_image': [float(10 * np.log10(255 ** 2 / max(v, 1e-12))) for v in mse.tolist()],
               'mean_rgb_diff_model_scale': (ref.float().div(127.5).sub(1).mean((2, 3)) - tpu.float().div(127.5).sub(1).mean((2, 3))).abs().mean(0).tolist(),
               'cpu_seconds': time.time() - t0, 'gen_precision': cfg['xla']['gen_precision']}
        save_json(res, P(cfg, 'results', 'hardware_check.json'))
        save_set(P(cfg, 'results', 'hardware_check_cpu_images.pt'), ref, {'noise_id': list(range(k))}, {'kind': 'hw_check'})
        log(f"  hardware check (TPU vs CPU fp32, {k} images): mean |diff| {res['mean_abs_diff_0_255']:.2f} of 255, "
            f"PSNR {np.mean(res['psnr_db_per_image']):.1f} dB")


# =============================================================================
# score (TPU): CBM-A, CBM-B, ISIC 2018 detector, Inception-v3 on every set
# =============================================================================
def _score_chunks(cfg):
    out = []
    for sp in sorted(glob.glob(os.path.join(cfg['work'], 'sets', '*.pt'))):
        name = os.path.basename(sp)[:-3]
        n = len(load_set(sp)['imgs'])
        for lo in range(0, n, cfg['xla']['score_batch']):
            out.append({'set': name, 'lo': lo, 'hi': min(lo + cfg['xla']['score_batch'], n)})
    return out


def _prep_eval(u8, size, mean, std):
    import torchvision.transforms.functional as TFn
    x = TFn.resize(u8.float() / 255.0, [size, size], antialias=True)
    return TFn.normalize(x, mean, std)


def _score_fn(index, cfg_path, single=False):
    cfg = load_cfg(cfg_path); X = XLA(); dev = X.device()
    rank, world = (0, 1) if single else (X.rank(), X.world())
    prec = X.set_precision(cfg['xla']['score_precision'])
    chunks = json.load(open(os.path.join(cfg['work'], 'plan', 'score_chunks.json')))
    out_dir = os.path.join(cfg['work'], 'chunks_score'); os.makedirs(out_dir, exist_ok=True)
    todo = [(i, c) for i, c in my_items(chunks, rank, world) if not os.path.exists(os.path.join(out_dir, f'{i:05d}.npz'))]
    log(f"[score {rank}/{world}] {dev} precision={prec}: {len(todo)} chunks")
    if not todo: return
    SB = cfg['xla']['score_batch']
    cbm, _ = load_cbm_a(cfg['paths']['cbma'], dev)
    nb = ConceptNetB(pretrained=False)
    nb.load_state_dict(torch.load(cfg['paths']['cbmb'], map_location='cpu', weights_only=False)['state_dict']); nb = nb.to(dev).eval()
    ni = None
    if cfg['paths'].get('isic18'):
        ni = AttrNet(pretrained=False)
        ni.load_state_dict(torch.load(cfg['paths']['isic18'], map_location='cpu', weights_only=False)['state_dict']); ni = ni.to(dev).eval()
    torch.manual_seed(0)   # only matters without pretrained weights (offline test)
    inc = build_inception(cfg, dev)
    cur, X_u8 = None, None; t0 = time.time(); n_img = 0
    with torch.no_grad():
        for ci, ch in todo:
            if ch['set'] != cur:
                cur = ch['set']; X_u8 = load_set(os.path.join(cfg['work'], 'sets', f'{cur}.pt'))['imgs']
            u = X_u8[ch['lo']:ch['hi']]; n = len(u)
            if n < SB: u = torch.cat([u, u[-1:].repeat(SB - n, 1, 1, 1)])
            xe = _prep_eval(u, cfg['eval_size'], IMNET_MEAN, IMNET_STD).to(dev)
            cp, cf, dp = cbm.forward_all(xe)
            e = nb.embed(xe); pb = torch.sigmoid(nb.head(e))
            res = {'cbma': cp, 'cbma_feat': cf, 'cbma_diag': dp, 'cbmb': pb, 'cbmb_feat': e}
            if ni is not None: res['isic18'] = torch.sigmoid(ni(xe))
            if cur.startswith(NEED_INCEP):
                f = inc(_prep_eval(u, cfg['incep_size'], [0.5] * 3, [0.5] * 3).to(dev))
                res['incep'] = f[0] if isinstance(f, tuple) else f
            X.sync()
            atomic_npz(os.path.join(out_dir, f'{ci:05d}.npz'), **{k: v.float().cpu().numpy()[:n] for k, v in res.items()})
            n_img += n
    log(f"[score {rank}] {n_img} images in {(time.time() - t0) / 60:.1f} min")


def job_score(cfg):
    ch = _score_chunks(cfg)
    with open(P(cfg, 'plan', 'score_chunks.json'), 'w') as f:
        json.dump(ch, f)
    log(f"score: {len(ch)} chunks over {len(set(c['set'] for c in ch))} sets")
    run_spawned(_score_fn, cfg['_cfg_path'])
    miss = [i for i in range(len(ch)) if not os.path.exists(P(cfg, 'chunks_score', f'{i:05d}.npz'))]
    if miss:
        raise RuntimeError(f'{len(miss)} scoring chunks are missing; run the cell again to finish them')


# =============================================================================
# finalize (CPU): score files in the Phase 1b format, darkness and CIELAB
# =============================================================================
def _dark_lab(args):
    u8 = args
    return darkness_metrics(u8), lab_means(u8)


def job_finalize(cfg):
    from multiprocessing import Pool
    ch = json.load(open(P(cfg, 'plan', 'score_chunks.json')))
    by_set = {}
    for i, c in enumerate(ch):
        by_set.setdefault(c['set'], []).append((c['lo'], i))
    with Pool(cfg['cpu_workers']) as pool:
        for name, lst in sorted(by_set.items()):
            fa, fb = P(cfg, 'scores', f'A_{name}.npz'), P(cfg, 'scores', f'B_{name}.npz')
            if os.path.exists(fa) and os.path.exists(fb): continue
            parts = [dict(np.load(P(cfg, 'chunks_score', f'{i:05d}.npz'))) for _, i in sorted(lst)]
            cat = {k: np.concatenate([p[k] for p in parts], 0) for k in parts[0]}
            u8 = load_set(P(cfg, 'sets', f'{name}.pt'))['imgs'].numpy()
            res = pool.map(_dark_lab, [u8[s:s + 256] for s in range(0, len(u8), 256)])
            dark = np.concatenate([r[0] for r in res], 0); lab = np.concatenate([r[1] for r in res], 0)
            A = {'cbma': cat['cbma'], 'cbma_feat': cat['cbma_feat'], 'cbma_diag': cat['cbma_diag'], 'dark': dark, 'lab': lab}
            if 'incep' in cat: A['incep'] = cat['incep']
            Bd = {'cbmb': cat['cbmb'], 'cbmb_feat': cat['cbmb_feat']}
            if 'isic18' in cat: Bd['isic18'] = cat['isic18']
            atomic_npz(fa, **A); atomic_npz(fb, **Bd)
            log(f"  scores {name}: n={len(u8)}")


JOBS = {'prep': job_prep, 'gen': job_gen, 'assemble': job_assemble, 'score': job_score, 'finalize': job_finalize}

if __name__ == '__main__':
    job, cfg_path = sys.argv[1], sys.argv[2]
    cfg = load_cfg(cfg_path); cfg['_cfg_path'] = cfg_path
    rid = os.environ.get('P1D_RUN_ID')
    if rid:
        try:
            cur = open(os.path.join(cfg['work'], 'run_id.txt')).read().strip()
        except Exception:
            cur = rid
        if cur != rid:
            print(f'job {job}: started by an older run of the cell; exit without work', flush=True); sys.exit(3)
    log(f"job {job} start | torch {torch.__version__} | PJRT_DEVICE={os.environ.get('PJRT_DEVICE')}")
    t0 = time.time()
    try:
        JOBS[job](cfg)
    except Exception:
        traceback.print_exc(); sys.exit(1)
    with open(P(cfg, 'done', f'{job}.ok'), 'w') as f:
        f.write(f"{time.time() - t0:.0f}s\n")
    log(f"job {job} done in {(time.time() - t0) / 60:.1f} min")
''',
    'analysis1d.py': r'''"""ClinDiffXAI Phase 1d analysis. Runs after analysis.py (the complete Phase 1 analysis,
applied here to the V2 image sets). Adds: colour fidelity of every set, colour
targeting, the class-agnostic colour control, diversity, the hardware check, and
the comparison with Phase 1b (V0 sampling).  Usage: python analysis1d.py <cfg.json>"""
import os, sys, json, time, traceback
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import CONCEPT_NAMES, load_cfg, load_set, save_json, log, lab_means
import analysis as A
from analysis import plt, stats, Out, fci, fp

REP, WARN = [], []
GENS = [('main', 'ClinDiffXAI, prototype'), ('base', 'Baseline'), ('realc', 'ClinDiffXAI, real vectors')]
C_GEN = {'main': '#2a78d6', 'base': '#eb6834', 'realc': '#1baf7a'}


def section(name):
    def deco(fn):
        def run(*a, **k):
            log(f"== {name}")
            try:
                return fn(*a, **k)
            except Exception as e:
                traceback.print_exc(); WARN.append(f"{name} failed: {e}"); REP.append(f"\n**{name}: FAILED ({e})**\n")
        return run
    return deco


class Data:
    def __init__(self, work):
        self.work = work; self.cache = {}

    def has(self, name):
        return os.path.exists(os.path.join(self.work, 'sets', f'{name}.pt'))

    def get(self, name):
        if name not in self.cache:
            s = load_set(os.path.join(self.work, 'sets', f'{name}.pt'))
            d = {'imgs': s['imgs'], 'meta': pd.DataFrame(s['meta']), 'info': s['info']}
            for t in ['A', 'B']:
                f = os.path.join(self.work, 'scores', f'{t}_{name}.npz')
                d[t] = dict(np.load(f)) if os.path.exists(f) else {}
            d['lab'] = d['A']['lab'] if 'lab' in d['A'] else lab_means(d['imgs'].numpy())
            self.cache[name] = d
        return self.cache[name]


def colour_rule(D):
    Rl = np.concatenate([D.get('real_mel')['lab'], D.get('real_ben')['lab']])
    mu = Rl.mean(0); icov = np.linalg.inv(np.cov(Rl, rowvar=False))
    md = lambda X: np.sqrt(np.einsum('ij,jk,ik->i', X - mu, icov, X - mu))
    return md, float(np.percentile(md(Rl), 99))


def set_names(D):
    names = []
    for pre in ['eval', 'pool']:
        for g, _ in GENS:
            for c in ['mel', 'ben']:
                if D.has(f'{pre}_{g}_{c}'): names.append(f'{pre}_{g}_{c}')
    for n in ['sweep_mel', 'sweep_ben', 'cfg_sweep']:
        if D.has(n): names.append(n)
    return names


@section('Hardware and settings')
def s_hw(cfg, out, R, D):
    f = os.path.join(cfg['work'], 'results', 'hardware_check.json')
    hw = json.load(open(f)) if os.path.exists(f) else None
    R['hardware_check'] = hw
    lines = [f"- Generation hardware: {cfg.get('hardware', 'TPU')} ({cfg.get('n_devices', '?')} devices), XLA matrix precision "
             f"for generation: {cfg['xla']['gen_precision']}; for scoring: {cfg['xla']['score_precision']}.",
             "- All images (real and generated) were scored in this run on the same hardware, so the comparisons inside "
             "Phase 1d use one pipeline. The evaluator-validity table (T1) is copied from Phase 1b (same evaluator files)."]
    if hw:
        lines.append(f"- Hardware check: {hw['n']} images of eval_main_mel made again in fp32 on the CPU. Mean |difference| "
                     f"{hw['mean_abs_diff_0_255']:.2f} of 255, max {hw['max_abs_diff']:.0f}, PSNR "
                     f"{np.mean(hw['psnr_db_per_image']):.1f} dB (per image: {', '.join(f'{v:.1f}' for v in hw['psnr_db_per_image'])}); "
                     f"mean colour difference {np.mean(hw['mean_rgb_diff_model_scale']):.4f} (model scale).")
    REP.append("\n## 1. Hardware and settings\n\n" + '\n'.join(lines))
    if hw:
        try:
            ref = load_set(os.path.join(cfg['work'], 'results', 'hardware_check_cpu_images.pt'))['imgs']
            tpu = D.get('eval_main_mel')['imgs'][:len(ref)]
            fig, axes = plt.subplots(2, len(ref), figsize=(1.3 * len(ref), 2.8))
            for i in range(len(ref)):
                for r, (X, lab) in enumerate([(tpu, 'TPU (XLA)'), (ref, 'CPU fp32')]):
                    ax = axes[r, i]; ax.imshow(X[i].permute(1, 2, 0).numpy()); ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
                    if i == 0: ax.set_ylabel(lab, fontsize=6)
            fig.suptitle('Same noise, same colour target, same weights', fontsize=7); fig.tight_layout(); out.fig(fig, 'figD_hardware_check')
        except Exception as e:
            WARN.append(f'hardware-check figure: {e}')


@section('Colour fidelity of every V2 set')
def s_colour_all(cfg, out, R, D):
    md, thr = colour_rule(D); rows = []
    for name in ['real_mel', 'real_ben'] + set_names(D):
        d = D.get(name); X = d['lab']
        if name == 'cfg_sweep':
            m = d['meta']
            for s in sorted(m['scale'].unique()):
                for c in ['mel', 'ben']:
                    sel = ((m['scale'] == s) & (m['cls'] == c)).values
                    rows.append({'Set': f'cfg_sweep s={s:g}', 'Class': c, 'n': int(sel.sum()), 'Colour outliers %': round(float(np.mean(md(X[sel]) > thr) * 100), 1),
                                 'L*': round(float(X[sel, 0].mean()), 1), 'a*': round(float(X[sel, 1].mean()), 1), 'b*': round(float(X[sel, 2].mean()), 1)})
            continue
        rows.append({'Set': name, 'Class': name.split('_')[-1], 'n': len(X), 'Colour outliers %': round(float(np.mean(md(X) > thr) * 100), 1),
                     'L*': round(float(X[:, 0].mean()), 1), 'a*': round(float(X[:, 1].mean()), 1), 'b*': round(float(X[:, 2].mean()), 1)})
    df = pd.DataFrame(rows); R['colour_all'] = df.to_dict(orient='list')
    out.table(df, 'TD1_colour_fidelity_all_sets', 'Colour fidelity of every Phase 1d set (V2 sampling). Colour outlier: '
              'Mahalanobis distance of the image mean CIELAB above the 99th percentile of real images (Phase 1b rule). '
              'pool_*: class-agnostic colour targets (control). cfg_sweep: the calibration map fitted at s = 3.5 is used at every scale.')
    REP.append("\n## 2. Colour fidelity (V2)\n\n" + df.to_string(index=False))
    # figure: a*-b* per class, class-specific versus class-agnostic targets
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), sharex=True, sharey=True)
    for ax, c in zip(axes, ['mel', 'ben']):
        X = D.get(f'real_{c}')['lab']; ax.scatter(X[:, 1], X[:, 2], s=2, color='#9a9993', alpha=0.4, lw=0, label='Real', rasterized=True)
        for pre, mk, lab in [('eval', 'o', 'class-specific targets'), ('pool', 'x', 'class-agnostic targets')]:
            if D.has(f'{pre}_main_{c}'):
                Y = D.get(f'{pre}_main_{c}')['lab']
                ax.scatter(Y[:, 1], Y[:, 2], s=3 if mk == 'o' else 5, marker=mk, lw=0 if mk == 'o' else 0.5,
                           color='#2a78d6' if pre == 'eval' else '#eb6834', alpha=0.45, label=f'ClinDiffXAI, {lab}', rasterized=True)
        ax.set_title('Melanoma' if c == 'mel' else 'Benign'); ax.set_xlabel('mean a* (green to red)')
    axes[0].set_ylabel('mean b* (blue to yellow)'); axes[0].legend(markerscale=3, fontsize=5.5, loc='lower left')
    fig.tight_layout(); out.fig(fig, 'figD_colour_class_specific_vs_agnostic')


@section('Colour targeting (does the image get its target colour?)')
def s_target(cfg, out, R, D):
    rows = []; pts = {}
    for name in set_names(D):
        d = D.get(name)
        if 'target_r' not in d['meta']: continue
        T = d['meta'][['target_r', 'target_g', 'target_b']].values
        Y = d['imgs'].float().div(127.5).sub(1.0).mean((2, 3)).numpy()
        r2 = [1 - ((T[:, k] - Y[:, k]) ** 2).sum() / max(((T[:, k] - T[:, k].mean()) ** 2).sum(), 1e-12) for k in range(3)]
        rows.append({'Set': name, 'n': len(T), 'MAE R': round(float(np.abs(T[:, 0] - Y[:, 0]).mean()), 3),
                     'MAE G': round(float(np.abs(T[:, 1] - Y[:, 1]).mean()), 3), 'MAE B': round(float(np.abs(T[:, 2] - Y[:, 2]).mean()), 3),
                     'R2 R': round(float(r2[0]), 2), 'R2 G': round(float(r2[1]), 2), 'R2 B': round(float(r2[2]), 2)})
        if name.startswith('eval_main'): pts[name] = (T, Y)
    df = pd.DataFrame(rows); R['targeting'] = df.to_dict(orient='list')
    out.table(df, 'TD2_colour_targeting', 'Colour targeting. Target: mean RGB (model scale [-1, 1]) drawn from a Gaussian fitted '
              'to the mean colours of real images of the class (pool_*: 50/50 mixture of the two class Gaussians). Achieved: '
              'mean RGB of the generated image. The calibration maps are the Phase 1c maps (fitted on GPU images).')
    REP.append("\n## 3. Colour targeting\n\n" + df.to_string(index=False))
    if pts:
        fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5))
        for k, ch in enumerate(['red', 'green', 'blue']):
            ax = axes[k]
            for name, (T, Y) in pts.items():
                ax.scatter(T[:, k], Y[:, k], s=3, alpha=0.4, lw=0, label='melanoma' if name.endswith('mel') else 'benign',
                           color='#2a78d6' if name.endswith('mel') else '#eb6834', rasterized=True)
            lim = [min(ax.get_xlim()[0], ax.get_ylim()[0]), max(ax.get_xlim()[1], ax.get_ylim()[1])]
            ax.plot(lim, lim, color='#52514e', lw=0.8, ls=':'); ax.set_title(ch, fontsize=7)
            ax.set_xlabel('Target mean (model scale)')
        axes[0].set_ylabel('Achieved mean'); axes[0].legend(markerscale=3, fontsize=6)
        fig.tight_layout(); out.fig(fig, 'figD_colour_targeting')


@section('Class signal: class-specific versus class-agnostic colour targets')
def s_class_signal(cfg, out, R, D):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.model_selection import cross_val_predict, StratifiedKFold
    B = cfg['analysis']['n_boot_auc']
    rm, rb = D.get('real_mel'), D.get('real_ben')
    feats = {'CBM-B': lambda d: d['B']['cbmb'], 'CBM-A': lambda d: d['A']['cbma'], 'Colour only (L*, a*, b*)': lambda d: d['lab']}
    rows, bars = [], []
    for ev, f in feats.items():
        Xr = np.concatenate([f(rm), f(rb)]); yr = np.r_[np.ones(len(f(rm))), np.zeros(len(f(rb)))]
        mk = lambda: make_pipeline(StandardScaler(), LogisticRegression(class_weight='balanced', max_iter=3000))
        pcv = cross_val_predict(mk(), Xr, yr, cv=StratifiedKFold(5, shuffle=True, random_state=0), method='predict_proba')[:, 1]
        a = A.auc_ci(pcv[yr == 1], pcv[yr == 0], B)
        rows.append({'Evaluator': ev, 'Images': 'Real (5-fold CV)', 'n mel/ben': f"{int(yr.sum())}/{int((1 - yr).sum())}", 'AUC [95% CI]': fci(*a)})
        clf = mk().fit(Xr, yr)
        for g, glab in GENS:
            for pre, plab in [('eval', 'class-specific colour targets'), ('pool', 'class-agnostic colour targets')]:
                if not (D.has(f'{pre}_{g}_mel') and D.has(f'{pre}_{g}_ben')): continue
                dm, db = D.get(f'{pre}_{g}_mel'), D.get(f'{pre}_{g}_ben')
                k = min(cfg['sizes']['n_pool'], len(dm['lab'])) if pre == 'eval' else len(dm['lab'])   # paired noise ids
                pm = clf.predict_proba(f(dm)[:k])[:, 1]; pb = clf.predict_proba(f(db)[:k])[:, 1]
                a = A.auc_ci(pm, pb, B)
                rows.append({'Evaluator': ev, 'Images': f'{glab}, {plab}', 'n mel/ben': f'{k}/{k}', 'AUC [95% CI]': fci(*a)})
                bars.append((ev, g, pre, a))
    df = pd.DataFrame(rows); R['class_signal'] = df.to_dict(orient='list')
    out.table(df, 'TD3_class_signal_colour_control', 'Melanoma versus benign separability (logistic regression trained on '
              'real images only). Class-specific: colour targets drawn from the colour distribution of the same class. '
              'Class-agnostic (control): the same targets for both classes, drawn from a 50/50 mixture. Both use noise ids '
              '0..%d. The drop from class-specific to class-agnostic is the part of the class signal that the colour '
              'targets add.' % (cfg['sizes']['n_pool'] - 1))
    REP.append("\n## 4. Class signal and the colour control\n\n" + df.to_string(index=False))
    evs = list(feats); fig, axes = plt.subplots(1, len(evs), figsize=(7.2, 2.5), sharey=True)
    for ax, ev in zip(axes, evs):
        x = np.arange(len(GENS))
        for j, (pre, col, lab) in enumerate([('eval', '#2a78d6', 'class-specific'), ('pool', '#eb6834', 'class-agnostic')]):
            v = [next((a for e, g2, p2, a in bars if e == ev and g2 == g and p2 == pre), (np.nan, np.nan, np.nan)) for g, _ in GENS]
            m = np.array([a[0] for a in v]); lo = np.array([a[1] for a in v]); hi = np.array([a[2] for a in v])
            ax.bar(x + (j - 0.5) * 0.38, m, 0.36, color=col, label=lab, edgecolor='white')
            ax.errorbar(x + (j - 0.5) * 0.38, m, yerr=[m - lo, hi - m], fmt='none', ecolor='#52514e', lw=0.8, capsize=2)
        ax.axhline(0.5, color='#52514e', ls=':', lw=0.8); ax.set_ylim(0.4, 1.0); ax.set_title(ev, fontsize=7)
        ax.set_xticks(x); ax.set_xticklabels(['ClinDiffXAI\nprototype', 'Baseline', 'ClinDiffXAI\nreal vectors'], fontsize=6)
    axes[0].set_ylabel('Melanoma vs benign AUC')
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, [f'{x} colour targets' for x in l], loc='upper center', ncol=2, fontsize=6.5, bbox_to_anchor=(0.5, 1.08))
    fig.tight_layout(); out.fig(fig, 'figD_class_signal_colour_control')


@section('Diversity (real versus generated)')
def s_div(cfg, out, R, D):
    rng = np.random.default_rng(5)

    def div(F, k=1000):
        F = F[rng.choice(len(F), min(k, len(F)), replace=False)]
        F = F / (np.linalg.norm(F, axis=1, keepdims=True) + 1e-9); S = F @ F.T
        return float(1 - S[np.triu_indices(len(F), 1)].mean())
    rows = []
    for name in ['real_mel', 'real_ben'] + [n for n in set_names(D) if n.startswith(('eval', 'pool'))]:
        d = D.get(name)
        rows.append({'Set': name, 'n': len(d['lab']), 'Diversity, Inception (1 - mean cosine)': round(div(d['A']['incep']), 3) if 'incep' in d['A'] else np.nan,
                     'Diversity, CBM-A feature': round(div(d['A']['cbma_feat']), 3), 'Diversity, CBM-B feature': round(div(d['B']['cbmb_feat']), 3)})
    df = pd.DataFrame(rows); R['diversity'] = df.to_dict(orient='list')
    out.table(df, 'TD4_diversity', 'Within-set diversity: 1 minus the mean pairwise cosine similarity of image features '
              '(up to 1,000 images per set). Real rows give the reference value.')
    REP.append("\n## 5. Diversity\n\n" + df.to_string(index=False))


def _kid_val(s):
    try:
        return float(str(s).split('±')[0])
    except Exception:
        return np.nan


@section('Phase 1b (V0, original sampling, GPU) versus Phase 1d (V2, TPU)')
def s_compare(cfg, out, R, D):
    ref = os.path.join(cfg['work'], 'ref'); new = os.path.join(cfg['outputs'], 'tables'); rows = []
    for t in ['T2a_concept_agreement', 'T2c_condition_discriminability', 'T2d_distribution_distances', 'T3_dose_response_mel_base',
              'T5_colour_fidelity', 'T6b_realistic_vector_tracking']:
        a, b = os.path.join(ref, f'p1b_{t}.csv'), os.path.join(new, f'{t}.csv')
        if os.path.exists(a) and os.path.exists(b):
            da = pd.read_csv(a); db = pd.read_csv(b)
            da.insert(0, 'Sampling', 'V0 original (Phase 1b, GPU)'); db.insert(0, 'Sampling', 'V2 colour-calibrated (Phase 1d, TPU)')
            pd.concat([da, db], ignore_index=True).to_csv(os.path.join(new, f'TD5_{t}_V0_vs_V2.csv'), index=False)
            rows.append(t)
    REP.append("\n## 6. V0 versus V2 tables (supplement)\n\n- Written: " + ', '.join(f'TD5_{t}_V0_vs_V2.csv' for t in rows) +
               "\n- V0 and V2 differ in the sampler AND the hardware. Use these for description, not for tests.")
    # acceptance of V2 on the full data, descriptive (the rule from Phase 1c)
    Dc = cfg['decision']; acc = []
    md, thr = colour_rule(D)
    k_old = pd.read_csv(os.path.join(ref, 'p1b_T2d_distribution_distances.csv'))
    k_new = pd.read_csv(os.path.join(new, 'T2d_distribution_distances.csv'))
    for g, glab, img in [('main', 'ClinDiffXAI, prototype', 'ClinDiffXAI'), ('base', 'Baseline', 'Baseline')]:
        for c in ['mel', 'ben']:
            o = float(np.mean(md(D.get(f'eval_{g}_{c}')['lab']) > thr) * 100)
            sel = lambda d: d[(d['Feature space'] == 'Inception') & (d['Split'] == c) & (d['Images'] == img)]
            ko, kn = _kid_val(sel(k_old)['KID x1e3 (mean ± sd)'].iloc[0]), _kid_val(sel(k_new)['KID x1e3 (mean ± sd)'].iloc[0])
            acc.append({'Generator': glab, 'Class': c, 'Colour outliers % (V2)': round(o, 1), 'colour ok (<= %g%%)' % Dc['max_colour_outliers']: o <= Dc['max_colour_outliers'],
                        'KID x1e3 V0 (Phase 1b)': ko, 'KID x1e3 V2 (Phase 1d)': kn,
                        'KID not >10% worse': bool(np.isfinite(ko) and np.isfinite(kn) and kn <= ko * (1 + Dc['kid_tolerance']))})
    res = json.load(open(os.path.join(cfg['outputs'], 'results', 'phase1_results.json')))
    dose = {(f['ev'], f['concept'], f['output']): f for f in res.get('dose', {}).get('mel', [])}
    ctrl = {}
    for c in ['pigmentation', 'pigment_network']:
        f = dose.get(('cbmb', c, c)); ctrl[c] = bool(f and f['p_bh'] < Dc['alpha'] and f['median_slope'] > 0)
    df = pd.DataFrame(acc)
    out.table(df, 'TD6_V2_acceptance_full_data', 'The Phase 1c acceptance rule checked again on the full Phase 1d data '
              '(1,000 images per class). Descriptive only: V0 values come from Phase 1b (other hardware).')
    ok = bool(df.iloc[:, 3].all() and df['KID not >10% worse'].all() and all(ctrl.values()))
    R['v2_acceptance'] = {'table': df.to_dict(orient='list'), 'control': ctrl, 'all_ok': ok}
    gate = json.load(open(os.path.join(cfg['outputs'], 'results', 'gate1.json'))) if os.path.exists(os.path.join(cfg['outputs'], 'results', 'gate1.json')) else {}
    R['gate1_v2'] = gate.get('decision')
    REP.insert(0, f"**Gate 1 on V2 images: {gate.get('decision', 'n/a')}.  The Phase 1c rule on the full V2 data: "
                  f"{'HOLDS' if ok else 'DOES NOT HOLD (see TD6)'}** (colour outliers <= {Dc['max_colour_outliers']:g}% and KID not more "
                  f"than {Dc['kid_tolerance'] * 100:.0f}% worse than Phase 1b for main and baseline; CBM-B pigmentation and "
                  f"pigment-network control: {ctrl}).\n")
    REP.append("\n## 7. Phase 1c rule on the full V2 data\n\n" + df.to_string(index=False) + f"\n\nCBM-B control: {ctrl}")


def main():
    cfg = load_cfg(sys.argv[1]); out = Out(cfg['outputs']); t0 = time.time(); R = {}
    D = Data(cfg['work'])
    s_hw(cfg, out, R, D); s_colour_all(cfg, out, R, D); s_target(cfg, out, R, D); s_class_signal(cfg, out, R, D)
    s_div(cfg, out, R, D); s_compare(cfg, out, R, D)
    head = ["# ClinDiffXAI Phase 1d report: final evaluation with colour-calibrated sampling (V2)\n"]
    if cfg.get('dry_run'):
        head.append("**DRY RUN: a few images only. These numbers are not results.**\n")
    if WARN: REP.append("\n## Warnings\n\n" + '\n'.join(f"- {w}" for w in WARN))
    REP.append("\n_The complete Phase 1 analysis of the V2 sets (tables T1 to T7, Gate 1, all figures) is in "
               "phase1d_full_analysis_report.md._")
    REP.append(f"\n_Analysis time {(time.time() - t0) / 60:.1f} min._\n")
    with open(os.path.join(out.root, 'phase1d_report.md'), 'w') as f:
        f.write('\n'.join(head + REP))
    save_json(R, os.path.join(out.root, 'results', 'phase1d_results.json'))
    print('\n'.join(head + REP))


if __name__ == '__main__':
    main()
''',
}
EMBED = {
    'results/derm7pt_evaluators.json': 'H4sIAL+P1GoC/4VXy24bORC85ysMnb1Bd5PsR25Jznvc0yIQZFuwhShSIMkJFkH+fYsTOR4BacqABXtITZH9qKr+8ebmZrFbng6rzW7x7qay3P568m213TzgiVA5Pzmtjyc8KFGnB3f4t+9aflmvdu//+YgleuscymwuVUqT5tPOp83xtD/8hw3/4t+bmx/TJxbWX/f3T3jMty9PpnMst/vjcXqdRlVtEUVaKfF7V0c93u8P62lTaxxCNdxdWVymXT9v/wwlKRQDhDRMm/R3JWBKlUsjb2RVpdYhWMnAmlKHsWgSxSkDsyqGuwOwUWi0IVhNwcioEjmFVBfPwmgKhGgWQSbSxmFsGVhtVpALJE4Cn+nNAhml6oi2UVwB0xSsiKPWrBlLKcwpGBs37K1aqtM4jJaCEamwGDH+YsnBzGoQAa0wKnII5hlYsYIO0oqcqZqHJmjOVMTcrbXmGmO0SNFqNSX2qK3gjyxpKNSKm4cVhBMZHIIxpWjCFfVci1XnptJSOK2Mg0WJonIFLaWQQoT69xY12KVxCobLqwfOFIWkjNFSFhEHORSp6mg585qmTdyboeWMK9OYRTilEdHWgMcIJZOZ1DxxoCwzQm+ShI/hUiJBy+LM1t+AWJLnRdnzFWgVpLhcaW5OqUQKBTdm8ARbuORwFXRFHF5B3OPu5pRLRNAFDgbEh4InWo6Gqu0cJ4TeHKOlZAJON0edIWu1ATkFw+0hfQ1K4GCdMVrKJkLUTArYVpBAtPigCeAAoHGIOJcYw6V0wkHIVvOiraiDB1M4NLcTrshglivyJimfgGWFo2lV/DIEJc8cDqWCSKjjZGO4lFAYUWSQRWgvUC853KUVGsNJDlfVSvcl4DGFz8mDiUjWzqdKhce5k5RSWCFgiiii9ERGUgAOY2oVBs3ruDClDtC0FoeBY4Zi5gQGzVX8OFtX4SuxbDkcKXXVgTURBlXnfYC6rPCd4LnaxvQsKaVwqwTeJXRVoIcH0oO8MgwMBKFdy5zlaEW74ygg35AgybUHkggcOIbuMcZwKadAw1FoVBSJg2LmXcBoSlhdA43hd4yWU0qPoAWOXR0TAQ/qpGl35mSwvWNznhNKte6mRBQU7Zaa2N4iSrCfvZ7iSlUWHsFxGCxKp8E6QHMojnUoyP1ZVfH5aRqyvu6Pm9Pm2/r4Mqf9OsHi6+bxy3p3Wu7Wp+/7w+ep+c9HWdxtn9fL70/43vFp+W292XZLeq4boB/vn7erw/J4Ojzfn54P637cdm7Rl/euTpv9bpK6cwUssH29+nycjMvL5of96bhc7R6Wj9v93fN2elN5obLFYf2Ilx/xokss5t55U0AXq+f71zt9/PD3X++Xu1W/7++p8qbHCMloGgTehfa8hpzeBlTGwCHB6uDt2YpjDIMSexGMc+jV2QryizkGMxh6q7xyZsepDW2EyZPitfRh/lvvd7j2giqEKZmtQOhrUWzHbAWdmhY+3c7vc7c/nbbr3fr+8+WdCG5LAiwrAvqbn8/BUDgGOBEO0HyOprBDEHucRmd3NTAoHLxABOHAeX5ywxwBwwpDzl2w5ncKzKkw7Vr6PDr/CtwnhA1MAOY5a8D8Sh/+mCLsFWS2YNZGOOrsOgHLXKvA88GKNZ+vYEZ0RWsr+mO+0AWE8SLwS7lcwDiILtH+XMo8Ec0RTJiv7oXmFYLB18EcDkuuL8PN5W2yBIFz2DCje8MxQuZFB15HjKQrd5hd3AjGX7vEIncXd8WIFQUTO8i1zrxTj1yDgGEwNO9mYB45Kp072K07ALoo1IJwAhwzfVfPX7f63VUPm9XjctRa9LbBXFO31rAeKIWWlyxGZJhZ3NTAVEgUKLgDvfn5P1a0Q4VkEgAA',
    'results/isic18_evaluator.json': 'H4sIAL+P1GoC/4WWT28bOQzF7/kUgc9BQYoU/+xt0VPPiz0tFgPHHSSDOHbgmaYognz3fWM7iY2unIsPEsc/knp80svV9fVi003bable/HFdaurNcWm3HDZY4mA7Lj0v18N3LEnkW1A/TicLT7seMf1m1XfL9fx/L1id14e7x34zdZt++rndPWCDvtRwLsqUxYIj6s0hdNPfLafhuT+LJZeiNSuRSiHWt+Bx2vXLh/EQI1EraWYUM0vLY8zjsB6W3Xp46LvVr3229KVYSVblZMePih5j79bb2x/r/vCH5cDkKiLs1Qwxr4cyt+Mw5zi+1d8sEwW26yr5Wxlcmlmn/55kfOS0/LH6SOTbX9++cnSbPRHL/+yXr1GUmxRjDtN0p2Ny84Z5YtldUk2MyseOl0qlMFqPhmie7BjVWiKCmNiC6uk3SVazZGh4Ruw3/r05y+52O03QSr96OM8Qx1gKidWaEnySootLRf4cknSGC4WGLN24QCmUJymqVmUXFvyZU5x8Q17cyBGhUeWY4ns/b3Gys+C7x365+fPvr3tJuEcmo+w0C7F9bov7YZy2u1/vRbwcEYv+abu6nw/1DbrYz1S33o4HhaUHRAuxBnJIfu/sYuaOq+2uP0wKJsQLM+pDlaXso15v/h9WmjD0ATOXFc1QCmnBnNGVuZtWrYhdhEkTxoL2QhqihOyjATMXqckYVhwAxHIRpi1YYIBxLjoPM/TbYHlRwVkbhtoz7XJhtcVyTAh0SQEcFfEWTDAxJgRWDXT9IsyaMLLMFAaM0J1mZa7oXoS6GmqrF2HegpkpEpUKpWnlJgvC96rqQjCQT+QRTRYX6B5FqcE6vF3Y+bhdhGULVqtQLRJGhCY1p8wNxgIHgfu5eupFGFOTRvAmZfQQfuMcTRqUQYWLwTwp5TKt6SCKxiBpyATXH660Fg03I7wD5omhxhlfpjUtRNHIqMCpY5q02cma8Bn0UNDKypc1wk0PmeVRE8JngrprWyRUBaJ1WFZQfEJrmghSRc5B4goY1xZNHUblSTBTlHbZi7lpIyLwBalVTQt8mZsqcUEf52cP3inyCa3pI2WWBowEkjPYVjSLM1YcbMV9iHfWJ7SmkRS4uc+F4eUouEWasHm2MdaV8Vr45NiaTlLwIFGIH1d74ikp0dYkXiJ4VQjuCnjzAXd1fJQshu/d9rnfrZdP3c9huu+GcVgVKtStxuf5+33Q+LQepm7s+/kNrOXq9T8MeQwgPQsAAA==',
    'results/derm7pt_test_preds.npz': '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',
    'ref/p1b_gate1.json': 'H4sIAL+P1GoC/62VXU/bMBSG7/kVR70CqYn8GTvbNKlU26jEAK2wXVZu4oJHGke2gSHEf99poOVjwA3LXY/fvD7Pee30ZgtgEC4aO/gAg6PRdApuAePd79kuxDN/FcFA56NL7tIOIbrT1i1cZdoEtY82CzZ2vo0WttOVz6KrbQ2/XFP5P77t1bbOgmnPIdmYAGudDVkK5retkg/XEBvf2TiE3b2s8iFgFQ06+AQkJ3IIS1s7096p4DOQHVj4ALuHx3vQudOlbZNJDl1NW68L0FpsJZzDIvglpDOLJo1p/RI5gk8+XXcIsnphtWYujWvMvLHQ+jZDbKjObHWO1KfBIlVj48q/NuG8tTH22z/e+SNMppMxMEI11Db1VI9F6252cpgcjA8Pxvsn08nPLw8z7ucyOhnDNmUa3+ps41q7009Akd7KOuw0QOXbynYph6PRj+PJaP/BozMx2gjzi4RhPeGABeLFHL6OJvvgVzZXLtp8MFyFXgWXbHAGg7/B31j5Rs3a8hHjZh0Vm3lhMYULO1wvrHr4pzabn2GN50yWBSsJpVQKqqTNSLkR3UU86yNGMcklF0pQqVmhVUGk2Chr2yQzizWqaK6kKHCZilJRSpTeqBbBVDM8sb1XiR6yX7kdrhHnTxHXCb2XkuaIqKmQREqliFbibUyiSoUPLyhnlCrNX+IkudBEKq4100RIxthLnDQnzxir10/sezmxc0Io0UVRSkpL7E0qrV/jFDnlQhSSE8G0Kkv1MiWOgCvGpeCCqKIQL6epNGXP06z7C5j1F/A/J9qTSjSmJQIIXuDhLd4IVGqqOKGKFJoLpl4JlFOJWXLKKVN4L9grqPecW/esg2q+nM8uTeNql65n5qKa4ffi4eY+S5jkWhBackVEQYjQ4r6ZtW72MKJeKrFzKbUUSpVss2dtKxfvDPt/hsHW7V8cnghkLgYAAA==',
    'ref/p1b_T2a_concept_agreement.csv': 'H4sIAL+P1GoC/5WSS2vjMBSF9/kVF0MhAaVIsvXwMnU33k5pKZQu1PgmCGTZ2JqBmV8/9o37oGQ66UoY3fPdc3RcBTeOkNxwxMTq1h1xZJHtjgNiizHBU6muoKqfVy0Glv1AF6C+qyuQXHIGQlrofY/BR8yYsgXL+HVRCHiajrxgMH/Z5+y8OrrkfyGs+xF/Nt02uBcMUN09bN5ROieU0oQyfEFV08Jbfzg87mpY77u4xz5tp7PxyXcRmw0TnPOZoCSfCUqYmaBkvhBu3EiuJ/n8AufFb+tJXOjyNcmH/Qy6vu9Gn3BLJPD0iNk7RBqCSEGQXC6Q++gP3dDC4GLTtdAP2Pj97GCEtYsu/E5+v2GMX2vJT6Gnq+SmTqYcH8dpRmua2YWw/YND9/leGLt6wfi/BuXi2eZ29mxPFdpCTJ7Pyr+o8I0lJbFESSypF9blHRpDNRhtZ4Ix5UK4qENTliQuc1rPi9co3+lQn34ErSiD1maBXNyhysUp9BcdqlzRzL86LLVa/QUpRMUGrwMAAA==',
    'ref/p1b_T2c_condition_discriminability.csv': 'H4sIAL+P1GoC/42RwUoDMRCG732KsCAoZEsmyWQS8dKuHnrwoiiC9LC1qRS3rXRrn8Tn8pncTFpoq4inMN/On/kme7Otm496s1rLu1g34hzL2aqZiurxQlbNfHk9n82eBiM5rNvYlVF+fYrBQyWeA56JajSWxSI2ZTN/i8vYtlJ0lZgv6tfYXop3KdaFLCZxedDQVacNh3PEOlls40tn1P71TWxbMdlL/Tb2n8kfPr1qeFsOZKH6hEY8d4cBKbqD3LhIOIBNOCjGQVPGpCFhFzx3W5MxBJ0wkE5Yg074SvVVjnsbDupS9a1FzqENKYfapjYkm69TynIbKHucQ6ez+jC1eWAZCjzV65DTPriEvc/qaocJKKsjq5sdBvKsjiFhCOFYnQhO1I3N6qiyusnqtFfXWR1Oco5M0Rvdj6pSK/BiGjf8o1LGAb8eekpXObN/a897kHO8ntphJN4aLe+BATPWgLxHYB9tTvdQJvs5HtId+TLgGdbz9miPflyZbgXus15xTLui9w3lEa6CTQMAAA==',
    'ref/p1b_T2d_distribution_distances.csv': 'H4sIAL+P1GoC/41Vy24bMQy89ysWOSWASot66xgnCGAUubSXXjeJHBhw1oHXCfpb/YV+WSlpHdsbaeODH2tjyOGQQ96Fdve2DU3/2j4G9ut1vdqxxUv7HHrWNc+hm23Dkt0tbtmPxW3zB4NsLl9C2zX//jb901X6tX976MPu26J7DK+71aZj7XrNbtar7na1XP6+XjDBOZ8hGsdQKHDMSZAYIyiQmmnOR9h52weChyOg82CZMoA+4gQoU8D9DO26ee+bbfy8XK43m+0V007N6MWkA2R8CECf48Qv4ZQ0xtwRKHhEovFENieXvoD9IP0BRO6pWE1kZcZxWcAVSQsvZvRihlMEDtJl0pzHf44CPITuM+moGvOCBNMcrBs421HuCD3lnHEcTBTa24yzvIArck4x4hsTMknNVWYtsNCri2PaOdB7eNxttv3FUdu9iQoSpUFBZwoKToU6NAMlcKYcaDEI4gqFfRkqaYRcgmLagzeDSFncm/n99+vm8j48kWGuJmxgXeSiNIgkEZkCXS1CwQxKgGbo9/mpMiNq8ElPIMZA4Ifx8liKUjUGUmMk89qC44kHCeFrET7bQyvwTCiaj2Qr4oLV/JMmEYJG1lHuE5OMotSdQlVQDCsofeJhwRfVrBiG9gpJYRUIO4jgXQ3+hW8MESFJUx3UGV/r6XneURiJ0R7gmZiiPVTT9ywLKVrb1HSqNzG0IE2t0PN8pDFOgLH7CbD7VREjzif8I0XcrJ4cqJOBtNlvhgOy4BtMrVbC0aQn30nwfASb9ItS5Fs9LDba6+IIXLWJkRElhJTgknDxC46Qhevh42ApQy4RQ/+Gm3VATdrCIV0AN2h7Yov5hBuQ7hUNnqENl5aLpEd1lLbiAlRxF2g0ILM4H1fyAPvqasTL852GVuX5t27UmvPGHkkxmgXyIc/n2oCyI93OuxjO0ZApuoXG7tesHJV05r2gSJpZyUFnTekxt/I/ilW6CH4JAAA=',
    'ref/p1b_T3_dose_response_mel_base.csv': 'H4sIAL+P1GoC/5VW22rjVhR9n68QgUBLFLH3uepAKTRpoYEOLWTeQjCKrSZuXDvIcvLcl35Ame9q/6Ff0n05UpSxp3heouj4XNZee6119MNzs9o1/aYrr1/ap76Yb9ZzepY/7/qnXV++bxfLZl1sV5unll6Gf4ub5E+Ly6vb8kPX/FYVL8v+If/ybQHlU/HVxY9fl92suyv//ljAv3/+hcVyXXRtszq//r7YrZf99nWPXzabVbso/vljHHp3efH+/LvyaXn/e7vuZ+u2f9l0j3vvZ1BBMOUJVIhQ3NBbxLKgN5duT8qEp+U3NAZYQpVMzfN85GkOeFaMNInGrPU8aHzgUWfr25N8/t1q185eHpb9cvswe26XqwMjjAGcYADZB5wVDIH2KesphlgTBlNZx9NSXRa2ikkxoFP8teO1xroRw3Ozne9WTTfb9t1u3u+6dntwTHBY3gsyDjC8FwAfEdJpqUNQBYwyzcgsIQxcpgJlEH0SKiKMMDL1Tb/crN++0MEJkVfXQSqLQYpInglAgJEBpL/ADMS6uDGVEwbA6tEhBl4cwEtv/CsDVGTbPG7H55kWIiWgVuq1CE9bRT+p1CSeZrRSmeRqPQ6S1WZJpRjNeNxi029nzXoxu19t7nYronZ/hCEk0H1q3Vz3sYGb7iZNr41AdYLUCDM+YzAgGNBEHrUYRwxde0893RK90xYfHhU6RNwQCMs59zQrkAXuw6n0nPlACCK2pKwJH5D5gCBrEXQtDvxfHOXCGtSFSgbpnPfw5rAAuAs8z4rhglEEDsWbVu3jbRoBHGlDVIujbg5JWDUIe1HA+scqkeOQXF8WpgpZhBhROyJaMd6OGL7AhsSddtfpXuIwU0skTZWRVMRBjnQyq/ZZGbWYAZMQ6Yz/tBmHfeitl5ALEibOS4dD5OJSnJ6cDDMQPTPAYWioZ9mGVtjzujaaMJ58yIYpqQdUUN5r+Ia94LMyz8k0k3TnIfg0rKJ2y73K7kgfopfswZC9JA7DlD4nvSDMesnG2gxGFGQYpQALOIL4QiNikELRa+s1gY1w+zYUqN3UABCu+VSs7BAKOYKdUSzc+qvrq8tzA+SuRdu3c76tjzGl11SImnRWEwpiDskRTEhybVg9VxU36NAoMRoolgPlEJhD0gj58HwnqjS4mrdMxCQmsHESj4M08g2l1wnGzxy+r4o3MW31snNyABiRNaFiEozeFLbkSLaTm0KYspmD4dNC08pISv/UcvOLRdM9rkkIb424kh9n449nju4FigRPdz61wlJTCUVNLkRGUU/ZCH7SCrGEG74Q6oxNoymZI2HQB13fNdu+PEtVYhj0SNSUQCIkGEg3sLF7+gzCWi2V+8j6hJA/EdSwxubr9Fg6GpLG7NeumUtXopgWalWH0TuHzT90BUPJu3vNpEEcBAOHpNalqL2S3PhfGCyKZjXpiqkMOY++Roy6NTEdlq6jqTYia8NHTRgNKqE/kwHRTWSF/HHx7j/niHufUgsAAA==',
    'ref/p1b_T5_colour_fidelity.csv': 'H4sIAL+P1GoC/32RO08DMRCE+/yKUySaaLPyrt8lHE2kVDTQOsiBSPeQ7hJ+P2snKAcFhdfWWd/M3HjXp488Q9uleYYB9pumz2mAdNsPt739nMY+Xc+v1Ow3ZaY6Dxtox268TM14OXenPM3Nw+olpw763IENBpxCDWTRARlkYMYACpVaDEZzZQ55AFbyxcUCeVRAJCeB7F9IoVu13Wl4Ph2Pb4+76keFtcWBBNgSGmAjzh69L/ZGpkP24D2GX3Rx/qEjOCQJpUDXGA6NRCAkCxSQDQSNfvWU5iwCeWGsBfOytkrcta3GWhWByEARWa4D2ju6dPUQxZkFF9cKOV8hU4oQgaDEdb0M3UxSWvOV38/jNK8XQUzxF7Waw4iaxeiKjI3lF4hLAfyv2D2alhZ1jaZBSxflMSUOK+RYy5FCSB7jGz6TfYVNAgAA',
    'ref/p1b_T6b_realistic_vector_tracking.csv': 'H4sIAL+P1GoC/42Sy26DMBBF9/mKiFUj0YrBL+iuyaKrbvoDyIFRgmJMhE3y+/UMVR8SlboZjfDxnTvXHJwNIT+MvsVrzLN3tO55O53HB29jf8N8e9i/Pb5soax2WZ69osfJRuwWph2HwfoOu0+MkHWF/f8U9rtsM6DLr/1pQB8bj/E+Tpe8eNIgU5UFUF+WqSoDzB7djM393Mc+nJsb9o4IpVMtK+Y03REgmb7Z0M7OTk2I09zGecKQTisgXjBvNE+qxE8naZXR02ENLEmIrivqlWEw6aG9kJiueDhbUKKmXi9MN8bQpHWbkxuPs+PRimYWPFmWJFuaxemEp2QupLm/vRrD8lqzBUX31OaIfi00w5szqZisa0ZXMwPyAEvOimxDyfB6ZEYIEieOrUgQP118BVZU/BKKPUj2Awx+B6Zq+g7MKKCAwSxiq4HxVoXkh1CsLg3TfyWmoeaMBf87hvX15gMOxOnE+gIAAA==',
}
for fn, src in FILES.items():
    with open(os.path.join(CODE, fn), 'w') as f:
        f.write(src)
for rel, b64 in EMBED.items():
    dst = os.path.join(WORK, rel); os.makedirs(os.path.dirname(dst), exist_ok=True)
    with open(dst, 'wb') as f:
        f.write(gzip.decompress(base64.b64decode(b64)))
say(f'wrote {len(FILES)} code files and {len(EMBED)} reference files')
if PRETRAINED:
    say('downloading the Inception-v3 weights once (torchvision)')
    import torchvision.models as tvm
    tvm.inception_v3(weights=tvm.Inception_V3_Weights.DEFAULT)

# ---------------------------------------------------------------- 4. config
full = dict(n_eval=1000, n_noise_eval=1000, n_pool=500, sweep_n=32, sweep_levels=11, cfg_n=128,
            cfg_scales=[0.0, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.5, 6.0], ddim_steps=DDIM_STEPS, cfg_scale=3.5,
            n_real_ben=2000, n_real_mel_max=None)
dry = dict(full, n_eval=16, n_pool=8, sweep_n=6, sweep_levels=3, cfg_n=4, cfg_scales=[0.0, 3.5], n_real_ben=48, n_real_mel_max=48)
ncpu = os.cpu_count() or 4
cfg = {
    'work': WORK, 'outputs': OUTD, 'paths': paths, 'image_size': IMAGE_SIZE, 'eval_size': EVAL_SIZE, 'incep_size': INCEP_SIZE,
    'pretrained': PRETRAINED, 'cpu_workers': max(1, min(32, ncpu - 2)), 'cpu_workers_io': max(2, min(32, ncpu)),
    'dry_run': DRY_RUN, 'hardware': HW, 'n_devices': int(N_DEV), 'torch': TORCH_V, 'torch_xla': XLA_V,
    'sizes': dry if DRY_RUN else full,
    'seeds': {'eval_mel': 1001, 'eval_ben': 1002, 'sweep': 2001, 'cfg': 3001, 'realc_mel': 4001, 'realc_ben': 4002,
              'target_mel': 7001, 'target_ben': 7002, 'target_pooled': 7003, 'target_sweep_mel': 7004, 'target_sweep_ben': 7005,
              'target_cfg_mel': 7006, 'target_cfg_ben': 7007},
    'xla': {'gen_batch': GEN_BATCH, 'score_batch': 64, 'gen_precision': GEN_PRECISION, 'score_precision': 'highest', 'hw_check_n': 4},
    'calibration': {"main|mel": {"W": [[30.9926700592041, 2.5316219329833984, 2.1724252700805664], [2.2206430435180664, 29.62668800354004, 1.7066510915756226], [0.9334173202514648, 1.9682159423828125, 30.009281158447266]], "b": [-0.05300125852227211, 0.0035169555339962244, 0.012779776006937027], "fit_r2": 0.8944032192230225}, "main|ben": {"W": [[48.36914825439453, 1.2046619653701782, 1.2589964866638184], [2.4488365650177, 45.28070831298828, 2.064800262451172], [0.934712827205658, 1.8212356567382812, 44.709842681884766]], "b": [0.012255029752850533, -0.03020394966006279, -0.007765494287014008], "fit_r2": 0.9043594002723694}, "base|mel": {"W": [[44.792274475097656, 2.068265914916992, 1.294365406036377], [2.991440773010254, 40.74657440185547, 1.9280140399932861], [0.8379920721054077, 2.480213165283203, 43.105262756347656]], "b": [0.013931483030319214, -0.033029839396476746, -0.0021379292011260986], "fit_r2": 0.9059123992919922}, "base|ben": {"W": [[47.52669906616211, 2.0041096210479736, 0.9458061456680298], [3.5854575634002686, 44.009300231933594, 2.3902764320373535], [1.0723220109939575, 2.0042877197265625, 44.46895217895508]], "b": [0.06852088123559952, -0.05352824553847313, -0.012109987437725067], "fit_r2": 0.907301664352417}, "realc|mel": {"W": [[42.051212310791016, 1.6122329235076904, 2.172402858734131], [2.19482421875, 39.857154846191406, 1.3308396339416504], [1.0457937717437744, 2.2469863891601562, 41.342201232910156]], "b": [0.005050957202911377, 0.011044031009078026, 0.011632918380200863], "fit_r2": 0.8911113739013672}, "realc|ben": {"W": [[45.277183532714844, 1.1170908212661743, 1.1978789567947388], [2.3206896781921387, 41.998138427734375, 2.004533052444458], [1.007563829421997, 1.8944206237792969, 41.53213882446289]], "b": [0.0036738021299242973, -0.006223663687705994, -0.0009653735905885696], "fit_r2": 0.8937494158744812}},
    'analysis': {'n_boot': 200 if DRY_RUN else 2000, 'n_boot_auc': 100 if DRY_RUN else 1000,
                 'kid_subset': 500, 'kid_subset_small': 100, 'kid_n_sub': 10 if DRY_RUN else 100},
    'gate1': {'alpha': 0.05, 'min_evaluator_auc': 0.70, 'text':
              'PASS if CBM-B shows a positive, significant dose-response (two-sided Wilcoxon signed-rank test on '
              'per-trajectory slopes, BH-corrected p < 0.05, median slope > 0) for BOTH pigmentation and pigment network '
              'from the melanoma prototype, and the available non-CBM checks agree (lesion darkness for pigmentation; '
              'ISIC 2018 detector for pigment network). INCONCLUSIVE if CBM-B test AUC (128 pipeline) < 0.70 for either '
              'concept. PARTIAL if CBM-B passes but a non-CBM check fails. FAIL otherwise.'},
    'decision': {'max_colour_outliers': 20.0, 'kid_tolerance': 0.10, 'alpha': 0.05},
}
CFG = os.path.join(WORK, 'cfg.json')
with open(CFG, 'w') as f:
    json.dump(cfg, f, indent=2)

# ---------------------------------------------------------------- 5. stages (one at a time; stopping the cell stops the stage)
STATUS = {}


PROG = {}


def progress(job, t0):
    try:
        if job == 'gen':
            if 'gen' not in PROG:
                import torch
                PROG['gen'] = len(torch.load(os.path.join(WORK, 'plan', 'plan.pt'), weights_only=False)['chunks'])
            n, d = PROG['gen'], len(glob.glob(os.path.join(WORK, 'chunks_gen', '*.pt')))
        elif job == 'score':
            n = len(json.load(open(os.path.join(WORK, 'plan', 'score_chunks.json'))))
            d = len(glob.glob(os.path.join(WORK, 'chunks_score', '*.npz')))
        else:
            return ''
        if not isinstance(PROG.get(job + '_start'), tuple):
            PROG[job + '_start'] = (time.time(), d)
        ts, d0 = PROG[job + '_start']; rate = (d - d0) / max(time.time() - ts, 1)
        eta = f", about {(n - d) / rate / 60:.0f} min left at the current speed" if rate > 0 and d > d0 else ''
        return f"{d}/{n} chunks{eta}"
    except Exception:
        return ''


def last_line(job):
    p = os.path.join(WORK, 'logs', f'{job}.log')
    if not os.path.exists(p): return ''
    with open(p, 'rb') as f:
        f.seek(0, 2); n = f.tell(); f.seek(max(0, n - 6000))
        lines = [l for l in f.read().decode('utf8', 'ignore').splitlines() if l.strip() and 'WARNING' not in l]
    return lines[-1][-120:] if lines else ''


def run_stage(job, script='worker1d.py', tpu=False):
    if script == 'worker1d.py' and os.path.exists(os.path.join(WORK, 'done', f'{job}.ok')):
        STATUS[job] = 'done earlier'; say(f'{job}: done earlier, skipped'); return
    say(f'===== {job} =====')
    env = {**(XENV if tpu else os.environ), 'PYTHONUNBUFFERED': '1', 'P1D_RUN_ID': RUN_ID}
    args = [sys.executable, '-u', os.path.join(CODE, script)] + ([job, CFG] if script == 'worker1d.py' else [CFG])
    t0 = time.time()
    with open(os.path.join(WORK, 'logs', f'{job}.log'), 'a') as lf:
        proc = subprocess.Popen(args, stdout=lf, stderr=subprocess.STDOUT, env=env, cwd=CODE, start_new_session=True)
        try:
            tick = 0
            while proc.poll() is None:
                time.sleep(5); tick += 5
                if tick % 60 == 0:
                    print(f"    {job}: {(time.time() - t0) / 60:.1f} min | {progress(job, t0)} | {last_line(job)}", flush=True)
        except KeyboardInterrupt:
            os.killpg(proc.pid, signal.SIGTERM); time.sleep(3)
            try: os.killpg(proc.pid, signal.SIGKILL)
            except Exception: pass
            raise
    rc = proc.returncode
    STATUS[job] = 'ok' if rc == 0 else f'FAILED (exit {rc})'
    say(f'{job}: {STATUS[job]} after {(time.time() - t0) / 60:.1f} min')
    if rc != 0:
        print('---- last lines of the log ----'); os.system(f"tail -n 40 {os.path.join(WORK, 'logs', job + '.log')}")
        raise RuntimeError(f'stage {job} failed; see {os.path.join(WORK, "logs", job + ".log")}')


T0 = time.time()
run_stage('prep')
run_stage('gen', tpu=True)
run_stage('assemble')
run_stage('score', tpu=True)
run_stage('finalize')
run_stage('analysis', script='analysis.py')
rep = os.path.join(OUTD, 'phase1_report.md')
if os.path.exists(rep):
    txt = open(rep).read()
    with open(os.path.join(OUTD, 'phase1d_full_analysis_report.md'), 'w') as f:
        f.write('# Phase 1d: complete Phase 1 analysis of the V2 image sets (TPU)\n\n'
                '_Section numbers follow the Phase 1 report. Reproduction (section 2) does not apply to Phase 1d._\n\n' + txt)
    os.remove(rep)
run_stage('analysis1d', script='analysis1d.py')

# ---------------------------------------------------------------- 6. package
for fn in FILES: shutil.copy(os.path.join(CODE, fn), os.path.join(OUTD, 'code_' + fn))
shutil.copy(CFG, os.path.join(OUTD, 'cfg.json'))
os.makedirs(os.path.join(OUTD, 'logs'), exist_ok=True)
for lg in glob.glob(os.path.join(WORK, 'logs', '*.log')): shutil.copy(lg, os.path.join(OUTD, 'logs'))
for fn in glob.glob(os.path.join(WORK, 'results', '*')):
    if not fn.endswith('.pt'): shutil.copy(fn, os.path.join(OUTD, 'results'))
for d in ['chunks_gen', 'chunks_score']:   # the assembled sets and scores hold the same data
    shutil.rmtree(os.path.join(WORK, d), ignore_errors=True)
zip_base = os.path.join(WORK_ROOT, 'phase1d_dryrun_outputs' if DRY_RUN else 'phase1d_outputs')
shutil.make_archive(zip_base, 'zip', OUTD)
print('\n' + '=' * 78)
print(f"PHASE 1d {'DRY RUN ' if DRY_RUN else ''}FINISHED in {(time.time() - T0) / 60:.1f} min on {HW}")
for j, s in STATUS.items(): print(f"  {j:<11} {s}")
rep = os.path.join(OUTD, 'phase1d_report.md')
print('=' * 78); print(open(rep).read() if os.path.exists(rep) else 'No report: see the analysis logs.')
print(f"\nDownload: {zip_base}.zip  (tables, figures, reports, logs, code)")
print(f"V2 image sets stay in {os.path.join(WORK, 'sets')} (in the notebook output after a commit).")
