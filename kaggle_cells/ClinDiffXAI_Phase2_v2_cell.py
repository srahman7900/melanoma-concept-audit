# =============================================================================
# ClinDiffXAI  PHASE 2: counterfactual concept audit of melanoma classifiers
#                                          (one Kaggle cell, GPU T4 x2)
# =============================================================================
# WHERE TO RUN
#   In the notebook session where Phase 1d ran, below the Phase 1d cell. The cell
#   reads /kaggle/working/phase1d (V2 images, plan, scores). If that folder is not
#   there, attach the committed Phase 1d output instead. Run this cell
#   interactively; do NOT commit this notebook (a commit runs Phase 1d again).
#   Download phase2_outputs.zip when it finishes.
#
# WHAT THIS CELL DOES
#   1. Trains the classifiers under audit on real ISIC 2020 images (all melanomas
#      plus 10 benign per melanoma, 128-pixel pipeline, test split by patient
#      when patient ids are found): ResNet-50, EfficientNet-B0, ViT-B/16, two
#      seeds each; plus ResNet-50 on Derm7pt if Derm7pt is attached.
#      Controls of the audit itself:
#        positive: ResNet-50 trained to detect high versus low pigmentation, and
#                  high versus low blue-whitish veil (pseudo-labels);
#        negative: ResNet-50 trained with permuted melanoma labels.
#   2. Makes realistic-base sweeps (REALISTIC_SWEEPS): the real concept vectors of
#      64 real benign and 64 real melanoma cases (the Phase 1d eval_realc cases,
#      same start noise and colour target), one concept set to 0, 0.2, ..., 1.
#   3. Audits every classifier on these sweeps and on the Phase 1d prototype
#      sweeps: change of the melanoma logit per concept, decision flips at a
#      90%-sensitivity threshold, a concept/colour decomposition with CBM-B, the
#      colour sensitivity (Phase 1d pairs with different colour targets),
#      agreement with the 7-point checklist weights and with real-data
#      association, and agreement between seeds and architectures.
#
# AUDIT VALIDITY RULE (fixed before the run)
#   VALID if (a) each positive control has its target concept as the largest
#   effect, positive and significant (two-sided Wilcoxon signed-rank test on
#   per-trajectory slopes, BH-corrected p < 0.05, median slope > 0), in at least
#   3 of 4 bases (or the same share of the available bases), and (b) the largest
#   |effect| of the permuted-label control is below 0.25 x the median largest
#   |effect| of the melanoma classifiers. NOT VALID otherwise.
#
# TIME (my estimate, not a measurement): 1.5 to 2 hours on T4 x2 with
#   REALISTIC_SWEEPS = True; about 45 minutes less with False.
#
# VERSION 2 (after the first run): every audit score is computed in float32
#   (the first run used float16, and EfficientNet-B0 gave non-finite and extreme
#   outputs on some generated images); non-finite outputs are excluded and
#   logits are clipped at +/-15 before the effects; the 90%-sensitivity threshold
#   is computed from float32 validation logits. These are numerical fixes; the
#   audit-validity rule is unchanged. In the same session, this version reuses the
#   trained classifiers and the images and only scores and analyses again (my
#   estimate: 10 to 20 minutes).
# =============================================================================

DRY_RUN          = False
FORCE_RERUN      = False
REALISTIC_SWEEPS = True

import os, sys, json, time, glob, shutil, subprocess, uuid, signal

WORK_ROOT = os.environ.get('P1_WORK_ROOT', '/kaggle/working')
INPUT_ROOT = os.environ.get('P1_INPUT_ROOT', '/kaggle/input')


def say(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------- 0. Phase 1d output
def is_1d(d):
    return all(os.path.exists(os.path.join(d, p)) for p in ['cfg.json', 'plan/plan.pt', 'sets/eval_realc_mel.pt', 'sets/real_mel.pt'])


W1D = os.environ.get('P2_WORK1D')
if not W1D:
    for d in [os.path.join(WORK_ROOT, 'phase1d')] + ([os.path.join(WORK_ROOT, 'phase1d_dryrun')] if DRY_RUN else []):
        if is_1d(d): W1D = d; break
if not W1D and os.path.isdir(INPUT_ROOT):
    for dp, dn, fn in os.walk(INPUT_ROOT):
        if os.path.basename(dp) == 'phase1d' and is_1d(dp): W1D = dp; break
        if dp.count(os.sep) - INPUT_ROOT.count(os.sep) >= 4: dn[:] = []
if not W1D:
    raise FileNotFoundError('Phase 1d output not found. Run this cell in the session where Phase 1d ran, or attach '
                            'the committed Phase 1d notebook output.')
cfg1d = json.load(open(os.path.join(W1D, 'cfg.json')))
if cfg1d.get('dry_run') and not DRY_RUN and not os.environ.get('P2_WORK1D'):
    raise RuntimeError('The Phase 1d folder found is a dry run. Use the full Phase 1d output, or set DRY_RUN = True.')
say(f'Phase 1d found at {W1D}')

WORK = os.path.join(WORK_ROOT, 'phase2_dryrun' if DRY_RUN else 'phase2')
CODE = os.path.join(WORK, 'code'); OUTD = os.path.join(WORK, 'outputs'); WORKER = os.path.join(CODE, 'worker2.py')
os.makedirs(WORK, exist_ok=True)
RUN_ID = uuid.uuid4().hex
with open(os.path.join(WORK, 'run_id.txt'), 'w') as f:
    f.write(RUN_ID)
for _ in range(2):   # stop worker processes of an interrupted earlier run (they would hold the GPUs)
    try:
        if subprocess.run(['pkill', '-f', WORKER], check=False).returncode == 0:
            say('stopped worker processes of an earlier run of this cell')
    except FileNotFoundError:
        pass
    time.sleep(2)
if FORCE_RERUN:
    for d in ['done', 'sets', 'scores', 'scores2', 'results', 'outputs', 'logs', 'plan', 'chunks_gen', 'cls', 'models']:
        shutil.rmtree(os.path.join(WORK, d), ignore_errors=True)
for d in [CODE, OUTD, os.path.join(WORK, 'logs'), os.path.join(WORK, 'done'), os.path.join(WORK, 'results')]:
    os.makedirs(d, exist_ok=True)

# ---------------------------------------------------------------- 1. inputs
paths = dict(cfg1d['paths'])
missing = [k for k in ['cbma', 'main_e150', 'pseudo_csv', 'isic2020_dir', 'cbmb'] if not paths.get(k) or not os.path.exists(paths[k])]
if missing:
    raise FileNotFoundError(f'Inputs from Phase 1d are not attached any more: {missing}. Attach the same datasets as Phase 1d.')
say(f'indexing {INPUT_ROOT} for patient ids and Derm7pt ...')
csvs, metas = [], []
for dp, dn, fn in os.walk(INPUT_ROOT):
    if len(fn) > 2000 and not any(f.endswith('.csv') for f in fn[:200]):
        continue
    for f in fn:
        if f.endswith('.csv'): csvs.append(os.path.join(dp, f))
        if f == 'meta.csv': metas.append(os.path.join(dp, f))


def header(p):
    try:
        with open(p, 'r', errors='ignore') as fh:
            return fh.readline().strip().lower()
    except Exception:
        return ''


if 'patient_id' in header(paths['pseudo_csv']):
    paths['isic2020_meta'] = None; pid_note = 'patient_id found in the pseudo-label CSV'
else:
    c = [p for p in csvs if 'image_name' in header(p) and 'patient_id' in header(p)]
    paths['isic2020_meta'] = c[0] if c else None
    pid_note = f'patient ids from {c[0]}' if c else 'NO patient ids found: the classifier test split will be by image'
d7 = [p for p in metas if os.path.exists(os.path.join(os.path.dirname(p), 'train_indexes.csv'))]
if d7:
    md = os.path.dirname(d7[0]); paths['derm7pt_csv'] = d7[0]; paths['derm7pt_img_dir'] = os.path.join(os.path.dirname(md), 'images')
    for k in ['train', 'valid', 'test']:
        paths[f'derm7pt_{k}_idx'] = os.path.join(md, f'{k}_indexes.csv')
print('\n' + '=' * 78 + '\nINPUTS\n' + '=' * 78)
for k in ['cbma', 'main_e150', 'pseudo_csv', 'isic2020_dir', 'cbmb', 'isic18']:
    print(f"  {'OK ' if paths.get(k) else '-- '} {k:<13} {paths.get(k) or 'NOT FOUND'}")
print(f"  note: {pid_note}")
print(f"  {'OK ' if d7 else '-- '} Derm7pt       {d7[0] if d7 else 'not attached: the Derm7pt classifier is skipped'}")

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
say(f'hardware: {HW}; {len(SHARDS)} parallel worker(s)')
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
    if spec['label'] in ('pig', 'bwv'):
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
AUDIT_1D = ['sweep_mel', 'sweep_ben'] + [f'{p}_{g}_{c}' for p in ['eval', 'pool'] for g in ['main', 'base', 'realc'] for c in ['mel', 'ben']]
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
    'analysis2.py': r'''"""ClinDiffXAI Phase 2 analysis: counterfactual concept audit of melanoma classifiers.
Usage: python analysis2.py <cfg.json>"""
import os, sys, json, time, traceback
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import CONCEPT_NAMES, SHORT, load_cfg, load_set, save_json, log
import analysis as A
from analysis import plt, stats, Out, fci, fp, INK, INK2, DIV

REP, WARN = [], []
SEVEN_POINT = {'pigment_network': 2, 'blue_whitish_veil': 2, 'vascular_structures': 2, 'pigmentation': 1,
               'streaks': 1, 'dots_and_globules': 1, 'regression_structures': 1}   # Argenziano et al. 1998
BASES = [('real_ben', 'rsweep_ben', 'Real benign case'), ('real_mel', 'rsweep_mel', 'Real melanoma case'),
         ('proto_ben', 'sweep_ben', 'Benign prototype'), ('proto_mel', 'sweep_mel', 'Melanoma prototype')]
ARCH_LAB = {'resnet50': 'ResNet-50', 'effb0': 'EfficientNet-B0', 'vitb16': 'ViT-B/16'}
ARCH_COL = {'resnet50': '#2a78d6', 'effb0': '#eb6834', 'vitb16': '#1baf7a'}
RNG = np.random.default_rng(2027)


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
    def __init__(self, cfg):
        self.cfg = cfg; self.meta_cache, self.score_cache = {}, {}
        self.models, self.res, self.lg = [], {}, {}
        C = cfg['analysis']['logit_clip']; self.numcheck = []
        yv = None
        for m in cfg['models']:
            fr = os.path.join(cfg['work'], 'results', f"train_{m['name']}.json"); fs = os.path.join(cfg['work'], cfg['scores_dir'], f"{m['name']}.npz")
            if os.path.exists(fr) and os.path.exists(fs):
                self.models.append(m); self.res[m['name']] = json.load(open(fr)); raw = dict(np.load(fs)); lg = {}
                for k, v in raw.items():
                    v = v.astype(np.float64); nf = ~np.isfinite(v)
                    self.numcheck.append({'classifier': m['name'], 'set': k, 'n': len(v), 'non_finite': int(nf.sum()),
                                          'abs_logit_gt_clip_pct': float(np.mean(np.abs(v[~nf]) > C) * 100) if (~nf).any() else np.nan,
                                          'min': float(np.nanmin(np.where(nf, np.nan, v))) if (~nf).any() else np.nan,
                                          'max': float(np.nanmax(np.where(nf, np.nan, v))) if (~nf).any() else np.nan})
                    v[nf] = np.nan; lg[k] = np.clip(v, -C, C)
                self.lg[m['name']] = lg
                if '_val' in lg:   # 90%-sensitivity threshold from float32 validation logits
                    if yv is None:
                        import torch
                        dat = torch.load(os.path.join(cfg['work'], 'cls', 'data.pt'), map_location='cpu', weights_only=False)
                        yv = dat['target'][np.array(dat['split']) == 'val']; del dat
                    pos = np.sort(lg['_val'][(yv == 1) & np.isfinite(lg['_val'])])
                    if len(pos): self.res[m['name']]['tau90_logit'] = float(pos[int(np.floor(0.10 * len(pos)))])
            else:
                WARN.append(f"classifier {m['name']} has no results (training or scoring did not finish)")
        self.mel = [m for m in self.models if m['label'] == 'mel']
        self.bases = [(k, s, lab) for k, s, lab in BASES if self.path(s)]

    def path(self, name):
        for root in [self.cfg['work'], self.cfg['work1d']]:
            p = os.path.join(root, 'sets', f'{name}.pt')
            if os.path.exists(p): return p
        return None

    def meta(self, name):
        if name not in self.meta_cache:
            self.meta_cache[name] = pd.DataFrame(load_set(self.path(name))['meta'])
        return self.meta_cache[name]

    def imgs(self, name):
        return load_set(self.path(name))['imgs']

    def scores(self, name):
        if name not in self.score_cache:
            d = {}
            for root in [self.cfg['work'], self.cfg['work1d']]:
                for t in ['A', 'B']:
                    f = os.path.join(root, 'scores', f'{t}_{name}.npz')
                    if os.path.exists(f) and t not in d: d[t] = dict(np.load(f))
            self.score_cache[name] = d
        return self.score_cache[name]


def trajectories(meta, y, j):
    """(n_traj, n_levels) array of y for concept j; trajectory = noise_id"""
    sel = meta['concept_idx'].values == j
    lev = np.sort(np.unique(meta['level'].values[sel])); nid = np.sort(np.unique(meta['noise_id'].values[sel]))
    T = np.full((len(nid), len(lev)), np.nan)
    T[np.searchsorted(nid, meta['noise_id'].values[sel]), np.searchsorted(lev, meta['level'].values[sel])] = y[sel]
    return T, lev, nid


def mlabel(m):
    if m['label'] == 'mel':
        return f"{ARCH_LAB[m['arch']]} s{m['seed']}" + (' (Derm7pt)' if m['data'] == 'derm7pt' else '')
    return {'pig': 'Control+: pigmentation', 'bwv': 'Control+: blue-whitish veil', 'perm': 'Control-: permuted labels'}[m['label']]


# -----------------------------------------------------------------------------
@section('Numerical check of the classifier outputs')
def s_numeric(cfg, out, R, D):
    df = pd.DataFrame(D.numcheck); C = cfg['analysis']['logit_clip']
    if not len(df): return
    df['group'] = np.where(df['set'].isin(['cls_test', '_val']), 'real images', 'generated images')
    g = df.groupby(['classifier', 'group']).apply(lambda x: pd.Series({
        'images': int(x['n'].sum()), 'non-finite': int(x['non_finite'].sum()),
        f'% |logit| > {C:g}': round(float(np.nansum(x['abs_logit_gt_clip_pct'] * x['n']) / x['n'].sum()), 2),
        'min logit': round(float(x['min'].min()), 1), 'max logit': round(float(x['max'].max()), 1)})).reset_index()
    g['images'] = g['images'].astype(int); g['non-finite'] = g['non-finite'].astype(int)
    out.table(g, 'TP0_numerical_check', f'Classifier outputs in float32. Non-finite values are excluded; logits are clipped '
              f'at +/-{C:g} (p(melanoma) between 3e-7 and 1 - 3e-7) before every effect, so that a few extreme outputs on '
              f'generated images cannot dominate the means.')
    R['numerical_check'] = g.to_dict(orient='list')
    bad = g[(g['group'] == 'generated images') & ((g['non-finite'] > 0) | (g[f'% |logit| > {C:g}'] > 0.5))]
    REP.append("\n## 0. Numerical check (float32 scores)\n\n" + g.to_string(index=False) +
               ("\n\n- Classifiers with extreme outputs on generated images: " + ', '.join(sorted(set(bad['classifier']))) if len(bad) else
                "\n\n- No classifier has extreme outputs on generated images."))


@section('Classifiers under audit')
def s_classifiers(cfg, out, R, D):
    B = cfg['analysis']['n_boot_auc']; cd = json.load(open(os.path.join(cfg['work'], 'results', 'cls_data.json')))
    tm = D.meta('cls_test'); yt = tm['target'].values; rows = []
    for m in D.models:
        r = D.res[m['name']]; lt = D.lg[m['name']]['cls_test']; tau = r['tau90_logit']
        a = A.auc_ci(lt[yt == 1], lt[yt == 0], B)
        r['test_sens_at_tau'] = float(np.mean(lt[yt == 1] >= tau)); r['test_spec_at_tau'] = float(np.mean(lt[yt == 0] < tau))
        row = {'Classifier': mlabel(m), 'Architecture': ARCH_LAB[m['arch']], 'Label': m['label'], 'Data': m['data'],
               'n train (positive)': f"{r['n_train']} ({r['n_train_pos']})", 'Best val AUC': round(r['best_val_auc'], 3),
               'Test AUC, melanoma [95% CI]': fci(*a), 'Sens / spec at tau90': f"{r['test_sens_at_tau']:.2f} / {r['test_spec_at_tau']:.2f}"}
        for g, gl in [('main', 'prototype'), ('base', 'baseline'), ('realc', 'real vectors')]:
            if f'eval_{g}_mel' in D.lg[m['name']]:
                a1, b1 = D.lg[m['name']][f'eval_{g}_mel'], D.lg[m['name']][f'eval_{g}_ben']
                row[f'AUC on V2 images ({gl})'] = round(A.auc_fast(a1[np.isfinite(a1)], b1[np.isfinite(b1)]), 3)
        if 'test_auc_own_label' in r: row['AUC own label (test)'] = round(r['test_auc_own_label'], 3)
        if 'derm7pt_test_auc' in r: row['AUC Derm7pt test'] = round(r['derm7pt_test_auc'], 3)
        rows.append(row)
    df = pd.DataFrame(rows).fillna(''); R['classifiers'] = df.to_dict(orient='list')
    split = 'patient' if cd['have_patient_ids'] else 'IMAGE (no patient ids found)'
    out.table(df, 'TP1_classifiers', f"Classifiers under audit. Trained on real ISIC 2020 images at the 128-pixel pipeline "
              f"({cd['n']} images, {cd['n_mel']} melanoma), split by {split}: train/val/test = "
              f"{cd['split_counts']['train']}/{cd['split_counts']['val']}/{cd['split_counts']['test']}. tau90: threshold with 90% "
              f"sensitivity on the validation set. AUC on V2 images: melanoma-conditioned versus benign-conditioned Phase 1d images.")
    REP.append(f"\n## 1. Classifiers under audit\n\n- Split by {split}; test set {cd['split_counts']['test']} images "
               f"({cd['split_mel']['test']} melanoma).\n\n" + df.to_string(index=False))
    if not cd['have_patient_ids']:
        WARN.append('No patient ids: the classifier test split is by image, so images of one patient can be in train and test.')


def measured_change(D, sname, j):
    """mean change (level 1 minus level 0) of the CBM-B output for concept j along its own sweep, in SD units of
    CBM-B on real test images"""
    key = (sname, j)
    if not hasattr(D, '_meas'): D._meas = {}
    if key in D._meas: return D._meas[key]
    try:
        sd = np.std(D.scores('cls_test')['B']['cbmb'][:, j]) + 1e-9
        T, lev, nid = trajectories(D.meta(sname), D.scores(sname)['B']['cbmb'][:, j], j)
        v = float(np.nanmean(T[:, -1] - T[:, 0]) / sd)
    except Exception:
        v = np.nan
    D._meas[key] = v
    return v


def effects(D, m, set_name):
    """per concept: dict with trajectories of logits"""
    meta = D.meta(set_name); y = D.lg[m['name']][set_name]; out = {}
    for j in range(7):
        T, lev, nid = trajectories(meta, y, j)
        ok = np.isfinite(T).all(1); out[j] = (T[ok], lev, nid[ok])
    return out


@section('Counterfactual concept effects')
def s_effects(cfg, out, R, D):
    Bn = cfg['analysis']['n_boot']; rows = []
    for m in D.models:
        for bk, sname, blab in D.bases:
            fam = []
            for j, (T, lev, nid) in effects(D, m, sname).items():
                if len(T) < 5: continue
                slope, rho = A.traj_stats(T, lev); p, rb = A.one_sample_test(slope)
                dl = T[:, -1] - T[:, 0]; dm, dlo, dhi = A.boot_mean(dl, Bn)
                pr = 1 / (1 + np.exp(-T)); dp = pr[:, -1] - pr[:, 0]
                meas = measured_change(D, sname, j)
                fam.append({'classifier': m['name'], 'label': m['label'], 'arch': m['arch'], 'seed': m['seed'], 'data': m['data'],
                            'base': bk, 'concept': CONCEPT_NAMES[j], 'n_traj': int(len(T)), 'median_slope': float(np.median(slope)),
                            'frac_rising': float(np.mean(slope > 0)), 'p': p, 'r_rb': rb, 'dlogit': dm, 'dlogit_lo': dlo,
                            'dlogit_hi': dhi, 'dlogit_median': float(np.median(dl)), 'dprob': float(np.mean(dp)),
                            'measured_change_sd': meas,
                            'dlogit_per_sd_measured': dm / meas if np.isfinite(meas) and meas >= cfg['analysis']['min_measured_sd'] else np.nan,
                            'curve': np.mean(pr, 0).tolist(), 'levels': lev.tolist()})
            q = A.bh([f['p'] for f in fam])
            for f, qq in zip(fam, q): f['p_bh'] = float(qq)
            rows += fam
    R['effects'] = [{k: v for k, v in r.items() if k not in ('curve',)} for r in rows]
    D.effects = rows
    df = pd.DataFrame([{k: v for k, v in r.items() if k not in ('curve', 'levels')} for r in rows])
    df.to_csv(os.path.join(out.root, 'tables', 'TP2_counterfactual_effects_all.csv'), index=False)
    tab = pd.DataFrame([{'Classifier': mlabel(next(m for m in D.models if m['name'] == r['classifier'])),
                         'Base': dict((k, l) for k, _, l in BASES)[r['base']], 'Concept': r['concept'],
                         'Δ logit 0→1 [95% CI]': fci(r['dlogit'], r['dlogit_lo'], r['dlogit_hi'], 2), 'Δ p(melanoma)': f"{r['dprob']:+.3f}",
                         'Median Δ logit': f"{r['dlogit_median']:+.2f}", 'Measured change (CBM-B, SD)': f"{r['measured_change_sd']:+.2f}",
                         'Δ logit per SD measured': f"{r['dlogit_per_sd_measured']:+.2f}" if np.isfinite(r['dlogit_per_sd_measured']) else 'n/a',
                         'Median slope': f"{r['median_slope']:+.2f}", '% rising': f"{r['frac_rising'] * 100:.0f}%", 'p (BH)': fp(r['p_bh'])}
                        for r in rows])
    out.table(tab, 'TP2_counterfactual_effects', 'Counterfactual concept effects. For each starting case (trajectory), one '
              'concept is set from 0 to 1 while the other six keep their values and the start noise and colour target stay '
              'the same. Δ logit: change of the classifier melanoma logit (float32, clipped at +/-%g) from level 0 to level 1 '
              '(mean over trajectories, bootstrap CI). Measured change: mean change of the CBM-B output for the same concept, '
              'in SD units of real test images; Δ logit per SD measured is given when the measured change is at least %g SD. '
              'p: two-sided Wilcoxon signed-rank test of per-trajectory OLS slopes, BH-corrected within each classifier and '
              'base. The effect includes the cross-talk of the generator (Phase 1).' % (cfg['analysis']['logit_clip'], cfg['analysis']['min_measured_sd']))
    # heat maps: mean over seeds of the same architecture, one panel per base
    groups = []
    for m in D.models:
        key = (m['arch'], m['label'], m['data'])
        if key not in [g[0] for g in groups]: groups.append((key, mlabel(m).replace(f" s{m['seed']}", '')))
    fig, axes = plt.subplots(1, len(D.bases), figsize=(2.2 * len(D.bases) + 1.2, 0.36 * len(groups) + 1.6), squeeze=False)
    vmax = max([abs(r['dlogit']) for r in rows if np.isfinite(r['dlogit'])] + [1e-6]); vmax = min(vmax, 6)
    for ax, (bk, sname, blab) in zip(axes[0], D.bases):
        M = np.full((len(groups), 7), np.nan)
        for gi, (key, _) in enumerate(groups):
            for j in range(7):
                v = [r['dlogit'] for r in rows if (r['arch'], r['label'], r['data']) == key and r['base'] == bk and r['concept'] == CONCEPT_NAMES[j]]
                if v: M[gi, j] = np.mean(v)
        im = ax.imshow(M, cmap=DIV.reversed(), vmin=-vmax, vmax=vmax, aspect='auto'); ax.grid(False)
        for a in range(M.shape[0]):
            for b in range(7):
                if np.isfinite(M[a, b]): ax.text(b, a, f"{M[a, b]:.1f}", ha='center', va='center', fontsize=5.5, color=INK)
        ax.set_xticks(range(7)); ax.set_xticklabels(SHORT, fontsize=6); ax.set_title(blab, fontsize=7)
        ax.set_yticks(range(len(groups)))
        ax.set_yticklabels([g[1] for g in groups] if ax is axes[0][0] else [''] * len(groups), fontsize=6)
    cb = fig.colorbar(im, ax=list(axes[0]), fraction=0.02, pad=0.01); cb.set_label('Δ melanoma logit, concept 0 → 1', fontsize=7)
    out.fig(fig, 'figP_effects_heatmap')
    # dose-response curves for the most realistic base
    bk, sname, blab = D.bases[0]
    fig, axes = plt.subplots(2, 4, figsize=(7.2, 3.8), sharex=True, sharey=True); axes = axes.ravel()
    for j in range(7):
        ax = axes[j]
        for arch in ARCH_LAB:
            cs = [r for r in rows if r['arch'] == arch and r['label'] == 'mel' and r['data'] == 'isic2020' and r['base'] == bk and r['concept'] == CONCEPT_NAMES[j]]
            if cs:
                ax.plot(cs[0]['levels'], np.mean([c['curve'] for c in cs], 0), color=ARCH_COL[arch], marker='o', ms=2.5, label=ARCH_LAB[arch])
        ax.set_title(f"{SHORT[j]} ({CONCEPT_NAMES[j].replace('_', ' ')})", fontsize=7)
    axes[7].axis('off'); h, l = axes[0].get_legend_handles_labels(); axes[7].legend(h, l, loc='center', fontsize=7)
    fig.supylabel('Mean p(melanoma)', fontsize=7); fig.supxlabel(f'Commanded concept level ({blab.lower()} as the start)', fontsize=7)
    fig.tight_layout(); out.fig(fig, f'figP_dose_response_{bk}')
    REP.append("\n## 2. Counterfactual concept effects (mean Δ logit 0→1, logits clipped at +/-%g; melanoma classifiers, "
               "mean over classifiers)\n" % cfg['analysis']['logit_clip'])
    for bk, sname, blab in D.bases:
        s = []
        for j in range(7):
            v = [r['dlogit'] for r in rows if r['label'] == 'mel' and r['base'] == bk and r['concept'] == CONCEPT_NAMES[j]]
            sig = [r['p_bh'] < 0.05 and r['median_slope'] > 0 for r in rows if r['label'] == 'mel' and r['base'] == bk and r['concept'] == CONCEPT_NAMES[j]]
            if v: s.append(f"{SHORT[j]} {np.mean(v):+.2f} ({sum(sig)}/{len(sig)} sig.)")
        REP.append(f"- {blab}: " + ', '.join(s))
        u = []
        for j in range(7):
            v = [r['dlogit_per_sd_measured'] for r in rows if r['label'] == 'mel' and r['base'] == bk and r['concept'] == CONCEPT_NAMES[j]]
            mc = next((r['measured_change_sd'] for r in rows if r['base'] == bk and r['concept'] == CONCEPT_NAMES[j]), np.nan)
            u.append(f"{SHORT[j]} {np.nanmean(v):+.2f} (knob moves CBM-B by {mc:+.2f} SD)" if v and np.isfinite(np.nanmean(v)) else
                     f"{SHORT[j]} n/a (knob moves CBM-B by {mc:+.2f} SD)")
        REP.append(f"  per SD of measured concept change: " + ', '.join(u))


@section('Decision flips at the 90%-sensitivity threshold')
def s_flips(cfg, out, R, D):
    rows = []
    for m in D.models:
        tau = D.res[m['name']]['tau90_logit']
        for bk, sname, blab in D.bases:
            for j, (T, lev, nid) in effects(D, m, sname).items():
                if len(T) == 0: continue
                lo, hi = T[:, 0] >= tau, T[:, -1] >= tau
                n_neg0 = int((~lo).sum()); n_pos1 = int(hi.sum())
                rows.append({'Classifier': mlabel(m), 'label': m['label'], 'name': m['name'], 'Base': blab, 'base': bk, 'Concept': CONCEPT_NAMES[j],
                             'p(mel) call at level 0': round(float(lo.mean()), 3), 'p(mel) call at level 1': round(float(hi.mean()), 3),
                             'Flip to melanoma (adding)': round(float((~lo & hi).sum() / max(n_neg0, 1)), 3) if n_neg0 else np.nan,
                             'Flip to benign (removing)': round(float((hi & ~lo).sum() / max(n_pos1, 1)), 3) if n_pos1 else np.nan,
                             'n traj': int(len(T))})
    df = pd.DataFrame(rows); R['flips'] = df.to_dict(orient='list')
    out.table(df.drop(columns=['label', 'name', 'base']), 'TP3_decision_flips', 'Decisions at tau90 (90% sensitivity on real '
              'validation images). Flip to melanoma: share of trajectories called benign at level 0 and melanoma at level 1. Flip to '
              'benign: share of trajectories called melanoma at level 1 and benign at level 0.')
    mm = df[df['label'] == 'mel']
    if len(mm):
        piv = mm.groupby(['Base', 'Concept'])['Flip to melanoma (adding)'].mean().unstack()
        REP.append("\n## 3. Decision flips (benign to melanoma when the concept is added; mean over melanoma classifiers)\n\n" +
                   piv.round(3).to_string())
        fig, ax = plt.subplots(figsize=(7.2, 2.4)); bases = list(piv.index); x = np.arange(7); w = 0.8 / max(len(bases), 1)
        for i, b in enumerate(bases):
            ax.bar(x + (i - (len(bases) - 1) / 2) * w, [piv.loc[b].get(c, np.nan) for c in CONCEPT_NAMES], w * 0.92, label=b,
                   color=['#2a78d6', '#eb6834', '#1baf7a', '#eda100'][i % 4], edgecolor='white')
        ax.set_xticks(x); ax.set_xticklabels(SHORT); ax.set_ylabel('Share of trajectories that flip\nbenign → melanoma', fontsize=7)
        ax.legend(fontsize=6, ncol=len(bases), loc='upper left', bbox_to_anchor=(0, 1.2)); fig.tight_layout(); out.fig(fig, 'figP_decision_flips')


def _deltas(D, m, sname):
    """Δlogit, ΔCBM-B (7) and ΔLab (3) of every image against the level-0 image of its trajectory"""
    meta = D.meta(sname); y = D.lg[m['name']][sname]; sc = D.scores(sname)
    Bc, L = sc['B']['cbmb'], sc['A']['lab']; out = []
    for j in range(7):
        sel = np.where(meta['concept_idx'].values == j)[0]
        lev = meta['level'].values[sel]; nid = meta['noise_id'].values[sel]
        for t in np.unique(nid):
            ii = sel[nid == t]; ll = meta['level'].values[ii]
            if len(ii) < 2: continue
            i0 = ii[np.argmin(ll)]
            for i in ii:
                if i == i0: continue
                out.append((f"{sname}|{j}|{t}", y[i] - y[i0], *(Bc[i] - Bc[i0]), *(L[i] - L[i0])))
    return out


@section('Decomposition of the classifier change into concept and colour parts')
def s_decomp(cfg, out, R, D):
    from sklearn.linear_model import Ridge
    Bn = cfg['analysis']['n_boot_reg']; names = [f'Δ CBM-B {s}' for s in SHORT] + ['Δ L*', 'Δ a*', 'Δ b*']; rows = []; D.decomp = {}
    for m in D.models:
        recs = []
        for bk, sname, blab in D.bases:
            if 'B' in D.scores(sname) and 'A' in D.scores(sname): recs += _deltas(D, m, sname)
        if len(recs) < 50: continue
        g = np.array([r[0] for r in recs]); Y = np.array([r[1] for r in recs]); X = np.array([r[2:] for r in recs], float)
        ok = np.isfinite(Y) & np.isfinite(X).all(1); g, Y, X = g[ok], Y[ok], X[ok]
        mu, sd = X.mean(0), X.std(0) + 1e-9; Z = (X - mu) / sd
        fit = Ridge(alpha=1.0).fit(Z, Y); coef = fit.coef_; r2 = fit.score(Z, Y)
        ug = np.unique(g); gi = {k: np.where(g == k)[0] for k in ug}; bs = []
        for _ in range(Bn):
            ii = np.concatenate([gi[k] for k in RNG.choice(ug, len(ug), replace=True)])
            bs.append(Ridge(alpha=1.0).fit(Z[ii], Y[ii]).coef_)
        lo, hi = np.percentile(bs, 2.5, 0), np.percentile(bs, 97.5, 0)
        D.decomp[m['name']] = coef
        row = {'Classifier': mlabel(m), 'R2': round(r2, 3), 'n images': len(Y)}
        for k, nm in enumerate(names): row[nm] = fci(coef[k], lo[k], hi[k], 2)
        rows.append(row)
    df = pd.DataFrame(rows); R['decomposition'] = df.to_dict(orient='list')
    out.table(df, 'TP4_decomposition_concept_colour', 'Ridge regression (alpha = 1, standardised predictors) of the change in '
              'melanoma logit along all sweeps on the measured changes: CBM-B concept outputs (independent evaluator) and image '
              'mean CIELAB. Coefficient: logit change per SD of the measured change. CI: bootstrap over trajectories. Shows which '
              'measured image changes carry the classifier response, including cross-talk and colour.')
    REP.append("\n## 4. Concept and colour parts of the classifier change\n\n" + df.to_string(index=False))
    # exploratory (defined after the pre-registered check): does the decomposition find each positive control's concept?
    ex = []
    for lab, j in [('pig', CONCEPT_NAMES.index('pigmentation')), ('bwv', CONCEPT_NAMES.index('blue_whitish_veil'))]:
        mm = [m for m in D.models if m['label'] == lab and m['name'] in D.decomp]
        if not mm: continue
        c = D.decomp[mm[0]['name']][:7]; top = int(np.argmax(c))
        ex.append(f"- {mlabel(mm[0])}: largest CBM-B coefficient {SHORT[top]} ({c[top]:+.2f}); target {SHORT[j]} {c[j]:+.2f} "
                  f"({'target is largest' if top == j else 'target is NOT largest'})")
    if ex:
        R['decomposition_control_check'] = ex
        REP.append("\nExploratory check (defined after the pre-registered check): positive controls in the decomposition\n\n" + '\n'.join(ex))


@section('Observational attribution on real test images')
def s_observational(cfg, out, R, D):
    from sklearn.linear_model import Ridge
    sc = D.scores('cls_test'); X = np.concatenate([sc['B']['cbmb'], sc['A']['lab']], 1)
    Z = (X - X.mean(0)) / (X.std(0) + 1e-9); rows = []; D.obs = {}
    for m in D.models:
        y = D.lg[m['name']]['cls_test']; fit = Ridge(alpha=1.0).fit(Z, y); D.obs[m['name']] = fit.coef_
        row = {'Classifier': mlabel(m), 'R2': round(fit.score(Z, y), 3)}
        for k, nm in enumerate([f'CBM-B {s}' for s in SHORT] + ['L*', 'a*', 'b*']): row[nm] = round(float(fit.coef_[k]), 3)
        rows.append(row)
    df = pd.DataFrame(rows); R['observational'] = df.to_dict(orient='list')
    out.table(df, 'TP5_observational_attribution', 'Observational attribution: ridge regression of the classifier melanoma '
              'logit on the CBM-B concept outputs and mean CIELAB of real held-out test images (standardised). It describes '
              'which image properties go together with the classifier output in real data; it is not an intervention.')


@section('Agreement of concept profiles')
def s_profiles(cfg, out, R, D):
    cd = json.load(open(os.path.join(cfg['work'], 'results', 'cls_data.json')))
    ref = {'7-point checklist weight': np.array([SEVEN_POINT[c] for c in CONCEPT_NAMES], float),
           'ISIC 2020 association (pseudo-labels)': np.array([cd['association_std_coef'][c] for c in CONCEPT_NAMES])}
    rows = []
    for m in D.mel:
        prof = {}
        for bk, _, blab in D.bases:
            v = [next((r['dlogit'] for r in D.effects if r['classifier'] == m['name'] and r['base'] == bk and r['concept'] == c), np.nan) for c in CONCEPT_NAMES]
            prof[f'Counterfactual, {blab.lower()}'] = np.array(v)
        if m['name'] in getattr(D, 'decomp', {}): prof['Decomposition (CBM-B part)'] = D.decomp[m['name']][:7]
        if m['name'] in getattr(D, 'obs', {}): prof['Observational (real test)'] = D.obs[m['name']][:7]
        for pk, pv in prof.items():
            row = {'Classifier': mlabel(m), 'Profile': pk}
            for rk, rv in ref.items():
                ok = np.isfinite(pv); row[f'ρ with {rk}'] = round(float(stats.spearmanr(pv[ok], rv[ok])[0]), 3) if ok.sum() >= 4 else np.nan
            if 'Observational (real test)' in prof and pk != 'Observational (real test)':
                ok = np.isfinite(pv); row['ρ with observational'] = round(float(stats.spearmanr(pv[ok], prof['Observational (real test)'][ok])[0]), 3)
            rows.append(row)
    df = pd.DataFrame(rows); R['profiles'] = df.to_dict(orient='list')
    out.table(df, 'TP6_profile_agreement', 'Spearman correlation across the 7 concepts between each classifier\'s concept '
              'profile and reference profiles. With 7 concepts these correlations are descriptive only (no test).')
    if len(df):
        summ = df.drop(columns=['Classifier']).groupby('Profile').mean(numeric_only=True).round(3)
        REP.append("\n## 5. Concept profiles versus references (mean Spearman ρ over melanoma classifiers; descriptive)\n\n" + summ.to_string())


@section('Colour sensitivity (same noise and concepts, different colour target)')
def s_colour(cfg, out, R, D):
    rows = []; npool = cfg['n_pool']
    for m in D.models:
        dl, dL = [], []
        for g in ['main', 'base', 'realc']:
            for c in ['mel', 'ben']:
                e, p = f'eval_{g}_{c}', f'pool_{g}_{c}'
                if e not in D.lg[m['name']] or p not in D.lg[m['name']]: continue
                k = min(npool, len(D.lg[m['name']][p]))
                dl.append(D.lg[m['name']][e][:k] - D.lg[m['name']][p][:k])
                dL.append(D.scores(e)['A']['lab'][:k] - D.scores(p)['A']['lab'][:k])
        if not dl: continue
        y = np.concatenate(dl); X = np.concatenate(dL) / 10.0
        Xd = np.c_[np.ones(len(X)), X]; beta = np.linalg.lstsq(Xd, y, rcond=None)[0]
        r2 = 1 - ((y - Xd @ beta) ** 2).sum() / ((y - y.mean()) ** 2).sum()
        conc = [abs(r['dlogit']) for r in getattr(D, 'effects', []) if r['classifier'] == m['name'] and r['base'] == D.bases[0][0]]
        rows.append({'Classifier': mlabel(m), 'n pairs': len(y), 'mean |Δ logit| from colour change': round(float(np.abs(y).mean()), 3),
                     'mean |Δ logit| concept 0→1 (' + D.bases[0][2].lower() + ')': round(float(np.mean(conc)), 3) if conc else np.nan,
                     'logit per +10 L*': round(float(beta[1]), 3), 'per +10 a*': round(float(beta[2]), 3), 'per +10 b*': round(float(beta[3]), 3),
                     'R2': round(float(r2), 3)})
    df = pd.DataFrame(rows); R['colour_sensitivity'] = df.to_dict(orient='list')
    out.table(df, 'TP7_colour_sensitivity', 'Colour sensitivity. Phase 1d image pairs with the same start noise and concept '
              'vector but a different colour target (class-specific versus class-agnostic). Linear regression of the change in '
              'melanoma logit on the change in image mean CIELAB (per 10 units).')
    REP.append("\n## 6. Colour sensitivity\n\n" + df.to_string(index=False))


@section('Controls and audit validity (rule fixed before the run)')
def s_controls(cfg, out, R, D):
    E = getattr(D, 'effects', []); res = {'rule': cfg['audit_rule']['text']}; lines = []
    need = int(np.ceil(cfg['audit_rule']['pos_fraction'] * len(D.bases)))
    pos_ok = []
    for lab, concept in [('pig', 'pigmentation'), ('bwv', 'blue_whitish_veil')]:
        mm = [m for m in D.models if m['label'] == lab]
        if not mm: pos_ok.append(False); lines.append(f"- positive control {lab}: MISSING"); continue
        top = 0
        for bk, _, blab in D.bases:
            rr = [r for r in E if r['classifier'] == mm[0]['name'] and r['base'] == bk]
            if not rr: continue
            best = max(rr, key=lambda r: r['dlogit'] if np.isfinite(r['dlogit']) else -1e9)
            tgt = next(r for r in rr if r['concept'] == concept)
            ok = best['concept'] == concept and tgt['p_bh'] < cfg['audit_rule']['alpha'] and tgt['median_slope'] > 0
            top += ok
            lines.append(f"- {mlabel(mm[0])}, {blab}: largest effect {best['concept']} ({best['dlogit']:+.2f}); "
                         f"{concept} {tgt['dlogit']:+.2f}, p_BH {fp(tgt['p_bh'])}")
        pos_ok.append(top >= need); res[f'positive_{lab}'] = {'bases_with_target_largest_and_significant': top, 'needed': need}
    neg = [m for m in D.models if m['label'] == 'perm']
    mel_max = [max([abs(r['dlogit']) for r in E if r['classifier'] == m['name'] and np.isfinite(r['dlogit'])] or [np.nan]) for m in D.mel]
    if neg:
        nmax = max([abs(r['dlogit']) for r in E if r['classifier'] == neg[0]['name'] and np.isfinite(r['dlogit'])] or [np.nan])
        ref = float(np.nanmedian(mel_max)) if mel_max else np.nan
        neg_ok = bool(np.isfinite(nmax) and np.isfinite(ref) and nmax < cfg['audit_rule']['neg_fraction'] * ref)
        res['negative'] = {'max_abs_dlogit': nmax, 'median_max_abs_dlogit_melanoma_classifiers': ref, 'ok': neg_ok}
        lines.append(f"- negative control (permuted labels): max |Δ logit| {nmax:.3f} versus {cfg['audit_rule']['neg_fraction']:.2f} x "
                     f"{ref:.3f} = {cfg['audit_rule']['neg_fraction'] * ref:.3f} (median over melanoma classifiers)")
    else:
        neg_ok = False; lines.append('- negative control: MISSING')
    verdict = 'VALID' if all(pos_ok) and neg_ok else 'NOT VALID'
    if cfg.get('dry_run'): verdict = 'DRY RUN ONLY, not a decision. Would read: ' + verdict
    res['verdict'] = verdict; R['audit_validity'] = res
    save_json(res, os.path.join(out.root, 'results', 'audit_validity.json'))
    REP.insert(0, f"**Audit validity (rule fixed before the run): {verdict}**\n\nRule: {cfg['audit_rule']['text']}\n\n" + '\n'.join(lines) + '\n')


@section('Agreement between seeds and architectures')
def s_reliability(cfg, out, R, D):
    E = getattr(D, 'effects', []); vec = {}
    for m in D.mel:
        vec[m['name']] = np.array([next((r['dlogit'] for r in E if r['classifier'] == m['name'] and r['base'] == bk and r['concept'] == c), np.nan)
                                   for bk, _, _ in D.bases for c in CONCEPT_NAMES])
    rows = []
    names = list(vec)
    for i in range(len(names)):
        for k in range(i + 1, len(names)):
            a, b = vec[names[i]], vec[names[k]]; ok = np.isfinite(a) & np.isfinite(b)
            ma = next(m for m in D.mel if m['name'] == names[i]); mb = next(m for m in D.mel if m['name'] == names[k])
            kind = 'same architecture, other seed' if (ma['arch'], ma['data']) == (mb['arch'], mb['data']) else 'different classifier'
            rows.append({'Classifier A': mlabel(ma), 'Classifier B': mlabel(mb), 'Pair': kind, 'n effects': int(ok.sum()),
                         'Spearman ρ of effects': round(float(stats.spearmanr(a[ok], b[ok])[0]), 3) if ok.sum() > 3 else np.nan,
                         'Pearson r': round(float(np.corrcoef(a[ok], b[ok])[0, 1]), 3) if ok.sum() > 3 else np.nan})
    df = pd.DataFrame(rows); R['reliability'] = df.to_dict(orient='list')
    out.table(df, 'TP8_reliability', 'Agreement of the counterfactual effect vectors (7 concepts x bases) between classifiers.')
    if len(df):
        REP.append("\n## 7. Reliability of the audit\n\n" + df.groupby('Pair')[['Spearman ρ of effects', 'Pearson r']].mean().round(3).to_string())


@section('Example trajectory')
def s_example(cfg, out, R, D):
    E = getattr(D, 'effects', []); bk, sname, blab = D.bases[0]
    mel_seed = {}
    for m in D.mel:
        if m['data'] == 'isic2020' and m['arch'] not in mel_seed: mel_seed[m['arch']] = m
    if not mel_seed: return
    mean_eff = {c: np.mean([r['dlogit'] for r in E if r['label'] == 'mel' and r['base'] == bk and r['concept'] == c]) for c in CONCEPT_NAMES}
    c = max(mean_eff, key=lambda k: mean_eff[k]); j = CONCEPT_NAMES.index(c)
    meta = D.meta(sname); P_ = {a: 1 / (1 + np.exp(-D.lg[m['name']][sname])) for a, m in mel_seed.items()}
    T = {a: trajectories(meta, P_[a], j) for a in P_}
    nid = T[list(T)[0]][2]; dp = np.mean([T[a][0][:, -1] - T[a][0][:, 0] for a in T], 0); t = int(np.argmax(dp))
    sel = np.where((meta['concept_idx'].values == j) & (meta['noise_id'].values == nid[t]))[0]
    sel = sel[np.argsort(meta['level'].values[sel])]; X = D.imgs(sname)
    fig, axes = plt.subplots(1, len(sel), figsize=(1.15 * len(sel), 1.9))
    for k, i in enumerate(sel):
        ax = axes[k]; ax.imshow(X[i].permute(1, 2, 0).numpy()); ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        ax.set_title(f"{c.replace('_', ' ')} = {meta['level'].values[i]:.1f}", fontsize=5.5)
        ax.set_xlabel('\n'.join(f"{ {'resnet50': 'ResNet', 'effb0': 'EffNet', 'vitb16': 'ViT'}[a]} {P_[a][i]:.2f}" for a in P_), fontsize=5.5)
    fig.suptitle(f'{blab}: p(melanoma) of each classifier as {c.replace("_", " ")} is added', fontsize=7)
    fig.tight_layout(); out.fig(fig, 'figP_example_trajectory')


def main():
    cfg = load_cfg(sys.argv[1]); out = Out(cfg['outputs']); t0 = time.time(); R = {}
    D = Data(cfg)
    if not D.models:
        raise RuntimeError('no classifier results found')
    s_numeric(cfg, out, R, D); s_classifiers(cfg, out, R, D); s_effects(cfg, out, R, D); s_flips(cfg, out, R, D); s_decomp(cfg, out, R, D)
    s_observational(cfg, out, R, D); s_profiles(cfg, out, R, D); s_colour(cfg, out, R, D); s_reliability(cfg, out, R, D)
    s_example(cfg, out, R, D); s_controls(cfg, out, R, D)
    head = ["# ClinDiffXAI Phase 2 report: counterfactual concept audit of melanoma classifiers\n"]
    if cfg.get('dry_run'): head.append("**DRY RUN: a few images and 1 epoch. These numbers are not results.**\n")
    if WARN: REP.append("\n## Warnings\n\n" + '\n'.join(f"- {w}" for w in WARN))
    REP.append(f"\n_Analysis time {(time.time() - t0) / 60:.1f} min._\n")
    with open(os.path.join(out.root, 'phase2_report.md'), 'w') as f:
        f.write('\n'.join(head + REP))
    save_json(R, os.path.join(out.root, 'results', 'phase2_results.json'))
    print('\n'.join(head + REP))


if __name__ == '__main__':
    main()
''',
}
for fn, src in FILES.items():
    with open(os.path.join(CODE, fn), 'w') as f:
        f.write(src)
if cfg1d['pretrained']:
    say('downloading ImageNet weights once (ResNet-50, EfficientNet-B0, ViT-B/16)')
    import torchvision.models as tvm
    tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V2); tvm.efficientnet_b0(weights=tvm.EfficientNet_B0_Weights.IMAGENET1K_V1)
    tvm.vit_b_16(weights=tvm.ViT_B_16_Weights.IMAGENET1K_V1)

# ---------------------------------------------------------------- 3. config
models = []
for arch in ['resnet50', 'effb0', 'vitb16']:
    for seed in ([1] if DRY_RUN else [1, 2]):
        models.append({'name': f'mel_{arch}_s{seed}', 'arch': arch, 'label': 'mel', 'seed': seed, 'data': 'isic2020'})
models += [{'name': 'ctrl_pig_resnet50', 'arch': 'resnet50', 'label': 'pig', 'seed': 1, 'data': 'isic2020'},
           {'name': 'ctrl_bwv_resnet50', 'arch': 'resnet50', 'label': 'bwv', 'seed': 1, 'data': 'isic2020'},
           {'name': 'ctrl_perm_resnet50', 'arch': 'resnet50', 'label': 'perm', 'seed': 1, 'data': 'isic2020'}]
if d7:
    models.append({'name': 'mel_resnet50_derm7pt', 'arch': 'resnet50', 'label': 'mel', 'seed': 1, 'data': 'derm7pt'})
COST = {'resnet50': 6, 'effb0': 5, 'vitb16': 11}
load = [0] * len(SHARDS)
for m in sorted(models, key=lambda m: -(COST[m['arch']] if m['data'] == 'isic2020' else 3)):
    g = load.index(min(load)); m['gpu'] = g; load[g] += COST[m['arch']] if m['data'] == 'isic2020' else 3
ncpu = os.cpu_count() or 4
cfg = {
    'work': WORK, 'work1d': W1D, 'outputs': OUTD, 'paths': paths, 'image_size': cfg1d['image_size'],
    'eval_size': cfg1d['eval_size'], 'pretrained': cfg1d['pretrained'], 'dry_run': DRY_RUN or bool(cfg1d.get('dry_run')),
    'hardware': HW, 'cpu_workers': max(1, ncpu), 'cpu_workers_io': max(2, ncpu),
    'ddim_steps': cfg1d['sizes']['ddim_steps'], 'cfg_scale': cfg1d['sizes']['cfg_scale'], 'n_pool': cfg1d['sizes']['n_pool'],
    'gpu': {'gen_batch': 32},
    'rsweep': {'enabled': REALISTIC_SWEEPS, 'n_bases': 6 if DRY_RUN else 64, 'n_levels': 3 if DRY_RUN else 6},
    'cls': {'ben_ratio': 3 if DRY_RUN else 10, 'n_mel_max': 40 if DRY_RUN else None, 'ctrl_q': 0.25},
    'train': {'epochs': 1 if DRY_RUN else 15, 'patience': 1 if DRY_RUN else 5, 'batch': 16 if DRY_RUN else 32,
              'lr_cnn': 1e-4, 'lr_vit': 3e-5},
    'models': models,
    'analysis': {'n_boot': 200 if DRY_RUN else 2000, 'n_boot_auc': 100 if DRY_RUN else 1000, 'n_boot_reg': 20 if DRY_RUN else 200,
                 'logit_clip': 15.0, 'min_measured_sd': 0.25},
    'scores_dir': 'scores2_fp32',
    'audit_rule': {'alpha': 0.05, 'pos_fraction': 0.75, 'neg_fraction': 0.25, 'text':
                   'VALID if each positive control (pigmentation, blue-whitish veil) has its target concept as the largest '
                   'effect, positive and significant (two-sided Wilcoxon signed-rank test on per-trajectory slopes, BH-corrected '
                   'p < 0.05, median slope > 0), in at least 3 of 4 bases (the same share of the available bases), and the '
                   'largest |effect| of the permuted-label control is below 0.25 x the median largest |effect| of the melanoma '
                   'classifiers. NOT VALID otherwise.'},
}
CFG = os.path.join(WORK, 'cfg.json')
with open(CFG, 'w') as f:
    json.dump(cfg, f, indent=2)
print('  training plan: ' + '; '.join(f"GPU{m['gpu']}: {m['name']}" for m in sorted(models, key=lambda m: m['gpu'])))

# ---------------------------------------------------------------- 4. stages
STATUS = {}


def last_line(tag):
    p = os.path.join(WORK, 'logs', f'{tag}.log')
    if not os.path.exists(p): return ''
    with open(p, 'rb') as f:
        f.seek(0, 2); n = f.tell(); f.seek(max(0, n - 6000))
        lines = [l for l in f.read().decode('utf8', 'ignore').splitlines() if l.strip() and 'WARNING' not in l and 'Warning' not in l]
    return lines[-1][-120:] if lines else ''


def run_group(title, jobs, script='worker2.py'):
    """jobs: list of (job, shards or None). All processes start together; the group ends when all have ended."""
    procs, t0 = [], time.time()
    for job, shards in jobs:
        tags = [job] if shards is None else [f'{job}_{r}' for _, r in shards]
        if script == 'worker2.py' and all(os.path.exists(os.path.join(WORK, 'done', f'{t}.ok')) for t in tags):
            STATUS[job] = 'done earlier'; say(f'{job}: done earlier, skipped'); continue
        for (vis, rank), tag in zip(shards or [(None, 0)], tags):
            env = {**os.environ, 'PYTHONUNBUFFERED': '1', 'P2_RUN_ID': RUN_ID}
            if shards is not None: env['CUDA_VISIBLE_DEVICES'] = vis
            args = [sys.executable, '-u', os.path.join(CODE, script)] + \
                   (([job, CFG] + ([str(rank), str(len(shards))] if shards else [])) if script == 'worker2.py' else [CFG])
            lf = open(os.path.join(WORK, 'logs', f'{tag}.log'), 'a')
            procs.append((job, tag, subprocess.Popen(args, stdout=lf, stderr=subprocess.STDOUT, env=env, cwd=CODE, start_new_session=True), lf))
    if not procs: return
    say(f'===== {title} =====')
    try:
        tick = 0
        while any(p.poll() is None for _, _, p, _ in procs):
            time.sleep(5); tick += 5
            if tick % 60 == 0:
                print(f"    {title}: {(time.time() - t0) / 60:.1f} min", flush=True)
                for _, tag, p, _ in procs:
                    if p.poll() is None: print(f"      {tag}: {last_line(tag)}", flush=True)
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
    bad = [(tag, p.returncode) for _, tag, p, _ in procs if p.returncode != 0]
    for job in {j for j, _, _, _ in procs}:
        STATUS[job] = 'ok' if not [b for b in bad if b[0].startswith(job)] else 'FAILED'
    say(f"{title}: {'ok' if not bad else 'FAILED ' + str(bad)} after {(time.time() - t0) / 60:.1f} min")
    if bad:
        for tag, _ in bad:
            print(f'---- last lines of {tag}.log ----'); os.system(f"tail -n 40 {os.path.join(WORK, 'logs', tag + '.log')}")
        raise RuntimeError(f'{title} failed; see the logs in {os.path.join(WORK, "logs")}')


T0 = time.time()
run_group('plan of the realistic sweeps', [('plan', None)])
run_group('realistic sweeps (GPUs) and classifier data (CPU)', [('gen', SHARDS), ('prep_cls', None)])
run_group('assemble', [('assemble', None)])
run_group('train the classifiers', [('train', SHARDS)])
run_group('score every audit set (float32)', [('score32', SHARDS)])
run_group('analysis', [('analysis2', None)], script='analysis2.py')

# ---------------------------------------------------------------- 5. package
for fn in FILES: shutil.copy(os.path.join(CODE, fn), os.path.join(OUTD, 'code_' + fn))
shutil.copy(CFG, os.path.join(OUTD, 'cfg.json'))
os.makedirs(os.path.join(OUTD, 'logs'), exist_ok=True); os.makedirs(os.path.join(OUTD, 'results'), exist_ok=True)
for lg in glob.glob(os.path.join(WORK, 'logs', '*.log')): shutil.copy(lg, os.path.join(OUTD, 'logs'))
for fn in glob.glob(os.path.join(WORK, 'results', '*.json')): shutil.copy(fn, os.path.join(OUTD, 'results'))
for fn in glob.glob(os.path.join(WORK, 'models', 'pred_*.npz')): shutil.copy(fn, os.path.join(OUTD, 'results'))
for fn in glob.glob(os.path.join(WORK, 'scores2_fp32', '*.npz')): shutil.copy(fn, os.path.join(OUTD, 'results'))
shutil.rmtree(os.path.join(WORK, 'chunks_gen'), ignore_errors=True)
zip_base = os.path.join(WORK_ROOT, 'phase2_dryrun_outputs' if DRY_RUN else 'phase2_outputs')
shutil.make_archive(zip_base, 'zip', OUTD)
print('\n' + '=' * 78)
print(f"PHASE 2 {'DRY RUN ' if DRY_RUN else ''}FINISHED in {(time.time() - T0) / 60:.1f} min on {HW}")
for j, s in STATUS.items(): print(f"  {j:<10} {s}")
rep = os.path.join(OUTD, 'phase2_report.md')
print('=' * 78); print(open(rep).read() if os.path.exists(rep) else 'No report: see the analysis log.')
print(f"\nDownload: {zip_base}.zip  (tables, figures, report, logs, classifier outputs)")
