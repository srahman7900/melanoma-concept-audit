# =============================================================================
# ClinDiffXAI  PHASE 1c: colour diagnosis and sampling-time fixes
#                                              (one Kaggle cell, GPU T4 x2)
# =============================================================================
# WHERE TO RUN
#   In the SAME notebook and session as Phase 1b, in a new cell below the 1b cell.
#   The cell reads /kaggle/working/phase1b (images, scores, CBM-B, ISIC 2018
#   detector, cfg.json). It writes only to /kaggle/working/phase1c and does not
#   change anything in phase1b. No model is trained again.
#   If you stopped a cell in this session before, restart the kernel first
#   (Run -> Restart). The files in /kaggle/working stay; old worker processes stop.
#
# WHAT THIS CELL DOES (option B: fixes at sampling time only)
#   Part A. Does the starting noise set the colour? For the 6 Phase 1 sets
#     (1,000 images each), the per-channel mean of each starting noise is compared
#     with the mean colour of its image (correlation, cross-validated R2).
#   Part B. The same 256 starting noises per class (Phase 1 noise ids 0..255) are
#     sampled again in two new ways, for the main model with the prototypes, the
#     baseline, and the main model with real concept vectors:
#       V1  DDIM starts at the last timestep t = 999 ("trailing" spacing,
#           Lin et al., WACV 2024) instead of t = 980.
#       V2  colour-calibrated start: the channel means of the starting noise are
#           set to m*, chosen with a linear map (fitted on noise ids >= 256 only)
#           so that the image gets the mean colour of a randomly drawn real image
#           of the same class.
#     V0 = the existing Phase 1 images of the same noises (no new sampling).
#   Part C. Concept control under V1 and V2: dose-response for pigmentation,
#     pigment network and blue-whitish veil (16 trajectories x 11 levels each).
#   Part D. Is the Phase 1 concept control only a colour effect? The Phase 1
#     sweep and the real anchors are re-scored after Shades-of-Gray colour
#     constancy (p = 6).
#
# DECISION RULE (fixed before the run)
#   A variant FIXES COLOUR if its colour-outlier rate (Phase 1b rule) is <= 20%
#   for the main model and the baseline in both classes. It is ACCEPTABLE if, in
#   addition, its Inception KID is not more than 10% worse than V0 for each of
#   these 4 sets, and the CBM-B dose-response for pigmentation AND pigment
#   network stays positive and significant (two-sided Wilcoxon signed-rank test
#   on per-trajectory slopes, BH-corrected p < 0.05, median slope > 0). If more
#   than one variant is acceptable, the one with the lower mean outlier rate is
#   chosen. If no variant is acceptable: option C (fine-tune with a
#   zero-terminal-SNR schedule).
#   Part A criterion: "the noise mean sets the colour" if the mean CV R2 >= 0.50.
#
# TIME: my estimate is 45 to 70 minutes on T4 x2 (about 4,100 new images,
#   split over the two GPUs, then scoring). This is an estimate, not a measurement.
# OUTPUT: /kaggle/working/phase1c_outputs.zip (tables, figures, report, logs).
# =============================================================================

DRY_RUN     = False   # True: a few images only, to test the cell (results are not valid)
FORCE_RERUN = False   # True: delete earlier Phase 1c results and start again

import os, sys, json, time, glob, shutil, subprocess, threading, uuid
import numpy as np

# testing overrides for an offline run on another machine; leave unset on Kaggle
WORK_ROOT = os.environ.get('P1_WORK_ROOT', '/kaggle/working')
INPUT_ROOT = os.environ.get('P1_INPUT_ROOT', '/kaggle/input')


def say(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------- 0. find Phase 1b
NEED_SETS = ['eval_main_mel', 'eval_main_ben', 'eval_base_mel', 'eval_base_ben', 'eval_realc_mel', 'eval_realc_ben',
             'real_mel', 'real_ben', 'sweep_mel']


def is_phase1b(d):
    return (os.path.exists(os.path.join(d, 'cfg.json')) and os.path.exists(os.path.join(d, 'evaluators', 'cbmb.pth'))
            and all(os.path.exists(os.path.join(d, 'sets', f'{n}.pt')) for n in NEED_SETS))


W1B = os.environ.get('P1C_WORK1B')
if not W1B:
    for d in [os.path.join(WORK_ROOT, 'phase1b')] + ([os.path.join(WORK_ROOT, 'phase1b_dryrun')] if DRY_RUN else []):
        if is_phase1b(d):
            W1B = d; break
if not W1B and os.path.isdir(INPUT_ROOT):   # a committed Phase 1b notebook attached as input
    say(f'phase1b not found in {WORK_ROOT}; searching {INPUT_ROOT} ...')
    for dp, dn, fn in os.walk(INPUT_ROOT):
        if os.path.basename(dp) == 'phase1b' and is_phase1b(dp):
            W1B = dp; break
        if dp.count(os.sep) - INPUT_ROOT.count(os.sep) >= 4:
            dn[:] = []
if not W1B or not is_phase1b(W1B):
    raise FileNotFoundError('Phase 1b output not found. Run this cell in the same session after the Phase 1b cell, '
                            'or attach the committed Phase 1b notebook output.')
cfg1b = json.load(open(os.path.join(W1B, 'cfg.json')))
S1B = cfg1b['sizes']
missing_scores = [f'{t}_{n}' for n in NEED_SETS for t in ['A', 'B'] if not os.path.exists(os.path.join(W1B, 'scores', f'{t}_{n}.npz'))]
if missing_scores:
    raise FileNotFoundError(f'Phase 1b scores missing in {W1B}/scores: {missing_scores}. Let the Phase 1b cell finish first.')
say(f'Phase 1b found at {W1B}' + ('  (Phase 1b was a DRY RUN)' if cfg1b.get('dry_run') else ''))
if cfg1b.get('dry_run') and not DRY_RUN and not os.environ.get('P1C_WORK1B'):
    raise RuntimeError('The Phase 1b folder found is a dry run. Run the full Phase 1b first, or set DRY_RUN = True.')

WORK = os.path.join(WORK_ROOT, 'phase1c_dryrun' if DRY_RUN else 'phase1c')
CODE = os.path.join(WORK, 'code'); OUTD = os.path.join(WORK, 'outputs')
WORKER = os.path.join(CODE, 'worker1c.py')
os.makedirs(WORK, exist_ok=True)
# run-id guard. When a cell is stopped, its background threads keep running, and a new run of the cell
# rebinds the global names they use. Each thread therefore keeps its own run id (an argument, not a global);
# it starts no job and records no result when the id in run_id.txt is not its own. Each worker process also
# checks the id when it starts. The new id is written FIRST, then old worker processes are stopped twice
# (the second time for a process that an old thread started in the moment before the new id was written).
RUN_ID = uuid.uuid4().hex
with open(os.path.join(WORK, 'run_id.txt'), 'w') as f:
    f.write(RUN_ID)


def current_run(rid):
    try:
        return open(os.path.join(WORK, 'run_id.txt')).read().strip() == rid
    except Exception:
        return False


for _ in range(2):
    if subprocess.run(['pkill', '-f', WORKER], check=False).returncode == 0:
        say('stopped worker processes of an earlier, interrupted run of this cell')
    time.sleep(4)
if FORCE_RERUN:
    for d in ['done', 'sets', 'scores', 'results', 'outputs', 'logs']:
        shutil.rmtree(os.path.join(WORK, d), ignore_errors=True)
for d in [CODE, OUTD, os.path.join(WORK, 'logs'), os.path.join(WORK, 'done'), os.path.join(WORK, 'results')]:
    os.makedirs(d, exist_ok=True)

# ---------------------------------------------------------------- 1. checkpoints
import torch
ck = {}
for nm in ['eval_main_mel', 'eval_base_mel', 'sweep_mel']:
    info = torch.load(os.path.join(W1B, 'sets', f'{nm}.pt'), map_location='cpu', weights_only=False).get('info', {})
    ck[nm] = (info.get('ckpt') or {}).get('path')
print('\n' + '=' * 78 + '\nCHECKPOINTS (taken from the Phase 1 image sets, so V0, V1 and V2 use the same weights)\n' + '=' * 78)
for nm, p in ck.items():
    print(f"  {'OK ' if p and os.path.exists(p) else '-- '} {nm:<14} {p}")
bad = [nm for nm, p in ck.items() if not p or not os.path.exists(p)]
if bad:
    alt = {nm: cfg1b['paths'].get(('base' if 'base' in nm else 'main') + '_e150') for nm in bad}
    if not all(p and os.path.exists(p) for p in alt.values()):
        raise FileNotFoundError(f'Checkpoints for {bad} are not attached. Attach the epoch-150 checkpoints used in Phase 1.')
    print('  NOTE: saved checkpoint path missing; the epoch-150 path from the Phase 1b cfg is used:', alt)
if not os.path.exists(cfg1b['paths']['cbma']):
    raise FileNotFoundError(f"CBM-A checkpoint not found: {cfg1b['paths']['cbma']}")

# ---------------------------------------------------------------- 2. packages + code files
if not os.environ.get('P1_INPUT_ROOT'):
    try:
        import diffusers
        if diffusers.__version__ != '0.33.0': raise ImportError
    except ImportError:
        say('installing diffusers==0.33.0 and accelerate==1.6.0 (same pins as NB4 and Phase 1b)')
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'diffusers==0.33.0', 'accelerate==1.6.0'], check=False)

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
    'worker1c.py': r'''"""ClinDiffXAI Phase 1c worker (colour diagnosis and sampling-time fixes).
Usage: python worker1c.py <job> <cfg.json>"""
import os, sys, json, time, glob, traceback
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *  # noqa


def P(cfg, *parts):
    p = os.path.join(cfg['work'], *parts)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    return p


def S1B(cfg, name):
    return os.path.join(cfg['work1b'], 'sets', f'{name}.pt')


def mean_rgb_pm1(u8):
    """per-image mean of each channel, in the model's [-1, 1] scale"""
    return u8.float().div(127.5).sub(1.0).mean((2, 3)).numpy()


GEN_SET = {'main': 'eval_main', 'base': 'eval_base', 'realc': 'eval_realc'}


def v0_ckpt(cfg, set_name):
    """checkpoint that made the Phase 1/1b set, so that V0, V1 and V2 use identical weights"""
    info = torch.load(S1B(cfg, set_name), map_location='cpu', weights_only=False).get('info', {})
    p = (info.get('ckpt') or {}).get('path')
    if p and os.path.exists(p):
        return p
    gen = 'base' if 'base' in set_name else 'main'
    p = cfg['paths'].get(f'{gen}_e150') or cfg['paths'][f'{gen}_best']
    log(f"  NOTE: checkpoint of {set_name} not found at its saved path; using {p}")
    return p


# =============================================================================
# prep: Part A (does the starting noise set the colour?), calibration fit,
# colour targets, and colour-constancy copies of the Phase 1 sweep (CPU)
# =============================================================================
def job_prep(cfg):
    from sklearn.linear_model import LinearRegression, RidgeCV
    from sklearn.model_selection import cross_val_predict, KFold
    IS = cfg['image_size']; N = cfg['n_var']; NF = cfg['n_fit_from']
    seeds = {'mel': cfg['seeds']['eval_mel'], 'ben': cfg['seeds']['eval_ben']}
    noise_m = {}; low = {}
    for cls in ['mel', 'ben']:
        z = cpu_noise(1000, IS, seeds[cls])
        noise_m[cls] = z.mean((2, 3)).numpy()
        low[cls] = torch.nn.functional.adaptive_avg_pool2d(z, 4).reshape(1000, -1).numpy()
    real = {c: load_set(S1B(cfg, f'real_{c}'))['imgs'] for c in ['mel', 'ben']}
    # colour targets: real images that are NOT used as the reference in the analysis
    # (melanoma: even indices; benign: second half. The analysis uses the other images as the reference.)
    pool = {'mel': mean_rgb_pm1(real['mel'][0::2]), 'ben': mean_rgb_pm1(real['ben'][len(real['ben']) // 2:])}
    rows, calib, raw = [], {}, {}
    kf = KFold(5, shuffle=True, random_state=0)
    for gen in ['main', 'base', 'realc']:
        for cls in ['mel', 'ben']:
            X = load_set(S1B(cfg, f'{GEN_SET[gen]}_{cls}'))['imgs']
            y = mean_rgb_pm1(X); m = noise_m[cls][:len(y)]; L = low[cls][:len(y)]
            cv3 = cross_val_predict(LinearRegression(), m, y, cv=kf)
            cv48 = cross_val_predict(RidgeCV(alphas=np.logspace(-3, 3, 13)), L, y, cv=kf)
            r2 = lambda yp: [float(1 - ((y[:, k] - yp[:, k]) ** 2).sum() / ((y[:, k] - y[:, k].mean()) ** 2).sum()) for k in range(3)]
            corr = [float(np.corrcoef(m[:, k], y[:, k])[0, 1]) for k in range(3)]
            # calibration map y = W m + b, fitted on noise ids NF..999 only (ids 0..N-1 are the test images)
            reg = LinearRegression().fit(m[NF:], y[NF:]); W, b = reg.coef_, reg.intercept_
            fit_r2 = float(reg.score(m[NF:], y[NF:]))
            rng = np.random.default_rng(cfg['seeds'][f'target_{cls}'])
            tgt = pool[cls][rng.integers(0, len(pool[cls]), N)]
            mstar = np.linalg.lstsq(W, (tgt - b).T, rcond=None)[0].T          # (N,3)
            sd = 1.0 / IS                                                     # SD of a channel mean of N(0,1) noise
            calib[f'{gen}|{cls}'] = {'W': W.tolist(), 'b': b.tolist(), 'fit_r2': fit_r2, 'targets': tgt.tolist(),
                                     'mstar': mstar.tolist()}
            rows.append({'Generator': gen, 'Class': cls, 'n': len(y),
                         'corr R': corr[0], 'corr G': corr[1], 'corr B': corr[2],
                         'CV R2 3-mean R': r2(cv3)[0], 'CV R2 3-mean G': r2(cv3)[1], 'CV R2 3-mean B': r2(cv3)[2],
                         'CV R2 4x4 R': r2(cv48)[0], 'CV R2 4x4 G': r2(cv48)[1], 'CV R2 4x4 B': r2(cv48)[2],
                         'Calibration fit R2': fit_r2,
                         'median |m*| (natural SD)': float(np.median(np.abs(mstar)) / sd),
                         'm* beyond 3 natural SD (%)': float(np.mean(np.abs(mstar) > 3 * sd) * 100)})
            raw[f'{gen}|{cls}|y'] = y
            log(f"  {gen:<5} {cls}: corr(noise mean, image mean) R/G/B = {corr[0]:.2f}/{corr[1]:.2f}/{corr[2]:.2f}; "
                f"CV R2 = {np.mean(r2(cv3)):.2f}; calibration fit R2 = {fit_r2:.2f}")
    # sweep targets (16 trajectories from the melanoma prototype) use the main-model melanoma map
    rng = np.random.default_rng(cfg['seeds']['target_sweep'])
    c = calib['main|mel']; W, b = np.array(c['W']), np.array(c['b'])
    tgt = pool['mel'][rng.integers(0, len(pool['mel']), cfg['sweep_n'])]
    calib['sweep'] = {'targets': tgt.tolist(), 'mstar': np.linalg.lstsq(W, (tgt - b).T, rcond=None)[0].T.tolist()}
    save_json(calib, P(cfg, 'results', 'calibration.json'))
    import pandas as pd
    pd.DataFrame(rows).to_csv(P(cfg, 'results', 'partA_noise_vs_colour.csv'), index=False)
    np.savez(P(cfg, 'results', 'partA_raw.npz'), noise_mel=noise_m['mel'], noise_ben=noise_m['ben'],
             **{k.replace('|', '__'): v for k, v in raw.items()})
    # colour-constancy copies (Shades of Gray, p = 6) of the Phase 1 sweep and the real anchors
    for name in ['sweep_mel', 'real_mel', 'real_ben']:
        s = load_set(S1B(cfg, name))
        X = s['imgs'].numpy()
        cc = torch.from_numpy(np.concatenate([shades_of_gray(X[i:i + 256]) for i in range(0, len(X), 256)], 0))
        save_set(P(cfg, 'sets', f'cc_{name}.pt'), cc, s['meta'], {**s['info'], 'colour_constancy': 'shades_of_gray_p6'})
        log(f"  colour-constancy copy of {name}: {len(cc)} images")


# =============================================================================
# generation with sampling-time variants (GPU)
#   V1: DDIM started at the last timestep ("trailing" spacing, Lin et al. 2024)
#   V2: colour-calibrated start: the channel means of the starting noise are
#       replaced by m*, chosen so that the fitted noise->colour map gives the
#       colour of a randomly drawn real image of the same class
# =============================================================================
def _variant_noise(noise, var, mstar):
    if var == 'V2':
        return noise - noise.mean((2, 3), keepdim=True) + torch.tensor(np.asarray(mstar), dtype=noise.dtype)[:, :, None, None]
    return noise


def job_gen_var(cfg, gen, var):
    dev = get_device(); torch.backends.cudnn.benchmark = True
    IS = cfg['image_size']; N = cfg['n_var']; S = cfg['sizes']
    calib = load_cfg(os.path.join(cfg['work'], 'results', 'calibration.json')) if var == 'V2' else None
    model, info = load_diffusion(v0_ckpt(cfg, f'{GEN_SET[gen]}_mel'), 2 if gen == 'base' else 7, dev, IS)
    for cls, seed in [('mel', cfg['seeds']['eval_mel']), ('ben', cfg['seeds']['eval_ben'])]:
        out = P(cfg, 'sets', f'var_{gen}_{var}_{cls}.pt')
        if os.path.exists(out): continue
        noise = cpu_noise(1000, IS, seed)[:N]                      # identical to Phase 1 noise ids 0..N-1
        if gen == 'main':
            cond = torch.FloatTensor(MEL_VEC_LIST if cls == 'mel' else BEN_VEC_LIST).unsqueeze(0).repeat(N, 1)
        elif gen == 'base':
            cond = torch.FloatTensor([0., 1.] if cls == 'mel' else [1., 0.]).unsqueeze(0).repeat(N, 1)
        else:
            meta = load_set(S1B(cfg, f'eval_realc_{cls}'))['meta']
            cond = torch.FloatTensor(np.stack([np.asarray(meta[f'concept_{c}'][:N], np.float32) for c in CONCEPT_NAMES], 1)).clamp(0, 1)
        c = calib[f'{gen}|{cls}'] if var == 'V2' else None
        z = _variant_noise(noise, var, c['mstar'] if c else None)
        t0 = time.time()
        imgs = generate_crn(model, cond, z, dev, cfg_scale=S['cfg_scale'], steps=S['ddim_steps'], bs=S['gen_bs'],
                            spacing='trailing' if var == 'V1' else 'leading')
        log(f"  var_{gen}_{var}_{cls}: {N} images in {time.time() - t0:.0f}s")
        meta = {'noise_id': list(range(N))}
        if gen == 'realc':
            for k in CONCEPT_NAMES:
                meta[f'concept_{k}'] = cond[:, CONCEPT_NAMES.index(k)].tolist()
        if var == 'V2':
            meta['target_r'], meta['target_g'], meta['target_b'] = [[float(v) for v in np.asarray(c['targets'])[:, k]] for k in range(3)]
        save_set(out, imgs, meta, {'kind': 'variant', 'gen': gen, 'variant': var, 'cls': cls, 'ckpt': info})


def job_gen_sweep_var(cfg, var):
    dev = get_device(); torch.backends.cudnn.benchmark = True
    IS = cfg['image_size']; S = cfg['sizes']; n = cfg['sweep_n']
    out = P(cfg, 'sets', f'sweepvar_{var}.pt')
    if os.path.exists(out): return
    calib = load_cfg(os.path.join(cfg['work'], 'results', 'calibration.json')) if var == 'V2' else None
    model, info = load_diffusion(v0_ckpt(cfg, 'sweep_mel'), 7, dev, IS)
    noise = cpu_noise(cfg['sizes']['sweep_n_phase1'], IS, cfg['seeds']['sweep'])[:n]   # Phase 1 sweep noises 0..n-1
    z = _variant_noise(noise, var, calib['sweep']['mstar'] if calib else None)
    levels = [round(float(x), 4) for x in np.linspace(0, 1, cfg['sizes']['sweep_levels'])]
    imgs, meta = [], {'concept_idx': [], 'level': [], 'noise_id': []}
    for j in cfg['sweep_concepts']:
        for l in levels:
            v = torch.FloatTensor(MEL_VEC_LIST); v[j] = l
            imgs.append(generate_crn(model, v.unsqueeze(0).repeat(n, 1), z, dev, cfg_scale=S['cfg_scale'], steps=S['ddim_steps'],
                                     bs=S['gen_bs'], spacing='trailing' if var == 'V1' else 'leading'))
            meta['concept_idx'] += [j] * n; meta['level'] += [l] * n; meta['noise_id'] += list(range(n))
        log(f"  sweepvar_{var}: concept {CONCEPT_NAMES[j]} done")
    save_set(out, torch.cat(imgs, 0), meta, {'kind': 'sweep_variant', 'variant': var, 'base': 'mel', 'ckpt': info})


# =============================================================================
# scoring (same evaluators and pipeline as Phase 1 / 1b)
# =============================================================================
def _atomic_npz(path, **arrays):
    with open(path + '.tmp', 'wb') as fh:
        np.savez(fh, **arrays)
    os.replace(path + '.tmp', path)


def job_score_A(cfg):
    dev = get_device()
    cbm, _ = load_cbm_a(cfg['paths']['cbma'], dev); incep = build_inception(cfg, dev)
    for sp in sorted(glob.glob(os.path.join(cfg['work'], 'sets', '*.pt'))):
        name = os.path.basename(sp)[:-3]; out = P(cfg, 'scores', f'A_{name}.npz')
        if os.path.exists(out): continue
        X = load_set(sp)['imgs']; cp, fi = [], []
        with torch.no_grad():
            for s in range(0, len(X), 64):
                a, _, _ = cbm.forward_all(u8_to_eval(X[s:s + 64], dev, cfg['eval_size'])); cp.append(a.cpu().numpy())
                if name.startswith('var_'):
                    f = incep(u8_to_inception(X[s:s + 64], dev, cfg['incep_size']))
                    fi.append((f[0] if isinstance(f, tuple) else f).float().cpu().numpy())
        arr = {'cbma': np.concatenate(cp, 0), 'dark': darkness_metrics(X.numpy()), 'lab': lab_means(X.numpy())}
        if fi: arr['incep'] = np.concatenate(fi, 0)
        _atomic_npz(out, **arr); log(f"  score_A {name}: n={len(X)}")


def job_score_B(cfg):
    dev = get_device(); nets = {}
    n = ConceptNetB(pretrained=False)
    n.load_state_dict(torch.load(os.path.join(cfg['work1b'], 'evaluators', 'cbmb.pth'), map_location='cpu', weights_only=False)['state_dict'])
    nets['cbmb'] = n.to(dev).eval()
    pi = os.path.join(cfg['work1b'], 'evaluators', 'isic18_detector.pth')
    if os.path.exists(pi):
        n = AttrNet(pretrained=False); n.load_state_dict(torch.load(pi, map_location='cpu', weights_only=False)['state_dict'])
        nets['isic18'] = n.to(dev).eval()
    for sp in sorted(glob.glob(os.path.join(cfg['work'], 'sets', '*.pt'))):
        name = os.path.basename(sp)[:-3]; out = P(cfg, 'scores', f'B_{name}.npz')
        if os.path.exists(out): continue
        X = load_set(sp)['imgs']; res = {k: [] for k in nets}
        with torch.no_grad():
            for s in range(0, len(X), 64):
                x = u8_to_eval(X[s:s + 64], dev, cfg['eval_size'])
                for k, net in nets.items():
                    res[k].append(torch.sigmoid(net(x)).cpu().numpy())
        _atomic_npz(out, **{k: np.concatenate(v, 0) for k, v in res.items()}); log(f"  score_B {name}: n={len(X)}")


JOBS = {'prep': job_prep, 'score_A': job_score_A, 'score_B': job_score_B,
        'sweep_V1': lambda c: job_gen_sweep_var(c, 'V1'), 'sweep_V2': lambda c: job_gen_sweep_var(c, 'V2')}
for _g in ['main', 'base', 'realc']:
    for _v in ['V1', 'V2']:
        JOBS[f'var_{_g}_{_v}'] = (lambda g, v: (lambda c: job_gen_var(c, g, v)))(_g, _v)

if __name__ == '__main__':
    job, cfg_path = sys.argv[1], sys.argv[2]
    cfg = load_cfg(cfg_path)
    rid = os.environ.get('P1C_RUN_ID')
    if rid:   # started by a thread of an older, interrupted run of the cell: do nothing
        try:
            cur = open(os.path.join(cfg['work'], 'run_id.txt')).read().strip()
        except Exception:
            cur = rid
        if cur != rid:
            print(f'job {job}: started by an older run of the cell; exit without work', flush=True); sys.exit(3)
    log(f"job {job} start | device={get_device()} | torch {torch.__version__} | "
        f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")
    t0 = time.time()
    try:
        JOBS[job](cfg)
    except Exception:
        traceback.print_exc(); sys.exit(1)
    with open(P(cfg, 'done', f'{job}.ok'), 'w') as f:
        f.write(f"{time.time() - t0:.0f}s\n")
    log(f"job {job} done in {(time.time() - t0) / 60:.1f} min")
''',
    'analysis1c.py': r'''"""ClinDiffXAI Phase 1c analysis: colour diagnosis and sampling-time fixes.
Usage: python analysis1c.py <cfg.json>"""
import os, sys, json, time, traceback
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (CONCEPT_NAMES, SHORT, MEL_VEC_LIST, BEN_VEC_LIST, ISIC18_ATTRS, ISIC18_TO_DERM, DARK_COLS, load_cfg,
                    load_set, save_json, log, lab_means)
import analysis as A   # shared statistics, styles and table/figure writers (Phase 1 code)
from analysis import plt, stats, Out, fci, fp, f3, INK, INK2, C_EV, M_EV

REP, WARN = [], []
GENS = [('main', 'ClinDiffXAI, prototype'), ('base', 'Baseline'), ('realc', 'ClinDiffXAI, real vectors')]
VARS = [('V0', 'V0 original sampling'), ('V1', 'V1 trailing start'), ('V2', 'V2 colour-calibrated noise')]
C_VAR = {'V0': '#52514e', 'V1': '#eb6834', 'V2': '#2a78d6'}
M_VAR = {'V0': 'o', 'V1': 's', 'V2': '^'}
C_GEN = {'main': '#2a78d6', 'base': '#eb6834', 'realc': '#1baf7a'}
TWO_LINE = {'main': 'ClinDiffXAI\nprototype', 'base': 'Baseline', 'realc': 'ClinDiffXAI\nreal vectors'}
DERM_TO_ISIC18 = {v: k for k, v in ISIC18_TO_DERM.items()}


def npz(path):
    if not os.path.exists(path): return {}
    z = np.load(path); return {k: z[k] for k in z.files}


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
    """V0 = Phase 1/1b images (noise ids 0..N-1); V1/V2 = Phase 1c images."""
    def __init__(self, cfg):
        self.cfg = cfg; self.N = cfg['n_var']; w1b, w1c = cfg['work1b'], cfg['work']
        self.s1b = lambda n: os.path.join(w1b, 'sets', f'{n}.pt')
        self.s1c = lambda n: os.path.join(w1c, 'sets', f'{n}.pt')
        self.A1b = lambda n: npz(os.path.join(w1b, 'scores', f'A_{n}.npz'))
        self.B1b = lambda n: npz(os.path.join(w1b, 'scores', f'B_{n}.npz'))
        self.A1c = lambda n: npz(os.path.join(w1c, 'scores', f'A_{n}.npz'))
        self.B1c = lambda n: npz(os.path.join(w1c, 'scores', f'B_{n}.npz'))
        self.real = {}
        for c in ['mel', 'ben']:
            s = load_set(self.s1b(f'real_{c}'))
            self.real[c] = {'lab': lab_means(s['imgs'].numpy()), 'A': self.A1b(f'real_{c}'), 'B': self.B1b(f'real_{c}'), 'n': len(s['imgs'])}
        # reference halves (not used as colour targets): melanoma odd indices, benign first half
        self.ref = {'mel': np.arange(self.real['mel']['n'])[1::2], 'ben': np.arange(self.real['ben']['n'] // 2)}
        self.cache = {}

    def get(self, gen, var, cls):
        key = (gen, var, cls)
        if key in self.cache: return self.cache[key]
        N = self.N
        if var == 'V0':
            name = {'main': 'eval_main', 'base': 'eval_base', 'realc': 'eval_realc'}[gen] + f'_{cls}'
            if not os.path.exists(self.s1b(name)): return None
            s = load_set(self.s1b(name)); a, b = self.A1b(name), self.B1b(name)
            d = {'imgs': s['imgs'][:N], 'meta': pd.DataFrame(s['meta']).iloc[:N].reset_index(drop=True),
                 'cbma': a['cbma'][:N], 'incep': a['incep'][:N], 'cbmb': b['cbmb'][:N],
                 'isic18': b['isic18'][:N] if 'isic18' in b else None, 'dark': a['dark'][:N]}
        else:
            name = f'var_{gen}_{var}_{cls}'
            if not os.path.exists(self.s1c(name)): return None
            s = load_set(self.s1c(name)); a, b = self.A1c(name), self.B1c(name)
            if not a or not b: return None
            d = {'imgs': s['imgs'], 'meta': pd.DataFrame(s['meta']), 'cbma': a['cbma'], 'incep': a['incep'], 'cbmb': b['cbmb'],
                 'isic18': b.get('isic18'), 'dark': a['dark']}
        d['lab'] = lab_means(d['imgs'].numpy())
        self.cache[key] = d
        return d


# -----------------------------------------------------------------------------
@section('Part A: does the starting noise set the colour?')
def s_partA(cfg, out, R):
    f = os.path.join(cfg['work'], 'results', 'partA_noise_vs_colour.csv')
    df = pd.read_csv(f); R['partA'] = df.to_dict(orient='list')
    out.table(df.round(3), 'TC1_noise_vs_colour', 'Part A. Phase 1 images (1,000 per set): relation between the per-channel '
              'spatial mean of the starting noise and the per-channel mean colour of the generated image (both in the model '
              'scale [-1, 1]). corr: Pearson correlation. CV R2 3-mean: 5-fold cross-validated R2 of a linear model from the 3 '
              'noise channel means. CV R2 4x4: ridge model from the 4x4 low-frequency noise (48 values). The V2 calibration map '
              'is fitted on noise ids >= %d only; m*: the noise means that V2 sets, in units of the natural SD of a noise '
              'channel mean (1/image size).' % cfg['n_fit_from'])
    c3 = ['CV R2 3-mean R', 'CV R2 3-mean G', 'CV R2 3-mean B']; c48 = ['CV R2 4x4 R', 'CV R2 4x4 G', 'CV R2 4x4 B']
    r3 = float(df[c3].values.mean()); r48 = float(df[c48].values.mean())
    R['partA_mean_cv_r2'] = r3; R['partA_mean_cv_r2_lowfreq'] = r48
    thr = cfg['decision']['partA_r2']
    REP.append(f"\n## Part A: starting noise versus image colour\n\n- Mean cross-validated R2, 3 noise channel means to image "
               f"mean colour: **{r3:.2f}**; with the 4x4 low-frequency noise: {r48:.2f}. Criterion fixed before the run for "
               f"'the noise mean sets the colour': R2 >= {thr:.2f}: **{'MET' if r3 >= thr else 'NOT MET'}**.")
    for _, r in df.iterrows():
        REP.append(f"- {r['Generator']}/{r['Class']}: corr R/G/B {r['corr R']:.2f}/{r['corr G']:.2f}/{r['corr B']:.2f}; "
                   f"CV R2 {np.mean([r[c] for c in c3]):.2f}; median |m*| {r['median |m*| (natural SD)']:.1f} natural SD; "
                   f"{r['m* beyond 3 natural SD (%)']:.0f}% of m* beyond 3 SD")
    z = np.load(os.path.join(cfg['work'], 'results', 'partA_raw.npz'))
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.4), sharex='col')
    for r_, cls in enumerate(['mel', 'ben']):
        m = z[f'noise_{cls}']
        for k, ch in enumerate(['red', 'green', 'blue']):
            ax = axes[r_, k]
            for gen, _ in GENS:
                key = f'{gen}__{cls}__y'
                if key in z.files:
                    y = z[key]; ax.scatter(m[:len(y), k] * cfg['image_size'], y[:, k], s=2, alpha=0.35, lw=0, color=C_GEN[gen],
                                           label=TWO_LINE[gen].replace('\n', ' '), rasterized=True)
            ax.set_title(f"{'Melanoma' if cls == 'mel' else 'Benign'} condition, {ch}", fontsize=7)
            if r_ == 1: ax.set_xlabel(f'Noise mean, {ch} (natural SD units)')
            if k == 0: ax.set_ylabel('Image mean (model scale)')
    axes[0, 0].legend(markerscale=4, fontsize=5.5, loc='upper left')
    fig.tight_layout(); out.fig(fig, 'figC_noise_mean_vs_image_colour')


@section('Part B: sampling-time variants')
def s_partB(cfg, out, R, D):
    B = cfg['analysis']['n_boot_auc']
    Rl = np.concatenate([D.real['mel']['lab'], D.real['ben']['lab']])
    mu = Rl.mean(0); icov = np.linalg.inv(np.cov(Rl, rowvar=False))
    md = lambda X: np.sqrt(np.einsum('ij,jk,ik->i', X - mu, icov, X - mu))
    thr = float(np.percentile(md(Rl), 99))
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    clf = {}
    for ev in ['cbmb', 'cbma']:
        key = 'B' if ev == 'cbmb' else 'A'
        Xr = np.concatenate([D.real['mel'][key][ev], D.real['ben'][key][ev]])
        yr = np.r_[np.ones(len(D.real['mel'][key][ev])), np.zeros(len(D.real['ben'][key][ev]))]
        clf[ev] = make_pipeline(StandardScaler(), LogisticRegression(class_weight='balanced', max_iter=2000)).fit(Xr, yr)
    rows, grp = [], []
    R['variants'] = {}
    for gen, glab in GENS:
        for var, vlab in VARS:
            got = {c: D.get(gen, var, c) for c in ['mel', 'ben']}
            if any(v is None for v in got.values()): continue
            for cls in ['mel', 'ben']:
                d = got[cls]; ref = D.ref[cls]
                ref_lab = D.real[cls]['lab'][ref]; ref_inc = D.real[cls]['A']['incep'][ref]
                o = float(np.mean(md(d['lab']) > thr) * 100)
                km, ks, mm = A.kid(d['incep'], ref_inc, cfg['analysis']['kid_subset'], cfg['analysis']['kid_n_sub'])
                f = d['incep'] / (np.linalg.norm(d['incep'], axis=1, keepdims=True) + 1e-9)
                div = 1 - (f @ f.T)[np.triu_indices(len(f), 1)].mean()
                row = {'Generator': glab, 'Class': cls, 'Variant': vlab, 'n': len(d['lab']), 'Colour outliers %': round(o, 1),
                       'L*': round(float(d['lab'][:, 0].mean()), 1), 'a*': round(float(d['lab'][:, 1].mean()), 1),
                       'b*': round(float(d['lab'][:, 2].mean()), 1),
                       'W1 a*': round(float(stats.wasserstein_distance(d['lab'][:, 1], ref_lab[:, 1])), 2),
                       'W1 b*': round(float(stats.wasserstein_distance(d['lab'][:, 2], ref_lab[:, 2])), 2),
                       'KID x1e3 (Inception)': round(km * 1e3, 2), 'Diversity': round(float(div), 3)}
                if gen == 'main':
                    c = np.array(MEL_VEC_LIST if cls == 'mel' else BEN_VEC_LIST)
                    row['Concept agreement (CBM-A)'] = round(float((1 - np.abs(d['cbma'] - c[None]).mean(1)).mean()), 3)
                if gen == 'realc':
                    Cg = d['meta'][[f'concept_{k}' for k in CONCEPT_NAMES]].values
                    row['Tracking rho (CBM-B)'] = round(float(np.nanmean([stats.spearmanr(Cg[:, k], d['cbmb'][:, k])[0] for k in range(7)])), 3)
                rows.append(row); R['variants'][f'{gen}|{var}|{cls}'] = {'outliers': o, 'kid': km}
            for ev in ['cbmb', 'cbma']:
                pm = clf[ev].predict_proba(got['mel'][ev])[:, 1]; pb = clf[ev].predict_proba(got['ben'][ev])[:, 1]
                a = A.auc_ci(pm, pb, B)
                grp.append({'Generator': glab, 'Variant': vlab, 'Evaluator': 'CBM-B' if ev == 'cbmb' else 'CBM-A',
                            'Melanoma vs benign AUC [95% CI]': fci(*a)})
                R['variants'][f'{gen}|{var}|auc_{ev}'] = a[0]
    out.table(pd.DataFrame(rows), 'TC2_variants_colour_realism', 'Part B. The same starting noises (ids 0..%d of Phase 1) '
              'sampled three ways. Colour outlier: Mahalanobis distance of the image mean CIELAB above the 99th percentile '
              'of real images (Phase 1b rule). KID and W1 use real images that were not used as colour targets.' % (D.N - 1))
    out.table(pd.DataFrame(grp), 'TC2b_variants_class_signal', 'Melanoma versus benign separability of each generator and '
              'variant, with a logistic regression trained on evaluator outputs of real images only.')
    # does V2 reach the colour it aims at? (targets are real-image colours; V0 images of the same noises as the reference)
    acc = []
    for gen, glab in GENS:
        for cls in ['mel', 'ben']:
            d2, d0 = D.get(gen, 'V2', cls), D.get(gen, 'V0', cls)
            if d2 is None or 'target_r' not in d2['meta']: continue
            T = d2['meta'][['target_r', 'target_g', 'target_b']].values
            for lab, d in [('V2', d2), ('V0 (same noises, no target)', d0)]:
                if d is None: continue
                Y = d['imgs'].float().div(127.5).sub(1.0).mean((2, 3)).numpy()
                r2 = [1 - ((T[:, k] - Y[:, k]) ** 2).sum() / ((T[:, k] - T[:, k].mean()) ** 2).sum() for k in range(3)]
                acc.append({'Generator': glab, 'Class': cls, 'Images': lab,
                            'MAE to target R/G/B': '/'.join(f"{np.abs(T[:, k] - Y[:, k]).mean():.3f}" for k in range(3)),
                            'R2 achieved vs target R/G/B': '/'.join(f"{x:.2f}" for x in r2)})
    if acc:
        out.table(pd.DataFrame(acc), 'TC2c_V2_colour_targeting', 'Does the colour-calibrated start reach its target colour? '
                  'Target: mean RGB (model scale [-1, 1]) of a randomly drawn real image of the same class. MAE: mean absolute '
                  'error per channel. The V0 rows compare the original images of the same noises with the same targets.')
        R['v2_targeting'] = acc
    REP.append("\n## Part B: sampling-time variants\n\n" + pd.DataFrame(rows)[['Generator', 'Class', 'Variant', 'Colour outliers %',
               'KID x1e3 (Inception)', 'Diversity']].to_string(index=False) + "\n\n" + pd.DataFrame(grp).to_string(index=False) + "\n")
    if acc:
        REP.append("\nV2 colour targeting:\n\n" + pd.DataFrame(acc).to_string(index=False) + "\n")
    # ---- figures
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6), sharex=True, sharey=True)
    for ax, (var, vlab) in zip(axes, VARS):
        ax.scatter(Rl[:, 1], Rl[:, 2], s=2, color='#9a9993', alpha=0.4, lw=0, label='Real', rasterized=True)
        for gen, col in C_GEN.items():
            for cls in ['mel', 'ben']:
                d = D.get(gen, var, cls)
                if d is not None:
                    ax.scatter(d['lab'][:, 1], d['lab'][:, 2], s=2, color=col, alpha=0.4, lw=0, rasterized=True,
                               label=dict(GENS)[gen] if cls == 'mel' else None)
        ax.set_title(vlab, fontsize=7); ax.set_xlabel('mean a* (green to red)')
    axes[0].set_ylabel('mean b* (blue to yellow)'); axes[0].legend(markerscale=4, fontsize=5.5, loc='lower left')
    fig.tight_layout(); out.fig(fig, 'figC_colour_by_variant')
    df = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.5))
    x = np.arange(len(GENS)); w = 0.26
    for i, (var, vlab) in enumerate(VARS):
        sub = df[df['Variant'] == vlab]
        o = [sub[sub['Generator'] == g]['Colour outliers %'].mean() for _, g in GENS]
        k = [sub[(sub['Generator'] == g) & (sub['Class'] == 'mel')]['KID x1e3 (Inception)'].mean() for _, g in GENS]
        axes[0].bar(x + (i - 1) * w, o, w * 0.92, color=C_VAR[var], label=vlab, edgecolor='white', linewidth=0.8)
        axes[1].bar(x + (i - 1) * w, k, w * 0.92, color=C_VAR[var], label=vlab, edgecolor='white', linewidth=0.8)
    for ax, t in [(axes[0], 'Colour outliers (%, mean of 2 classes)'), (axes[1], 'Inception KID x1e3, melanoma')]:
        ax.set_xticks(x); ax.set_xticklabels([TWO_LINE[g] for g, _ in GENS], fontsize=6.5); ax.set_ylabel(t, fontsize=7)
    axes[0].axhline(cfg['decision']['max_colour_outliers'], color=INK2, ls=':', lw=0.8)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc='upper center', ncol=3, fontsize=6.5, bbox_to_anchor=(0.5, 1.09))
    fig.tight_layout(); out.fig(fig, 'figC_outliers_and_kid')
    # image grid: 4 noises x 3 variants per generator/class
    k = min(4, D.N); rows_ = [(g, c) for g, _ in GENS for c in ['mel', 'ben']]
    fig, axes = plt.subplots(len(rows_), 3 * k, figsize=(7.2, 0.62 * len(rows_) + 0.4))
    for r, (gen, cls) in enumerate(rows_):
        for vi, (var, vlab) in enumerate(VARS):
            d = D.get(gen, var, cls)
            for c in range(k):
                ax = axes[r, vi * k + c]; ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
                for sp in ax.spines.values(): sp.set_visible(False)
                if d is not None: ax.imshow(d['imgs'][c].permute(1, 2, 0).numpy())
                if r == 0 and c == 0: ax.set_title(vlab, fontsize=6, loc='left')
        axes[r, 0].set_ylabel(TWO_LINE[gen] + ('\n(melanoma)' if cls == 'mel' else '\n(benign)'), fontsize=5)
    fig.subplots_adjust(left=0.09, right=0.995, top=0.95, bottom=0.005, wspace=0.04, hspace=0.04)
    fig.text(0.5, 0.985, 'Same column position = same starting noise', ha='center', fontsize=6)
    out.fig(fig, 'figC_examples_by_variant')


def traj_table(rec_meta, Y, concept_idx, levels_expected=None):
    """meta: DataFrame with concept_idx, level, noise_id; Y: (n, K). Returns {j: (T (traj x levels x K), levels)}"""
    out = {}
    for j in concept_idx:
        sel = rec_meta['concept_idx'].values == j
        if sel.sum() == 0: continue
        lev = np.sort(np.unique(rec_meta['level'].values[sel])); nid = np.sort(np.unique(rec_meta['noise_id'].values[sel]))
        li = np.searchsorted(lev, rec_meta['level'].values[sel]); ni = np.searchsorted(nid, rec_meta['noise_id'].values[sel])
        T = np.full((len(nid), len(lev), Y.shape[1]), np.nan); T[ni, li] = Y[sel]
        out[j] = (T, lev)
    return out


def dose_rows(tag, meta, outs, anchors, concepts, B):
    """outs: {ev: Y}; anchors: {ev: (real_mel_Y, real_ben_Y)}"""
    rows = []
    for ev, Y in outs.items():
        if Y is None: continue
        rm, rb = anchors[ev]; sd = np.sqrt((np.nanvar(rm, 0) + np.nanvar(rb, 0)) / 2) + 1e-9
        tt = traj_table(meta, Y, concepts); fam = []
        for j, (T, lev) in tt.items():
            if ev in ('cbma', 'cbmb'): ks = [j]
            elif ev == 'isic18': ks = [ISIC18_ATTRS.index(DERM_TO_ISIC18[CONCEPT_NAMES[j]])] if CONCEPT_NAMES[j] in DERM_TO_ISIC18 else []
            else: ks = [DARK_COLS.index('lesion_darkness')] if CONCEPT_NAMES[j] == 'pigmentation' else []
            for k in ks:
                Tk = T[:, :, k]; ok = np.isfinite(Tk).all(1)
                if ok.sum() < 5: continue
                slope, rho = A.traj_stats(Tk[ok], lev); p, rb_ = A.one_sample_test(slope)
                dz = (Tk[ok][:, -1] - Tk[ok][:, 0]) / sd[k]; dm, dlo, dhi = A.boot_mean(dz, B)
                fam.append({'Set': tag, 'Evaluator': {'cbma': 'CBM-A', 'cbmb': 'CBM-B', 'isic18': 'ISIC-2018 detector', 'dark': 'Lesion darkness'}[ev],
                            'Concept': CONCEPT_NAMES[j], 'n traj': int(ok.sum()), 'Median slope': float(np.median(slope)),
                            '% rising': float(np.mean(slope > 0) * 100), 'p': p, 'Δ 0→1 (real-SD units)': dm, 'Δ lo': dlo, 'Δ hi': dhi,
                            'curve': np.nanmean((Tk[ok] - np.nanmean(rb, 0)[k]) / sd[k], 0).tolist(), 'levels': lev.tolist()})
        q = A.bh([f['p'] for f in fam])
        for f, qq in zip(fam, q): f['p_BH'] = float(qq)
        rows += fam
    return rows


@section('Part C: does concept control survive each variant?')
def s_partC(cfg, out, R, D):
    B = cfg['analysis']['n_boot']; conc = cfg['sweep_concepts']; n = cfg['sweep_n']
    anchors = {ev: (D.real['mel'][src][ev], D.real['ben'][src][ev]) for ev, src in [('cbma', 'A'), ('cbmb', 'B'), ('isic18', 'B'), ('dark', 'A')]
               if ev in D.real['mel'][src]}
    rows = []
    s = load_set(D.s1b('sweep_mel')); m = pd.DataFrame(s['meta']); a, b = D.A1b('sweep_mel'), D.B1b('sweep_mel')
    sel = (m['concept_idx'].isin(conc) & (m['noise_id'] < n)).values
    outs = {'cbmb': b['cbmb'][sel], 'cbma': a['cbma'][sel], 'isic18': b['isic18'][sel] if 'isic18' in b else None, 'dark': a['dark'][sel]}
    rows += dose_rows('V0 original sampling', m[sel].reset_index(drop=True), outs, anchors, conc, B)
    for var, vlab in VARS[1:]:
        p = D.s1c(f'sweepvar_{var}')
        if not os.path.exists(p): continue
        s = load_set(p); mm = pd.DataFrame(s['meta']); a, b = D.A1c(f'sweepvar_{var}'), D.B1c(f'sweepvar_{var}')
        rows += dose_rows(vlab, mm, {'cbmb': b['cbmb'], 'cbma': a['cbma'], 'isic18': b.get('isic18'), 'dark': a['dark']}, anchors, conc, B)
    R['partC'] = [{k: v for k, v in r.items() if k not in ('curve', 'levels')} for r in rows]
    tab = pd.DataFrame([{'Sampling': r['Set'], 'Evaluator': r['Evaluator'], 'Concept': r['Concept'], 'n traj': r['n traj'],
                         'Median slope': f"{r['Median slope']:+.3f}", '% rising': f"{r['% rising']:.0f}%", 'p (BH)': fp(r['p_BH']),
                         'Δ 0→1 in real-SD units [95% CI]': fci(r['Δ 0→1 (real-SD units)'], r['Δ lo'], r['Δ hi'], 2)} for r in rows])
    out.table(tab, 'TC3_dose_response_by_variant', 'Part C. Dose-response from the melanoma prototype for pigmentation, '
              'pigment network and blue-whitish veil, with the same %d starting noises under each sampling variant. '
              'p: two-sided Wilcoxon signed-rank test of per-trajectory slopes, BH-corrected within each evaluator and variant.' % n)
    REP.append("\n## Part C: concept control under each variant\n\n" + tab.to_string(index=False) + "\n")
    fig, axes = plt.subplots(1, len(conc), figsize=(7.2, 2.4), sharey=False)
    for ax, j in zip(np.atleast_1d(axes), conc):
        for var, vlab in VARS:
            r = [x for x in rows if x['Set'] == vlab and x['Evaluator'] == 'CBM-B' and x['Concept'] == CONCEPT_NAMES[j]]
            if r: ax.plot(r[0]['levels'], r[0]['curve'], color=C_VAR[var], marker=M_VAR[var], ms=3, label=vlab)
        ax.set_title(f"{SHORT[j]} ({CONCEPT_NAMES[j].replace('_', ' ')})", fontsize=7); ax.set_xlabel('Commanded level')
    np.atleast_1d(axes)[0].set_ylabel('CBM-B output (real-SD units)'); np.atleast_1d(axes)[0].legend(fontsize=5.5)
    fig.tight_layout(); out.fig(fig, 'figC_dose_response_by_variant')
    # pigmentation strips: V0 versus V2
    j = CONCEPT_NAMES.index('pigmentation')
    if j in conc and os.path.exists(D.s1c('sweepvar_V2')):
        s0 = load_set(D.s1b('sweep_mel')); m0 = pd.DataFrame(s0['meta'])
        s2 = load_set(D.s1c('sweepvar_V2')); m2 = pd.DataFrame(s2['meta']); lev = np.sort(m2['level'].unique())
        fig, axes = plt.subplots(6, len(lev), figsize=(0.62 * len(lev) + 0.6, 4.2))
        for blk, (S_, M_, lab) in enumerate([(s0, m0, 'V0'), (s2, m2, 'V2')]):
            for r in range(3):
                for c, l in enumerate(lev):
                    i = np.where((M_['concept_idx'].values == j) & (np.isclose(M_['level'].values, l)) & (M_['noise_id'].values == r))[0][0]
                    ax = axes[blk * 3 + r, c]; ax.imshow(S_['imgs'][i].permute(1, 2, 0).numpy()); ax.set_xticks([]); ax.set_yticks([])
                    ax.grid(False)
                    for sp in ax.spines.values(): sp.set_visible(False)
                    if blk * 3 + r == 5: ax.set_xlabel(f"{l:.1f}", fontsize=6)
                axes[blk * 3 + r, 0].set_ylabel(f'{lab}\nnoise {r}', fontsize=6)
        fig.suptitle('Pigmentation 0 to 1 from the melanoma prototype\nV0: original sampling (top 3 rows); '
                     'V2: colour-calibrated start (bottom 3 rows); same noise in each row pair', fontsize=6.5)
        fig.tight_layout(pad=0.2); out.fig(fig, 'figC_pigmentation_strip_V0_vs_V2')


@section('Part D: is concept control a colour effect? (colour constancy)')
def s_partD(cfg, out, R, D):
    B = cfg['analysis']['n_boot']; rows = []
    s = load_set(D.s1b('sweep_mel')); m = pd.DataFrame(s['meta']); a, b = D.A1b('sweep_mel'), D.B1b('sweep_mel')
    anchors = {ev: (D.real['mel'][src][ev], D.real['ben'][src][ev]) for ev, src in [('cbma', 'A'), ('cbmb', 'B'), ('isic18', 'B'), ('dark', 'A')]
               if ev in D.real['mel'][src]}
    rows += dose_rows('Original images', m, {'cbmb': b['cbmb'], 'cbma': a['cbma'], 'isic18': b.get('isic18'), 'dark': a['dark']},
                      anchors, list(range(7)), B)
    ac, bc = D.A1c('cc_sweep_mel'), D.B1c('cc_sweep_mel')
    ccm, ccb = (D.A1c('cc_real_mel'), D.B1c('cc_real_mel')), (D.A1c('cc_real_ben'), D.B1c('cc_real_ben'))
    anchors_cc = {'cbma': (ccm[0]['cbma'], ccb[0]['cbma']), 'cbmb': (ccm[1]['cbmb'], ccb[1]['cbmb']), 'dark': (ccm[0]['dark'], ccb[0]['dark'])}
    if 'isic18' in ccm[1]: anchors_cc['isic18'] = (ccm[1]['isic18'], ccb[1]['isic18'])
    rows += dose_rows('After colour constancy', m, {'cbmb': bc['cbmb'], 'cbma': ac['cbma'], 'isic18': bc.get('isic18'), 'dark': ac['dark']},
                      anchors_cc, list(range(7)), B)
    R['partD'] = [{k: v for k, v in r.items() if k not in ('curve', 'levels')} for r in rows]
    tab = pd.DataFrame([{'Images': r['Set'], 'Evaluator': r['Evaluator'], 'Concept': r['Concept'],
                         'Median slope': f"{r['Median slope']:+.3f}", '% rising': f"{r['% rising']:.0f}%", 'p (BH)': fp(r['p_BH']),
                         'Δ 0→1 in real-SD units [95% CI]': fci(r['Δ 0→1 (real-SD units)'], r['Δ lo'], r['Δ hi'], 2)} for r in rows])
    out.table(tab, 'TC4_colour_constancy_robustness', 'Part D. Phase 1 dose-response (melanoma prototype, 32 noises) re-scored '
              'after Shades-of-Gray colour constancy (p = 6) of every image, including the real anchors. If the control were a '
              'colour cast effect, it would disappear here.')
    REP.append("\n## Part D: colour-constancy robustness (Phase 1 sweep)\n\n" + tab[tab['Evaluator'] == 'CBM-B'].to_string(index=False) + "\n")


@section('Decision (rule fixed before the run)')
def s_decision(cfg, out, R):
    Dc = cfg['decision']; V = R.get('variants', {}); C = R.get('partC', []); Dd = R.get('partD', [])
    def ctrl(setlab, concept, rows=C, ev='CBM-B'):
        r = [x for x in rows if x['Set'] == setlab and x['Evaluator'] == ev and x['Concept'] == concept]
        return bool(r) and r[0]['p_BH'] < Dc['alpha'] and r[0]['Median slope'] > 0
    res = {'rule': Dc['text'], 'variants': {}}
    for var, vlab in VARS[1:]:
        o = [V.get(f'{g}|{var}|{c}', {}).get('outliers', np.nan) for g in ['main', 'base'] for c in ['mel', 'ben']]
        k = [(V.get(f'{g}|{var}|{c}', {}).get('kid', np.nan), V.get(f'{g}|V0|{c}', {}).get('kid', np.nan)) for g in ['main', 'base'] for c in ['mel', 'ben']]
        colour_ok = all(np.isfinite(x) and x <= Dc['max_colour_outliers'] for x in o)
        kid_ok = all(np.isfinite(a) and np.isfinite(b) and a <= b * (1 + Dc['kid_tolerance']) for a, b in k)
        control_ok = ctrl(vlab, 'pigmentation') and ctrl(vlab, 'pigment_network')
        res['variants'][var] = {'colour_outliers_main_base': o, 'colour_ok': colour_ok, 'kid_ok': kid_ok, 'control_ok': control_ok,
                                'acceptable': colour_ok and kid_ok and control_ok, 'mean_outliers': float(np.nanmean(o))}
    acc = [v for v, x in res['variants'].items() if x['acceptable']]
    if acc:
        best = min(acc, key=lambda v: res['variants'][v]['mean_outliers'])
        res['decision'] = f"ADOPT {best}: it fixes the colour and keeps realism and concept control. Use it for all final results (option B)."
    else:
        res['decision'] = "NO SAMPLING-TIME FIX IS ACCEPTABLE: go to option C (fine-tune with a zero-terminal-SNR schedule)."
    cc_ok = ctrl('After colour constancy', 'pigmentation', Dd) and ctrl('After colour constancy', 'pigment_network', Dd)
    res['colour_constancy_control_survives'] = cc_ok
    if cfg.get('dry_run'): res['decision'] = 'DRY RUN ONLY, not a decision. Would read: ' + res['decision']
    save_json(res, os.path.join(out.root, 'results', 'phase1c_decision.json')); R['decision'] = res
    lines = [f"# ClinDiffXAI Phase 1c report (colour)\n\n**Decision: {res['decision']}**\n", f"Rule (fixed before the run): {Dc['text']}\n"]
    for v, x in res['variants'].items():
        lines.append(f"- {v}: colour outliers (main/base, mel/ben) {', '.join(f'{y:.1f}%' for y in x['colour_outliers_main_base'])} | "
                     f"colour {'ok' if x['colour_ok'] else 'NOT ok'} | KID {'ok' if x['kid_ok'] else 'NOT ok'} | "
                     f"control {'ok' if x['control_ok'] else 'NOT ok'}")
    lines.append(f"- Colour constancy (Phase 1 sweep): pigmentation and pigment-network control {'SURVIVES' if cc_ok else 'DOES NOT survive'} "
                 f"(CBM-B, p_BH < {Dc['alpha']}, median slope > 0).")
    REP.insert(0, '\n'.join(lines) + '\n')


def main():
    cfg = load_cfg(sys.argv[1]); out = Out(cfg['outputs']); t0 = time.time(); R = {}
    D = Data(cfg)
    s_partA(cfg, out, R); s_partB(cfg, out, R, D); s_partC(cfg, out, R, D); s_partD(cfg, out, R, D); s_decision(cfg, out, R)
    if WARN: REP.append("\n## Warnings\n\n" + '\n'.join(f"- {w}" for w in WARN))
    REP.append(f"\n_Analysis time {(time.time() - t0) / 60:.1f} min._\n")
    with open(os.path.join(out.root, 'phase1c_report.md'), 'w') as f:
        f.write('\n'.join(REP))
    save_json(R, os.path.join(out.root, 'results', 'phase1c_results.json'))
    print('\n'.join(REP))


if __name__ == '__main__':
    main()
''',
}
for fn, src in FILES.items():
    with open(os.path.join(CODE, fn), 'w') as f:
        f.write(src)
say(f'wrote {len(FILES)} code files to {CODE}')

# ---------------------------------------------------------------- 3. config
n_eval = int(S1B['n_eval'])
N_VAR = max(4, n_eval // 3) if DRY_RUN else 256
assert n_eval - N_VAR >= 12, 'too few Phase 1 images left to fit the calibration map'
cfg = {
    'work': WORK, 'work1b': W1B, 'outputs': OUTD, 'paths': cfg1b['paths'], 'image_size': cfg1b['image_size'],
    'eval_size': cfg1b['eval_size'], 'incep_size': cfg1b['incep_size'], 'pretrained': cfg1b['pretrained'],
    'dry_run': DRY_RUN or bool(cfg1b.get('dry_run')),
    'n_var': N_VAR, 'n_fit_from': N_VAR,
    'sweep_n': min(6 if DRY_RUN else 16, int(S1B['sweep_n'])), 'sweep_concepts': [3, 0, 1],   # Pig, PN, BWV
    'seeds': {**cfg1b['seeds'], 'target_mel': 5001, 'target_ben': 5002, 'target_sweep': 5003},
    'sizes': {'cfg_scale': S1B['cfg_scale'], 'ddim_steps': S1B['ddim_steps'], 'gen_bs': S1B['gen_bs'],
              'sweep_levels': S1B['sweep_levels'], 'sweep_n_phase1': S1B['sweep_n']},
    'analysis': {'n_boot': 200 if DRY_RUN else 2000, 'n_boot_auc': 100 if DRY_RUN else 1000,
                 'kid_subset': 200, 'kid_n_sub': 10 if DRY_RUN else 100},
    'decision': {'max_colour_outliers': 20.0, 'kid_tolerance': 0.10, 'alpha': 0.05, 'partA_r2': 0.50, 'text':
                 'A variant fixes colour if its colour-outlier rate is <= 20% for the main model and the baseline in both '
                 'classes. It is acceptable if, in addition, its Inception KID is not more than 10% worse than V0 for each of '
                 'these 4 sets and the CBM-B dose-response for pigmentation and pigment network stays positive and '
                 'significant (two-sided Wilcoxon signed-rank test on per-trajectory slopes, BH-corrected p < 0.05, median '
                 'slope > 0). Among acceptable variants, the lower mean outlier rate wins. If none is acceptable: option C.'},
    'run_id': RUN_ID,
}
CFG = os.path.join(WORK, 'cfg.json')
with open(CFG, 'w') as f:
    json.dump(cfg, f, indent=2)

try:
    n_gpu = len([l for l in subprocess.run(['nvidia-smi', '-L'], capture_output=True, text=True).stdout.splitlines() if l.startswith('GPU')])
except Exception:
    n_gpu = 0
say(f'GPUs visible: {n_gpu}')
if n_gpu == 0 and not os.environ.get('P1_INPUT_ROOT'):
    raise RuntimeError('No GPU found. Set Accelerator = GPU T4 x2.')

# ---------------------------------------------------------------- 4. job runner (dependency-aware, both GPUs)
GEN_V1 = ['var_main_V1', 'var_base_V1', 'var_realc_V1', 'sweep_V1']      # need nothing
GEN_V2 = ['var_main_V2', 'var_base_V2', 'var_realc_V2', 'sweep_V2']      # need the calibration from prep
DEPS = {j: [] for j in GEN_V1}
DEPS.update({j: ['prep'] for j in GEN_V2})
DEPS['score_A'] = ['prep'] + GEN_V1 + GEN_V2
DEPS['score_B'] = ['prep'] + GEN_V1 + GEN_V2
# estimated minutes on one T4 (about 1 s per CFG image at 50 DDIM steps in Phase 1); used only for the order
EST = {j: 9 for j in GEN_V1 + GEN_V2}; EST.update({'sweep_V1': 9.5, 'sweep_V2': 9.5, 'score_A': 6, 'score_B': 5})
GPU_JOBS = GEN_V1 + GEN_V2 + ['score_A', 'score_B']
STATUS = {}; LOCK = threading.Lock()


def is_done(job):
    return os.path.exists(os.path.join(WORK, 'done', f'{job}.ok'))


for j in ['prep'] + GPU_JOBS:
    if is_done(j): STATUS[j] = 'done earlier'


def run_job(job, gpu, rid):
    if not current_run(rid): return
    env = os.environ.copy(); env['CUDA_VISIBLE_DEVICES'] = '' if gpu is None else str(gpu)
    env['PYTHONUNBUFFERED'] = '1'; env['P1C_RUN_ID'] = rid
    with open(os.path.join(WORK, 'logs', f'{job}.log'), 'a') as lf:
        lf.write(f"\n===== run {rid} | GPU {gpu} | {time.strftime('%H:%M:%S')}\n"); lf.flush()
        rc = subprocess.run([sys.executable, '-u', WORKER, job, CFG], stdout=lf, stderr=subprocess.STDOUT, env=env, cwd=CODE).returncode
    if not current_run(rid): return
    with LOCK:
        STATUS[job] = 'ok' if rc == 0 and is_done(job) else f'FAILED (exit {rc})'


def ok(job):
    return STATUS.get(job) in ('ok', 'done earlier')


def next_job():
    """first ready job (longest first); None if nothing is left; 'wait' if jobs wait for others"""
    with LOCK:
        left = [j for j in GPU_JOBS if j not in STATUS]
        if not left: return None
        for j in left:
            if any(STATUS.get(d, '').startswith(('FAILED', 'skipped')) for d in DEPS[j]):
                STATUS[j] = 'skipped (a job it needs failed)'
        left = [j for j in GPU_JOBS if j not in STATUS]
        if not left: return None
        ready = sorted([j for j in left if all(ok(d) for d in DEPS[j])], key=lambda j: -EST[j])
        if not ready: return 'wait'
        STATUS[ready[0]] = 'running'
        return ready[0]


def gpu_worker(gpu, rid):
    while current_run(rid):
        j = next_job()
        if j is None: return
        if j == 'wait':
            time.sleep(5); continue
        RUNNING[gpu] = j; run_job(j, gpu, rid); RUNNING[gpu] = None


def cpu_prep(rid):
    if 'prep' in STATUS: return
    with LOCK: STATUS['prep'] = 'running'
    RUNNING['cpu'] = 'prep'; run_job('prep', None, rid); RUNNING['cpu'] = None


def last_line(job):
    p = os.path.join(WORK, 'logs', f'{job}.log')
    if not os.path.exists(p): return ''
    with open(p, 'rb') as f:
        f.seek(0, 2); n = f.tell(); f.seek(max(0, n - 4000))
        lines = [l for l in f.read().decode('utf8', 'ignore').splitlines() if l.strip()]
    return lines[-1][-110:] if lines else ''


gpus = list(range(n_gpu)) if n_gpu >= 1 else [None, None]   # [None, None]: offline test on CPU only
if n_gpu == 1:
    print('  WARNING: only one GPU; the GPU jobs run one after another (slower).')
RUNNING = {g: None for g in gpus}; RUNNING['cpu'] = None
T0 = time.time()
say('===== STAGE 1: prep on CPU | V1 generation at once, V2 after prep | scoring at the end =====')
threads = [threading.Thread(target=cpu_prep, args=(RUN_ID,), daemon=True)]
threads += [threading.Thread(target=gpu_worker, args=(g, RUN_ID), daemon=True) for g in gpus]
for i, t in enumerate(threads):
    t.start()
    if i == 0: time.sleep(2)
tick = 0
while any(t.is_alive() for t in threads):
    time.sleep(5); tick += 5
    if tick % 60 == 0:
        print(f"--- {(time.time() - T0) / 60:.1f} min elapsed", flush=True)
        for g, j in list(RUNNING.items()):
            if j: print(f"    {'GPU' + str(g) if g not in (None, 'cpu') else ('CPU' if g == 'cpu' else 'CPU-test')}: {j:<13} | {last_line(j)}", flush=True)
for j in ['prep'] + GPU_JOBS:
    print(f"    {j:<13} {STATUS.get(j, 'not run')}", flush=True)

say('===== STAGE 2: analysis (CPU) =====')
with open(os.path.join(WORK, 'logs', 'analysis1c.log'), 'w') as lf:
    rc = subprocess.run([sys.executable, '-u', os.path.join(CODE, 'analysis1c.py'), CFG], stdout=lf, stderr=subprocess.STDOUT,
                        cwd=CODE, env={**os.environ, 'CUDA_VISIBLE_DEVICES': ''}).returncode
STATUS['analysis'] = 'ok' if rc == 0 else f'FAILED (exit {rc})'

# ---------------------------------------------------------------- 5. package
for fn in FILES: shutil.copy(os.path.join(CODE, fn), os.path.join(OUTD, 'code_' + fn))
shutil.copy(CFG, os.path.join(OUTD, 'cfg.json'))
os.makedirs(os.path.join(OUTD, 'logs'), exist_ok=True); os.makedirs(os.path.join(OUTD, 'results'), exist_ok=True)
for lg in glob.glob(os.path.join(WORK, 'logs', '*.log')): shutil.copy(lg, os.path.join(OUTD, 'logs'))
for fn in glob.glob(os.path.join(WORK, 'results', '*')): shutil.copy(fn, os.path.join(OUTD, 'results'))
zip_base = os.path.join(WORK_ROOT, 'phase1c_dryrun_outputs' if DRY_RUN else 'phase1c_outputs')
shutil.make_archive(zip_base, 'zip', OUTD)
print('\n' + '=' * 78)
print(f"PHASE 1c {'DRY RUN ' if DRY_RUN else ''}FINISHED in {(time.time() - T0) / 60:.1f} min")
for j, s in STATUS.items(): print(f"  {j:<13} {s}")
rep = os.path.join(OUTD, 'phase1c_report.md')
if os.path.exists(rep):
    print('=' * 78); print(open(rep).read())
else:
    print('No report: see', os.path.join(WORK, 'logs', 'analysis1c.log'))
print(f"\nDownload: {zip_base}.zip  (tables, figures, report, logs, code)")
print(f"New images stay in {os.path.join(WORK, 'sets')}.")
