# ============================================================
# NOTEBOOK 3b-TPU — BASELINE: Class-Label-Conditioned DDPM, 128×128
# TPU v5e-8 (8-core multi-process)
#
# ⚠⚠⚠ READ BEFORE RUNNING ⚠⚠⚠
# Same untested-on-real-hardware caveat as NB3-TPU applies here. This
# mirrors NB3-TPU exactly except for conditioning dimensionality — same
# architecture, same CFG/EMA, same optimizer, same effective batch size.
#
# MANDATORY SEQUENCE (same as NB3-TPU):
#   1. Run TPU_Smoke_Test_8core.py first (or confirm it already passed
#      for NB3-TPU in a FRESH kernel — see note below).
#   2. Run THIS script with SESSION_END=3 as a verification run. Compare
#      loss trajectory shape against NB3-TPU's verification run AND
#      against the GPU NB3b baseline. They should look similar in shape
#      (not necessarily identical numbers, since this is class-label
#      conditioning, not concept conditioning — but the loss SCALE and
#      DECREASE RATE should be comparable).
#   3. Only then commit to SESSION_END=40, 80, 120, 150.
#
# ⚠ CRITICAL — RUNTIME CONTAMINATION ACROSS NOTEBOOKS:
#   If you already ran NB3-TPU successfully in a Kaggle kernel and are
#   now switching to this baseline script, you MUST use a FRESH kernel
#   session (restart/factory-reset), not the same running kernel — once
#   xmp.spawn() has run once in a kernel, do not run any further xm.*
#   calls at module scope in that same kernel before spawning again.
#   This script (like NB3-TPU) never touches xm at module scope, so it
#   is itself safe — but only if the kernel it runs in hasn't been
#   contaminated by something else first.
#
# WHAT'S DIFFERENT FROM NB3-TPU (mirrors the GPU NB3b exactly):
#   - Conditioning: one-hot class label [is_benign, is_melanoma] instead
#     of the 7-dim CBM-A-predicted concept vector
#   - CCFS evaluation still uses the SAME 7-dim EVAL_MEL_VEC/EVAL_BEN_VEC
#     prototypes and SAME CBM-A model as NB3-TPU — isolates exactly one
#     variable (conditioning richness) while holding the metric fixed
#   - All output filenames suffixed "_baseline_tpu"
# ============================================================

import os, time, csv, warnings

# ⚠ KAGGLE TPU PLATFORM BUG FIX — must run before any torch_xla import/use.
# Kaggle's TPU kernels pre-set TPU_PROCESS_ADDRESSES at the environment level
# in a way that conflicts with xmp.spawn()'s own single-host multi-core setup,
# causing: "TPU initialization failed: Invalid --..._slice_builder_worker_
# addresses specified. Expected 8 worker addresses, got 1." This is a
# confirmed Kaggle platform issue, not a bug in this script — see
# https://www.kaggle.com/discussions/product-feedback/473974 and
# https://github.com/Lightning-AI/pytorch-lightning/issues/20244
os.environ.pop('TPU_PROCESS_ADDRESSES', None)

from copy import deepcopy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import make_grid, save_image
import torchvision.models as tv_models
from diffusers import DDPMScheduler, DDIMScheduler
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
warnings.filterwarnings('ignore')

import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.distributed.parallel_loader as pl


# ============================================================
# CONFIG
# ============================================================

PSEUDO_CSV       = "/kaggle/input/datasets/shohanurrahman7/pseudo-labels-v3/isic2020_pseudo_labels_v3.csv"
ISIC_IMG_DIR     = "/kaggle/input/datasets/sumaiyabinteshahid/isic-challenge-dataset-2020/ISIC_2020_Dataset/train"
CBM_A_CHECKPOINT = "/kaggle/input/datasets/shohanurrahman7/cbma-v3/best_cbm_model.pth"

SESSION_END  = 3     # ⚠ START WITH 3 FOR THE VERIFICATION RUN
RESUME_FROM  = None

IMAGE_SIZE        = 128
PER_CORE_BATCH    = 4
GRAD_ACCUM        = 1      # 8 cores × 4 × 1 = effective batch 32 (matches all other runs)
NUM_TIMESTEPS     = 1000
SAVE_EVERY        = 10
EVAL_EVERY        = 40
SEED              = 42

CFG_DROP_PROB = 0.15
CFG_SCALE     = 3.5
EMA_DECAY     = 0.999

LOG_CSV = 'nb3b_tpu_train_log.csv'

# Needed only for CCFS evaluation against CBM-A's output, NOT as model input
CONCEPT_NAMES = ['pigment_network', 'blue_whitish_veil', 'vascular_structures',
                  'pigmentation', 'streaks', 'dots_and_globules', 'regression_structures']
NUM_CONCEPTS = len(CONCEPT_NAMES)

# Model conditioning: one-hot class label, NOT concepts
CLASS_NAMES = ['benign', 'melanoma']
NUM_CLASSES = 2
GEN_BEN_VEC_LIST = [1.0, 0.0]
GEN_MEL_VEC_LIST = [0.0, 1.0]

# Evaluation only: same 7-dim prototypes as NB3-TPU, for fair CCFS comparison
EVAL_MEL_VEC_LIST = [0.8292, 0.7781, 0.9926, 0.8918, 0.4585, 0.9256, 0.8769]
EVAL_BEN_VEC_LIST = [0.0800, 0.0002, 0.0047, 0.0055, 0.0018, 0.1501, 0.0003]


# ============================================================
# ARCHITECTURE — byte-identical to NB3-TPU (verified shape-trace there).
# Only the instantiation parameter (num_concepts=2) differs, far below.
# ============================================================

class ConceptEmbedding(nn.Module):
    def __init__(self, n, d=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n,d),nn.SiLU(),nn.Linear(d,d),nn.SiLU())
    def forward(self, c): return self.net(c)


class ResBlock(nn.Module):
    def __init__(self, ic, oc, td, cd):
        super().__init__()
        self.n1=nn.GroupNorm(8,ic); self.c1=nn.Conv2d(ic,oc,3,padding=1)
        self.n2=nn.GroupNorm(8,oc); self.c2=nn.Conv2d(oc,oc,3,padding=1)
        self.tp=nn.Linear(td,oc);   self.cp=nn.Linear(cd,oc)
        self.sk=nn.Conv2d(ic,oc,1) if ic!=oc else nn.Identity()
        self.a=nn.SiLU()
    def forward(self, x, t, c):
        h=self.c1(self.a(self.n1(x)))
        h=h+self.tp(self.a(t))[:,:,None,None]
        h=h+self.cp(self.a(c))[:,:,None,None]
        return self.c2(self.a(self.n2(h)))+self.sk(x)


class SelfAttention(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.n=nn.GroupNorm(8,ch)
        self.a=nn.MultiheadAttention(ch,num_heads=4,batch_first=True)
    def forward(self, x):
        B,C,H,W=x.shape
        h=self.n(x).reshape(B,C,H*W).permute(0,2,1)
        h,_=self.a(h,h,h)
        return x+h.permute(0,2,1).reshape(B,C,H,W)


class ConceptConditionedUNet128(nn.Module):
    def __init__(self, image_size=128, num_concepts=7, base_channels=64):
        super().__init__()
        self.image_size = image_size
        C = [64, 64, 128, 256, 256]
        td = base_channels*4; cd = base_channels*2
        self.time_mlp = nn.Sequential(nn.Linear(base_channels,td),nn.SiLU(),nn.Linear(td,td))
        self.concept_emb = ConceptEmbedding(num_concepts, cd)
        self.enc_in = nn.Conv2d(3, C[0], 3, padding=1)

        self.enc = nn.ModuleList([
            nn.ModuleList([ResBlock(C[0],C[0],td,cd), ResBlock(C[0],C[0],td,cd)]),
            nn.ModuleList([ResBlock(C[0],C[1],td,cd), ResBlock(C[1],C[1],td,cd)]),
            nn.ModuleList([ResBlock(C[1],C[2],td,cd), ResBlock(C[2],C[2],td,cd), SelfAttention(C[2])]),
            nn.ModuleList([ResBlock(C[2],C[3],td,cd), ResBlock(C[3],C[3],td,cd), SelfAttention(C[3])]),
            nn.ModuleList([ResBlock(C[3],C[4],td,cd), ResBlock(C[4],C[4],td,cd), SelfAttention(C[4])]),
        ])
        self.down = nn.ModuleList([nn.Conv2d(C[i],C[i],3,stride=2,padding=1) for i in range(4)])
        self.mid  = nn.ModuleList([ResBlock(C[4],C[4],td,cd), SelfAttention(C[4]), ResBlock(C[4],C[4],td,cd)])

        self.dec = nn.ModuleList([
            nn.ModuleList([ResBlock(C[4]+C[3],C[3],td,cd), ResBlock(C[3],C[3],td,cd), SelfAttention(C[3])]),
            nn.ModuleList([ResBlock(C[3]+C[2],C[2],td,cd), ResBlock(C[2],C[2],td,cd), SelfAttention(C[2])]),
            nn.ModuleList([ResBlock(C[2]+C[1],C[1],td,cd), ResBlock(C[1],C[1],td,cd)]),
            nn.ModuleList([ResBlock(C[1]+C[0],C[0],td,cd), ResBlock(C[0],C[0],td,cd)]),
        ])
        self.up = nn.ModuleList([
            nn.ConvTranspose2d(C[4],C[4],2,stride=2),
            nn.ConvTranspose2d(C[3],C[3],2,stride=2),
            nn.ConvTranspose2d(C[2],C[2],2,stride=2),
            nn.ConvTranspose2d(C[1],C[1],2,stride=2),
        ])
        self.out = nn.Sequential(nn.GroupNorm(8,C[0]), nn.SiLU(), nn.Conv2d(C[0],3,3,padding=1))

    def sin_emb(self, t, d):
        half = d//2
        f = torch.exp(-torch.arange(half,device=t.device)*(np.log(10000)/(half-1)))
        a = t[:,None].float()*f[None]
        return torch.cat([torch.cos(a),torch.sin(a)],dim=-1)

    def forward(self, x, ts, concepts):
        te = self.time_mlp(self.sin_emb(ts, self.time_mlp[0].in_features))
        ce = self.concept_emb(concepts)
        h = self.enc_in(x); sk = []
        for i, blks in enumerate(self.enc):
            for b in blks: h = b(h) if isinstance(b,SelfAttention) else b(h,te,ce)
            sk.append(h)
            if i < 4: h = self.down[i](h)
        for b in self.mid:
            h = b(h) if isinstance(b,SelfAttention) else b(h,te,ce)

        usable_sk = sk[:-1]
        for i, (blks, up) in enumerate(zip(self.dec, self.up)):
            h = up(h)
            s = usable_sk[-(i+1)]
            if h.shape != s.shape:
                h = F.interpolate(h, size=s.shape[-2:])
            h = torch.cat([h, s], dim=1)
            for b in blks: h = b(h) if isinstance(b,SelfAttention) else b(h,te,ce)

        o = self.out(h)
        if o.shape[-1] != self.image_size:
            o = F.interpolate(o, size=(self.image_size, self.image_size))
        return o


# ============================================================
# CBM-A loader, CCFS, generation — identical to NB3-TPU
# ============================================================

def load_cbm_a(path, device):
    class _CBM(nn.Module):
        def __init__(self):
            super().__init__()
            bb = tv_models.efficientnet_v2_s(weights=None)
            in_f = bb.classifier[1].in_features; bb.classifier = nn.Identity()
            self.backbone = bb
            self.concept_feature = nn.Sequential(
                nn.Linear(in_f,512), nn.LayerNorm(512), nn.GELU(), nn.Dropout(0.3))
            self.concept_heads = nn.ModuleList([
                nn.Sequential(nn.Linear(512,128), nn.GELU(), nn.Dropout(0.2), nn.Linear(128,1))
                for _ in range(7)])
            self.diagnosis_head = nn.Sequential(
                nn.Linear(7,64), nn.GELU(), nn.Dropout(0.3), nn.Linear(64,2))
        def forward(self, x):
            f = self.backbone(x)
            if f.dim() > 2: f = f.mean([-2,-1])
            cf = self.concept_feature(f)
            cl = torch.cat([h(cf) for h in self.concept_heads], dim=1)
            cp = torch.sigmoid(cl)
            return cl, cp, self.diagnosis_head(cp)

    m = _CBM()
    ck = torch.load(path, map_location='cpu', weights_only=False)
    m.load_state_dict(ck['model_state_dict'], strict=True)
    return m.to(device).eval()


def ccfs_feedback(imgs, cvec, cbm, device):
    """Compares generated images against the SAME 7-dim eval prototype
    used in NB3-TPU — identical metric regardless of generator input."""
    norm = transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    rsz  = transforms.Resize((224,224))
    inp  = torch.stack([norm(rsz(i)) for i in imgs]).to(device)
    with torch.no_grad():
        _, det, _ = cbm(inp)
    det  = det.cpu()
    cond = cvec.unsqueeze(0).repeat(len(imgs), 1)
    return F.cosine_similarity(cond, det, dim=1).mean().item()


@torch.no_grad()
def generate_images(model, gen_cvec, n, steps, device, sz, cfg_scale=CFG_SCALE):
    model.eval()
    ddim = DDIMScheduler(num_train_timesteps=1000, beta_schedule='linear', clip_sample=True)
    ddim.set_timesteps(steps)
    imgs = torch.randn(n,3,sz,sz, device=device)
    cv = gen_cvec.unsqueeze(0).repeat(n,1).to(device)
    for t in ddim.timesteps:
        ts = torch.full((n,), t, device=device, dtype=torch.long)
        eps_cond   = model(imgs, ts, cv)
        eps_uncond = model(imgs, ts, torch.zeros_like(cv))
        noise_pred = eps_uncond + cfg_scale*(eps_cond - eps_uncond)
        imgs = ddim.step(noise_pred, t, imgs).prev_sample
        xm.mark_step()
    return ((imgs.clamp(-1,1)+1)/2).cpu()


# ============================================================
# DATA — same balancing as NB3-TPU, ClassLabelDataset instead of
# ConceptDataset (returns one-hot class label, not concept vector)
# ============================================================

def build_balanced(df, seed=42):
    mel = df[df['target'] == 1].copy()
    ben = df[df['target'] == 0].copy()
    mel5 = pd.concat([mel]*5, ignore_index=True)
    ben3 = ben.sample(n=len(mel5)*3, random_state=seed)
    out = pd.concat([mel5, ben3], ignore_index=True).sample(frac=1, random_state=seed)
    return out


class ClassLabelDataset(Dataset):
    def __init__(self, df, image_dir, image_size=128, augment=True):
        self.image_size = image_size
        tfms = ([transforms.Resize((image_size+16, image_size+16)),
                 transforms.RandomCrop(image_size),
                 transforms.RandomHorizontalFlip(0.5),
                 transforms.RandomVerticalFlip(0.5)]
                if augment else [transforms.Resize((image_size, image_size))])
        self.tfm = transforms.Compose(
            tfms + [transforms.ToTensor(), transforms.Normalize([0.5]*3, [0.5]*3)]
        )
        self.rows, self.paths = [], []
        for _, row in df.reset_index(drop=True).iterrows():
            p = os.path.join(image_dir, f"{row['image_name']}.jpg")
            if os.path.exists(p):
                self.rows.append(row.to_dict())
                self.paths.append(p)

    def __len__(self): return len(self.rows)

    def __getitem__(self, idx):
        try:
            img = self.tfm(Image.open(self.paths[idx]).convert('RGB'))
        except Exception:
            img = torch.zeros(3, self.image_size, self.image_size)
        target = int(self.rows[idx]['target'])
        c = torch.FloatTensor([1.0, 0.0] if target == 0 else [0.0, 1.0])
        return img, c


def save_ckpt(path, epoch, model, ema_model, optimizer, lr_sched, val_loss, history):
    tmp_path = path + '.tmp'
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'ema_state_dict': ema_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'lr_scheduler_state_dict': lr_sched.state_dict(),
        'val_loss': val_loss, 'history': history, 'image_size': IMAGE_SIZE,
        'num_concepts': NUM_CLASSES, 'class_names': CLASS_NAMES,
        'cfg_scale': CFG_SCALE, 'cfg_drop_prob': CFG_DROP_PROB,
        'model_type': 'baseline_class_conditioned',
    }, tmp_path)
    os.replace(tmp_path, path)


# ============================================================
# _mp_fn — runs once PER CORE (8 total)
# ============================================================

def _mp_fn(rank):
    device = xm.xla_device()
    world_size = xr.world_size()
    ordinal = xr.global_ordinal()
    is_master = xm.is_master_ordinal()

    def log(msg):
        if is_master:
            print(msg)

    log(f"[TPU-BASELINE] world_size={world_size} | starting on core {ordinal}")
    log(f"Conditioning: {NUM_CLASSES}-dim one-hot class label (NOT 7-dim concepts)")

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    if not os.path.exists(PSEUDO_CSV):
        raise FileNotFoundError(f"⚠ PSEUDO_CSV not found: {PSEUDO_CSV}")
    pseudo_df = pd.read_csv(PSEUDO_CSV)

    balanced_df = build_balanced(pseudo_df, seed=SEED)
    full_ds = ClassLabelDataset(balanced_df, ISIC_IMG_DIR, image_size=IMAGE_SIZE, augment=True)
    n_val = max(100, len(full_ds)//10)
    n_train = len(full_ds) - n_val
    tr_ds, vl_ds = torch.utils.data.random_split(
        full_ds, [n_train, n_val], generator=torch.Generator().manual_seed(SEED)
    )
    log(f"Train: {n_train} | Val: {n_val}")

    train_sampler = torch.utils.data.distributed.DistributedSampler(
        tr_ds, num_replicas=world_size, rank=ordinal, shuffle=True, seed=SEED)
    val_sampler = torch.utils.data.distributed.DistributedSampler(
        vl_ds, num_replicas=world_size, rank=ordinal, shuffle=False)

    tr_loader = DataLoader(tr_ds, batch_size=PER_CORE_BATCH, sampler=train_sampler,
                           num_workers=2, drop_last=True)
    vl_loader = DataLoader(vl_ds, batch_size=PER_CORE_BATCH, sampler=val_sampler, num_workers=2)

    tr_device_loader = pl.MpDeviceLoader(tr_loader, device)
    vl_device_loader = pl.MpDeviceLoader(vl_loader, device)

    model = ConceptConditionedUNet128(image_size=IMAGE_SIZE, num_concepts=NUM_CLASSES, base_channels=64).to(device)
    if is_master:
        log(f"Model params: {sum(p.numel() for p in model.parameters()):,}  "
            f"(should match NB3-TPU's main model — same architecture)")
    ema_model = deepcopy(model).eval()
    for p in ema_model.parameters(): p.requires_grad_(False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0.01)
    noise_sched = DDPMScheduler(num_train_timesteps=NUM_TIMESTEPS, beta_schedule='linear',
                                beta_start=0.0001, beta_end=0.02, clip_sample=True)

    def lr_fn(ep, warmup=10, total=150):
        if ep < warmup: return ep/max(warmup,1)
        p = (ep-warmup)/(total-warmup)
        return max(1e-6, 0.5*(1+np.cos(np.pi*p)))

    lr_sched = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda e: lr_fn(e, warmup=10, total=150))

    GEN_BEN_VEC = torch.FloatTensor(GEN_BEN_VEC_LIST)
    GEN_MEL_VEC = torch.FloatTensor(GEN_MEL_VEC_LIST)
    EVAL_MEL_VEC = torch.FloatTensor(EVAL_MEL_VEC_LIST)
    EVAL_BEN_VEC = torch.FloatTensor(EVAL_BEN_VEC_LIST)

    start_epoch = 0
    best_val = float('inf')
    history = {'train_loss':[], 'val_loss':[], 'ccfs_training':[], 'ccfs_mel':[],
               'ccfs_ben':[], 'epoch_times':[]}

    if RESUME_FROM is not None:
        if not os.path.exists(RESUME_FROM):
            raise FileNotFoundError(f"Checkpoint not found: {RESUME_FROM}")
        ck = torch.load(RESUME_FROM, map_location='cpu', weights_only=False)
        model.load_state_dict(ck['model_state_dict'])
        if 'ema_state_dict' in ck:
            ema_model.load_state_dict(ck['ema_state_dict'])
        else:
            ema_model.load_state_dict(ck['model_state_dict'])
        optimizer.load_state_dict(ck['optimizer_state_dict'])
        lr_sched.load_state_dict(ck['lr_scheduler_state_dict'])
        start_epoch = ck['epoch'] + 1
        best_val = ck.get('val_loss', float('inf'))
        history = ck.get('history', history)
        log(f"✓ Resumed | last epoch {ck['epoch']+1} | will train {start_epoch+1}→{SESSION_END}")
    else:
        log("Fresh start.")

    cbm_a = load_cbm_a(CBM_A_CHECKPOINT, device) if is_master else None
    if is_master:
        log("✓ CBM-A loaded (master only — used for monitoring)")

    log_file, log_writer = None, None
    if is_master:
        log_exists = os.path.exists(LOG_CSV)
        log_file = open(LOG_CSV, 'a', newline='')
        log_writer = csv.writer(log_file)
        if not log_exists:
            log_writer.writerow(['epoch','train_loss','val_loss','epoch_time_sec','lr','timestamp'])
            log_file.flush()

    def update_ema_(ema_model, model, decay=EMA_DECAY):
        with torch.no_grad():
            for ep_, sp in zip(ema_model.parameters(), model.parameters()):
                ep_.data.mul_(decay).add_(sp.data, alpha=1-decay)

    log(f"\n{'='*60}\n[TPU-8core BASELINE] TRAINING ep {start_epoch+1}→{SESSION_END}\n{'='*60}\n")

    ep_times = []
    for epoch in range(start_epoch, SESSION_END):
        t0 = time.time()

        model.train()
        tr_losses = []
        for batch_idx, (imgs, cvecs) in enumerate(tr_device_loader):
            B = imgs.shape[0]
            t = torch.randint(0, NUM_TIMESTEPS, (B,), device=device).long()
            noise = torch.randn(imgs.shape, device=device)

            cond_input = cvecs.clone()
            drop_mask = torch.rand(B, device=device) < CFG_DROP_PROB
            cond_input[drop_mask] = 0.0

            noisy = noise_sched.add_noise(imgs, noise, t)
            pred = model(noisy, t, cond_input)
            loss = F.mse_loss(pred, noise)

            (loss / GRAD_ACCUM).backward()
            if (batch_idx + 1) % GRAD_ACCUM == 0:
                xm.optimizer_step(optimizer)
                optimizer.zero_grad()
                update_ema_(ema_model, model)

            tr_losses.append(loss)
            xm.mark_step()

        tr_loss_local = torch.stack(tr_losses).mean()
        tr_loss_global = xm.mesh_reduce('tr_loss', tr_loss_local.item(), lambda xs: sum(xs)/len(xs))

        model.eval()
        vl_losses = []
        with torch.no_grad():
            for imgs, cvecs in vl_device_loader:
                B = imgs.shape[0]
                t = torch.randint(0, NUM_TIMESTEPS, (B,), device=device).long()
                noise = torch.randn(imgs.shape, device=device)
                noisy = noise_sched.add_noise(imgs, noise, t)
                pred = model(noisy, t, cvecs)
                loss = F.mse_loss(pred, noise)
                vl_losses.append(loss)
                xm.mark_step()
        vl_loss_local = torch.stack(vl_losses).mean()
        vl_loss_global = xm.mesh_reduce('vl_loss', vl_loss_local.item(), lambda xs: sum(xs)/len(xs))

        lr_sched.step()
        elapsed = time.time() - t0
        ep_times.append(elapsed)
        eta = ((SESSION_END-(epoch+1)) * np.mean(ep_times[-5:])) / 3600

        if is_master:
            history['train_loss'].append(tr_loss_global)
            history['val_loss'].append(vl_loss_global)
            history['epoch_times'].append(elapsed)
            log_writer.writerow([epoch+1, tr_loss_global, vl_loss_global, elapsed,
                                 optimizer.param_groups[0]['lr'], time.time()])
            log_file.flush()

            print(f"Ep {epoch+1:4d}/{SESSION_END} | tr={tr_loss_global:.4f} | vl={vl_loss_global:.4f} | "
                  f"{elapsed:.0f}s | ETA={eta:.1f}h", end='')

            if (epoch+1) % EVAL_EVERY == 0:
                mi = generate_images(ema_model, GEN_MEL_VEC, n=8, steps=30, device=device, sz=IMAGE_SIZE)
                bi = generate_images(ema_model, GEN_BEN_VEC, n=8, steps=30, device=device, sz=IMAGE_SIZE)
                mc = ccfs_feedback(mi, EVAL_MEL_VEC, cbm_a, device)
                bc = ccfs_feedback(bi, EVAL_BEN_VEC, cbm_a, device)
                mean_c = (mc+bc)/2
                history['ccfs_training'].append(mean_c)
                history['ccfs_mel'].append(mc); history['ccfs_ben'].append(bc)
                print(f" | CCFS={mean_c:.4f} [mel={mc:.3f} ben={bc:.3f}]", end='')
                grid = make_grid(torch.cat([mi[:4], bi[:4]]), nrow=4, normalize=False)
                save_image(grid, f'gen_baseline_tpu_ep{epoch+1:04d}.png')
                print(" [grid saved]")
            else:
                print()

            if (epoch+1) % SAVE_EVERY == 0:
                ckpt_name = f'checkpoint_baseline_tpu_epoch{epoch+1}.pth'
                save_ckpt(ckpt_name, epoch, model, ema_model, optimizer, lr_sched, vl_loss_global, history)
                print(f"  → {ckpt_name} saved")

            if vl_loss_global < best_val:
                best_val = vl_loss_global
                save_ckpt('best_clindiffxai_baseline_tpu.pth', epoch, model, ema_model, optimizer, lr_sched,
                          vl_loss_global, history)
                print(f"  ★ best model (vl={vl_loss_global:.4f})")

        xm.rendezvous(f'epoch_{epoch}_done')

    if is_master and log_file:
        log_file.close()
        print(f"\nSession complete. Best val_loss={best_val:.4f}")
        print(f"Download: checkpoint_baseline_tpu_epoch{SESSION_END}.pth + "
              f"best_clindiffxai_baseline_tpu.pth + {LOG_CSV}")
        if SESSION_END < 150:
            print(f"\nNext: SESSION_END={min(SESSION_END+40,150)}, "
                  f"RESUME_FROM='…/checkpoint_baseline_tpu_epoch{SESSION_END}.pth'")


if __name__ == '__main__':
    xmp.spawn(_mp_fn, args=())

    # ── Training curves (PNG + PDF) — runs ONCE in the parent process,
    # after all 8 spawned cores complete and exit. Reads the CSV log
    # rather than in-process history, since history lives inside the
    # now-terminated child processes and isn't accessible here. ──
    import pandas as pd

    def save_fig_both(fig, name, dpi=300):
        fig.savefig(f'{name}.png', dpi=dpi, bbox_inches='tight', facecolor='white')
        fig.savefig(f'{name}.pdf', bbox_inches='tight', facecolor='white')
        print(f"  Saved: {name}.png + {name}.pdf")

    if os.path.exists(LOG_CSV):
        log_df = pd.read_csv(LOG_CSV)
        fig, ax = plt.subplots(1, 1, figsize=(7, 4))
        fig.suptitle(f'BASELINE (TPU, class-conditioned, 128×128) — '
                     f'ep 1→{int(log_df["epoch"].max())}', fontweight='bold')
        ax.plot(log_df['epoch'], log_df['train_loss'], 'b-', lw=2, label='Train')
        ax.plot(log_df['epoch'], log_df['val_loss'], 'r-', lw=2, label='Val')
        for b in [40, 80, 120]:
            if b < log_df['epoch'].max():
                ax.axvline(x=b, color='gray', ls='--', alpha=0.5)
        ax.set_title('MSE Loss'); ax.set_xlabel('Epoch'); ax.legend(); ax.grid(alpha=0.3)
        plt.tight_layout()
        save_fig_both(fig, f'nb3b_tpu_baseline_upto_ep{int(log_df["epoch"].max())}_curves')
        plt.show()
    else:
        print(f"⚠ {LOG_CSV} not found — skipping curve plot.")
