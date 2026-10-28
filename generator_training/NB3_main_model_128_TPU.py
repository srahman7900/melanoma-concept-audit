# ============================================================
# NOTEBOOK 3-TPU — Main Model, 128×128, TPU v5e-8 (8-core multi-process)
#
# ⚠⚠⚠ READ BEFORE RUNNING ⚠⚠⚠
# This is a structurally different codebase from the GPU NB3 — multi-core
# TPU training requires spawning 8 separate processes (one per core), not
# a flat script. I have NO way to test torch_xla code in my development
# environment (no TPU hardware, no network access to even install torch).
# Every line below is written from documented API behavior, reasoned
# through carefully, but NOT verified by execution.
#
# MANDATORY SEQUENCE:
#   1. Run TPU_Smoke_Test_8core.py FIRST. Confirm all 8 cores participate
#      (all_reduce_sum == 28). Do not proceed if it fails.
#   2. Run THIS script with SESSION_END=3 (just 3 epochs) as a verification
#      run. Compare the loss trajectory shape to your GPU NB3 run at the
#      same epoch count. Loss should decrease at a SIMILAR rate. If it
#      explodes (NaN/huge) or completely flatlines, the gradient
#      all-reduce scaling is very likely wrong — STOP and report back
#      with the printed losses before committing to a real session.
#   3. Only after step 2 looks sane, commit to SESSION_END=40 etc.
#
# KNOWN SPECIFIC RISK — gradient reduction scaling:
#   xm.optimizer_step(optimizer) all-reduces gradients across the 8 cores
#   before stepping. Different torch_xla versions have handled this as
#   either SUM or MEAN across replicas. This code assumes the standard
#   documented default (no manual rescaling). If your verification run
#   in step 2 shows loss exploding, try dividing the loss by 8 before
#   .backward() as the likely fix (search "xm.optimizer_step reduce_type
#   torch_xla <your version>" for your exact installed version's behavior).
#
# EFFECTIVE BATCH SIZE — set to match the GPU runs for a fair comparison:
#   GPU NB3 used micro_batch=8, accum=4 → effective batch = 32
#   Here: 8 cores × per-core batch=4 × accum=1 → effective batch = 32 (same)
# ============================================================

import os, time, csv, warnings

# ⚠ KAGGLE TPU PLATFORM BUG FIX — must run before any torch_xla import/use.
# See https://www.kaggle.com/discussions/product-feedback/473974 — confirmed
# Kaggle platform issue, not specific to this script.
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
# CONFIG — module-level globals, visible to _mp_fn via closure (fork)
# ============================================================

PSEUDO_CSV       = "/kaggle/input/datasets/shohanurrahman7/pseudo-labels-v3/isic2020_pseudo_labels_v3.csv"
ISIC_IMG_DIR     = "/kaggle/input/datasets/sumaiyabinteshahid/isic-challenge-dataset-2020/ISIC_2020_Dataset/train"
CBM_A_CHECKPOINT = "/kaggle/input/datasets/shohanurrahman7/cbma-v3/best_cbm_model.pth"

SESSION_END  = 150   # ⚠ ADJUST to match your actual remaining TPU quota.
                      # At ~165s/epoch steady-state: 100 epochs (50→150) ≈ 4.6h.
                      # If your quota is less, lower this — e.g. SESSION_END=110
                      # for ~60 epochs (50→110) ≈ 2.75h. You can always resume
                      # a later session from wherever this one stops.
RESUME_FROM  = "/kaggle/input/datasets/REPLACE_WITH_YOUR_PATH/checkpoint_epoch50.pth"
                      # ⚠ UPDATE to your actual GPU checkpoint_epoch50.pth path,
                      # uploaded as a Kaggle dataset on this account. This is the
                      # SAME cross-platform resume pattern already validated twice
                      # on the baseline (GPU checkpoint → TPU multi-core training).

IMAGE_SIZE        = 128
PER_CORE_BATCH    = 4      # ⚠ NOT the same meaning as GPU's BATCH_SIZE — this is PER CORE
GRAD_ACCUM        = 1      # 8 cores × 4 × 1 = effective batch 32 (matches GPU runs)
NUM_TIMESTEPS     = 1000
SAVE_EVERY        = 10
EVAL_EVERY        = 10     # ⚠ CHANGED from 40 — CCFS now tracked every 10 epochs
SEED              = 42

CFG_DROP_PROB = 0.15
CFG_SCALE     = 3.5
EMA_DECAY     = 0.999

LOG_CSV = 'nb3_tpu_train_log.csv'

CONCEPT_NAMES = ['pigment_network', 'blue_whitish_veil', 'vascular_structures',
                  'pigmentation', 'streaks', 'dots_and_globules', 'regression_structures']
NUM_CONCEPTS = len(CONCEPT_NAMES)

MEL_VEC_LIST = [0.8292, 0.7781, 0.9926, 0.8918, 0.4585, 0.9256, 0.8769]
BEN_VEC_LIST = [0.0800, 0.0002, 0.0047, 0.0055, 0.0018, 0.1501, 0.0003]


# ============================================================
# ARCHITECTURE — byte-identical to NB3/NB3b (verified shape-trace there).
# Copy-pasted verbatim, not re-derived, to guarantee zero drift.
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
# CBM-A loader, CCFS, generation — identical logic to NB3, device-generic
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
    norm = transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    rsz  = transforms.Resize((224,224))
    inp  = torch.stack([norm(rsz(i)) for i in imgs]).to(device)
    with torch.no_grad():
        _, det, _ = cbm(inp)
    det  = det.cpu()
    cond = cvec.unsqueeze(0).repeat(len(imgs), 1)
    return F.cosine_similarity(cond, det, dim=1).mean().item()


@torch.no_grad()
def generate_images(model, cvec, n, steps, device, sz, cfg_scale=CFG_SCALE):
    model.eval()
    ddim = DDIMScheduler(num_train_timesteps=1000, beta_schedule='linear', clip_sample=True)
    ddim.set_timesteps(steps)
    imgs = torch.randn(n,3,sz,sz, device=device)
    cv = cvec.unsqueeze(0).repeat(n,1).to(device)
    for t in ddim.timesteps:
        ts = torch.full((n,), t, device=device, dtype=torch.long)
        eps_cond   = model(imgs, ts, cv)
        eps_uncond = model(imgs, ts, torch.zeros_like(cv))
        noise_pred = eps_uncond + cfg_scale*(eps_cond - eps_uncond)
        imgs = ddim.step(noise_pred, t, imgs).prev_sample
        xm.mark_step()   # bound the lazy graph during the 30-step sampling loop too
    return ((imgs.clamp(-1,1)+1)/2).cpu()


def save_sample_grid_raw(mi, bi, epoch, prefix='gen_tpu'):
    """Lightweight, torchvision-only sample save — proven pattern, matches
    what ran successfully on the baseline across 6 TPU sessions. Deliberately
    avoids matplotlib here: matplotlib figure creation inside the live
    8-core spawned process is untested on TPU and a plausible trigger for
    the runtime_metric_aggregator crash hit on the first real run of this
    script. Labeled, publication-quality figures are now generated
    separately, AFTER training, from these saved PNGs — see the post-spawn
    section at the bottom of this file."""
    grid = make_grid(torch.cat([mi[:4], bi[:4]]), nrow=4, normalize=False)
    name = f'{prefix}_ep{epoch:04d}'
    save_image(grid, f'{name}.png')
    return name


# ============================================================
# DATA — identical balancing/dataset logic to NB3
# ============================================================

def build_balanced(df, seed=42):
    mel = df[df['target'] == 1].copy()
    ben = df[df['target'] == 0].copy()
    mel5 = pd.concat([mel]*5, ignore_index=True)
    ben3 = ben.sample(n=len(mel5)*3, random_state=seed)
    out = pd.concat([mel5, ben3], ignore_index=True).sample(frac=1, random_state=seed)
    return out


class ConceptDataset(Dataset):
    def __init__(self, df, image_dir, concept_cols, image_size=128, augment=True):
        self.concept_cols = concept_cols
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
        c = torch.FloatTensor(
            [float(self.rows[idx].get(col, 0.)) for col in self.concept_cols]
        ).clamp(0., 1.)
        return img, c


# ============================================================
# save_ckpt — master-only writer (avoid 8 processes racing on one file)
# ============================================================

def save_ckpt(path, epoch, model, ema_model, optimizer, lr_sched, val_loss, history):
    tmp_path = path + '.tmp'
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'ema_state_dict': ema_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'lr_scheduler_state_dict': lr_sched.state_dict(),
        'val_loss': val_loss, 'history': history, 'image_size': IMAGE_SIZE,
        'num_concepts': NUM_CONCEPTS, 'concept_names': CONCEPT_NAMES,
        'cfg_scale': CFG_SCALE, 'cfg_drop_prob': CFG_DROP_PROB,
    }, tmp_path)
    os.replace(tmp_path, path)


# ============================================================
# _mp_fn — runs once PER CORE (8 total). All setup happens here, not at
# module scope, except the lightweight config/architecture defs above.
# ============================================================

def _mp_fn(rank):
    device = xm.xla_device()
    world_size = xr.world_size()
    ordinal = xr.global_ordinal()
    is_master = xm.is_master_ordinal()

    def log(msg):
        if is_master:
            print(msg)

    log(f"[TPU] world_size={world_size} | starting on core {ordinal}")

    # Same seed on every core → identical initial model weights across
    # replicas. This invariant is preserved at every step because
    # xm.optimizer_step all-reduces gradients BEFORE applying them, so
    # if all replicas agree before a step, they agree after it too.
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    if not os.path.exists(PSEUDO_CSV):
        raise FileNotFoundError(f"⚠ PSEUDO_CSV not found: {PSEUDO_CSV}")
    pseudo_df = pd.read_csv(PSEUDO_CSV)
    concept_cols = [f'concept_{n}' for n in CONCEPT_NAMES]

    balanced_df = build_balanced(pseudo_df, seed=SEED)   # identical across cores (same seed)
    full_ds = ConceptDataset(balanced_df, ISIC_IMG_DIR, concept_cols, image_size=IMAGE_SIZE, augment=True)
    n_val = max(100, len(full_ds)//10)
    n_train = len(full_ds) - n_val
    tr_ds, vl_ds = torch.utils.data.random_split(
        full_ds, [n_train, n_val], generator=torch.Generator().manual_seed(SEED)
    )
    log(f"Train: {n_train} | Val: {n_val} (full corpus — each core sees its own shard)")

    # DistributedSampler: each of the 8 cores gets a DIFFERENT 1/8 shard
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        tr_ds, num_replicas=world_size, rank=ordinal, shuffle=True, seed=SEED)
    val_sampler = torch.utils.data.distributed.DistributedSampler(
        vl_ds, num_replicas=world_size, rank=ordinal, shuffle=False)

    tr_loader = DataLoader(tr_ds, batch_size=PER_CORE_BATCH, sampler=train_sampler,
                           num_workers=2, drop_last=True)
    vl_loader = DataLoader(vl_ds, batch_size=PER_CORE_BATCH, sampler=val_sampler, num_workers=2)

    tr_device_loader = pl.MpDeviceLoader(tr_loader, device)
    vl_device_loader = pl.MpDeviceLoader(vl_loader, device)

    model = ConceptConditionedUNet128(image_size=IMAGE_SIZE, num_concepts=NUM_CONCEPTS, base_channels=64).to(device)
    if is_master:
        log(f"Model params: {sum(p.numel() for p in model.parameters()):,}")
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

    MEL_VEC = torch.FloatTensor(MEL_VEC_LIST)
    BEN_VEC = torch.FloatTensor(BEN_VEC_LIST)

    start_epoch = 0
    best_val = float('inf')
    history = {'train_loss':[], 'val_loss':[], 'ccfs_training':[], 'ccfs_mel':[],
               'ccfs_ben':[], 'epoch_times':[]}

    if RESUME_FROM is not None:
        if not os.path.exists(RESUME_FROM):
            raise FileNotFoundError(f"Checkpoint not found: {RESUME_FROM}")
        # Every core loads the SAME file independently — no broadcast
        # needed since they all read identical bytes from disk.
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
            log_writer.writerow(['epoch','train_loss','val_loss','ccfs_mean','ccfs_mel','ccfs_ben',
                                 'epoch_time_sec','lr','timestamp'])
            log_file.flush()

    def update_ema_(ema_model, model, decay=EMA_DECAY):
        with torch.no_grad():
            for ep_, sp in zip(ema_model.parameters(), model.parameters()):
                ep_.data.mul_(decay).add_(sp.data, alpha=1-decay)

    log(f"\n{'='*60}\n[TPU-8core] TRAINING ep {start_epoch+1}→{SESSION_END}\n{'='*60}\n")

    ep_times = []
    for epoch in range(start_epoch, SESSION_END):
        t0 = time.time()

        # ── TRAIN ──
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
                xm.optimizer_step(optimizer)   # all-reduces grads across 8 cores, then steps
                optimizer.zero_grad()
                update_ema_(ema_model, model)

            tr_losses.append(loss)   # keep as tensor — avoid .item() every micro-step
            xm.mark_step()           # bound the lazy graph every iteration

        tr_loss_local = torch.stack(tr_losses).mean()
        tr_loss_global = xm.mesh_reduce('tr_loss', tr_loss_local.item(), lambda xs: sum(xs)/len(xs))

        # ── VALIDATE ──
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

            # ── Eval computed BEFORE the CSV write, so CCFS lands in the same
            # row as this epoch's loss — survives across sessions via the CSV,
            # not just the in-memory history dict.
            # ⚠ Also forces an eval on the SESSION'S FINAL epoch even if it
            # isn't a multiple of EVAL_EVERY — this closes the exact gap hit
            # on the baseline run, where epoch 150 (not a multiple of 40) had
            # no CCFS reading at all. ──
            is_eval_epoch = ((epoch+1) % EVAL_EVERY == 0) or ((epoch+1) == SESSION_END)
            mc_str = bc_str = mean_c_str = ''
            ccfs_suffix = ''

            if is_eval_epoch:
                mi = generate_images(ema_model, MEL_VEC, n=8, steps=30, device=device, sz=IMAGE_SIZE)
                bi = generate_images(ema_model, BEN_VEC, n=8, steps=30, device=device, sz=IMAGE_SIZE)
                mc = ccfs_feedback(mi, MEL_VEC, cbm_a, device)
                bc = ccfs_feedback(bi, BEN_VEC, cbm_a, device)
                mean_c = (mc+bc)/2
                history['ccfs_training'].append(mean_c)
                history['ccfs_mel'].append(mc); history['ccfs_ben'].append(bc)
                mc_str, bc_str, mean_c_str = f'{mc:.4f}', f'{bc:.4f}', f'{mean_c:.4f}'
                ccfs_suffix = f" | CCFS={mean_c:.4f} [mel={mc:.3f} ben={bc:.3f}]"
                fig_name = save_sample_grid_raw(mi, bi, epoch+1)
                ccfs_suffix += f" [{fig_name}.png saved]"

            log_writer.writerow([epoch+1, tr_loss_global, vl_loss_global, mean_c_str, mc_str, bc_str,
                                 elapsed, optimizer.param_groups[0]['lr'], time.time()])
            log_file.flush()

            print(f"Ep {epoch+1:4d}/{SESSION_END} | tr={tr_loss_global:.4f} | vl={vl_loss_global:.4f} | "
                  f"{elapsed:.0f}s | ETA={eta:.1f}h{ccfs_suffix}")

            if (epoch+1) % SAVE_EVERY == 0:
                ckpt_name = f'checkpoint_tpu_epoch{epoch+1}.pth'
                save_ckpt(ckpt_name, epoch, model, ema_model, optimizer, lr_sched, vl_loss_global, history)
                print(f"  → {ckpt_name} saved")

            if vl_loss_global < best_val:
                best_val = vl_loss_global
                save_ckpt('best_clindiffxai_v3_tpu.pth', epoch, model, ema_model, optimizer, lr_sched,
                          vl_loss_global, history)
                print(f"  ★ best model (vl={vl_loss_global:.4f})")

        # barrier: keep all 8 cores in lockstep at epoch boundaries, so a
        # slow core (or master's extra I/O work above) doesn't let other
        # cores race ahead into the next epoch's data loading
        xm.rendezvous(f'epoch_{epoch}_done')

    if is_master and log_file:
        log_file.close()
        print(f"\nSession complete. Best val_loss={best_val:.4f}")
        print(f"Download: checkpoint_tpu_epoch{SESSION_END}.pth + best_clindiffxai_v3_tpu.pth + {LOG_CSV}")
        if SESSION_END < 150:
            print(f"\nNext: SESSION_END={min(SESSION_END+40,150)}, "
                  f"RESUME_FROM='…/checkpoint_tpu_epoch{SESSION_END}.pth'")


if __name__ == '__main__':
    xmp.spawn(_mp_fn, args=())

    # ── Paper figures — run ONCE in the parent process, after all 8 spawned
    # cores complete and exit. Read from the CSV log (the only thing that
    # survives the child processes terminating). ──
    import pandas as pd

    def save_fig_both(fig, name, dpi=300):
        fig.savefig(f'{name}.png', dpi=dpi, bbox_inches='tight', facecolor='white')
        fig.savefig(f'{name}.pdf', bbox_inches='tight', facecolor='white')
        print(f"  Saved: {name}.png + {name}.pdf")

    if os.path.exists(LOG_CSV):
        log_df = pd.read_csv(LOG_CSV)
        max_ep = int(log_df['epoch'].max())

        # ── Figure 1: Loss curves ──
        fig, ax = plt.subplots(1, 1, figsize=(7, 4))
        fig.suptitle(f'ClinDiffXAI v3 (TPU, 128×128) — ep 1→{max_ep}', fontweight='bold')
        ax.plot(log_df['epoch'], log_df['train_loss'], 'b-', lw=2, label='Train')
        ax.plot(log_df['epoch'], log_df['val_loss'], 'r-', lw=2, label='Val')
        for b in [40, 80, 120]:
            if b < max_ep:
                ax.axvline(x=b, color='gray', ls='--', alpha=0.5)
        ax.set_title('MSE Loss'); ax.set_xlabel('Epoch'); ax.legend(); ax.grid(alpha=0.3)
        plt.tight_layout()
        save_fig_both(fig, f'nb3_tpu_main_upto_ep{max_ep}_loss_curves')
        plt.show()

        # ── Figure 2: CCFS trajectory (every 10 epochs, now with full history) ──
        ccfs_df = log_df.dropna(subset=['ccfs_mean'])
        if len(ccfs_df) > 0:
            fig, ax = plt.subplots(1, 1, figsize=(8, 4.5))
            fig.suptitle(f'ClinDiffXAI v3 (TPU) — CCFS Trajectory, ep 1→{max_ep}', fontweight='bold')
            ax.plot(ccfs_df['epoch'], ccfs_df['ccfs_mean'], 'g-o', lw=2, ms=6, label='Mean CCFS')
            ax.plot(ccfs_df['epoch'], ccfs_df['ccfs_mel'], 'r--s', lw=1.5, ms=5, label='Melanoma CCFS')
            ax.plot(ccfs_df['epoch'], ccfs_df['ccfs_ben'], 'b--^', lw=1.5, ms=5, label='Benign CCFS')
            ax.axhline(0.70, color='gray', ls=':', alpha=0.7, label='0.70 threshold')
            ax.set_title('Concept-Conditioning Fidelity Score (EMA model)')
            ax.set_xlabel('Epoch'); ax.set_ylabel('CCFS')
            ax.legend(fontsize=9); ax.grid(alpha=0.3)
            plt.tight_layout()
            save_fig_both(fig, f'nb3_tpu_main_upto_ep{max_ep}_ccfs_trajectory')
            plt.show()
        else:
            print("⚠ No CCFS readings found in log — skipping CCFS trajectory plot.")

        # ── Figure 3: Conditioning vectors reference chart (static — doesn't
        # depend on training progress, but useful as a methods-section figure) ──
        short = ['PN', 'BWV', 'VS', 'Pig', 'Str', 'D/G', 'Reg']
        x = np.arange(NUM_CONCEPTS)
        fig, ax = plt.subplots(1, 1, figsize=(7, 4))
        ax.bar(x-0.2, MEL_VEC_LIST, 0.35, label='Melanoma', color='#d62728', alpha=0.8)
        ax.bar(x+0.2, BEN_VEC_LIST, 0.35, label='Benign', color='#1f77b4', alpha=0.8)
        ax.set_xticks(x); ax.set_xticklabels(short, fontsize=10)
        ax.set_title('Conditioning Vectors (image-verified, NB2/NB2b)', fontweight='bold')
        ax.set_ylim(0, 1.1); ax.legend(); ax.grid(alpha=0.3, axis='y')
        plt.tight_layout()
        save_fig_both(fig, 'nb3_tpu_main_conditioning_vectors')
        plt.show()

        print(f"\nAll figures saved to /kaggle/working/. Raw sample grids per eval epoch: "
              f"gen_tpu_ep*.png (every {EVAL_EVERY} epochs + final epoch of this session).")
    else:
        print(f"⚠ {LOG_CSV} not found — skipping figure generation.")

    # ── Labeled paper-quality versions of the sample grids — generated HERE,
    # after training is fully done and the TPU process pool has exited, not
    # during live training. The raw grid (torchvision, proven-safe pattern)
    # was already saved during training; this step only adds text labels
    # and a PDF export, with no XLA/TPU interaction at all. ──
    import glob
    raw_grids = sorted(glob.glob('gen_tpu_ep*.png'))
    if raw_grids:
        print(f"\nLabeling {len(raw_grids)} sample grid(s) for paper use...")
        for path in raw_grids:
            epoch_str = path.replace('gen_tpu_ep', '').replace('.png', '')
            img = plt.imread(path)
            fig, ax = plt.subplots(figsize=(9.5, 5.2))
            ax.imshow(img)
            ax.axis('off')
            fig.suptitle(f'ClinDiffXAI v3 — Generated Samples, Epoch {int(epoch_str)}',
                         fontweight='bold', fontsize=12, y=0.98)
            h = img.shape[0]
            ax.text(-0.04, 0.75, 'Melanoma-\nconditioned', rotation=90, fontsize=10,
                    fontweight='bold', ha='center', va='center', transform=ax.transAxes)
            ax.text(-0.04, 0.25, 'Benign-\nconditioned', rotation=90, fontsize=10,
                    fontweight='bold', ha='center', va='center', transform=ax.transAxes)
            plt.tight_layout()
            labeled_name = f'gen_tpu_ep{epoch_str}_labeled'
            fig.savefig(f'{labeled_name}.png', dpi=200, bbox_inches='tight', facecolor='white')
            fig.savefig(f'{labeled_name}.pdf', bbox_inches='tight', facecolor='white')
            plt.close(fig)
        print(f"  Labeled PNG+PDF versions saved alongside the raw grids.")
