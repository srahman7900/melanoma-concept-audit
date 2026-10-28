# ============================================================
# NOTEBOOK 3 v3 — ClinDiffXAI MAIN MODEL, 128×128
# Single-cell version. Simple local Kaggle save (no Drive).
# Platform: Kaggle T4×2 GPU
#
# ⚠ ARCHITECTURE FIX vs v1/v2: tracing v2's exact decoder skip-pairing
# revealed an off-by-one bug — the finest skip connection was NEVER used,
# the first learned upsample was immediately discarded by a forced
# interpolate, and the final 32→64 step was a naive non-learned upsample.
# v3 uses corrected skip indexing (decoder consumes sk[:-1] in reverse,
# dropping only the bottleneck-resolution skip, which is redundant with
# mid's own output). Verified via shape-trace: all 4 decoder stages match
# cleanly, no fallback interpolate ever fires, finest 128×128 skip is used.
#
# ⚠ BEFORE RUNNING — VERIFY:
#   1. PSEUDO_CSV — path to isic2020_pseudo_labels_v3.csv from NB2
#      (upload as a Kaggle dataset, update path below)
#   2. ISIC_IMG_DIR — same as NB2
#   3. CBM_A_CHECKPOINT — best_cbm_model.pth from NB1 (for CCFS eval)
#
# MEMORY: 128×128 needs ~4x the activation memory of 64×64. Using
# BATCH_SIZE=8 with GRAD_ACCUM_STEPS=4 (effective batch=32, same as v2)
# to fit in T4×2's 15.6GB each.
#
# TO SWITCH SESSIONS: change SESSION_END / RESUME_FROM below.
#   Session 1: SESSION_END=40,  RESUME_FROM=None
#   Session 2: SESSION_END=80,  RESUME_FROM=".../checkpoint_epoch40.pth"
#   Session 3: SESSION_END=120, RESUME_FROM=".../checkpoint_epoch80.pth"
#   Session 4: SESSION_END=150, RESUME_FROM=".../checkpoint_epoch120.pth"
# ============================================================

# ── INSTALL ────────────────────────────────────────────────────
import subprocess, sys
for pkg in ["diffusers==0.33.0", "accelerate==1.6.0", "timm==1.0.15"]:
    subprocess.run([sys.executable, '-m', 'pip', 'install', pkg, '-q'], check=False)

import os, time, csv, warnings
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

device = torch.device('cuda')
torch.backends.cudnn.benchmark = True
print(f"GPU: {torch.cuda.get_device_name(0)} | Count: {torch.cuda.device_count()}")


# ============================================================
# CONFIG — verify paths, edit session plan
# ============================================================

# ⚠ VERIFY these paths
PSEUDO_CSV       = "/kaggle/input/datasets/shohanurrahman7/pseudo-labels-v3/isic2020_pseudo_labels_v3.csv"
ISIC_IMG_DIR     = "/kaggle/input/datasets/sumaiyabinteshahid/isic-challenge-dataset-2020/ISIC_2020_Dataset/train"
CBM_A_CHECKPOINT = "/kaggle/input/datasets/shohanurrahman7/cbma-v3/best_cbm_model.pth"

SESSION_END  = 40
RESUME_FROM  = None

IMAGE_SIZE     = 128
BATCH_SIZE     = 8          # micro-batch — memory constrained at 128x128
GRAD_ACCUM     = 4          # effective batch = 8*4 = 32 (matches v2)
NUM_TIMESTEPS  = 1000
NUM_WORKERS    = 6
SAVE_EVERY     = 10
EVAL_EVERY     = 40
SEED           = 42

CFG_DROP_PROB = 0.15
CFG_SCALE     = 3.5
EMA_DECAY     = 0.999

LOG_CSV = 'nb3_train_log.csv'   # per-epoch log, survives session boundaries

CONCEPT_NAMES = ['pigment_network', 'blue_whitish_veil', 'vascular_structures',
                  'pigmentation', 'streaks', 'dots_and_globules', 'regression_structures']
NUM_CONCEPTS = len(CONCEPT_NAMES)

# Finalized conditioning vectors — derived from real CBM-A v3 predictions on
# ISIC 2020 (NB2), verified against actual images (NB2b). p90/p10 prototype
# rule for 6 concepts; blue_whitish_veil uses p98 (its positive signal is
# real but confined to ~2% of melanoma images — p90 would fall inside the
# "truly absent" bulk and undersell genuine, image-verified signal).
MEL_VEC = torch.FloatTensor([0.8292, 0.7781, 0.9926, 0.8918, 0.4585, 0.9256, 0.8769])
BEN_VEC = torch.FloatTensor([0.0800, 0.0002, 0.0047, 0.0055, 0.0018, 0.1501, 0.0003])

torch.manual_seed(SEED)
np.random.seed(SEED)

print(f"\nSession target: epochs {1 if RESUME_FROM is None else '?'} → {SESSION_END}")
print(f"Resume from:   {RESUME_FROM or 'None (fresh start)'}")
print(f"Config: IMAGE_SIZE={IMAGE_SIZE} | batch={BATCH_SIZE}×{GRAD_ACCUM}accum=eff.{BATCH_SIZE*GRAD_ACCUM} | "
      f"CFG(p={CFG_DROP_PROB},s={CFG_SCALE}) | EMA(d={EMA_DECAY})")


# ============================================================
# DATA — balanced dataset from NB2's per-image pseudo-labels
# ============================================================

if not os.path.exists(PSEUDO_CSV):
    raise FileNotFoundError(f"⚠ PSEUDO_CSV not found: {PSEUDO_CSV}")

pseudo_df = pd.read_csv(PSEUDO_CSV)
concept_cols = [f'concept_{n}' for n in CONCEPT_NAMES]
print(f"\nLoaded {len(pseudo_df)} pseudo-labeled rows")

found = sum(1 for n in pseudo_df['image_name'].head(10)
            if os.path.exists(os.path.join(ISIC_IMG_DIR, f"{n}.jpg")))
print(f"Data check: {found}/10 images found at {ISIC_IMG_DIR}")
if found == 0:
    raise FileNotFoundError(f"⚠ Images not found at {ISIC_IMG_DIR}")


def build_balanced(df, seed=42):
    mel = df[df['target'] == 1].copy()
    ben = df[df['target'] == 0].copy()
    mel5 = pd.concat([mel]*5, ignore_index=True)
    ben3 = ben.sample(n=len(mel5)*3, random_state=seed)
    out = pd.concat([mel5, ben3], ignore_index=True).sample(frac=1, random_state=seed)
    print(f"Balanced dataset: {len(mel5)} melanoma + {len(ben3)} benign = {len(out)} total")
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
        print(f"Dataset: {len(self.rows)}/{len(df)} valid images")

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


balanced_df = build_balanced(pseudo_df, seed=SEED)
full_ds = ConceptDataset(balanced_df, ISIC_IMG_DIR, concept_cols, image_size=IMAGE_SIZE, augment=True)
n_val = max(100, len(full_ds)//10)
n_train = len(full_ds) - n_val
tr_ds, vl_ds = torch.utils.data.random_split(
    full_ds, [n_train, n_val], generator=torch.Generator().manual_seed(SEED)
)
tr_loader = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True,
                       num_workers=NUM_WORKERS, pin_memory=True, drop_last=True,
                       persistent_workers=True, prefetch_factor=4)
vl_loader = DataLoader(vl_ds, batch_size=BATCH_SIZE, shuffle=False,
                       num_workers=NUM_WORKERS, persistent_workers=True, prefetch_factor=4)
print(f"Train: {n_train} | Val: {n_val}")

t0 = time.time()
_ = next(iter(tr_loader))
batch_time = time.time() - t0
bpe = n_train // BATCH_SIZE
est_min = batch_time * bpe / 60
print(f"~{est_min:.1f} min/epoch estimated (live, from your first batch)")
if RESUME_FROM is None:
    print(f"Session 1 ({SESSION_END} epochs): ~{SESSION_END*est_min/60:.1f}h estimated")


# ============================================================
# ARCHITECTURE — 128×128, corrected skip indexing (see header)
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
    """5 encoder levels (128,64,32,16,8), 4 real downsamples. Attention only
    at 32/16/8 (full attention at 128/64 would be computationally absurd —
    a 128x128 attention map is 16,384^2 elements per head). Decoder skip
    indexing verified correct via shape-trace (see header comment)."""
    def __init__(self, image_size=128, num_concepts=7, base_channels=64):
        super().__init__()
        self.image_size = image_size
        C = [64, 64, 128, 256, 256]
        td = base_channels*4; cd = base_channels*2
        self.time_mlp = nn.Sequential(nn.Linear(base_channels,td),nn.SiLU(),nn.Linear(td,td))
        self.concept_emb = ConceptEmbedding(num_concepts, cd)
        self.enc_in = nn.Conv2d(3, C[0], 3, padding=1)

        self.enc = nn.ModuleList([
            nn.ModuleList([ResBlock(C[0],C[0],td,cd), ResBlock(C[0],C[0],td,cd)]),                        # L0 128x128 no-attn
            nn.ModuleList([ResBlock(C[0],C[1],td,cd), ResBlock(C[1],C[1],td,cd)]),                        # L1 64x64  no-attn
            nn.ModuleList([ResBlock(C[1],C[2],td,cd), ResBlock(C[2],C[2],td,cd), SelfAttention(C[2])]),   # L2 32x32  attn
            nn.ModuleList([ResBlock(C[2],C[3],td,cd), ResBlock(C[3],C[3],td,cd), SelfAttention(C[3])]),   # L3 16x16  attn
            nn.ModuleList([ResBlock(C[3],C[4],td,cd), ResBlock(C[4],C[4],td,cd), SelfAttention(C[4])]),   # L4 8x8    attn
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

        usable_sk = sk[:-1]   # drop sk[-1] — same resolution as bottleneck, redundant
        for i, (blks, up) in enumerate(zip(self.dec, self.up)):
            h = up(h)
            s = usable_sk[-(i+1)]
            if h.shape != s.shape:   # defensive only — verified this never fires
                h = F.interpolate(h, size=s.shape[-2:])
            h = torch.cat([h, s], dim=1)
            for b in blks: h = b(h) if isinstance(b,SelfAttention) else b(h,te,ce)

        o = self.out(h)
        if o.shape[-1] != self.image_size:   # defensive only
            o = F.interpolate(o, size=(self.image_size, self.image_size))
        return o


# ============================================================
# CBM-A loader (for CCFS evaluation — matches NB1 exactly)
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
    print(f"✓ CBM-A loaded (epoch {ck.get('epoch','?')})")
    return m.to(device).eval()


def ccfs_feedback(imgs, cvec, cbm, device):
    norm = transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    rsz  = transforms.Resize((224,224))   # CBM-A always expects 224, regardless of diffusion resolution
    inp  = torch.stack([norm(rsz(i)) for i in imgs]).to(device)
    with torch.no_grad():
        _, det, _ = cbm(inp)
    det  = det.cpu()
    cond = cvec.unsqueeze(0).repeat(len(imgs), 1)
    return F.cosine_similarity(cond, det, dim=1).mean().item()


# ============================================================
# GENERATION — CFG DDIM sampling
# ============================================================

@torch.no_grad()
def generate_images(model, cvec, n=8, steps=30, device='cuda', sz=128, cfg_scale=CFG_SCALE):
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
    return ((imgs.clamp(-1,1)+1)/2).cpu()


# ============================================================
# MODEL + EMA INIT
# ============================================================

model = ConceptConditionedUNet128(image_size=IMAGE_SIZE, num_concepts=NUM_CONCEPTS, base_channels=64).to(device)
print(f"\nModel params: {sum(p.numel() for p in model.parameters()):,}")

if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs via DataParallel")
    model = nn.DataParallel(model)

_base = model.module if hasattr(model,'module') else model
ema_model = deepcopy(_base).to(device).eval()
for p in ema_model.parameters(): p.requires_grad_(False)
print(f"EMA model initialised (decay={EMA_DECAY})")

noise_sched = DDPMScheduler(num_train_timesteps=NUM_TIMESTEPS, beta_schedule='linear',
                            beta_start=0.0001, beta_end=0.02, clip_sample=True)
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0.01)

def lr_fn(ep, warmup=10, total=150):
    if ep < warmup: return ep/max(warmup,1)
    p = (ep-warmup)/(total-warmup)
    return max(1e-6, 0.5*(1+np.cos(np.pi*p)))

lr_sched = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda e: lr_fn(e, warmup=10, total=150))


# ============================================================
# RESUME LOGIC
# ============================================================

start_epoch = 0
best_val = float('inf')
history = {'train_loss':[], 'val_loss':[], 'ccfs_training':[], 'ccfs_mel':[],
           'ccfs_ben':[], 'epoch_times':[]}

if RESUME_FROM is not None:
    if not os.path.exists(RESUME_FROM):
        raise FileNotFoundError(f"Checkpoint not found: {RESUME_FROM}")
    ck = torch.load(RESUME_FROM, map_location='cpu', weights_only=False)
    raw_model = model.module if hasattr(model,'module') else model
    raw_model.load_state_dict(ck['model_state_dict'])
    if 'ema_state_dict' in ck:
        ema_model.load_state_dict(ck['ema_state_dict'])
        print("  ✓ EMA state restored")
    else:
        ema_model.load_state_dict(raw_model.state_dict())
        print("  ⚠ No EMA in checkpoint — initialised from training weights")
    optimizer.load_state_dict(ck['optimizer_state_dict'])
    lr_sched.load_state_dict(ck['lr_scheduler_state_dict'])
    start_epoch = ck['epoch'] + 1
    best_val = ck.get('val_loss', float('inf'))
    history = ck.get('history', history)
    for k in ['ccfs_mel','ccfs_ben','epoch_times']:
        if k not in history: history[k] = []
    print(f"✓ Resumed | last epoch {ck['epoch']+1} | will train {start_epoch+1}→{SESSION_END} | "
          f"best_val={best_val:.4f}")
else:
    print("Fresh start.")

cbm_a = load_cbm_a(CBM_A_CHECKPOINT, device)

# ── per-epoch CSV logger, survives session boundaries ──
log_exists = os.path.exists(LOG_CSV)
log_file = open(LOG_CSV, 'a', newline='')
log_writer = csv.writer(log_file)
if not log_exists:
    log_writer.writerow(['epoch','train_loss','val_loss','epoch_time_sec','lr','timestamp'])
    log_file.flush()


# ============================================================
# TRAINING LOOP — gradient accumulation + EMA + CSV logging
# ============================================================

@torch.no_grad()
def update_ema(ema_model, model, decay=EMA_DECAY):
    src = model.module if hasattr(model,'module') else model
    for ep_, sp in zip(ema_model.parameters(), src.parameters()):
        ep_.data.mul_(decay).add_(sp.data, alpha=1-decay)


def run_epoch(model, loader, optim, sched, device, train=True, grad_accum=1):
    model.train() if train else model.eval()
    losses = []
    ctx = torch.enable_grad() if train else torch.no_grad()
    if train: optim.zero_grad()

    with ctx:
        for batch_idx, (imgs, cvecs) in enumerate(tqdm(loader, desc='tr' if train else 'vl', leave=False)):
            imgs  = imgs.to(device, non_blocking=True)
            cvecs = cvecs.to(device, non_blocking=True)
            B = imgs.shape[0]
            t = torch.randint(0, sched.config.num_train_timesteps, (B,), device=device).long()
            noise = torch.randn_like(imgs)

            cond_input = cvecs.clone()
            drop_mask = torch.rand(B, device=device) < CFG_DROP_PROB
            cond_input[drop_mask] = 0.0

            pred = model(sched.add_noise(imgs, noise, t), t, cond_input)
            loss = F.mse_loss(pred, noise)
            if torch.isnan(loss): continue

            if train:
                (loss / grad_accum).backward()
                if (batch_idx + 1) % grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optim.step()
                    optim.zero_grad()
                    update_ema(ema_model, model)

            losses.append(loss.item())

    return float(np.mean(losses)) if losses else float('nan')


def save_ckpt(path, epoch, model, optimizer, lr_sched, val_loss, history):
    train_state = model.module.state_dict() if hasattr(model,'module') else model.state_dict()
    ema_state = ema_model.state_dict()
    tmp_path = path + '.tmp'
    torch.save({
        'epoch': epoch, 'model_state_dict': train_state, 'ema_state_dict': ema_state,
        'optimizer_state_dict': optimizer.state_dict(), 'lr_scheduler_state_dict': lr_sched.state_dict(),
        'val_loss': val_loss, 'history': history, 'image_size': IMAGE_SIZE,
        'num_concepts': NUM_CONCEPTS, 'concept_names': CONCEPT_NAMES,
        'cfg_scale': CFG_SCALE, 'cfg_drop_prob': CFG_DROP_PROB,
    }, tmp_path)
    os.replace(tmp_path, path)


print(f"\n{'='*65}")
print(f"TRAINING  ep {start_epoch+1}→{SESSION_END}  |  {torch.cuda.get_device_name(0)}")
print(f"CFG(p={CFG_DROP_PROB},s={CFG_SCALE}) | EMA(d={EMA_DECAY}) | 128x128, corrected skips")
print(f"{'='*65}\n")

ep_times = []
for epoch in range(start_epoch, SESSION_END):
    t0 = time.time()
    tr = run_epoch(model, tr_loader, optimizer, noise_sched, device, train=True, grad_accum=GRAD_ACCUM)
    vl = run_epoch(model, vl_loader, optimizer, noise_sched, device, train=False)
    lr_sched.step()

    elapsed = time.time() - t0
    ep_times.append(elapsed)
    eta = ((SESSION_END-(epoch+1)) * np.mean(ep_times[-5:])) / 3600

    history['train_loss'].append(tr)
    history['val_loss'].append(vl)
    history['epoch_times'].append(elapsed)

    log_writer.writerow([epoch+1, tr, vl, elapsed, optimizer.param_groups[0]['lr'], time.time()])
    log_file.flush()   # write immediately — survives a session crash

    print(f"Ep {epoch+1:4d}/{SESSION_END} | tr={tr:.4f} | vl={vl:.4f} | {elapsed:.0f}s | ETA={eta:.1f}h", end='')

    if (epoch+1) % EVAL_EVERY == 0:
        mi = generate_images(ema_model, MEL_VEC, n=8, steps=30, device=device, sz=IMAGE_SIZE)
        bi = generate_images(ema_model, BEN_VEC, n=8, steps=30, device=device, sz=IMAGE_SIZE)
        mc = ccfs_feedback(mi, MEL_VEC, cbm_a, device)
        bc = ccfs_feedback(bi, BEN_VEC, cbm_a, device)
        mean_c = (mc+bc)/2
        history['ccfs_training'].append(mean_c)
        history['ccfs_mel'].append(mc); history['ccfs_ben'].append(bc)
        print(f" | CCFS={mean_c:.4f} [mel={mc:.3f} ben={bc:.3f}]", end='')
        grid = make_grid(torch.cat([mi[:4], bi[:4]]), nrow=4, normalize=False)
        save_image(grid, f'gen_ep{epoch+1:04d}.png')
        print(" [grid saved]")
    else:
        print()

    if (epoch+1) % SAVE_EVERY == 0:
        ckpt_name = f'checkpoint_epoch{epoch+1}.pth'
        save_ckpt(ckpt_name, epoch, model, optimizer, lr_sched, vl, history)
        print(f"  → {ckpt_name} saved (train + EMA weights)")

    if vl < best_val:
        best_val = vl
        save_ckpt('best_clindiffxai_v3.pth', epoch, model, optimizer, lr_sched, vl, history)
        print(f"  ★ best model (vl={vl:.4f})")

log_file.close()
print(f"\nSession complete. Best val_loss={best_val:.4f}")

session_num = SESSION_END // 40
print(f"\n{'='*60}\nSESSION {session_num} DONE\n{'='*60}")
print(f"Download: checkpoint_epoch{SESSION_END}.pth + best_clindiffxai_v3.pth + {LOG_CSV}")
if SESSION_END < 150:
    print(f"\nNext: SESSION_END={min(SESSION_END+40,150)}, "
          f"RESUME_FROM='…/checkpoint_epoch{SESSION_END}.pth'")
else:
    print("\nAll 150 epochs complete. Proceed to NB4.")


# ============================================================
# TRAINING CURVES (PNG + PDF)
# ============================================================

def save_fig_both(fig, name, dpi=300):
    fig.savefig(f'{name}.png', dpi=dpi, bbox_inches='tight', facecolor='white')
    fig.savefig(f'{name}.pdf', bbox_inches='tight', facecolor='white')
    print(f"  Saved: {name}.png + {name}.pdf")


fig, axes = plt.subplots(1, 3, figsize=(15,4))
fig.suptitle(f'ClinDiffXAI v3 (128×128) — ep 1→{len(history["train_loss"])}', fontweight='bold')
er = range(1, len(history['train_loss'])+1)
axes[0].plot(er, history['train_loss'], 'b-', lw=2, label='Train')
axes[0].plot(er, history['val_loss'], 'r-', lw=2, label='Val')
for b in [40,80,120]:
    if b < len(history['train_loss']): axes[0].axvline(x=b, color='gray', ls='--', alpha=0.5)
axes[0].set_title('MSE Loss'); axes[0].set_xlabel('Epoch'); axes[0].legend(); axes[0].grid(alpha=0.3)

if history['ccfs_training']:
    ep_pts = [EVAL_EVERY*(i+1) for i in range(len(history['ccfs_training']))]
    axes[1].plot(ep_pts, history['ccfs_training'], 'g-o', lw=2, ms=8, label='Mean CCFS')
    axes[1].plot(ep_pts, history['ccfs_mel'], 'r--s', lw=1.5, ms=6, label='Mel CCFS')
    axes[1].plot(ep_pts, history['ccfs_ben'], 'b--^', lw=1.5, ms=6, label='Ben CCFS')
    axes[1].axhline(0.70, color='r', ls=':', alpha=0.7, label='0.70 threshold')
    axes[1].set_title('CCFS (EMA model)'); axes[1].set_xlabel('Epoch')
    axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)

short = ['PN','BWV','VS','Pig','Str','D/G','Reg']
x = np.arange(NUM_CONCEPTS)
axes[2].bar(x-0.2, MEL_VEC.numpy(), 0.35, label='Melanoma', color='#d62728', alpha=0.8)
axes[2].bar(x+0.2, BEN_VEC.numpy(), 0.35, label='Benign', color='#1f77b4', alpha=0.8)
axes[2].set_xticks(x); axes[2].set_xticklabels(short, fontsize=9)
axes[2].set_title('Conditioning vectors (v3, image-verified)'); axes[2].set_ylim(0,1.1)
axes[2].legend(); axes[2].grid(alpha=0.3, axis='y')

plt.tight_layout()
save_fig_both(fig, f'nb3v3_session{session_num}_curves')
plt.show()
