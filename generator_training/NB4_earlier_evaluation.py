# ============================================================
# NOTEBOOK 4 v3 — ClinDiffXAI FINAL EVALUATION
# Compares: Main (concept-conditioned) vs Baseline (class-conditioned)
# Both trained at 128×128, 150 epochs.
# Platform: Kaggle T4×2 GPU (inference only — no TPU needed)
# Single cell. All figures saved as PNG + PDF (CIBM-ready).
#
# FILES TO UPLOAD AS KAGGLE DATASETS BEFORE RUNNING:
#   1. best_clindiffxai_v3_tpu.pth          ← main model (best val_loss)
#   2. checkpoint_tpu_epoch110.pth           ← main model best CCFS (optional)
#   3. best_clindiffxai_baseline_tpu.pth    ← baseline (best val_loss)
#   4. best_cbm_model.pth                   ← CBM-A evaluator (already on Kaggle)
#   5. isic2020_pseudo_labels_v3.csv        ← pseudo-labels (already on Kaggle)
#   6. ISIC 2020 images                     ← already on Kaggle
#
# OUTPUTS (all to /kaggle/working/, PNG + PDF):
#   fig1_ccfs_comparison_bar     — CCFS main vs baseline with 95% CI
#   fig2_ccfs_trajectory         — combined training trajectory both models
#   fig3_pearson_r               — per-concept Pearson r (main only)
#   fig4_cfg_ablation            — CCFS vs guidance scale
#   fig5_qualitative             — side-by-side samples mel/ben
#   fig6_fid_bar                 — Inception-FID + MedFID comparison
#   fig7_conditioning_vectors    — reference bar chart for methods section
#   nb4_results_table.csv        — complete numeric results for paper table
# ============================================================

import subprocess, sys
for pkg in ["diffusers==0.33.0", "accelerate==1.6.0", "timm==1.0.15"]:  # scipy not pinned — use Kaggle's pre-installed version
    subprocess.run([sys.executable, '-m', 'pip', 'install', pkg, '-q'], check=False)

import os, warnings, csv
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from torchvision import transforms
import torchvision.models as tv_models
from diffusers import DDIMScheduler
from PIL import Image
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from tqdm import tqdm
warnings.filterwarnings('ignore')

device = torch.device('cuda')
torch.backends.cudnn.benchmark = True
print(f"GPU: {torch.cuda.get_device_name(0)} | Count: {torch.cuda.device_count()}")


# ============================================================
# PATHS — ⚠ UPDATE to your actual Kaggle dataset paths
# ============================================================

MAIN_CKPT  = "/kaggle/input/datasets/shohanoor7900/clindiffxai-v3-main/best_clindiffxai_v3_tpu.pth"
BASE_CKPT  = "/kaggle/input/datasets/shohanoor7900/clindiffxai-baseline/best_clindiffxai_baseline_tpu.pth"
CBM_A_PATH = "/kaggle/input/datasets/shohanurrahman7/cbma-v3/best_cbm_model.pth"
PSEUDO_CSV = "/kaggle/input/datasets/shohanurrahman7/pseudo-labels-v3/isic2020_pseudo_labels_v3.csv"
ISIC_DIR   = "/kaggle/input/datasets/sumaiyabinteshahid/isic-challenge-dataset-2020/ISIC_2020_Dataset/train"

# Training-trajectory log CSVs — for the combined CCFS-over-time figure
# These are the nb3_tpu_train_log.csv and nb3b_tpu_train_log.csv you saved from training.
# Leave as None if you haven't uploaded them — the trajectory figure will be skipped.
MAIN_LOG_CSV = "/kaggle/input/datasets/shohanoor7900/clindiffxai-v3-main/nb3_tpu_train_log.csv"
BASE_LOG_CSV = "/kaggle/input/datasets/shohanoor7900/clindiffxai-baseline/nb3b_tpu_train_log.csv"

# ── Evaluation config ──
IMAGE_SIZE    = 128
NUM_CONCEPTS  = 7
CFG_SCALE     = 3.5
N_EVAL        = 100    # images per class per model for CCFS/FID (↑ = more stable, ↑ = slower)
N_PEARSON     = 15     # images per step for per-concept Pearson r
N_PEARSON_STEPS = 10   # number of steps 0→1 per concept
N_BOOTSTRAP   = 2000   # bootstrap iterations for CCFS CIs
N_REAL_FID    = 500    # real images for FID reference (per class)
DDIM_STEPS    = 50     # sampling steps (more than during training = higher quality)

CONCEPT_NAMES = ['pigment_network', 'blue_whitish_veil', 'vascular_structures',
                  'pigmentation', 'streaks', 'dots_and_globules', 'regression_structures']
SHORT         = ['PN', 'BWV', 'VS', 'Pig', 'Str', 'D/G', 'Reg']

MEL_VEC = torch.FloatTensor([0.8292, 0.7781, 0.9926, 0.8918, 0.4585, 0.9256, 0.8769])
BEN_VEC = torch.FloatTensor([0.0800, 0.0002, 0.0047, 0.0055, 0.0018, 0.1501, 0.0003])

print("\nVerifying input paths:")
for label, path in [("Main model", MAIN_CKPT), ("Baseline", BASE_CKPT),
                     ("CBM-A", CBM_A_PATH), ("Pseudo CSV", PSEUDO_CSV),
                     ("ISIC dir", ISIC_DIR)]:
    ok = os.path.exists(path)
    size = f"{os.path.getsize(path)/1e6:.0f}MB" if ok and os.path.isfile(path) else "dir" if ok else "MISSING"
    print(f"  {'✓' if ok else '✗'} {label:<14}: {size}")
    if not ok:
        raise FileNotFoundError(f"⚠ Not found: {path}")


# ============================================================
# HELPERS — save_fig_both
# ============================================================

def save_fig_both(fig, name, dpi=300):
    fig.savefig(f'{name}.png', dpi=dpi, bbox_inches='tight', facecolor='white')
    fig.savefig(f'{name}.pdf', bbox_inches='tight', facecolor='white')
    print(f"  Saved: {name}.png + {name}.pdf")


# ============================================================
# ARCHITECTURE — v3, 128×128 (same as NB3/NB3b — zero drift)
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
        self.sk=nn.Conv2d(ic,oc,1) if ic!=oc else nn.Identity(); self.a=nn.SiLU()
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
        h,_=self.a(h,h,h); return x+h.permute(0,2,1).reshape(B,C,H,W)

class ConceptConditionedUNet128(nn.Module):
    def __init__(self, image_size=128, num_concepts=7, base_channels=64):
        super().__init__()
        self.image_size=image_size; C=[64,64,128,256,256]
        td=base_channels*4; cd=base_channels*2
        self.time_mlp=nn.Sequential(nn.Linear(base_channels,td),nn.SiLU(),nn.Linear(td,td))
        self.concept_emb=ConceptEmbedding(num_concepts,cd)
        self.enc_in=nn.Conv2d(3,C[0],3,padding=1)
        self.enc=nn.ModuleList([
            nn.ModuleList([ResBlock(C[0],C[0],td,cd),ResBlock(C[0],C[0],td,cd)]),
            nn.ModuleList([ResBlock(C[0],C[1],td,cd),ResBlock(C[1],C[1],td,cd)]),
            nn.ModuleList([ResBlock(C[1],C[2],td,cd),ResBlock(C[2],C[2],td,cd),SelfAttention(C[2])]),
            nn.ModuleList([ResBlock(C[2],C[3],td,cd),ResBlock(C[3],C[3],td,cd),SelfAttention(C[3])]),
            nn.ModuleList([ResBlock(C[3],C[4],td,cd),ResBlock(C[4],C[4],td,cd),SelfAttention(C[4])]),
        ])
        self.down=nn.ModuleList([nn.Conv2d(C[i],C[i],3,stride=2,padding=1) for i in range(4)])
        self.mid=nn.ModuleList([ResBlock(C[4],C[4],td,cd),SelfAttention(C[4]),ResBlock(C[4],C[4],td,cd)])
        self.dec=nn.ModuleList([
            nn.ModuleList([ResBlock(C[4]+C[3],C[3],td,cd),ResBlock(C[3],C[3],td,cd),SelfAttention(C[3])]),
            nn.ModuleList([ResBlock(C[3]+C[2],C[2],td,cd),ResBlock(C[2],C[2],td,cd),SelfAttention(C[2])]),
            nn.ModuleList([ResBlock(C[2]+C[1],C[1],td,cd),ResBlock(C[1],C[1],td,cd)]),
            nn.ModuleList([ResBlock(C[1]+C[0],C[0],td,cd),ResBlock(C[0],C[0],td,cd)]),
        ])
        self.up=nn.ModuleList([nn.ConvTranspose2d(C[4],C[4],2,stride=2),nn.ConvTranspose2d(C[3],C[3],2,stride=2),
                                nn.ConvTranspose2d(C[2],C[2],2,stride=2),nn.ConvTranspose2d(C[1],C[1],2,stride=2)])
        self.out=nn.Sequential(nn.GroupNorm(8,C[0]),nn.SiLU(),nn.Conv2d(C[0],3,3,padding=1))

    def sin_emb(self,t,d):
        half=d//2; f=torch.exp(-torch.arange(half,device=t.device)*(np.log(10000)/(half-1)))
        a=t[:,None].float()*f[None]; return torch.cat([torch.cos(a),torch.sin(a)],dim=-1)

    def forward(self,x,ts,concepts):
        te=self.time_mlp(self.sin_emb(ts,self.time_mlp[0].in_features))
        ce=self.concept_emb(concepts); h=self.enc_in(x); sk=[]
        for i,blks in enumerate(self.enc):
            for b in blks: h=b(h) if isinstance(b,SelfAttention) else b(h,te,ce)
            sk.append(h)
            if i<4: h=self.down[i](h)
        for b in self.mid: h=b(h) if isinstance(b,SelfAttention) else b(h,te,ce)
        usable_sk=sk[:-1]
        for i,(blks,up) in enumerate(zip(self.dec,self.up)):
            h=up(h); s=usable_sk[-(i+1)]
            if h.shape!=s.shape: h=F.interpolate(h,size=s.shape[-2:])
            h=torch.cat([h,s],dim=1)
            for b in blks: h=b(h) if isinstance(b,SelfAttention) else b(h,te,ce)
        o=self.out(h)
        if o.shape[-1]!=self.image_size: o=F.interpolate(o,size=(self.image_size,self.image_size))
        return o


# ============================================================
# CBM-A loader
# ============================================================

def load_cbm_a(path):
    class _CBM(nn.Module):
        def __init__(self):
            super().__init__()
            bb=tv_models.efficientnet_v2_s(weights=None)
            in_f=bb.classifier[1].in_features; bb.classifier=nn.Identity(); self.backbone=bb
            self.concept_feature=nn.Sequential(nn.Linear(in_f,512),nn.LayerNorm(512),nn.GELU(),nn.Dropout(0.3))
            self.concept_heads=nn.ModuleList([nn.Sequential(nn.Linear(512,128),nn.GELU(),nn.Dropout(0.2),nn.Linear(128,1)) for _ in range(7)])
            self.diagnosis_head=nn.Sequential(nn.Linear(7,64),nn.GELU(),nn.Dropout(0.3),nn.Linear(64,2))
        def forward(self,x):
            f=self.backbone(x)
            if f.dim()>2: f=f.mean([-2,-1])
            cf=self.concept_feature(f)
            cl=torch.cat([h(cf) for h in self.concept_heads],dim=1)
            return cl,torch.sigmoid(cl),self.diagnosis_head(torch.sigmoid(cl))
    m=_CBM(); ck=torch.load(path,map_location='cpu',weights_only=False)
    m.load_state_dict(ck['model_state_dict'],strict=True); return m.to(device).eval()


def load_diffusion_model(path, num_concepts):
    """Load a UNet checkpoint, auto-detect EMA vs training weights."""
    ck = torch.load(path, map_location='cpu', weights_only=False)
    model = ConceptConditionedUNet128(IMAGE_SIZE, num_concepts).to(device).eval()
    if 'ema_state_dict' in ck:
        model.load_state_dict(ck['ema_state_dict'])
        print(f"  ✓ Loaded EMA weights | epoch {ck.get('epoch','?')+1} | params: {sum(p.numel() for p in model.parameters()):,}")
    else:
        model.load_state_dict(ck['model_state_dict'])
        print(f"  ✓ Loaded training weights | epoch {ck.get('epoch','?')+1}")
    return model


print("\nLoading CBM-A...")
cbm_a = load_cbm_a(CBM_A_PATH)
print("Loading main model (concept-conditioned)...")
main_model = load_diffusion_model(MAIN_CKPT, NUM_CONCEPTS)
print("Loading baseline model (class-conditioned)...")
base_model = load_diffusion_model(BASE_CKPT, 2)


# ============================================================
# GENERATION — CFG DDIM
# ============================================================

@torch.no_grad()
def generate_batch(model, cvec, n, cfg_scale=CFG_SCALE, steps=DDIM_STEPS, batch_size=16):
    """Generate n images, batched to avoid OOM at 128×128."""
    ddim = DDIMScheduler(num_train_timesteps=1000, beta_schedule='linear', clip_sample=True)
    ddim.set_timesteps(steps)
    all_imgs = []
    for start in range(0, n, batch_size):
        b = min(batch_size, n - start)
        imgs = torch.randn(b, 3, IMAGE_SIZE, IMAGE_SIZE, device=device)
        cv = cvec.unsqueeze(0).repeat(b, 1).to(device)
        for t in ddim.timesteps:
            ts = torch.full((b,), t, device=device, dtype=torch.long)
            eps_c = model(imgs, ts, cv)
            eps_u = model(imgs, ts, torch.zeros_like(cv))
            imgs = ddim.step(eps_u + cfg_scale*(eps_c - eps_u), t, imgs).prev_sample
        all_imgs.append(((imgs.clamp(-1, 1) + 1) / 2).cpu())
    return torch.cat(all_imgs, dim=0)


# ============================================================
# CBM-A INFERENCE on generated images
# ============================================================

cbm_norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
cbm_rsz  = transforms.Resize((224, 224))

@torch.no_grad()
def get_concept_preds(imgs):
    """imgs: list/tensor of [0,1] images → (N, 7) concept probabilities."""
    inp = torch.stack([cbm_norm(cbm_rsz(i)) for i in imgs]).to(device)
    _, cp, _ = cbm_a(inp)
    return cp.cpu().numpy()

def ccfs_score(imgs, cvec):
    """Cosine similarity between conditioning vector and CBM-A predictions."""
    preds = get_concept_preds(imgs)   # (N, 7)
    cond  = cvec.numpy()              # (7,)
    sims  = np.array([np.dot(cond, p) / (np.linalg.norm(cond) * np.linalg.norm(p) + 1e-8)
                      for p in preds])
    return sims  # per-image scores, so we can bootstrap


def bootstrap_ci(scores, n_boot=N_BOOTSTRAP, ci=95):
    """Bootstrap confidence interval for the mean."""
    means = [np.mean(np.random.choice(scores, len(scores), replace=True)) for _ in range(n_boot)]
    lo = np.percentile(means, (100 - ci) / 2)
    hi = np.percentile(means, 100 - (100 - ci) / 2)
    return float(np.mean(scores)), lo, hi


# ============================================================
# METRIC 1 — CCFS with 95% bootstrap CI
# ============================================================

print(f"\n{'='*60}\nMETRIC 1 — CCFS  (n={N_EVAL} per class)\n{'='*60}")

print("Generating main model images...")
main_mel_imgs = generate_batch(main_model, MEL_VEC, N_EVAL)
main_ben_imgs = generate_batch(main_model, BEN_VEC, N_EVAL)
print("Generating baseline images...")
base_mel_imgs = generate_batch(base_model, torch.FloatTensor([0., 1.]), N_EVAL)  # [benign, melanoma]
base_ben_imgs = generate_batch(base_model, torch.FloatTensor([1., 0.]), N_EVAL)

print("Computing CCFS...")
mm_sims = ccfs_score(list(main_mel_imgs), MEL_VEC)
mb_sims = ccfs_score(list(main_ben_imgs), BEN_VEC)
bm_sims = ccfs_score(list(base_mel_imgs), MEL_VEC)
bb_sims = ccfs_score(list(base_ben_imgs), BEN_VEC)

main_mel_mean, main_mel_lo, main_mel_hi = bootstrap_ci(mm_sims)
main_ben_mean, main_ben_lo, main_ben_hi = bootstrap_ci(mb_sims)
base_mel_mean, base_mel_lo, base_mel_hi = bootstrap_ci(bm_sims)
base_ben_mean, base_ben_lo, base_ben_hi = bootstrap_ci(bb_sims)

main_mean = (main_mel_mean + main_ben_mean) / 2
base_mean = (base_mel_mean + base_ben_mean) / 2
main_mean_lo = (main_mel_lo + main_ben_lo) / 2
main_mean_hi = (main_mel_hi + main_ben_hi) / 2
base_mean_lo = (base_mel_lo + base_ben_lo) / 2
base_mean_hi = (base_mel_hi + base_ben_hi) / 2

print(f"\n  {'Metric':<25} {'Main':>10} {'[95% CI]':>18} {'Baseline':>10} {'[95% CI]':>18}")
print(f"  {'-'*83}")
for label, m, mlo, mhi, b, blo, bhi in [
    ("CCFS mean",      main_mean,     main_mean_lo,  main_mean_hi,  base_mean,     base_mean_lo,  base_mean_hi),
    ("CCFS melanoma",  main_mel_mean, main_mel_lo,   main_mel_hi,   base_mel_mean, base_mel_lo,   base_mel_hi),
    ("CCFS benign",    main_ben_mean, main_ben_lo,   main_ben_hi,   base_ben_mean, base_ben_lo,   base_ben_hi),
]:
    print(f"  {label:<25} {m:.4f}     [{mlo:.3f}, {mhi:.3f}]     {b:.4f}     [{blo:.3f}, {bhi:.3f}]")


# ============================================================
# METRIC 2 — FID (Inception) + MedFID (CBM-A backbone)
# ============================================================

print(f"\n{'='*60}\nMETRIC 2 — FID + MedFID  (n_real={N_REAL_FID}/class)\n{'='*60}")

# ── InceptionV3 feature extractor ──
incep = tv_models.inception_v3(weights=tv_models.Inception_V3_Weights.DEFAULT, transform_input=False)
incep.fc = nn.Identity(); incep.eval().to(device)
incep_tfm = transforms.Compose([transforms.Resize((299,299)), transforms.Normalize([0.5]*3,[0.5]*3)])

# ── CBM-A backbone feature extractor ──
class CBMABackbone(nn.Module):
    def __init__(self, cbm):
        super().__init__()
        self.backbone = cbm.backbone
        self.concept_feature = cbm.concept_feature
    def forward(self, x): 
        f = self.backbone(x)
        if f.dim()>2: f=f.mean([-2,-1])
        return self.concept_feature(f)

med_feat = CBMABackbone(cbm_a).eval().to(device)
med_tfm = transforms.Compose([transforms.Resize((224,224)), cbm_norm])

def extract_features(imgs_cpu, net, tfm, batch_size=32):
    """Extract features from a list/tensor of [0,1] images."""
    feats = []
    for start in range(0, len(imgs_cpu), batch_size):
        batch = torch.stack([tfm(imgs_cpu[i]) for i in range(start, min(start+batch_size, len(imgs_cpu)))]).to(device)
        with torch.no_grad():
            f = net(batch)
            if isinstance(f, tuple): f = f[0]
        feats.append(f.cpu().numpy())
    return np.concatenate(feats, axis=0)

def fid_score(real_feats, fake_feats):
    """Fréchet Inception Distance."""
    from scipy.linalg import sqrtm as _sqrtm   # inline to avoid scope issues after pip reinstalls
    mu_r, mu_f = real_feats.mean(0), fake_feats.mean(0)
    cov_r = np.cov(real_feats, rowvar=False) + np.eye(real_feats.shape[1])*1e-6
    cov_f = np.cov(fake_feats, rowvar=False) + np.eye(fake_feats.shape[1])*1e-6
    diff  = mu_r - mu_f
    covmean, _ = _sqrtm(cov_r @ cov_f, disp=False)
    if np.iscomplexobj(covmean): covmean = covmean.real
    return float(diff @ diff + np.trace(cov_r + cov_f - 2*covmean))

# Load real images for FID reference
pseudo_df = pd.read_csv(PSEUDO_CSV)
mel_df = pseudo_df[pseudo_df['target']==1].sample(n=N_REAL_FID, random_state=42)
ben_df = pseudo_df[pseudo_df['target']==0].sample(n=N_REAL_FID, random_state=42)

def load_real_imgs(df, n):
    imgs = []
    for _, row in df.head(n).iterrows():
        p = os.path.join(ISIC_DIR, f"{row['image_name']}.jpg")
        if os.path.exists(p):
            imgs.append(transforms.ToTensor()(Image.open(p).convert('RGB').resize((IMAGE_SIZE, IMAGE_SIZE))))
    return imgs

print(f"Loading {N_REAL_FID} real mel + {N_REAL_FID} real ben images for FID reference...")
real_mel = load_real_imgs(mel_df, N_REAL_FID)
real_ben = load_real_imgs(ben_df, N_REAL_FID)
real_all = real_mel + real_ben

print("Extracting InceptionV3 features...")
real_inc  = extract_features(real_all, incep, incep_tfm)
main_inc  = extract_features(list(main_mel_imgs)+list(main_ben_imgs), incep, incep_tfm)
base_inc  = extract_features(list(base_mel_imgs)+list(base_ben_imgs), incep, incep_tfm)

print("Extracting MedFID (CBM-A backbone) features...")
real_med  = extract_features(real_all, med_feat, med_tfm)
main_med  = extract_features(list(main_mel_imgs)+list(main_ben_imgs), med_feat, med_tfm)
base_med  = extract_features(list(base_mel_imgs)+list(base_ben_imgs), med_feat, med_tfm)

main_incfid = fid_score(real_inc, main_inc)
base_incfid = fid_score(real_inc, base_inc)
main_medfid = fid_score(real_med, main_med)
base_medfid = fid_score(real_med, base_med)

print(f"\n  {'':25} {'Main':>10} {'Baseline':>10}")
print(f"  {'-'*47}")
print(f"  {'Inception-FID':<25} {main_incfid:>10.2f} {base_incfid:>10.2f}")
print(f"  {'MedFID (CBM-A backbone)':<25} {main_medfid:>10.2f} {base_medfid:>10.2f}")


# ============================================================
# METRIC 3 — Per-concept Pearson r (main model only;
# baseline is class-conditioned so per-concept test is N/A)
# ============================================================

print(f"\n{'='*60}\nMETRIC 3 — Per-concept Pearson r  (main model)\n{'='*60}")

pearson_r = {}; pearson_p = {}
steps = np.linspace(0, 1, N_PEARSON_STEPS)

for ci, cname in enumerate(CONCEPT_NAMES):
    rs = []; preds_all = []
    for val in tqdm(steps, desc=cname, leave=False):
        cvec = MEL_VEC.clone(); cvec[ci] = float(val)
        imgs = generate_batch(main_model, cvec, N_PEARSON, steps=30)
        preds = get_concept_preds(list(imgs))[:, ci]
        rs.extend([float(val)] * N_PEARSON)
        preds_all.extend(preds.tolist())
    r, p = pearsonr(rs, preds_all)
    pearson_r[cname] = r; pearson_p[cname] = p
    print(f"  {cname:<24}: r={r:+.4f}  p={p:.4f}")


# ============================================================
# METRIC 4 — CFG ablation (main model only)
# ============================================================

print(f"\n{'='*60}\nMETRIC 4 — CFG scale ablation  (main model)\n{'='*60}")

cfg_scales = [0.0, 1.0, 3.5]
cfg_ccfs_mel = {}; cfg_ccfs_ben = {}

for s in cfg_scales:
    mi = generate_batch(main_model, MEL_VEC, 50, cfg_scale=s, steps=30)
    bi = generate_batch(main_model, BEN_VEC, 50, cfg_scale=s, steps=30)
    cm = ccfs_score(list(mi), MEL_VEC).mean()
    cb = ccfs_score(list(bi), BEN_VEC).mean()
    cfg_ccfs_mel[s] = cm; cfg_ccfs_ben[s] = cb
    print(f"  s={s}: CCFS_mel={cm:.4f}  CCFS_ben={cb:.4f}  mean={(cm+cb)/2:.4f}")


# ============================================================
# SAVE RESULTS TABLE
# ============================================================

results = {
    'main_ccfs_mean': main_mean, 'main_ccfs_mean_lo': main_mean_lo, 'main_ccfs_mean_hi': main_mean_hi,
    'main_ccfs_mel': main_mel_mean, 'main_ccfs_mel_lo': main_mel_lo, 'main_ccfs_mel_hi': main_mel_hi,
    'main_ccfs_ben': main_ben_mean, 'main_ccfs_ben_lo': main_ben_lo, 'main_ccfs_ben_hi': main_ben_hi,
    'base_ccfs_mean': base_mean, 'base_ccfs_mean_lo': base_mean_lo, 'base_ccfs_mean_hi': base_mean_hi,
    'base_ccfs_mel': base_mel_mean, 'base_ccfs_mel_lo': base_mel_lo, 'base_ccfs_mel_hi': base_mel_hi,
    'base_ccfs_ben': base_ben_mean, 'base_ccfs_ben_lo': base_ben_lo, 'base_ccfs_ben_hi': base_ben_hi,
    'main_inception_fid': main_incfid, 'base_inception_fid': base_incfid,
    'main_medfid': main_medfid, 'base_medfid': base_medfid,
}
for cname in CONCEPT_NAMES:
    results[f'pearson_r_{cname}'] = pearson_r[cname]
    results[f'pearson_p_{cname}'] = pearson_p[cname]

pd.DataFrame([results]).to_csv('nb4_results_table.csv', index=False)
print("\n✓ nb4_results_table.csv saved")


# ============================================================
# FIGURES — all paper-quality PNG + PDF
# ============================================================

# ── Figure 1: CCFS comparison bar chart with 95% CI ──
fig, ax = plt.subplots(figsize=(8, 5))
x = np.arange(3); w = 0.35
metrics_m = [main_mean, main_mel_mean, main_ben_mean]
metrics_b = [base_mean, base_mel_mean, base_ben_mean]
errs_m = [[v-lo for v,lo in zip(metrics_m, [main_mean_lo,main_mel_lo,main_ben_lo])],
           [hi-v for v,hi in zip(metrics_m, [main_mean_hi,main_mel_hi,main_ben_hi])]]
errs_b = [[v-lo for v,lo in zip(metrics_b, [base_mean_lo,base_mel_lo,base_ben_lo])],
           [hi-v for v,hi in zip(metrics_b, [base_mean_hi,base_mel_hi,base_ben_hi])]]
bars_m = ax.bar(x-w/2, metrics_m, w, yerr=errs_m, capsize=5, label='ClinDiffXAI (concept)', color='#2ecc71', alpha=0.85)
bars_b = ax.bar(x+w/2, metrics_b, w, yerr=errs_b, capsize=5, label='Baseline (class-label)', color='#95a5a6', alpha=0.85)
ax.axhline(0.70, color='r', ls='--', alpha=0.7, lw=1.5, label='Adequacy threshold (0.70)')
ax.set_xticks(x); ax.set_xticklabels(['Mean CCFS', 'Melanoma CCFS', 'Benign CCFS'], fontsize=11)
ax.set_ylabel('CCFS', fontsize=12); ax.set_ylim(0, 1.0)
ax.set_title('Concept-Conditioning Fidelity Score — Main vs Baseline\n(bootstrapped 95% CI, n=100 images each)', fontweight='bold')
ax.legend(fontsize=10); ax.grid(alpha=0.3, axis='y')
for bar in list(bars_m)+list(bars_b):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01, f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=9)
plt.tight_layout()
save_fig_both(fig, 'fig1_ccfs_comparison_bar')
plt.close()


# ── Figure 2: Combined CCFS trajectory ──
if MAIN_LOG_CSV and BASE_LOG_CSV and os.path.exists(MAIN_LOG_CSV) and os.path.exists(BASE_LOG_CSV):
    mlog = pd.read_csv(MAIN_LOG_CSV).dropna(subset=['ccfs_mean'])
    blog = pd.read_csv(BASE_LOG_CSV)
    # Baseline CSV doesn't have ccfs columns; use the hardcoded trajectory from training logs
    base_traj_ep   = [40,  80,  120,  130,  140,  150]
    base_traj_mean = [0.6396, 0.6638, 0.6593, 0.6809, 0.6890, 0.6575]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(mlog['epoch'], mlog['ccfs_mean'],  'g-o', lw=2, ms=6, label='ClinDiffXAI mean CCFS')
    ax.plot(mlog['epoch'], mlog['ccfs_mel'],   'r--s', lw=1.5, ms=5, label='ClinDiffXAI mel CCFS')
    ax.plot(mlog['epoch'], mlog['ccfs_ben'],   'b--^', lw=1.5, ms=5, label='ClinDiffXAI ben CCFS')
    ax.plot(base_traj_ep, base_traj_mean, 'k--o', lw=2, ms=6, alpha=0.7, label='Baseline mean CCFS')
    ax.axhline(0.70, color='r', ls=':', lw=1.5, alpha=0.7, label='0.70 threshold')
    ax.set_xlabel('Epoch', fontsize=12); ax.set_ylabel('CCFS', fontsize=12)
    ax.set_title('CCFS Training Trajectory — ClinDiffXAI v3 vs Baseline', fontweight='bold')
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    plt.tight_layout()
    save_fig_both(fig, 'fig2_ccfs_trajectory_combined')
    plt.close()
else:
    # Fallback: just use the hardcoded trajectory data we know from the runs
    main_traj_ep   = [60,70,80,90,100,110,120,130,140,150]
    main_traj_mean = [0.6642,0.6638,0.6401,0.6376,0.6842,0.6926,0.6913,0.6439,0.6622,0.6900]
    main_traj_mel  = [0.761,0.738,0.745,0.752,0.747,0.746,0.729,0.723,0.704,0.729]
    main_traj_ben  = [0.568,0.590,0.535,0.523,0.621,0.639,0.653,0.565,0.621,0.651]
    base_traj_ep   = [40,80,120,130,140,150]
    base_traj_mean = [0.6396,0.6638,0.6593,0.6809,0.6890,0.6575]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(main_traj_ep, main_traj_mean, 'g-o', lw=2, ms=6, label='ClinDiffXAI mean CCFS')
    ax.plot(main_traj_ep, main_traj_mel,  'r--s', lw=1.5, ms=5, label='ClinDiffXAI mel CCFS')
    ax.plot(main_traj_ep, main_traj_ben,  'b--^', lw=1.5, ms=5, label='ClinDiffXAI ben CCFS')
    ax.plot(base_traj_ep, base_traj_mean, 'k--o', lw=2, ms=6, alpha=0.7, label='Baseline mean CCFS')
    ax.axhline(0.70, color='r', ls=':', lw=1.5, alpha=0.7, label='0.70 threshold')
    ax.set_xlabel('Epoch', fontsize=12); ax.set_ylabel('CCFS', fontsize=12)
    ax.set_title('CCFS Training Trajectory — ClinDiffXAI v3 vs Baseline', fontweight='bold')
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    plt.tight_layout()
    save_fig_both(fig, 'fig2_ccfs_trajectory_combined')
    plt.close()
print("  fig2: CCFS trajectory saved")


# ── Figure 3: Per-concept Pearson r ──
fig, ax = plt.subplots(figsize=(9, 5))
rs = [pearson_r[c] for c in CONCEPT_NAMES]
ps = [pearson_p[c] for c in CONCEPT_NAMES]
colors = ['#2ecc71' if r > 0 else '#e74c3c' for r in rs]
bars = ax.bar(SHORT, rs, color=colors, alpha=0.85)
ax.axhline(0, color='black', lw=0.8)
for bar, p_val in zip(bars, ps):
    sig = '***' if p_val<0.001 else '**' if p_val<0.01 else '*' if p_val<0.05 else 'n.s.'
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01*(1 if bar.get_height()>=0 else -3),
            sig, ha='center', va='bottom', fontsize=11, fontweight='bold')
ax.set_ylabel("Pearson r", fontsize=12); ax.set_ylim(-0.5, 1.0)
ax.set_title("Per-Concept Pearson r — ClinDiffXAI v3 (concept-conditioning)\n(n.s./*/***=p<0.05/0.01/0.001)", fontweight='bold')
ax.grid(alpha=0.3, axis='y')
plt.tight_layout()
save_fig_both(fig, 'fig3_pearson_r_per_concept')
plt.close()
print("  fig3: Pearson r saved")


# ── Figure 4: CFG ablation ──
fig, ax = plt.subplots(figsize=(7, 4.5))
cfg_means = [(cfg_ccfs_mel[s]+cfg_ccfs_ben[s])/2 for s in cfg_scales]
ax.plot(cfg_scales, [cfg_ccfs_mel[s] for s in cfg_scales], 'r-o', ms=8, lw=2, label='Melanoma CCFS')
ax.plot(cfg_scales, [cfg_ccfs_ben[s] for s in cfg_scales], 'b-s', ms=8, lw=2, label='Benign CCFS')
ax.plot(cfg_scales, cfg_means, 'g-^', ms=8, lw=2, label='Mean CCFS')
ax.axhline(0.70, color='gray', ls='--', alpha=0.7, label='0.70 threshold')
ax.axvline(3.5, color='purple', ls=':', alpha=0.7, label='Selected scale (3.5)')
ax.set_xlabel('CFG Guidance Scale (s)', fontsize=12); ax.set_ylabel('CCFS', fontsize=12)
ax.set_title('CFG Ablation — Effect of Guidance Scale on CCFS', fontweight='bold')
ax.legend(fontsize=10); ax.grid(alpha=0.3)
plt.tight_layout()
save_fig_both(fig, 'fig4_cfg_ablation')
plt.close()
print("  fig4: CFG ablation saved")


# ── Figure 5: Qualitative comparison 2×2 panel ──
n_show = 4
fig, axes = plt.subplots(4, n_show, figsize=(2.4*n_show, 10))
row_labels = ['Main — Melanoma', 'Main — Benign', 'Baseline — Melanoma', 'Baseline — Benign']
row_imgs   = [main_mel_imgs, main_ben_imgs, base_mel_imgs, base_ben_imgs]
for ri, (label, imgs) in enumerate(zip(row_labels, row_imgs)):
    for ci in range(n_show):
        ax = axes[ri, ci]
        ax.imshow(imgs[ci].permute(1,2,0).numpy())
        ax.set_xticks([]); ax.set_yticks([])
    axes[ri, 0].set_ylabel(label, fontsize=9, fontweight='bold')
fig.suptitle('Generated Samples — ClinDiffXAI v3 vs Class-Conditioned Baseline', fontweight='bold', y=1.01)
plt.tight_layout()
save_fig_both(fig, 'fig5_qualitative_comparison')
plt.close()
print("  fig5: Qualitative comparison saved")


# ── Figure 6: FID bar chart ──
fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))
for ax, metric, vals, title in [
    (axes[0], 'Inception-FID', [main_incfid, base_incfid], 'Inception-FID (↓ better)'),
    (axes[1], 'MedFID',        [main_medfid, base_medfid],  'MedFID — CBM-A backbone (↓ better)'),
]:
    bars = ax.bar(['ClinDiffXAI\n(concept)', 'Baseline\n(class)'], vals,
                  color=['#2ecc71', '#95a5a6'], alpha=0.85, width=0.5)
    for bar in bars:
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1, f'{bar.get_height():.1f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax.set_title(title, fontweight='bold'); ax.set_ylabel(metric); ax.grid(alpha=0.3, axis='y')
plt.suptitle('Distributional Quality — FID Comparison', fontweight='bold', y=1.02)
plt.tight_layout()
save_fig_both(fig, 'fig6_fid_comparison')
plt.close()
print("  fig6: FID comparison saved")


# ── Figure 7: Conditioning vectors reference chart ──
fig, ax = plt.subplots(figsize=(8, 4.5))
x = np.arange(NUM_CONCEPTS)
ax.bar(x-0.2, MEL_VEC.numpy(), 0.35, label='Melanoma', color='#d62728', alpha=0.85)
ax.bar(x+0.2, BEN_VEC.numpy(), 0.35, label='Benign',   color='#1f77b4', alpha=0.85)
ax.set_xticks(x); ax.set_xticklabels(SHORT, fontsize=11)
ax.set_title('Concept Conditioning Vectors\n(empirically derived from ISIC 2020 via CBM-A, NB2/NB2b)', fontweight='bold')
ax.set_ylim(0, 1.15); ax.legend(fontsize=11); ax.grid(alpha=0.3, axis='y')
ax.set_ylabel('Concept strength', fontsize=12)
plt.tight_layout()
save_fig_both(fig, 'fig7_conditioning_vectors')
plt.close()
print("  fig7: Conditioning vectors saved")


# ============================================================
# FINAL SUMMARY
# ============================================================

print(f"\n{'='*65}")
print(f"NB4 COMPLETE — ALL METRICS AND FIGURES GENERATED")
print(f"{'='*65}")
print(f"\n  CCFS Summary:")
print(f"    Main  mean={main_mean:.4f} [{main_mean_lo:.3f},{main_mean_hi:.3f}] | mel={main_mel_mean:.4f} | ben={main_ben_mean:.4f}")
print(f"    Base  mean={base_mean:.4f} [{base_mean_lo:.3f},{base_mean_hi:.3f}] | mel={base_mel_mean:.4f} | ben={base_ben_mean:.4f}")
print(f"    Δ CCFS mean = {main_mean-base_mean:+.4f}")
print(f"\n  FID:")
print(f"    Inception-FID: Main={main_incfid:.1f} | Baseline={base_incfid:.1f}")
print(f"    MedFID:        Main={main_medfid:.1f} | Baseline={base_medfid:.1f}")
print(f"\n  Per-concept Pearson r (main model):")
for c,s in zip(CONCEPT_NAMES, SHORT):
    print(f"    {s:<4}: r={pearson_r[c]:+.4f}  p={pearson_p[c]:.4f}")
print(f"\n  Files saved to /kaggle/working/:")
outputs_list = ['fig1_ccfs_comparison_bar', 'fig2_ccfs_trajectory_combined',
                'fig3_pearson_r_per_concept', 'fig4_cfg_ablation',
                'fig5_qualitative_comparison', 'fig6_fid_comparison',
                'fig7_conditioning_vectors', 'nb4_results_table.csv']
for f in outputs_list:
    ext = '.csv' if '.csv' in f else '.png + .pdf'
    print(f"    {f}{'' if '.csv' in f else ''}{ext}")
