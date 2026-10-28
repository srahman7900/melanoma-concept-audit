# ============================================================
# NB4-STATS — Statistical Supplement
# Run AFTER NB4 completes (uses its saved generated images/scores).
# Computes: Wilcoxon test, Spearman rho, BH-corrected Pearson,
# CIs for all correlations.
# Platform: GPU T4×2 (inference only, ~20 min)
# ============================================================

import subprocess, sys
subprocess.run([sys.executable, '-m', 'pip', 'install', 'diffusers==0.33.0',
                'accelerate==1.6.0', 'timm==1.0.15', '-q'], check=False)

import os, warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models
from torchvision import transforms
from diffusers import DDIMScheduler
from PIL import Image
from scipy.stats import wilcoxon, spearmanr, pearsonr
from statsmodels.stats.multitest import multipletests
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
warnings.filterwarnings('ignore')

device = torch.device('cuda')
torch.backends.cudnn.benchmark = True
print(f"GPU: {torch.cuda.get_device_name(0)}")


# ============================================================
# PATHS — same as NB4, update if different
# ============================================================

MAIN_CKPT  = "/kaggle/input/datasets/shohanoor7900/clindiffxai-v3-main/best_clindiffxai_v3_tpu.pth"
BASE_CKPT  = "/kaggle/input/datasets/shohanoor7900/clindiffxai-baseline/best_clindiffxai_baseline_tpu.pth"
CBM_A_PATH = "/kaggle/input/datasets/shohanurrahman7/cbma-v3/best_cbm_model.pth"

IMAGE_SIZE = 128
NUM_CONCEPTS = 7
CFG_SCALE = 3.5
N_EVAL = 200        # ↑ from 100 for tighter CIs on statistical tests
N_PEARSON = 20      # ↑ from 15 for more power per step
N_PEARSON_STEPS = 10
DDIM_STEPS = 50

CONCEPT_NAMES = ['pigment_network', 'blue_whitish_veil', 'vascular_structures',
                  'pigmentation', 'streaks', 'dots_and_globules', 'regression_structures']
SHORT = ['PN', 'BWV', 'VS', 'Pig', 'Str', 'D/G', 'Reg']

MEL_VEC = torch.FloatTensor([0.8292, 0.7781, 0.9926, 0.8918, 0.4585, 0.9256, 0.8769])
BEN_VEC = torch.FloatTensor([0.0800, 0.0002, 0.0047, 0.0055, 0.0018, 0.1501, 0.0003])

def save_fig_both(fig, name, dpi=300):
    fig.savefig(f'{name}.png', dpi=dpi, bbox_inches='tight', facecolor='white')
    fig.savefig(f'{name}.pdf', bbox_inches='tight', facecolor='white')
    print(f"  Saved: {name}.png + {name}.pdf")


# ============================================================
# ARCHITECTURE + CBM-A (identical to NB4 — copy-paste, not import)
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
        self.tp=nn.Linear(td,oc); self.cp=nn.Linear(cd,oc)
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


def load_model(path, nc):
    ck=torch.load(path,map_location='cpu',weights_only=False)
    m=ConceptConditionedUNet128(IMAGE_SIZE,nc).to(device).eval()
    m.load_state_dict(ck.get('ema_state_dict',ck['model_state_dict']))
    print(f"  ✓ Loaded (epoch {ck.get('epoch','?')+1})")
    return m


# ── Load models ──
print("Loading models...")
cbm_a = load_cbm_a(CBM_A_PATH)
main_model = load_model(MAIN_CKPT, 7)
base_model = load_model(BASE_CKPT, 2)

cbm_norm = transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
cbm_rsz  = transforms.Resize((224,224))


# ── Generation + CCFS helpers ──
@torch.no_grad()
def generate_batch(model, cvec, n, cfg_scale=CFG_SCALE, steps=DDIM_STEPS, bs=16):
    ddim=DDIMScheduler(num_train_timesteps=1000,beta_schedule='linear',clip_sample=True)
    ddim.set_timesteps(steps); all_imgs=[]
    for start in range(0,n,bs):
        b=min(bs,n-start); imgs=torch.randn(b,3,IMAGE_SIZE,IMAGE_SIZE,device=device)
        cv=cvec.unsqueeze(0).repeat(b,1).to(device)
        for t in ddim.timesteps:
            ts=torch.full((b,),t,device=device,dtype=torch.long)
            eps_c=model(imgs,ts,cv); eps_u=model(imgs,ts,torch.zeros_like(cv))
            imgs=ddim.step(eps_u+cfg_scale*(eps_c-eps_u),t,imgs).prev_sample
        all_imgs.append(((imgs.clamp(-1,1)+1)/2).cpu())
    return torch.cat(all_imgs,0)

@torch.no_grad()
def get_concept_preds(imgs):
    inp=torch.stack([cbm_norm(cbm_rsz(i)) for i in imgs]).to(device)
    _,cp,_=cbm_a(inp); return cp.cpu().numpy()

def per_image_ccfs(imgs, cvec):
    preds=get_concept_preds(imgs); cond=cvec.numpy()
    return np.array([np.dot(cond,p)/(np.linalg.norm(cond)*np.linalg.norm(p)+1e-8) for p in preds])


# ============================================================
# TEST 1 — Wilcoxon signed-rank test on CCFS distributions
# ============================================================

print(f"\n{'='*65}")
print(f"TEST 1 — Wilcoxon signed-rank test (n={N_EVAL} paired images/class)")
print(f"{'='*65}")

# Generate PAIRED samples: same random seed for both models
print("Generating paired samples (same noise seeds)...")
torch.manual_seed(42)
seeds_mel = [torch.randn(1,3,IMAGE_SIZE,IMAGE_SIZE) for _ in range(N_EVAL)]
torch.manual_seed(42)
seeds_ben = [torch.randn(1,3,IMAGE_SIZE,IMAGE_SIZE) for _ in range(N_EVAL)]

# Since we can't easily share exact noise across the generation pipeline,
# we generate independently but use the SAME random seed for both models
# to get comparable (not identical) noise trajectories.
torch.manual_seed(42)
main_mel = generate_batch(main_model, MEL_VEC, N_EVAL)
torch.manual_seed(42)
base_mel = generate_batch(base_model, torch.FloatTensor([0.,1.]), N_EVAL)
torch.manual_seed(123)
main_ben = generate_batch(main_model, BEN_VEC, N_EVAL)
torch.manual_seed(123)
base_ben = generate_batch(base_model, torch.FloatTensor([1.,0.]), N_EVAL)

print("Computing per-image CCFS scores...")
mm_scores = per_image_ccfs(list(main_mel), MEL_VEC)
bm_scores = per_image_ccfs(list(base_mel), MEL_VEC)
mb_scores = per_image_ccfs(list(main_ben), BEN_VEC)
bb_scores = per_image_ccfs(list(base_ben), BEN_VEC)

# Combined scores (mean of mel + ben per image-pair)
main_combined = np.concatenate([mm_scores, mb_scores])
base_combined = np.concatenate([bm_scores, bb_scores])

# Wilcoxon tests
stat_mel, p_mel = wilcoxon(mm_scores, bm_scores, alternative='greater')
stat_ben, p_ben = wilcoxon(mb_scores, bb_scores, alternative='greater')
stat_all, p_all = wilcoxon(main_combined, base_combined, alternative='greater')

# Effect size: rank-biserial r = 1 - (2*W)/(n*(n+1)) where W = sum of negative ranks
def rank_biserial(x, y):
    d = x - y
    d = d[d != 0]
    n = len(d)
    ranks = np.argsort(np.argsort(np.abs(d))) + 1
    W_neg = ranks[d < 0].sum()
    return 1 - (2 * W_neg) / (n * (n + 1) / 2)

rb_mel = rank_biserial(mm_scores, bm_scores)
rb_ben = rank_biserial(mb_scores, bb_scores)
rb_all = rank_biserial(main_combined, base_combined)

print(f"\n  {'Test':<25} {'W-stat':>10} {'p-value':>10} {'Effect (r_rb)':>14} {'Interpret':>12}")
print(f"  {'-'*73}")
for label, stat, p, rb in [
    ("CCFS mel (paired)", stat_mel, p_mel, rb_mel),
    ("CCFS ben (paired)", stat_ben, p_ben, rb_ben),
    ("CCFS combined", stat_all, p_all, rb_all),
]:
    sig = '***' if p<0.001 else '**' if p<0.01 else '*' if p<0.05 else 'n.s.'
    size = 'large' if abs(rb)>0.5 else 'medium' if abs(rb)>0.3 else 'small'
    print(f"  {label:<25} {stat:>10.0f} {p:>10.4f} {rb:>14.4f} {sig:>6} ({size})")


# ============================================================
# TEST 2 — Per-concept Pearson + Spearman + BH correction + CIs
# ============================================================

print(f"\n{'='*65}")
print(f"TEST 2 — Per-concept correlations with BH correction")
print(f"  (Pearson + Spearman, n={N_PEARSON_STEPS}×{N_PEARSON}={N_PEARSON_STEPS*N_PEARSON}/concept)")
print(f"{'='*65}")

steps = np.linspace(0, 1, N_PEARSON_STEPS)
pearson_results = {}
spearman_results = {}

for ci, cname in enumerate(CONCEPT_NAMES):
    input_vals = []; pred_vals = []
    for val in tqdm(steps, desc=SHORT[ci], leave=False):
        cvec = MEL_VEC.clone(); cvec[ci] = float(val)
        imgs = generate_batch(main_model, cvec, N_PEARSON, steps=30)
        preds = get_concept_preds(list(imgs))[:, ci]
        input_vals.extend([float(val)] * N_PEARSON)
        pred_vals.extend(preds.tolist())

    r_p, p_p = pearsonr(input_vals, pred_vals)
    r_s, p_s = spearmanr(input_vals, pred_vals)

    # Fisher z-transform CI for Pearson r
    n = len(input_vals)
    z = np.arctanh(r_p)
    se = 1 / np.sqrt(n - 3)
    z_lo, z_hi = z - 1.96*se, z + 1.96*se
    r_lo, r_hi = np.tanh(z_lo), np.tanh(z_hi)

    pearson_results[cname] = {'r': r_p, 'p': p_p, 'ci_lo': r_lo, 'ci_hi': r_hi}
    spearman_results[cname] = {'r': r_s, 'p': p_s}

# Benjamini-Hochberg FDR correction
p_vals_pearson = [pearson_results[c]['p'] for c in CONCEPT_NAMES]
p_vals_spearman = [spearman_results[c]['p'] for c in CONCEPT_NAMES]
_, p_bh_pearson, _, _ = multipletests(p_vals_pearson, method='fdr_bh')
_, p_bh_spearman, _, _ = multipletests(p_vals_spearman, method='fdr_bh')

print(f"\n  {'Concept':<24} {'Pearson r':>10} {'95% CI':>16} {'p(raw)':>10} {'p(BH)':>10} {'Spearman ρ':>11} {'p(BH)':>10}")
print(f"  {'-'*95}")
for i, c in enumerate(CONCEPT_NAMES):
    pr = pearson_results[c]; sr = spearman_results[c]
    sig_p = '***' if p_bh_pearson[i]<0.001 else '**' if p_bh_pearson[i]<0.01 else '*' if p_bh_pearson[i]<0.05 else 'n.s.'
    sig_s = '***' if p_bh_spearman[i]<0.001 else '**' if p_bh_spearman[i]<0.01 else '*' if p_bh_spearman[i]<0.05 else 'n.s.'
    print(f"  {c:<24} {pr['r']:>+10.4f} [{pr['ci_lo']:+.3f},{pr['ci_hi']:+.3f}] "
          f"{pr['p']:>10.4f} {p_bh_pearson[i]:>10.4f}{sig_p:>4} "
          f"{sr['r']:>+11.4f} {p_bh_spearman[i]:>10.4f}{sig_s:>4}")

# Count how many survive BH correction
n_sig_pearson = sum(1 for p in p_bh_pearson if p < 0.05)
n_sig_spearman = sum(1 for p in p_bh_spearman if p < 0.05)
print(f"\n  Concepts surviving BH correction: Pearson={n_sig_pearson}/7, Spearman={n_sig_spearman}/7")


# ============================================================
# SAVE ALL RESULTS
# ============================================================

results = {
    'wilcoxon_mel_stat': stat_mel, 'wilcoxon_mel_p': p_mel, 'wilcoxon_mel_rb': rb_mel,
    'wilcoxon_ben_stat': stat_ben, 'wilcoxon_ben_p': p_ben, 'wilcoxon_ben_rb': rb_ben,
    'wilcoxon_all_stat': stat_all, 'wilcoxon_all_p': p_all, 'wilcoxon_all_rb': rb_all,
}
for i, c in enumerate(CONCEPT_NAMES):
    results[f'pearson_r_{c}'] = pearson_results[c]['r']
    results[f'pearson_ci_lo_{c}'] = pearson_results[c]['ci_lo']
    results[f'pearson_ci_hi_{c}'] = pearson_results[c]['ci_hi']
    results[f'pearson_p_raw_{c}'] = pearson_results[c]['p']
    results[f'pearson_p_bh_{c}'] = p_bh_pearson[i]
    results[f'spearman_r_{c}'] = spearman_results[c]['r']
    results[f'spearman_p_bh_{c}'] = p_bh_spearman[i]

pd.DataFrame([results]).to_csv('nb4_stats_supplement.csv', index=False)
print("\n✓ nb4_stats_supplement.csv saved")


# ============================================================
# UPDATED FIGURE 3 — Pearson r with BH-corrected significance + CIs
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Panel A: Pearson r with error bars (CIs) and BH-corrected significance
rs = [pearson_results[c]['r'] for c in CONCEPT_NAMES]
ci_lo = [pearson_results[c]['ci_lo'] for c in CONCEPT_NAMES]
ci_hi = [pearson_results[c]['ci_hi'] for c in CONCEPT_NAMES]
errs = [[r-lo for r,lo in zip(rs,ci_lo)], [hi-r for r,hi in zip(rs,ci_hi)]]
colors_p = ['#2ecc71' if r > 0 else '#e74c3c' for r in rs]
bars = axes[0].bar(SHORT, rs, yerr=errs, capsize=5, color=colors_p, alpha=0.85)
axes[0].axhline(0, color='black', lw=0.8)
for bar, p_bh in zip(bars, p_bh_pearson):
    sig = '***' if p_bh<0.001 else '**' if p_bh<0.01 else '*' if p_bh<0.05 else 'n.s.'
    y = bar.get_height()
    axes[0].text(bar.get_x()+bar.get_width()/2, y + (0.02 if y>=0 else -0.05),
                 sig, ha='center', va='bottom', fontsize=11, fontweight='bold')
axes[0].set_ylabel("Pearson r", fontsize=12); axes[0].set_ylim(-0.5, 1.0)
axes[0].set_title("(a) Pearson r (BH-corrected, 95% CI)", fontweight='bold')
axes[0].grid(alpha=0.3, axis='y')

# Panel B: Spearman rho
rs_s = [spearman_results[c]['r'] for c in CONCEPT_NAMES]
colors_s = ['#3498db' if r > 0 else '#e74c3c' for r in rs_s]
bars_s = axes[1].bar(SHORT, rs_s, color=colors_s, alpha=0.85)
axes[1].axhline(0, color='black', lw=0.8)
for bar, p_bh in zip(bars_s, p_bh_spearman):
    sig = '***' if p_bh<0.001 else '**' if p_bh<0.01 else '*' if p_bh<0.05 else 'n.s.'
    y = bar.get_height()
    axes[1].text(bar.get_x()+bar.get_width()/2, y + (0.02 if y>=0 else -0.05),
                 sig, ha='center', va='bottom', fontsize=11, fontweight='bold')
axes[1].set_ylabel("Spearman ρ", fontsize=12); axes[1].set_ylim(-0.5, 1.0)
axes[1].set_title("(b) Spearman ρ (BH-corrected)", fontweight='bold')
axes[1].grid(alpha=0.3, axis='y')

plt.suptitle("Per-Concept Correlation — ClinDiffXAI v3", fontweight='bold', y=1.02)
plt.tight_layout()
save_fig_both(fig, 'fig3_pearson_spearman_bh_corrected')
plt.close()


# ============================================================
# FINAL SUMMARY
# ============================================================

print(f"\n{'='*65}")
print(f"STATISTICAL SUPPLEMENT COMPLETE")
print(f"{'='*65}")
print(f"\n  Wilcoxon (one-sided, H1: main > baseline):")
print(f"    Combined: p={p_all:.4f}, effect size r_rb={rb_all:.4f}")
print(f"    Melanoma: p={p_mel:.4f}, effect size r_rb={rb_mel:.4f}")
print(f"    Benign:   p={p_ben:.4f}, effect size r_rb={rb_ben:.4f}")
print(f"\n  BH-corrected significance:")
print(f"    Pearson: {n_sig_pearson}/7 concepts survive")
print(f"    Spearman: {n_sig_spearman}/7 concepts survive")
print(f"\n  Files saved:")
print(f"    nb4_stats_supplement.csv")
print(f"    fig3_pearson_spearman_bh_corrected.png + .pdf")
