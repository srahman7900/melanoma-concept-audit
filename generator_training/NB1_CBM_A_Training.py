# ============================================================
# NOTEBOOK 1 — CBM-A Training from Scratch (Derm7pt)
# Single-cell version. Simple local Kaggle save (no Drive).
# Platform: Kaggle T4×2 GPU
#
# v1.1 FIXES APPLIED:
#   - Uses OFFICIAL Derm7pt train/valid/test split (Kawahara et al. 2019)
#     instead of a random stratified split — directly comparable to
#     published benchmarks
#   - Model selection now uses combined_score = 0.5*diag_AUC + 0.5*concept_AUC
#     (was diag_AUC alone, which let concept-head quality silently decay
#     even as the checkpoint kept getting saved as "best")
#   - Patience-based early stopping (was a fixed 60 epochs regardless of
#     plateau)
#   - Final numbers reported on a TRUE held-out TEST set, never touched
#     during training or model selection (was previously re-evaluating on
#     the same validation set used for checkpoint selection — mild leakage)
#
# ⚠ BEFORE RUNNING — VERIFY THESE 4 THINGS:
#   1. DERM7PT_CSV / DERM7PT_IMG_DIR / split index CSV paths below match
#      your actual Kaggle dataset
#   2. CONCEPT_COLS below match your CSV's actual column names
#      (printed automatically — check the output)
#   3. DIAGNOSIS unique values printed at runtime — verify melanoma/benign
#      substring mapping is correct
#   4. Official split index files — the code prints their structure and
#      tries case_num matching first, positional-index fallback second;
#      it will raise a clear error if neither covers ≥80% of rows
#
# TO SWITCH SESSIONS (unlikely needed — this dataset is small and trains
# in well under an hour with early stopping):
#   change SESSION_END / RESUME_FROM in CONFIG section below.
# ============================================================

# ── INSTALL ────────────────────────────────────────────────────
import subprocess, sys
for pkg in ["timm==1.0.15", "scikit-learn"]:
    subprocess.run([sys.executable, '-m', 'pip', 'install', pkg, '-q'], check=False)

import os, time, warnings, json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.models as tv_models
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from tqdm import tqdm
warnings.filterwarnings('ignore')

device = torch.device('cuda')
torch.backends.cudnn.benchmark = True
print(f"GPU: {torch.cuda.get_device_name(0)} | Count: {torch.cuda.device_count()}")


# ============================================================
# CONFIG — edit these for your dataset / session plan
# ============================================================

# ── ⚠ VERIFY THIS PATH ──
DERM7PT_CSV     = "/kaggle/input/datasets/shohanurrahman7/derm7pt/derm7pt_clean.zip/meta/meta.csv"            # ⚠ CHECK
DERM7PT_IMG_DIR = "/kaggle/input/datasets/shohanurrahman7/derm7pt/derm7pt_clean.zip/images"                    # ⚠ CHECK
DERM7PT_TRAIN_IDX = "/kaggle/input/datasets/shohanurrahman7/derm7pt/derm7pt_clean.zip/meta/train_indexes.csv"  # ⚠ CHECK
DERM7PT_VALID_IDX = "/kaggle/input/datasets/shohanurrahman7/derm7pt/derm7pt_clean.zip/meta/valid_indexes.csv"  # ⚠ CHECK
DERM7PT_TEST_IDX  = "/kaggle/input/datasets/shohanurrahman7/derm7pt/derm7pt_clean.zip/meta/test_indexes.csv"   # ⚠ CHECK

IMAGE_SIZE  = 224          # EfficientNetV2-S standard input (NOT diffusion's 128)
BATCH_SIZE  = 32
NUM_WORKERS = 4
LR          = 1e-4
WEIGHT_DECAY= 0.01
DIAG_LOSS_WEIGHT = 1.0     # balance concept loss vs diagnosis loss

SESSION_END  = 60          # ceiling — early stopping will likely trigger sooner
RESUME_FROM  = None        # set to checkpoint path to resume, e.g.:
                            # "/kaggle/input/datasets/.../checkpoint_epoch30.pth"

SAVE_EVERY = 10
SEED = 42

EARLY_STOP_PATIENCE = 15   # stop if combined score doesn't improve for N epochs
EARLY_STOP_MIN_DELTA = 1e-4

# ── ⚠ VERIFY these column names match your CSV exactly ──
# Standard Derm7pt release columns (Kawahara et al. 2019):
CONCEPT_COLS_RAW = {
    'pigment_network':       'pigment_network',
    'blue_whitish_veil':     'blue_whitish_veil',
    'vascular_structures':   'vascular_structures',
    'pigmentation':          'pigmentation',
    'streaks':               'streaks',
    'dots_and_globules':     'dots_and_globules',
    'regression_structures': 'regression_structures',
}
DIAGNOSIS_COL = 'diagnosis'   # ⚠ VERIFY
IMAGE_COL     = 'derm'        # ⚠ VERIFY — column holding image filename

CONCEPT_NAMES = list(CONCEPT_COLS_RAW.keys())
NUM_CONCEPTS  = len(CONCEPT_NAMES)

torch.manual_seed(SEED)
np.random.seed(SEED)

print(f"\nConfig: IMAGE_SIZE={IMAGE_SIZE} | epochs target={SESSION_END} | "
      f"resume={'yes' if RESUME_FROM else 'no'}")


# ============================================================
# DATA — load, inspect, map labels
# ============================================================

if not os.path.exists(DERM7PT_CSV):
    raise FileNotFoundError(
        f"Derm7pt CSV not found at {DERM7PT_CSV}\n"
        f"⚠ Update DERM7PT_CSV above to your actual Kaggle dataset path."
    )

raw_df = pd.read_csv(DERM7PT_CSV)
raw_df['_orig_idx'] = raw_df.index   # preserved through later row filtering, for split matching
print(f"\nLoaded {len(raw_df)} rows. Columns found:")
print(list(raw_df.columns))

print(f"\n⚠ VERIFY: unique values in '{DIAGNOSIS_COL}' column (check mapping below matches):")
if DIAGNOSIS_COL in raw_df.columns:
    print(raw_df[DIAGNOSIS_COL].value_counts())
else:
    raise KeyError(f"Column '{DIAGNOSIS_COL}' not found. Check CSV columns printed above "
                    f"and update DIAGNOSIS_COL.")

# ── 7-point checklist scoring (Argenziano et al. 1998) ──
# Each raw category mapped to a normalized concept-strength in [0, 0.5, 1.0].
# This is the STANDARD scheme — but raw category STRING VALUES vary by release.
# The mapping below handles the most common Derm7pt string encodings; if your
# CSV uses different strings, this will silently produce all-zero columns —
# CHECK THE PRINTED VALUE COUNTS BELOW before trusting training.

def score_pigment_network(v):
    v = str(v).lower()
    if 'atypical' in v: return 1.0
    if 'typical' in v:  return 0.5
    return 0.0

def score_blue_veil(v):
    v = str(v).lower()
    return 1.0 if 'present' in v else 0.0

def score_vascular(v):
    v = str(v).lower()
    if 'within regression' in v or 'absent' in v: return 0.0
    return 1.0   # any other vascular pattern present

def score_binary_irregular(v):
    # used for pigmentation, streaks, dots_and_globules
    v = str(v).lower()
    if 'irregular' in v: return 1.0
    if 'regular' in v:   return 0.5
    return 0.0

def score_regression(v):
    v = str(v).lower()
    # Actual Derm7pt categories: 'absent', 'blue areas', 'combinations', 'white areas'
    # (NOT literally 'present' — any non-absent subtype counts as present)
    return 0.0 if 'absent' in v else 1.0

SCORERS = {
    'pigment_network':       score_pigment_network,
    'blue_whitish_veil':     score_blue_veil,
    'vascular_structures':   score_vascular,
    'pigmentation':          score_binary_irregular,
    'streaks':               score_binary_irregular,
    'dots_and_globules':     score_binary_irregular,
    'regression_structures': score_regression,
}

processed_rows = []
missing_cols = []
for concept, raw_col in CONCEPT_COLS_RAW.items():
    if raw_col not in raw_df.columns:
        missing_cols.append(raw_col)

if missing_cols:
    raise KeyError(
        f"⚠ These raw concept columns were not found: {missing_cols}\n"
        f"Available columns: {list(raw_df.columns)}\n"
        f"Update CONCEPT_COLS_RAW above to match your CSV."
    )

df = raw_df.copy()
for concept, raw_col in CONCEPT_COLS_RAW.items():
    df[f'concept_{concept}'] = df[raw_col].apply(SCORERS[concept])

print("\n⚠ VERIFY: raw category values found per concept column (sanity check the SCORERS logic above against these):")
for concept, raw_col in CONCEPT_COLS_RAW.items():
    print(f"  {raw_col:<24}: {sorted(raw_df[raw_col].astype(str).str.lower().unique().tolist())}")

print("\n⚠ VERIFY: concept score distributions AFTER mapping (each must show >1 distinct value):")
degenerate = []
for concept in CONCEPT_NAMES:
    col = f'concept_{concept}'
    vc = df[col].value_counts()
    print(f"  {concept:<24} value_counts: {dict(vc)}")
    if len(vc) < 2:
        degenerate.append(concept)

if degenerate:
    raise ValueError(
        f"⚠ DEGENERATE CONCEPT COLUMN(S) DETECTED: {degenerate}\n"
        f"These produced only one distinct value across the entire dataset — the\n"
        f"SCORERS function for this concept does not match the actual category\n"
        f"strings printed above. Fix the relevant score_* function before proceeding.\n"
        f"Training would otherwise silently produce a useless concept head."
    )

# ── AUTOMATIC SELF-VALIDATION against the dataset's own official score ──
# Derm7pt provides 'seven_point_score', the ground-truth weighted sum per the
# checklist (Argenziano et al. 1998): PN=2, BWV=2, VS=2, PIG=1, STR=1, DG=1, RS=1.
# If our derived score doesn't correlate strongly with theirs, a SCORERS mapping
# is wrong — this check catches that class of bug automatically, rather than
# relying on manually re-deriving every category string from memory.
if 'seven_point_score' in df.columns:
    POINT_WEIGHTS = {
        'pigment_network': 2, 'blue_whitish_veil': 2, 'vascular_structures': 2,
        'pigmentation': 1, 'streaks': 1, 'dots_and_globules': 1, 'regression_structures': 1,
    }
    df['derived_7pt_score'] = sum(
        df[f'concept_{c}'] * w for c, w in POINT_WEIGHTS.items()
    )
    score_corr = df['derived_7pt_score'].corr(df['seven_point_score'])
    print(f"\n⚠ VALIDATION: correlation between derived score and official "
          f"'seven_point_score' column: {score_corr:.4f}")
    if score_corr < 0.70:
        raise ValueError(
            f"⚠ CONCEPT SCORING VALIDATION FAILED (correlation={score_corr:.4f} < 0.70).\n"
            f"At least one SCORERS function does not match this dataset's actual\n"
            f"category encoding. Compare the raw category values printed above against\n"
            f"each score_* function and fix before training. Stopping here to avoid\n"
            f"wasting GPU time training on incorrect concept labels."
        )
    print(f"  ✓ Validation passed — derived scores are consistent with the dataset's "
          f"own official checklist score.")
else:
    print("\n⚠ No 'seven_point_score' column found — skipping automatic validation. "
          "Manually verify the value_counts above are sensible before training.")

# ── Diagnosis binarization: melanoma vs benign ──
# ⚠ VERIFY against the value_counts printed above for DIAGNOSIS_COL.
# This substring match assumes diagnosis strings contain 'melanoma' for
# melanoma cases (e.g. 'melanoma', 'melanoma (in situ)', etc.) and do NOT
# contain 'melanoma' for benign cases (e.g. 'nevus', 'seborrheic keratosis').
df['target'] = df[DIAGNOSIS_COL].astype(str).str.lower().str.contains('melanoma').astype(int)
print(f"\nDiagnosis binarization result: {df['target'].value_counts().to_dict()} "
      f"(1=melanoma, 0=benign)")
if df['target'].sum() == 0 or df['target'].sum() == len(df):
    raise ValueError(
        "⚠ Diagnosis binarization produced all-0 or all-1 labels — "
        "the substring mapping is wrong for your CSV. Check the printed "
        "diagnosis value_counts above and fix the mapping logic."
    )

# ── Resolve image paths ──
def resolve_path(name):
    name = str(name)
    for ext in ['', '.jpg', '.jpeg', '.png']:
        p = os.path.join(DERM7PT_IMG_DIR, name + ext)
        if os.path.exists(p):
            return p
    return None

df['image_path'] = df[IMAGE_COL].apply(resolve_path)
found = df['image_path'].notna().sum()
print(f"\nImage resolution: {found}/{len(df)} found at {DERM7PT_IMG_DIR}")
if found == 0:
    raise FileNotFoundError(
        f"⚠ No images found. Check DERM7PT_IMG_DIR and IMAGE_COL above."
    )
df = df[df['image_path'].notna()].reset_index(drop=True)

# ── Official Derm7pt train/valid/test split (Kawahara et al. 2019) ──
# Using the dataset's own official split — directly comparable to published
# Derm7pt benchmarks, rather than an ad hoc random split.

def load_official_split(df, raw_df, train_path, valid_path, test_path):
    for p, label in [(train_path, 'train'), (valid_path, 'valid'), (test_path, 'test')]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"⚠ {label}_indexes.csv not found at {p}")

    train_idx_df = pd.read_csv(train_path)
    valid_idx_df = pd.read_csv(valid_path)
    test_idx_df  = pd.read_csv(test_path)

    print(f"\n⚠ VERIFY: official split index file structure:")
    for name, idf in [('train', train_idx_df), ('valid', valid_idx_df), ('test', test_idx_df)]:
        print(f"  {name}_indexes.csv: shape={idf.shape}, columns={list(idf.columns)}, "
              f"head={idf.iloc[:3, 0].tolist()}")

    def extract_indexes(idx_df):
        return idx_df[idx_df.columns[0]].astype(int).values

    train_idx = extract_indexes(train_idx_df)
    valid_idx = extract_indexes(valid_idx_df)
    test_idx  = extract_indexes(test_idx_df)

    # Strategy A: match against the explicit case_num identifier column
    tr_a = df[df['case_num'].isin(train_idx)]
    vl_a = df[df['case_num'].isin(valid_idx)]
    te_a = df[df['case_num'].isin(test_idx)]
    matched_a = len(tr_a) + len(vl_a) + len(te_a)

    # Strategy B: positional index into the ORIGINAL (pre-filtering) row order,
    # via the _orig_idx column preserved before any image-path filtering
    max_idx = max(train_idx.max(), valid_idx.max(), test_idx.max())
    if max_idx < len(raw_df):
        tr_b = df[df['_orig_idx'].isin(train_idx)]
        vl_b = df[df['_orig_idx'].isin(valid_idx)]
        te_b = df[df['_orig_idx'].isin(test_idx)]
        matched_b = len(tr_b) + len(vl_b) + len(te_b)
    else:
        tr_b, vl_b, te_b, matched_b = None, None, None, -1

    print(f"\n  Match via case_num:        {matched_a}/{len(df)} rows assigned")
    print(f"  Match via positional index: "
          f"{matched_b if matched_b >= 0 else 'N/A (indexes out of range)'}/{len(df)} rows assigned")

    if matched_a >= 0.8 * len(df):
        print(f"  → Using case_num matching "
              f"(train={len(tr_a)}, valid={len(vl_a)}, test={len(te_a)})")
        return tr_a.reset_index(drop=True), vl_a.reset_index(drop=True), te_a.reset_index(drop=True)
    elif matched_b >= 0.8 * len(df):
        print(f"  → Using positional-index matching "
              f"(train={len(tr_b)}, valid={len(vl_b)}, test={len(te_b)})")
        return tr_b.reset_index(drop=True), vl_b.reset_index(drop=True), te_b.reset_index(drop=True)
    else:
        raise ValueError(
            f"⚠ OFFICIAL SPLIT MATCHING FAILED — neither case_num nor positional-index\n"
            f"matching covered ≥80% of rows (case_num={matched_a}, positional={matched_b},\n"
            f"total={len(df)}). Inspect the printed index file structure above against the\n"
            f"'case_num' values and adjust load_official_split() matching logic.\n"
            f"A random stratified split would not be directly comparable to published\n"
            f"Derm7pt benchmarks, so this is a hard stop rather than a silent fallback."
        )

train_df, val_df, test_df = load_official_split(
    df, raw_df, DERM7PT_TRAIN_IDX, DERM7PT_VALID_IDX, DERM7PT_TEST_IDX
)

# Sanity: confirm no case appears in more than one split
overlap_tv = set(train_df['case_num']) & set(val_df['case_num'])
overlap_tt = set(train_df['case_num']) & set(test_df['case_num'])
overlap_vt = set(val_df['case_num']) & set(test_df['case_num'])
if overlap_tv or overlap_tt or overlap_vt:
    raise ValueError(
        f"⚠ SPLIT OVERLAP DETECTED — train∩valid={len(overlap_tv)}, "
        f"train∩test={len(overlap_tt)}, valid∩test={len(overlap_vt)}.\n"
        f"Fix load_official_split() before proceeding — this would silently leak "
        f"test images into training."
    )

print(f"\nOfficial split — Train: {len(train_df)} | Valid: {len(val_df)} | Test: {len(test_df)}")
print(f"(Valid set drives model selection + early stopping. "
      f"Test set is held out and reported only once, at the very end.)")


# ============================================================
# DATASET CLASS
# ============================================================

class Derm7ptDataset(Dataset):
    def __init__(self, df, image_size, augment=True):
        self.df = df.reset_index(drop=True)
        self.image_size = image_size
        norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        if augment:
            self.tfm = transforms.Compose([
                transforms.Resize((image_size + 16, image_size + 16)),
                transforms.RandomCrop(image_size),
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandomVerticalFlip(0.5),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.1, contrast=0.1),
                transforms.ToTensor(),
                norm,
            ])
        else:
            self.tfm = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                norm,
            ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        try:
            img = self.tfm(Image.open(row['image_path']).convert('RGB'))
        except Exception:
            img = torch.zeros(3, self.image_size, self.image_size)
        concepts = torch.FloatTensor(
            [row[f'concept_{c}'] for c in CONCEPT_NAMES]
        ).clamp(0., 1.)
        target = torch.tensor(int(row['target']), dtype=torch.long)
        return img, concepts, target


train_ds = Derm7ptDataset(train_df, IMAGE_SIZE, augment=True)
val_ds   = Derm7ptDataset(val_df,   IMAGE_SIZE, augment=False)
test_ds  = Derm7ptDataset(test_df,  IMAGE_SIZE, augment=False)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=NUM_WORKERS, pin_memory=True, drop_last=True,
                          persistent_workers=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=NUM_WORKERS, persistent_workers=True)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                         num_workers=NUM_WORKERS, persistent_workers=True)

print(f"Batches/epoch — train: {len(train_loader)} | val: {len(val_loader)} | test: {len(test_loader)}")

# Time estimate
t0 = time.time()
_ = next(iter(train_loader))
batch_time = time.time() - t0
est_min = batch_time * len(train_loader) / 60
print(f"~{est_min:.2f} min/epoch estimated → {SESSION_END} epochs ≈ {est_min*SESSION_END/60:.1f}h")


# ============================================================
# MODEL — CBM-A (MUST match v2 architecture exactly for downstream
# compatibility with NB2/NB3/NB4, which load this with strict=True)
# ============================================================

class CBM_A(nn.Module):
    def __init__(self, num_concepts=7):
        super().__init__()
        bb = tv_models.efficientnet_v2_s(
            weights=tv_models.EfficientNet_V2_S_Weights.DEFAULT
        )
        in_f = bb.classifier[1].in_features
        bb.classifier = nn.Identity()
        self.backbone = bb

        self.concept_feature = nn.Sequential(
            nn.Linear(in_f, 512), nn.LayerNorm(512), nn.GELU(), nn.Dropout(0.3)
        )
        self.concept_heads = nn.ModuleList([
            nn.Sequential(nn.Linear(512, 128), nn.GELU(), nn.Dropout(0.2), nn.Linear(128, 1))
            for _ in range(num_concepts)
        ])
        self.diagnosis_head = nn.Sequential(
            nn.Linear(num_concepts, 64), nn.GELU(), nn.Dropout(0.3), nn.Linear(64, 2)
        )

    def forward(self, x):
        f = self.backbone(x)
        if f.dim() > 2:
            f = f.mean([-2, -1])
        cf = self.concept_feature(f)
        cl = torch.cat([h(cf) for h in self.concept_heads], dim=1)   # concept logits
        cp = torch.sigmoid(cl)                                       # concept probs [0,1]
        diag_logits = self.diagnosis_head(cp)                        # diagnosis from concepts (CBM bottleneck)
        return cl, cp, diag_logits


model = CBM_A(num_concepts=NUM_CONCEPTS).to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f"\nModel params: {n_params:,}")

if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs via DataParallel")
    model = nn.DataParallel(model)

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
lr_sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=SESSION_END)


# ============================================================
# RESUME LOGIC
# ============================================================

start_epoch = 0
best_combined_score = 0.0
history = {'train_loss': [], 'val_loss': [], 'val_diag_auc': [],
           'val_concept_auc_mean': [], 'val_combined_score': []}

if RESUME_FROM is not None:
    if not os.path.exists(RESUME_FROM):
        raise FileNotFoundError(f"Checkpoint not found: {RESUME_FROM}")
    ck = torch.load(RESUME_FROM, map_location='cpu', weights_only=False)
    raw_model = model.module if hasattr(model, 'module') else model
    raw_model.load_state_dict(ck['model_state_dict'])
    optimizer.load_state_dict(ck['optimizer_state_dict'])
    lr_sched.load_state_dict(ck['lr_scheduler_state_dict'])
    start_epoch = ck['epoch'] + 1
    best_combined_score = ck.get('best_combined_score', 0.0)
    history = ck.get('history', history)
    print(f"✓ Resumed from epoch {ck['epoch']+1} | best_combined_score={best_combined_score:.4f}")
else:
    print("Fresh start.")


# ============================================================
# TRAIN / EVAL FUNCTIONS
# ============================================================

def run_epoch(model, loader, optimizer, train=True):
    model.train() if train else model.eval()
    losses = []
    all_concept_probs, all_concept_targets = [], []
    all_diag_probs, all_diag_targets = [], []

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for imgs, concepts, targets in tqdm(loader, desc='tr' if train else 'vl', leave=False):
            imgs = imgs.to(device, non_blocking=True)
            concepts = concepts.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            cl, cp, diag_logits = model(imgs)

            concept_loss = F.binary_cross_entropy_with_logits(cl, concepts)
            diag_loss = F.cross_entropy(diag_logits, targets)
            loss = concept_loss + DIAG_LOSS_WEIGHT * diag_loss

            if torch.isnan(loss):
                continue

            if train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            losses.append(loss.item())
            all_concept_probs.append(cp.detach().cpu())
            all_concept_targets.append(concepts.detach().cpu())
            all_diag_probs.append(torch.softmax(diag_logits, dim=1)[:, 1].detach().cpu())
            all_diag_targets.append(targets.detach().cpu())

    mean_loss = float(np.mean(losses)) if losses else float('nan')

    concept_probs = torch.cat(all_concept_probs).numpy()
    concept_targets_bin = (torch.cat(all_concept_targets).numpy() > 0).astype(int)
    diag_probs = torch.cat(all_diag_probs).numpy()
    diag_targets = torch.cat(all_diag_targets).numpy()

    concept_aucs = []
    for i in range(NUM_CONCEPTS):
        try:
            auc = roc_auc_score(concept_targets_bin[:, i], concept_probs[:, i])
        except ValueError:
            auc = float('nan')  # only one class present in this batch set
        concept_aucs.append(auc)

    try:
        diag_auc = roc_auc_score(diag_targets, diag_probs)
    except ValueError:
        diag_auc = float('nan')

    return mean_loss, concept_aucs, diag_auc


def save_ckpt(path, epoch, model, optimizer, lr_sched, best_combined_score, history):
    state = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
    tmp_path = path + '.tmp'
    torch.save({
        'epoch': epoch,
        'model_state_dict': state,
        'optimizer_state_dict': optimizer.state_dict(),
        'lr_scheduler_state_dict': lr_sched.state_dict(),
        'best_combined_score': best_combined_score,
        'history': history,
        'concept_names': CONCEPT_NAMES,
        'image_size': IMAGE_SIZE,
    }, tmp_path)
    os.replace(tmp_path, path)   # atomic write


# ============================================================
# TRAINING LOOP
# ============================================================

print(f"\n{'='*60}")
print(f"TRAINING CBM-A  ep {start_epoch+1} → {SESSION_END}  "
      f"(early stop patience={EARLY_STOP_PATIENCE})")
print(f"{'='*60}\n")

ep_times = []
epochs_without_improvement = 0
early_stopped_epoch = None

for epoch in range(start_epoch, SESSION_END):
    t0 = time.time()

    tr_loss, _, _ = run_epoch(model, train_loader, optimizer, train=True)
    vl_loss, vl_concept_aucs, vl_diag_auc = run_epoch(model, val_loader, optimizer, train=False)
    lr_sched.step()

    elapsed = time.time() - t0
    ep_times.append(elapsed)
    eta = ((SESSION_END - (epoch + 1)) * np.mean(ep_times[-5:])) / 3600

    mean_concept_auc = float(np.nanmean(vl_concept_aucs))
    # ── Combined selection criterion — balances diagnosis AND concept
    # quality, since downstream CCFS/MedFID/Pearson-r all depend on the
    # concept heads, not the diagnosis head. ──
    combined_score = 0.5 * vl_diag_auc + 0.5 * mean_concept_auc

    history['train_loss'].append(tr_loss)
    history['val_loss'].append(vl_loss)
    history['val_diag_auc'].append(vl_diag_auc)
    history['val_concept_auc_mean'].append(mean_concept_auc)
    history['val_combined_score'].append(combined_score)

    print(f"Ep {epoch+1:3d}/{SESSION_END} | tr={tr_loss:.4f} | vl={vl_loss:.4f} | "
          f"diag_AUC={vl_diag_auc:.4f} | concept_AUC={mean_concept_auc:.4f} | "
          f"combined={combined_score:.4f} | {elapsed:.0f}s | ETA={eta:.1f}h")

    # ── simple Kaggle local save (no Drive) — periodic checkpoint ──
    if (epoch + 1) % SAVE_EVERY == 0:
        ckpt_name = f'cbma_checkpoint_epoch{epoch+1}.pth'
        save_ckpt(ckpt_name, epoch, model, optimizer, lr_sched, best_combined_score, history)
        print(f"  → {ckpt_name} saved")

    # ── model selection on combined score, with early-stopping patience ──
    if combined_score > best_combined_score + EARLY_STOP_MIN_DELTA:
        best_combined_score = combined_score
        epochs_without_improvement = 0
        save_ckpt('best_cbm_model.pth', epoch, model, optimizer, lr_sched, best_combined_score, history)
        print(f"  ★ best model (combined={combined_score:.4f} = "
              f"0.5×diag[{vl_diag_auc:.4f}] + 0.5×concept[{mean_concept_auc:.4f}])")
    else:
        epochs_without_improvement += 1

    if epochs_without_improvement >= EARLY_STOP_PATIENCE:
        early_stopped_epoch = epoch + 1
        print(f"\n⚠ Early stopping at epoch {early_stopped_epoch} — no improvement in "
              f"combined score for {EARLY_STOP_PATIENCE} consecutive epochs.")
        break

print(f"\nTraining complete. Best combined score: {best_combined_score:.4f}"
      f"{f' (early stopped at epoch {early_stopped_epoch})' if early_stopped_epoch else ''}")


# ============================================================
# FINAL PER-CONCEPT AUC REPORT (on best model)
# ============================================================

best_ck = torch.load('best_cbm_model.pth', map_location='cpu', weights_only=False)
raw_model = model.module if hasattr(model, 'module') else model
raw_model.load_state_dict(best_ck['model_state_dict'])

# Validation-set numbers at the selected checkpoint — shown for transparency
# only; this is what model selection saw, NOT the number to report in a paper.
_, val_concept_aucs_final, val_diag_auc_final = run_epoch(model, val_loader, optimizer, train=False)

# TRUE held-out TEST set — never touched during training or model selection.
_, test_concept_aucs, test_diag_auc = run_epoch(model, test_loader, optimizer, train=False)

print(f"\n{'='*60}")
print(f"FINAL CBM-A EVALUATION (best checkpoint, epoch {best_ck['epoch']+1})")
print(f"{'='*60}")
print(f"  [Validation set — used for model selection; reference only]")
print(f"    Diagnosis AUC: {val_diag_auc_final:.4f} | "
      f"Mean concept AUC: {np.nanmean(val_concept_aucs_final):.4f}")

print(f"\n  [HELD-OUT TEST SET — never seen during training or selection]")
print(f"    Diagnosis AUC: {test_diag_auc:.4f}")
print(f"\n  Per-concept AUC (test set):")
for name, auc in zip(CONCEPT_NAMES, test_concept_aucs):
    print(f"    {name:<24}: {auc:.4f}")
print(f"  Mean concept AUC (test set): {np.nanmean(test_concept_aucs):.4f}")
print(f"{'='*60}")
print(f"\n  → Report the HELD-OUT TEST SET numbers in the paper, not the")
print(f"    validation numbers above (those were used for model selection).")


# ============================================================
# TRAINING CURVES
# ============================================================

# ============================================================
# FIGURE EXPORT HELPER — every figure saved as PNG (viewing) AND
# PDF (vector, CIBM submission quality). Reused in NB2/NB3/NB4.
# ============================================================

def save_fig_both(fig, name, dpi=300):
    """Save a matplotlib figure as both .png (raster, for quick viewing)
    and .pdf (vector, embedded fonts, journal-submission quality)."""
    fig.savefig(f'{name}.png', dpi=dpi, bbox_inches='tight', facecolor='white')
    fig.savefig(f'{name}.pdf', bbox_inches='tight', facecolor='white')
    print(f"  Saved: {name}.png + {name}.pdf")


fig, axes = plt.subplots(1, 4, figsize=(19, 4))
ep_range = range(1, len(history['train_loss']) + 1)

axes[0].plot(ep_range, history['train_loss'], 'b-', label='Train')
axes[0].plot(ep_range, history['val_loss'], 'r-', label='Val')
axes[0].set_title('Loss'); axes[0].set_xlabel('Epoch'); axes[0].legend(); axes[0].grid(alpha=0.3)

axes[1].plot(ep_range, history['val_diag_auc'], 'g-o', ms=4)
axes[1].axhline(0.70, color='r', ls='--', alpha=0.5, label='0.70 threshold')
axes[1].set_title('Diagnosis AUC (val)'); axes[1].set_xlabel('Epoch')
axes[1].legend(); axes[1].grid(alpha=0.3)

axes[2].plot(ep_range, history['val_concept_auc_mean'], 'm-o', ms=4)
axes[2].set_title('Mean Concept AUC (val)'); axes[2].set_xlabel('Epoch')
axes[2].grid(alpha=0.3)

axes[3].plot(ep_range, history['val_combined_score'], 'k-o', ms=4, label='Combined score')
axes[3].axhline(best_combined_score, color='green', ls='--', alpha=0.5,
                label=f'Best={best_combined_score:.4f}')
if early_stopped_epoch is not None:
    axes[3].axvline(early_stopped_epoch, color='red', ls=':', alpha=0.7,
                     label=f'Early stop (ep {early_stopped_epoch})')
axes[3].set_title('Selection Score\n(0.5×diag_AUC + 0.5×concept_AUC)')
axes[3].set_xlabel('Epoch'); axes[3].legend(fontsize=8); axes[3].grid(alpha=0.3)

plt.tight_layout()
save_fig_both(fig, 'nb1_cbma_training_curves')
plt.show()

print(f"\nFiles saved to /kaggle/working/:")
print(f"  best_cbm_model.pth  (selected via combined score, NOT diag_AUC alone)")
print(f"  cbma_checkpoint_epoch*.pth (every {SAVE_EVERY} epochs)")
print(f"  nb1_cbma_training_curves.png + .pdf")
print(f"\nDownload best_cbm_model.pth — this is what NB2/NB3/NB4 will use as CBM_A_PATH.")
