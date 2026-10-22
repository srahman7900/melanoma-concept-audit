# =============================================================================
# ClinDiffXAI Phase 2: archive cell (CPU only, about 3 to 8 min)
#
# Run this cell in the SAME live session, after the Phase 2 v2 cell.
# It does not use the GPUs and it does not change any result.
#
# It writes two files in /kaggle/working:
#   1. phase2_analysis_inputs.zip  (about 50 to 250 MB)
#      Everything needed to repeat or extend the Phase 2 analysis offline,
#      without a GPU: evaluator scores (CBM-A, CBM-B, ISIC 2018 detector,
#      darkness, CIELAB) for every audit set, the metadata of every set
#      (no full image sets), classifier data labels and splits, a slim copy of
#      the pseudo-label table, and small image subsets for paper figures.
#   2. phase2_models.zip  (about 1.3 GB, only if SAVE_MODELS = True)
#      The 10 trained classifiers. Needed only for later work on the same
#      classifiers (for example a TCAV comparison) after this session ends.
#
# Download both from the Output panel. Send phase2_analysis_inputs.zip back.
# Keep phase2_models.zip on your computer or in a private Kaggle dataset.
# =============================================================================
import os, glob, json, time, shutil, zipfile, io
import numpy as np
import torch

SAVE_MODELS = True          # set False if you do not want the 1.3 GB model archive
KEEP_FEATURES = False       # True keeps the CBM feature vectors (much larger zip)
N_RSWEEP_BASES = 8          # image subset for figures: first N bases of each realistic sweep
N_SWEEP_BASES = 4           # first N trajectories of each Phase 1d prototype sweep
N_EVAL_IMAGES = 24          # first N images of each Phase 1d eval / pool set and of the real test set

ROOT = os.environ.get('P1_WORK_ROOT', '/kaggle/working')
W2, W1D = os.path.join(ROOT, 'phase2'), os.path.join(ROOT, 'phase1d')
T0 = time.time()


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def mb(n):
    return f"{n / 2**20:.1f} MB"


for d in [W2, W1D]:
    if not os.path.isdir(d):
        raise SystemExit(f'{d} not found: this cell must run in the live session that ran Phase 1d and Phase 2.')

free = shutil.disk_usage(ROOT).free
log(f'free disk in {ROOT}: {mb(free)}')

out_zip = os.path.join(ROOT, 'phase2_analysis_inputs.zip')
if os.path.exists(out_zip):
    os.remove(out_zip)
manifest = {'created': time.strftime('%Y-%m-%d %H:%M:%S'), 'keep_features': KEEP_FEATURES, 'files': {}}
zf = zipfile.ZipFile(out_zip, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6)


def add_bytes(arcname, data):
    zf.writestr(arcname, data); manifest['files'][arcname] = len(data)


def add_npz(arcname, arrays):
    buf = io.BytesIO(); np.savez_compressed(buf, **arrays); add_bytes(arcname, buf.getvalue())


def add_file(arcname, path):
    zf.write(path, arcname); manifest['files'][arcname] = os.path.getsize(path)


# ---------------------------------------------------------------- 1. evaluator scores
n_sc = 0
for root, tag in [(W1D, 'phase1d'), (W2, 'phase2')]:
    for f in sorted(glob.glob(os.path.join(root, 'scores', '*.npz'))):
        try:
            d = dict(np.load(f, allow_pickle=False))
            if not KEEP_FEATURES:
                d = {k: v for k, v in d.items() if not (k.endswith('_feat') or k == 'incep')}
            add_npz(f'{tag}/scores/{os.path.basename(f)}', d); n_sc += 1
        except Exception as e:
            log(f'  could not read {f}: {e}')
    for f in sorted(glob.glob(os.path.join(root, 'scores2_fp32', '*.npz'))):
        add_file(f'{tag}/scores2_fp32/{os.path.basename(f)}', f); n_sc += 1
    for f in sorted(glob.glob(os.path.join(root, 'results', '*.json'))):
        add_file(f'{tag}/results/{os.path.basename(f)}', f)
    for f in sorted(glob.glob(os.path.join(root, 'models', 'pred_*.npz'))):
        add_file(f'{tag}/models/{os.path.basename(f)}', f)
    for f in [os.path.join(root, 'cfg.json'), os.path.join(root, 'code', 'cfg.json')]:
        if os.path.exists(f):
            add_file(f'{tag}/{os.path.relpath(f, root)}', f)
log(f'1. evaluator and classifier scores: {n_sc} files')


# ---------------------------------------------------------------- 2. set metadata and figure subsets
def meta_to_json(meta):
    out = {}
    for k, v in meta.items():
        if torch.is_tensor(v):
            v = v.tolist()
        elif isinstance(v, np.ndarray):
            v = v.tolist()
        out[k] = v
    return out


def info_to_json(info):
    try:
        return json.loads(json.dumps(info, default=str))
    except Exception:
        return {'note': 'info not serialisable'}


n_sets = 0
for root, tag in [(W1D, 'phase1d'), (W2, 'phase2')]:
    for f in sorted(glob.glob(os.path.join(root, 'sets', '*.pt'))):
        name = os.path.basename(f)[:-3]
        try:
            S = torch.load(f, map_location='cpu', weights_only=False)
        except Exception as e:
            log(f'  could not read {f}: {e}'); continue
        imgs = S['imgs']; meta = meta_to_json(S.get('meta', {}))
        add_bytes(f'{tag}/sets_meta/{name}.json',
                  json.dumps({'n': int(len(imgs)), 'shape': list(imgs.shape), 'meta': meta,
                              'info': info_to_json(S.get('info', {}))}).encode())
        # small image subsets for paper figures
        sel = None
        nid = np.asarray(meta.get('noise_id', []))
        if name.startswith('rsweep_') and len(nid) == len(imgs):
            sel = np.where(nid < N_RSWEEP_BASES)[0]
        elif name.startswith('sweep_') and len(nid) == len(imgs):
            sel = np.where(nid < N_SWEEP_BASES)[0]
        elif name.startswith(('eval_', 'pool_', 'cls_test')):
            sel = np.arange(min(N_EVAL_IMAGES, len(imgs)))
        if sel is not None and len(sel):
            add_npz(f'{tag}/fig_images/{name}.npz', {'index': sel.astype(np.int64), 'imgs': imgs[torch.from_numpy(sel)].numpy()})
        n_sets += 1
        del S, imgs
log(f'2. set metadata and figure subsets: {n_sets} sets')

# ---------------------------------------------------------------- 3. classifier data (labels, splits; no images)
fcls = os.path.join(W2, 'cls', 'data.pt')
if os.path.exists(fcls):
    dat = torch.load(fcls, map_location='cpu', weights_only=False)
    add_npz('phase2/cls/cls_labels.npz', {
        'target': np.asarray(dat['target']), 'concepts': np.asarray(dat['concepts']),
        'split': np.asarray(dat['split']).astype('U8'), 'image_name': np.asarray(dat['image_name']).astype('U32'),
        'group': np.asarray(dat['group']).astype('U32')})
    del dat
    log('3. classifier labels and splits saved')
else:
    log('3. classifier data not found (skipped)')

# ---------------------------------------------------------------- 4. slim pseudo-label table
try:
    import pandas as pd
    cfg2 = json.load(open(os.path.join(W2, 'outputs', 'cfg.json'))) if os.path.exists(os.path.join(W2, 'outputs', 'cfg.json')) else None
    csv = cfg2['paths']['pseudo_csv'] if cfg2 else None
    if csv and os.path.exists(csv):
        df = pd.read_csv(csv)
        keep = [c for c in df.columns if c in ('image_name', 'target', 'patient_id') or c.startswith('concept_')]
        meta_csv = cfg2['paths'].get('isic2020_meta')
        if 'patient_id' not in df.columns and meta_csv and os.path.exists(meta_csv):
            m = pd.read_csv(meta_csv)
            m = m[[c for c in ['image_name', 'patient_id'] if c in m.columns]].drop_duplicates('image_name')
            df = df.merge(m, on='image_name', how='left'); keep = keep + ['patient_id']
        add_bytes('phase2/pseudo_labels_slim.csv', df[list(dict.fromkeys(keep))].to_csv(index=False).encode())
        log(f'4. pseudo-label table: {len(df)} rows, columns {keep}')
    else:
        log('4. pseudo-label CSV not found (skipped)')
except Exception as e:
    log(f'4. pseudo-label table failed: {e}')

add_bytes('manifest.json', json.dumps(manifest, indent=1).encode())
zf.close()
log(f'phase2_analysis_inputs.zip: {mb(os.path.getsize(out_zip))}')

# ---------------------------------------------------------------- 5. trained classifiers (optional)
mz = os.path.join(ROOT, 'phase2_models.zip')
if SAVE_MODELS:
    pts = sorted(glob.glob(os.path.join(W2, 'models', '*.pt')))
    need = sum(os.path.getsize(p) for p in pts)
    free = shutil.disk_usage(ROOT).free
    if not pts:
        log('5. no model files found (skipped)')
    elif free < need + 2 * 2**30:
        log(f'5. not enough free disk for the model archive (need {mb(need)} + 2 GB, free {mb(free)}); skipped. '
            f'Delete /kaggle/working/phase2/chunks_gen or other large folders and run again.')
    else:
        if os.path.exists(mz):
            os.remove(mz)
        with zipfile.ZipFile(mz, 'w', compression=zipfile.ZIP_STORED) as z:
            for p in pts:
                z.write(p, f'models/{os.path.basename(p)}')
            for p in sorted(glob.glob(os.path.join(W2, 'results', 'train_*.json'))):
                z.write(p, f'models/{os.path.basename(p)}')
        log(f'5. phase2_models.zip: {len(pts)} classifiers, {mb(os.path.getsize(mz))}')
else:
    log('5. SAVE_MODELS = False: model archive not made')

print('\n' + '=' * 78)
print(f'ARCHIVE FINISHED in {(time.time() - T0) / 60:.1f} min')
for f in [out_zip, mz]:
    if os.path.exists(f):
        print(f'  {f}  {mb(os.path.getsize(f))}')
print('=' * 78)
print('Download phase2_analysis_inputs.zip and send it to Claude. Keep phase2_models.zip.')
