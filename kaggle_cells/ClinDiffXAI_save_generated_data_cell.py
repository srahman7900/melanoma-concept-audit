# =============================================================================
# ClinDiffXAI: save the generated audit images and the Phase 2c controls
# (CPU only; my estimate: 3 to 6 minutes; optional)
#
# Run in the live session after Phase 2c. It does not change any result.
# The generated images exist only in /kaggle/working. Keep them if you want to
# (a) share them in the paper's data-availability statement, or (b) run new
# analyses on the same images later without a GPU.
#
# Output (download from the Output panel):
#   clindiffxai_generated_images.zip   all Phase 1d and Phase 2 image sets
#       one compressed .npz per set: 'imgs' uint8 (N, 3, 128, 128), plus
#       <set>.json with the per-image metadata (concept vector, level, noise
#       id, colour target, source image) and the set information
#   phase2c_models.zip                 the 3 matched control classifiers
# =============================================================================
import os, io, glob, json, time, zipfile
import numpy as np
import torch

ROOT = '/kaggle/working'
T0 = time.time()


def mb(p):
    return f"{os.path.getsize(p) / 2**20:.1f} MB"


def to_json(x):
    try:
        return json.loads(json.dumps(x, default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o)))
    except Exception:
        return {'note': 'not serialisable'}


out = os.path.join(ROOT, 'clindiffxai_generated_images.zip')
sets = sorted(glob.glob(os.path.join(ROOT, 'phase1d', 'sets', '*.pt'))) + \
       sorted(glob.glob(os.path.join(ROOT, 'phase2', 'sets', 'rsweep_*.pt')))
if not sets:
    raise SystemExit('No image sets found: run this cell in the live session that ran Phase 1d and Phase 2.')
with zipfile.ZipFile(out, 'w', compression=zipfile.ZIP_STORED) as z:
    for p in sets:
        name = os.path.basename(p)[:-3]
        if name.startswith('real_'):
            continue   # real ISIC 2020 images: not ours to redistribute; the metadata names them
        S = torch.load(p, map_location='cpu', weights_only=False)
        buf = io.BytesIO(); np.savez_compressed(buf, imgs=S['imgs'].numpy()); z.writestr(f'{name}.npz', buf.getvalue())
        meta = {k: (v.tolist() if hasattr(v, 'tolist') else v) for k, v in S.get('meta', {}).items()}
        z.writestr(f'{name}.json', json.dumps({'n': int(len(S['imgs'])), 'meta': to_json(meta), 'info': to_json(S.get('info', {}))}))
        print(f"[{time.strftime('%H:%M:%S')}] {name}: {len(S['imgs'])} images", flush=True)
        del S
    z.writestr('README.txt', 'ClinDiffXAI generated audit images (128 x 128, uint8, channel first). One .npz per set '
               '(key imgs) and one .json with the per-image metadata. Sets: eval_* / pool_* (Phase 1d evaluation and '
               'colour control), sweep_* (prototype concept sweeps), cfg_sweep (guidance scale), rsweep_* (Phase 2 '
               'realistic-base sweeps). Real ISIC 2020 images are not included.\n')
print(f'{out}: {mb(out)}')

mz = os.path.join(ROOT, 'phase2c_models.zip')
pts = sorted(glob.glob(os.path.join(ROOT, 'phase2c', 'models', '*.pt')))
if pts:
    with zipfile.ZipFile(mz, 'w', compression=zipfile.ZIP_STORED) as z:
        for p in pts: z.write(p, f'models/{os.path.basename(p)}')
        for p in sorted(glob.glob(os.path.join(ROOT, 'phase2c', 'results', 'train_ctrl_m*.json'))): z.write(p, f'models/{os.path.basename(p)}')
    print(f'{mz}: {mb(mz)}')
print(f'finished in {(time.time() - T0) / 60:.1f} min. Download both files; then the session can end.')
