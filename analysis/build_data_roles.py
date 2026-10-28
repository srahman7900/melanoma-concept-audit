"""Per-image list of data roles (written for the manuscript's Methods, "Data").
Inputs: figure_inputs/p2ainp (classifier labels and splits, real reference sets, audit-base metadata).
Output: results/data_roles.csv. Generator-training membership is known for every melanoma (all 584 were used)
and for the held-out benign reference set (not used); for other benign images it was not recorded per image."""
import os, json, numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AI = os.environ.get('DATA_ROOT', os.path.join(ROOT, 'figure_inputs')) + '/p2ainp'
z = np.load(f'{AI}/phase2/cls/cls_labels.npz', allow_pickle=True)
cls = pd.DataFrame({'image_name': z['image_name'], 'patient_id': z['group'], 'melanoma': z['target'].astype(int), 'classifier_split': z['split']})
meta = lambda p: json.load(open(f'{AI}/{p}'))['meta']
real_ben = set(meta('phase1d/sets_meta/real_ben.json')['image_name']); real_mel = set(meta('phase1d/sets_meta/real_mel.json')['image_name'])
base_ben = pd.Series(meta('phase2/sets_meta/rsweep_ben.json')['source_image']); base_mel = pd.Series(meta('phase2/sets_meta/rsweep_mel.json')['source_image'])
# each base vector appears once per knob and level; count the distinct base draws per image
nb = {}
for s, m in [(base_ben, 'phase2/sets_meta/rsweep_ben.json'), (base_mel, 'phase2/sets_meta/rsweep_mel.json')]:
    mm = meta(m); df = pd.DataFrame({'img': mm['source_image'], 'noise': mm['noise_id']}).drop_duplicates()
    for k, v in df.groupby('img').size().items(): nb[k] = int(v)
names = sorted(set(cls.image_name) | real_ben | real_mel | set(base_ben) | set(base_mel))
out = pd.DataFrame({'image_name': names}).merge(cls, on='image_name', how='left')
out['melanoma'] = out['melanoma'].fillna(out.image_name.isin(real_mel).astype(int)).astype(int)
out['classifier_split'] = out['classifier_split'].fillna('not in classifier data')
def gen(r):
    if r.melanoma == 1: return 'yes (all 584 melanomas, 5x)'
    if r.image_name in real_ben: return 'no (held-out benign reference set)'
    if r.image_name in set(base_ben): return 'yes (benign generator-training sample)'
    return 'not recorded per image (benign sample of 8,760, seed 42)'
out['generator_training'] = out.apply(gen, axis=1)
out['real_reference_set'] = np.where(out.image_name.isin(real_mel), 'real melanoma (584)', np.where(out.image_name.isin(real_ben), 'real benign held-out (2,000)', ''))
out['audit_base'] = [('benign base' if n in set(base_ben) else 'melanoma base' if n in set(base_mel) else '') for n in out.image_name]
out['audit_base_draws'] = [nb.get(n, 0) for n in out.image_name]
os.makedirs(os.path.join(ROOT, 'results'), exist_ok=True)
out.to_csv(os.path.join(ROOT, 'results', 'data_roles.csv'), index=False)
print(len(out), 'images'); print(out.groupby(['classifier_split', 'generator_training']).size().to_string())
print('audit bases:', (out.audit_base == 'benign base').sum(), 'benign images,', (out.audit_base == 'melanoma base').sum(), 'melanoma images,', out.audit_base_draws.sum(), 'draws')
