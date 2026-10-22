"""Check the official ISIC 2020 duplicate list against patients and the classifier split.
Usage: python check_isic_duplicates.py ISIC_2020_Training_Duplicates.csv pseudo_labels.csv cls_labels.npz
(pseudo_labels.csv needs image_name, target and patient_id; cls_labels.npz needs image_name and split)"""
import sys
import numpy as np
import pandas as pd

dup = pd.read_csv(sys.argv[1])
meta = pd.read_csv(sys.argv[2])[['image_name', 'target', 'patient_id']]
z = np.load(sys.argv[3], allow_pickle=True)
cls = pd.DataFrame({'image_name': z['image_name'].astype(str), 'split': z['split'].astype(str)})
d = dup.merge(meta.add_suffix('_1'), on='image_name_1').merge(meta.add_suffix('_2'), on='image_name_2')
print('pairs:', len(dup), '| same patient:', int((d.patient_id_1 == d.patient_id_2).sum()),
      '| both melanoma:', int(((d.target_1 == 1) & (d.target_2 == 1)).sum()))
c = dup.merge(cls.add_suffix('_1'), on='image_name_1').merge(cls.add_suffix('_2'), on='image_name_2')
print('pairs inside the classifier data:', len(c), '| across splits:', int((c.split_1 != c.split_2).sum()))
