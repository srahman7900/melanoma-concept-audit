# Generator and evaluator training (earlier notebooks)

These are the notebooks that produced the models the audit cells load. They ran on Kaggle
before the audit phases and are released as they were run. Their file names and comments keep
the project's working name, ClinDiffXAI.

| File | What it produced | Hardware |
|---|---|---|
| `NB1_CBM_A_Training.py` | CBM-A, the concept bottleneck model, trained on the official Derm7pt split | T4 x2 |
| `NB2_pseudo_label_generation.ipynb` | `isic2020_pseudo_labels_v3.csv`: CBM-A pseudo-labels for all 33,126 ISIC 2020 images, and the p90/p10 prototype vectors. Cell outputs are kept as the printed record of that run. | T4 x2 |
| `NB3_main_model_128_GPU.py` | Concept-conditioned DDPM, epochs 1 to 50 | T4 x2 |
| `NB3_main_model_128_TPU.py` | The same model, epochs 51 to 150 | TPU v5e-8 |
| `NB3b_baseline_128_GPU.py` | Class-conditioned baseline DDPM, epochs 1 to 45 | T4 x2 |
| `NB3b_baseline_128_TPU.py` | The same baseline, epochs 46 to 150 | TPU v5e-8 |
| `NB4_earlier_evaluation.py` | An earlier evaluation of the two generators. **Not** part of this manuscript. | T4 x2 |
| `NB4_earlier_evaluation_stats.py` | Statistical supplement to that earlier evaluation. **Not** part of this manuscript. | T4 x2 |

## Which checkpoints this manuscript used

The audit used the **epoch-150 EMA weights of both models**:
`checkpoint_tpu_epoch150.pth` and `checkpoint_baseline_tpu_epoch150.pth`.
The Phase 1d log records this for every generated set.

The two `NB4_earlier_*` scripts load different files (`best_clindiffxai_v3_tpu.pth`, epoch 149,
and `best_clindiffxai_baseline_tpu.pth`, epoch 120), which were selected by the lowest
validation loss within a session. They are included for completeness only. No number in this
manuscript comes from them.

## Which CBM-A checkpoint

NB2 and the audit cells both read CBM-A from the Kaggle dataset `best-cbm-model-128`. On the
Derm7pt test set that checkpoint gives a mean concept AUC of 0.8224 and a diagnosis AUC of
0.5425, which are the values NB1 printed for its best epoch (epoch 48) and the values reported
in the manuscript. The NB3 scripts name a second Kaggle dataset, `cbma-v3`, for the same
file name. We did not compare the two files byte for byte.

## Session settings

The session variables in these files (`SESSION_END`, `RESUME_FROM`, `EVAL_EVERY`) hold the
values of one session. Training ran in several sessions, each resuming from the previous
checkpoint, because of Kaggle's session limits. The epoch ranges in the table above are the
ranges that were actually run.

## What is not here

The trained checkpoints and the pseudo-label CSV are not in this repository because of their
size. They are available from the corresponding author on request, including to reviewers.
