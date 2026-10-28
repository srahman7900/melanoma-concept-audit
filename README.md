# Testing concept-level audits of melanoma classifiers

Code, inputs and result tables for the manuscript "Testing concept-level audits of melanoma classifiers with diffusion counterfactuals and matched positive controls" (Md. Shohanoor Rahman et al.).

The code files keep the working name of the project, ClinDiffXAI, because they are released exactly as they were run. In the paper, the generator is called the concept-conditioned DDPM, and the class-conditioned model is called the baseline.

## Contents

| Folder | What it holds |
|---|---|
| `generator_training/` | The earlier notebooks that trained CBM-A, made the pseudo-labels, and trained the two diffusion models. See `generator_training/README.md` for which checkpoint each audit used. |
| `kaggle_cells/` | Single-cell Kaggle notebooks for the audit phases, as run. The GPU cells use two T4 GPUs, and each cell writes its tables, figures and logs. |
| `analysis/` | Offline scripts (no GPU): `make_figures.py` (main Figures 2 to 6), `make_fig_design.py` (Figure 1), `negative_control_scale.py` (Supplementary Table S22), `build_data_roles.py` (role of each image), `numbers_sheet.py`, `integrate_2e.py`, `pairs_cbma.py`, `check_isic_duplicates.py`, `make_supplementary.py`. |
| `figure_inputs/` | The run outputs that the offline scripts read: audit effects, validity results, TCAV results, per-classifier logits, classifier labels and splits, and the generated images shown in Figure 2a. |
| `results/` | Result tables, the numbers sheet, Supplementary Information, Supplementary Data 1, `data_roles.csv` and `rule_2e.txt` (rule 4 with its SHA-256 fingerprint). |
| `figures/` | Main figures (PDF); `figures/supplementary/` holds Supplementary Figures S1 to S15 (PDF). |

## Reproduction route

**1. From the released outputs (any computer, Python 3 with numpy, pandas, matplotlib and scipy).** These commands were tested from a clean copy of this repository:

```
FIG_DIR=figures_rebuilt python analysis/make_figures.py      # Figures 2 to 6
FIG_DIR=figures_rebuilt python analysis/make_fig_design.py   # Figure 1
python analysis/negative_control_scale.py                    # Supplementary Table S22
python analysis/build_data_roles.py                          # results/data_roles.csv
```

The scripts read `figure_inputs/` by default; set `DATA_ROOT` to use another folder with the same layout.

`requirements.txt` lists the package versions used.

**2. Re-running the experiments (Kaggle).** First run the notebooks in `generator_training/` (NB1, then NB2, then NB3 and NB3b), which train CBM-A, write the pseudo-labels, and train the two diffusion models. Then run the cells in `kaggle_cells/` in the order of the table below. Besides the public datasets (ISIC 2020, Derm7pt, ISIC 2018 Task 2), the audit cells need the checkpoints and the pseudo-label CSV that the first group produces; these files are available from the corresponding author on request, so that a reviewer does not have to repeat the training.

`make_supplementary.py` uses the authors' working folders (paths at the top of the file); it documents how Supplementary Information and Supplementary Data 1 were assembled from the tables in `results/`.

## Kaggle cells and the paper

| Cell | Part of the paper |
|---|---|
| `Phase1`, `Phase1b` | Evaluators (CBM-B, ISIC 2018 detector), dose responses on the prototypes, images from real concept vectors |
| `Phase1c` | Colour diagnosis and the choice of the colour-calibrated start noise |
| `Phase1d_GPU`, `Phase1d_TPU` | Final generator evaluation with colour-calibrated sampling (one GPU and one TPU version of the cell) |
| `Phase2`, `Phase2_v2` | Classifiers, quartile and negative controls, counterfactual audit (`_v2` is the revised cell) |
| `Phase2b` | TCAV, our implementation |
| `Phase2c` | First matched positive controls (CBM-A labels) |
| `Phase2d` | Further matched controls and TCAV with Captum |
| `Phase2e` | Matched controls labelled by CBM-B (rule 4) |
| `Phase2_archive`, `save_generated_data` | Archive of the inputs and generated images; no result changes |

## Data roles

The generator was trained on all 584 ISIC 2020 melanomas (each repeated five times) and 8,760 benign images sampled with seed 42, a melanoma-to-benign ratio of 1:3 (11,680 images). It therefore saw every melanoma in the classifier data, including the 85 test melanomas. A held-out set of 2,000 benign images was not used for generator training. The realistic audit bases are pseudo-label vectors of generator-training images. `results/data_roles.csv` lists, for each image, its patient, its classifier split, its generator-training status (known for all melanomas, the held-out benign set and the audit bases; not recorded per image for the other benign images) and its use as a real reference or audit base.

## Where to find the evidence for each claim

| Manuscript item | File |
|---|---|
| Audit effects of every classifier, base and knob | `figure_inputs/p2dout/audit/tables/TP2_counterfactual_effects_all.csv` |
| Matched-control decisions (CBM-A and CBM-B) | `figure_inputs/p2dout/audit/results/`, `figure_inputs/p2eout/audit/results/` |
| Rule 4 text and its fingerprint | `results/rule_2e.txt` |
| Negative-control scale analysis (Table S22) | `analysis/negative_control_scale.py`, `results/tables/T_S22a_*.csv`, `T_S22b_*.csv` |
| Role of each image | `results/data_roles.csv`, built by `analysis/build_data_roles.py` |
| Every number in the manuscript with its source file | `results/numbers/` |
| Supplementary Figures S1 to S15 and Tables S1 to S22 | `results/Supplementary_Information.pdf`, `results/Supplementary_Data_1.xlsx` |

## Decision rules

Each validity rule was written into the code before the run that tested it (see Methods, "Validity rules"). The fingerprint of rule 4 is
`919a511cf55ae6f5b249ba11c465ea25a460529fb775e49913c4d5a360ccf880`; the Phase 2e cell prints the same value when it starts, and `results/rule_2e.txt` holds the rule text. A fingerprint identifies the text of a rule; on its own, it does not prove when the text was written. The rules were not registered externally.

## Data

ISIC 2020, Derm7pt and ISIC 2018 Task 2 are public and are not redistributed here. The trained checkpoints, the CBM-A pseudo-labels, all generated images and the per-image scores are available from the corresponding author on request, including to reviewers.

## Contact

Md. Shohanoor Rahman, srahman7900@gmail.com
