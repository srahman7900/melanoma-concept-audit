# Testing concept-level audits of melanoma classifiers

Code, inputs and result tables for the manuscript "Testing concept-level audits of melanoma classifiers with diffusion counterfactuals and matched positive controls" (Md. Shohanoor Rahman et al.).

The code files keep the working name of the project, ClinDiffXAI, because they are released exactly as they were run. In the paper, the generator is called the concept-conditioned DDPM, and the class-conditioned model is called the baseline.

## Contents

| Folder | What it holds |
|---|---|
| `generator_training/` | The earlier notebooks that trained CBM-A, made the pseudo-labels, and trained the two diffusion models. See `generator_training/README.md` for which checkpoint each audit used. |
| `kaggle_cells/` | Single-cell Kaggle notebooks for the audit phases, as run. The GPU cells use two T4 GPUs, and each cell writes its tables, figures and logs. |
| `analysis/` | Offline scripts (no GPU): `make_figures.py` (main Figures 2 to 7 and Supplementary Figure S16), `make_fig_design.py` (Figure 1), `negative_control_scale.py` (Supplementary Table S22), `build_data_roles.py` (role of each image), `numbers_sheet.py`, `integrate_2e.py`, `pairs_cbma.py`, `check_isic_duplicates.py`, `make_supplementary.py`. `analysis/phase3/` holds the post hoc and Phase 3 analyses (decomposition, trajectory filter, standardised margins, known-function controls, null checks). |
| `figure_inputs/` | The run outputs that the offline scripts read: audit effects, validity results, TCAV results, per-classifier logits (all controls and classifiers, including the 32 Phase 3 classifiers in `p3out/`), evaluator outputs on the realistic and prototype sweeps, classifier labels and splits, the Phase 3b PH2 and HAM10000 checks (`p3bout/`), and the generated images shown in Figure 2a. |
| `results/` | Result tables, the numbers sheet, Supplementary Information, Supplementary Data 1, `data_roles.csv`, `rule_2e.txt` (rule 4 with its SHA-256 fingerprint) and `protocol/` (rules 5 to 8 with their fingerprint, and the PH2 mapping and post-run reporting protocol frozen before the Phase 3 results were seen). |
| `figures/` | Main figures 1 to 7 (PDF); `figures/supplementary/` holds Supplementary Figures (PDF). |

## Reproduction route

**1. From the released outputs (any computer, Python 3 with numpy, pandas, matplotlib and scipy).** These commands were tested from a clean copy of this repository:

```
FIG_DIR=figures_rebuilt python analysis/make_figures.py      # Figures 2 to 7, Supplementary Figure S16
FIG_DIR=figures_rebuilt python analysis/make_fig_design.py   # Figure 1
python analysis/negative_control_scale.py                    # Supplementary Table S22
python analysis/build_data_roles.py                          # results/data_roles.csv
python analysis/phase3/b_decomp.py; python analysis/phase3/b2_contrib.py   # decomposition (Fig. 7, Tables S23a,b)
python analysis/phase3/a_preservation.py                     # trajectory filter, six primary classifiers (Table S24b)
python analysis/phase3/e1_margins.py                         # standardised margins with paired bootstrap (Table S29b)
python analysis/phase3/e3_known_function.py                  # known-function controls (Table S28)
A3_TABLES=results/phase3_rebuilt python analysis/phase3/null_checks.py   # null checks (Tables S27f, S27h, S29a); copy P3_T2_effects_all.csv there first
```

The Phase 3 tables `P3_*` come from `analysis/phase3/analysis3_with_reporting.py` run on the Phase 3 scores; the Kaggle cell writes the same tables (`figure_inputs/p3out/analysis3/`), and the offline run reproduced them exactly.

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
| `Phase3` | Rules 5 to 8: 17 more permuted-label classifiers (null distribution), 9 retrained ISIC 2020 melanoma classifiers, 6 HAM10000 classifiers, random heads on ImageNet features (as run; its PH2 step failed to parse the file) |
| `Phase3b` | PH2 with the corrected parser (rule 8) and the HAM10000 pixel-hash and near-duplicate checks; no training |

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
| Null distribution, retraining, HAM10000 (rules 5 to 7; Fig. 6) | `results/tables/P3_*.csv`, `figure_inputs/p3out/` |
| PH2 evaluator check (rule 8) and HAM10000 overlap checks | `results/tables/P3b_*.csv`, `figure_inputs/p3bout/` |
| Decomposition, trajectory filter, margins, known-function controls (post hoc) | `results/tables/T_decomposition_*`, `T_preservation_*`, `T_margins_*`, `T_control_margins_*`, `T_known_function_*`, `T_null_*` |
| Supplementary Figures and Tables | `results/Supplementary_Information.pdf`, `results/Supplementary_Data_1.xlsx` |

## Decision rules

Each validity rule was written into the code before the run that tested it (see Methods, "Validity rules"). The fingerprint of rule 4 is
`919a511cf55ae6f5b249ba11c465ea25a460529fb775e49913c4d5a360ccf880`; the Phase 2e cell prints the same value when it starts, and `results/rule_2e.txt` holds the rule text. The fingerprint of rules 5 to 8 is `287492fb0ca91d0dbaee3f839345472d28fec594ac208c02e6cdf87dad20e318`; the Phase 3 cell checks and prints it, and `results/protocol/rule_3.txt` holds the text. The PH2 mapping and the extra reporting of the Phase 3 results were frozen in `results/protocol/PH2_mapping_and_post_run_protocol.txt` (SHA-256 `35eeefd1e0ad476d2486e27d73508b0e67132bcb1fab0b20e84fa674b26518c1`) while the Phase 3 run was in progress, before its results were seen. A fingerprint identifies the text of a rule; on its own, it does not prove when the text was written. The rules were not registered externally.

## Data

ISIC 2020, Derm7pt, ISIC 2018 Task 2 and HAM10000 are public and are not redistributed here. PH2 is available from its providers after registration and may not be redistributed; this repository holds only our derived scores and results. The trained checkpoints, the CBM-A pseudo-labels, all generated images and the per-image scores are available from the corresponding author on request, including to reviewers.

## Contact

Md. Shohanoor Rahman, srahman7900@gmail.com
