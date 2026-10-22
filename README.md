# Validated concept-level audit of melanoma classifiers

Code and result tables for the manuscript "Validating concept-level audits of melanoma classifiers with diffusion counterfactuals and matched positive controls" (Md. Shohanoor Rahman et al.).

The code files keep the working name of the project, ClinDiffXAI, because they are released exactly as they were run. In the paper, the generator is called the concept-conditioned DDPM, and the class-conditioned model is called the baseline.

## Contents

| Folder | What it holds |
|---|---|
| `kaggle_cells/` | Single-cell Kaggle notebooks for every phase (generator checks, colour calibration, classifier training, audit, TCAV, matched controls, Phase 2e independent-label controls). The GPU cells use two T4 GPUs, and each cell writes its tables, figures and logs. |
| `analysis/` | Offline scripts: `numbers_sheet.py` (every number in the paper with its source file), `integrate_2e.py` (Phase 2e decision, balance in both label sets, calibration check), `pairs_cbma.py` (exact rebuild of the matched pairs), `make_fig_design.py`, `make_figures.py`, `make_supplementary.py`, `check_isic_duplicates.py`. |
| `results/` | Tables behind the figures, the numbers sheet, Supplementary Data 1 and `rule_2e.txt` (the independent-label rule with its SHA-256 fingerprint). |
| `figures/` | Main figures (PDF); `figures/supplementary/` holds Supplementary Figures S1 to S15 (PDF). |

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
| `Phase2e` | Matched controls with CBM-B labels (rule 4) |
| `Phase2_archive`, `save_generated_data` | Archive of the inputs and generated images; no result changes |

## Decision rules

Each validity rule was written into the code before the run that tested it (see Methods, "Validity rules"). The fingerprint of rule 4 is
`919a511cf55ae6f5b249ba11c465ea25a460529fb775e49913c4d5a360ccf880`; the Phase 2e cell prints the same value when it starts.

## Data

ISIC 2020, Derm7pt and ISIC 2018 Task 2 are public and are not redistributed here. The generated images, evaluator and classifier scores and trained control classifiers are available from the corresponding author on reasonable request.

## Contact

Md. Shohanoor Rahman, srahman7900@gmail.com
