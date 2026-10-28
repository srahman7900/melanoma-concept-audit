# ClinDiffXAI Phase 3 report: empirical null distribution and retraining variability

Rules fixed before the run (SHA-256 of the text: 287492fb0ca91d0dbaee3f839345472d28fec594ac208c02e6cdf87dad20e318):

Phase 3 rules, fixed before the run. Data: the two realistic bases (64 benign and 64 melanoma base vectors), the four target-dominant knobs (pigment network, vascular structures, pigmentation, regression structures), logits clipped at +/-15, effect = mean logit change from knob level 0 to 1 over trajectories. Scale-free effect = effect divided by the SD of the classifier logits on the 969 real test images, mean of the two bases per classifier. Rule 5 (trained null): null classifiers are the Phase 2 permuted-label ResNet-50 and 17 new permuted-label classifiers (ResNet-50 seeds 2 to 10, EfficientNet-B0 seeds 1 to 5, ViT-B/16 seeds 1 to 3), trained with the Phase 2 recipe. For each knob, a melanoma finding is above null when the mean scale-free effect of the six primary melanoma classifiers is above the 95th percentile of the null classifiers. Reported for all four knobs; the same comparison is reported for the observational coefficients. Rule 6 (retraining): nine new melanoma classifiers (ResNet-50, EfficientNet-B0 and ViT-B/16, seeds 3 to 5) are added to the six primary ones. The pigmentation-sweep finding is stable if (a) its bootstrap 95% CI is above 0 on both realistic bases for at least 12 of the 15 classifiers and (b) pigmentation has the largest mean effect of the four knobs averaged over the 15 classifiers. Rule 7 (second training dataset): six melanoma classifiers trained on HAM10000 (melanoma against all other diagnoses; split by lesion; images whose names occur in ISIC 2020 removed; ResNet-50, EfficientNet-B0 and ViT-B/16, seeds 1 and 2; Phase 2 recipe). The pigmentation-sweep finding transfers if its bootstrap 95% CI is above 0 on both realistic bases in at least 5 of the 6 and pigmentation has the largest mean effect of the four knobs averaged over the 6. Rule 8 (independent expert criteria, descriptive): CBM-B and the ISIC 2018 detector are compared with the PH2 expert criteria (pigment network atypical against typical; dots and globules; streaks; regression areas; blue-whitish veil) by AUC with a 95% bootstrap CI; a criterion is confirmed on PH2 when the lower bound is above 0.5. If PH2 is not attached, rule 8 is not run. Descriptive only: the false-positive rate of the per-classifier Wilcoxon test in the null classifiers; 100 random linear heads per architecture on ImageNet features; off-target preservation filters for all melanoma classifiers. Earlier verdicts are not changed.


## Verdicts

- Rule 5, PN: NOT above null
- Rule 5, VS: above null
- Rule 5, Pig: above null
- Rule 5, Reg: above null
- Rule 6 (retraining): stable (pigmentation largest of 4 knobs in 12 of 15 classifiers, descriptive)
- Rule 7 (HAM10000): transfers
- Rule 8 (PH2, descriptive): parse failed: PH2 header not recognised: ['Name', 'Histological Diagnosis', 'Clinical Diagnosis', 'Asymmetry | Pigment Network | Dots/Globules | Streaks | Regression Areas | Blue-Whitish Veil', 'Colors']

### T1_classifiers

Classifiers. Test AUC for the melanoma label on the 969 real ISIC 2020 test images (bootstrap 95% CI, 500 resamples; for HAM10000 classifiers this is a cross-dataset test); own-split test AUC is on the test split of the training dataset; test logit SD (ISIC 2020) is the scale of the scale-free effects.

            classifier        group            arch  seed label  test AUC (melanoma)  AUC lo  AUC hi  test logit SD  best val AUC (own label)  own-split test AUC (melanoma)  epochs  minutes
       mel_resnet50_s1  mel_primary       ResNet-50     1   mel                0.889   0.855   0.916          3.346                     0.906                          0.889      14    5.428
       mel_resnet50_s2  mel_primary       ResNet-50     2   mel                0.876   0.835   0.910          3.391                     0.903                          0.876      15    5.593
          mel_effb0_s1  mel_primary EfficientNet-B0     1   mel                0.884   0.851   0.912          2.136                     0.907                          0.884       9    3.256
          mel_effb0_s2  mel_primary EfficientNet-B0     2   mel                0.870   0.838   0.903          2.809                     0.908                          0.870      14    3.250
         mel_vitb16_s1  mel_primary        ViT-B/16     1   mel                0.913   0.884   0.939          2.165                     0.916                          0.913       7    6.310
         mel_vitb16_s2  mel_primary        ViT-B/16     2   mel                0.903   0.874   0.931          2.524                     0.921                          0.903      10    8.599
     ctrl_pig_resnet50        other       ResNet-50     1   pig                0.599   0.529   0.662          3.468                     0.954                          0.599       9    1.661
     ctrl_bwv_resnet50        other       ResNet-50     1   bwv                0.605   0.538   0.672          4.341                     0.959                          0.604      13    2.297
    ctrl_perm_resnet50 null_trained       ResNet-50     1  perm                0.425   0.367   0.493          0.123                     0.530                          0.423       6    2.166
  mel_resnet50_derm7pt        other       ResNet-50     1   mel                0.673   0.612   0.734          0.946                     0.812                          0.672      11    0.382
 ctrl_perm_resnet50_s2 null_trained       ResNet-50     2  perm                0.439   0.370   0.507          1.584                     0.518                          0.437      15    5.676
 ctrl_perm_resnet50_s3 null_trained       ResNet-50     3  perm                0.445   0.371   0.516          1.816                     0.547                          0.445      15    5.424
 ctrl_perm_resnet50_s4 null_trained       ResNet-50     4  perm                0.440   0.376   0.502          0.553                     0.523                          0.439       8    3.134
 ctrl_perm_resnet50_s5 null_trained       ResNet-50     5  perm                0.451   0.398   0.509          1.300                     0.538                          0.452      14    5.052
 ctrl_perm_resnet50_s6 null_trained       ResNet-50     6  perm                0.473   0.412   0.533          1.968                     0.537                          0.472      15    5.340
 ctrl_perm_resnet50_s7 null_trained       ResNet-50     7  perm                0.454   0.398   0.522          0.676                     0.547                          0.453      10    3.623
 ctrl_perm_resnet50_s8 null_trained       ResNet-50     8  perm                0.419   0.345   0.485          0.482                     0.530                          0.420       9    3.162
 ctrl_perm_resnet50_s9 null_trained       ResNet-50     9  perm                0.559   0.498   0.623          0.707                     0.552                          0.560       9    3.257
ctrl_perm_resnet50_s10 null_trained       ResNet-50    10  perm                0.405   0.337   0.474          0.610                     0.488                          0.405       7    2.486
    ctrl_perm_effb0_s1 null_trained EfficientNet-B0     1  perm                0.369   0.305   0.426          1.427                     0.538                          0.369      14    4.610
    ctrl_perm_effb0_s2 null_trained EfficientNet-B0     2  perm                0.431   0.365   0.494          1.371                     0.508                          0.431      15    4.878
    ctrl_perm_effb0_s3 null_trained EfficientNet-B0     3  perm                0.466   0.395   0.526          0.196                     0.546                          0.467       6    1.405
    ctrl_perm_effb0_s4 null_trained EfficientNet-B0     4  perm                0.499   0.431   0.565          0.230                     0.560                          0.500       6    1.374
    ctrl_perm_effb0_s5 null_trained EfficientNet-B0     5  perm                0.480   0.416   0.544          0.568                     0.560                          0.480       9    2.104
   ctrl_perm_vitb16_s1 null_trained        ViT-B/16     1  perm                0.303   0.235   0.377          0.916                     0.591                          0.303      15   13.183
   ctrl_perm_vitb16_s2 null_trained        ViT-B/16     2  perm                0.593   0.512   0.668          0.051                     0.540                          0.593       9    7.604
   ctrl_perm_vitb16_s3 null_trained        ViT-B/16     3  perm                0.617   0.550   0.682          0.111                     0.519                          0.617       6    5.217
       mel_resnet50_s3  mel_retrain       ResNet-50     3   mel                0.892   0.858   0.920          2.554                     0.901                          0.892       9    3.277
       mel_resnet50_s4  mel_retrain       ResNet-50     4   mel                0.911   0.878   0.943          3.652                     0.890                          0.911      15    5.439
       mel_resnet50_s5  mel_retrain       ResNet-50     5   mel                0.889   0.849   0.924          3.651                     0.904                          0.890      14    4.938
          mel_effb0_s3  mel_retrain EfficientNet-B0     3   mel                0.869   0.833   0.903          2.359                     0.900                          0.868      11    2.512
          mel_effb0_s4  mel_retrain EfficientNet-B0     4   mel                0.868   0.823   0.906          2.454                     0.908                          0.869      12    2.879
          mel_effb0_s5  mel_retrain EfficientNet-B0     5   mel                0.862   0.812   0.900          2.920                     0.909                          0.862      15    3.394
         mel_vitb16_s3  mel_retrain        ViT-B/16     3   mel                0.913   0.887   0.939          3.096                     0.917                          0.913      15   12.316
         mel_vitb16_s4  mel_retrain        ViT-B/16     4   mel                0.909   0.881   0.935          3.187                     0.928                          0.909      15   12.966
         mel_vitb16_s5  mel_retrain        ViT-B/16     5   mel                0.892   0.859   0.920          2.948                     0.922                          0.892      14   11.617
   mel_ham_resnet50_s1      mel_ham       ResNet-50     1   mel                0.776   0.719   0.825          1.925                     0.921                          0.911       8    4.691
   mel_ham_resnet50_s2      mel_ham       ResNet-50     2   mel                0.669   0.597   0.733          2.371                     0.927                          0.918      10    5.917
      mel_ham_effb0_s1      mel_ham EfficientNet-B0     1   mel                0.659   0.585   0.723          1.964                     0.919                          0.929      12    5.913
      mel_ham_effb0_s2      mel_ham EfficientNet-B0     2   mel                0.652   0.570   0.721          1.889                     0.935                          0.920      11    5.640
     mel_ham_vitb16_s1      mel_ham        ViT-B/16     1   mel                0.757   0.699   0.808          2.018                     0.927                          0.928      11   14.236
     mel_ham_vitb16_s2      mel_ham        ViT-B/16     2   mel                0.672   0.604   0.745          2.245                     0.922                          0.923      14   18.970

### T3_null_distribution

Rule 5. Scale-free sweep effect (logit change from level 0 to 1 divided by the SD of the classifier logits on the real test images), mean of the two realistic bases per classifier. Null: classifiers trained on permuted melanoma labels. A finding is above null when the mean of the six primary melanoma classifiers exceeds the 95th percentile of the null classifiers.

knob  target-dominant  n null models  null mean  null SD  null max  null 95th pct  primary melanoma mean  primary melanoma min  all melanoma mean  empirical p (null >= melanoma mean)  n primary above null 95th  above null (rule 5)
  PN             True             18     -0.196    0.520     0.500          0.487                  0.028                -0.154             -0.001                                0.421                          0                False
  VS             True             18     -0.147    0.172     0.114          0.047                  0.109                -0.048              0.083                                0.105                          2                 True
 Pig             True             18      0.226    0.553     1.012          0.884                  1.013                 0.760              1.079                                0.053                          5                 True
 Reg             True             18     -0.261    0.414     0.524          0.432                  0.779                 0.418              0.801                                0.053                          5                 True
 BWV            False             18     -0.089    0.291     0.382          0.368                  0.607                 0.554              0.596                                0.053                          6                 True
 Str            False             18     -0.012    0.406     0.625          0.498                 -0.277                -1.392             -0.222                                0.789                          0                False
 D/G            False             18     -0.023    0.177     0.266          0.238                  0.302                 0.276              0.325                                0.053                          6                 True

### T4_test_false_positive_rate

Share of classifier x base x knob cells in which the per-classifier test of the paper (two-sided Wilcoxon signed-rank test of per-trajectory slopes, BH within the 7 knobs) is significant with a rising median slope. For null classifiers this is a false-positive rate of the per-model test, which treats trajectories, not models, as the unit.

       group  cells (model x base x knob)  BH p < 0.05 and rising  fraction  fraction, all 7 knobs
null_trained                          144                      19     0.132                  0.135
null_generic                         2400                     632     0.263                  0.238
 mel_primary                           48                      28     0.583                  0.631
 mel_retrain                           72                      39     0.542                  0.619
     mel_ham                           48                      18     0.375                  0.548

### T5_retraining

Rule 6. Sweep effects of all ISIC 2020 melanoma classifiers (six primary and the retrained seeds), mean of the two realistic bases. The pigmentation finding is stable if its bootstrap CI is above 0 on both bases for at least 80% of the classifiers and pigmentation has the largest mean effect of the four target-dominant knobs over all classifiers. Last column: number of classifiers in which each knob has the largest effect (descriptive).

knob  n classifiers  mean dlogit  SD between classifiers    min   max  n positive  n CI > 0 on both bases  Pig largest of 4 (n)
  PN             15       -0.038                   0.376 -0.619 0.480           8                       0                     0
  VS             15        0.219                   0.390 -0.251 1.245          10                       2                     0
 Pig             15        3.117                   0.792  1.904 4.276          15                      15                    12
 Reg             15        2.299                   0.680  0.905 3.542          15                      15                     3

Per classifier (mean of two bases):

knob               PN   Pig   Reg    VS
classifier                             
mel_effb0_s1     0.03  1.90  1.80  0.45
mel_effb0_s2     0.15  2.70  2.99  1.24
mel_effb0_s3     0.13  2.56  2.19  0.44
mel_effb0_s4     0.24  2.45  2.55  0.29
mel_effb0_s5     0.46  3.33  2.50 -0.25
mel_resnet50_s1  0.41  2.54  2.06  0.09
mel_resnet50_s2 -0.52  4.16  3.54 -0.06
mel_resnet50_s3 -0.08  2.49  2.67  0.69
mel_resnet50_s4 -0.62  4.14  3.40  0.24
mel_resnet50_s5 -0.55  3.19  2.22  0.15
mel_vitb16_s1    0.48  2.17  0.90 -0.10
mel_vitb16_s2   -0.23  3.12  1.74  0.10
mel_vitb16_s3   -0.44  4.11  1.74 -0.11
mel_vitb16_s4   -0.21  4.28  2.20  0.34
mel_vitb16_s5    0.18  3.62  2.00 -0.22

### T6_profile

Share of classifiers in which each knob has the largest scale-free effect (mean of the two realistic bases). A sweep that moves any image-sensitive model would be largest also for the null classifiers.

       group   n  PN largest  BWV largest  VS largest  Pig largest  Str largest  D/G largest  Reg largest  mean |effect| over 7 knobs
null_trained  18       0.278         0.00       0.000        0.611        0.056        0.056        0.000                       0.305
null_generic 300       0.217         0.02       0.143        0.310        0.110        0.050        0.150                       0.344
 mel_primary   6       0.000         0.00       0.000        0.833        0.000        0.000        0.167                       0.471
 mel_retrain   9       0.000         0.00       0.000        0.778        0.000        0.000        0.222                       0.489
     mel_ham   6       0.000         0.00       0.000        1.000        0.000        0.000        0.000                       0.623

### T7_observational_null

Observational attribution against the same null: ridge coefficient per SD of each predictor, divided by the SD of the classifier logits on the real test images.

concept  null mean  null 95th pct  primary melanoma mean  empirical p  above null
     PN     -0.070          0.082                  0.129        0.053        True
     VS     -0.069          0.097                  0.301        0.053        True
    Pig      0.010          0.097                  0.156        0.053        True
    Reg     -0.012          0.136                 -0.076        0.737       False

### T8_preservation

Sweep effects on trajectories with small off-target change. Off-target change: largest absolute change of the six other CBM-B outputs (level 1 minus level 0, SD units of real test images).

      group       base knob                    filter  n trajectories  target change (SD)  mean dlogit     lo     hi  n classifiers > 0  n classifiers
mel_primary rsweep_ben   PN                       all              64               1.717        0.330 -0.219  0.852                  5              6
mel_primary rsweep_ben   PN        off-target <= 2 SD              39               1.601        0.062 -0.509  0.607                  4              6
mel_primary rsweep_ben   PN target > every off-target              29               2.354        0.909  0.251  1.560                  5              6
mel_retrain rsweep_ben   PN                       all              64               1.717        0.153 -0.400  0.679                  5              9
mel_retrain rsweep_ben   PN        off-target <= 2 SD              39               1.601       -0.075 -0.708  0.563                  3              9
mel_retrain rsweep_ben   PN target > every off-target              29               2.354        0.873  0.136  1.646                  9              9
    mel_ham rsweep_ben   PN                       all              64               1.717        0.046 -0.381  0.484                  5              6
    mel_ham rsweep_ben   PN        off-target <= 2 SD              39               1.601       -0.110 -0.617  0.414                  3              6
    mel_ham rsweep_ben   PN target > every off-target              29               2.354        0.391 -0.369  1.233                  4              6
mel_primary rsweep_ben   VS                       all              64               1.574        0.157 -0.335  0.675                  2              6
mel_primary rsweep_ben   VS        off-target <= 2 SD              30               1.016        0.395 -0.259  1.119                  4              6
mel_primary rsweep_ben   VS target > every off-target              17               3.056        1.035  0.100  2.048                  5              6
mel_retrain rsweep_ben   VS                       all              64               1.574        0.012 -0.493  0.500                  4              9
mel_retrain rsweep_ben   VS        off-target <= 2 SD              30               1.016        0.284 -0.429  1.009                  8              9
mel_retrain rsweep_ben   VS target > every off-target              17               3.056        0.959  0.116  1.762                  9              9
    mel_ham rsweep_ben   VS                       all              64               1.574       -1.042 -1.572 -0.543                  0              6
    mel_ham rsweep_ben   VS        off-target <= 2 SD              30               1.016       -0.580 -1.084 -0.090                  0              6
    mel_ham rsweep_ben   VS target > every off-target              17               3.056       -0.006 -0.719  0.749                  3              6
mel_primary rsweep_ben  Pig                       all              64               3.488        2.774  2.306  3.240                  6              6
mel_primary rsweep_ben  Pig        off-target <= 2 SD               6               2.696        1.023  0.115  1.906                  5              6
mel_primary rsweep_ben  Pig target > every off-target              17               3.412        1.983  1.209  2.818                  6              6
mel_retrain rsweep_ben  Pig                       all              64               3.488        3.396  2.875  3.972                  9              9
mel_retrain rsweep_ben  Pig        off-target <= 2 SD               6               2.696        1.762  0.550  3.124                  8              9
mel_retrain rsweep_ben  Pig target > every off-target              17               3.412        2.751  1.826  3.720                  9              9
    mel_ham rsweep_ben  Pig                       all              64               3.488        3.955  3.420  4.487                  6              6
    mel_ham rsweep_ben  Pig        off-target <= 2 SD               6               2.696        1.888  1.191  2.658                  6              6
    mel_ham rsweep_ben  Pig target > every off-target              17               3.412        2.837  1.886  3.945                  6              6
mel_primary rsweep_ben  Reg                       all              64               3.623        2.094  1.579  2.582                  6              6
mel_primary rsweep_ben  Reg        off-target <= 2 SD              37               3.786        2.072  1.371  2.779                  6              6
mel_primary rsweep_ben  Reg target > every off-target              43               4.488        1.913  1.258  2.566                  6              6
mel_retrain rsweep_ben  Reg                       all              64               3.623        2.366  1.876  2.868                  9              9
mel_retrain rsweep_ben  Reg        off-target <= 2 SD              37               3.786        2.275  1.567  3.014                  9              9
mel_retrain rsweep_ben  Reg target > every off-target              43               4.488        2.016  1.345  2.665                  9              9
    mel_ham rsweep_ben  Reg                       all              64               3.623        0.228 -0.256  0.739                  4              6
    mel_ham rsweep_ben  Reg        off-target <= 2 SD              37               3.786        0.040 -0.574  0.664                  4              6
    mel_ham rsweep_ben  Reg target > every off-target              43               4.488       -0.146 -0.716  0.413                  4              6
mel_primary rsweep_mel   PN                       all              64               1.687       -0.224 -0.776  0.326                  2              6
mel_primary rsweep_mel   PN        off-target <= 2 SD              27               1.352       -0.712 -1.544  0.140                  2              6
mel_primary rsweep_mel   PN target > every off-target              22               2.158       -0.427 -1.362  0.550                  2              6
mel_retrain rsweep_mel   PN                       all              64               1.687       -0.351 -0.988  0.300                  1              9
mel_retrain rsweep_mel   PN        off-target <= 2 SD              27               1.352       -1.106 -2.051 -0.160                  0              9
mel_retrain rsweep_mel   PN target > every off-target              22               2.158       -0.597 -1.576  0.340                  0              9
    mel_ham rsweep_mel   PN                       all              64               1.687       -0.143 -0.550  0.301                  2              6
    mel_ham rsweep_mel   PN        off-target <= 2 SD              27               1.352       -0.474 -1.005  0.065                  2              6
    mel_ham rsweep_mel   PN target > every off-target              22               2.158       -0.112 -0.724  0.479                  3              6
mel_primary rsweep_mel   VS                       all              64               1.880        0.416 -0.026  0.865                  5              6
mel_primary rsweep_mel   VS        off-target <= 2 SD              24               1.467        0.422 -0.385  1.283                  4              6
mel_primary rsweep_mel   VS target > every off-target              25               2.589        0.550 -0.167  1.277                  5              6
mel_retrain rsweep_mel   VS                       all              64               1.880        0.337 -0.126  0.778                  7              9
mel_retrain rsweep_mel   VS        off-target <= 2 SD              24               1.467        0.259 -0.535  1.101                  7              9
mel_retrain rsweep_mel   VS target > every off-target              25               2.589        0.435 -0.424  1.212                  6              9
    mel_ham rsweep_mel   VS                       all              64               1.880       -1.114 -1.570 -0.656                  0              6
    mel_ham rsweep_mel   VS        off-target <= 2 SD              24               1.467       -0.877 -1.436 -0.277                  0              6
    mel_ham rsweep_mel   VS target > every off-target              25               2.589       -0.929 -1.594 -0.259                  0              6
mel_primary rsweep_mel  Pig                       all              64               2.885        2.756  2.290  3.210                  6              6
mel_primary rsweep_mel  Pig        off-target <= 2 SD               4               2.144        3.351  2.750  3.835                  6              6
mel_primary rsweep_mel  Pig target > every off-target              14               3.327        2.300  1.379  3.274                  6              6
mel_retrain rsweep_mel  Pig                       all              64               2.885        3.306  2.760  3.862                  9              9
mel_retrain rsweep_mel  Pig        off-target <= 2 SD               4               2.144        4.752  3.983  5.520                  9              9
mel_retrain rsweep_mel  Pig target > every off-target              14               3.327        2.855  1.854  3.932                  9              9
    mel_ham rsweep_mel  Pig                       all              64               2.885        4.068  3.554  4.609                  6              6
    mel_ham rsweep_mel  Pig        off-target <= 2 SD               4               2.144        1.808  0.644  3.226                  6              6
    mel_ham rsweep_mel  Pig target > every off-target              14               3.327        2.342  1.234  3.403                  6              6
mel_primary rsweep_mel  Reg                       all              64               3.724        2.250  1.751  2.789                  6              6
mel_primary rsweep_mel  Reg        off-target <= 2 SD              33               3.051        2.242  1.682  2.783                  6              6
mel_primary rsweep_mel  Reg target > every off-target              38               4.480        2.503  1.821  3.183                  6              6
mel_retrain rsweep_mel  Reg                       all              64               3.724        2.404  1.817  2.960                  9              9
mel_retrain rsweep_mel  Reg        off-target <= 2 SD              33               3.051        2.355  1.760  2.972                  9              9
mel_retrain rsweep_mel  Reg target > every off-target              38               4.480        2.678  1.995  3.383                  9              9
    mel_ham rsweep_mel  Reg                       all              64               3.724        0.826  0.319  1.383                  6              6
    mel_ham rsweep_mel  Reg        off-target <= 2 SD              33               3.051        0.671 -0.013  1.360                  6              6
    mel_ham rsweep_mel  Reg target > every off-target              38               4.480        0.832  0.087  1.590                  6              6

### T9_ham10000

Rule 7. Sweep effects of the melanoma classifiers trained on HAM10000 (mean of the two realistic bases). The pigmentation finding transfers if its CI is above 0 on both bases in all but at most one HAM10000 classifier and pigmentation has the largest mean effect of the four knobs.

knob  n classifiers  mean dlogit    min    max  n CI > 0 on both bases  ISIC 2020 primary mean
  PN              6       -0.048 -0.593  0.552                       0                   0.053
  VS              6       -1.078 -1.478 -0.688                       0                   0.287
 Pig              6        4.012  2.928  5.996                       6                   2.765
 Reg              6        0.527 -0.116  1.123                       1                   2.172

HAM10000 classifiers, all seven knobs (mean of two bases):

knob                  BWV   D/G    PN   Pig   Reg   Str    VS
classifier                                                   
mel_ham_effb0_s1     1.40  0.64  0.55  3.18  0.70 -1.26 -0.89
mel_ham_effb0_s2     1.44  0.79  0.03  3.20 -0.09 -0.16 -1.17
mel_ham_resnet50_s1  1.39  0.65 -0.03  2.93  0.79  0.67 -0.94
mel_ham_resnet50_s2  1.62  0.74 -0.59  4.01 -0.12  1.00 -1.30
mel_ham_vitb16_s1    1.74  0.85 -0.28  4.75  1.12  0.86 -0.69
mel_ham_vitb16_s2    1.94  0.85  0.04  6.00  0.76  0.71 -1.48

### PH2

PH2 step: parse failed: PH2 header not recognised: ['Name', 'Histological Diagnosis', 'Clinical Diagnosis', 'Asymmetry | Pigment Network | Dots/Globules | Streaks | Regression Areas | Blue-Whitish Veil', 'Colors']

First lines of the annotation file:

||   Name || Histological Diagnosis || Clinical Diagnosis || Asymmetry | Pigment Network | Dots/Globules | Streaks | Regression Areas | Blue-Whitish Veil ||           Colors ||
|| IMD003 ||                        ||                  0 ||         0 |               T |             A |       A |                A |                 A ||                4 ||
|| IMD009 ||                        ||                  0 ||         0 |               T |             A |       A |                A |                 A ||                3 ||
|| IMD016 ||                        ||                  0 ||         0 |               T |             T |       A |                A |                 A ||             3  4 ||
|| IMD022 ||                        ||                  0 ||         0 |               T |             A |       A |                A |                 A ||                3 ||
|| IMD024 ||                        ||                  0 ||         0 |               T |             A |       A |                A |                 A ||             3  4 ||
|| IMD025 ||                        ||                  0 ||         0 |               T |             T |       A |                A |                 A ||                3 ||
|| IMD035 ||                        ||                  0 ||         2 |               T |             A |       A |                A |                 A ||             2  3 ||
|| IMD038 ||                        ||                  0 ||         0 |               T |             T |       A |                A |                 A ||             4  6 ||
|| IMD042 ||                        ||                  0 ||         0 |               T |             T |       A |                A |                 A ||             3  4 ||
|| IMD044 ||                        ||                  0 ||         0 |               T |             T |       A |                A |                 A ||             4  5 ||
|| IMD045 ||                        ||                  0 ||         0 |               T |             T |       A |                A |                 A ||                3 ||
|| IMD050 ||                        ||                  0 ||         0 |               T |             T |       A |                A |                 A ||                3 ||
|| IMD092 ||                        ||                  0 ||         0 |               T |             T |       A |                A |                 A ||             3  4 ||
|| IMD101 ||                        ||                  0 ||         0 |               T |             A |       A |                A |                 A ||                3 ||
|| IMD103 ||                        ||                  0 ||         0 |               T |             T |       A |                A |                 A ||             3  4 ||
|| IMD112 ||                        ||                  0 ||         0 |               T |             T |       A |                A |                 A ||          2  3  4 ||
|| IMD118 ||                        ||                  0 ||         0 |               T |             T |       A |                A |                 A ||             3  4 ||
|| IMD125 ||                        ||                  0 ||         0 |               T |             T |       A |                A |                 A ||             3  4 ||
|| IMD132 ||                        ||                  0 ||         0 |               T |             T |       A |                A |                 A ||             3  4 ||
|| IMD134 ||                        ||                  0 ||         0 |               T |             T |       A |                A |                 A ||          2  3  4 ||
|| IMD135 ||                        ||                  0 ||         0 |               T |             T |       A |                A |                 A ||             3  4 ||
|| IMD144 ||                        ||                  0 ||         0 |               T |             T |       A |                A |                 A ||             3  4 ||
|| IMD146 ||                        ||                  0 ||         1 |               T |             A |       P |                A |                 A ||          2  3  4 ||
|| IMD147 ||                        ||                  0 ||         0 |               T |             T |       A |                A |                 A ||                3 ||