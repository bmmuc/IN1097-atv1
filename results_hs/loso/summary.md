# LOSO (leave-one-source-out) results summary
Total runtime: 34.0s.
Repeats the LOO evaluation (`results_hs/`) but, on every round, removes **every row sharing the held-out row's source group** from the meta-model's training set (not just the held-out row itself). Evaluation stays per-dataset (N=50), so the numbers below are directly comparable to the plain-LOO ones.

## Source groups (13 groups)
`sbic` (1 row) and `sbic_hf` (6 rows) manually merged: same underlying corpus (Social Bias Frames, Sap et al.), two ingestion pipelines. No other pair of the manifest's 14 raw `source` labels collides (checked both by domain knowledge and by an automated provenance-key match -- see script output / `verify_no_other_collisions`).
| source_group | n_datasets |
| --- | --- |
| ihc | 16 |
| sbic+sbic_hf | 7 |
| hateval | 6 |
| ethos | 5 |
| mlma | 3 |
| measuring_hate_speech | 2 |
| hatebr | 2 |
| davidson | 2 |
| hatexplain | 2 |
| tweet_eval | 2 |
| hatecheck | 1 |
| hatewic | 1 |
| toxic_conversations | 1 |

HARRIS fixed at lam = **0.5** (matches the existing LOO six-approach comparison; not re-swept here).

## Mean Spearman correlation and AUC_loss (all approaches, LOSO)
| approach | mean_spearman | std_spearman | mean_auc_loss | std_auc_loss |
| --- | --- | --- | --- | --- |
| AR | 0.7348 | 0.2945 | 0.0022 | 0.0034 |
| MR | 0.7421 | 0.3097 | 0.0036 | 0.0045 |
| SW | 0.7407 | 0.2975 | 0.0024 | 0.0039 |
| RegressorOnP | 0.8012 | 0.2639 | 0.0013 | 0.0021 |
| RegressorOnR | 0.7797 | 0.2258 | 0.0017 | 0.0027 |
| Harris | 0.7981 | 0.2052 | 0.0013 | 0.0018 |
| RandomBaseline | 0.007 | 0.3407 | 0.0196 | 0.0159 |


## Loss curve (mean loss at t=1..10) -- six approaches + RandomBaseline
| approach | t1 | t2 | t3 | t4 | t5 | t6 | t7 | t8 | t9 | t10 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| AR | 0.0164 | 0.005 | 0.0004 | 0.0002 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| MR | 0.0296 | 0.0058 | 0.0004 | 0.0002 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| SW | 0.0164 | 0.0075 | 0.0004 | 0.0002 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| RegressorOnP | 0.0085 | 0.0034 | 0.0007 | 0.0002 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| RegressorOnR | 0.0104 | 0.0054 | 0.0012 | 0.0002 | 0.0001 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| Harris | 0.0099 | 0.0018 | 0.0007 | 0.0002 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| RandomBaseline | 0.0879 | 0.0441 | 0.0239 | 0.0164 | 0.0096 | 0.0071 | 0.0041 | 0.0022 | 0.0007 | 0.0 |


## Friedman + Nemenyi -- Spearman (six approaches, LOSO)
statistic = 23.1520, p = 0.000315705, CD = 1.0664
Mean ranks: RegressorOnP=2.800, Harris=2.960, RegressorOnR=3.370, MR=3.700, SW=3.990, AR=4.180
Significant pairs: AR vs RegressorOnP (d=1.380), AR vs Harris (d=1.220), SW vs RegressorOnP (d=1.190)

## Friedman + Nemenyi -- AUC_loss (six approaches, LOSO)
statistic = 19.3037, p = 0.00168716, CD = 1.0664
Mean ranks: Harris=3.010, RegressorOnP=3.080, AR=3.500, SW=3.680, RegressorOnR=3.760, MR=3.970
Significant pairs: none

## English (n=43) vs non-English (n=7) Spearman, LOSO vs LOO
LOSO breakdown:
| approach | spearman_english | spearman_non_english | drop |
| --- | --- | --- | --- |
| AR | 0.8354 | 0.1169 | -0.7185 |
| MR | 0.8482 | 0.0901 | -0.7581 |
| SW | 0.8431 | 0.1117 | -0.7314 |
| RegressorOnP | 0.8774 | 0.3333 | -0.544 |
| RegressorOnR | 0.8468 | 0.368 | -0.4788 |
| Harris | 0.8593 | 0.4216 | -0.4377 |

## LOO vs LOSO comparison (T8)
| approach | spearman_loo | spearman_loso | spearman_diff_loso_minus_loo | aucloss_loo | aucloss_loso |
| --- | --- | --- | --- | --- | --- |
| AR | 0.7484 | 0.7348 | -0.0136 | 0.0021 | 0.0022 |
| MR | 0.7465 | 0.7421 | -0.0044 | 0.0021 | 0.0036 |
| SW | 0.7406 | 0.7407 | 0.0001 | 0.0024 | 0.0024 |
| RegressorOnP | 0.8383 | 0.8012 | -0.0371 | 0.001 | 0.0013 |
| RegressorOnR | 0.8403 | 0.7797 | -0.0605 | 0.001 | 0.0017 |
| Harris | 0.8398 | 0.7981 | -0.0417 | 0.001 | 0.0013 |

## Sanity checks
- RandomBaseline mean Spearman = 0.0070 (expect near 0)
- RandomBaseline mean AUC_loss = 0.0196
- ALPHA (SignificantWins) = 0.05
