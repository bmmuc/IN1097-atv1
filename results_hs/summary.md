# Experimental results summary
Total runtime: 150.5s. Max CPU temp observed: 76.5 C. Max GPU temp observed: 50.0 C.
HARRIS lam swept over [0.0, 0.25, 0.5, 0.75, 1.0]; selected lam = **0.5** for the six-approach comparison (best mean Spearman).
## Mean Spearman correlation and AUC_loss (all approaches)
| approach | mean_spearman | std_spearman | mean_auc_loss | std_auc_loss |
| --- | --- | --- | --- | --- |
| AR | 0.7484 | 0.2997 | 0.0021 | 0.0034 |
| MR | 0.7465 | 0.3069 | 0.0021 | 0.0035 |
| SW | 0.7406 | 0.2971 | 0.0024 | 0.0039 |
| RegressorOnP | 0.8383 | 0.1923 | 0.001 | 0.002 |
| RegressorOnR | 0.8403 | 0.1725 | 0.001 | 0.0017 |
| Harris(lam=0.0) | 0.8303 | 0.161 | 0.001 | 0.0016 |
| Harris(lam=0.25) | 0.8344 | 0.153 | 0.0011 | 0.0016 |
| Harris(lam=0.5) | 0.8398 | 0.1358 | 0.001 | 0.0016 |
| Harris(lam=0.75) | 0.8393 | 0.1397 | 0.001 | 0.0016 |
| Harris(lam=1.0) | 0.8368 | 0.1326 | 0.001 | 0.0017 |
| Harris | 0.8398 | 0.1358 | 0.001 | 0.0016 |
| RandomBaseline | 0.007 | 0.3407 | 0.0196 | 0.0159 |


## HARRIS lambda sweep
| lam | mean_spearman | std_spearman | mean_auc_loss | std_auc_loss |
| --- | --- | --- | --- | --- |
| 0.0 | 0.8303 | 0.161 | 0.001 | 0.0016 |
| 0.25 | 0.8344 | 0.153 | 0.0011 | 0.0016 |
| 0.5 | 0.8398 | 0.1358 | 0.001 | 0.0016 |
| 0.75 | 0.8393 | 0.1397 | 0.001 | 0.0016 |
| 1.0 | 0.8368 | 0.1326 | 0.001 | 0.0017 |


## Loss curve (mean loss at t=1..10) -- six approaches + RandomBaseline
| approach | t1 | t2 | t3 | t4 | t5 | t6 | t7 | t8 | t9 | t10 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| AR | 0.0164 | 0.0041 | 0.0004 | 0.0002 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| MR | 0.0164 | 0.0039 | 0.0004 | 0.0002 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| SW | 0.0164 | 0.0071 | 0.0004 | 0.0002 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| RegressorOnP | 0.0068 | 0.0027 | 0.0006 | 0.0002 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| RegressorOnR | 0.0077 | 0.0018 | 0.0003 | 0.0001 | 0.0001 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| Harris | 0.0076 | 0.0015 | 0.0003 | 0.0002 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| RandomBaseline | 0.0879 | 0.0441 | 0.0239 | 0.0164 | 0.0096 | 0.0071 | 0.0041 | 0.0022 | 0.0007 | 0.0 |


## Friedman + Nemenyi -- Spearman (six approaches)
statistic = 34.5033, p = 1.89018e-06, CD = 1.0664
Mean ranks: RegressorOnR=2.780, RegressorOnP=2.880, Harris=3.090, AR=3.820, MR=3.980, SW=4.450
Significant pairs: MR vs RegressorOnP (d=1.100), MR vs RegressorOnR (d=1.200), SW vs RegressorOnP (d=1.570), SW vs RegressorOnR (d=1.670), SW vs Harris (d=1.360)

## Friedman + Nemenyi -- AUC_loss (six approaches)
statistic = 8.4161, p = 0.134747, CD = 1.0664
Mean ranks: Harris=3.220, RegressorOnR=3.260, RegressorOnP=3.440, MR=3.600, AR=3.640, SW=3.840
Significant pairs: none

## Sanity checks
- RandomBaseline mean Spearman = 0.0070 (expect near 0)
- RandomBaseline mean AUC_loss = 0.0196 vs best six-approach AUC_loss = 0.0010
- lam=0 finite: True; lam=1 finite: True
- ALPHA (SignificantWins) = 0.05
