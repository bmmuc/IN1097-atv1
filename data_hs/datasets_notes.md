# datasets_notes.md

Provenance and build log for `data_hs/datasets/` (hate-speech meta-dataset acquisition layer).

Built by `build_datasets.py`. Runtime: 21.9s. Seed: 42. Cap: 3000. Min total: 300. Min minority frac: 0.1. Min class n: 30.

## Result: 50 datasets written

### Summary table

| dataset_id                             | source                | task                               | language   |   n_instances |   n_classes |   minority_pct |
|:---------------------------------------|:----------------------|:-----------------------------------|:-----------|--------------:|------------:|---------------:|
| davidson_3way                          | davidson              | hate_offensive_neither_3way        | en         |          3000 |           3 |         0.1    |
| davidson_hate_vs_not                   | davidson              | hate_vs_not                        | en         |          3000 |           2 |         0.1    |
| ethos_binary                           | ethos                 | hate_vs_not                        | en         |           998 |           2 |         0.4339 |
| ethos_category_gender_vs_rest          | ethos                 | gender_vs_rest_among_hate          | en         |           433 |           2 |         0.1986 |
| ethos_category_national_origin_vs_rest | ethos                 | national_origin_vs_rest_among_hate | en         |           433 |           2 |         0.1709 |
| ethos_category_race_vs_rest            | ethos                 | race_vs_rest_among_hate            | en         |           433 |           2 |         0.1755 |
| ethos_category_religion_vs_rest        | ethos                 | religion_vs_rest_among_hate        | en         |           433 |           2 |         0.1871 |
| hatebr_offensive_binary                | hatebr                | offensive_binary                   | pt         |          3000 |           2 |         0.4967 |
| hatebr_partyism_vs_rest                | hatebr                | partyism_vs_rest                   | pt         |          3000 |           2 |         0.1    |
| hatecheck_en                           | hatecheck             | hateful_vs_non_hateful             | en         |          3000 |           2 |         0.3127 |
| hateval_en_AG                          | hateval               | aggressiveness_binary              | en         |          3000 |           2 |         0.1773 |
| hateval_en_HS                          | hateval               | hate_vs_not                        | en         |          3000 |           2 |         0.418  |
| hateval_en_TR                          | hateval               | target_range_ind_vs_gen            | en         |          3000 |           2 |         0.1633 |
| hateval_es_AG                          | hateval               | aggressiveness_binary              | es         |          3000 |           2 |         0.326  |
| hateval_es_HS                          | hateval               | hate_vs_not                        | es         |          3000 |           2 |         0.4147 |
| hateval_es_TR                          | hateval               | target_range_ind_vs_gen            | es         |          3000 |           2 |         0.256  |
| hatewic_all                            | hatewic               | word_in_context_hateful_vs_not     | en         |          3000 |           2 |         0.4717 |
| hatexplain_3way                        | hatexplain            | hate_normal_offensive_3way         | en         |          3000 |           3 |         0.285  |
| hatexplain_hate_vs_rest                | hatexplain            | hate_vs_rest                       | en         |          3000 |           2 |         0.309  |
| ihc_stg1_3way                          | ihc                   | implicit_explicit_not_3way         | en         |          3000 |           3 |         0.1    |
| ihc_stg1_binary                        | ihc                   | hate_vs_not                        | en         |          3000 |           2 |         0.381  |
| ihc_stg1_explicit_vs_not               | ihc                   | explicit_vs_not                    | en         |          3000 |           2 |         0.1    |
| ihc_stg1_implicit_vs_not               | ihc                   | implicit_vs_not                    | en         |          3000 |           2 |         0.348  |
| ihc_stg2_6way                          | ihc                   | implicit_class_6way                | en         |          3000 |           6 |         0.1063 |
| ihc_stg2_incitement_vs_rest            | ihc                   | incitement_vs_rest                 | en         |          3000 |           2 |         0.2027 |
| ihc_stg2_inferiority_vs_rest           | ihc                   | inferiority_vs_rest                | en         |          3000 |           2 |         0.1377 |
| ihc_stg2_irony_vs_rest                 | ihc                   | irony_vs_rest                      | en         |          3000 |           2 |         0.1273 |
| ihc_stg2_stereotypical_vs_rest         | ihc                   | stereotypical_vs_rest              | en         |          3000 |           2 |         0.181  |
| ihc_stg2_threatening_vs_rest           | ihc                   | threatening_vs_rest                | en         |          3000 |           2 |         0.1063 |
| ihc_stg2_white_grievance_vs_rest       | ihc                   | white_grievance_vs_rest            | en         |          3000 |           2 |         0.2453 |
| ihc_target_black_vs_not                | ihc                   | hate_target_black_vs_not           | en         |          3000 |           2 |         0.2    |
| ihc_target_immigrant_vs_not            | ihc                   | hate_target_immigrant_vs_not       | en         |          3000 |           2 |         0.2    |
| ihc_target_jewish_vs_not               | ihc                   | hate_target_jewish_vs_not          | en         |          2655 |           2 |         0.2    |
| ihc_target_lgbt_vs_not                 | ihc                   | hate_target_lgbt_vs_not            | en         |           340 |           2 |         0.2    |
| ihc_target_muslim_vs_not               | ihc                   | hate_target_muslim_vs_not          | en         |          3000 |           2 |         0.2    |
| mlma_ar_hate_vs_normal                 | mlma                  | hateful_or_abusive_vs_normal       | ar         |          2062 |           2 |         0.4437 |
| mlma_en_hate_vs_normal                 | mlma                  | hateful_or_abusive_vs_normal       | en         |          3000 |           2 |         0.313  |
| mlma_fr_hate_vs_normal                 | mlma                  | hateful_or_abusive_vs_normal       | fr         |          1358 |           2 |         0.3711 |
| sbic_hf_group_targeted                 | sbic_hf               | targets_group_binary               | en         |          3000 |           2 |         0.492  |
| sbic_hf_offensive_gab                  | sbic_hf               | offensive_binary                   | en         |          3000 |           2 |         0.3697 |
| sbic_hf_offensive_reddit               | sbic_hf               | offensive_binary                   | en         |          3000 |           2 |         0.268  |
| sbic_hf_offensive_stormfront           | sbic_hf               | offensive_binary                   | en         |          3000 |           2 |         0.336  |
| sbic_hf_offensive_twitter              | sbic_hf               | offensive_binary                   | en         |          3000 |           2 |         0.306  |
| sbic_hf_sex                            | sbic_hf               | sexual_lewd_binary                 | en         |          3000 |           2 |         0.1    |
| sbic_offensive_binary                  | sbic                  | offensive_binary                   | en         |          3000 |           2 |         0.4633 |
| toxic_conversations_binary             | toxic_conversations   | toxic_binary                       | en         |          3000 |           2 |         0.1    |
| tweeteval_hate                         | tweet_eval            | hate_vs_not                        | en         |          3000 |           2 |         0.4187 |
| tweeteval_offensive                    | tweet_eval            | offensive_vs_not                   | en         |          3000 |           2 |         0.3297 |
| ucb_hate_binary                        | measuring_hate_speech | hate_vs_not                        | en         |          3000 |           2 |         0.3753 |
| ucb_target_identity_5way               | measuring_hate_speech | target_identity_5way               | en         |          3000 |           5 |         0.134  |

### What FAILED / was dropped, and why

- ihc_target_women_vs_not: DROPPED - only 235 instances after balancing (< 300 minimum)

### Diversity notes

- Languages: ['ar', 'en', 'es', 'fr', 'pt']
- Platforms: ['gab', 'mixed', 'reddit', 'stormfront', 'twitter', 'unknown']
- Sources: ['davidson', 'ethos', 'hatebr', 'hatecheck', 'hateval', 'hatewic', 'hatexplain', 'ihc', 'measuring_hate_speech', 'mlma', 'sbic', 'sbic_hf', 'toxic_conversations', 'tweet_eval']
- n_classes distribution: {2: 45, 3: 3, 6: 1, 5: 1}
- Deliberately spans: explicit vs implicit hate (IHC), offensive vs hate (Davidson/HateXplain/tweet_eval), target/identity identification (IHC stg3 groups, UC Berkeley target_identity_5way, SBIC whoTarget), word-in-context hate sense (HateWiC), synthetic functional testing (HateCheck), and varying imbalance levels (from near-50/50 to the 10% rebalanced floor).

### Verification against hard rules

All datasets pass every hard rule (dedup, no null text, >=2 classes >=30 instances, minority >=10%, 300 <= n <= 3000, manifest counts match parquet).

### Full build log

```
# Local sources


## IHC (Implicit Hate Corpus)
ihc_stg1_3way: rebalanced (undersampled majority classes) so minority class 'explicit_hate' reaches >= 10%
ihc_stg1_3way: OK n=3000 n_classes=3 minority_pct=0.100 classes=['explicit_hate', 'implicit_hate', 'not_hate']
ihc_stg1_binary: OK n=3000 n_classes=2 minority_pct=0.381 classes=['hate', 'not_hate']
ihc_stg1_implicit_vs_not: OK n=3000 n_classes=2 minority_pct=0.348 classes=['implicit_hate', 'not_hate']
ihc_stg1_explicit_vs_not: rebalanced (undersampled majority classes) so minority class 'explicit_hate' reaches >= 10%
ihc_stg1_explicit_vs_not: OK n=3000 n_classes=2 minority_pct=0.100 classes=['explicit_hate', 'not_hate']
ihc_stg2_6way: OK n=3000 n_classes=6 minority_pct=0.106 classes=['incitement', 'inferiority', 'irony', 'stereotypical', 'threatening', 'white_grievance']
ihc_stg2_white_grievance_vs_rest: OK n=3000 n_classes=2 minority_pct=0.245 classes=['rest', 'white_grievance']
ihc_stg2_incitement_vs_rest: OK n=3000 n_classes=2 minority_pct=0.203 classes=['incitement', 'rest']
ihc_stg2_inferiority_vs_rest: OK n=3000 n_classes=2 minority_pct=0.138 classes=['inferiority', 'rest']
ihc_stg2_irony_vs_rest: OK n=3000 n_classes=2 minority_pct=0.127 classes=['irony', 'rest']
ihc_stg2_stereotypical_vs_rest: OK n=3000 n_classes=2 minority_pct=0.181 classes=['rest', 'stereotypical']
ihc_stg2_threatening_vs_rest: OK n=3000 n_classes=2 minority_pct=0.106 classes=['rest', 'threatening']
ihc_target_black_vs_not: OK n=3000 n_classes=2 minority_pct=0.200 classes=['not_hate', 'target_black']
ihc_target_jewish_vs_not: OK n=2655 n_classes=2 minority_pct=0.200 classes=['not_hate', 'target_jewish']
ihc_target_muslim_vs_not: OK n=3000 n_classes=2 minority_pct=0.200 classes=['not_hate', 'target_muslim']
ihc_target_immigrant_vs_not: OK n=3000 n_classes=2 minority_pct=0.200 classes=['not_hate', 'target_immigrant']
ihc_target_women_vs_not: DROPPED - only 235 instances after balancing (< 300 minimum)
ihc_target_lgbt_vs_not: OK n=340 n_classes=2 minority_pct=0.200 classes=['not_hate', 'target_lgbt']

## HatEval (SemEval-2019 Task 5)
hateval_en_HS: OK n=3000 n_classes=2 minority_pct=0.418 classes=[np.int64(0), np.int64(1)]
hateval_en_TR: OK n=3000 n_classes=2 minority_pct=0.163 classes=[np.int64(0), np.int64(1)]
hateval_en_AG: OK n=3000 n_classes=2 minority_pct=0.177 classes=[np.int64(0), np.int64(1)]
hateval_es_HS: OK n=3000 n_classes=2 minority_pct=0.415 classes=[np.int64(0), np.int64(1)]
hateval_es_TR: OK n=3000 n_classes=2 minority_pct=0.256 classes=[np.int64(0), np.int64(1)]
hateval_es_AG: OK n=3000 n_classes=2 minority_pct=0.326 classes=[np.int64(0), np.int64(1)]

## SBIC (local terms_clean cut)
sbic_offensive_binary: OK n=3000 n_classes=2 minority_pct=0.463 classes=[np.int64(0), np.int64(1)]

## HateWiC
hatewic_all: OK n=3000 n_classes=2 minority_pct=0.472 classes=['hateful', 'not_hateful']

# HuggingFace sources (bonus)


## SBIC (HF allenai/social_bias_frames)
sbic_hf_offensive_twitter: OK n=3000 n_classes=2 minority_pct=0.306 classes=[np.float64(0.0), np.float64(1.0)]
sbic_hf_offensive_reddit: OK n=3000 n_classes=2 minority_pct=0.268 classes=[np.float64(0.0), np.float64(1.0)]
sbic_hf_offensive_stormfront: OK n=3000 n_classes=2 minority_pct=0.336 classes=[np.float64(0.0), np.float64(1.0)]
sbic_hf_offensive_gab: OK n=3000 n_classes=2 minority_pct=0.370 classes=[np.float64(0.0), np.float64(1.0)]
sbic_hf_sex: rebalanced (undersampled majority classes) so minority class '1.0' reaches >= 10%
sbic_hf_sex: OK n=3000 n_classes=2 minority_pct=0.100 classes=[np.float64(0.0), np.float64(1.0)]
sbic_hf_group_targeted: OK n=3000 n_classes=2 minority_pct=0.492 classes=[np.float64(0.0), np.float64(1.0)]

## tweet_eval (hate, offensive)
tweeteval_hate: OK n=3000 n_classes=2 minority_pct=0.419 classes=[np.int64(0), np.int64(1)]
tweeteval_offensive: OK n=3000 n_classes=2 minority_pct=0.330 classes=[np.int64(0), np.int64(1)]

## Davidson hate_speech_offensive
davidson_3way: rebalanced (undersampled majority classes) so minority class 'hate_speech' reaches >= 10%
davidson_3way: OK n=3000 n_classes=3 minority_pct=0.100 classes=['hate_speech', 'neither', 'offensive_language']
davidson_hate_vs_not: rebalanced (undersampled majority classes) so minority class 'hate_speech' reaches >= 10%
davidson_hate_vs_not: OK n=3000 n_classes=2 minority_pct=0.100 classes=['hate_speech', 'not_hate_speech']

## HateXplain
hatexplain_3way: OK n=3000 n_classes=3 minority_pct=0.285 classes=['hatespeech', 'normal', 'offensive']
hatexplain_hate_vs_rest: OK n=3000 n_classes=2 minority_pct=0.309 classes=['hatespeech', 'not_hatespeech']

## HateCheck
hatecheck_en: OK n=3000 n_classes=2 minority_pct=0.313 classes=['hateful', 'non-hateful']

## Measuring Hate Speech (UC Berkeley)
ucb_hate_binary: OK n=3000 n_classes=2 minority_pct=0.375 classes=['hate', 'not_hate']
ucb_target_identity_5way: OK n=3000 n_classes=5 minority_pct=0.134 classes=['gender', 'origin', 'race', 'religion', 'sexuality']

## ETHOS
ethos_binary: OK n=998 n_classes=2 minority_pct=0.434 classes=[np.int64(0), np.int64(1)]
ethos_category_race_vs_rest: OK n=433 n_classes=2 minority_pct=0.176 classes=['race', 'rest']
ethos_category_religion_vs_rest: OK n=433 n_classes=2 minority_pct=0.187 classes=['religion', 'rest']
ethos_category_gender_vs_rest: OK n=433 n_classes=2 minority_pct=0.199 classes=['gender', 'rest']
ethos_category_national_origin_vs_rest: OK n=433 n_classes=2 minority_pct=0.171 classes=['national_origin', 'rest']

## MLMA_hate_speech (multilingual)
mlma_en_hate_vs_normal: OK n=3000 n_classes=2 minority_pct=0.313 classes=['hateful_or_abusive', 'normal']
mlma_fr_hate_vs_normal: OK n=1358 n_classes=2 minority_pct=0.371 classes=['hateful_or_abusive', 'normal']
mlma_ar_hate_vs_normal: OK n=2062 n_classes=2 minority_pct=0.444 classes=['hateful_or_abusive', 'normal']

## HateBR (Portuguese)
hatebr_offensive_binary: OK n=3000 n_classes=2 minority_pct=0.497 classes=[np.False_, np.True_]
hatebr_partyism_vs_rest: rebalanced (undersampled majority classes) so minority class 'partyism' reaches >= 10%
hatebr_partyism_vs_rest: OK n=3000 n_classes=2 minority_pct=0.100 classes=['not_partyism', 'partyism']

## toxic_conversations (Jigsaw/Civil Comments)
toxic_conversations_binary: rebalanced (undersampled majority classes) so minority class '1' reaches >= 10%
toxic_conversations_binary: OK n=3000 n_classes=2 minority_pct=0.100 classes=[np.int64(0), np.int64(1)]
```
