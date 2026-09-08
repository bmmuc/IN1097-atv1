# Hate-speech meta-dataset provenance notes

## Datasets

- Panel of hate-speech / offensive-language classification datasets assembled by a parallel agent into `data_hs/datasets/<dataset_id>.parquet` (columns `text`, `label`) with a manifest at `data_hs/datasets_manifest.csv` (`dataset_id,source,slice,task,language,platform,n_instances,n_classes,minority_pct,provenance`). Each dataset is capped at 3000 instances, minimum 300, minimum 2 classes, minimum 10% minority.

- Datasets used in this build: **50**.

## Deduplication rule

- No cross-dataset deduplication was needed at this stage: the datasets contract already guarantees exactly one row (one dataset_id) per source/slice, produced by the parallel dataset-building agent. `hs_build_metadataset.py` takes the inner join of manifest rows and parquet files present on disk and logs any mismatch.

## Algorithm panel (10, fixed)

TF-IDF family (vectorizer/SVD fit INSIDE each fold on the training split only -- via `sklearn.pipeline.Pipeline`, no leakage):

| algorithm | pipeline |
|---|---|
| tfidf_word_lr | TfidfVectorizer(word, 1-2gram, max_features=20000, min_df=2, sublinear_tf) + LogisticRegression(balanced) |
| tfidf_char_svm | TfidfVectorizer(char_wb, 3-5gram, max_features=30000, min_df=2, sublinear_tf) + LinearSVC(balanced) |
| tfidf_nb | TfidfVectorizer(word, 1-2gram, max_features=20000, min_df=2) + ComplementNB |
| svd_rf | TF-IDF word + TruncatedSVD(<=200) + RandomForestClassifier(200 trees, balanced) |
| svd_histgb | TF-IDF word + TruncatedSVD(<=200) + HistGradientBoostingClassifier |
| svd_knn | TF-IDF word + TruncatedSVD(<=200) + KNeighborsClassifier(k=5, cosine) |

For the `svd_*` algorithms, `n_components` is computed PER FOLD from the actual vocabulary size of the training split (`min(200, n_features_train - 1)`), so it never leaks test-set structure and never exceeds the available TF-IDF dimensionality on tiny folds.

Frozen-embedding family (embeddings computed ONCE per dataset, OUTSIDE the CV loop, cached to `data_hs/embeddings/<dataset_id>__<model_tag>.npy`; only the classifier head is fit inside each fold). This is NOT leakage: the encoders (MiniLM / hateBERT / mBERT) are pretrained and never see this task's labels -- encoding is a fixed, label-blind feature map, exactly analogous to using a fixed dictionary/embedding table as input features.

| algorithm | embedding | head |
|---|---|
| minilm_lr | sentence-transformers/all-MiniLM-L6-v2 (mean-pooled) | LogisticRegression(balanced) |
| minilm_mlp | same MiniLM embeddings | MLPClassifier((128,), max_iter=500) |
| hatebert_lr | GroNLP/hateBERT last-hidden-state, mean-pooled | LogisticRegression(balanced) |
| mbert_lr | google-bert/bert-base-multilingual-cased, mean-pooled | LogisticRegression(balanced) |

By design, `hatebert_lr` and `minilm_*` are English-centric encoders while `mbert_lr` is multilingual; the English encoders are expected to (and, per the numbers below, do) degrade on the Spanish/Portuguese rows relative to `mbert_lr`. This is deliberate meta-learning signal that the `is_english` meta-feature is meant to predict -- it is not a bug and was not corrected.

All estimators are deterministic: `random_state=42` everywhere it applies.

## Performance measure

**Macro-F1** (`sklearn.metrics.f1_score(average="macro")`), used everywhere in P/P_folds. Chosen because hate-speech datasets are class-imbalanced and accuracy is dominated by the majority (non-hate) class.

## Cross-validation

`StratifiedKFold(n_splits=10, shuffle=True, random_state=42)`, fit ONCE per dataset -- the same 10 fold index sets are reused across all 10 algorithms on a given dataset, which is required because a paired Wilcoxon signed-rank test is run over the folds downstream.
- P_folds.csv has 5000 rows = 50 datasets x 10 algorithms x 10 folds.

## Meta-features (X)

Computed on the FULL dataset (a dataset descriptor, not a per-fold quantity).

General / dimension: `n_instances`, `n_classes`, `log_instances = log10(n_instances)`, `instances_per_class = n_instances / n_classes`.

Class balance: `majority_pct`, `minority_pct`, `imbalance_ratio = majority_count / minority_count`, `class_entropy` = Shannon entropy (bits) of the label distribution.

Lexical: `vocab_size` (unique lowercased whitespace tokens), `log_vocab_size`, `vocab_per_instance = vocab_size / n_instances`, `type_token_ratio = vocab_size / total_tokens`, `hapax_ratio` = fraction of vocabulary types occurring exactly once, `mean_doc_len_words`, `std_doc_len_words`, `mean_doc_len_chars`, `mean_word_len`.

Social-media / domain-specific (justification: hate speech is overwhelmingly studied on social-media text, and these descriptors characterize how 'platform-native' / noisy / emphatic a dataset's language is -- properties known to interact with how well TF-IDF vs. pretrained-embedding models perform):
- `pct_docs_with_mention` / `mean_mentions_per_doc` (regex `@\w+`) -- reply/targeting culture (Twitter/Reddit-style).
- `pct_docs_with_url` (`https?://` or `www\.`) -- link-sharing behavior.
- `pct_docs_with_hashtag` / `mean_hashtags_per_doc` (regex `#\w+`) -- topic-tagging behavior.
- `pct_docs_with_emoji` (unicode emoji ranges) -- paralinguistic/affective markup.
- `uppercase_char_ratio` (uppercase letters / all letters) -- shouting / emphasis.
- `exclamation_rate` (mean `!` per doc) -- emphatic/aggressive punctuation.
- `elongation_rate` (fraction of docs matching `(.)\1{2,}`) -- expressive lengthening (e.g. 'soooo').
- `pct_docs_with_profanity` / `profanity_token_rate` -- rate of a small (~40 term) hardcoded English profanity/slur-adjacent lexicon; used purely to characterize how 'raw' a dataset's language is, not to label or classify any person or group.

**Caveat (English-only lexicon):** `PROFANITY_LEXICON` is English-only by construction. `pct_docs_with_profanity` and `profanity_token_rate` will therefore read near-zero on the Spanish/Portuguese/other-language rows regardless of how profane those datasets actually are. The lexicon was deliberately NOT translated/extended -- these two columns are partly confounded with `is_english` and must NOT be interpreted as a language-independent measure of how raw/profane a dataset is; treat them as reliable only within the English-language rows.

Signal / separability / hardness (the highest-value features for this problem -- domain-motivated by instance-hardness research in hate speech):
- `mean_mutual_info`, `max_mutual_info` -- mean/max mutual information between the top-1000 TF-IDF terms and the label (`sklearn.feature_selection.mutual_info_classif`, random_state=42).
- `centroid_cosine_gap = 1 - mean_pairwise_cosine(class centroids)` in MiniLM embedding space (mean over all class pairs for multiclass). Higher = more separable.
- `knn1_disagreement` -- fraction of instances whose nearest neighbour (cosine, MiniLM space, excluding self) carries a different label; a direct instance-hardness / label-noise proxy.
- `silhouette_by_label` -- `sklearn.metrics.silhouette_score` on MiniLM embeddings with the true labels as the partition, cosine metric.

Language / platform (from the manifest): `is_english` (language == 'en'), `is_multilingual_row` (language != 'en'), `is_twitter` (platform == 'twitter').

## Constant columns dropped

- None.

## Imputation

Remaining missing values (e.g. degenerate `type_token_ratio`/`silhouette_by_label` on unusual tiny datasets) imputed with the column median. Total imputed cells: **0**.

## Final matrix shapes

- P : 50 datasets x 10 algorithms

- X : 50 datasets x 36 meta-features

- P_folds : 5000 rows
