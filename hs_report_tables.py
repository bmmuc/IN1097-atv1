from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

import approaches
from approaches import (
    ALPHA,
    AverageRank,
    MedianRank,
    SignificantWins,
    compute_win_tensor,
    ranks_from_P,
)

BASE = Path(__file__).resolve().parent
DATA = BASE / "data_hs"
RESULTS = BASE / "results_hs"
TABLES = RESULTS / "tables"
TEX = TABLES / "tex"
TABLES.mkdir(parents=True, exist_ok=True)
TEX.mkdir(parents=True, exist_ok=True)


# formatting helpers

def esc(s) -> str:
    """Escape LaTeX specials in a text cell."""
    s = str(s)
    s = s.replace("\\", r"\textbackslash{}")
    s = s.replace("_", r"\_")
    s = s.replace("%", r"\%")
    s = s.replace("&", r"\&")
    s = s.replace("#", r"\#")
    return s


def fnum(x, decimals: int) -> str:
    """Format a float with `decimals` places, comma as decimal separator."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    s = f"{float(x):.{decimals}f}"
    return s.replace(".", ",")


def fint(x) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    return str(int(round(float(x))))


def write_tabular(path: Path, header, rows, aligns, comment_lines=None):
    """rows: list of list-of-strings (already formatted/escaped)."""
    lines = []
    if comment_lines:
        for c in comment_lines:
            lines.append(f"% {c}")
    col_spec = "".join(aligns)
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append("\\toprule")
    lines.append(" & ".join(header) + r" \\")
    lines.append("\\midrule")
    for r in rows:
        lines.append(" & ".join(r) + r" \\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    path.write_text("\n".join(lines) + "\n")


def hr(title: str):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


# load data

P = pd.read_csv(DATA / "P.csv", index_col=0)
X = pd.read_csv(DATA / "X.csv", index_col=0)
P_folds = pd.read_csv(DATA / "P_folds.csv")
manifest = pd.read_csv(DATA / "datasets_manifest.csv", index_col=0)
per_dataset_spearman = pd.read_csv(RESULTS / "per_dataset_spearman.csv", index_col=0)
loss_curves = pd.read_csv(RESULTS / "loss_curves.csv", index_col=0)
auc_loss = pd.read_csv(RESULTS / "auc_loss.csv", index_col=0)

ALGOS = list(P.columns)
DATASET_IDS = list(P.index)

R = ranks_from_P(P)
assert R.shape == P.shape, f"R shape {R.shape} != P shape {P.shape}"
assert list(R.index) == DATASET_IDS and list(R.columns) == ALGOS

ALGO_ABBR = {
    "tfidf_word_lr": "TWL",
    "tfidf_char_svm": "TCS",
    "tfidf_nb": "TNB",
    "svd_rf": "SRF",
    "svd_histgb": "SHG",
    "svd_knn": "SKN",
    "minilm_lr": "MLR",
    "minilm_mlp": "MMP",
    "hatebert_lr": "HBL",
    "mbert_lr": "MBL",
}


# T1 -- rank matrix R summarized per algorithm

def build_T1():
    rows = []
    for algo in ALGOS:
        col = R[algo]
        rows.append({
            "algorithm": algo,
            "mean_rank": col.mean(),
            "median_rank": col.median(),
            "std_rank": col.std(ddof=1),
            "best_rank": col.min(),
            "worst_rank": col.max(),
            "wins_rank1": int((col == 1.0).sum()),
            "top3_count": int((col <= 3.0).sum()),
        })
    df = pd.DataFrame(rows).sort_values("mean_rank", kind="mergesort").reset_index(drop=True)
    return df


def emit_T1(df):
    csv_path = TABLES / "T1_rank_matrix_summary.csv"
    df.to_csv(csv_path, index=False)

    header = ["Algoritmo", "Rank m\\'edio", "Rank mediano", "Desvio-padr\\~ao",
              "Melhor rank", "Pior rank", "Vit\\'orias (1\\textordmasculine)", "Top-3"]
    tex_rows = []
    for _, r in df.iterrows():
        tex_rows.append([
            esc(r["algorithm"]), fnum(r["mean_rank"], 2), fnum(r["median_rank"], 2),
            fnum(r["std_rank"], 2), fnum(r["best_rank"], 2), fnum(r["worst_rank"], 2),
            fint(r["wins_rank1"]), fint(r["top3_count"]),
        ])
    aligns = ["l", "r", "r", "r", "r", "r", "r", "r"]
    write_tabular(TEX / "T1_rank_matrix_summary.tex", header, tex_rows, aligns)

    hr("T1 -- rank matrix R summarized per algorithm (sorted by mean rank)")
    print(df.to_string(index=False))
    return csv_path


# T2 -- the three fixed rankings (AR, MR, SW) on the full meta-dataset

def build_T2(win_tensor):
    ar = AverageRank().fit(None, P, R)
    mr = MedianRank().fit(None, P, R)
    sw = SignificantWins(win_tensor, ALGOS).fit(None, P, R)

    ar_raw = R.mean(axis=0)   # mean rank per algorithm (raw aggregate behind AR)
    mr_raw = R.median(axis=0)  # median rank per algorithm (raw aggregate behind MR)
    sw_raw = sw.win_counts_    # total significant-win count per algorithm (raw behind SW)

    for name, series in [("AR", ar.rank_scores_), ("MR", mr.rank_scores_), ("SW", sw.rank_scores_)]:
        assert set(series.index) == set(ALGOS) and len(series) == len(ALGOS), \
            f"{name} ranking does not contain all {len(ALGOS)} algorithms exactly once"

    rows = []
    for algo in ALGOS:
        rows.append({
            "algorithm": algo,
            "AR_position": ar.rank_scores_[algo],
            "AR_mean_rank": ar_raw[algo],
            "MR_position": mr.rank_scores_[algo],
            "MR_median_rank": mr_raw[algo],
            "SW_position": sw.rank_scores_[algo],
            "SW_total_sig_wins": sw_raw[algo],
        })
    df = pd.DataFrame(rows).sort_values("AR_position", kind="mergesort").reset_index(drop=True)
    return df


def emit_T2(df):
    csv_path = TABLES / "T2_fixed_rankings_AR_MR_SW.csv"
    df.to_csv(csv_path, index=False)

    header = ["Algoritmo", "Posi\\c{c}\\~ao AR", "Rank m\\'edio (AR)",
              "Posi\\c{c}\\~ao MR", "Rank mediano (MR)",
              "Posi\\c{c}\\~ao SW", "Total vit. signif. (SW)"]
    tex_rows = []
    for _, r in df.iterrows():
        tex_rows.append([
            esc(r["algorithm"]), fnum(r["AR_position"], 2), fnum(r["AR_mean_rank"], 2),
            fnum(r["MR_position"], 2), fnum(r["MR_median_rank"], 2),
            fnum(r["SW_position"], 2), fint(r["SW_total_sig_wins"]),
        ])
    aligns = ["l", "r", "r", "r", "r", "r", "r"]
    write_tabular(TEX / "T2_fixed_rankings_AR_MR_SW.tex", header, tex_rows, aligns)

    hr("T2 -- fixed rankings AR / MR / SW, fit on the full 50-dataset meta-dataset")
    print(df.to_string(index=False))
    return csv_path


# T3 -- Wilcoxon significant-wins matrix

def build_T3(win_tensor):
    m = len(ALGOS)
    total = np.zeros((m, m), dtype=np.int64)
    for did in DATASET_IDS:
        total += win_tensor[did]
    df = pd.DataFrame(total, index=ALGOS, columns=ALGOS)
    row_totals = df.sum(axis=1)
    col_totals = df.sum(axis=0)
    return df, row_totals, col_totals


def emit_T3(df, row_totals, col_totals, sw_raw_from_T2):
    # sanity: row totals must equal SignificantWins' own summed win counts
    for algo in ALGOS:
        assert row_totals[algo] == sw_raw_from_T2[algo], (
            f"T3 row total for {algo} ({row_totals[algo]}) != SW win count "
            f"({sw_raw_from_T2[algo]})"
        )

    out = df.copy()
    out.index.name = "algorithm"
    out["row_total_sig_wins"] = row_totals
    out.loc["col_total_sig_losses", :] = list(col_totals.values) + [np.nan]
    csv_path = TABLES / "T3_wilcoxon_significant_wins_matrix.csv"
    out.to_csv(csv_path)

    abbr_comment = [
        "Abreviacoes das colunas/linhas (algoritmo -> sigla):",
        *[f"  {a} -> {ALGO_ABBR[a]}" for a in ALGOS],
        f"Celula (linha i, coluna j) = numero de datasets (de 50) em que i bate j",
        f"com significancia estatistica (Wilcoxon pareado sobre os 10 folds, alpha={ALPHA}).",
    ]
    header = ["Alg."] + [ALGO_ABBR[a] for a in ALGOS] + ["Total"]
    tex_rows = []
    for i_algo in ALGOS:
        row = [ALGO_ABBR[i_algo]]
        for j_algo in ALGOS:
            if i_algo == j_algo:
                row.append("")
            else:
                row.append(fint(df.loc[i_algo, j_algo]))
        row.append(fint(row_totals[i_algo]))
        tex_rows.append(row)
    col_row = ["Perdas"] + [fint(col_totals[a]) for a in ALGOS] + [""]
    tex_rows.append(col_row)
    aligns = ["l"] + ["r"] * (len(ALGOS) + 1)
    write_tabular(TEX / "T3_wilcoxon_significant_wins_matrix.tex", header, tex_rows, aligns,
                  comment_lines=abbr_comment)

    hr(f"T3 -- Wilcoxon significant-wins matrix (alpha = {ALPHA}, over 50 datasets x 10 folds)")
    print(df.to_string())
    print("\nRow totals (significant wins):")
    print(row_totals.to_string())
    print("\nColumn totals (significant losses):")
    print(col_totals.to_string())
    return csv_path


# T4 -- full mean loss curve t=1..10 + AUC_loss cross-check

APPROACH_DISPLAY = {
    "AR": "AR", "MR": "MR", "SW": "SW",
    "RegressorOnP": "RegressorOnP", "RegressorOnR": "RegressorOnR",
    "Harris": "HARRIS", "RandomBaseline": "RandomBaseline",
}
T4_ROW_ORDER = ["AR", "MR", "SW", "RegressorOnP", "RegressorOnR", "Harris", "RandomBaseline"]


def build_T4():
    t_cols = [f"t{i}" for i in range(1, 11)]
    sub = loss_curves.loc[T4_ROW_ORDER, t_cols].copy()
    sub["AUC_loss_recomputed"] = sub[t_cols].mean(axis=1)
    sub["AUC_loss_stored"] = auc_loss.loc[T4_ROW_ORDER, "mean_auc_loss"]
    sub["matches_stored"] = np.isclose(
        sub["AUC_loss_recomputed"], sub["AUC_loss_stored"], atol=1e-9, rtol=1e-6)
    all_match = bool(sub["matches_stored"].all())
    assert all_match, f"AUC_loss mismatch:\n{sub[~sub['matches_stored']]}"
    sub.index = [APPROACH_DISPLAY[a] for a in sub.index]
    sub.index.name = "approach"
    return sub, all_match


def emit_T4(df, all_match):
    csv_path = TABLES / "T4_loss_curve_full.csv"
    df.to_csv(csv_path)

    t_cols = [f"t{i}" for i in range(1, 11)]
    header = ["Abordagem"] + [f"t{i}" for i in range(1, 11)] + ["AUC\\_loss"]
    tex_rows = []
    for algo, r in df.iterrows():
        row = [esc(algo)] + [fnum(r[c], 5) for c in t_cols] + [fnum(r["AUC_loss_recomputed"], 5)]
        tex_rows.append(row)
    aligns = ["l"] + ["r"] * 11
    write_tabular(TEX / "T4_loss_curve_full.tex", header, tex_rows, aligns)

    hr("T4 -- full mean loss curve t=1..10, six approaches + RandomBaseline")
    print(df.to_string())
    print(f"\nRecomputed AUC_loss matches results_hs/auc_loss.csv for all rows: {all_match}")
    return csv_path


# T5 -- the 36 meta-features, grouped, with importance

FAMILY = {
    "n_instances": "Dimens\\~ao", "n_classes": "Dimens\\~ao",
    "log_instances": "Dimens\\~ao", "instances_per_class": "Dimens\\~ao",
    "majority_pct": "Balanceamento", "minority_pct": "Balanceamento",
    "imbalance_ratio": "Balanceamento", "class_entropy": "Balanceamento",
    "vocab_size": "Lexical", "log_vocab_size": "Lexical",
    "vocab_per_instance": "Lexical", "type_token_ratio": "Lexical",
    "hapax_ratio": "Lexical", "mean_doc_len_words": "Lexical",
    "std_doc_len_words": "Lexical", "mean_doc_len_chars": "Lexical",
    "mean_word_len": "Lexical",
    "pct_docs_with_mention": "Dom\\'inio/redes sociais",
    "mean_mentions_per_doc": "Dom\\'inio/redes sociais",
    "pct_docs_with_url": "Dom\\'inio/redes sociais",
    "pct_docs_with_hashtag": "Dom\\'inio/redes sociais",
    "mean_hashtags_per_doc": "Dom\\'inio/redes sociais",
    "pct_docs_with_emoji": "Dom\\'inio/redes sociais",
    "uppercase_char_ratio": "Dom\\'inio/redes sociais",
    "exclamation_rate": "Dom\\'inio/redes sociais",
    "elongation_rate": "Dom\\'inio/redes sociais",
    "pct_docs_with_profanity": "Dom\\'inio/redes sociais",
    "profanity_token_rate": "Dom\\'inio/redes sociais",
    "mean_mutual_info": "Sinal e dificuldade", "max_mutual_info": "Sinal e dificuldade",
    "centroid_cosine_gap": "Sinal e dificuldade", "knn1_disagreement": "Sinal e dificuldade",
    "silhouette_by_label": "Sinal e dificuldade",
    "is_english": "Derivada", "is_multilingual_row": "Derivada", "is_twitter": "Derivada",
}

# plain-text (non-escaped) family labels for CSV / stdout
FAMILY_PLAIN = {
    "n_instances": "Dimensão", "n_classes": "Dimensão",
    "log_instances": "Dimensão", "instances_per_class": "Dimensão",
    "majority_pct": "Balanceamento", "minority_pct": "Balanceamento",
    "imbalance_ratio": "Balanceamento", "class_entropy": "Balanceamento",
    "vocab_size": "Lexical", "log_vocab_size": "Lexical",
    "vocab_per_instance": "Lexical", "type_token_ratio": "Lexical",
    "hapax_ratio": "Lexical", "mean_doc_len_words": "Lexical",
    "std_doc_len_words": "Lexical", "mean_doc_len_chars": "Lexical",
    "mean_word_len": "Lexical",
    "pct_docs_with_mention": "Domínio/redes sociais",
    "mean_mentions_per_doc": "Domínio/redes sociais",
    "pct_docs_with_url": "Domínio/redes sociais",
    "pct_docs_with_hashtag": "Domínio/redes sociais",
    "mean_hashtags_per_doc": "Domínio/redes sociais",
    "pct_docs_with_emoji": "Domínio/redes sociais",
    "uppercase_char_ratio": "Domínio/redes sociais",
    "exclamation_rate": "Domínio/redes sociais",
    "elongation_rate": "Domínio/redes sociais",
    "pct_docs_with_profanity": "Domínio/redes sociais",
    "profanity_token_rate": "Domínio/redes sociais",
    "mean_mutual_info": "Sinal e dificuldade", "max_mutual_info": "Sinal e dificuldade",
    "centroid_cosine_gap": "Sinal e dificuldade", "knn1_disagreement": "Sinal e dificuldade",
    "silhouette_by_label": "Sinal e dificuldade",
    "is_english": "Derivada", "is_multilingual_row": "Derivada", "is_twitter": "Derivada",
}

FAMILY_ORDER = ["Dimensão", "Balanceamento", "Lexical", "Domínio/redes sociais",
                "Sinal e dificuldade", "Derivada"]

# Formulas transcribed from hs_meta_features.py / data_hs/metadataset_notes.md
FORMULA = {
    "n_instances": "contagem direta de instâncias",
    "n_classes": "contagem direta de classes",
    "log_instances": "log10(n_instances)",
    "instances_per_class": "n_instances / n_classes",
    "majority_pct": "contagem_classe_majoritaria / n_instances",
    "minority_pct": "contagem_classe_minoritaria / n_instances",
    "imbalance_ratio": "contagem_classe_majoritaria / contagem_classe_minoritaria",
    "class_entropy": "-soma_c p_c * log2(p_c), bits",
    "vocab_size": "nº de tokens únicos (lowercased, split por espaço)",
    "log_vocab_size": "log10(vocab_size)",
    "vocab_per_instance": "vocab_size / n_instances",
    "type_token_ratio": "vocab_size / total_tokens",
    "hapax_ratio": "(nº tipos com freq==1) / vocab_size",
    "mean_doc_len_words": "média de nº de palavras por documento",
    "std_doc_len_words": "desvio-padrão de nº de palavras por documento",
    "mean_doc_len_chars": "média de nº de caracteres por documento",
    "mean_word_len": "média do comprimento (chars) dos tokens do dataset",
    "pct_docs_with_mention": "média de 1[regex @\\w+ encontrado no doc]",
    "mean_mentions_per_doc": "média da contagem de matches de @\\w+ por doc",
    "pct_docs_with_url": "média de 1[regex https?://|www\\. encontrado]",
    "pct_docs_with_hashtag": "média de 1[regex #\\w+ encontrado]",
    "mean_hashtags_per_doc": "média da contagem de matches de #\\w+ por doc",
    "pct_docs_with_emoji": "média de 1[regex de blocos unicode de emoji encontrado]",
    "uppercase_char_ratio": "letras maiúsculas / total de letras",
    "exclamation_rate": "média da contagem de \"!\" por doc",
    "elongation_rate": "média de 1[regex (.)\\1{2,} encontrado] (ex: soooo)",
    "pct_docs_with_profanity": "(nº docs com hit no léxico ~40 termos) / n_instances",
    "profanity_token_rate": "(total tokens no léxico de profanidade) / total_tokens",
    "mean_mutual_info": "média da MI(top-1000 termos TF-IDF; label), mutual_info_classif",
    "max_mutual_info": "máximo da MI(top-1000 termos TF-IDF; label)",
    "centroid_cosine_gap": "1 - média_pares(cosine(centróides de classe)) em espaço MiniLM",
    "knn1_disagreement": "média de 1[label(1-NN cosine, MiniLM) != label]",
    "silhouette_by_label": "silhouette_score(embeddings MiniLM, labels, metric=cosine)",
    "is_english": "1 se language == 'en' senão 0",
    "is_multilingual_row": "1 se language != 'en' senão 0",
    "is_twitter": "1 se platform == 'twitter' senão 0",
}


def build_T5():
    feats = list(X.columns)
    assert len(feats) == 36, f"expected 36 meta-features, found {len(feats)}"
    assert set(feats) == set(FAMILY_PLAIN.keys()), "family mapping does not cover all columns"

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X.values)
    rf = RandomForestRegressor(n_estimators=300, random_state=0)
    rf.fit(Xs, R.loc[X.index].values)
    importance = pd.Series(rf.feature_importances_, index=feats)

    rows = []
    for f in feats:
        rows.append({
            "feature": f,
            "family": FAMILY_PLAIN[f],
            "min": X[f].min(),
            "median": X[f].median(),
            "max": X[f].max(),
            "formula": FORMULA[f],
            "rf_importance": importance[f],
        })
    df = pd.DataFrame(rows)
    df["family"] = pd.Categorical(df["family"], categories=FAMILY_ORDER, ordered=True)
    df = df.sort_values(["family", "rf_importance"], ascending=[True, False]).reset_index(drop=True)
    df["family"] = df["family"].astype(str)
    return df


def emit_T5(df):
    csv_path = TABLES / "T5_meta_features_grouped.csv"
    df.to_csv(csv_path, index=False)

    header = ["Feature", "Fam\\'ilia", "Min", "Mediana", "M\\'ax", "F\\'ormula", "Import. RF"]
    tex_rows = []
    for _, r in df.iterrows():
        tex_rows.append([
            esc(r["feature"]), FAMILY[r["feature"]],
            fnum(r["min"], 4), fnum(r["median"], 4), fnum(r["max"], 4),
            esc(r["formula"]), fnum(r["rf_importance"], 4),
        ])
    aligns = ["l", "l", "r", "r", "r", "l", "r"]
    write_tabular(TEX / "T5_meta_features_grouped.tex", header, tex_rows, aligns)

    hr("T5 -- 36 meta-features grouped by family, with RF(300, rs=0) importance")
    print(f"n_features = {len(df)}  (expected 36)")
    with pd.option_context("display.max_rows", None, "display.width", 160):
        print(df[["feature", "family", "min", "median", "max", "rf_importance"]].to_string(index=False))
    return csv_path


# T6 -- per-dataset appendix (50 rows)

def build_T6():
    winner = P.idxmax(axis=1)
    winner_val = P.max(axis=1)
    second_val = P.apply(lambda row: row.nlargest(2).iloc[1], axis=1)
    gap = winner_val - second_val

    rows = []
    for did in DATASET_IDS:
        m = manifest.loc[did]
        rows.append({
            "dataset_id": did,
            "source": m["source"],
            "language": m["language"],
            "platform": m["platform"],
            "n_instances": m["n_instances"],
            "n_classes": m["n_classes"],
            "minority_pct": m["minority_pct"],
            "winner": winner[did],
            "winner_macro_f1": winner_val[did],
            "gap_to_2nd": gap[did],
            "spearman_AR": per_dataset_spearman.loc[did, "AR"],
            "spearman_HARRIS": per_dataset_spearman.loc[did, "Harris"],
        })
    df = pd.DataFrame(rows).sort_values(
        ["language", "dataset_id"], kind="mergesort").reset_index(drop=True)

    assert len(df) == 50, f"expected 50 rows, got {len(df)}"
    n_noneng = int((df["language"] != "en").sum())
    assert n_noneng == 7, f"expected 7 non-English rows, got {n_noneng}"
    return df, gap


def emit_T6(df):
    csv_path = TABLES / "T6_per_dataset_appendix.csv"
    df.to_csv(csv_path, index=False)

    abbr_comment = [
        "Abreviacoes da coluna 'venc.' (algoritmo vencedor -> sigla):",
        *[f"  {a} -> {ALGO_ABBR[a]}" for a in ALGOS],
    ]
    header = ["dataset\\_id", "fonte", "idi.", "plat.", "n\\_inst", "n\\_cls",
              "min\\_pct", "venc.", "F1\\_venc.", "gap\\_2o", "$\\rho$AR", "$\\rho$HARRIS"]
    tex_rows = []
    for _, r in df.iterrows():
        tex_rows.append([
            esc(r["dataset_id"]), esc(r["source"]), esc(r["language"]), esc(r["platform"]),
            fint(r["n_instances"]), fint(r["n_classes"]), fnum(r["minority_pct"], 4),
            ALGO_ABBR[r["winner"]], fnum(r["winner_macro_f1"], 4), fnum(r["gap_to_2nd"], 4),
            fnum(r["spearman_AR"], 4), fnum(r["spearman_HARRIS"], 4),
        ])
    aligns = ["l", "l", "l", "l", "r", "r", "r", "l", "r", "r", "r", "r"]

    lines = ["% " + abbr_comment[0]]
    lines += [f"% {c}" for c in abbr_comment[1:]]
    lines.append("\\tiny")
    body = []
    col_spec = "".join(aligns)
    body.append(f"\\begin{{tabular}}{{{col_spec}}}")
    body.append("\\toprule")
    body.append(" & ".join(header) + r" \\")
    body.append("\\midrule")
    for r in tex_rows:
        body.append(" & ".join(r) + r" \\")
    body.append("\\bottomrule")
    body.append("\\end{tabular}")
    (TEX / "T6_per_dataset_appendix.tex").write_text("\n".join(lines + body) + "\n")

    n_noneng = int((df["language"] != "en").sum())
    hr("T6 -- per-dataset appendix (50 rows, sorted by language then dataset_id)")
    print(f"n_rows = {len(df)} (expected 50), n_non_english = {n_noneng} (expected 7)")
    print("Non-English rows:")
    print(df[df["language"] != "en"].to_string(index=False))
    return csv_path


# T7 -- English vs non-English breakdown per algorithm

def build_T7():
    en_mask = manifest.loc[DATASET_IDS, "language"] == "en"
    en_mask = en_mask.reindex(DATASET_IDS)
    n_en = int(en_mask.sum())
    n_noneng = int((~en_mask).sum())
    assert n_en == 43 and n_noneng == 7, f"expected 43/7 split, got {n_en}/{n_noneng}"

    rows = []
    for algo in ALGOS:
        f1_en = P.loc[en_mask, algo].mean()
        f1_ne = P.loc[~en_mask, algo].mean()
        rank_en = R.loc[en_mask, algo].mean()
        rank_ne = R.loc[~en_mask, algo].mean()
        rows.append({
            "algorithm": algo,
            "mean_f1_english": f1_en,
            "mean_f1_non_english": f1_ne,
            "mean_rank_english": rank_en,
            "mean_rank_non_english": rank_ne,
            "rank_delta_noneng_minus_eng": rank_ne - rank_en,
        })
    df = pd.DataFrame(rows).sort_values(
        "rank_delta_noneng_minus_eng", ascending=False, kind="mergesort").reset_index(drop=True)
    return df


def emit_T7(df):
    csv_path = TABLES / "T7_english_vs_nonenglish.csv"
    df.to_csv(csv_path, index=False)

    header = ["Algoritmo", "F1 (en, n=43)", "F1 (n\\~ao-en, n=7)",
              "Rank (en)", "Rank (n\\~ao-en)", "$\\Delta$ rank"]
    tex_rows = []
    for _, r in df.iterrows():
        tex_rows.append([
            esc(r["algorithm"]), fnum(r["mean_f1_english"], 4), fnum(r["mean_f1_non_english"], 4),
            fnum(r["mean_rank_english"], 2), fnum(r["mean_rank_non_english"], 2),
            fnum(r["rank_delta_noneng_minus_eng"], 2),
        ])
    aligns = ["l", "r", "r", "r", "r", "r"]
    write_tabular(TEX / "T7_english_vs_nonenglish.tex", header, tex_rows, aligns)

    hr("T7 -- English (n=43) vs non-English (n=7) breakdown per algorithm")
    print(df.to_string(index=False))
    return csv_path


# extra sanity checks against claims already in the draft report

def sanity_checks(t6_df):
    hr("Sanity checks against claims in the draft report")

    en_mask = manifest.loc[DATASET_IDS, "language"] == "en"
    en_mask = en_mask.reindex(DATASET_IDS)

    sp = per_dataset_spearman.copy()
    sp_en = sp.loc[en_mask.values]
    sp_ne = sp.loc[~en_mask.values]
    for col in ["AR", "MR", "SW", "Harris"]:
        print(f"  spearman[{col}] mean: english={sp_en[col].mean():.4f}  "
              f"non-english={sp_ne[col].mean():.4f}")

    winner = P.idxmax(axis=1)
    noneng_ids = [d for d in DATASET_IDS if not en_mask[d]]
    noneng_winner_counts = winner.loc[noneng_ids].value_counts()
    tfidf_wins = int(noneng_winner_counts.get("tfidf_word_lr", 0))
    print(f"  tfidf_word_lr wins on non-English: {tfidf_wins} / {len(noneng_ids)}")

    overall_winner_counts = winner.value_counts()
    minilm_mlp_wins = int(overall_winner_counts.get("minilm_mlp", 0))
    print(f"  minilm_mlp wins overall: {minilm_mlp_wins} / {len(DATASET_IDS)}")

    median_gap = t6_df["gap_to_2nd"].median()
    print(f"  median best-vs-second-best gap (macro-F1): {median_gap:.4f}")


# main

def main():
    win_tensor = compute_win_tensor(P_folds, DATASET_IDS, ALGOS, alpha=ALPHA)

    t1 = build_T1()
    emit_T1(t1)

    t2 = build_T2(win_tensor)
    emit_T2(t2)

    sw_raw = pd.Series(dict(zip(t2["algorithm"], t2["SW_total_sig_wins"])))
    t3_df, row_totals, col_totals = build_T3(win_tensor)
    emit_T3(t3_df, row_totals, col_totals, sw_raw)

    t4, t4_match = build_T4()
    emit_T4(t4, t4_match)

    t5 = build_T5()
    emit_T5(t5)

    t6, gap = build_T6()
    emit_T6(t6)

    t7 = build_T7()
    emit_T7(t7)

    sanity_checks(t6)

    hr("Done")
    print(f"CSV tables written to: {TABLES}")
    print(f"LaTeX tabular bodies written to: {TEX}")


if __name__ == "__main__":
    main()
