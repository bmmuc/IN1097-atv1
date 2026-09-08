from __future__ import annotations

from pathlib import Path

import pandas as pd

import evaluate as ev

ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results_hs"

SIX_APPROACHES = ["AR", "MR", "SW", "RegressorOnP", "RegressorOnR", "Harris"]

# Figure WIDTH sets the effective print font size once LaTeX scales the PNG
# to a fixed print width: effective_pt = fontsize * print_width_in / fig_width_in.
# Figure HEIGHT is grown independently (breaking the original 9:4.2 aspect)
# to give the hand-drawn stacked labels room so bigger glyphs don't collide.
NEW_FIGSIZE = (5.5, 3.0)
NEW_DPI = 220
TICK_FS = 13
LABEL_FS = 13
CD_FS = 13
TITLE_FS = 9


def main():
    per_dataset_spearman = pd.read_csv(RESULTS_DIR / "per_dataset_spearman.csv", index_col="dataset_id")
    per_dataset_auc_loss = pd.read_csv(RESULTS_DIR / "per_dataset_auc_loss.csv", index_col="dataset_id")

    fr_spearman = ev.friedman_nemenyi(per_dataset_spearman[SIX_APPROACHES], higher_is_better=True)
    fr_aucloss = ev.friedman_nemenyi(per_dataset_auc_loss[SIX_APPROACHES], higher_is_better=False)

    print("Spearman: statistic=%.4f p=%.6g CD=%.4f" %
          (fr_spearman["statistic"], fr_spearman["pvalue"], fr_spearman["cd"]))
    print("  mean ranks:", fr_spearman["mean_ranks"].round(3).to_dict())
    print("  significant pairs:", fr_spearman["significant_pairs"])
    print("AUC_loss: statistic=%.4f p=%.6g CD=%.4f" %
          (fr_aucloss["statistic"], fr_aucloss["pvalue"], fr_aucloss["cd"]))
    print("  mean ranks:", fr_aucloss["mean_ranks"].round(3).to_dict())
    print("  significant pairs:", fr_aucloss["significant_pairs"])

    best_lam = 0.5  # matches the existing Harris(lam=0.5) selection, cosmetic (title text) only

    ev.plot_cd_diagram(
        fr_spearman["mean_ranks"], fr_spearman["cd"],
        f"CD diagram -- mean Spearman rank (CD={fr_spearman['cd']:.3f}, "
        f"HARRIS lam={best_lam}) [hs]",
        str(RESULTS_DIR / "cd_diagram_spearman.png"),
        figsize=NEW_FIGSIZE, dpi=NEW_DPI,
        tick_fontsize=TICK_FS, label_fontsize=LABEL_FS,
        cd_fontsize=CD_FS, title_fontsize=TITLE_FS,
    )
    ev.plot_cd_diagram(
        fr_aucloss["mean_ranks"], fr_aucloss["cd"],
        f"CD diagram -- mean AUC_loss rank (CD={fr_aucloss['cd']:.3f}, "
        f"HARRIS lam={best_lam}) [hs]",
        str(RESULTS_DIR / "cd_diagram_aucloss.png"),
        figsize=NEW_FIGSIZE, dpi=NEW_DPI,
        tick_fontsize=TICK_FS, label_fontsize=LABEL_FS,
        cd_fontsize=CD_FS, title_fontsize=TITLE_FS,
    )
    print(f"\nWrote {RESULTS_DIR / 'cd_diagram_spearman.png'} and "
          f"{RESULTS_DIR / 'cd_diagram_aucloss.png'}")


if __name__ == "__main__":
    main()
