# src/data_analysis/EDA.py
from __future__ import annotations
import os
from typing import Dict, Any, Optional, Sequence

import pandas as pd
import matplotlib.pyplot as plt

# Shared logging & progress helpers
from src.pipeline.logger import get_logger, task, pbar

log = get_logger("EDA")


def run_eda(
    df: Optional[pd.DataFrame] = None,
    data_path: str = "PhiUSIIL_Phishing_URL_Dataset.csv",
    label_col: str = "label",
    preview_cols: Sequence[str] = ("URLLength", "DomainLength", "NoOfJS", "NoOfImage", "NoOfExternalRef"),
    save_dir: str = "outputs/eda",
    save_fig: bool = True,
    show_fig: bool = False,
) -> Dict[str, Any]:
    """
    Lightweight, non-intrusive EDA with logging. Keeps original analytics intact.
    - If `df` is provided, uses it. Otherwise reads `data_path`.
    - Returns a structured dict (so the notebook can aggregate results cleanly).
    - Figure display is controlled by `show_fig` (not forced).

    Returns:
        {
          "shape": tuple,
          "columns": list[str],
          "dtypes_counts": pd.Series,
          "missing_summary": pd.Series,
          "describe_numeric": pd.DataFrame,
          "label_distribution_pct": pd.Series,
          "preview_describe": pd.DataFrame,
          "fig_label_dist": matplotlib.figure.Figure | None,
          "fig_label_dist_path": str | None
        }
    """
    # 1) Load data
    if df is None:
        with task(log, f"Reading CSV from '{data_path}'"):
            df = pd.read_csv(data_path)

    # 2) Basic dataset info
    shape = df.shape
    columns = df.columns.tolist()
    dtypes_counts = df.dtypes.value_counts()

    # 3) Missing values
    missing = df.isna().sum()
    missing_summary = missing[missing > 0]

    # 4) Numeric describe
    describe_numeric = df.describe().T

    # 5) Label distribution
    if label_col not in df.columns:
        raise KeyError(f"Label column '{label_col}' not found in dataframe columns.")
    label_distribution_pct = (df[label_col].value_counts(normalize=True) * 100).sort_index(ascending=False)

    # 6) Preview of selected numeric columns (if present)
    preview_cols = [c for c in preview_cols if c in df.columns]
    if preview_cols:
        with task(log, f"Describing preview columns ({len(preview_cols)} cols)"):
            # Trivial loop shown with pbar for consistency/feedback on large sets
            _ = [c for c in pbar(preview_cols, desc="Collecting stats")]
            preview_describe = df[preview_cols].describe()
    else:
        preview_describe = pd.DataFrame()

    # 7) Label distribution figure (saved, optionally shown)
    fig, fig_path = None, None
    with task(log, "Creating label distribution plot"):
        try:
            fig = plt.figure(figsize=(6, 4))
            # Keep order: phishing (1) first, then legitimate (0), when present
            order = [1, 0]
            vals = df[label_col].value_counts().reindex(order)
            vals.plot(kind='bar', color=['tab:red', 'tab:blue'], rot=0)
            plt.xticks([0, 1], ['Phishing (1)', 'Légitime (0)'])
            plt.ylabel('Nombre de sites')
            plt.title('Distribution de la variable cible (label)')
            plt.grid(axis='y', alpha=0.4)

            if save_fig:
                os.makedirs(save_dir, exist_ok=True)
                fig_path = os.path.join(save_dir, 'distribution_label.png')
                plt.savefig(fig_path, dpi=300, bbox_inches='tight')
                log.info(f"Figure saved to: {fig_path}")

            if show_fig:
                plt.show()
            else:
                plt.close(fig)
        except Exception as e:
            log.warning(f"Could not create/save label distribution plot: {e}")
            fig, fig_path = None, None

    return {
        "shape": shape,
        "columns": columns,
        "dtypes_counts": dtypes_counts,
        "missing_summary": missing_summary,
        "describe_numeric": describe_numeric,
        "label_distribution_pct": label_distribution_pct,
        "preview_describe": preview_describe,
        "fig_label_dist": fig,
        "fig_label_dist_path": fig_path,
    }


if __name__ == "__main__":
    # Script-mode: keep original behavior (read CSV, print, show + save figure)
    out = run_eda(
        df=None,
        data_path="PhiUSIIL_Phishing_URL_Dataset.csv",
        label_col="label",
        save_dir=".",
        save_fig=True,
        show_fig=True,  # preserve interactive plotting when run as a script
    )

    print("Dimensions du dataset :", out["shape"])
    print("\nColonnes disponibles :")
    print(out["columns"])

    print("\nTypes de variables :")
    print(out["dtypes_counts"])

    print("\nColonnes avec valeurs manquantes :")
    if out["missing_summary"].empty:
        print("Aucune valeur manquante détectée.")
    else:
        print(out["missing_summary"])

    print("\nStatistiques descriptives (numériques) – premières colonnes :")
    sel = [c for c in ["mean", "std", "min", "max"] if c in out["describe_numeric"].columns]
    if sel:
        print(out["describe_numeric"][sel].head(10))
    else:
        print(out["describe_numeric"].head(10))

    print("\nRépartition des classes (en %):")
    print(out["label_distribution_pct"])

    if not out["preview_describe"].empty:
        print("\nAperçu de quelques variables :")
        print(out["preview_describe"])
