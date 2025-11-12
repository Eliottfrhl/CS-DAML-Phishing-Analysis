# src/data_analysis/ACP.py
from __future__ import annotations
from typing import Dict, Any, Optional, Sequence, Tuple

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Shared logging & progress helpers
from src.pipeline.logger import get_logger, task

log = get_logger("ACP")


def run_pca(
    df: Optional[pd.DataFrame] = None,
    data_path: str = "PhiUSIIL_Phishing_URL_Dataset.csv",
    label_col: str = "label",
    drop_cols: Sequence[str] = (),
    scale: bool = True,
    n_components: float | int = 0.95,  # keep 95% variance by default (same intent as original)
    make_scree_plot: bool = True,
    make_scatter_plot: bool = True,
    save_dir: str = "outputs/pca",
    save_fig: bool = True,
    show_fig: bool = False,
) -> Dict[str, Any]:
    """
    PCA/ACP non-intrusive with shared logging. Mirrors the original analysis behavior:
      - Keep only numeric columns, drop `label` from X, fit StandardScaler (optional) + PCA(n_components=0.95 by default)
      - Report retained components count, cumulative explained variance
      - Return top 10 loadings for PC1 and PC2, and (optionally) generate the same plots

    Returns a dict with:
        {
          "numeric_columns": list[str],
          "X_columns": list[str],
          "scaler": StandardScaler | None,
          "pca": PCA,
          "X_pca": np.ndarray,
          "explained_variance_ratio": np.ndarray,
          "n_components_": int,
          "components_df": pd.DataFrame,   # PCA loadings matrix
          "pc1_top_loadings": pd.Series,
          "pc2_top_loadings": pd.Series,
          "fig_scree": Figure | None,
          "fig_scree_path": str | None,
          "fig_scatter": Figure | None,
          "fig_scatter_path": str | None,
        }
    """
    # 1) Load
    if df is None:
        with task(log, f"Reading CSV from '{data_path}'"):
            df = pd.read_csv(data_path)

    # 2) Keep numeric only (as in original)
    with task(log, "Selecting numeric columns"):
        df_num = df.select_dtypes(include=[np.number])
        numeric_columns = df_num.columns.tolist()

    # 3) Split X / y (drop label from X)
    if label_col not in df_num.columns:
        raise KeyError(f"Label column '{label_col}' not found among numeric columns.")
    X = df_num.drop(columns=[label_col, *[c for c in drop_cols if c in df_num.columns]])
    y = df_num[label_col].copy()
    X_columns = X.columns.tolist()

    # 4) Scale (original script used StandardScaler)
    scaler = None
    if scale:
        with task(log, "Scaling features (StandardScaler)"):
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = X.values

    # 5) PCA
    with task(log, f"Fitting PCA (n_components={n_components})"):
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X_scaled)
        explained_variance_ratio = pca.explained_variance_ratio_
        n_retained = pca.n_components_

    log.info(f"Retained components: {n_retained}")
    log.info(f"Cumulative explained variance: {explained_variance_ratio.sum():.4f}")

    # 6) Components (loadings) DataFrame like original
    with task(log, "Building components (loadings) DataFrame"):
        components_df = pd.DataFrame(
            pca.components_,
            columns=X.columns,
            index=[f"PC{i+1}" for i in range(n_retained)]
        )

    pc1_top = components_df.loc["PC1"].abs().sort_values(ascending=False).head(10) if n_retained >= 1 else pd.Series(dtype=float)
    pc2_top = components_df.loc["PC2"].abs().sort_values(ascending=False).head(10) if n_retained >= 2 else pd.Series(dtype=float)

    # 7) Plots (same visuals as commented in original)
    fig_scree, fig_scatter = None, None
    fig_scree_path, fig_scatter_path = None, None
    os.makedirs(save_dir, exist_ok=True)

    if make_scree_plot:
        with task(log, "Creating Scree plot (cumulative explained variance)"):
            try:
                fig_scree = plt.figure(figsize=(8, 5))
                plt.plot(np.cumsum(explained_variance_ratio), marker='o')
                plt.xlabel('Nombre de composantes')
                plt.ylabel('Variance expliquée cumulée')
                plt.title('Scree plot, ACP')
                plt.grid(True)
                if save_fig:
                    fig_scree_path = os.path.join(save_dir, "pca_scree.png")
                    plt.savefig(fig_scree_path, dpi=300, bbox_inches='tight')
                if show_fig:
                    plt.show()
                else:
                    plt.close(fig_scree)
            except Exception as e:
                log.warning(f"Could not render/save Scree plot: {e}")
                fig_scree, fig_scree_path = None, None

    if make_scatter_plot and X_pca.shape[1] >= 2:
        with task(log, "Creating 2D scatter on first two PCs"):
            try:
                fig_scatter = plt.figure(figsize=(8, 6))
                plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='coolwarm', alpha=0.6)
                plt.xlabel('Composante principale 1')
                plt.ylabel('Composante principale 2')
                plt.title('ACP - Visualisation des deux premières composantes')
                cbar = plt.colorbar()
                cbar.set_label('Label (1=Phishing, 0=Légitime)')
                if save_fig:
                    fig_scatter_path = os.path.join(save_dir, "pca_scatter_pc1_pc2.png")
                    plt.savefig(fig_scatter_path, dpi=300, bbox_inches='tight')
                if show_fig:
                    plt.show()
                else:
                    plt.close(fig_scatter)
            except Exception as e:
                log.warning(f"Could not render/save 2D PCA scatter: {e}")
                fig_scatter, fig_scatter_path = None, None

    # Return structured results
    return {
        "numeric_columns": numeric_columns,
        "X_columns": X_columns,
        "scaler": scaler,
        "pca": pca,
        "X_pca": X_pca,
        "explained_variance_ratio": explained_variance_ratio,
        "n_components_": n_retained,
        "components_df": components_df,
        "pc1_top_loadings": pc1_top,
        "pc2_top_loadings": pc2_top,
        "fig_scree": fig_scree,
        "fig_scree_path": fig_scree_path,
        "fig_scatter": fig_scatter,
        "fig_scatter_path": fig_scatter_path,
    }


if __name__ == "__main__":
    # Script-mode: preserve the original behavior (read CSV, print, show plots)
    out = run_pca(
        df=None,
        data_path="PhiUSIIL_Phishing_URL_Dataset.csv",
        label_col="label",
        scale=True,
        n_components=0.95,
        make_scree_plot=True,
        make_scatter_plot=True,
        save_dir=".",
        save_fig=True,
        show_fig=True,
    )

    # Original-like prints (kept under __main__ only)
    print(f"Nombre de composantes retenues : {out['n_components_']}")
    print("Variance expliquée cumulée :", out["explained_variance_ratio"].sum())

    print("\nTop 10 variables influentes sur PC1 :")
    print(out["pc1_top_loadings"])
    print("\nTop 10 variables influentes sur PC2 :")
    print(out["pc2_top_loadings"])
