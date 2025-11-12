# src/models/regression logistique.py
from __future__ import annotations
from typing import Dict, Any, Optional, Sequence, Tuple

import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)

# Shared logger/progress
from src.pipeline.logger import get_logger, task

log = get_logger("LogRegTab")

DEFAULT_FEATURES: Tuple[str, ...] = (
    'URLLength', 'DomainLength', 'NoOfSubDomain', 'IsDomainIP',
    'NoOfLettersInURL', 'NoOfDegitsInURL', 'NoOfEqualsInURL',
    'NoOfQMarkInURL', 'NoOfAmpersandInURL', 'NoOfOtherSpecialCharsInURL',
    'SpacialCharRatioInURL', 'TLDLength'
)

def _train_and_evaluate(
    X: pd.DataFrame,
    y: pd.Series,
    title: str = "Modèle",
    random_state: int = 42
) -> Dict[str, Any]:
    """Internal helper replicating the original training/eval logic."""
    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )

    # Standardisation
    scaler = StandardScaler()
    with task(log, "Scaling features (StandardScaler)"):
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

    # Modèle (same hyperparams as original)
    with task(log, "Training LogisticRegression"):
        logreg = LogisticRegression(max_iter=500, random_state=random_state)
        logreg.fit(X_train_scaled, y_train)

    # Prédictions
    with task(log, "Scoring"):
        y_pred = logreg.predict(X_test_scaled)
        y_proba = logreg.predict_proba(X_test_scaled)[:, 1]

    # Métriques
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
    }
    log.info(
        f"{title} | Acc={metrics['accuracy']:.4f} | Prec={metrics['precision']:.4f} | "
        f"Rec={metrics['recall']:.4f} | F1={metrics['f1']:.4f} | AUC={metrics['roc_auc']:.4f}"
    )

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    return {
        "model": logreg,
        "scaler": scaler,
        "X_columns": list(X.columns),
        "metrics": metrics,
        "confusion_matrix": cm,
        "splits": {
            "X_train": X_train, "X_test": X_test,
            "y_train": y_train, "y_test": y_test,
            "X_train_scaled": X_train_scaled, "X_test_scaled": X_test_scaled
        }
    }

def run_logistic_regression(
    df: Optional[pd.DataFrame] = None,
    data_path: str = "PhiUSIIL_Phishing_URL_Dataset.csv",
    label_col: str = "label",
    features: Sequence[str] = DEFAULT_FEATURES,
    sample_n: Optional[int] = 10000,      # same 10k stratified subset as original
    random_state: int = 42,
    save_dir: str = "outputs/logreg",
    save_fig: bool = True,
    show_fig: bool = False,
    title: str = "Modèle basé sur 10 000 lignes et les caractéristiques de l'URL"
) -> Dict[str, Any]:
    """
    Train/evaluate Logistic Regression exactly like the original script,
    but as a reusable function with logging and controllable plotting.

    Returns:
      {
        "model": LogisticRegression,
        "scaler": StandardScaler,
        "X_columns": list[str],
        "metrics": dict,
        "confusion_matrix": ndarray,
        "coefficients_df": pd.DataFrame,
        "fig_cm": Figure | None, "fig_cm_path": str | None,
        "fig_coeff_pos": Figure | None, "fig_coeff_pos_path": str | None,
        "fig_coeff_neg": Figure | None, "fig_coeff_neg_path": str | None,
        "n_train": int, "n_test": int, "n_used_features": int
      }
    """
    os.makedirs(save_dir, exist_ok=True)

    # Load data
    if df is None:
        with task(log, f"Reading CSV from '{data_path}'"):
            df = pd.read_csv(data_path)
        log.info(f"Dataset initial : {df.shape[0]} lignes, {df.shape[1]} colonnes")

    # Stratified sample to 10k like original
    if sample_n is not None and sample_n < len(df):
        with task(log, f"Stratified sampling to n={sample_n}"):
            _, df_small = train_test_split(
                df, test_size=sample_n, random_state=random_state, stratify=df[label_col]
            )
    else:
        df_small = df
    log.info(f"Échantillon sélectionné : {df_small.shape[0]} lignes")

    # Feature selection (keep only present columns)
    cols_to_keep = [c for c in features if c in df_small.columns]
    if not cols_to_keep:
        raise ValueError("None of the requested features are present in the dataframe.")
    X = df_small[cols_to_keep]
    y = df_small[label_col]
    log.info(f"Variables conservées ({len(cols_to_keep)} au total) : {cols_to_keep}")

    # Train & evaluate
    out = _train_and_evaluate(X, y, title=title, random_state=random_state)
    logreg = out["model"]
    scaler = out["scaler"]
    cm = out["confusion_matrix"]
    metrics = out["metrics"]
    X_train = out["splits"]["X_train"]
    X_test = out["splits"]["X_test"]
    y_train = out["splits"]["y_train"]
    y_test = out["splits"]["y_test"]

    # Confusion matrix plot
    fig_cm, fig_cm_path = None, None
    with task(log, "Rendering confusion matrix"):
        try:
            fig_cm = plt.figure(figsize=(6, 5))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.xlabel('Prédiction')
            plt.ylabel('Valeur réelle')
            plt.title(f'Matrice de confusion - {title}')
            if save_fig:
                fig_cm_path = os.path.join(save_dir, "logreg_confusion_matrix.png")
                plt.savefig(fig_cm_path, dpi=300, bbox_inches='tight')
            if show_fig:
                plt.show()
            else:
                plt.close(fig_cm)
        except Exception as e:
            log.warning(f"Could not render/save confusion matrix: {e}")
            fig_cm, fig_cm_path = None, None

    # Coefficients dataframe (same content as original)
    coeffs_df = pd.DataFrame({
        'Variable': X.columns,
        'Coefficient': logreg.coef_[0]
    }).sort_values(by='Coefficient', ascending=False)

    # Positive top-10 coefficients
    fig_pos, fig_pos_path = None, None
    with task(log, "Rendering top-10 positive coefficients"):
        try:
            fig_pos = plt.figure(figsize=(10, 6))
            sns.barplot(x='Coefficient', y='Variable', data=coeffs_df.head(10), color='steelblue')
            plt.title("Top 10 variables influentes (poids positifs) - Modèle URL-based")
            if save_fig:
                fig_pos_path = os.path.join(save_dir, "logreg_top10_positive_coeffs.png")
                plt.savefig(fig_pos_path, dpi=300, bbox_inches='tight')
            if show_fig:
                plt.show()
            else:
                plt.close(fig_pos)
        except Exception as e:
            log.warning(f"Could not render/save positive coeffs plot: {e}")
            fig_pos, fig_pos_path = None, None

    # Negative top-10 coefficients
    fig_neg, fig_neg_path = None, None
    with task(log, "Rendering top-10 negative coefficients"):
        try:
            fig_neg = plt.figure(figsize=(10, 6))
            sns.barplot(x='Coefficient', y='Variable', data=coeffs_df.tail(10), color='tomato')
            plt.title("Top 10 variables influentes (poids négatifs) - Modèle URL-based")
            if save_fig:
                fig_neg_path = os.path.join(save_dir, "logreg_top10_negative_coeffs.png")
                plt.savefig(fig_neg_path, dpi=300, bbox_inches='tight')
            if show_fig:
                plt.show()
            else:
                plt.close(fig_neg)
        except Exception as e:
            log.warning(f"Could not render/save negative coeffs plot: {e}")
            fig_neg, fig_neg_path = None, None

    return {
        "model": logreg,
        "scaler": scaler,
        "X_columns": cols_to_keep,
        "metrics": metrics,
        "confusion_matrix": cm,
        "coefficients_df": coeffs_df,
        "fig_cm": fig_cm, "fig_cm_path": fig_cm_path,
        "fig_coeff_pos": fig_pos, "fig_coeff_pos_path": fig_pos_path,
        "fig_coeff_neg": fig_neg, "fig_coeff_neg_path": fig_neg_path,
        "n_train": int(len(y_train)), "n_test": int(len(y_test)),
        "n_used_features": int(len(cols_to_keep)),
    }

# --- Script-mode: preserve original prints/plots ---
if __name__ == "__main__":
    out = run_logistic_regression(
        df=None,
        data_path="PhiUSIIL_Phishing_URL_Dataset.csv",
        label_col="label",
        features=DEFAULT_FEATURES,
        sample_n=10000,         # original 10k stratified subset
        random_state=42,
        save_dir=".",
        save_fig=False,         # original script showed plots only
        show_fig=True,
        title="Modèle basé sur 10 000 lignes et les caractéristiques de l'URL"
    )

    print("===== Modèle (Régression Logistique) =====")
    print(f"Accuracy : {out['metrics']['accuracy']:.4f}")
    print(f"Precision : {out['metrics']['precision']:.4f}")
    print(f"Recall : {out['metrics']['recall']:.4f}")
    print(f"F1-score : {out['metrics']['f1']:.4f}")
    print(f"ROC AUC : {out['metrics']['roc_auc']:.4f}\n")
