# src/models/random forest.py
from __future__ import annotations
from typing import Dict, Any, Optional, Sequence, Tuple

import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)

# Shared logger/progress
from src.pipeline.logger import get_logger, task

log = get_logger("RandomForestTab")


DEFAULT_FEATURES: Tuple[str, ...] = (
    'URLLength', 'DomainLength', 'NoOfSubDomain', 'IsDomainIP',
    'NoOfLettersInURL', 'NoOfDegitsInURL', 'NoOfEqualsInURL',
    'NoOfQMarkInURL', 'NoOfAmpersandInURL', 'NoOfOtherSpecialCharsInURL',
    'SpacialCharRatioInURL', 'TLDLength'
)


def run_random_forest(
    df: Optional[pd.DataFrame] = None,
    data_path: str = "PhiUSIIL_Phishing_URL_Dataset.csv",
    label_col: str = "label",
    features: Sequence[str] = DEFAULT_FEATURES,
    sample_n: Optional[int] = 10000,           # same as original (10k stratified)
    test_size: float = 0.2,
    random_state: int = 42,
    scale_features: bool = True,               # original code scaled before RF
    rf_params: Optional[Dict[str, Any]] = None,
    save_dir: str = "outputs/random_forest",
    save_fig: bool = True,
    show_fig: bool = False
) -> Dict[str, Any]:
    """
    Train/evaluate the Random Forest exactly like the original script,
    but as a reusable function with logging and controlled plotting.

    Returns a dict with:
      {
        "model": RandomForestClassifier,
        "scaler": StandardScaler | None,
        "X_columns": list[str],
        "metrics": {accuracy, precision, recall, f1, roc_auc},
        "confusion_matrix": ndarray,
        "feature_importances_df": pd.DataFrame,
        "fig_cm": Figure | None,
        "fig_cm_path": str | None,
        "fig_importances": Figure | None,
        "fig_importances_path": str | None,
        "n_train": int,
        "n_test": int,
        "n_used_features": int
      }
    """
    os.makedirs(save_dir, exist_ok=True)

    # --- Load data ---
    if df is None:
        with task(log, f"Reading CSV from '{data_path}'"):
            df = pd.read_csv(data_path)

    # --- Stratified sample (10k) like the original ---
    if sample_n is not None and sample_n < len(df):
        with task(log, f"Stratified sampling to n={sample_n}"):
            _, df_small = train_test_split(
                df, test_size=sample_n, stratify=df[label_col], random_state=random_state
            )
    else:
        df_small = df

    # --- Feature selection (keep only present columns) ---
    cols_to_keep = [c for c in features if c in df_small.columns]
    if not cols_to_keep:
        raise ValueError("None of the requested features are present in the dataframe.")
    X = df_small[cols_to_keep]
    y = df_small[label_col]

    log.info(f"Using {len(cols_to_keep)} features: {cols_to_keep}")

    # --- Train/valid split ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    # --- Scaling (as per original script) ---
    scaler = None
    if scale_features:
        with task(log, "Scaling features (StandardScaler)"):
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
    else:
        X_train_scaled, X_test_scaled = X_train.values, X_test.values

    # --- RF params (identical defaults to original) ---
    if rf_params is None:
        rf_params = dict(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=-1
        )

    # --- Train RF ---
    with task(log, "Training RandomForest"):
        rf_model = RandomForestClassifier(**rf_params)
        rf_model.fit(X_train_scaled, y_train)

    # --- Predict & metrics ---
    with task(log, "Scoring"):
        y_pred = rf_model.predict(X_test_scaled)
        y_proba = rf_model.predict_proba(X_test_scaled)[:, 1]

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
    }
    log.info(
        f"Acc={metrics['accuracy']:.4f} | Prec={metrics['precision']:.4f} | "
        f"Rec={metrics['recall']:.4f} | F1={metrics['f1']:.4f} | AUC={metrics['roc_auc']:.4f}"
    )

    # --- Confusion matrix plot ---
    cm = confusion_matrix(y_test, y_pred)
    fig_cm, fig_cm_path = None, None
    with task(log, "Rendering confusion matrix"):
        try:
            fig_cm = plt.figure(figsize=(6, 5))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Purples')
            plt.xlabel('Prédiction')
            plt.ylabel('Valeur réelle')
            plt.title('Matrice de confusion - Modèle Random Forest')
            if save_fig:
                fig_cm_path = os.path.join(save_dir, "rf_confusion_matrix.png")
                plt.savefig(fig_cm_path, dpi=300, bbox_inches='tight')
            if show_fig:
                plt.show()
            else:
                plt.close(fig_cm)
        except Exception as e:
            log.warning(f"Could not render/save confusion matrix: {e}")
            fig_cm, fig_cm_path = None, None

    # --- Feature importances plot ---
    feat_imp_df = pd.DataFrame({
        'Variable': X.columns,
        'Importance': rf_model.feature_importances_
    }).sort_values(by='Importance', ascending=False)

    fig_imp, fig_imp_path = None, None
    with task(log, "Rendering feature importances"):
        try:
            fig_imp = plt.figure(figsize=(10, 6))
            sns.barplot(x='Importance', y='Variable', data=feat_imp_df, color='mediumpurple')
            plt.title("Importance des variables - Modèle Random Forest")
            plt.tight_layout()
            if save_fig:
                fig_imp_path = os.path.join(save_dir, "rf_feature_importances.png")
                plt.savefig(fig_imp_path, dpi=300, bbox_inches='tight')
            if show_fig:
                plt.show()
            else:
                plt.close(fig_imp)
        except Exception as e:
            log.warning(f"Could not render/save feature importances: {e}")
            fig_imp, fig_imp_path = None, None

    return {
        "model": rf_model,
        "scaler": scaler,
        "X_columns": cols_to_keep,
        "metrics": metrics,
        "confusion_matrix": cm,
        "feature_importances_df": feat_imp_df,
        "fig_cm": fig_cm,
        "fig_cm_path": fig_cm_path,
        "fig_importances": fig_imp,
        "fig_importances_path": fig_imp_path,
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "n_used_features": int(len(cols_to_keep)),
    }


# --- Script-mode: preserve original behavior (prints + shown plots) ---
if __name__ == "__main__":
    out = run_random_forest(
        df=None,
        data_path="PhiUSIIL_Phishing_URL_Dataset.csv",
        label_col="label",
        features=DEFAULT_FEATURES,
        sample_n=10000,          # original 10k stratified subset
        test_size=0.2,
        random_state=42,
        scale_features=True,
        rf_params=None,          # use the original defaults
        save_dir=".",
        save_fig=False,          # original script didn't save to files by default
        show_fig=True            # original behavior: show figures interactively
    )

    # Original-style prints
    print("===== Modèle Random Forest =====")
    print(f"Accuracy : {out['metrics']['accuracy']:.4f}")
    print(f"Precision : {out['metrics']['precision']:.4f}")
    print(f"Recall : {out['metrics']['recall']:.4f}")
    print(f"F1-score : {out['metrics']['f1']:.4f}")
    print(f"ROC AUC : {out['metrics']['roc_auc']:.4f}\n")
    print(out["feature_importances_df"])
