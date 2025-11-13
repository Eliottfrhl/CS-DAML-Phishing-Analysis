# src/models/NN.py
from __future__ import annotations
from typing import Dict, Any, Optional, Sequence, Tuple

import os
import pandas as pd
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping

# Shared logger/progress
from src.pipeline.logger import get_logger, task

log = get_logger("NeuralNetworkTab")

DEFAULT_FEATURES: Tuple[str, ...] = (
    "URLLength",
    "DomainLength",
    "NoOfSubDomain",
    "IsDomainIP",
    "NoOfLettersInURL",
    "NoOfDegitsInURL",
    "NoOfEqualsInURL",
    "NoOfQMarkInURL",
    "NoOfAmpersandInURL",
    "NoOfOtherSpecialCharsInURL",
    "SpacialCharRatioInURL",
    "TLDLength",
)


def run_neural_network(
    df: Optional[pd.DataFrame] = None,
    data_path: str = "PhiUSIIL_Phishing_URL_Dataset.csv",
    label_col: str = "label",
    features: Sequence[str] = DEFAULT_FEATURES,
    test_size: float = 0.2,
    random_state: int = 42,
    scale_features: bool = True,  # original code scaled before RF
    early_stop: bool = False,
    early_stop_patience: int = 5,
    epochs: int = 50,
    batch_size: int = 64,
    save_dir: str = "outputs/neural_network",
    save_fig: bool = True,
    show_fig: bool = False,
) -> Dict[str, Any]:
    """
    Train/evaluate the Random Forest exactly like the original script,
    but as a reusable function with logging and controlled plotting.

    Returns a dict with:
      {
        "model": Sequential,
        "scaler": MinMaxScaler | None,
        "X_columns": list[str],
        "metrics": {accuracy, precision, recall, f1, roc_auc},
        "confusion_matrix": ndarray,
        "fig_cm": Figure | None,
        "fig_cm_path": str | None,
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

    # --- Feature selection (keep only present columns) ---
    cols_to_keep = [c for c in features if c in df.columns]
    if not cols_to_keep:
        raise ValueError("None of the requested features are present in the dataframe.")
    X = df[cols_to_keep]
    y = df[label_col]

    log.info(f"Using {len(cols_to_keep)} features: {cols_to_keep}")

    # --- Scaling (as per original script) ---
    scaler = None
    if scale_features:
        with task(log, "Scaling features (StandardScaler)"):
            scaler = MinMaxScaler()
            X_scaled = scaler.fit_transform(X)

    else:
        X_scaled = X

    # --- Train/test split ---
    X_train, X_val, Y_train, Y_val = train_test_split(
        X_scaled, y, test_size=test_size, random_state=random_state, stratify=y
    )
    # --- Build Neural Network model ---
    with task(log, "Building Neural Network model"):
        model = Sequential()
        input_dim = X_train.shape[1]
        model.add(Dense(128, input_dim=input_dim, activation="relu"))
        model.add(Dense(64, activation="relu"))
        model.add(Dense(32, activation="relu"))
        model.add(Dense(1, activation="sigmoid"))
        model.summary()

    # --- Compile model ---
    with task(log, "Compiling Neural Network model"):
        model.compile(
            optimizer="adam",
            loss="binary_crossentropy",  # cost function for binary classification
            metrics=["accuracy"],
        )

    # --- Train model ---
    with task(log, "Training Neural Network model"):
        if early_stop:
            early_stopper = EarlyStopping(
                monitor="val_loss", patience=5, restore_best_weights=True
            )

            history = model.fit(
                X_train,
                Y_train,
                epochs=100,
                batch_size=batch_size,
                validation_data=(X_val, Y_val),
                callbacks=[early_stopper],
                verbose=1,
            )
        else:
            history = model.fit(
                X_train,
                Y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(X_val, Y_val),
                verbose=1,
            )

    with task(log, "Plotting training report"):
        try:
            fig_training_report, fig_training_report_path = None, None
            fig_training_report, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
            ax1.plot(history.history["accuracy"], label="Training data")
            ax1.plot(history.history["val_accuracy"], label="Validation data")
            ax1.set_xlabel("Epoch number")
            ax1.set_ylabel("Accuracy")
            ax1.legend(loc="lower right")
            ax1.grid()

            ax2.plot(history.history["loss"], label="Training data")
            ax2.plot(history.history["val_loss"], label="Validation data")
            ax2.set_xlabel("Epoch number")
            ax2.set_ylabel("Loss")
            ax2.legend(loc="upper right")
            ax2.grid()

            plt.title("Training and Validation Accuracy/Loss")
            if save_fig:
                fig_training_report_path = os.path.join(
                    save_dir, "nn_training_report.png"
                )
                plt.savefig(fig_training_report_path, dpi=300, bbox_inches="tight")
            if show_fig:
                plt.show()
            else:
                plt.close(fig_training_report)
        except Exception as e:
            log.warning(f"Could not render/save training report: {e}")
            fig_training_report, fig_training_report_path = None, None

    # --- Predict & metrics ---
    with task(log, "Scoring"):
        y_pred_probs = model.predict(X_val)
        y_pred_classes = (y_pred_probs > 0.5).astype(int)

        try:
          metrics = {
              "accuracy": float(accuracy_score(Y_val, y_pred_classes)),
              "precision": float(precision_score(Y_val, y_pred_classes)),
              "recall": float(recall_score(Y_val, y_pred_classes)),
              "f1": float(f1_score(Y_val, y_pred_classes)),
              "roc_auc": float(roc_auc_score(Y_val, y_pred_probs)),
          }
        except:
            print("error")
        log.info(
            f"Acc={metrics['accuracy']:.4f} | Prec={metrics['precision']:.4f} | "
            f"Rec={metrics['recall']:.4f} | F1={metrics['f1']:.4f} | AUC={metrics['roc_auc']:.4f}"
        )

    # --- Confusion matrix plot ---
    cm = confusion_matrix(Y_val, y_pred_classes)
    fig_cm, fig_cm_path = None, None
    with task(log, "Rendering confusion matrix"):
        try:
            fig_cm = plt.figure(figsize=(6, 5))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
            plt.xlabel("Prediction")
            plt.ylabel("Real Value")
            plt.title("Confusion Matrix - Neural Network Model")
            if save_fig:
                fig_cm_path = os.path.join(save_dir, "nn_confusion_matrix.png")
                plt.savefig(fig_cm_path, dpi=300, bbox_inches="tight")
            if show_fig:
                plt.show()
            else:
                plt.close(fig_cm)
        except Exception as e:
            log.warning(f"Could not render/save confusion matrix: {e}")
            fig_cm, fig_cm_path = None, None
    return {
        "model": model,
        "scaler": scaler,
        "X_columns": cols_to_keep,
        "metrics": metrics,
        "confusion_matrix": cm,
        "fig_training_report": fig_training_report,
        "fig_training_report_path": fig_training_report_path,
        "fig_cm": fig_cm,
        "fig_cm_path": fig_cm_path,
        "n_train": int(len(Y_train)),
        "n_test": int(len(Y_val)),
        "n_used_features": int(len(cols_to_keep)),
    }


# --- Script-mode: preserve original behavior (prints + shown plots) ---
if __name__ == "__main__":
    out = run_neural_network(
        df=None,
        data_path="PhiUSIIL_Phishing_URL_Dataset.csv",
        label_col="label",
        features=DEFAULT_FEATURES,
        test_size=0.2,
        random_state=42,
        scale_features=True,
        early_stop=False,
        early_stop_patience=5,
        epochs=50,
        batch_size=64,
        save_dir=".",
        save_fig=False,  # original script didn't save to files by default
        show_fig=True,  # original behavior: show figures interactively
    )

    # Original-style prints
    print("===== Neural Network Model =====")
    print(f"Accuracy : {out['metrics']['accuracy']:.4f}")
    print(f"Precision : {out['metrics']['precision']:.4f}")
    print(f"Recall : {out['metrics']['recall']:.4f}")
    print(f"F1-score : {out['metrics']['f1']:.4f}")
    print(f"ROC AUC : {out['metrics']['roc_auc']:.4f}\n")
