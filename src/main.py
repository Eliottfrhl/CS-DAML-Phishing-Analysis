# src/main.py
from __future__ import annotations
import os
import time
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, average_precision_score,
    roc_curve, precision_recall_curve, f1_score
)
from sklearn.model_selection import train_test_split

# Module "niveau 2" (baselines sémantiques)
from src.semantic_models.phishing_url_semantic_baselines import (
    train_and_compare_semantic_baselines,
    PhiUSIILSpec,
    load_urls_and_labels,
)

PLOTS_DIR = "plots"
MODELS_DIR = "models"

# ───────────────────────────────────────────────────────────────────────────────
# Helpers de visualisation/évaluation
# ───────────────────────────────────────────────────────────────────────────────
def ensure_dirs():
    os.makedirs(PLOTS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

def plot_confusion_matrix(cm, title: str, fname: str | None = None):
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Légitime (0)', 'Phishing (1)'],
                yticklabels=['Légitime (0)', 'Phishing (1)'])
    plt.xlabel('Prédit'); plt.ylabel('Réel')
    plt.title(title)
    plt.tight_layout()
    if fname:
        plt.savefig(os.path.join(PLOTS_DIR, fname), dpi=150)
    plt.show()

def plot_roc_pr(y_true, scores, title: str, base_name: str):
    fpr, tpr, _ = roc_curve(y_true, scores)
    prec, rec, _ = precision_recall_curve(y_true, scores)
    roc_auc = roc_auc_score(y_true, scores)
    pr_auc = average_precision_score(y_true, scores)

    # ROC
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("Taux de faux positifs")
    plt.ylabel("Taux de vrais positifs")
    plt.title(f"Courbe ROC – {title}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f"{base_name}_roc.png"), dpi=150)
    plt.show()

    # PR
    plt.figure(figsize=(6, 5))
    plt.plot(rec, prec, label=f"AP = {pr_auc:.3f}")
    plt.xlabel("Rappel"); plt.ylabel("Précision")
    plt.title(f"Courbe Precision-Recall – {title}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f"{base_name}_pr.png"), dpi=150)
    plt.show()

def show_top_tfidf_features(vectorizer, clf, n: int = 20):
    import numpy as np
    feature_names = np.array(vectorizer.get_feature_names_out())
    coefs = clf.coef_[0]
    top_phish_idx = np.argsort(coefs)[-n:][::-1]
    top_legit_idx = np.argsort(coefs)[:n]

    print("\nTop n-grams indicateurs de PHISHING (+) :")
    print(feature_names[top_phish_idx])
    print("\nTop n-grams indicateurs de LÉGITIME (−) :")
    print(feature_names[top_legit_idx])

def optimize_threshold(scores, y_true, metric="f1"):
    import numpy as np
    thresholds = np.linspace(scores.min(), scores.max(), num=200)
    best_t, best_s = 0.0, -1.0
    for t in thresholds:
        y_pred = (scores >= t).astype(int)
        s = f1_score(y_true, y_pred)
        if s > best_s:
            best_s, best_t = s, t
    return best_t, best_s

def full_eval(y_true, scores, title: str, base_name: str, default_thresh: float = 0.0, prob_like: bool = False):
    ensure_dirs()
    roc_auc = roc_auc_score(y_true, scores)
    pr_auc  = average_precision_score(y_true, scores)
    print(f"\n[{title}] ROC-AUC={roc_auc:.4f} | PR-AUC={pr_auc:.4f}")
    plot_roc_pr(y_true, scores, title, base_name)

    y_pred_default = (scores >= default_thresh).astype(int)
    cm_default = confusion_matrix(y_true, y_pred_default)
    print(f"\n--- Rapport @ seuil par défaut ({'0.5' if prob_like else '0.0'}) ---")
    print(classification_report(y_true, y_pred_default, digits=3))
    plot_confusion_matrix(cm_default, f"Matrice de confusion – {title} @default", f"{base_name}_cm_default.png")

    best_t, best_f1 = optimize_threshold(scores, y_true, metric="f1")
    y_pred_best = (scores >= best_t).astype(int)
    cm_best = confusion_matrix(y_true, y_pred_best)
    print(f"--- Rapport @ seuil optimisé (F1 max) t={best_t:.4f}, F1={best_f1:.4f} ---")
    print(classification_report(y_true, y_pred_best, digits=3))
    plot_confusion_matrix(cm_best, f"Matrice de confusion – {title} @opt", f"{base_name}_cm_opt.png")

# ───────────────────────────────────────────────────────────────────────────────
# Programme principal
# ───────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ensure_dirs()

    print("[MAIN] Starting semantic baselines training/eval...")
    t0 = time.time()
    spec = PhiUSIILSpec(dataset_id=967)

    # → FAST text mode ON to avoid long freezes on full corpus
    results = train_and_compare_semantic_baselines(
        spec=spec,
        test_size=0.2,
        random_state=42,
        run_char_tfidf=True,
        run_fasttext=True,
        fasttext_fast_mode=True
    )
    print(f"[MAIN] Baselines done in {time.time()-t0:.1f}s")

    print("[MAIN] Recreating the same split for plots & thresholds...")
    urls, y_all, _meta = load_urls_and_labels(spec)
    from sklearn.model_selection import train_test_split
    X_train_urls, X_test_urls, y_train, y_test = train_test_split(
        urls, y_all, test_size=0.2, random_state=42, stratify=y_all
    )

    # ───────────────────────────────────────────────────────────────────────────
    # A) TF-IDF (char 3–5) + LinearSVC
    # ───────────────────────────────────────────────────────────────────────────
    print("\n[MAIN] === Post-eval: TF-IDF + LinearSVC ===")
    tfidf = results["char_tfidf_svc"]["vectorizer"]
    svc   = results["char_tfidf_svc"]["clf"]
    char_transformer = results["char_tfidf_svc"]["char_transformer"]

    print("[MAIN] Transforming test set with TF-IDF...")
    X_test_text = char_transformer.transform(X_test_urls, use_tqdm=True)
    X_test_tfidf = tfidf.transform(X_test_text)
    scores_svc = svc.decision_function(X_test_tfidf)
    full_eval(
        y_true=y_test.values,
        scores=scores_svc,
        title="TF-IDF char (3–5) + LinearSVC",
        base_name="char_tfidf_svc",
        default_thresh=0.0,
        prob_like=False
    )
    show_top_tfidf_features(tfidf, svc, n=20)

    print("[MAIN] Saving TF-IDF+SVC model...")
    joblib.dump((char_transformer, tfidf, svc), os.path.join(MODELS_DIR, "char_tfidf_linear_svc.joblib"))

    # ───────────────────────────────────────────────────────────────────────────
    # B) FastText (moyenne) + LogisticRegression
    # ───────────────────────────────────────────────────────────────────────────
    print("\n[MAIN] === Post-eval: FastText + LogisticRegression ===")
    ft_embedder = results["fasttext_logreg"]["embedder"]
    tok_transformer = results["fasttext_logreg"]["token_transformer"]
    lr = results["fasttext_logreg"]["clf"]

    print("[MAIN] Embedding test set with FastText (this part shows progress bars)...")
    X_test_tokens = tok_transformer.transform(X_test_urls, use_tqdm=True)
    X_test_ft = ft_embedder.transform(X_test_tokens)
    scores_lr = lr.predict_proba(X_test_ft)[:, 1]
    full_eval(
        y_true=y_test.values,
        scores=scores_lr,
        title="FastText (moyenne) + LogisticRegression",
        base_name="fasttext_logreg",
        default_thresh=0.5,
        prob_like=True
    )

    print("[MAIN] Saving FastText+LR model...")
    joblib.dump((tok_transformer, ft_embedder, lr), os.path.join(MODELS_DIR, "fasttext_logreg.joblib"))

    print("\n✅ Évaluation complète terminée. Modèles et graphiques sauvegardés.")
