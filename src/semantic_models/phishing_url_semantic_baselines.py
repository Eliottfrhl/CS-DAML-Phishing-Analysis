# src/semantic_models/phishing_url_semantic_baselines.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Dict, Any, List, Optional

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, average_precision_score,
    accuracy_score, precision_score, recall_score, f1_score
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from gensim.models import FastText

# Shared logging/progress
from src.pipeline.logger import get_logger, task, pbar

# Utils génériques
from src.utils.data_loader import DataLoader
from src.utils.url_parser import URLParser

log = get_logger("Semantic")


# =========================
# 0) Spécification dataset
# =========================
@dataclass
class PhiUSIILSpec:
    dataset_id: int = 967
    url_col_candidates: Tuple[str, ...] = ("URL", "url", "Url")
    label_col_candidates: Tuple[str, ...] = ("label", "Label", "target", "Target")


def load_urls_and_labels(
    spec: PhiUSIILSpec,
    df: Optional[pd.DataFrame] = None,
    url_col_candidates: Optional[Tuple[str, ...]] = None,
    label_col_candidates: Optional[Tuple[str, ...]] = None,
) -> Tuple[pd.Series, pd.Series, Dict[str, Any]]:
    """
    Si `df` est fourni, on l’utilise (comme les autres modèles tabulaires).
    Sinon, on charge via DataLoader(UCI).
    """
    url_cands = url_col_candidates or spec.url_col_candidates
    label_cands = label_col_candidates or spec.label_col_candidates

    if df is None:
        loader = DataLoader(spec.dataset_id)
        X_all, y_series, meta = loader.get_xy_as_dataframes()
        url_col = DataLoader.find_first_column(url_cands, X_all.columns)
        if url_col is None:
            raise ValueError(f"Colonne URL introuvable parmi {url_cands}.")
        urls = X_all[url_col].astype(str)
    else:
        # Cherche URL + label dans df fourni
        url_col = DataLoader.find_first_column(url_cands, df.columns)
        lab_col = DataLoader.find_first_column(label_cands, df.columns)
        if url_col is None or lab_col is None:
            raise ValueError(
                f"Colonnes introuvables (url={url_cands}, label={label_cands}) dans df."
            )
        urls = df[url_col].astype(str)
        y_series = df[lab_col]
        meta = {"source": "provided_df"}

    # Label -> int si possible
    y = y_series.copy()
    if not np.issubdtype(y.dtype, np.integer):
        try:
            y = y.astype(int)
        except Exception:
            pass

    return urls, y, meta


# ============================================
# 1) Transformeurs texte (réutilisent URLParser)
# ============================================
class URLCharDocumentTransformer:
    """Transforme une série d’URLs en textes nettoyés pour les char-ngrams (TF-IDF)."""
    def fit(self, X: pd.Series, y=None):
        return self

    def transform(self, X: pd.Series, use_tqdm: bool = False) -> List[str]:
        it = pbar(X.astype(str), desc="Cleaning URLs (char-ngrams)") if use_tqdm else X.astype(str)
        return [URLParser.simple_clean(u) for u in it]

    def fit_transform(self, X: pd.Series, y=None, use_tqdm: bool = False) -> List[str]:
        return self.transform(X, use_tqdm=use_tqdm)


class URLWordTokenTransformer:
    """Transforme une série d’URLs en liste de tokens (mots) pour embeddings (FastText)."""
    def fit(self, X: pd.Series, y=None):
        return self

    def transform(self, X: pd.Series, use_tqdm: bool = True) -> List[List[str]]:
        it = pbar(X.astype(str), desc="Tokenizing URLs (word tokens)") if use_tqdm else X.astype(str)
        out: List[List[str]] = []
        for u in it:
            out.append(URLParser.tokenize_words(u))
        return out

    def fit_transform(self, X: pd.Series, y=None, use_tqdm: bool = True) -> List[List[str]]:
        return self.transform(X, use_tqdm=use_tqdm)


# ============================================
# 2) Vectoriseur FastText (embeddings moyens)
# ============================================
class FastTextEmbedder:
    def __init__(
        self,
        vector_size: int = 100,
        window: int = 5,
        min_count: int = 2,
        epochs: int = 10,
        sg: int = 1,
        workers: int = 1,
    ):
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.epochs = epochs
        self.sg = sg
        self.workers = workers
        self.model: Optional[FastText] = None

    def fit(self, token_lists: List[List[str]]):
        with task(
            log,
            f"FastText build (vs={self.vector_size}, win={self.window}, "
            f"min={self.min_count}, ep={self.epochs}, sg={self.sg}, workers={self.workers})"
        ):
            self.model = FastText(
                sentences=token_lists,
                vector_size=self.vector_size,
                window=self.window,
                min_count=self.min_count,
                sg=self.sg,
                workers=self.workers,
            )
            log.info(f"Vocab size: {len(self.model.wv)}")

        with task(log, "FastText train"):
            assert self.model is not None
            self.model.train(token_lists, total_examples=len(token_lists), epochs=self.epochs)
        return self

    def transform(self, token_lists: List[List[str]]) -> np.ndarray:
        assert self.model is not None, "FastTextEmbedder.fit must be called before transform."
        emb = []
        it = pbar(token_lists, desc="Averaging embeddings")
        for tokens in it:
            vecs = [self.model.wv[t] for t in tokens if t in self.model.wv]
            if not vecs:
                emb.append(np.zeros(self.vector_size, dtype=np.float32))
            else:
                emb.append(np.mean(vecs, axis=0))
        return np.vstack(emb)

    def fit_transform(self, token_lists: List[List[str]]) -> np.ndarray:
        self.fit(token_lists)
        return self.transform(token_lists)


# ============================================
# 3) Évaluation commune
# ============================================
def evaluate_binary(y_true, scores_continuous, y_pred_labels) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    # seuil-indep metrics
    try:
        out["roc_auc"] = roc_auc_score(y_true, scores_continuous)
    except Exception:
        out["roc_auc"] = np.nan
    try:
        out["pr_auc"] = average_precision_score(y_true, scores_continuous)
    except Exception:
        out["pr_auc"] = np.nan
    # seuil-dep metrics
    out["accuracy"] = float(accuracy_score(y_true, y_pred_labels))
    out["precision"] = float(precision_score(y_true, y_pred_labels))
    out["recall"] = float(recall_score(y_true, y_pred_labels))
    out["f1"] = float(f1_score(y_true, y_pred_labels))
    # report + CM
    out["report"] = classification_report(y_true, y_pred_labels, digits=3)
    out["confusion_matrix"] = confusion_matrix(y_true, y_pred_labels)
    return out


def print_eval(title: str, res: Dict[str, Any]):
    # Gardé pour compatibilité avec les sorties textuelles existantes.
    print(f"\n=== {title} ===")
    print(f"ROC-AUC: {res.get('roc_auc', float('nan')):.4f}")
    print(f"PR-AUC : {res.get('pr_auc', float('nan')):.4f}")
    print(
        f"Acc: {res['accuracy']:.4f} | Prec: {res['precision']:.4f} | "
        f"Rec: {res['recall']:.4f} | F1: {res['f1']:.4f}"
    )
    print("\n-- Classification report --")
    print(res["report"])
    print("-- Confusion matrix --")
    print(res["confusion_matrix"])


# ============================================
# 3.5) Signaux lexicaux & top n-grams
# ============================================
def compute_lexical_signals(urls: pd.Series) -> Dict[str, Any]:
    s = urls.astype(str)
    n = len(s)
    lengths = s.str.len()
    denom = lengths.replace(0, np.nan)

    digit_ratio = s.str.count(r"\d") / denom
    special_ratio = s.str.count(r"[^A-Za-z0-9]") / denom
    alpha_ratio = s.str.count(r"[A-Za-z]") / denom

    return {
        "n_urls": int(n),
        "length_mean": float(lengths.mean()),
        "length_std": float(lengths.std(ddof=0)),
        "digit_ratio_mean": float(digit_ratio.mean()),
        "special_char_ratio_mean": float(special_ratio.mean()),
        "alpha_ratio_mean": float(alpha_ratio.mean()),
        "length_describe": lengths.describe(),
    }


def get_tfidf_top_ngrams(vectorizer: TfidfVectorizer, clf: LinearSVC, n: int = 20) -> Dict[str, List[str]]:
    names = np.array(vectorizer.get_feature_names_out())
    coefs = clf.coef_[0]
    top_phishing_idx = np.argsort(coefs)[-n:][::-1]
    top_legit_idx = np.argsort(coefs)[:n]
    return {
        "top_phishing_ngrams": names[top_phishing_idx].tolist(),
        "top_legit_ngrams": names[top_legit_idx].tolist(),
    }


# ============================================
# 4) Pipelines modèles (spécifiques)
# ============================================
def train_tfidf_char_svc(
    urls_train: pd.Series,
    urls_test: pd.Series,
    y_train: pd.Series,
    y_test: pd.Series,
    ngram_range: Tuple[int, int] = (3, 5),
    max_features: int = 50000,
) -> Dict[str, Any]:
    """
    Baseline sémantique forte : TF-IDF de n-grammes de caractères + LinearSVC.
    """
    log.info("[TF-IDF+SVC] Cleaning & vectorizing")
    char_tf = URLCharDocumentTransformer()
    tr_txt = char_tf.fit_transform(urls_train, use_tqdm=True)
    te_txt = char_tf.transform(urls_test, use_tqdm=True)

    tfidf = TfidfVectorizer(
        analyzer="char",
        ngram_range=ngram_range,
        lowercase=False,
        max_features=max_features,
    )
    Xtr = tfidf.fit_transform(tr_txt)
    Xte = tfidf.transform(te_txt)

    log.info("[TF-IDF+SVC] Training LinearSVC")
    clf = LinearSVC()
    clf.fit(Xtr, y_train)

    log.info("[TF-IDF+SVC] Evaluating")
    scores = clf.decision_function(Xte)
    y_pred = clf.predict(Xte)

    out = evaluate_binary(y_test, scores, y_pred)
    out.update({"vectorizer": tfidf, "clf": clf, "char_transformer": char_tf})
    out.update(get_tfidf_top_ngrams(tfidf, clf, n=20))  # top n-grams for interpretability
    return out


def train_fasttext_logreg(
    urls_train: pd.Series,
    urls_test: pd.Series,
    y_train: pd.Series,
    y_test: pd.Series,
    vector_size: int = 100,
    window: int = 5,
    min_count: int = 5,
    epochs: int = 3,
    sg: int = 1,
    C: float = 1.0,
    workers: int = 1,
    fast_mode: bool = True,
    train_frac: float = 0.25,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Embeddings FastText (moyenne par URL) + Logistic Regression.

    fast_mode=True  -> échantillonne une fraction des URLs pour entraîner FastText (train_frac).
    """
    log.info("[FastText+LR] Tokenizing train/test URLs")
    tok_tf = URLWordTokenTransformer()
    tr_tokens_full = tok_tf.fit_transform(urls_train, use_tqdm=True)
    te_tokens = tok_tf.transform(urls_test, use_tqdm=True)

    # Sous-échantillonnage pour l'entraînement de FastText (plus léger)
    if fast_mode:
        log.info(f"[FastText+LR] FAST mode ON: sampling train tokens with frac={train_frac}")
        rng = np.random.default_rng(random_state)
        idx = rng.choice(len(tr_tokens_full), size=int(len(tr_tokens_full) * train_frac), replace=False)
        tr_tokens = [tr_tokens_full[i] for i in sorted(idx)]
    else:
        tr_tokens = tr_tokens_full

    log.info("[FastText+LR] Fitting FastText")
    ft = FastTextEmbedder(
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        epochs=epochs,
        sg=sg,
        workers=workers,
    )
    _ = ft.fit_transform(tr_tokens)  # fit + quick embeddings on the sampled set (for timing symmetry)

    # Embeddings pour tout le train/test à partir du modèle appris
    log.info("[FastText+LR] Embedding FULL train/test sets")
    Xtr_full = ft.transform(tr_tokens_full)
    Xte = ft.transform(te_tokens)

    log.info("[FastText+LR] Training LogisticRegression")
    clf = LogisticRegression(max_iter=2000, C=C, n_jobs=None)
    clf.fit(Xtr_full, y_train)

    log.info("[FastText+LR] Evaluating")
    prob = clf.predict_proba(Xte)[:, 1]
    y_pred = (prob >= 0.5).astype(int)

    out = evaluate_binary(y_test, prob, y_pred)
    out.update({"embedder": ft, "token_transformer": tok_tf, "clf": clf})
    out.update({"fasttext_vocab_size": int(len(ft.model.wv)) if ft.model else 0})
    return out


# ============================================
# 5) Routine de comparaison réutilisable
# ============================================
def train_and_compare_semantic_baselines(
    spec: PhiUSIILSpec = PhiUSIILSpec(),
    df: Optional[pd.DataFrame] = None,          # NEW: allow passing a preloaded dataframe
    test_size: float = 0.2,
    random_state: int = 42,
    run_char_tfidf: bool = True,
    run_fasttext: bool = True,
    fasttext_fast_mode: bool = True,
) -> Dict[str, Dict[str, Any]]:
    """
    Charge les URLs + labels (depuis df si fourni, sinon UCI) et entraîne les 2 baselines sémantiques.
    Retourne un dict avec résultats + objets utiles (vectorizers, modèles...) et des infos lexicales.
    """
    log.info("[Data] Loading URLs & labels")
    urls, y, _meta = load_urls_and_labels(spec, df=df)

    log.info("[Data] Train/test split")
    X_train, X_test, y_train, y_test = train_test_split(
        urls, y, test_size=test_size, random_state=random_state, stratify=y
    )
    log.info(f"[Data] Train={len(X_train)} | Test={len(X_test)}")

    # signaux lexicaux globaux (sur le test pour reporting)
    lexical = compute_lexical_signals(X_test)

    results: Dict[str, Dict[str, Any]] = {"lexical_signals": lexical}

    if run_char_tfidf:
        res1 = train_tfidf_char_svc(X_train, X_test, y_train, y_test)
        print_eval("TF-IDF char (3–5) + LinearSVC", res1)
        results["char_tfidf_svc"] = res1

    if run_fasttext:
        res2 = train_fasttext_logreg(
            X_train, X_test, y_train, y_test,
            vector_size=100, window=5, min_count=5, epochs=3, sg=1,
            workers=1, fast_mode=fasttext_fast_mode, train_frac=0.25, random_state=random_state
        )
        print_eval("FastText (moyenne) + LogisticRegression", res2)
        results["fasttext_logreg"] = res2

    return results


# ============================================
# 6) Exécution directe
# ============================================
if __name__ == "__main__":
    _ = train_and_compare_semantic_baselines()
