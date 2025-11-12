# src/utils/data_loader.py
from __future__ import annotations
from typing import Dict, Any, Iterable, Tuple, Optional
import pandas as pd
from ucimlrepo import fetch_ucirepo

class DataLoader:
    """
    Chargement générique depuis UCI ML Repo.
    Fournit des helpers pour récupérer X/y sous forme pandas et
    pour trouver des colonnes candidates (url/label) sans logique modèle.
    """
    def __init__(self, dataset_id: int):
        self.dataset_id = dataset_id
        self.dataset = None

    def load_data(self):
        self.dataset = fetch_ucirepo(id=self.dataset_id)
        return (
            self.dataset.data.features,
            self.dataset.data.targets,
            self.dataset.metadata,
            self.dataset.variables,
        )

    def get_xy_as_dataframes(self) -> Tuple[pd.DataFrame, pd.Series, Dict[str, Any]]:
        """
        Retourne X (DataFrame), y (Series) et meta dict.
        Ne choisit pas les colonnes sémantiquement — reste générique.
        """
        X_all, y_all, meta, vars_ = self.load_data()

        # Normalise vers pandas
        if not isinstance(X_all, pd.DataFrame):
            # Essaie de récupérer des noms via vars_
            cols = None
            if hasattr(vars_, "feature_names"):
                cols = getattr(vars_, "feature_names")
            X_all = pd.DataFrame(X_all, columns=cols)

        if isinstance(y_all, pd.DataFrame):
            if y_all.shape[1] == 1:
                y_series = y_all.iloc[:, 0]
            else:
                # Si plusieurs colonnes cibles, laisse à l'utilisateur le choix ensuite
                y_series = y_all.iloc[:, 0]
        elif isinstance(y_all, pd.Series):
            y_series = y_all
        else:
            # Dernier recours : construit une Series
            y_series = pd.Series(y_all, name="target")

        return X_all, y_series, {"metadata": meta, "variables": vars_}

    @staticmethod
    def find_first_column(candidates: Iterable[str], available: Iterable[str]) -> Optional[str]:
        """
        Renvoie la première colonne présente parmi des candidates, sinon None.
        """
        available_set = set(available)
        for c in candidates:
            if c in available_set:
                return c
        return None
