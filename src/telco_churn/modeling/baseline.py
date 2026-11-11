"""Baselines de développement : prétraitement local au train et seuil hors test."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    log_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler


def normalize_categories(frame):
    """Convertir les chaînes pandas et pd.NA en objets et np.nan pour sklearn."""
    return frame.astype(object).where(frame.notna(), np.nan)


def build_baselines(train: pd.DataFrame) -> dict[str, Pipeline]:
    """Inférer les types sur train ; chaque pipeline possède son prétraitement."""
    numeric = train.select_dtypes(include="number").columns.tolist()
    categorical = [name for name in train if name not in numeric]

    def pipeline(model):
        preprocessing = ColumnTransformer(
            [
                (
                    "numeric",
                    Pipeline(
                        [
                            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                            ("scale", StandardScaler()),
                        ]
                    ),
                    numeric,
                ),
                (
                    "categorical",
                    Pipeline(
                        [
                            ("normalize", FunctionTransformer(normalize_categories)),
                            (
                                "imputer",
                                SimpleImputer(strategy="most_frequent", keep_empty_features=True),
                            ),
                            ("encode", OneHotEncoder(handle_unknown="ignore")),
                        ]
                    ),
                    categorical,
                ),
            ]
        )
        return Pipeline([("preprocessing", preprocessing), ("model", model)])

    return {
        "dummy_prior": pipeline(DummyClassifier(strategy="prior")),
        "logistic_regression": pipeline(LogisticRegression(max_iter=2000, random_state=42)),
    }


def select_threshold(y, probability) -> float:
    """Maximiser F1 sur validation uniquement ; égalité : seuil le plus élevé."""
    if set(np.unique(y)) != {0, 1}:
        raise ValueError("Deux classes requises pour choisir le seuil.")
    precision, recall, thresholds = precision_recall_curve(y, probability)
    f1 = (
        2
        * precision[:-1]
        * recall[:-1]
        / np.maximum(precision[:-1] + recall[:-1], np.finfo(float).eps)
    )
    return float(thresholds[np.flatnonzero(f1 == f1.max())[-1]])


def evaluate(y, probability, threshold: float = 0.5) -> dict:
    """Métriques par snapshot ; AP désigne average precision, pas une aire trapèze."""
    y = np.asarray(y)
    probability = np.asarray(probability)
    if set(np.unique(y)) != {0, 1}:
        raise ValueError("Deux classes requises pour l'évaluation.")
    if probability.shape != y.shape or not np.isfinite(probability).all():
        raise ValueError("Probabilités finies et alignées requises.")
    if ((probability < 0) | (probability > 1)).any():
        raise ValueError("Probabilités hors [0, 1].")
    predicted = probability >= threshold
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    precision = precision_score(y, predicted, zero_division=0)
    recall = recall_score(y, predicted, zero_division=0)
    return {
        "n_rows": len(y),
        "prevalence": float(y.mean()),
        "average_precision": average_precision_score(y, probability),
        "roc_auc": roc_auc_score(y, probability),
        "brier": brier_score_loss(y, probability),
        "log_loss": log_loss(y, probability, labels=[0, 1]),
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0,
        "selected_fraction": float(predicted.mean()),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }
