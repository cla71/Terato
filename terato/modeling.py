"""Model training, evaluation, and serialization."""

from __future__ import annotations

from dataclasses import dataclass
import importlib
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from .features import FeatureConfig


@dataclass
class ModelBundle:
    model: Any
    model_name: str
    feature_config: FeatureConfig
    feature_columns: list[str]
    label_mapping: dict[str, int]


def _optional_model(name: str):
    return importlib.util.find_spec(name) is not None


def _build_models(random_state: int) -> dict[str, Any]:
    models: dict[str, Any] = {
        "RandomForest": RandomForestClassifier(
            n_estimators=400,
            random_state=random_state,
            class_weight="balanced",
        ),
        "GradientBoosting": GradientBoostingClassifier(random_state=random_state),
        "LogisticRegression": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "SVM": SVC(probability=True, class_weight="balanced"),
    }
    if _optional_model("xgboost"):
        xgboost = importlib.import_module("xgboost")
        models["XGBoost"] = xgboost.XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=random_state,
        )
    if _optional_model("lightgbm"):
        lightgbm = importlib.import_module("lightgbm")
        models["LightGBM"] = lightgbm.LGBMClassifier(
            n_estimators=400,
            learning_rate=0.05,
            random_state=random_state,
        )
    return models


def _build_preprocessor(feature_columns: list[str]) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                feature_columns,
            )
        ],
        remainder="drop",
    )


def evaluate_models(X: pd.DataFrame, y: np.ndarray, random_state: int = 13) -> tuple[pd.DataFrame, dict[str, Any]]:
    models = _build_models(random_state)
    results = []
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    best_model_name = None
    best_auc = -np.inf
    best_pipeline: Any = None

    for name, model in models.items():
        preprocessor = _build_preprocessor(list(X.columns))
        pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
        probas = cross_val_predict(pipeline, X, y, cv=cv, method="predict_proba")[:, 1]
        preds = (probas >= 0.5).astype(int)

        metrics = {
            "model": name,
            "roc_auc": roc_auc_score(y, probas),
            "accuracy": accuracy_score(y, preds),
            "balanced_accuracy": balanced_accuracy_score(y, preds),
            "f1_score": f1_score(y, preds),
            "mcc": matthews_corrcoef(y, preds),
        }
        results.append(metrics)

        if metrics["roc_auc"] > best_auc:
            best_auc = metrics["roc_auc"]
            best_model_name = name
            best_pipeline = pipeline

    summary = pd.DataFrame(results).sort_values(by="roc_auc", ascending=False).reset_index(drop=True)
    return summary, {"name": best_model_name, "pipeline": best_pipeline}


def fit_best_model(X: pd.DataFrame, y: np.ndarray, random_state: int = 13) -> tuple[Pipeline, pd.DataFrame, str]:
    summary, best = evaluate_models(X, y, random_state=random_state)
    pipeline: Pipeline = best["pipeline"]
    pipeline.fit(X, y)
    return pipeline, summary, best["name"]
