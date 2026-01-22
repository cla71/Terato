"""End-to-end training and prediction pipeline."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import json
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report

from .data_ingestion import load_dataset
from .features import build_feature_matrix
from .modeling import ModelBundle, fit_best_model
from .sar import generate_sar_outputs


def _prepare_labels(series: pd.Series) -> tuple[np.ndarray, dict[str, int]]:
    label_mapping = {label: idx for idx, label in enumerate(sorted(series.dropna().unique()))}
    y = series.map(label_mapping).to_numpy()
    return y, label_mapping


def train_and_evaluate(
    train_path: str,
    output_dir: str,
    fingerprint_radius: int = 2,
    fingerprint_bits: int = 2048,
    use_fingerprints: bool = True,
) -> ModelBundle:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    df, schema = load_dataset(train_path, training=True)
    if not schema.label_column:
        raise ValueError("Training requires a label column (e.g., Teratogenic).")

    y, label_mapping = _prepare_labels(df[schema.label_column])
    X, feature_config, bit_info = build_feature_matrix(
        df,
        schema.smiles_column,
        schema.numeric_descriptor_columns,
        schema.lineage_columns,
        fingerprint_radius=fingerprint_radius,
        fingerprint_bits=fingerprint_bits,
        use_fingerprints=use_fingerprints,
    )

    model, summary, best_name = fit_best_model(X, y)

    predictions = model.predict(X)
    report = classification_report(y, predictions, target_names=sorted(label_mapping, key=label_mapping.get))

    summary.to_csv(output / "model_performance.csv", index=False)
    with open(output / "classification_report.txt", "w", encoding="utf-8") as handle:
        handle.write(report)

    sar_dir = output / "sar_outputs"
    sar_result = generate_sar_outputs(
        model,
        list(X.columns),
        bit_info,
        df[schema.smiles_column],
        sar_dir,
    )
    sar_result.summary_table.to_csv(output / "sar_summary.csv", index=False)

    bundle = ModelBundle(
        model=model,
        model_name=best_name,
        feature_config=feature_config,
        feature_columns=list(X.columns),
        label_mapping=label_mapping,
    )

    metadata = {
        "model_name": best_name,
        "feature_config": asdict(feature_config),
        "feature_columns": list(X.columns),
        "label_mapping": label_mapping,
    }
    joblib.dump(bundle, output / "model_bundle.joblib")
    with open(output / "model_metadata.json", "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    return bundle


def predict_with_model(model_bundle_path: str, data_path: str, output_dir: str) -> pd.DataFrame:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    bundle: ModelBundle = joblib.load(model_bundle_path)
    df, schema = load_dataset(data_path, training=False)

    X, _, _ = build_feature_matrix(
        df,
        schema.smiles_column,
        bundle.feature_config.descriptor_columns,
        bundle.feature_config.lineage_columns,
        fingerprint_radius=bundle.feature_config.fingerprint_radius,
        fingerprint_bits=bundle.feature_config.fingerprint_bits,
        use_fingerprints=bundle.feature_config.use_fingerprints,
    )

    X = X[bundle.feature_columns]
    probabilities = bundle.model.predict_proba(X)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    inverse_map = {v: k for k, v in bundle.label_mapping.items()}
    labels = [inverse_map[pred] for pred in predictions]

    output_df = df.copy()
    output_df["predicted_label"] = labels
    output_df["predicted_probability"] = probabilities

    output_df.to_csv(output / "predictions.csv", index=False)
    return output_df
