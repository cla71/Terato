"""Local SAR and hotspot interpretation helpers."""

from __future__ import annotations

from dataclasses import dataclass
import importlib
from pathlib import Path

import numpy as np
import pandas as pd


def _rdkit_available() -> bool:
    return importlib.util.find_spec("rdkit") is not None


@dataclass
class SarResult:
    summary_table: pd.DataFrame
    image_paths: list[Path]


def _extract_importances(model) -> np.ndarray | None:
    if hasattr(model, "feature_importances_"):
        return np.asarray(model.feature_importances_)
    if hasattr(model, "coef_"):
        coef = np.asarray(model.coef_)
        if coef.ndim == 2:
            coef = coef[0]
        return np.abs(coef)
    return None


def generate_sar_outputs(
    model_pipeline,
    feature_columns: list[str],
    bit_infos: list[dict[int, list[tuple[int, int]]]] | None,
    smiles: pd.Series,
    output_dir: Path,
    top_n: int = 10,
) -> SarResult:
    model = model_pipeline.named_steps["model"]
    importances = _extract_importances(model)

    summary_rows = []
    image_paths: list[Path] = []

    if importances is None:
        summary = pd.DataFrame(columns=["feature", "importance", "notes"])
        return SarResult(summary_table=summary, image_paths=image_paths)

    top_indices = np.argsort(importances)[::-1][:top_n]
    for idx in top_indices:
        feature = feature_columns[idx]
        summary_rows.append({
            "feature": feature,
            "importance": float(importances[idx]),
            "notes": "fingerprint" if feature.startswith("ECFP_") else "descriptor",
        })

    summary = pd.DataFrame(summary_rows)

    if bit_infos and _rdkit_available():
        chem = importlib.import_module("rdkit.Chem")
        draw = importlib.import_module("rdkit.Chem.Draw")
        output_dir.mkdir(parents=True, exist_ok=True)
        for i, idx in enumerate(top_indices):
            feature = feature_columns[idx]
            if not feature.startswith("ECFP_"):
                continue
            bit = int(feature.split("_")[1])
            for row_index, bit_info in enumerate(bit_infos):
                if bit not in bit_info:
                    continue
                mol = chem.MolFromSmiles(smiles.iloc[row_index])
                if mol is None:
                    continue
                atom_ids = [atom_id for atom_id, _ in bit_info[bit]]
                img = draw.MolToImage(mol, highlightAtoms=atom_ids)
                path = output_dir / f"sar_bit_{bit}_mol_{row_index}.png"
                img.save(path)
                image_paths.append(path)
                summary.loc[summary["feature"] == feature, "notes"] = "fingerprint-mapped"
                break
    return SarResult(summary_table=summary, image_paths=image_paths)
