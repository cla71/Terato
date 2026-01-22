"""Feature generation for teratogenicity modeling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import importlib

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FeatureConfig:
    fingerprint_radius: int
    fingerprint_bits: int
    use_fingerprints: bool
    descriptor_columns: list[str]
    lineage_columns: list[str]


def _rdkit_available() -> bool:
    return importlib.util.find_spec("rdkit") is not None


def featurize_smiles(smiles: pd.Series, radius: int, bits: int) -> tuple[np.ndarray, list[dict[int, list[tuple[int, int]]]]]:
    if not _rdkit_available():
        raise RuntimeError("RDKit is required for fingerprint generation but is not installed.")

    rdkit = importlib.import_module("rdkit")
    chem = importlib.import_module("rdkit.Chem")
    all_chem = importlib.import_module("rdkit.Chem.AllChem")

    arrays = []
    bit_infos: list[dict[int, list[tuple[int, int]]]] = []
    for value in smiles.fillna(""):
        mol = chem.MolFromSmiles(value)
        if mol is None:
            arrays.append(np.zeros(bits, dtype=int))
            bit_infos.append({})
            continue
        bit_info: dict[int, list[tuple[int, int]]] = {}
        fp = all_chem.GetMorganFingerprintAsBitVect(mol, radius, nBits=bits, bitInfo=bit_info)
        arr = np.zeros(bits, dtype=int)
        chem.DataStructs.ConvertToNumpyArray(fp, arr)
        arrays.append(arr)
        bit_infos.append(bit_info)
    return np.vstack(arrays), bit_infos


def build_feature_matrix(
    df: pd.DataFrame,
    smiles_column: str,
    descriptor_columns: list[str],
    lineage_columns: list[str],
    fingerprint_radius: int = 2,
    fingerprint_bits: int = 2048,
    use_fingerprints: bool = True,
) -> tuple[pd.DataFrame, FeatureConfig, list[dict[int, list[tuple[int, int]]]] | None]:
    descriptor_frame = df[descriptor_columns + lineage_columns].copy()
    descriptor_frame = descriptor_frame.apply(pd.to_numeric, errors="coerce")

    bit_info = None
    if use_fingerprints:
        fp_matrix, bit_info = featurize_smiles(df[smiles_column], fingerprint_radius, fingerprint_bits)
        fp_cols = [f"ECFP_{idx}" for idx in range(fp_matrix.shape[1])]
        fp_frame = pd.DataFrame(fp_matrix, columns=fp_cols, index=df.index)
        features = pd.concat([fp_frame, descriptor_frame], axis=1)
    else:
        features = descriptor_frame

    config = FeatureConfig(
        fingerprint_radius=fingerprint_radius,
        fingerprint_bits=fingerprint_bits,
        use_fingerprints=use_fingerprints,
        descriptor_columns=descriptor_columns,
        lineage_columns=lineage_columns,
    )
    return features, config, bit_info
