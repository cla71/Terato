"""Data ingestion and schema harmonization utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd

LINEAGE_COLUMNS = ["LPM_Analysis", "DE_Analysis", "NR_Analysis", "NC_Analysis"]
LITERATURE_LABEL_COLUMNS = [
    "Classification",
    "Teratogenic",
    "Teratogenicity",
    "Label",
    "label",
    "Class",
    "ClassLabel",
    "Status",
]
SMILES_COLUMNS = ["SMILES", "Smiles", "smiles"]


@dataclass(frozen=True)
class DatasetSchema:
    smiles_column: str
    label_column: str | None
    lineage_columns: list[str]
    numeric_descriptor_columns: list[str]


def infer_smiles_column(columns: Iterable[str]) -> str:
    for candidate in SMILES_COLUMNS:
        if candidate in columns:
            return candidate
    raise ValueError("SMILES column not found. Expected one of: SMILES, Smiles, smiles.")


def infer_label_column(columns: Iterable[str]) -> str | None:
    for candidate in LITERATURE_LABEL_COLUMNS:
        if candidate in columns:
            return candidate
    return None


def normalize_lineage_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in LINEAGE_COLUMNS:
        if col in df.columns:
            continue
        lower_map = {c.lower(): c for c in df.columns}
        if col.lower() in lower_map:
            df[col] = df[lower_map[col.lower()]]
    return df


def derive_lineage_from_cc_ts(df: pd.DataFrame) -> pd.DataFrame:
    """Derive lineage-level activity from CC/TS assay signals when available."""
    df = df.copy()
    lineage_map = {
        "LPM": "LPM_Analysis",
        "DE": "DE_Analysis",
        "NR": "NR_Analysis",
        "NC": "NC_Analysis",
    }
    legacy_map = {
        "LPM": {"cc_active": "LPMCCAct", "ts_active": "LPMTSAct", "cc_ic50": "LPMCCIC50", "ts_ic50": "LPMTSIC50"},
        "DE": {"cc_active": "DECCAct", "ts_active": "DETSAct", "cc_ic50": "DECCIC50", "ts_ic50": "DETSIC50"},
        "NR": {"cc_active": "NRCCAct", "ts_active": "NRTSAct", "cc_ic50": "NRCCIC50", "ts_ic50": "NRTSIC50"},
        "NC": {"cc_active": "NCCCAct", "ts_active": "NCTSAct", "cc_ic50": "NCCCIC50", "ts_ic50": "NCTSIC50"},
    }
    for short, lineage_col in lineage_map.items():
        if lineage_col in df.columns:
            continue
        cc_flag = f"{short}_CC_Active"
        ts_flag = f"{short}_TS_Active"
        cc_ic50 = f"{short}_CC_IC50"
        ts_ic50 = f"{short}_TS_IC50"
        legacy = legacy_map.get(short, {})
        cc_flag_legacy = legacy.get("cc_active")
        ts_flag_legacy = legacy.get("ts_active")
        cc_ic50_legacy = legacy.get("cc_ic50")
        ts_ic50_legacy = legacy.get("ts_ic50")
        available = [
            col
            for col in (
                cc_flag,
                ts_flag,
                cc_ic50,
                ts_ic50,
                cc_flag_legacy,
                ts_flag_legacy,
                cc_ic50_legacy,
                ts_ic50_legacy,
            )
            if col and col in df.columns
        ]
        if not available:
            continue
        if cc_flag in df.columns or ts_flag in df.columns or cc_flag_legacy in df.columns or ts_flag_legacy in df.columns:
            flags = pd.DataFrame({
                "cc": df.get(cc_flag) if cc_flag in df.columns else df.get(cc_flag_legacy),
                "ts": df.get(ts_flag) if ts_flag in df.columns else df.get(ts_flag_legacy),
            })
            df[lineage_col] = flags.max(axis=1, skipna=True)
        elif cc_ic50 in df.columns or ts_ic50 in df.columns or cc_ic50_legacy in df.columns or ts_ic50_legacy in df.columns:
            ic50 = pd.DataFrame({
                "cc": df.get(cc_ic50) if cc_ic50 in df.columns else df.get(cc_ic50_legacy),
                "ts": df.get(ts_ic50) if ts_ic50 in df.columns else df.get(ts_ic50_legacy),
            })
            df[lineage_col] = (ic50.min(axis=1, skipna=True) <= 10).astype("float")
    return df


def identify_numeric_descriptors(df: pd.DataFrame, exclude: Iterable[str]) -> list[str]:
    numeric_cols = [
        col
        for col in df.columns
        if pd.api.types.is_numeric_dtype(df[col]) and col not in set(exclude)
    ]
    return numeric_cols


def load_dataset(path: str, training: bool = True) -> tuple[pd.DataFrame, DatasetSchema]:
    df = pd.read_csv(path)
    df = normalize_lineage_columns(df)
    df = derive_lineage_from_cc_ts(df)

    smiles_column = infer_smiles_column(df.columns)
    label_column = infer_label_column(df.columns) if training else None
    lineage_cols = [col for col in LINEAGE_COLUMNS if col in df.columns]

    exclude_cols = [smiles_column] + lineage_cols
    if label_column:
        exclude_cols.append(label_column)

    numeric_descriptor_columns = identify_numeric_descriptors(df, exclude_cols)
    schema = DatasetSchema(
        smiles_column=smiles_column,
        label_column=label_column,
        lineage_columns=lineage_cols,
        numeric_descriptor_columns=numeric_descriptor_columns,
    )
    return df, schema
