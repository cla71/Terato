# Terato

Teratogenicity SAR Visualizer & Predictor is a local, PyCharm-compatible Python tool for predictive toxicology and SAR interpretation. It supports heterogeneous assay inputs, multiple ML algorithms, and local substructure interpretation for mechanistic insights.

## Features

- Trains multiple ML models (Random Forest, Gradient Boosting, Logistic Regression, SVM; optional XGBoost/LightGBM if installed)
- Uses Morgan fingerprints + numeric descriptors + lineage assay aggregates
- Automatically selects the best model based on ROC-AUC
- Produces performance metrics, classification report, and SAR summaries
- Local-only outputs (no APIs, no cloud dependencies)

## Installation

```bash
pip install -r requirements.txt
```

Optional:
- `rdkit` for Morgan fingerprints and SAR substructure mapping
- `xgboost` / `lightgbm` for additional models

## Usage

### Demo data

Sample CSVs are available under `data/`:

- `data/demo_literature.csv` (training example with literature headers)
- `data/demo_novel.csv` (prediction example with CC/TS assay headers)

### Train

```bash
python -m terato.cli train --train data/demo_literature.csv --outdir outputs
```

Options:
- `--radius` (default 2)
- `--bits` (default 2048)
- `--no-fingerprints` to disable RDKit-based fingerprints

Outputs:
- `model_bundle.joblib`
- `model_metadata.json`
- `model_performance.csv`
- `classification_report.txt`
- `sar_summary.csv`
- `sar_outputs/` (images if RDKit is installed)

### Predict

```bash
python -m terato.cli predict --model outputs/model_bundle.joblib --input data/demo_novel.csv --outdir predictions
```

Outputs:
- `predictions.csv`

## Input Data Expectations

### Training dataset

Required:
- SMILES column: `SMILES` / `Smiles` / `smiles`
- Label column: `Classification`, `Status`, `Teratogenic`, `Teratogenicity`, `Label`, `Class`, or `ClassLabel`

Optional:
- Lineage analysis columns: `LPM_Analysis`, `DE_Analysis`, `NR_Analysis`, `NC_Analysis`
- CC/TS assay activity columns to derive lineage activity (e.g., `LPM_CC_Active`, `LPM_TS_IC50`)
- Legacy assay column headers (e.g., `LPMCCAct`, `LPMTSIC50`, `DECCAct`, `DETSIC50`, `NRCCAct`, `NRTSIC50`, `NCCCAct`, `NCTSIC50`)
- Any numeric descriptors (Quick Properties, cLogP, MW, PSA, etc.)

### Prediction dataset

Required:
- SMILES column

Optional:
- Lineage activity columns or CC/TS assay readouts (including legacy headers such as `LPMCCIC50`, `LPMCCAct`, `LPMTSIC50`, `LPMTSAct`)
- Numeric descriptors

All extra columns are ignored safely. Missing values are imputed during model fitting.
