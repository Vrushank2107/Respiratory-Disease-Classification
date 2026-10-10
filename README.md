# Respiratory Disease Classification

This project classifies respiratory diseases from lung-sound audio recordings sourced from the [ICBHI 2017 Challenge](https://bhichallenge.med.auth.gr/) dataset. It covers the full ML pipeline — data cleaning, exploratory analysis, audio preprocessing, feature engineering, feature selection, classical ML, deep learning (CNN / multi-feature CNN / CNN-LSTM), patient-level aggregation, and explainability (Grad-CAM) — plus a locally-run research dashboard for exploring saved results and running inference on new audio.

## Repository Structure

```text
Respiratory-Disease-Classification/
├── .gitignore
├── .gitattributes
├── README.md                               
├── requirements.txt                     
├── vercel.json                            
│
├── data/
│   ├── raw/                               # Original ICBHI dataset (git-ignored)
│   │   └── ICBHI/
│   ├── interim/                           # Intermediate parsed / validated CSVs
│   │   ├── audio_info.csv
│   │   ├── cycles.csv
│   │   ├── cycles_validation.csv
│   │   ├── filename_validation.csv
│   │   ├── patients.csv
│   │   └── recordings.csv
│   └── processed/                         # Final cleaned data, features, and model outputs
│       ├── audio_info_clean.csv
│       ├── cleaning_summary.csv
│       ├── cycles_clean.csv
│       ├── patients_clean.csv
│       ├── processed_cycles.csv
│       ├── processing_errors.csv
│       ├── recordings_clean.csv
│       ├── analysis/                      # EDA outputs
│       ├── audio/                         # Preprocessed audio cycles (git-ignored)
│       ├── classical_ml/                  # Classical ML comparison CSVs
│       ├── deep_learning/                 # DL metrics, predictions & checkpoints
│       │   ├── cnn_baseline/
│       │   ├── cnn_lstm/
│       │   ├── multifeature_cnn/
│       │   ├── multifeature_cnn_12b/
│       │   └── patient_level_prediction/
│       ├── eda/
│       ├── features/                      # Extracted feature matrices
│       ├── feature_selection/             # Feature importance & selection outputs
│       └── pca_clustering/                # PCA / clustering results
│
├── models/
│   ├── deep_learning/                     # Saved PyTorch checkpoints (.pt)
│   │   ├── cnn_lstm/
│   │   ├── lightweight_mel_cnn/
│   │   ├── multifeature_cnn/
│   │   └── regularized_multifeature_cnn/
│   └── traditional_ml/                   # Saved scikit-learn estimators (.joblib)
│       ├── classical_ml_metadata.json
│       ├── label_encoder.joblib
│       ├── logistic_regression.joblib
│       ├── random_forest.joblib
│       ├── scaler.joblib
│       ├── selected_features.json
│       └── svm.joblib
│
├── notebooks/                             # End-to-end experiment notebooks
│   ├── 01_dataset_understanding.ipynb
│   ├── 02_data_cleaning.ipynb
│   ├── 03_eda.ipynb
│   ├── 04_audio_preprocessing.ipynb
│   ├── 05_feature_engineering.ipynb
│   ├── 06_feature_analysis.ipynb
│   ├── 07_feature_selection.ipynb
│   ├── 08_pca_and_clustering.ipynb
│   ├── 09_classical_ml.ipynb
│   ├── 10_deep_learning_data_preparation.ipynb
│   ├── 11_cnn_baseline.ipynb
│   ├── 12_multifeature_cnn.ipynb
│   ├── 13_cnn_lstm_optional.ipynb
│   ├── 14_patient_level_prediction.ipynb
│   ├── 15_xai.ipynb
│   ├── 16_final_model_comparison.ipynb
│   ├── 17_Final_Artifact_Generation_and_Validation.ipynb
│   └── 18_model_validation.ipynb
│
├── reports/
│   ├── artifact_inventory.csv             # Full inventory of generated artifacts
│   ├── experiment_manifest.json           # Experiment metadata & provenance
│   ├── final_artifacts/                   # Consolidated comparison tables & validation
│   │   ├── cycle_level_model_comparison.csv
│   │   ├── patient_level_model_comparison.csv
│   │   ├── final_results_report.md
│   │   └── … (24 validation / registry files)
│   └── validation/
│       ├── model_validation_summary.csv
│       └── multifeature_performance_metrics.csv
│
├── tests/
│   └── test_final_artifacts.py            # Pytest suite for artifact integrity
│
├── app/                                   # Research dashboard (FastAPI + Vite/React/TS)
│   ├── README.md
│   ├── artifacts/
│   ├── docs/
│   │   ├── api.md
│   │   ├── architecture.md
│   │   ├── artifact_mapping.md
│   │   └── vercel-deployment.md
│   ├── backend/                           # FastAPI inference & artifact API
│   │   ├── main.py
│   │   ├── schemas.py
│   │   ├── requirements.txt               # Backend-only dependencies
│   │   ├── services/
│   │   │   ├── audio_processing.py
│   │   │   ├── classical.py
│   │   │   ├── lightweight_cnn.py
│   │   │   └── models.py
│   │   └── tests/
│   └── frontend/                          # Vite + React + TypeScript UI
│       ├── package.json
│       ├── vite.config.ts
│       ├── tsconfig.json
│       └── src/
│
└── .venv/                                 # Local Python virtual environment
```

## Pipeline Overview

| Stage | Notebooks | Description |
|-------|-----------|-------------|
| **Data Ingestion** | 01 | Parse ICBHI filenames, patient demographics, and respiratory cycle annotations |
| **Data Cleaning** | 02 | Validate and clean patient, recording, and cycle-level data |
| **EDA** | 03 | Class distributions, recording durations, disease breakdowns |
| **Audio Preprocessing** | 04 | Waveform segmentation into respiratory cycles, resampling |
| **Feature Engineering** | 05 | Time-domain, frequency-domain, and spectral features per cycle |
| **Feature Analysis** | 06 | Correlation analysis, distribution checks, feature statistics |
| **Feature Selection** | 07 | Mutual information, importance ranking, and final feature set |
| **PCA & Clustering** | 08 | Dimensionality reduction and unsupervised cluster analysis |
| **Classical ML** | 09 | Random Forest, SVM, and Logistic Regression training and evaluation |
| **DL Data Prep** | 10 | Spectrogram generation, train/val/test splitting for neural models |
| **CNN Baseline** | 11 | Lightweight Mel-spectrogram CNN |
| **Multi-feature CNN** | 12 | CNN with Log-Mel + MFCC + Chroma input channels |
| **CNN-LSTM** | 13 | Temporal CNN-LSTM architecture (optional) |
| **Patient-Level** | 14 | Majority-vote and mean-probability patient-level aggregation |
| **Explainability** | 15 | Grad-CAM visualizations for CNN models |
| **Comparison** | 16 | Consolidated cycle-level model comparison across all architectures |
| **Artifact Generation** | 17 | Final artifact packaging, inventory, and validation |
| **Model Validation** | 18 | End-to-end checkpoint integrity and metric reproduction checks |

## Models

### Classical ML (scikit-learn)
- **Random Forest** — Accuracy: 87.4%, Balanced Accuracy: 42.6%, Macro F1: 28.0%
- **SVM** — Accuracy: 80.7%, Balanced Accuracy: 45.8%, Macro F1: 26.6%
- **Logistic Regression** — Accuracy: 64.2%, Balanced Accuracy: 41.2%, Macro F1: 21.8%

### Deep Learning (PyTorch)
- **Lightweight Mel CNN** — Accuracy: 71.7%, Balanced Accuracy: 31.0%, Macro F1: 21.3%
- **Multi-Feature CNN (12A)** — Accuracy: 85.5%, Balanced Accuracy: 16.6%, Macro F1: 15.4%
- **Regularized Multi-Feature CNN (12B)** — Accuracy: 85.7%, Balanced Accuracy: 16.7%, Macro F1: 15.4%
- **CNN-LSTM** — Accuracy: 81.4%, Balanced Accuracy: 22.6%, Macro F1: 14.6%

### Patient-Level Aggregation
- **Majority Vote / Mean Probability** — Accuracy: 56.0%, Balanced Accuracy: 33.3%

> **Note:** High accuracy with low balanced accuracy reflects significant class imbalance (COPD dominates the dataset). See `reports/final_artifacts/final_results_report.md` for full details.

## Getting Started

### Prerequisites

- Python 3.14+ (tested on 3.14.7)
- Node.js 18+ (for the dashboard frontend)
- macOS / Linux

### Installation

```bash
# Clone the repository
git clone https://github.com/<your-username>/Respiratory-Disease-Classification.git
cd Respiratory-Disease-Classification

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate

# Install all Python dependencies
pip install -r requirements.txt
```

### Running the Notebooks

```bash
source .venv/bin/activate
jupyter lab
# Open notebooks in notebooks/ and run them in order (01 → 18)
```

### Running the Research Dashboard

The dashboard provides a web UI for exploring saved results and running inference on uploaded `.wav` audio files. It does **not** retrain models.

```bash
# Terminal 1 — Backend
source .venv/bin/activate
pip install -r app/backend/requirements.txt
uvicorn app.backend.main:app --reload --host 127.0.0.1 --port 8000

# Terminal 2 — Frontend
cd app/frontend
npm install
npm run dev
```

Open http://localhost:5173 for the UI, http://localhost:8000/docs for the API docs.

### Running Tests

```bash
source .venv/bin/activate
python -m pytest tests/test_final_artifacts.py       # Artifact integrity
python -m pytest app/backend/tests                   # Backend API tests
cd app/frontend && npm run build                     # Frontend type-check & build
```

## Key Technologies

| Category | Libraries |
|----------|-----------|
| **Data & Analysis** | pandas, numpy, scipy, openpyxl, imbalanced-learn |
| **Audio Processing** | librosa, soundfile, soxr, PyWavelets |
| **Visualization** | matplotlib, seaborn, mlxtend |
| **Classical ML** | scikit-learn, joblib |
| **Deep Learning** | PyTorch (torch), numba |
| **Notebooks** | JupyterLab, ipywidgets |
| **Backend API** | FastAPI, uvicorn, pydantic, python-multipart |
| **Frontend** | React 19, TypeScript, Vite, Recharts, Lucide React |
| **Testing** | pytest, httpx |

## Deployment

The React frontend can be hosted on Vercel, with the FastAPI backend deployed separately to Google Cloud Run from the root `Dockerfile`. See `app/docs/cloud-run-deployment.md` for deployment steps. For local backend development, run `docker compose up --build` from the repository root.

## Project Notes

- The ICBHI raw dataset (`data/raw/ICBHI/`) is git-ignored; obtain it from the [ICBHI 2017 Challenge](https://bhichallenge.med.auth.gr/).
- Processed audio cycles (`data/processed/audio/cycles/`) are also git-ignored due to size.
- Notebook source is treated as read-only by the dashboard. Uploaded audio is processed locally and not persisted.
- Model predictions are research outputs, **not** clinical diagnoses.
