# Respiratory Disease Classification

This project focuses on classifying respiratory diseases using signal and audio-based data, with a pipeline that includes data preprocessing, feature extraction, model training, evaluation, and explainability analysis.

## Repository Structure

```text
Respiratory-Disease-Classification/
├── .vscode/                          # VS Code workspace configuration
├── .gitignore                        # Git ignore rules
├── .gitattributes                    # Git attributes
├── README.md                         # Project overview and repository layout
├── requirements.txt                  # Core Python dependencies
├── requirements-post-05-optional.txt # Optional dependencies for post-processing tasks
├── app/                              # Application-level project files
│   └── .gitkeep
├── data/                             # Dataset storage and processed outputs
│   ├── raw/                          # Original/raw dataset files
│   ├── interim/                      # Intermediate cleaned or validated data files
│   │   ├── audio_info.csv
│   │   ├── cycles.csv
│   │   ├── filename_validation.csv
│   │   ├── patients.csv
│   │   └── recordings.csv
│   └── processed/                    # Final cleaned/processed datasets for modeling
│       ├── audio_info_clean.csv
│       ├── cleaning_summary.csv
│       ├── cycles_clean.csv
│       ├── patients_clean.csv
│       ├── processed_cycles.csv
│       └── processing_errors.csv
├── models/                           # Saved model artifacts and outputs
│   ├── deep_learning/
│   └── traditional/
├── notebooks/                       # Exploratory and modeling notebooks
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
│   └── 16_final_model_comparison.ipynb
├── outputs/                          # Output artifacts from experiment runs
│   └── post_05/
├── reports/                          # Reporting and visualization assets
│   ├── figures/
│   ├── results/
│   └── tables/
├── src/                              # Core source code
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loader.py
│   │   ├── parser.py
│   │   └── validator.py
│   ├── evaluation/
│   │   ├── __init__.py
│   │   └── metrics.py
│   ├── features/
│   │   ├── __init__.py
│   │   ├── frequency_domain.py
│   │   ├── spectral.py
│   │   └── time_domain.py
│   ├── mining/
│   │   ├── __init__.py
│   │   ├── association.py
│   │   ├── clustering.py
│   │   └── dimensionality_reduction.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── cnn.py
│   │   ├── mil.py
│   │   └── traditional_ml.py
│   ├── post_05/
│   │   ├── __init__.py
│   │   ├── classical.py
│   │   ├── cnn_utils.py
│   │   ├── common.py
│   │   ├── explainability.py
│   │   ├── features.py
│   │   ├── mining.py
│   │   ├── patient_level.py
│   │   ├── selection.py
│   │   └── splitting.py
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   ├── audio.py
│   │   └── segmentation.py
│   ├── visualization/
│   │   ├── __init__.py
│   │   └── plots.py
│   └── __pycache__/
├── tests/                            # Test files
│   └── .gitkeep
├── .venv/                            # Local Python environment
└── .DS_Store                         # macOS metadata file
```

## Project Flow

The codebase is organized into a modular ML pipeline:

- Data loading and validation live under `src/data/`
- Feature engineering and mining logic live under `src/features/` and `src/mining/`
- Model architectures and training code live under `src/models/`
- Evaluation metrics and reporting are under `src/evaluation/`
- Notebooks in `notebooks/` walk through experimentation and analysis
- Final outputs are stored in `reports/`, `outputs/`, and `models/`

## Getting Started

1. Create and activate a Python environment.
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Explore the notebooks in `notebooks/` to understand the workflow.
4. Run scripts or training modules under `src/` for preprocessing, modeling, and evaluation.
