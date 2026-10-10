# Respiratory Disease Classification — Full Project Documentation

**Project:** Respiratory Disease Classification from Lung Sounds
**Dataset:** ICBHI 2017 respiratory sound dataset
**Documentation audit:** 11 October 2026
**Scope:** Notebooks 01–18, saved research artifacts, RespiraLab web application, and its Vercel/Render deployment

> **Research-use notice:** This is an academic machine-learning project. Predictions and score values are experimental, are not calibrated clinical confidence, and must not be used to diagnose, treat, or triage a person.

## 1. Project overview

The project investigates whether acoustic information in respiratory recordings can be used to classify a patient's respiratory disease. Its workflow starts with ICBHI patient, recording, and cycle annotations; validates and cleans those data; transforms audio into handcrafted or time-frequency features; evaluates classical and neural models; aggregates cycle predictions to the patient level; and records explainability and validation artifacts.

The project has two related deliverables:

1. **Research pipeline:** 18 ordered Jupyter notebooks, the intermediate and processed tables they produce, model checkpoints/estimators, evaluation outputs, and validation reports.
2. **RespiraLab application:** a React/Vite dashboard backed by a FastAPI inference and artifact API. It reads saved research results and runs saved models on WAV uploads. A matching ICBHI-style cycle-annotation `.txt` file is optional: annotations extract known cycles; without them, the API scores consecutive five-second windows, not detected cycles. It does not train models from the browser.

### Key verified dataset totals

The cleaning notebook and `data/processed/cleaning_summary.csv` report:

| Unit | Count | Definition |
|---|---:|---|
| Patients | 126 | Distinct patient identifiers |
| Recordings | 920 | Distinct WAV recordings; the cleaning audit found 920 unique audio hashes |
| Respiratory cycles | 6,898 | Annotated cycle rows retained after cleaning |

These units are different levels of the data hierarchy: **patient → recording → respiratory cycle**. A patient's disease label is repeated across that patient's cycle records for cycle-level training, but the sound label (Normal, Crackles, Wheezes, or Both) belongs to the respiratory cycle.

### Diagnosis and respiratory sound labels

The eight disease classes are mapped in `data/processed/deep_learning/disease_class_mapping.csv`:

| Class ID | Diagnosis | Patients |
|---:|---|---:|
| 0 | Asthma | 1 |
| 1 | Bronchiectasis | 7 |
| 2 | Bronchiolitis | 6 |
| 3 | COPD | 64 |
| 4 | Healthy | 26 |
| 5 | LRTI | 2 |
| 6 | Pneumonia | 6 |
| 7 | URTI | 14 |

The patient distribution is substantially imbalanced: COPD accounts for 64 of 126 patients. The 6,898 cycle sound labels are Normal (3,642), Crackles (1,864), Wheezes (886), and Both (506). The imbalance is central to how results should be read: raw accuracy can be high while balanced accuracy and macro F1 remain low.

## 2. Research notebooks, in order

The notebooks are under `notebooks/`. Their numbered order describes the intended workflow. Notebooks 17 and 18 consolidate and validate existing artifacts; they do not replace the original model experiments.

### Notebook 01 — Dataset understanding

**File:** `notebooks/01_dataset_understanding.ipynb`

This notebook establishes the data model before training. It parses recording names and ICBHI annotation text into patient, recording, and respiratory-cycle tables; reads diagnosis/demographic information; associates each recording with chest location, acquisition mode, equipment, path, and official split; and validates cycle boundaries and labels.

The notebook reports 126 patients, 920 WAV recordings, and 6,898 annotated cycles. It preserves the distinction between patient diagnosis and cycle sound events. It also reviews the official recording-level train/test assignment. That assignment has 79 patients represented in the training partition and 49 in the test partition, with patients 156 and 218 present in both partitions. This overlap is why the later deep-learning work creates an explicitly patient-disjoint split rather than treating the official recording split as patient-disjoint.

**Main artifacts:** `data/interim/patients.csv`, `recordings.csv`, `cycles.csv`, `audio_info.csv`, `cycles_validation.csv`, `filename_validation.csv`, and related validation tables. These are intermediate tables, not the final cleaned tables.

### Notebook 02 — Data cleaning and integrity checks

**File:** `notebooks/02_data_cleaning.ipynb`

The raw WAV and annotation files are treated as source material and left unchanged. The notebook validates patient, recording, audio, and cycle tables for duplicate rows, missing metadata, unreadable WAVs, filename/metadata consistency, cycle time bounds, invalid durations, overlapping cycles, and recordings without cycles. It records checks instead of silently discarding questionable records.

The saved cleaning summary reports 126 patients, 920 recordings, and 6,898 cycles retained; zero duplicate patients, duplicate recordings, or duplicate cycles; zero unreadable WAV files; 920 unique audio hashes; no recordings without cycles; no invalid cycle timing; and no overlapping cycles. Patient 223 has incomplete demographic fields but has a diagnosis and valid recording/cycle data, so its audio examples remain useful for the sound task.

**Main artifacts:** cleaned tables under `data/processed/` such as `patients_clean.csv`, `recordings_clean.csv`, `cycles_clean.csv`, `audio_info_clean.csv`, `processed_cycles.csv`, and `cleaning_summary.csv`.

### Notebook 03 — Exploratory data analysis

**File:** `notebooks/03_eda.ipynb`

This notebook summarizes the cleaned cohort at the patient, recording, and cycle levels. It explores disease counts; sound-label counts and their relationship to disease; patient sound composition; recordings per patient; age and sex distributions; chest recording locations; acquisition modes and equipment; and sampling-rate distributions. Its purpose is to expose imbalance and data structure before feature and model decisions.

The EDA confirms eight diagnoses, four cycle sound classes, three source sampling rates (4,000, 10,000, and 44,100 Hz), seven chest-location codes, and four recording-equipment types. Disease and sound plots should use the saved distribution tables rather than assuming the two labels describe the same target.

**Main artifacts:** CSV summaries under `data/processed/eda/`, including `disease_distribution.csv`, `sound_distribution.csv`, `disease_sound_counts.csv`, `disease_sound_percent.csv`, `patient_audio_summary.csv`, and `patient_sound_by_disease.csv`.

### Notebook 04 — Audio preprocessing

**File:** `notebooks/04_audio_preprocessing.ipynb`

This notebook turns annotation intervals into individual respiratory-cycle WAVs. It reads the cleaned recording and cycle metadata, inspects source sampling rates, extracts each cycle using its annotated start/end time, converts to mono as needed, resamples to a common 4 kHz rate, applies peak normalization, and checks the written outputs.

The final notebook summary reports all 6,898 input cycles processed, zero failures, zero duplicate processed-cycle IDs, and no missing output cycle audio. The output cycle WAV directory is `data/processed/audio/cycles/`. That generated audio directory is excluded from Git and the Docker build context; a clone of the repository therefore does not contain those individual WAVs. Re-running later training/preparation steps that require cycle audio requires running this preprocessing workflow from the separately obtained raw ICBHI data.

### Notebook 05 — Handcrafted feature engineering

**File:** `notebooks/05_feature_engineering.ipynb`

This notebook produces one tabular feature row per respiratory cycle. It combines 39 handcrafted acoustic features across several groups:

- **Waveform/time statistics:** mean, standard deviation, variance, RMS, maximum, minimum, peak-to-peak range, median, skewness, kurtosis, zero-crossing rate, and energy.
- **Spectrum:** spectral centroid, bandwidth, rolloff, flatness, spectral energy, dominant frequency, and spectral entropy.
- **Band energy:** normalized energy in 0–500, 500–1,000, 1,000–1,500, and 1,500–2,000 Hz bands.
- **MFCC:** mean values for coefficients 1 through 13.
- **Wavelet:** wavelet energy, entropy, and standard deviation.

Patient and label metadata are retained alongside the measurements for analysis, but are not treated as acoustic predictors. No model is trained here. The feature table is the input to Notebooks 06–09.

**Main artifacts:** `data/processed/features/respiratory_cycle_features.csv` and `feature_extraction_errors.csv`.

### Notebook 06 — Feature analysis

**File:** `notebooks/06_feature_analysis.ipynb`

This notebook analyzes the 39 handcrafted features before selecting a subset. It calculates descriptive statistics and feature variance, examines correlation and highly correlated pairs, and compares feature behavior across disease and sound labels. It is exploratory: it does not perform final feature selection, PCA, or model fitting.

The notebook reports a feature-table shape of 6,898 cycles by 52 columns, including metadata and target columns, with no missing or infinite feature values. It explicitly distinguishes the patient-level disease target from the cycle-level sound target. Cycles from the same patient are related observations, so cycle counts must not be misread as independent patient counts.

**Main artifacts:** `data/processed/analysis/feature_statistics.csv`, `feature_variance.csv`, correlation-pair tables, disease/sound feature means, and crackle/wheeze feature analyses.

### Notebook 07 — Feature selection

**File:** `notebooks/07_feature_selection.ipynb`

This notebook ranks and reduces the handcrafted features using correlation-based redundancy checks, ANOVA F-scores, mutual information, Random Forest feature importance, consensus ranking, and redundancy-aware selection. The notebook states that the official test recordings are not used to calculate selection scores or rankings.

The active deployed traditional classifiers use the final 33-feature list in the exact order stored in `models/traditional_ml_patient_disjoint/selected_features.json`. That order is part of the inference contract; changing it without retraining/repackaging the estimators would make their inputs incorrect. The earlier official-split package remains under `models/traditional_ml/` for historical reproducibility.

The selected predictors are: `spectral_entropy`, `mfcc_11_mean`, `zero_crossing_rate`, `dominant_frequency`, `mfcc_13_mean`, `wavelet_entropy`, `mfcc_1_mean`, `spectral_bandwidth`, `spectral_rolloff`, `mfcc_5_mean`, `mfcc_3_mean`, `mfcc_8_mean`, `peak_to_peak`, `mfcc_9_mean`, `mfcc_12_mean`, `mfcc_7_mean`, `mfcc_2_mean`, `energy`, `rms`, `mfcc_6_mean`, `band_energy_500_1000`, `spectral_flatness`, `band_energy_0_500`, `mfcc_4_mean`, `band_energy_1000_1500`, `min`, `mfcc_10_mean`, `skewness`, `mean`, `kurtosis`, `median`, `band_energy_1500_2000`, and `max`.

**Main artifacts:** `data/processed/feature_selection/selected_features.csv`, `final_feature_selection_ranking.csv`, ANOVA/MI scores, Random Forest importance, and consensus ranking.

### Notebook 08 — PCA and clustering

**File:** `notebooks/08_pca_and_clustering.ipynb`

This notebook standardizes the selected feature space and applies PCA, K-Means, and hierarchical clustering. Disease and sound labels are held out of cluster creation and used only afterward to interpret cluster alignment. The notebook compares candidate K values using inertia and silhouette score and examines agreement using post-hoc measures such as Adjusted Rand Index (ARI) and Normalized Mutual Information (NMI).

K-Means with K=2 has the highest silhouette score among the tested K values (approximately 0.527). That separation does **not** mean the clusters recover disease classes: agreement with disease labels is near zero (ARI 0.0014, NMI 0.0242), and agreement with sound labels is also near zero (ARI -0.0093, NMI 0.0087). The unsupervised structure is therefore descriptive and not a disease classifier.

**Main artifacts:** `data/processed/pca_clustering/pca_explained_variance.csv` and `kmeans_cluster_analysis.csv`, plus notebook plots and analyses.

### Notebook 09 — Classical machine learning

**File:** `notebooks/09_classical_ml.ipynb`

The notebook trains and evaluates Logistic Regression, an RBF-kernel SVM, and a Random Forest on the 33 selected handcrafted features. The original saved experiment uses the official split and remains available as a historical result. For the common patient-held-out audit, all three estimators were retrained with feature selection and scaling fit on training patients only; the active serving package is saved in `models/traditional_ml_patient_disjoint/`. Logistic Regression and SVM use the saved scaler; Random Forest does not.

The artifact metadata records Logistic Regression with `C=1`, `max_iter=3000`, and `random_state=42`; an RBF `SVC` with `C=1`, `probability=True`, and balanced class weights; and a 300-tree Random Forest with balanced class weights, `max_features='sqrt'`, and `random_state=42`. The saved inference package comprises three `.joblib` estimators, a scaler, label encoder, selected feature list, and metadata.

The reported test metrics are: Random Forest accuracy 0.8745, balanced accuracy 0.4265, macro F1 0.2796; SVM accuracy 0.8066, balanced accuracy 0.4583, macro F1 0.2656; and Logistic Regression accuracy 0.6415, balanced accuracy 0.4119, macro F1 0.2182. The class imbalance means the accuracy values should not be read as uniform performance across diseases.

**Main artifacts:** `models/traditional_ml/` and `data/processed/classical_ml/`, including per-model predictions, comparison tables, per-disease results, and Random Forest feature importance.

### Notebook 10 — Deep-learning data preparation

**File:** `notebooks/10_deep_learning_data_preparation.ipynb`

This notebook prepares cycle-level inputs for the deep-learning experiments. It uses the preprocessed 4 kHz cycle audio, the patient-disjoint partition, and Log-Mel/MFCC/Chroma transforms. For the Log-Mel path, cycles are represented at a fixed five-second duration: 20,000 samples at 4 kHz. Short examples are zero-padded and long examples are truncated. The Mel configuration uses FFT size 512, hop length 128, 64 Mel bands, and a 0–2,000 Hz range; the resulting spectrogram is 64 by 157 frames.

Training-set statistics are used for normalization. Test examples are not used to calculate normalization statistics. Feature caches and manifests support later notebooks; augmentation is applied in training loaders rather than by altering validation/test inputs.

**Main artifacts:** `data/processed/deep_learning/` manifests, label mapping, training normalization files, cached arrays, and error/summary tables.

### Notebook 11 — Lightweight Mel CNN baseline

**File:** `notebooks/11_cnn_baseline.ipynb`

This is the single-channel neural baseline. It uses a Log-Mel spectrogram input and a compact CNN with convolutional blocks (16, 32, and 64 channels), pooling, global average pooling, and an eight-class output layer. The configuration records Adam, learning rate 0.001, weight decay 0.0001, batch size 32, up to 40 epochs, weighted cross-entropy, validation-loss checkpoint selection, ReduceLROnPlateau, and early stopping. Gaussian-noise and time-shift augmentation are training-only.

The experiment uses a patient-disjoint train/validation/test partition. Its saved test set contains 1,561 cycles from 25 patients. The reported result is accuracy 0.7168, balanced accuracy 0.3102, macro F1 0.2129, and weighted F1 0.7687; the best checkpoint was selected at epoch 32. Among the deep-learning experiments reported here, this model has the strongest balanced accuracy and macro F1, despite lower raw accuracy than the multi-feature networks.

**Main artifacts:** the checkpoint and training config in `models/deep_learning/lightweight_mel_cnn/`; metrics, predictions, history, class weights, split manifests, and architecture metadata in `data/processed/deep_learning/cnn_baseline/`.

### Notebook 12 — Multi-Feature CNN experiments 12A and 12B

**File:** `notebooks/12_multifeature_cnn.ipynb`

Notebook 12 evaluates two related model variants; they are not accidental duplicates. Both receive Log-Mel, MFCC, and Chroma representations through separate CNN branches. Branch representations are concatenated and passed to a fusion classifier. They use the same patient-disjoint split and training-only augmentation as the baseline so the experiments can be compared on the same test examples.

- **12A — Multi-Feature CNN:** original configuration, learning rate 0.001, branch dropout 0.30 and classifier dropout 0.40 in the deployed inference architecture.
- **12B — Regularized Multi-Feature CNN:** experiment intended to address validation-loss deterioration through a lower learning rate (0.0001) and stronger dropout (0.50 in the branches and classifier). It shares the trained feature normalization generated by the notebook's training-only data pipeline.

Both saved test results cover 1,561 cycles. 12A reports accuracy 0.8552, balanced accuracy 0.1664, macro F1 0.1537. 12B reports accuracy 0.8565, balanced accuracy 0.1667, macro F1 0.1538. The small increase in accuracy does not translate into broad class-balanced performance. The very high raw accuracy is strongly influenced by COPD examples.

**Main artifacts:** 12A checkpoint and metadata in `models/deep_learning/multifeature_cnn/`; 12B checkpoint and config in `models/deep_learning/regularized_multifeature_cnn/`; both sets of metrics, predictions, and histories in `data/processed/deep_learning/`.

### Notebook 13 — CNN-LSTM

**File:** `notebooks/13_cnn_lstm_optional.ipynb`

Although the notebook title calls this experiment optional/deferred, the repository contains its trained checkpoint, configuration, metrics, predictions, and history, and the application loads it. The CNN extracts local time-frequency features while retaining the temporal axis; a two-layer LSTM with 64 hidden units models the 157-frame sequence. The configuration records 4 kHz input, five seconds, 64 Mel bands, batch size 32, Adam at 0.001, dropout 0.30, and early stopping.

The saved test covers 1,561 cycles from 25 patients. It reports accuracy 0.8136, balanced accuracy 0.2261, macro F1 0.1465, and weighted F1 0.7867, with best epoch 8. It did not improve the baseline's balanced accuracy or macro F1 in the recorded experiment.

**Main artifacts:** checkpoint/config under `models/deep_learning/cnn_lstm/` and the saved metrics, predictions, history, confusion matrix, and comparison under `data/processed/deep_learning/cnn_lstm/`.

### Notebook 14 — Patient-level prediction

**File:** `notebooks/14_patient_level_prediction.ipynb`

This notebook converts cycle outputs from the Lightweight Mel CNN into one prediction per patient. It compares majority voting across a patient's cycle labels with mean-probability aggregation across that patient's cycles. Patient-level evaluation is reported separately from cycle-level evaluation because the clinical label belongs to the patient, while the model receives cycles.

The test portion contains only 25 patients, so class support is very small for several diseases. Both aggregation methods produce the same predicted class for all 25 saved test patients and therefore have identical saved metrics: accuracy 0.56, balanced accuracy 0.3333, macro precision 0.1328, macro recall 0.25, macro F1 0.1621, and weighted F1 0.4822. Matching results are a property of this test set, not proof that the methods are equivalent in general.

**Main artifacts:** per-patient prediction tables, aggregation comparison, experiment summary, and two confusion matrices under `data/processed/deep_learning/patient_level_prediction/`; consolidated metrics are in `reports/final_artifacts/patient_level_model_comparison.csv`.

### Notebook 15 — Explainable AI

**File:** `notebooks/15_xai.ipynb`

This notebook reuses the Lightweight Mel CNN and generates Grad-CAM-style visual attribution over Log-Mel time-frequency inputs. The saved examples are selected from correctly classified test cycles for classes where such examples exist. It does not retrain the model.

Grad-CAM indicates input regions that influenced the selected model output. It does not establish that a region corresponds to a specific clinical event or prove the prediction is correct. The app displays saved Grad-CAM research examples. For uploaded audio it uses distinct live methods—input-gradient saliency for neural models and single-feature baseline ablation for traditional models—and limits the explanation to the first segment. Live explanations are not Grad-CAM and are not clinical evidence.

**Main artifacts:** images and metadata under `data/processed/deep_learning/xai/`, including `multi_class_xai_summary.csv`, `gradcam_example_metadata.json`, and `gradcam_*.png`.

### Notebook 16 — Final model comparison

**File:** `notebooks/16_final_model_comparison.ipynb`

This notebook consolidates classical cycle-level scores, deep-learning cycle-level results, and patient-level aggregation. It explicitly keeps the evaluation units separate. Classical experiments use the official ICBHI split, while neural experiments use a patient-disjoint test split and patient-level work aggregates predictions for 25 test patients.

The final comparison artifacts mark cross-model split comparability as **unverified**. Accordingly, the table is a record of the reported experiments, not a statistically controlled single leaderboard across every model family. The results favor the Lightweight Mel CNN on the recorded deep-learning balanced accuracy/macro-F1 criteria, while the SVM leads the reported classical balanced accuracy; their test protocols differ.

**Main artifacts:** `reports/final_artifacts/cycle_level_model_comparison.csv`, `patient_level_model_comparison.csv`, `final_results_report.md`, and supporting comparison manifests.

### Notebook 17 — Artifact generation and validation

**File:** `notebooks/17_Final_Artifact_Generation_and_Validation.ipynb`

Notebook 17 discovers existing notebooks, models, images, tables, and reports; categorizes them; registers actual result sources; validates source paths; loads registered results; generates inventories/manifests; and writes a validation summary. It is intentionally conservative: a missing or empty result is an action item, not a zero score or an implied successful experiment. It does not silently retrain models or invent metrics.

**Main artifacts:** `reports/artifact_inventory.csv`, `reports/final_artifacts/artifact_inventory.csv`, model registries/manifests, prediction/source validation tables, and `final_results_report.md`.

### Notebook 18 — Model validation

**File:** `notebooks/18_model_validation.ipynb`

The final notebook validates artifacts without retraining. It checks saved file presence, selected-feature counts and estimator input dimensions, checkpoint/config compatibility, class mappings, prediction dimensions and numerical validity, and comparisons between saved and regenerated predictions where supported. It also reviews split information and source artifacts.

Validation establishes that artifacts load and meet the checks implemented in the notebook; it does not establish clinical validity, prove generalization to external populations, or make differing test splits comparable. Validation outputs are retained in `reports/final_artifacts/` and `reports/validation/`.

## 3. Experimental design and results

### Evaluation protocols

Historical notebook results preserve their original experiment splits and are not necessarily directly comparable. In addition, `reports/research_audit/` contains a common patient-held-out comparison of all seven serving model variants: the same 25 held-out patients, 53 recordings, and 1,561 cycles are used across model predictions. Neural model outputs are drawn from the saved patient-disjoint test exports; the three classical estimators were retrained on the shared training patients with feature ranking, pruning, and scaling fit on that training portion only. Patient-level majority-vote metrics are reported separately from cycle-level metrics. The held-out cohort has no Asthma or LRTI patients, so performance for those diagnoses is not established. See [`reports/research_audit/README.md`](../reports/research_audit/README.md) for the detailed protocol and support counts.

### Saved headline metrics

Values below come from the consolidated CSV artifacts; they are reported here, not recalculated independently.

| Experiment | Unit | Accuracy | Balanced accuracy | Macro F1 | Weighted F1 |
|---|---|---:|---:|---:|---:|
| Random Forest | Cycle | 0.8745 | 0.4265 | 0.2796 | 0.8775 |
| SVM | Cycle | 0.8066 | 0.4583 | 0.2656 | 0.8471 |
| Logistic Regression | Cycle | 0.6415 | 0.4119 | 0.2182 | 0.7458 |
| Lightweight Mel CNN | Cycle | 0.7168 | 0.3102 | 0.2129 | 0.7687 |
| Multi-Feature CNN 12A | Cycle | 0.8552 | 0.1664 | 0.1537 | 0.7897 |
| Regularized Multi-Feature CNN 12B | Cycle | 0.8565 | 0.1667 | 0.1538 | 0.7903 |
| CNN-LSTM | Cycle | 0.8136 | 0.2261 | 0.1465 | 0.7867 |
| Majority Vote | Patient | 0.5600 | 0.3333 | 0.1621 | 0.4822 |
| Mean Probability | Patient | 0.5600 | 0.3333 | 0.1621 | 0.4822 |

This table preserves the historical notebook-level experiment results and is not the app's common-split ranking. For a fairer seven-model comparison, use the shared patient-held-out results summarized in `reports/research_audit/README.md` and displayed separately in Model Comparison. In either protocol, consider balanced accuracy, macro metrics, per-class support, and confusion matrices alongside raw accuracy.

## 4. RespiraLab application

### Architecture

```text
Browser
  └── React 19 + TypeScript + Vite frontend (Vercel)
        └── HTTPS JSON / multipart requests
              └── FastAPI backend (Render Docker web service)
                    ├── saved research CSV/JSON reports
                    ├── scikit-learn estimators + fixed 33-feature order
                    ├── PyTorch checkpoints + training normalization
                    └── WAV decoding, preprocessing, and inference
```

The API derives its repository root from the backend module path, so its data/model paths do not depend on a caller's current working directory. Model checkpoints are loaded lazily and cached. Deep-learning checkpoints are loaded with strict state-dictionary matching and verified class order.

### Frontend pages

The Vite/React interface contains seven consolidated sections. Live audio analysis is part of the Predict workflow and appears with the results for that upload:

1. **Overview:** dataset summary, saved comparisons, model readiness, and high-level charts.
2. **Dataset & EDA:** disease/sound distributions, patient/audio summaries, cleaning checks, and saved EDA outputs.
3. **Audio & features:** saved preprocessing, feature-selection, PCA, and cluster results from Notebooks 04–08.
4. **Model development:** classical and neural training results from Notebooks 09–13.
5. **Evaluation & XAI:** common and historical model comparisons, patient-level metrics, saved Grad-CAM examples, and live per-upload model explanations.
6. **Artifacts & validation:** allowlisted reports, model artifact registry, validation records, and research paper provenance.
7. **Predict audio:** the sole live-inference page, with optional ICBHI timestamp annotations, fixed-window fallback, processing details, scores, and segment-specific explanations.

The layout has responsive desktop and mobile styling. On small screens navigation is horizontally scrollable, content/cards stack, and dense tables can scroll within their own bounded area.

### Backend routes

The implemented API includes:

| Route | Purpose |
|---|---|
| `GET /api/health` | Health/status response |
| `GET /api/models` | Model availability and load validation |
| `GET /api/overview` | Cleaned dataset summary, saved seven-model artifact inventory, and highlights |
| `GET /api/eda/summary` | Saved EDA and cleaning summaries |
| `GET /api/features/summary` | Feature selection/statistic summaries |
| `GET /api/pipeline/summary` | Saved audio/preprocessing and feature pipeline artifacts |
| `GET /api/development/summary` | Saved classical/neural metrics and histories |
| `GET /api/clustering/summary` | PCA and clustering outputs |
| `GET /api/evaluation/models` | Cycle comparisons and model confusion matrices |
| `GET /api/evaluation/cycle-level` | Consolidated cycle-level metrics |
| `GET /api/evaluation/patient-level` | Patient aggregation metrics and matrices |
| `GET /api/xai/summary` | Saved XAI metadata and image names |
| `GET /api/xai/image/{name}` | Allowlisted saved PNG image |
| `GET /api/reports` | Allowlisted reports and URLs |
| `GET /api/reports/download/{name}` | Download an allowlisted report |
| `GET /api/system/status` | Saved artifact inventory and validation records; no checkpoint loading |
| `POST /api/audio/preview` | Audio waveform and derived feature preview |
| `POST /api/predict` | One selected model prediction |
| `POST /api/predict/compare` | Predictions from multiple selected models |

### Uploaded-audio prediction flow

1. The user selects a readable WAV (20 MiB and 120 seconds maximum by default), optionally attaches a matching ICBHI-style timestamp `.txt`, and chooses one or more available models in **Predict audio**. Both attachments can be removed before submitting.
2. The browser sends a multipart request to `/api/predict/compare` with the WAV, selected model IDs, and optional annotation file.
3. The backend validates the file and decoded signal, rejects silent/non-finite audio, and reads its duration, source sample rate, and channel count.
4. If annotations are supplied, their start/end times are validated and used to extract respiratory cycles. Otherwise, the complete recording is resampled and split into consecutive five-second windows; the final window can be shorter. The windows are not cycle detection and predictions on these windows are exploratory because training/evaluation use respiratory cycles.
5. Each cycle/window is downmixed to mono, resampled to 4 kHz, and peak-normalized. Traditional estimators compute the selected 33 acoustic features per segment, sharing feature extraction where possible. Logistic Regression and SVM use the saved scaler; Random Forest is unscaled.
6. Neural models receive 20,000 samples (five seconds), padding or truncating each segment as necessary. The API calculates Log-Mel and, for Multi-Feature CNN variants, MFCC and Chroma inputs, applies saved training normalization, and runs the cached CPU model.
7. Every segment is scored. Recording-level class scores are the unweighted mean of segment scores, and the predicted class is the top mean score. These outputs are not calibrated probabilities or clinical confidence.
8. The response includes each segment's predicted class, timing/count context, preprocessing steps, a full-source waveform preview, first-segment prepared input and features, and an input-specific explanation for the first segment only. Neural explanations use input-gradient saliency; traditional explanations use feature baseline ablation. Recording scores are the unweighted mean of segment scores when scores are available; a majority segment vote is used when a model does not return class scores. Neither explanations nor outputs establish clinical evidence.
9. Temporary upload files are removed after processing. The application does not retain the submitted audio.

### Saved artifact and count behavior

The CNN Baseline confusion matrix artifact was missing as a standalone CSV even though `cnn_test_predictions.csv` existed. The API now derives the matrix from the saved `true_disease` and `predicted_disease` test columns when the standalone matrix file is absent; the UI labels that source explicitly.

Overview reports the cleaned patient, unique WAV recording, and cycle counts separately. Its seven-row saved model inventory checks artifact paths without loading model weights. Runtime model availability is validated separately when Predict audio opens. Saved validation records are informational and should not be confused with a live health check or scientific validation.

## 5. Deployment history and current hosting

### Frontend: Vercel

The Vite frontend is deployed from the GitHub repository with `app/frontend` as its project root. Vercel serves the built static frontend. `app/frontend/.env.production` sets `VITE_API_URL` to the Render API origin. A Vercel project environment variable with the same name overrides the file and should match that origin.

### Backend: Render

The FastAPI backend is deployed as a Render **Docker Web Service**. Render builds the root `Dockerfile` from the repository root. The Dockerfile uses `python:3.12-slim`, installs `libsndfile`, installs a CPU-only PyTorch wheel, installs the backend requirements, copies the backend plus the models, processed tables, and final reports, and starts Uvicorn on Render's `PORT`.

The `.dockerignore` excludes `.git`, local virtual environments, notebooks, raw data, tests, the frontend and generated audio-cycle directory. The backend's production Vercel origin is in its CORS allowlist; additional origins can be configured with the `FRONTEND_ORIGINS` environment variable.

Current deployment URLs:

- Frontend: [https://respiratory-disease-classification-seven.vercel.app/](https://respiratory-disease-classification-seven.vercel.app/)
- Backend: [https://respiratory-disease-classification.onrender.com/](https://respiratory-disease-classification.onrender.com/)
- Health: [https://respiratory-disease-classification.onrender.com/api/health](https://respiratory-disease-classification.onrender.com/api/health)
- OpenAPI docs: [https://respiratory-disease-classification.onrender.com/docs](https://respiratory-disease-classification.onrender.com/docs)

The API service does not serve the frontend at `/`; a `404` at its root is expected. Check `/api/health` or `/docs` instead. The Render compute plan controls available memory/CPU and whether the service sleeps; verify its current plan in Render before assuming performance or cost.

### Hosting decisions

The first deployment direction was a single Vercel project for frontend and Python backend. Vercel's build log reported a 5,720.12 MB Python function bundle against a 500 MB function-size limit, so the backend was separated from the frontend. Google Cloud Run was evaluated as a container host, but potential billing and account setup made it unsuitable for the user's no-unexpected-charges requirement. The project was then deployed as a Docker service on Render, leaving the frontend on Vercel.

The repository's `app/docs/cloud-run-deployment.md` remains an alternative deployment guide. Cloud Run and Render are separate hosting choices; the current production API is Render.

## 6. Local setup and reproduction

### Run the current application locally with Docker

From the repository root, with Docker Desktop running:

```bash
docker compose up --build
```

The API is available at `http://localhost:8000`, including its Swagger UI at `http://localhost:8000/docs`. In a second terminal, run the frontend:

```bash
cd app/frontend
npm ci
npm run dev
```

Open `http://localhost:5173`. In Vite development mode, `/api` is proxied to the local API. Stop Docker Compose with `Ctrl+C` or `docker compose down`.

### Run the API directly with Python

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r app/backend/requirements.txt
uvicorn app.backend.main:app --reload --host 127.0.0.1 --port 8000
```

The container uses Python 3.12 and CPU-only PyTorch. Notebook/research environment requirements are separate from backend requirements; `requirements.txt` at repository root contains the broader research environment.

### Re-run the research workflow

1. Obtain the ICBHI 2017 recordings and annotation files from the official dataset source; raw recordings are not stored in this Git repository.
2. Place the data in the paths expected by Notebook 01 and the source/helper code.
3. Install the root research dependencies in a virtual environment and use a Jupyter environment.
4. Run Notebooks 01–09 for the classical-data path, and Notebooks 10–13 for deep-learning preparation/experiments. Notebook 14 aggregates patient predictions; Notebook 15 creates saved XAI outputs.
5. Run Notebook 16 to consolidate results, Notebook 17 to inventory/generate final artifacts, and Notebook 18 for model/artifact validation.
6. Review split definitions, class support, and validation caveats before comparing or reporting results.

Re-running preprocessing and neural training requires local processed cycle audio, which is intentionally excluded from Git and the production image. The dashboard instead uses the packaged checkpoints and saved tables.

## 7. Limitations and interpretation

- The cohort is small at the patient level and strongly class-imbalanced.
- Cycle rows from one patient are related; evaluation must preserve patient grouping when the claim is patient generalization.
- Historical classical notebook results use the official recording split and include two patients with recordings on both sides. The active classical package was retrained for the shared patient-disjoint audit; neural test predictions use saved outputs from that same patient split. The common-split comparison is more consistent, but inherited hyperparameter selection and other benchmark limitations remain.
- Patient aggregation is evaluated on only 25 patients; identical predictions across two aggregation methods on this set do not prove the methods are generally interchangeable.
- High raw accuracy on an imbalanced dataset can coexist with weak minority-class performance. Review balanced accuracy, macro metrics, per-class support and confusion matrices.
- Softmax and estimator probability outputs are not calibrated confidence. Calibration has not been established.
- Saved Grad-CAM examples are historical research artifacts. Live uploaded-audio explanations use input-gradient saliency for neural models and feature baseline ablation for traditional models, and explain only the first segment; they are model-behavior diagnostics, not clinical evidence.
- A healthy dataset label or predicted label is not a medical assessment. This app must not be used clinically.
- The public API has no user authentication or request-level account system. Public hosting and plan limits apply; add access control/rate limiting before exposing it to sensitive or high-volume use.

## 8. Project map and further reading

- `notebooks/` — numbered research workflow, 01 through 18.
- `data/interim/` — parsed source metadata and intermediate validation tables.
- `data/processed/` — cleaned tables, feature matrices, metrics, predictions, manifests, and XAI examples.
- `models/traditional_ml_patient_disjoint/` — active classical inference estimators and their exact feature/scaler metadata for the common patient-held-out model package.
- `models/traditional_ml/` — preserved legacy official-split estimators and metadata.
- `models/deep_learning/` — neural checkpoints, training configs, and normalization assets.
- `reports/final_artifacts/` — consolidated comparisons, validation logs, manifests, and final report.
- `app/backend/` — FastAPI routes, audio processing, classical prediction, and neural model adapters.
- `app/frontend/` — React page components, API client, responsive styles, Vite config, and Vercel config.
- `app/docs/api.md` — endpoint reference.
- `app/docs/architecture.md` — application architecture and inference decisions.
- `app/docs/artifact_mapping.md` — source-artifact provenance for dashboard pages.
- `app/docs/vercel-deployment.md` — current Vercel frontend settings and split deployment details.
- `app/docs/cloud-run-deployment.md` — alternative container deployment instructions.
