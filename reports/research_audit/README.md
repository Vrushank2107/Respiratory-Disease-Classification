# Research audit updates

## Common patient-held-out evaluation

The reproducible evaluation in [`common_patient_split`](common_patient_split/) compares all seven model variants using the existing neural patient-disjoint split. It has 82 training patients, 19 validation patients (not used to fit the classical estimators), and 25 held-out test patients. The test cohort contains 53 recordings and 1,561 respiratory cycles.

The three classical estimators were retrained for this comparison. Notebook 07's ANOVA, mutual-information, and random-forest consensus ranking and correlation pruning were recomputed using training patients only (33 selected features). Scaling was fit on the training cycles only. The resulting estimators, scaler, label encoder, ordered feature list, and package metadata are saved under `models/traditional_ml_patient_disjoint/`; the app loads this package when present. The prior official-split package remains under `models/traditional_ml/` as the legacy model set.

The four neural model predictions come from their saved test-prediction exports. Each export was checked against the canonical test labels before comparison. The common cohort results are also available from `GET /api/evaluation/models` under `common_patient_split` and are shown separately from the legacy saved evaluations in the Model Comparison page.

### Headline results

Cycle-level macro averages include only classes represented in the test set. Patient-level results use majority vote over each patient's cycle predictions.

| Model | Cycle accuracy | Cycle balanced accuracy | Cycle macro F1 | Patient accuracy | Patient macro F1 |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.782 | 0.403 | 0.389 | 0.640 | 0.364 |
| SVM | 0.847 | 0.373 | 0.373 | 0.680 | 0.297 |
| Random Forest | 0.855 | 0.285 | 0.283 | 0.680 | 0.313 |
| Lightweight Mel CNN | 0.717 | 0.310 | 0.248 | 0.560 | 0.216 |
| Multi-Feature CNN | 0.855 | 0.166 | 0.154 | 0.520 | 0.114 |
| Regularized Multi-Feature CNN (12B) | 0.857 | 0.167 | 0.154 | 0.520 | 0.114 |
| CNN-LSTM | 0.814 | 0.226 | 0.195 | 0.480 | 0.111 |

| Test disease | Patients | Recordings | Cycles |
|---|---:|---:|---:|
| Asthma | 0 | 0 | 0 |
| Bronchiectasis | 2 | 3 | 20 |
| Bronchiolitis | 1 | 1 | 19 |
| COPD | 13 | 37 | 1,337 |
| Healthy | 5 | 6 | 84 |
| LRTI | 0 | 0 | 0 |
| Pneumonia | 1 | 3 | 52 |
| URTI | 3 | 3 | 49 |

### Interpretation limits

- Cycle-level scores count each cycle as one observation. Cycles from the same patient are correlated, so use the separate patient-level majority-vote results when the unit of interest is a patient.
- The held-out set has no Asthma or LRTI patients. Their metrics are marked unavailable; the current data cannot validate those diagnoses on unseen patients.
- Macro precision, recall, and F1 average over classes that have test support. Unsupported disease classes are excluded from the macro averages and marked unavailable in the per-class tables.
- Bronchiectasis has 2 held-out patients, Bronchiolitis 1, Pneumonia 1, URTI 3, Healthy 5, and COPD 13. Estimates for rare classes are highly uncertain.
- Patient-level aggregation uses one majority vote across a patient's cycle predictions; tied votes resolve to the lowest class index. It is a transparent comparison rule, not a validated clinical decision rule.
- Classical model hyperparameters were held at the saved experiment values rather than tuned on the shared split. Deep-learning checkpoints were not retrained in this audit.
- Earlier hyperparameter-selection history still needs review before calling this a fully untouched final benchmark.
- Classical latency and serialized fitted-estimator bytes refer to the newly fit audit estimators. Neural inference timing varies by device and is measured dynamically by the app; the result table does not invent a cross-device benchmark.
- These results are research findings, not evidence for clinical use.

## Full-recording upload flow

Notebook 04 used known cycle start/end timestamps from the ICBHI annotation files. Predict audio accepts a WAV with an optional paired `.txt` annotation file; when supplied, validated timestamps select the cycles before inference. Without annotations, the app covers the recording with consecutive five-second windows. These are fixed-duration windows, not detected respiratory cycles, and their predictions are exploratory because the trained models use cycle examples. There is no SHA lookup or cycle detector.

## Reproduction

From the repository root, run:

```bash
./.venv/bin/python scripts/evaluate_common_patient_split.py
```

The script writes the overall and per-class cycle metrics, patient-level majority-vote metrics, predictions, confusion matrices, cohort protocol, deployed model sizes, and selected feature list under this directory. It also regenerates the active patient-disjoint classical model package without overwriting the legacy artifacts.
