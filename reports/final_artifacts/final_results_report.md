# Respiratory Disease Classification — Final Results

Generated: 2026-10-10 00:22:50

This report consolidates existing metric and comparison CSV files. It does not retrain models or independently recalculate all metrics.

---

## classical_ml_model_comparison.csv

Source: `data/processed/classical_ml/classical_ml_model_comparison.csv`

| model | accuracy | balanced_accuracy | macro_precision | macro_recall | macro_f1 | weighted_f1 | roc_auc_ovr_macro |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Random Forest | 0.874455732946299 | 0.4264946695370397 | 0.3706142888913095 | 0.2665591684606498 | 0.2796043237510104 | 0.8775357362174079 |  |
| SVM | 0.8066037735849056 | 0.4582559389921606 | 0.2605028028520621 | 0.2864099618701004 | 0.2655802677862422 | 0.8470781234574646 |  |
| Logistic Regression | 0.6415094339622641 | 0.4119199839840171 | 0.2338070089691278 | 0.2574499899900107 | 0.2182179173467437 | 0.7457950658138452 |  |

---

## final_classical_ml_comparison.csv

Source: `data/processed/classical_ml/final_classical_ml_comparison.csv`

| Model | Accuracy | Balanced Accuracy | Macro Precision | Macro Recall | Macro F1 | Weighted F1 | ROC AUC (Macro) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Random Forest | 0.874455732946299 | 0.4264946695370397 | 0.3706142888913095 | 0.2665591684606498 | 0.2796043237510104 | 0.8775357362174079 |  |
| SVM | 0.8066037735849056 | 0.4582559389921606 | 0.2605028028520621 | 0.2864099618701004 | 0.2655802677862422 | 0.8470781234574646 |  |
| Logistic Regression | 0.6415094339622641 | 0.4119199839840171 | 0.2338070089691278 | 0.2574499899900107 | 0.2182179173467437 | 0.7457950658138452 |  |

---

## cnn_baseline_metrics.csv

Source: `data/processed/deep_learning/cnn_baseline/cnn_baseline_metrics.csv`

| model | accuracy | balanced_accuracy | macro_precision | macro_recall | macro_f1 | weighted_f1 | best_epoch | best_validation_loss | test_samples | test_patients |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Lightweight Mel CNN | 0.7168481742472774 | 0.3101678079997569 | 0.2130225938117892 | 0.2658581211426488 | 0.2128802054337031 | 0.7687212728918015 | 32 | 1.2070167528983018 | 1561 | 25 |

---

## cnn_lstm_metrics.csv

Source: `data/processed/deep_learning/cnn_lstm/cnn_lstm_metrics.csv`

| model | accuracy | balanced_accuracy | macro_precision | macro_recall | macro_f1 | weighted_f1 | test_loss | best_epoch | best_validation_loss | test_samples | test_patients |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CNN-LSTM | 0.8135810377962844 | 0.2260849970274054 | 0.1344729300835053 | 0.169563747770554 | 0.146471249640457 | 0.7867351773881051 | 0.8616819239670768 | 8 | 1.271948614057879 | 1561 | 25 |

---

## cnn_model_comparison.csv

Source: `data/processed/deep_learning/cnn_lstm/cnn_model_comparison.csv`

| rank | model | accuracy | balanced_accuracy | macro_precision | macro_recall | macro_f1 | weighted_f1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | Lightweight Mel CNN | 0.7168481742472774 | 0.3101678079997569 | 0.2130225938117892 | 0.2658581211426488 | 0.2128802054337031 | 0.7687212728918015 |
| 2 | Multi-feature CNN (12B) | 0.8565022421524664 | 0.1666666666666666 | 0.1427503736920777 | 0.1666666666666666 | 0.1537842190016103 | 0.790299170295271 |
| 3 | Multi-feature CNN (12A) | 0.8552210121716848 | 0.1664173522812266 | 0.1427196921103271 | 0.1664173522812266 | 0.1536602209944751 | 0.7896619428684687 |
| 4 | CNN-LSTM | 0.8135810377962844 | 0.2260849970274054 | 0.1344729300835053 | 0.169563747770554 | 0.146471249640457 | 0.7867351773881051 |

---

## final_deep_learning_comparison.csv

Source: `data/processed/deep_learning/final_deep_learning_comparison.csv`

| Model | Accuracy | Balanced Accuracy | Macro Precision | Macro Recall | Macro F1 | Weighted F1 |
| --- | --- | --- | --- | --- | --- | --- |
| Lightweight Mel CNN | 0.716848 | 0.310168 | 0.213023 | 0.265858 | 0.21288 | 0.768721 |
| Multi-feature CNN (12A) | 0.855221 | 0.166417 | 0.14272 | 0.166417 | 0.15366 | 0.789662 |
| Multi-feature CNN (12B) | 0.856502 | 0.166667 | 0.14275 | 0.166667 | 0.153784 | 0.790299 |
| CNN-LSTM | 0.813581 | 0.226085 | 0.134473 | 0.169564 | 0.146471 | 0.786735 |

---

## cnn_baseline_vs_multifeature_comparison.csv

Source: `data/processed/deep_learning/multifeature_cnn/cnn_baseline_vs_multifeature_comparison.csv`

| Metric | Lightweight Mel CNN | Multi-Feature CNN | Difference |
| --- | --- | --- | --- |
| Accuracy | 0.7168481742472774 | 0.8552210121716848 | 0.1383728379244073 |
| Balanced Accuracy | 0.3101678079997569 | 0.1664173522812266 | -0.1437504557185303 |
| Macro Precision | 0.2130225938117892 | 0.1427196921103271 | -0.0703029017014621 |
| Macro Recall | 0.2658581211426488 | 0.1664173522812266 | -0.0994407688614222 |
| Macro F1 | 0.2128802054337031 | 0.1536602209944751 | -0.0592199844392279 |
| Weighted F1 | 0.7687212728918015 | 0.7896619428684687 | 0.0209406699766672 |

---

## disease_level_cnn_comparison.csv

Source: `data/processed/deep_learning/multifeature_cnn/disease_level_cnn_comparison.csv`

| Disease | Baseline_F1 | MultiFeature_F1 | F1_Difference | Baseline_Support | MultiFeature_Support |
| --- | --- | --- | --- | --- | --- |
| Asthma | 0.0 | 0.0 | 0.0 | 0 | 0 |
| Bronchiectasis | 0.0857142857142857 | 0.0 | -0.0857142857142857 | 20 | 20 |
| Bronchiolitis | 0.1267605633802817 | 0.0 | -0.1267605633802817 | 19 | 19 |
| COPD | 0.8768233387358185 | 0.9219613259668508 | 0.0451379872310323 | 1337 | 1337 |
| Healthy | 0.0898876404494382 | 0.0 | -0.0898876404494382 | 84 | 84 |
| LRTI | 0.0 | 0.0 | 0.0 | 0 | 0 |
| Pneumonia | 0.25 | 0.0 | -0.25 | 52 | 52 |
| URTI | 0.0609756097560975 | 0.0 | -0.0609756097560975 | 49 | 49 |

---

## final_cnn_experiment_comparison.csv

Source: `data/processed/deep_learning/multifeature_cnn/final_cnn_experiment_comparison.csv`

| rank | model | features | accuracy | balanced_accuracy | macro_precision | macro_recall | macro_f1 | weighted_f1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | Lightweight Mel CNN | Log-Mel | 0.7168481742472774 | 0.3101678079997569 | 0.2130225938117892 | 0.2658581211426488 | 0.2128802054337031 | 0.7687212728918015 |
| 2 | Regularized Multi-Feature CNN 12B | Log-Mel + MFCC + Chroma | 0.8565022421524664 | 0.1666666666666666 | 0.1427503736920777 | 0.1666666666666666 | 0.1537842190016103 | 0.790299170295271 |
| 3 | Multi-Feature CNN 12A | Log-Mel + MFCC + Chroma | 0.8552210121716848 | 0.1664173522812266 | 0.1427196921103271 | 0.1664173522812266 | 0.1536602209944751 | 0.7896619428684687 |

---

## multifeature_cnn_metrics.csv

Source: `data/processed/deep_learning/multifeature_cnn/multifeature_cnn_metrics.csv`

| model | best_epoch | best_validation_loss | test_samples | accuracy | balanced_accuracy | macro_precision | macro_recall | macro_f1 | weighted_f1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Multi-Feature CNN | 1 | 2.45279103893322 | 1561 | 0.8552210121716848 | 0.1664173522812266 | 0.1427196921103271 | 0.1664173522812266 | 0.1536602209944751 | 0.7896619428684687 |

---

## metrics.csv

Source: `data/processed/deep_learning/multifeature_cnn_12b/metrics.csv`

| model | best_epoch | best_validation_loss | test_samples | accuracy | balanced_accuracy | macro_precision | macro_recall | macro_f1 | weighted_f1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Regularized Multi-Feature CNN (12B) | 2 | 1.33220093725025 | 1561 | 0.8565022421524664 | 0.1666666666666666 | 0.1427503736920777 | 0.1666666666666666 | 0.1537842190016103 | 0.790299170295271 |

---

## final_patient_level_comparison.csv

Source: `data/processed/deep_learning/patient_level_prediction/final_patient_level_comparison.csv`

| rank | aggregation_method | accuracy | balanced_accuracy | macro_precision | macro_recall | macro_f1 | weighted_f1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | Majority Vote | 0.56 | 0.3333333333333333 | 0.1328125 | 0.25 | 0.1620689655172413 | 0.4822068965517241 |
| 2 | Mean Probability | 0.56 | 0.3333333333333333 | 0.1328125 | 0.25 | 0.1620689655172413 | 0.4822068965517241 |

---

## patient_aggregation_comparison.csv

Source: `data/processed/deep_learning/patient_level_prediction/patient_aggregation_comparison.csv`

| rank | aggregation_method | accuracy | balanced_accuracy | macro_precision | macro_recall | macro_f1 | weighted_f1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | Majority Vote | 0.56 | 0.3333333333333333 | 0.1328125 | 0.25 | 0.1620689655172413 | 0.4822068965517241 |
| 2 | Mean Probability | 0.56 | 0.3333333333333333 | 0.1328125 | 0.25 | 0.1620689655172413 | 0.4822068965517241 |

---

## cnn_lstm_original_metric_recheck.csv

Source: `reports/final_artifacts/cnn_lstm_original_metric_recheck.csv`

| metric | recomputed | saved | difference | matches |
| --- | --- | --- | --- | --- |
| accuracy | 0.8135810377962844 | 0.8135810377962844 | 0.0 | True |
| balanced_accuracy | 0.2260849970274054 | 0.2260849970274054 | 0.0 | True |
| macro_precision | 0.1344729300835053 | 0.1344729300835053 | 5.551115123125783e-17 | True |
| macro_recall | 0.169563747770554 | 0.169563747770554 | 5.551115123125783e-17 | True |
| macro_f1 | 0.146471249640457 | 0.146471249640457 | 8.326672684688674e-17 | True |
| weighted_f1 | 0.7867351773881051 | 0.7867351773881051 | 0.0 | True |

---

## cycle_level_model_comparison.csv

Source: `reports/final_artifacts/cycle_level_model_comparison.csv`

| model_family | model | evaluation_unit | accuracy | balanced_accuracy | macro_precision | macro_recall | macro_f1 | weighted_f1 | roc_auc_(macro) | test_samples | test_patients | best_epoch | best_validation_loss | test_loss | split_comparability | source_file |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Classical ML | Random Forest | cycle | 0.874455732946299 | 0.4264946695370397 | 0.3706142888913095 | 0.2665591684606498 | 0.2796043237510104 | 0.8775357362174079 |  |  |  |  |  |  | UNVERIFIED — confirm identical test patients/cycles | classical_ml/final_classical_ml_comparison.csv |
| Classical ML | SVM | cycle | 0.8066037735849056 | 0.4582559389921606 | 0.2605028028520621 | 0.2864099618701004 | 0.2655802677862422 | 0.8470781234574646 |  |  |  |  |  |  | UNVERIFIED — confirm identical test patients/cycles | classical_ml/final_classical_ml_comparison.csv |
| Classical ML | Logistic Regression | cycle | 0.6415094339622641 | 0.4119199839840171 | 0.2338070089691278 | 0.2574499899900107 | 0.2182179173467437 | 0.7457950658138452 |  |  |  |  |  |  | UNVERIFIED — confirm identical test patients/cycles | classical_ml/final_classical_ml_comparison.csv |
| Deep Learning | Lightweight Mel CNN | cycle | 0.7168481742472774 | 0.3101678079997569 | 0.2130225938117892 | 0.2658581211426488 | 0.2128802054337031 | 0.7687212728918015 |  | 1561.0 | 25.0 | 32.0 | 1.2070167528983018 |  | UNVERIFIED — confirm identical test patients/cycles | deep_learning/cnn_baseline/cnn_baseline_metrics.csv |
| Deep Learning | CNN-LSTM | cycle | 0.8135810377962844 | 0.2260849970274054 | 0.1344729300835053 | 0.169563747770554 | 0.146471249640457 | 0.7867351773881051 |  | 1561.0 | 25.0 | 8.0 | 1.271948614057879 | 0.8616819239670768 | UNVERIFIED — confirm identical test patients/cycles | deep_learning/cnn_lstm/cnn_lstm_metrics.csv |
| Deep Learning | Multi-Feature CNN | cycle | 0.8552210121716848 | 0.1664173522812266 | 0.1427196921103271 | 0.1664173522812266 | 0.1536602209944751 | 0.7896619428684687 |  | 1561.0 |  | 1.0 | 2.45279103893322 |  | UNVERIFIED — confirm identical test patients/cycles | deep_learning/multifeature_cnn/multifeature_cnn_metrics.csv |
| Deep Learning | Regularized Multi-Feature CNN (12B) | cycle | 0.8565022421524664 | 0.1666666666666666 | 0.1427503736920777 | 0.1666666666666666 | 0.1537842190016103 | 0.790299170295271 |  | 1561.0 |  | 2.0 | 1.33220093725025 |  | UNVERIFIED — confirm identical test patients/cycles | deep_learning/multifeature_cnn_12b/metrics.csv |

---

## patient_level_model_comparison.csv

Source: `reports/final_artifacts/patient_level_model_comparison.csv`

| rank | aggregation_method | accuracy | balanced_accuracy | macro_precision | macro_recall | macro_f1 | weighted_f1 | evaluation_unit | source_file |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | Majority Vote | 0.56 | 0.3333333333333333 | 0.1328125 | 0.25 | 0.1620689655172413 | 0.4822068965517241 | patient | data/processed/deep_learning/patient_level_prediction/patient_aggregation_comparison.csv |
| 2 | Mean Probability | 0.56 | 0.3333333333333333 | 0.1328125 | 0.25 | 0.1620689655172413 | 0.4822068965517241 | patient | data/processed/deep_learning/patient_level_prediction/patient_aggregation_comparison.csv |

---

## prediction_metric_count_checks.csv

Source: `reports/final_artifacts/prediction_metric_count_checks.csv`

| model | metric_file_exists | prediction_file_exists | reported_test_samples | prediction_rows | counts_match | status |
| --- | --- | --- | --- | --- | --- | --- |
| Lightweight Mel CNN | True | True | 1561 | 1561 | True | PASS |
| CNN-LSTM | True | True | 1561 | 1561 | True | PASS |
| Multi-Feature CNN | True | True | 1561 | 1561 | True | PASS |
| Regularized Multi-Feature CNN (12B) | True | True | 1561 | 1561 | True | PASS |

---

## prediction_metric_reproduction_check.csv

Source: `reports/final_artifacts/prediction_metric_reproduction_check.csv`

| model | prediction_rows | recomputed_accuracy | saved_accuracy | recomputed_balanced_accuracy | saved_balanced_accuracy | recomputed_macro_f1 | saved_macro_f1 | status | details |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Lightweight Mel CNN | 1561 | 0.7168481742472774 | 0.7168481742472774 | 0.3101678079997569 | 0.3101678079997569 | 0.2128802054337031 | 0.2128802054337031 | PASS: metrics reproduced |  |
| CNN-LSTM | 1561 | 0.8135810377962844 | 0.8135810377962844 | 0.2260849970274054 | 0.2260849970274054 | 0.1673957138748081 | 0.146471249640457 | REVIEW: discrepancy | macro_f1: recomputed=0.167396, saved=0.146471 |
| Multi-Feature CNN | 1561 | 0.8552210121716848 | 0.8552210121716848 | 0.1664173522812266 | 0.1664173522812266 | 0.1536602209944751 | 0.1536602209944751 | PASS: metrics reproduced |  |
| Regularized Multi-Feature CNN (12B) | 1561 | 0.8565022421524664 | 0.8565022421524664 | 0.1666666666666666 | 0.1666666666666666 | 0.1537842190016103 | 0.1537842190016103 | PASS: metrics reproduced |  |

---

## Evaluation notes

- The general deep-learning test manifest contains 2,756 rows, while each of the four saved deep-learning prediction files contains 1,561 rows. Verify evaluation-split provenance before comparing models.
- Cycle-level and patient-level results use different evaluation units and must be reported separately.
- Accuracy should be interpreted alongside balanced accuracy and macro F1 because class imbalance may affect the results.
- This report preserves existing saved results; it does not prove that all models used identical test samples.
