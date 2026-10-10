"""Retrain classical baselines and compare all models on the CNN patient holdout.

The script does not overwrite app artifacts or legacy experiment reports. It uses
the CNN patient-disjoint manifests, fits feature selection/scaling/models on the
training patients only, and checks row alignment for saved neural predictions.
"""

from __future__ import annotations

import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import f_classif, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "reports" / "research_audit" / "common_patient_split"
MODEL_PACKAGE = ROOT / "models" / "traditional_ml_patient_disjoint"
FEATURES_PATH = ROOT / "data/processed/features/respiratory_cycle_features.csv"
TRAIN_PATH = ROOT / "data/processed/deep_learning/cnn_baseline/train_cnn_manifest.csv"
VALIDATION_PATH = (
    ROOT / "data/processed/deep_learning/cnn_baseline/validation_cnn_manifest.csv"
)
TEST_PATH = ROOT / "data/processed/deep_learning/cnn_baseline/test_cnn_manifest.csv"
CLASS_NAMES = [
    "Asthma",
    "Bronchiectasis",
    "Bronchiolitis",
    "COPD",
    "Healthy",
    "LRTI",
    "Pneumonia",
    "URTI",
]
DEPLOYED_ARTIFACTS = {
    "logistic_regression": ROOT / "models/traditional_ml_patient_disjoint/logistic_regression.joblib",
    "svm": ROOT / "models/traditional_ml_patient_disjoint/svm.joblib",
    "random_forest": ROOT / "models/traditional_ml_patient_disjoint/random_forest.joblib",
    "lightweight_mel_cnn": ROOT / "models/deep_learning/lightweight_mel_cnn/best_lightweight_mel_cnn.pt",
    "multifeature_cnn": ROOT / "models/deep_learning/multifeature_cnn/best_multifeature_cnn.pt",
    "regularized_multifeature_cnn": ROOT / "models/deep_learning/regularized_multifeature_cnn/best_regularized_multifeature_cnn.pt",
    "cnn_lstm": ROOT / "models/deep_learning/cnn_lstm/best_cnn_lstm.pt",
}
METADATA_COLUMNS = {
    "patient_id",
    "recording_id",
    "chest_location",
    "acquisition_mode",
    "equipment",
    "cycle_number",
    "start_time",
    "end_time",
    "sound_label",
    "crackles",
    "wheezes",
    "diagnosis",
    "processed_path",
}


def rank_training_features(x: pd.DataFrame, y: np.ndarray) -> list[str]:
    """Reproduce Notebook 07 consensus ranking and correlation pruning on train only."""
    anova, _ = f_classif(x, y)
    mi = mutual_info_classif(x, y, random_state=42)
    selector = RandomForestClassifier(
        n_estimators=300, random_state=42, n_jobs=-1, class_weight="balanced"
    )
    selector.fit(x, y)
    ranks = pd.DataFrame(
        {
            "feature": x.columns,
            "anova_rank": pd.Series(anova).rank(ascending=False, method="min").values,
            "mi_rank": pd.Series(mi).rank(ascending=False, method="min").values,
            "rf_rank": pd.Series(selector.feature_importances_)
            .rank(ascending=False, method="min")
            .values,
        }
    )
    for rank in ("anova_rank", "mi_rank", "rf_rank"):
        ranks[rank.replace("rank", "score")] = (
            ranks[rank].max() - ranks[rank] + 1
        ) / ranks[rank].max()
    ranks["consensus"] = ranks[["anova_score", "mi_score", "rf_score"]].mean(axis=1)
    ranked = ranks.sort_values("consensus", ascending=False)["feature"].tolist()
    corr = x.corr().abs()
    selected: list[str] = []
    for feature in ranked:
        if not selected or corr.loc[feature, selected].max() < 0.90:
            selected.append(feature)
    return selected


def metric_rows(model_id: str, y_true: np.ndarray, y_pred: np.ndarray) -> tuple[dict, list[dict], pd.DataFrame]:
    supported_labels = np.unique(y_true)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=np.arange(len(CLASS_NAMES)), zero_division=0
    )
    overall = {
        "model_id": model_id,
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_precision": precision_recall_fscore_support(
            y_true, y_pred, labels=supported_labels, average="macro", zero_division=0
        )[0],
        "macro_recall": precision_recall_fscore_support(
            y_true, y_pred, labels=supported_labels, average="macro", zero_division=0
        )[1],
        "macro_f1": f1_score(y_true, y_pred, labels=supported_labels, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, labels=np.arange(len(CLASS_NAMES)), average="weighted", zero_division=0),
        "test_cycles": len(y_true),
    }
    per_class = [
        {
            "model_id": model_id,
            "disease": name,
            "precision": float(precision[i]) if support[i] else np.nan,
            "recall": float(recall[i]) if support[i] else np.nan,
            "f1": float(f1[i]) if support[i] else np.nan,
            "support_cycles": int(support[i]),
            "metric_status": "no held-out support" if support[i] == 0 else "measured",
        }
        for i, name in enumerate(CLASS_NAMES)
    ]
    matrix = pd.DataFrame(
        confusion_matrix(y_true, y_pred, labels=np.arange(len(CLASS_NAMES))),
        index=CLASS_NAMES,
        columns=CLASS_NAMES,
    )
    matrix.index.name = "true_disease"
    return overall, per_class, matrix


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(TRAIN_PATH)
    validation = pd.read_csv(VALIDATION_PATH)
    test = pd.read_csv(TEST_PATH)
    if (
        train.patient_id.isin(test.patient_id).any()
        or validation.patient_id.isin(test.patient_id).any()
        or train.patient_id.isin(validation.patient_id).any()
    ):
        raise RuntimeError("Patient overlap found in canonical training/validation/test manifests")

    frame = pd.read_csv(FEATURES_PATH)
    feature_columns = [name for name in frame.columns if name not in METADATA_COLUMNS]
    if not np.isfinite(frame[feature_columns].to_numpy(dtype=float)).all():
        raise RuntimeError("Feature table has missing or non-finite model inputs")
    train_patients = set(train.patient_id)
    test_patients = set(test.patient_id)
    train_rows = frame[frame.patient_id.isin(train_patients)].copy()
    test_rows = frame[frame.patient_id.isin(test_patients)].copy()
    train_rows = train_rows.merge(train[["patient_id", "disease_label"]].drop_duplicates(), on="patient_id")
    test_rows = test_rows.merge(test[["patient_id", "disease_label"]].drop_duplicates(), on="patient_id")
    if len(train_rows) != len(train) or len(test_rows) != len(test):
        raise RuntimeError("Feature and canonical manifests do not have one-to-one cycle coverage")
    if not np.array_equal(test_rows.disease_label.to_numpy(), test.disease_label.to_numpy()):
        raise RuntimeError("Feature rows do not align with the canonical test manifest order")

    x_train_all = train_rows[feature_columns]
    y_train = train_rows.disease_label.to_numpy(dtype=int)
    x_test_all = test_rows[feature_columns]
    y_test = test_rows.disease_label.to_numpy(dtype=int)
    selected = rank_training_features(x_train_all, y_train)
    x_train, x_test = x_train_all[selected], x_test_all[selected]
    scaler = StandardScaler().fit(x_train)
    transformed_train = scaler.transform(x_train)
    transformed_test = scaler.transform(x_test)

    estimators = {
        "logistic_regression": LogisticRegression(
            C=1.0, class_weight="balanced", max_iter=3000, random_state=42, solver="lbfgs"
        ),
        "svm": SVC(
            C=1.0, class_weight="balanced", probability=True, random_state=42
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300, class_weight="balanced", random_state=42, n_jobs=-1
        ),
    }
    metrics = []
    per_class = []
    predictions = []
    matrices = {}
    for model_id, estimator in estimators.items():
        train_x = x_train if model_id == "random_forest" else transformed_train
        test_x = x_test if model_id == "random_forest" else transformed_test
        estimator.fit(train_x, y_train)
        started = time.perf_counter()
        y_pred = estimator.predict(test_x)
        elapsed_ms = (time.perf_counter() - started) * 1000
        overall, classes, matrix = metric_rows(model_id, y_test, y_pred)
        overall["estimator_predict_ms_per_cycle"] = elapsed_ms / len(y_pred)
        overall["serialized_estimator_bytes"] = len(pickle.dumps(estimator))
        for index, row in enumerate(classes):
            class_rows = test_rows[test_rows.disease_label == index]
            row["support_patients"] = int(class_rows.patient_id.nunique())
            row["support_recordings"] = int(
                class_rows[["patient_id", "recording_id"]].drop_duplicates().shape[0]
            )
        metrics.append(overall)
        per_class.extend(classes)
        matrices[model_id] = matrix
        predictions.append(
            pd.DataFrame(
                {
                    "patient_id": test_rows.patient_id,
                    "recording_id": test_rows.recording_id,
                    "chest_location": test_rows.chest_location,
                    "true_label": y_test,
                    "true_disease": [CLASS_NAMES[i] for i in y_test],
                    "predicted_label": y_pred,
                    "predicted_disease": [CLASS_NAMES[i] for i in y_pred],
                    "model_id": model_id,
                }
            )
        )

    MODEL_PACKAGE.mkdir(parents=True, exist_ok=True)
    for model_id, estimator in estimators.items():
        joblib.dump(estimator, MODEL_PACKAGE / f"{model_id}.joblib", compress=3)
    joblib.dump(scaler, MODEL_PACKAGE / "scaler.joblib", compress=3)
    encoder = LabelEncoder().fit(CLASS_NAMES)
    joblib.dump(encoder, MODEL_PACKAGE / "label_encoder.joblib", compress=3)
    (MODEL_PACKAGE / "selected_features.json").write_text(
        json.dumps(selected, indent=2) + "\n"
    )
    package_metadata = {
        "project": "Respiratory Disease Classification",
        "package_version": "patient-disjoint-v1",
        "training_rows": len(train_rows),
        "test_rows": len(test_rows),
        "training_patients": int(train.patient_id.nunique()),
        "test_patients": int(test.patient_id.nunique()),
        "validation_rows_not_used_for_training": len(validation),
        "feature_count": len(selected),
        "selected_feature_names": selected,
        "class_names": CLASS_NAMES,
        "scaler_used": {
            "logistic_regression": True,
            "svm": True,
            "random_forest": False,
        },
        "training_split": "CNN patient-disjoint split; feature selection and estimators fit on train patients only",
        "source": "scripts/evaluate_common_patient_split.py",
        "models": {
            model_id: {
                "file": f"{model_id}.joblib",
                "class": type(estimator).__name__,
                "parameters": estimator.get_params(),
            }
            for model_id, estimator in estimators.items()
        },
    }
    (MODEL_PACKAGE / "classical_ml_metadata.json").write_text(
        json.dumps(package_metadata, indent=2, default=str) + "\n"
    )

    neural_sources = {
        "lightweight_mel_cnn": (
            ROOT / "data/processed/deep_learning/cnn_baseline/cnn_test_predictions.csv",
            "predicted_label",
            "true_label",
        ),
        "multifeature_cnn": (
            ROOT / "data/processed/deep_learning/multifeature_cnn/multifeature_cnn_test_predictions.csv",
            "predicted_label",
            "true_label",
        ),
        "regularized_multifeature_cnn": (
            ROOT / "data/processed/deep_learning/multifeature_cnn_12b/test_predictions.csv",
            "predicted_label",
            "true_label",
        ),
        "cnn_lstm": (
            ROOT / "data/processed/deep_learning/cnn_lstm/cnn_lstm_test_predictions.csv",
            "predicted_disease_label",
            "disease_label",
        ),
    }
    for model_id, (path, pred_col, true_col) in neural_sources.items():
        saved = pd.read_csv(path)
        if len(saved) != len(test) or not np.array_equal(saved[true_col].to_numpy(), y_test):
            raise RuntimeError(f"Saved {model_id} predictions do not align to the canonical test manifest")
        y_pred = saved[pred_col].to_numpy(dtype=int)
        overall, classes, matrix = metric_rows(model_id, y_test, y_pred)
        for index, row in enumerate(classes):
            class_rows = test[test.disease_label == index]
            row["support_patients"] = int(class_rows.patient_id.nunique())
            row["support_recordings"] = int(
                class_rows[["patient_id", "recording_id"]].drop_duplicates().shape[0]
            )
        metrics.append(overall)
        per_class.extend(classes)
        matrices[model_id] = matrix
        predictions.append(
            pd.DataFrame(
                {
                    "patient_id": test.patient_id,
                    "recording_id": test.recording_id,
                    "chest_location": test.chest_location,
                    "true_label": y_test,
                    "true_disease": [CLASS_NAMES[i] for i in y_test],
                    "predicted_label": y_pred,
                    "predicted_disease": [CLASS_NAMES[i] for i in y_pred],
                    "model_id": model_id,
                }
            )
        )

    all_predictions = pd.concat(predictions, ignore_index=True)
    patient_metrics = []
    patient_classes = []
    patient_predictions = []
    patient_matrices = {}
    for model_id, group in all_predictions.groupby("model_id", sort=False):
        rows = []
        for patient_id, patient_rows in group.groupby("patient_id", sort=False):
            true_labels = patient_rows.true_label.unique()
            if len(true_labels) != 1:
                raise RuntimeError(f"Patient {patient_id} has conflicting ground-truth diagnoses")
            votes = np.bincount(patient_rows.predicted_label, minlength=len(CLASS_NAMES))
            rows.append(
                {
                    "patient_id": patient_id,
                    "true_label": int(true_labels[0]),
                    "predicted_label": int(votes.argmax()),
                    "true_disease": CLASS_NAMES[int(true_labels[0])],
                    "predicted_disease": CLASS_NAMES[int(votes.argmax())],
                    "model_id": model_id,
                    "recordings": patient_rows.recording_id.nunique(),
                    "cycles": len(patient_rows),
                }
            )
        patient_frame = pd.DataFrame(rows)
        y_patient_true = patient_frame.true_label.to_numpy(dtype=int)
        y_patient_pred = patient_frame.predicted_label.to_numpy(dtype=int)
        overall, classes, matrix = metric_rows(model_id, y_patient_true, y_patient_pred)
        overall["test_patients"] = len(patient_frame)
        overall.pop("test_cycles")
        for index, row in enumerate(classes):
            row["support_patients"] = int((y_patient_true == index).sum())
            row["support_recordings"] = int(
                patient_frame.loc[patient_frame.true_label == index, "recordings"].sum()
            )
            row.pop("support_cycles")
        patient_metrics.append(overall)
        patient_classes.extend(classes)
        patient_predictions.extend(rows)
        patient_matrices[model_id] = matrix

    result = pd.DataFrame(metrics)
    result["deployed_artifact_bytes"] = result.model_id.map(
        {model_id: path.stat().st_size for model_id, path in DEPLOYED_ARTIFACTS.items()}
    )
    result.insert(1, "training_patients", train.patient_id.nunique())
    result.insert(2, "validation_patients_not_used", validation.patient_id.nunique())
    result.insert(3, "test_patients", test.patient_id.nunique())
    result.insert(4, "test_recordings", test[["patient_id", "recording_id"]].drop_duplicates().shape[0])
    result.insert(5, "test_cycle_support", len(test))
    result.to_csv(OUTPUT / "overall_metrics.csv", index=False)
    pd.DataFrame(per_class).to_csv(OUTPUT / "per_class_metrics.csv", index=False)
    all_predictions.to_csv(OUTPUT / "test_predictions.csv", index=False)
    for model_id, matrix in matrices.items():
        matrix.to_csv(OUTPUT / f"confusion_matrix_{model_id}.csv")
    pd.DataFrame(patient_metrics).to_csv(OUTPUT / "patient_level_overall_metrics.csv", index=False)
    pd.DataFrame(patient_classes).to_csv(OUTPUT / "patient_level_per_class_metrics.csv", index=False)
    pd.DataFrame(patient_predictions).to_csv(OUTPUT / "patient_level_predictions.csv", index=False)
    for model_id, matrix in patient_matrices.items():
        matrix.to_csv(OUTPUT / f"patient_level_confusion_matrix_{model_id}.csv")
    (OUTPUT / "classical_selected_features.json").write_text(
        json.dumps(selected, indent=2) + "\n"
    )
    protocol = {
        "protocol": "cycle-level evaluation on the existing CNN patient-disjoint holdout",
        "train_patients": sorted(map(int, train.patient_id.unique())),
        "validation_patients_not_used_for_classical_fit": sorted(map(int, validation.patient_id.unique())),
        "test_patients": sorted(map(int, test.patient_id.unique())),
        "train_cycles": len(train),
        "validation_cycles": len(validation),
        "test_cycles": len(test),
        "test_recordings": int(test[["patient_id", "recording_id"]].drop_duplicates().shape[0]),
        "test_class_support": {
            name: int((y_test == index).sum()) for index, name in enumerate(CLASS_NAMES)
        },
        "feature_selection": "Notebook 07 consensus ranking and 0.90 correlation pruning fitted on patient training data only",
        "feature_count": len(selected),
        "neural_predictions": "Loaded from saved test predictions after true-label row-alignment checks. Notebook 12 uses the canonical patient-disjoint manifest with an unshuffled test loader; notebooks 11 and 13 retain row-level manifest identifiers.",
        "macro_average": "Macro precision/recall/F1 are averaged over classes with held-out support; unsupported classes remain explicitly unavailable in per-class tables.",
        "limitations": [
            "Cycle rows from a held-out patient are correlated; metrics are cycle-level, not independent patient-level estimates.",
            "The patient test cohort has no Asthma or LRTI support; their per-class metrics are reported as unavailable.",
            "Classical estimators were retrained with the saved experiment hyperparameters, not re-tuned on this split.",
            "A complete audit of whether historical hyperparameter decisions were influenced by earlier test results remains necessary before claiming a final untouched benchmark.",
            "Saved neural checkpoints are the existing patient-split models; they were not retrained in this script.",
            "Inference timing and serialized size are reported only for the newly fitted classical estimators.",
        ],
    }
    (OUTPUT / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    pd.DataFrame(
        [
            {"model_id": model_id, "artifact_path": path.relative_to(ROOT).as_posix(), "artifact_bytes": path.stat().st_size}
            for model_id, path in DEPLOYED_ARTIFACTS.items()
        ]
    ).to_csv(OUTPUT / "model_artifact_sizes.csv", index=False)
    print(result.to_string(index=False))
    print(f"Wrote common-split audit artifacts to {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
