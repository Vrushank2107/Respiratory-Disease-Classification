from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score


def aggregate_patient_predictions(frame: pd.DataFrame, probability_columns: list[str], method: str = "mean_probability") -> pd.DataFrame:
    required = {"patient_id", "diagnosis", *probability_columns}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Prediction table is missing columns: {sorted(missing)}")
    classes = [column.removeprefix("prob_") for column in probability_columns]
    patients = []
    for patient_id, group in frame.groupby("patient_id", sort=True):
        if group["diagnosis"].nunique() != 1:
            raise ValueError(f"Patient {patient_id} has inconsistent diagnosis labels")
        probs = group[probability_columns].to_numpy(dtype=float)
        if method == "mean_probability":
            aggregate = probs.mean(axis=0)
        elif method == "majority_vote":
            votes = np.argmax(probs, axis=1)
            aggregate = np.bincount(votes, minlength=len(classes)).astype(float)
            aggregate /= max(aggregate.sum(), 1)
        elif method == "weighted_probability":
            weights = group.get("cycle_weight", pd.Series(1.0, index=group.index)).to_numpy(dtype=float)
            aggregate = np.average(probs, axis=0, weights=weights)
        else:
            raise ValueError("method must be mean_probability, majority_vote, or weighted_probability")
        patients.append({"patient_id": patient_id, "diagnosis": group["diagnosis"].iloc[0], **{f"prob_{name}": float(value) for name, value in zip(classes, aggregate)}, "prediction": classes[int(np.argmax(aggregate))], "n_cycles": len(group)})
    return pd.DataFrame(patients)


def classification_metrics(y_true, y_pred, labels: list[str]) -> dict:
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    per_class = {}
    for index, label in enumerate(labels):
        tp = matrix[index, index]
        fn = matrix[index, :].sum() - tp
        fp = matrix[:, index].sum() - tp
        tn = matrix.sum() - tp - fn - fp
        per_class[label] = {
            "precision": float(tp / (tp + fp)) if tp + fp else 0.0,
            "recall": float(tp / (tp + fn)) if tp + fn else 0.0,
            "f1": float(2 * tp / (2 * tp + fp + fn)) if 2 * tp + fp + fn else 0.0,
            "specificity": float(tn / (tn + fp)) if tn + fp else 0.0,
        }
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
        "report": classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0),
        "per_class_specificity": per_class,
        "confusion_matrix": matrix.tolist(),
        "labels": labels,
    }
