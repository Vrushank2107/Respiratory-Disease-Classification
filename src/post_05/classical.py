from __future__ import annotations

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .patient_level import aggregate_patient_predictions, classification_metrics


def train_classical_models(train: pd.DataFrame, test: pd.DataFrame, features: list[str]) -> tuple[dict, dict, dict]:
    x_train = train[features].replace([float("inf"), -float("inf")], pd.NA).fillna(train[features].median()).fillna(0)
    x_test = test[features].replace([float("inf"), -float("inf")], pd.NA).fillna(train[features].median()).fillna(0)
    y_train, y_test = train["diagnosis"].astype(str), test["diagnosis"].astype(str)
    models = {
        "logistic_regression": make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=2000, random_state=42)),
        "svm": make_pipeline(StandardScaler(), SVC(class_weight="balanced", probability=True, random_state=42)),
        "random_forest": RandomForestClassifier(n_estimators=500, class_weight="balanced_subsample", random_state=42, n_jobs=-1),
    }
    metrics, predictions, fitted = {}, {}, {}
    all_classes = sorted(set(y_train) | set(y_test))
    for name, model in models.items():
        model.fit(x_train, y_train)
        predicted = model.predict(x_test)
        probs = model.predict_proba(x_test)
        model_classes = list(model.classes_ if hasattr(model, "classes_") else model[-1].classes_)
        prediction_frame = test[["patient_id", "recording_id", "cycle_number", "diagnosis", "split"]].copy()
        prediction_frame["prediction"] = predicted
        for class_name in all_classes:
            prediction_frame[f"prob_{class_name}"] = probs[:, model_classes.index(class_name)] if class_name in model_classes else 0.0
        probability_columns = [f"prob_{class_name}" for class_name in all_classes]
        metrics[name] = {
            "cycle": classification_metrics(y_test, predicted, all_classes),
            "patient": classification_metrics(
                (patient := aggregate_patient_predictions(prediction_frame, probability_columns))["diagnosis"],
                patient["prediction"], all_classes,
            ),
        }
        predictions[name] = prediction_frame
        fitted[name] = model
    return metrics, predictions, fitted
