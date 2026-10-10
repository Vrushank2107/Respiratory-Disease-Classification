"""Saved scikit-learn classifiers and Notebook 05 feature extraction."""

import json
from functools import lru_cache
from pathlib import Path
import time
import joblib
import librosa
import numpy as np
import pandas as pd
from scipy import stats
import pywt
from .models import ROOT, CLASS_NAMES

LEGACY_DIR = ROOT / "models" / "traditional_ml"
PATIENT_DISJOINT_DIR = ROOT / "models" / "traditional_ml_patient_disjoint"
DIR = (
    PATIENT_DISJOINT_DIR
    if (PATIENT_DISJOINT_DIR / "classical_ml_metadata.json").is_file()
    else LEGACY_DIR
)


@lru_cache(maxsize=1)
def artifacts():
    metadata = json.loads((DIR / "classical_ml_metadata.json").read_text())
    features = json.loads((DIR / "selected_features.json").read_text())
    return (
        metadata,
        features,
        joblib.load(DIR / "scaler.joblib"),
        joblib.load(DIR / "label_encoder.joblib"),
        {
            "logistic_regression": joblib.load(DIR / "logistic_regression.joblib"),
            "svm": joblib.load(DIR / "svm.joblib"),
            "random_forest": joblib.load(DIR / "random_forest.joblib"),
        },
    )


def extract_all_features(y, sr=4000):
    y = np.asarray(y, dtype=np.float64)
    features = {}
    features.update(
        mean=np.mean(y),
        std=np.std(y),
        variance=np.var(y),
        rms=np.sqrt(np.mean(y**2)),
        max=np.max(y),
        min=np.min(y),
        peak_to_peak=np.ptp(y),
        median=np.median(y),
        skewness=stats.skew(y),
        kurtosis=stats.kurtosis(y),
        energy=np.sum(y**2),
        zero_crossing_rate=float(librosa.feature.zero_crossing_rate(y).mean()),
    )
    stft = librosa.stft(y, n_fft=512, hop_length=128)
    mag = np.abs(stft)
    power = mag**2
    features.update(
        spectral_centroid=float(librosa.feature.spectral_centroid(S=mag, sr=sr).mean()),
        spectral_bandwidth=float(
            librosa.feature.spectral_bandwidth(S=mag, sr=sr).mean()
        ),
        spectral_rolloff=float(
            librosa.feature.spectral_rolloff(S=mag, sr=sr, roll_percent=0.85).mean()
        ),
        spectral_flatness=float(librosa.feature.spectral_flatness(S=mag).mean()),
        spectral_energy=float(np.sum(power)),
    )
    freqs = librosa.fft_frequencies(sr=sr, n_fft=512)
    features["dominant_frequency"] = float(freqs[np.argmax(mag.mean(axis=1))])
    norm = power / (np.sum(power, axis=0, keepdims=True) + 1e-12)
    features["spectral_entropy"] = float(
        np.mean(-np.sum(norm * np.log2(norm + 1e-12), axis=0))
    )
    spectrum = np.abs(np.fft.rfft(y)) ** 2
    f = np.fft.rfftfreq(len(y), d=1 / sr)
    total = spectrum.sum() + 1e-12
    for key, lo, hi in [
        ("band_energy_0_500", 0, 500),
        ("band_energy_500_1000", 500, 1000),
        ("band_energy_1000_1500", 1000, 1500),
        ("band_energy_1500_2000", 1500, 2000),
    ]:
        features[key] = float(spectrum[(f >= lo) & (f < hi)].sum() / total)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=512, hop_length=128)
    features.update({f"mfcc_{i + 1}_mean": float(mfcc[i].mean()) for i in range(13)})
    coeff = pywt.wavedec(y, "db4", level=4)
    energies = np.array([np.sum(c**2) for c in coeff])
    norm_e = energies / (energies.sum() + 1e-12)
    features.update(
        wavelet_energy=float(energies.sum() + 1e-12),
        wavelet_entropy=float(-np.sum(norm_e * np.log2(norm_e + 1e-12))),
        wavelet_std=float(np.std(np.concatenate(coeff))),
    )
    return features


def predict_classical(y, sr=4000, model_ids=None, extracted_features=None):
    """Predict only requested classifiers (all three when model_ids is omitted)."""
    metadata, selected, scaler, encoder, models = artifacts()
    features = (
        extracted_features
        if extracted_features is not None
        else extract_all_features(y, sr)
    )
    missing = [k for k in selected if k not in features]
    if missing:
        raise ValueError(
            "Selected feature extraction incomplete: " + ", ".join(missing)
        )
    x = pd.DataFrame([[features[k] for k in selected]], columns=selected, dtype=float)
    if not np.isfinite(x.to_numpy()).all():
        raise ValueError(
            "Feature extraction produced invalid values; use a non-silent recording with enough signal variation"
        )
    out = []
    requested = set(model_ids) if model_ids is not None else set(models)
    unknown = requested - set(models)
    if unknown:
        raise ValueError("Unknown traditional model selection")
    for key, model in models.items():
        if key not in requested:
            continue
        start = time.perf_counter()
        inp = scaler.transform(x) if metadata["scaler_used"][key] else x

        def class_name(value):
            if isinstance(value, (int, np.integer)) or (
                isinstance(value, str) and value.isdigit()
            ):
                return str(encoder.inverse_transform([int(value)])[0])
            return str(value)

        predicted = model.predict(inp)[0]
        pred = class_name(predicted)
        scores = None
        if hasattr(model, "predict_proba"):
            vals = model.predict_proba(inp)[0]
            names = [class_name(c) for c in model.classes_]
            scores = dict(zip(names, map(float, vals)))
        out.append(
            {
                "model_id": key,
                "model_name": {
                    "logistic_regression": "Logistic Regression",
                    "svm": "SVM",
                    "random_forest": "Random Forest",
                }[key],
                "model_family": "traditional_ml",
                "predicted_class": pred,
                "scores": scores,
                "score_type": "predict_proba output; calibration not established"
                if scores
                else "class prediction only",
                "inference_ms": round((time.perf_counter() - start) * 1000, 2),
                "warnings": [],
            }
        )
    return out


def explain_classical(model_id, extracted_features, target_class):
    """Estimate local feature effects by replacing one feature with its baseline."""
    metadata, selected, scaler, encoder, models = artifacts()
    if model_id not in models:
        raise ValueError("Unknown traditional model selection")
    x = pd.DataFrame(
        [[extracted_features[key] for key in selected]], columns=selected, dtype=float
    )
    scaled = bool(metadata["scaler_used"][model_id])
    model_input = scaler.transform(x) if scaled else x.to_numpy(dtype=float)
    model = models[model_id]

    def class_name(value):
        if isinstance(value, (int, np.integer)) or (
            isinstance(value, str) and value.isdigit()
        ):
            return str(encoder.inverse_transform([int(value)])[0])
        return str(value)

    class_names = [class_name(value) for value in model.classes_]
    if target_class not in class_names:
        raise ValueError("Predicted class is missing from the model class mapping")
    target_index = class_names.index(target_class)

    def target_score(values):
        if hasattr(model, "predict_proba"):
            return float(model.predict_proba(values)[0, target_index]), "class probability change"
        decision = np.asarray(model.decision_function(values))[0]
        if np.ndim(decision) == 0:
            score = float(decision if target_index == 1 else -decision)
        else:
            score = float(decision[target_index])
        return score, "decision-score change"

    full_score, score_label = target_score(model_input)
    baseline_values = (
        np.zeros_like(model_input, dtype=float)
        if scaled
        else np.asarray(scaler.mean_, dtype=float)[None, :]
    )
    effects = []
    for index, feature_name in enumerate(selected):
        ablated = model_input.copy()
        ablated[0, index] = baseline_values[0, index]
        baseline_score, _ = target_score(ablated)
        effects.append(
            {"feature": feature_name, "impact": float(full_score - baseline_score)}
        )
    effects.sort(key=lambda item: abs(item["impact"]), reverse=True)
    return {
        "method": "Single-feature baseline ablation",
        "target_class": target_class,
        "score_label": score_label,
        "features": effects[:10],
    }
