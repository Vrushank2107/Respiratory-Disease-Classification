"""Model definitions and inference adapters for saved project checkpoints."""

from __future__ import annotations

import json
import time
from pathlib import Path
from functools import lru_cache

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[3]
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
TARGET_SR, TARGET_SAMPLES, N_FFT, HOP, N_MELS = 4000, 20000, 512, 128, 64
EXPERIMENT_B_MODEL_ID = "experiment_b_sound_phenotype"
EXPERIMENT_B_CLASS_NAMES = ["Normal", "Crackles", "Wheezes", "Both"]
EXPERIMENT_B_DIR = ROOT / "models" / "deep_learning" / "experiment_b"


class LightweightMelCNN(nn.Module):
    def __init__(self, num_classes=8):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.30),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


class FeatureBranch(nn.Module):
    def __init__(self, dropout=0.30):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.projection = nn.Sequential(
            nn.Flatten(), nn.Linear(64, 64), nn.ReLU(), nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.projection(self.features(x))


class MultiFeatureCNN(nn.Module):
    def __init__(self, num_classes=8, dropout=0.30, classifier_dropout=0.40):
        super().__init__()
        self.logmel_branch = FeatureBranch(dropout)
        self.mfcc_branch = FeatureBranch(dropout)
        self.chroma_branch = FeatureBranch(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(192, 128),
            nn.ReLU(),
            nn.Dropout(classifier_dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, logmel, mfcc, chroma):
        return self.classifier(
            torch.cat(
                [
                    self.logmel_branch(logmel),
                    self.mfcc_branch(mfcc),
                    self.chroma_branch(chroma),
                ],
                dim=1,
            )
        )


class CNNLSTM(nn.Module):
    def __init__(self, num_classes=8):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d((2, 1)),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d((2, 1)),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.lstm = nn.LSTM(64, 64, num_layers=2, batch_first=True, dropout=0.30)
        self.dropout = nn.Dropout(0.30)
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.cnn(x).mean(dim=2).transpose(1, 2)
        x, _ = self.lstm(x)
        return self.classifier(self.dropout(x.mean(dim=1)))


class CompactRespiratoryCNN(nn.Module):
    def __init__(self, num_classes=4, dropout=0.25):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=15, stride=2, padding=7, bias=False),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=4),
            nn.Conv1d(16, 32, kernel_size=9, stride=1, padding=4, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=4),
            nn.Conv1d(32, 64, kernel_size=7, stride=1, padding=3, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=4),
            nn.Conv1d(64, 64, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.pool(self.features(x)))


MODEL_SPECS = {
    "lightweight_mel_cnn": (
        "Lightweight Mel CNN",
        "deep_learning/lightweight_mel_cnn/best_lightweight_mel_cnn.pt",
        "deep_learning/lightweight_mel_cnn/logmel_train_normalization.csv",
        "logmel",
    ),
    "multifeature_cnn": (
        "Multi-Feature CNN",
        "deep_learning/multifeature_cnn/best_multifeature_cnn.pt",
        "deep_learning/multifeature_cnn/multifeature_train_normalization.csv",
        "multi",
    ),
    "regularized_multifeature_cnn": (
        "Regularized Multi-Feature CNN (12B)",
        "deep_learning/regularized_multifeature_cnn/best_regularized_multifeature_cnn.pt",
        "deep_learning/multifeature_cnn/multifeature_train_normalization.csv",
        "regularized",
    ),
    "cnn_lstm": (
        "CNN-LSTM",
        "deep_learning/cnn_lstm/best_cnn_lstm.pt",
        "deep_learning/cnn_lstm/logmel_train_normalization.csv",
        "logmel_lstm",
    ),
}


def load_signal(path: str | Path):
    from .audio_processing import decode_audio

    return decode_audio(path, fixed_duration=True)


def extract_features(y):
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=TARGET_SR,
        n_fft=N_FFT,
        hop_length=HOP,
        n_mels=N_MELS,
        fmin=0,
        fmax=2000,
        power=2.0,
    )
    logmel = librosa.power_to_db(mel, ref=np.max).astype(np.float32)
    mfcc = librosa.feature.mfcc(
        y=y, sr=TARGET_SR, n_mfcc=13, n_fft=N_FFT, hop_length=HOP, fmin=0, fmax=2000
    ).astype(np.float32)
    chroma = librosa.feature.chroma_stft(
        y=y, sr=TARGET_SR, n_fft=N_FFT, hop_length=HOP, n_chroma=12, tuning=0.0
    ).astype(np.float32)
    return {"logmel": logmel, "mfcc": mfcc, "chroma": chroma}


def _stats(path, feature=None):
    df = pd.read_csv(path)
    if feature is None:
        stats = df.set_index("statistic")["value"]
        mean, std = float(stats["mean"]), float(stats["std"])
        return {"logmel": (mean, std)}
    return {str(r.feature): (float(r.mean), float(r.std)) for r in df.itertuples()}


@lru_cache(maxsize=1)
def load_experiment_b_model():
    config_path = EXPERIMENT_B_DIR / "config.json"
    classes_path = EXPERIMENT_B_DIR / "classes.json"
    checkpoint_path = EXPERIMENT_B_DIR / "experiment_b_cnn.pt"
    for path in (config_path, classes_path, checkpoint_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required Experiment B artifact missing: {path.name}")

    config = json.loads(config_path.read_text())
    class_mapping = json.loads(classes_path.read_text())
    class_names = [
        class_mapping[str(index)] for index in range(len(class_mapping))
    ]
    if class_names != EXPERIMENT_B_CLASS_NAMES:
        raise ValueError("Experiment B class mapping does not match the trained output order")
    if (
        config.get("sample_rate") != TARGET_SR
        or config.get("input_samples") != 12000
        or config.get("duration_seconds") != 3.0
        or config.get("channels") != 1
        or config.get("input_representation") != "raw_waveform"
        or config.get("amplitude_normalization") != "none"
        or config.get("long_input") != "center_crop"
        or config.get("short_input") != "symmetric_zero_pad"
    ):
        raise ValueError("Experiment B preprocessing configuration is unsupported")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if (
        checkpoint.get("experiment") != "B"
        or checkpoint.get("strategy") != "balanced"
        or checkpoint.get("seed") != config.get("seed")
        or checkpoint.get("class_names") != class_names
        or checkpoint.get("target_sr") != config.get("sample_rate")
        or checkpoint.get("target_samples") != config.get("input_samples")
    ):
        raise ValueError("Experiment B checkpoint metadata does not match its configuration")

    model = CompactRespiratoryCNN(num_classes=len(class_names))
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return model, class_names


def prepare_experiment_b_waveform(waveform: np.ndarray) -> np.ndarray:
    """Apply Experiment B's 3-second center-crop/symmetric-pad transform."""
    waveform = np.asarray(waveform, dtype=np.float32)
    if waveform.ndim != 1 or waveform.size == 0 or not np.isfinite(waveform).all():
        raise ValueError("Experiment B requires a finite, non-empty mono waveform")
    target_samples = 12000
    if len(waveform) > target_samples:
        start = (len(waveform) - target_samples) // 2
        waveform = waveform[start : start + target_samples]
    elif len(waveform) < target_samples:
        missing = target_samples - len(waveform)
        waveform = np.pad(waveform, (missing // 2, missing - missing // 2))
    return np.asarray(waveform, dtype=np.float32)


def predict_experiment_b(waveform: np.ndarray):
    model, class_names = load_experiment_b_model()
    prepared = prepare_experiment_b_waveform(waveform)
    input_tensor = torch.from_numpy(prepared[None, None])
    start = time.perf_counter()
    with torch.inference_mode():
        probabilities = torch.softmax(model(input_tensor), dim=1)[0].tolist()
    predicted_index = int(np.argmax(probabilities))
    return {
        "model_id": EXPERIMENT_B_MODEL_ID,
        "model_name": "Experiment B Sound Phenotype CNN",
        "model_family": "deep_learning",
        "predicted_class": class_names[predicted_index],
        "scores": dict(zip(class_names, probabilities)),
        "score_type": "softmax score; calibration not established",
        "inference_ms": round((time.perf_counter() - start) * 1000, 2),
        "warnings": [],
    }


@lru_cache(maxsize=4)
def load_dl_model(model_id):
    if model_id not in MODEL_SPECS:
        raise ValueError("Unknown neural model")
    name, rel, normalization, kind = MODEL_SPECS[model_id]
    ckpt = ROOT / "models" / rel
    norm = ROOT / "models" / normalization
    for p in (ckpt, norm):
        if not p.is_file():
            raise FileNotFoundError(f"Required artifact missing: {p.name}")
    checkpoint = torch.load(ckpt, map_location="cpu", weights_only=True)
    mapping = checkpoint.get("class_mapping") if isinstance(checkpoint, dict) else None
    if mapping:
        names = [
            x["diagnosis"] for x in sorted(mapping, key=lambda x: x["disease_label"])
        ]
        if names != CLASS_NAMES:
            raise ValueError(
                "Checkpoint class order does not match verified disease mapping"
            )
    state = (
        checkpoint.get("model_state_dict", checkpoint)
        if isinstance(checkpoint, dict)
        else checkpoint
    )
    if kind == "logmel":
        model = LightweightMelCNN(8)
    elif kind == "logmel_lstm":
        model = CNNLSTM(8)
    else:
        model = MultiFeatureCNN(
            8,
            dropout=0.50 if kind == "regularized" else 0.30,
            classifier_dropout=0.50 if kind == "regularized" else 0.40,
        )
    model.load_state_dict(state, strict=True)
    model.eval()
    stats = _stats(norm, None if kind.startswith("logmel") else "multi")
    return model, stats


def predict_dl(model_id, y):
    name, _, _, kind = MODEL_SPECS[model_id]
    model, stats = load_dl_model(model_id)
    f = extract_features(y)
    if kind.startswith("logmel"):
        mean, std = stats["logmel"]
        x = (f["logmel"] - mean) / std
        args = [torch.from_numpy(x[None, None]).float()]
    else:
        tensors = []
        for k in ("logmel", "mfcc", "chroma"):
            mean, std = stats[k]
            tensors.append(torch.from_numpy(((f[k] - mean) / std)[None, None]).float())
        args = tensors
    start = time.perf_counter()
    with torch.inference_mode():
        probs = torch.softmax(model(*args), dim=1)[0].numpy().tolist()
    return {
        "model_id": model_id,
        "model_name": name,
        "model_family": "deep_learning",
        "predicted_class": CLASS_NAMES[int(np.argmax(probs))],
        "scores": dict(zip(CLASS_NAMES, probs)),
        "score_type": "softmax score; calibration not established",
        "inference_ms": round((time.perf_counter() - start) * 1000, 2),
        "warnings": [],
    }


def explain_dl(model_id, y, target_class):
    """Return input-gradient saliency for one exact neural-model input cycle."""
    _, _, _, kind = MODEL_SPECS[model_id]
    model, stats = load_dl_model(model_id)
    features = extract_features(y)
    inputs = {}
    if kind.startswith("logmel"):
        mean, std = stats["logmel"]
        inputs["logmel"] = torch.from_numpy(
            ((features["logmel"] - mean) / std)[None, None]
        ).float().requires_grad_(True)
        args = [inputs["logmel"]]
    else:
        for key in ("logmel", "mfcc", "chroma"):
            mean, std = stats[key]
            inputs[key] = torch.from_numpy(
                ((features[key] - mean) / std)[None, None]
            ).float().requires_grad_(True)
        args = [inputs[key] for key in ("logmel", "mfcc", "chroma")]

    target_index = CLASS_NAMES.index(target_class)
    model.zero_grad(set_to_none=True)
    with torch.enable_grad():
        logits = model(*args)
        logits[0, target_index].backward()

    attributions = {}
    for key, value in inputs.items():
        saliency = (value.grad * value).abs().squeeze(0).squeeze(0)
        saliency = saliency.detach().cpu().numpy()
        peak = float(np.max(saliency))
        if peak > 0:
            saliency = saliency / peak
        attributions[key] = saliency.astype(np.float32)
    return {
        "method": "Input-gradient saliency (absolute gradient × normalized input)",
        "target_class": target_class,
        "feature_attributions": attributions,
    }
