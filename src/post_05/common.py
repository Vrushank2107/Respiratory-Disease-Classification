from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
OUTPUTS = ROOT / "outputs" / "post_05"
FEATURES_DIR = OUTPUTS / "features"
MODELS_DIR = OUTPUTS / "models"
FIGURES_DIR = OUTPUTS / "figures"
METRICS_DIR = OUTPUTS / "metrics"
PREDICTIONS_DIR = OUTPUTS / "predictions"
SEED = 42
TARGET_SR = 4000
TEST_SIZE = 0.20
VALIDATION_SIZE = 0.15
FEATURE_MAX_COUNT = 40
CORRELATION_THRESHOLD = 0.95
STFT_N_FFT = 512
STFT_HOP_LENGTH = 128
MEL_CHANNELS = 64
SPECTROGRAM_FRAMES = 128
CNN_EPOCHS = 20
CNN_BATCH_SIZE = 32
CNN_EARLY_STOPPING_PATIENCE = 4

META_COLUMNS = {
    "patient_id", "diagnosis", "recording_id", "chest_location",
    "acquisition_mode", "equipment", "cycle_number", "sound_label",
    "crackles", "wheezes", "start_time", "end_time", "original_sample_rate",
    "target_sample_rate", "original_samples", "processed_samples",
    "original_duration", "processed_duration", "original_peak", "original_rms",
    "normalized_peak", "processed_path", "processing_status", "wav_path",
    "split", "cycle_key", "filename", "status",
}


def ensure_output_dirs() -> None:
    for path in (FEATURES_DIR, MODELS_DIR, FIGURES_DIR, METRICS_DIR, PREDICTIONS_DIR):
        path.mkdir(parents=True, exist_ok=True)


def new_artifact_path(path: Path) -> Path:
    """Return a fresh path; never overwrite an existing result."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        return path
    raise FileExistsError(
        f"Refusing to overwrite existing artifact: {path}. "
        "Use a new output filename or archive the previous run first."
    )


def save_csv_new(frame: pd.DataFrame, path: Path, *, index: bool = False) -> Path:
    destination = new_artifact_path(path)
    frame.to_csv(destination, index=index)
    return destination


def feature_columns(frame: pd.DataFrame) -> list[str]:
    return [
        column for column in frame.select_dtypes(include="number").columns
        if column not in META_COLUMNS
    ]


def require_columns(frame: pd.DataFrame, columns: set[str], name: str) -> None:
    missing = sorted(columns - set(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")
