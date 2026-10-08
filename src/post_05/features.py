from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
from scipy.stats import kurtosis, skew

from .common import PROCESSED, require_columns


def _waveform_features(y: np.ndarray, sr: int) -> dict[str, float]:
    eps = np.finfo(float).eps
    rms = float(np.sqrt(np.mean(np.square(y))))
    abs_y = np.abs(y)
    crossings = np.mean(np.signbit(y[1:]) != np.signbit(y[:-1])) if len(y) > 1 else 0.0
    out = {
        "duration_s": len(y) / sr,
        "amplitude_mean": float(np.mean(y)),
        "amplitude_std": float(np.std(y)),
        "amplitude_min": float(np.min(y)),
        "amplitude_max": float(np.max(y)),
        "peak_abs": float(np.max(abs_y)),
        "rms": rms,
        "zero_crossing_rate": float(crossings),
        "crest_factor": float(np.max(abs_y) / max(rms, eps)),
        "skewness": float(skew(y, bias=False)) if len(y) > 2 else 0.0,
        "kurtosis": float(kurtosis(y, bias=False)) if len(y) > 3 else 0.0,
    }
    spectrum = np.abs(np.fft.rfft(y)) ** 2
    frequencies = np.fft.rfftfreq(len(y), d=1 / sr)
    total = float(spectrum.sum()) + eps
    for low, high in ((0, 100), (100, 250), (250, 500), (500, 1000), (1000, sr // 2 + 1)):
        mask = (frequencies >= low) & (frequencies < high)
        out[f"band_power_{low}_{high}"] = float(spectrum[mask].sum() / total)
    out["spectral_centroid_hz"] = float(np.sum(frequencies * spectrum) / total)
    out["spectral_spread_hz"] = float(np.sqrt(np.sum(((frequencies - out["spectral_centroid_hz"]) ** 2) * spectrum) / total))
    cumulative = np.cumsum(spectrum)
    out["spectral_rolloff_85_hz"] = float(frequencies[min(np.searchsorted(cumulative, 0.85 * total), len(frequencies) - 1)])
    return out


def extract_cycle_features(audio_path: Path, sr: int = 4000) -> dict[str, float]:
    y, actual_sr = sf.read(audio_path, dtype="float32", always_2d=False)
    if y.ndim == 2:
        y = y.mean(axis=1)
    if actual_sr != sr:
        y = librosa.resample(y, orig_sr=actual_sr, target_sr=sr)
    if y.size < 8 or not np.isfinite(y).all():
        raise ValueError("Audio is empty, too short, or contains non-finite samples")

    features = _waveform_features(y, sr)
    n_fft, hop = 512, 128
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=n_fft, hop_length=hop)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=32, n_fft=n_fft, hop_length=hop)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=n_fft, hop_length=hop)
    for name, matrix in (("mfcc", mfcc), ("mel_log", librosa.power_to_db(mel, ref=np.max)), ("chroma", chroma)):
        for index, row in enumerate(matrix):
            features[f"{name}_{index:02d}_mean"] = float(np.mean(row))
            features[f"{name}_{index:02d}_std"] = float(np.std(row))

    try:
        import pywt
    except ImportError:
        pass
    else:
        coeffs = pywt.wavedec(y, "db4", level=4)
        energy = np.asarray([np.sum(np.square(c)) for c in coeffs], dtype=float)
        energy /= max(float(energy.sum()), np.finfo(float).eps)
        for index, value in enumerate(energy):
            features[f"wavelet_energy_{index}"] = float(value)
    return features


def build_cycle_feature_table(metadata_path: Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    metadata_path = metadata_path or PROCESSED / "processed_cycles.csv"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Missing Notebook 04 cycle metadata: {metadata_path}")
    metadata = pd.read_csv(metadata_path)
    require_columns(metadata, {"patient_id", "diagnosis", "recording_id", "cycle_number", "sound_label", "processed_path", "processing_status"}, "processed_cycles.csv")
    metadata = metadata.loc[metadata["processing_status"].eq("success")].copy()
    rows, failures = [], []
    for row in metadata.itertuples(index=False):
        relative = Path(str(row.processed_path))
        audio_path = relative if relative.is_absolute() else PROCESSED / relative
        key = f"{int(row.patient_id)}_{row.recording_id}_{row.cycle_number}"
        if not audio_path.is_file():
            failures.append({"cycle_key": key, "error": f"Processed WAV not found: {audio_path}"})
            continue
        try:
            values = extract_cycle_features(audio_path, sr=int(getattr(row, "target_sample_rate", 4000)))
            record = row._asdict()
            record["cycle_key"] = key
            record.update(values)
            rows.append(record)
        except Exception as exc:
            failures.append({"cycle_key": key, "error": str(exc)})
    return pd.DataFrame(rows), pd.DataFrame(failures, columns=["cycle_key", "error"])
