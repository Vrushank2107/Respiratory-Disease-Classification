"""Audio decoding and training-compatible length transforms."""

from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

TARGET_SR = 4000
TARGET_SAMPLES = 20000


def decode_audio(path: str | Path, *, fixed_duration: bool = True):
    try:
        audio, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    except Exception as exc:
        raise ValueError("Invalid or unsupported WAV audio") from exc
    if audio.size == 0 or not np.isfinite(audio).all():
        raise ValueError("Audio is empty or contains invalid samples")

    channels = int(audio.shape[1])
    # Arithmetic channel mean matches the existing inference preprocessing.
    waveform = audio.mean(axis=1)
    source_rate = int(sample_rate)
    if source_rate <= 0:
        raise ValueError("Unsupported sample rate")
    if source_rate != TARGET_SR:
        waveform = librosa.resample(
            waveform, orig_sr=source_rate, target_sr=TARGET_SR, res_type="soxr_hq"
        )
    waveform = np.asarray(waveform, dtype=np.float32)
    if waveform.size == 0 or not np.isfinite(waveform).all():
        raise ValueError("Audio could not be resampled safely")
    if not np.any(waveform):
        raise ValueError("Audio is silent; provide a recording with a non-zero signal")
    if fixed_duration:
        waveform = fixed_length_waveform(waveform)
    return waveform, source_rate, channels


def fixed_length_waveform(waveform: np.ndarray) -> np.ndarray:
    """Pad or truncate to the five-second neural model input length."""
    waveform = np.asarray(waveform, dtype=np.float32)
    if len(waveform) < TARGET_SAMPLES:
        return np.pad(waveform, (0, TARGET_SAMPLES - len(waveform)))
    return waveform[:TARGET_SAMPLES]


def read_native_mono(path: str | Path):
    """Load an input like Notebook 04: float audio, arithmetic mono mix."""
    try:
        audio, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    except Exception as exc:
        raise ValueError("Invalid or unsupported WAV audio") from exc
    if audio.size == 0 or not np.isfinite(audio).all():
        raise ValueError("Audio is empty or contains invalid samples")
    sample_rate = int(sample_rate)
    if sample_rate <= 0:
        raise ValueError("Unsupported sample rate")
    channels = int(audio.shape[1])
    return audio.mean(axis=1).astype(np.float32), sample_rate, channels


def resample_peak_normalize(waveform: np.ndarray, sample_rate: int) -> np.ndarray:
    """Resample and peak-normalize one cycle as in Notebook 04."""
    waveform = np.asarray(waveform, dtype=np.float32)
    if waveform.size == 0 or not np.isfinite(waveform).all():
        raise ValueError("Audio cycle is empty or contains invalid samples")
    if sample_rate != TARGET_SR:
        waveform = librosa.resample(
            waveform,
            orig_sr=sample_rate,
            target_sr=TARGET_SR,
            res_type="soxr_hq",
        )
    waveform = np.asarray(waveform, dtype=np.float32)
    peak = float(np.max(np.abs(waveform))) if waveform.size else 0.0
    if peak <= 0:
        raise ValueError("Audio cycle is silent; provide a non-zero signal")
    normalized = waveform / peak
    if not np.isfinite(normalized).all():
        raise ValueError("Audio cycle could not be normalized safely")
    return normalized.astype(np.float32)


def split_recording_into_windows(
    waveform: np.ndarray, sample_rate: int, window_samples: int = TARGET_SAMPLES
) -> list[tuple[int, np.ndarray]]:
    """Cover an unannotated WAV with consecutive, independently normalized windows.

    This is a generic-recording fallback, not respiratory-cycle detection. The
    model's fixed five-second input size defines the default window length.
    """
    waveform = np.asarray(waveform, dtype=np.float32)
    if waveform.size == 0 or not np.isfinite(waveform).all():
        raise ValueError("Audio recording is empty or contains invalid samples")
    if sample_rate != TARGET_SR:
        waveform = librosa.resample(
            waveform,
            orig_sr=sample_rate,
            target_sr=TARGET_SR,
            res_type="soxr_hq",
        )
    waveform = np.asarray(waveform, dtype=np.float32)
    if not np.isfinite(waveform).all() or not np.any(waveform):
        raise ValueError("Audio is silent or could not be resampled safely")
    windows = []
    for offset in range(0, len(waveform), window_samples):
        segment = waveform[offset : offset + window_samples]
        if not segment.size or not np.any(segment):
            continue
        peak = float(np.max(np.abs(segment)))
        windows.append((len(windows) + 1, (segment / peak).astype(np.float32)))
    if not windows:
        raise ValueError("No non-silent audio windows were found")
    if len(windows) > 64:
        raise ValueError("Recording creates more than 64 windows; use a shorter WAV")
    return windows


def parse_cycle_annotations(
    text: str, duration_seconds: float
) -> list[tuple[float, float]]:
    """Read ICBHI-style start/end timestamps and validate them against a WAV."""
    if not np.isfinite(duration_seconds) or duration_seconds <= 0:
        raise ValueError("WAV duration is invalid")
    intervals = []
    previous_end = 0.0
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        columns = line.split()
        if len(columns) < 2:
            raise ValueError(f"Annotation line {line_number} needs start and end times")
        try:
            start, end = float(columns[0]), float(columns[1])
        except ValueError as exc:
            raise ValueError(f"Annotation line {line_number} has invalid timestamps") from exc
        if not np.isfinite([start, end]).all() or start < 0 or end <= start:
            raise ValueError(f"Annotation line {line_number} has invalid cycle boundaries")
        if end > duration_seconds + 1 / TARGET_SR:
            raise ValueError(f"Annotation line {line_number} extends beyond the WAV duration")
        if intervals and start < previous_end:
            raise ValueError(f"Annotation line {line_number} overlaps the previous cycle")
        intervals.append((start, min(end, duration_seconds)))
        previous_end = end
        if len(intervals) > 64:
            raise ValueError("Annotation file contains more than 64 cycles; use a shorter recording")
    if not intervals:
        raise ValueError("Annotation file contains no respiratory cycles")
    return intervals


def extract_annotated_cycle(
    waveform: np.ndarray, sample_rate: int, start_time: float, end_time: float
) -> np.ndarray:
    """Extract and apply Notebook 04 resampling/normalization to one annotated cycle."""
    start = int(round(start_time * sample_rate))
    end = int(round(end_time * sample_rate))
    if start < 0 or end <= start or start >= len(waveform):
        raise ValueError("Cycle annotation has invalid audio boundaries")
    return resample_peak_normalize(waveform[start : min(end, len(waveform))], sample_rate)
