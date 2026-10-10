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
