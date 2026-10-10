"""Stable import surface for migrated Lightweight Mel CNN inference."""
from .models import LightweightMelCNN, load_dl_model, predict_dl

MODEL_ID = 'lightweight_mel_cnn'

def load_model():
    """Load the saved network and verified training normalization strictly."""
    return load_dl_model(MODEL_ID)

def predict_waveform(waveform):
    """Predict from a prepared 4 kHz waveform using the saved pipeline."""
    return predict_dl(MODEL_ID, waveform)

__all__ = ['LightweightMelCNN', 'load_model', 'predict_waveform']
