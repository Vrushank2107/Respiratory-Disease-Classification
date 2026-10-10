from pathlib import Path
import numpy as np
import pytest
from app.backend.services.models import ROOT,MODEL_SPECS,load_dl_model,predict_dl,CLASS_NAMES
from app.backend.services.audio_processing import decode_audio
from app.backend.services.classical import artifacts,predict_classical

@pytest.mark.parametrize('model_id',list(MODEL_SPECS))
def test_neural_artifacts_load_strictly(model_id):
    model,stats=load_dl_model(model_id)
    assert model is not None and stats

def test_lightweight_regression_reference():
    wav=ROOT/'data/processed/audio/cycles/101/1b1_Al_001.wav'
    y,_,_=decode_audio(wav)
    result=predict_dl('lightweight_mel_cnn',y)
    assert result['predicted_class']=='LRTI'
    assert abs(max(result['scores'].values())-0.3026)<0.002

def test_classical_artifacts_and_prediction():
    _,features,scaler,encoder,models=artifacts()
    assert len(features)==33 and len(models)==3
    wav=ROOT/'data/processed/audio/cycles/101/1b1_Al_001.wav'
    y,_,_=decode_audio(wav)
    predictions=predict_classical(y)
    assert len(predictions)==3
    assert all(p['predicted_class'] in CLASS_NAMES for p in predictions)

def test_resample_and_fixed_length():
    wav=ROOT/'data/processed/audio/cycles/101/1b1_Al_001.wav'
    y,sr,channels=decode_audio(wav)
    assert y.shape==(20000,) and sr==4000 and channels==1 and np.isfinite(y).all()

def test_notebook_sources_remain_present():
    assert all((ROOT/'notebooks'/f'{i:02d}_').parent.exists() for i in range(1,19))
