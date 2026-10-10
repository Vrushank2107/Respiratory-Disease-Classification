from fastapi.testclient import TestClient
from app.backend.main import app
from app.backend.services.models import ROOT

client=TestClient(app)

def test_health_and_models():
    assert client.get('/api/health').json()['status']=='ok'
    models=client.get('/api/models').json()['models']
    assert len(models)==7
    assert all(m['available'] for m in models)

def test_saved_summaries_and_allowlisted_report():
    assert client.get('/api/overview').status_code==200
    assert client.get('/api/features/summary').status_code==200
    assert client.get('/api/evaluation/patient-level').status_code==200
    assert client.get('/api/reports/download/../../README.md').status_code==404

def test_predict_upload_reference():
    audio=ROOT/'data/processed/audio/cycles/101/1b1_Al_001.wav'
    response=client.post('/api/predict',data={'model_id':'lightweight_mel_cnn'},files={'file':('sample.wav',audio.read_bytes(),'audio/wav')})
    assert response.status_code==200,response.text
    assert response.json()['results'][0]['predicted_class']=='LRTI'

def test_invalid_upload_is_rejected():
    response=client.post('/api/predict',data={'model_id':'lightweight_mel_cnn'},files={'file':('sample.mp3',b'bad','audio/mpeg')})
    assert response.status_code==415


def test_all_seven_models_run_on_same_upload():
    import json
    audio=ROOT/'data/processed/audio/cycles/101/1b1_Al_001.wav'
    ids=['logistic_regression','svm','random_forest','lightweight_mel_cnn','multifeature_cnn','regularized_multifeature_cnn','cnn_lstm']
    response=client.post('/api/predict/compare',data={'model_ids':json.dumps(ids)},files={'file':('sample.wav',audio.read_bytes(),'audio/wav')})
    assert response.status_code==200,response.text
    result=response.json()
    assert {x['model_id'] for x in result['results']}==set(ids)
    assert result['failures']==[]

def test_audio_preview_has_real_signal_data():
    audio=ROOT/'data/processed/audio/cycles/101/1b1_Al_001.wav'
    response=client.post('/api/audio/preview',files={'file':('sample.wav',audio.read_bytes(),'audio/wav')})
    assert response.status_code==200,response.text
    payload=response.json()
    assert payload['waveform'] and len(payload['logmel'])==64 and len(payload['mfcc'])==13 and len(payload['chroma'])==12

def test_eda_chart_payloads_have_named_counts():
    payload=client.get('/api/eda/summary').json()
    assert sum(row['count'] for row in payload['disease_distribution'])==payload['patient_count']
    assert sum(row['count'] for row in payload['sound_distribution'])==payload['cycle_count']
    assert all(row.get('diagnosis') for row in payload['disease_distribution'])
    assert all(row.get('sound_label') for row in payload['sound_distribution'])

def test_patient_aggregation_explains_identical_saved_results():
    payload=client.get('/api/evaluation/patient-level').json()
    diagnostic=payload['aggregation_diagnostics']
    assert diagnostic['test_patients']==25
    assert diagnostic['matching_predicted_classes']==25
    assert diagnostic['predictions_identical'] is True
    assert 'true_class' in payload['majority_confusion'][0]
    assert 'Unnamed: 0' not in payload['majority_confusion'][0]

def test_status_returns_current_runtime_checks_without_paths():
    rows=client.get('/api/system/status').json()['validation_records']
    assert len(rows)==7
    assert all('path' not in row for row in rows)
    assert all(row['status']=='PASS' for row in rows)

def test_classical_probability_scores_use_disease_names():
    audio=ROOT/'data/processed/audio/cycles/101/1b1_Al_001.wav'
    response=client.post('/api/predict',data={'model_id':'logistic_regression'},files={'file':('sample.wav',audio.read_bytes(),'audio/wav')})
    assert response.status_code==200,response.text
    labels=set(response.json()['results'][0]['scores'])
    assert labels=={'Asthma','Bronchiectasis','Bronchiolitis','COPD','Healthy','LRTI','Pneumonia','URTI'}


def test_prediction_response_reports_duration_and_preprocessing():
    audio=ROOT/'data/processed/audio/cycles/101/1b1_Al_001.wav'
    response=client.post('/api/predict',data={'model_id':'lightweight_mel_cnn'},files={'file':('sample.wav',audio.read_bytes(),'audio/wav')})
    payload=response.json()
    assert payload['status']=='success'
    assert payload['input_duration_seconds']>0
    assert payload['neural_input_duration_seconds']==5.0
    assert 'first five seconds' in payload['handling']


def test_prediction_rejects_audio_over_configured_duration(monkeypatch):
    import app.backend.main as api
    audio=ROOT/'data/processed/audio/cycles/101/1b1_Al_001.wav'
    monkeypatch.setattr(api,'MAX_DURATION_SECONDS',0.001)
    response=client.post('/api/predict',data={'model_id':'lightweight_mel_cnn'},files={'file':('sample.wav',audio.read_bytes(),'audio/wav')})
    assert response.status_code==413
    assert 'duration limit' in response.json()['detail']


def test_prediction_rejects_silent_audio():
    import io
    import numpy as np
    import soundfile as sf
    buffer=io.BytesIO()
    sf.write(buffer,np.zeros(4000,dtype=np.float32),4000,format='WAV')
    response=client.post('/api/predict',data={'model_id':'lightweight_mel_cnn'},files={'file':('silent.wav',buffer.getvalue(),'audio/wav')})
    assert response.status_code==422
    assert 'silent' in response.json()['detail']
