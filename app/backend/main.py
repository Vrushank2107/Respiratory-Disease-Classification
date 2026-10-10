from __future__ import annotations
import json, tempfile, time, os, logging
from pathlib import Path
import pandas as pd
import soundfile as sf
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from starlette.concurrency import run_in_threadpool
from .services.models import ROOT, MODEL_SPECS, load_signal, load_dl_model, predict_dl, CLASS_NAMES
from .services.audio_processing import decode_audio, fixed_length_waveform
from .services.classical import artifacts, extract_all_features, predict_classical
from .schemas import PredictionResponse

app=FastAPI(title='Respiratory Sound Research API',version='1.0.0',description='Local academic research interface; predictions are not diagnoses.')
configured_origins=['https://respiratory-disease-classification-seven.vercel.app',*[origin.strip().rstrip('/') for origin in os.getenv('FRONTEND_ORIGINS','').split(',') if origin.strip()]]
app.add_middleware(CORSMiddleware,allow_origins=['http://localhost:5173','http://127.0.0.1:5173',*configured_origins],allow_origin_regex=r'https?://(localhost|127\.0\.0\.1|192\.168\.\d{1,3}\.\d{1,3}|10\.\d{1,3}\.\d{1,3}\.\d{1,3}|172\.(1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3}):5173',allow_methods=['*'],allow_headers=['*'])
MAX_BYTES=int(os.getenv("MAX_UPLOAD_BYTES",20*1024*1024))
MAX_DURATION_SECONDS=float(os.getenv("MAX_AUDIO_DURATION_SECONDS",120))
logger=logging.getLogger(__name__)
MODEL_META={
'logistic_regression':('Logistic Regression','traditional_ml'),'svm':('SVM','traditional_ml'),'random_forest':('Random Forest','traditional_ml'),
'lightweight_mel_cnn':('Lightweight Mel CNN','deep_learning'),'multifeature_cnn':('Multi-Feature CNN','deep_learning'),'regularized_multifeature_cnn':('Regularized Multi-Feature CNN (12B)','deep_learning'),'cnn_lstm':('CNN-LSTM','deep_learning')}

def csv(path):
 p=ROOT/path
 if not p.is_file(): return []
 try: return json.loads(pd.read_csv(p).to_json(orient='records'))
 except Exception: return []

def public_model(mid):
 name,family=MODEL_META[mid]; reasons=[]; checked=False
 try:
  if family=='deep_learning': load_dl_model(mid)
  else:
   metadata,features,scaler,encoder,estimators=artifacts(); checked=mid in estimators
  checked=True
 except Exception as e: reasons.append(str(e))
 return {'model_id':mid,'display_name':name,'family':family,'available':checked,'artifact_validation':'loaded strictly' if checked else 'failed','reason':None if checked else (reasons[0] if reasons else 'required artifact unavailable')}

@app.get('/api/health')
def health(): return {'status':'ok','service':'respiratory-research-api'}
@app.get('/api/models')
def models(): return {'models':[public_model(m) for m in MODEL_META]}
@app.get('/api/overview')
def overview():
 pats=pd.read_csv(ROOT/'data/processed/patients_clean.csv'); rec=pd.read_csv(ROOT/'data/processed/recordings_clean.csv'); cycles=pd.read_csv(ROOT/'data/processed/cycles_clean.csv')
 return {'title':'Respiratory Disease Classification','description':'Research platform for lung-sound data processing, classification, evaluation, and interpretation.','dataset':{'patients':int(pats.patient_id.nunique()),'recordings':int(rec.recording_id.nunique()),'cycles':int(len(cycles)),'classes':pats.diagnosis.value_counts().rename_axis('diagnosis').reset_index(name='count').to_dict('records')},'models':[public_model(m) for m in MODEL_META],'highlights':csv('reports/final_artifacts/cycle_level_model_comparison.csv'),'disclaimer':'For academic research only. Model outputs are not medical diagnoses.'}
@app.get('/api/eda/summary')
def eda(): return {'disease_distribution':csv('data/processed/eda/disease_distribution.csv'),'sound_distribution':csv('data/processed/eda/sound_distribution.csv'),'patient_audio_summary':csv('data/processed/eda/patient_audio_summary.csv'),'disease_sound_counts':csv('data/processed/eda/disease_sound_counts.csv'),'cleaning_summary':csv('data/processed/cleaning_summary.csv'),'patient_count':int(pd.read_csv(ROOT/'data/processed/patients_clean.csv').patient_id.nunique()),'recording_count':int(pd.read_csv(ROOT/'data/processed/recordings_clean.csv').recording_id.nunique()),'cycle_count':len(pd.read_csv(ROOT/'data/processed/cycles_clean.csv')),'pca':csv('data/processed/pca_clustering/pca_explained_variance.csv'),'clusters':csv('data/processed/pca_clustering/kmeans_cluster_analysis.csv')}
@app.get('/api/features/summary')
def features(): return {'selected_features':csv('data/processed/feature_selection/selected_features.csv'),'ranking':csv('data/processed/feature_selection/final_feature_selection_ranking.csv'),'statistics':csv('data/processed/analysis/feature_statistics.csv'),'variance':csv('data/processed/analysis/feature_variance.csv'),'pca':csv('data/processed/pca_clustering/pca_explained_variance.csv'),'clusters':csv('data/processed/pca_clustering/kmeans_cluster_analysis.csv')}
@app.get('/api/clustering/summary')
def clustering(): return {'pca':csv('data/processed/pca_clustering/pca_explained_variance.csv'),'clusters':csv('data/processed/pca_clustering/kmeans_cluster_analysis.csv')}
def cnn_baseline_confusion():
    saved_matrix = ROOT/'data/processed/deep_learning/cnn_baseline/cnn_baseline_confusion_matrix.csv'
    if saved_matrix.is_file():
        return csv('data/processed/deep_learning/cnn_baseline/cnn_baseline_confusion_matrix.csv')

    predictions = csv('data/processed/deep_learning/cnn_baseline/cnn_test_predictions.csv')
    if not predictions:
        return []

    matrix = {label: {prediction: 0 for prediction in CLASS_NAMES} for label in CLASS_NAMES}
    for row in predictions:
        actual = row.get('true_disease')
        predicted = row.get('predicted_disease')
        if actual in matrix and predicted in matrix[actual]:
            matrix[actual][predicted] += 1
    return [{'true_class': label, **matrix[label]} for label in CLASS_NAMES]

@app.get('/api/evaluation/models')
def evaluations(): return {'cycle_level':csv('reports/final_artifacts/cycle_level_model_comparison.csv'),'classical':csv('data/processed/classical_ml/classical_ml_per_disease_results.csv'),'confusion':{'cnn_baseline':cnn_baseline_confusion(),**{k:csv(f'data/processed/deep_learning/{k}/{f}') for k,f in [('cnn_lstm','cnn_lstm_confusion_matrix.csv'),('multifeature_cnn','multifeature_cnn_confusion_matrix.csv'),('multifeature_cnn_12b','confusion_matrix.csv')]}}}
@app.get('/api/evaluation/cycle-level')
def cycle_eval(): return csv('reports/final_artifacts/cycle_level_model_comparison.csv')
@app.get('/api/evaluation/patient-level')
def patient_eval():
    base=ROOT/'data/processed/deep_learning/patient_level_prediction'
    def matrix(filename):
        frame=pd.read_csv(base/filename,index_col=0)
        frame.index.name='true_class'
        return json.loads(frame.reset_index().to_json(orient='records'))
    majority=pd.read_csv(base/'patient_majority_predictions.csv')
    mean=pd.read_csv(base/'patient_probability_predictions.csv')
    same=majority[['patient_id','predicted_disease']].merge(mean[['patient_id','predicted_disease']],on='patient_id',suffixes=('_majority','_mean'))
    identical=int((same.predicted_disease_majority==same.predicted_disease_mean).sum())
    return {'metrics':csv('reports/final_artifacts/patient_level_model_comparison.csv'),'majority_confusion':matrix('majority_vote_confusion_matrix.csv'),'mean_confusion':matrix('mean_probability_confusion_matrix.csv'),'aggregation_diagnostics':{'test_patients':int(len(same)),'matching_predicted_classes':identical,'predictions_identical':identical==len(same),'explanation':'Both aggregation methods produced the same winning class for every saved test patient; their reported metrics and confusion matrices therefore match.' if identical==len(same) else 'The two saved aggregation methods differ on at least one patient.'}}
@app.get('/api/xai/summary')
def xai(): return {'summary':csv('data/processed/deep_learning/xai/multi_class_xai_summary.csv'),'metadata':json.loads((ROOT/'data/processed/deep_learning/xai/gradcam_example_metadata.json').read_text()),'images':[p.name for p in (ROOT/'data/processed/deep_learning/xai').glob('*.png')]}
@app.get('/api/xai/image/{name}')
def xai_image(name:str):
 p=(ROOT/'data/processed/deep_learning/xai'/name).resolve(); base=(ROOT/'data/processed/deep_learning/xai').resolve()
 if p.parent!=base or p.suffix.lower()!='.png' or not p.is_file(): raise HTTPException(404,'Image not found')
 return FileResponse(p,media_type='image/png')
@app.get('/api/reports')
def reports():
 names=['final_results_report.md','model_registry.csv','cycle_level_model_comparison.csv','patient_level_model_comparison.csv','final_project_validation.csv','checkpoint_integrity_check.csv','prediction_validation.csv','model_package_manifest.csv']
 return {'reports':[{'name':n,'url':'/api/reports/download/'+n} for n in names if (ROOT/'reports/final_artifacts'/n).is_file()]}
@app.get('/api/reports/download/{name}')
def report_download(name:str):
 p=(ROOT/'reports/final_artifacts'/name).resolve(); base=(ROOT/'reports/final_artifacts').resolve()
 if p.parent!=base or not p.is_file(): raise HTTPException(404,'Report not found')
 return FileResponse(p,filename=p.name)
@app.get('/api/system/status')
def status():
 checked=[public_model(m) for m in MODEL_META]
 records=[{'category':'Model runtime','item':m['display_name'],'requirement':'Checkpoint, normalization, strict load','status':'PASS' if m['available'] else 'UNAVAILABLE','details':'Loaded and ready for inference' if m['available'] else m['reason']} for m in checked]
 return {'models':checked,'validation_records':records,'api':'ok','checked_at':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
async def run_prediction(file,model_ids):
 raw=await file.read(MAX_BYTES+1)
 if len(raw)>MAX_BYTES: raise HTTPException(413,f'Upload exceeds the {MAX_BYTES / (1024 * 1024):g} MiB limit')
 if not file.filename or Path(file.filename).suffix.lower()!='.wav': raise HTTPException(415,'Upload a WAV file')
 if not model_ids: raise HTTPException(422,'Select at least one model')
 if len(set(model_ids))!=len(model_ids) or any(m not in MODEL_META for m in model_ids): raise HTTPException(422,'Unknown or duplicate model identifier')
 path=None
 try:
  with tempfile.NamedTemporaryFile(suffix='.wav',delete=False) as f: f.write(raw); path=Path(f.name)
  info=sf.info(path)
  if info.samplerate<=0 or info.frames<=0:
   raise HTTPException(422,'WAV file has no valid audio frames')
  input_duration=float(info.frames/info.samplerate)
  if input_duration>MAX_DURATION_SECONDS:
   raise HTTPException(413,f'Audio exceeds the {MAX_DURATION_SECONDS:g}-second duration limit')
  classical_y,source_sr,channels=decode_audio(path,fixed_duration=False)
  y=fixed_length_waveform(classical_y)
  results=[]; failures=[]
  classical_ids=[mid for mid in model_ids if MODEL_META[mid][1]=='traditional_ml']
  classical_features=await run_in_threadpool(extract_all_features,classical_y,4000) if classical_ids else None
  for mid in model_ids:
   try:
    if MODEL_META[mid][1]=='deep_learning': results.append(await run_in_threadpool(predict_dl,mid,y))
    else:
     model_result=await run_in_threadpool(predict_classical,classical_y,4000,[mid],classical_features)
     results.extend(model_result)
   except Exception:
    logger.exception('Inference failed for selected model %s',mid)
    failures.append({'model_id':mid,'error':'This model could not process the recording. Check that the audio is valid and review backend logs.'})
  outcome='failure' if failures and not results else ('partial_failure' if failures else 'success')
  return {'status':outcome,'filename':Path(file.filename).name,'source_sample_rate':source_sr,'target_sample_rate':4000,'channels':channels,'input_duration_seconds':round(input_duration,3),'neural_input_duration_seconds':5.0,'handling':'Downmixed by arithmetic channel mean and resampled with soxr_hq to 4 kHz. Neural models use the first five seconds, padded when shorter. Traditional models use the full resampled recording.','results':results,'failures':failures,'partial_success':bool(results and failures),'disclaimer':'Research output only; not a medical diagnosis.'}
 except HTTPException: raise
 except ValueError as e: raise HTTPException(422,str(e))
 except Exception:
  logger.exception('Uploaded audio could not be processed')
  raise HTTPException(422,'Could not decode or process this WAV recording. Check that it is valid PCM audio.')
 finally:
  if path:
   try: path.unlink(missing_ok=True)
   except OSError: pass
@app.post('/api/audio/preview')
async def audio_preview(file:UploadFile=File(...)):
 raw=await file.read(MAX_BYTES+1)
 if len(raw)>MAX_BYTES: raise HTTPException(413,f'Upload exceeds the {MAX_BYTES / (1024 * 1024):g} MiB limit')
 if not file.filename or Path(file.filename).suffix.lower()!='.wav': raise HTTPException(415,'Upload a WAV file')
 path=None
 try:
  with tempfile.NamedTemporaryFile(suffix='.wav',delete=False) as f: f.write(raw); path=Path(f.name)
  info=sf.info(path)
  if info.samplerate<=0 or info.frames<=0:
   raise HTTPException(422,'WAV file has no valid audio frames')
  duration=info.frames/info.samplerate
  if duration>MAX_DURATION_SECONDS:
   raise HTTPException(413,f'Audio exceeds the {MAX_DURATION_SECONDS:g}-second duration limit')
  y,sr,channels=decode_audio(path,fixed_duration=False)
  from .services.models import extract_features
  feats=await run_in_threadpool(extract_features,y)
  # Bound payload size for long recordings while preserving a useful overview.
  stride=max(1,len(y)//600)
  return {'filename':Path(file.filename).name,'source_sample_rate':sr,'target_sample_rate':4000,'channels':channels,'duration_seconds':round(len(y)/4000,3),'waveform':y[::stride][:600].round(5).tolist(),'waveform_time_step_seconds':stride/4000,'feature_settings':'Derived at 4 kHz with the Notebook 12 FFT/hop configuration; each feature is displayed without training normalization.','logmel':feats['logmel'].round(2).tolist(),'mfcc':feats['mfcc'].round(2).tolist(),'chroma':feats['chroma'].round(3).tolist()}
 except HTTPException: raise
 except ValueError as e: raise HTTPException(422,str(e))
 except Exception:
  logger.exception('Uploaded audio preview could not be processed')
  raise HTTPException(422,'Could not decode or process audio. Check that the WAV file is valid.')
 finally:
  if path:
   try: path.unlink(missing_ok=True)
   except OSError: pass

@app.post('/api/predict',response_model=PredictionResponse)
async def predict(file:UploadFile=File(...),model_id:str=Form(...)): return await run_prediction(file,[model_id])
@app.post('/api/predict/compare',response_model=PredictionResponse)
async def compare(file:UploadFile=File(...),model_ids:str=Form(...)):
 try: ids=json.loads(model_ids)
 except Exception: raise HTTPException(422,'model_ids must be a JSON array')
 if not isinstance(ids,list): raise HTTPException(422,'model_ids must be a JSON array')
 return await run_prediction(file,ids)
