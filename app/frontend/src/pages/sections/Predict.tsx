import { useObjectUrl } from '../../hooks/useObjectUrl';
import { Download, Mic2, Play, UploadCloud } from 'lucide-react';
import { Card } from '../../components/shared';

type Obj = Record<string, any>;

export default function Predict({ models, selected, setSelected, file, setFile, setResult, run, result, loading }: any) {
  const available = models.filter((model: Obj) => model.available);
  const audioUrl = useObjectUrl(file);

  function toggle(id: string) {
    setSelected((current: string[]) => current.includes(id)
      ? current.filter((value) => value !== id)
      : [...current, id]);
  }

  function downloadResult() {
    if (!result) return;
    const blob = new Blob([JSON.stringify(result, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = 'respiratory-prediction.json';
    link.click();
    setTimeout(() => URL.revokeObjectURL(url), 0);
  }

  return <div className="predict-layout">
    <div className="predict-controls">
      <Card title="1 · Choose an audio recording">
        <label className="dropzone">
          <input type="file" accept="audio/wav,.wav" onChange={(event) => {
            setFile(event.target.files?.[0] || null);
            setResult(null);
          }}/>
          <div className="upload-icon"><UploadCloud/></div>
          <b>{file ? file.name : 'Choose a WAV recording'}</b>
          <span>{file ? `${(file.size / 1024 / 1024).toFixed(2)} MB · WAV` : 'Click to browse · maximum 20 MB'}</span>
        </label>
        {file && <audio controls src={audioUrl}/>}
        <p className="small-note">The selected recording is sent to the configured API for analysis. Mono and multichannel recordings are supported.</p>
      </Card>

      <Card title="2 · Select models" aside={<button className="text-button" onClick={() => setSelected(available.map((model: Obj) => model.model_id))}>Select all available</button>}>
        <div className="select-list">{models.map((model: Obj) => <label className={`select-row ${!model.available ? 'disabled' : ''}`} key={model.model_id}>
          <input type="checkbox" disabled={!model.available} checked={selected.includes(model.model_id)} onChange={() => toggle(model.model_id)}/>
          <span><b>{model.display_name}</b><small>{model.family === 'deep_learning' ? 'Deep learning' : 'Traditional ML'} · {model.available ? 'Ready' : model.reason}</small></span>
          <span className={`status-pill ${model.available ? 'ready' : 'off'}`}>{model.available ? 'Ready' : 'Unavailable'}</span>
        </label>)}</div>
      </Card>
      <button className="primary run" disabled={!file || !selected.length || loading} onClick={run}>
        <Play size={16}/>{loading ? 'Running…' : 'Run prediction'}<span>{selected.length} selected</span>
      </button>
    </div>

    <Card title="Prediction results" aside={result?.results?.length ? <span className="results-tools"><span className="status-pill ready">{result.results.length} model{result.results.length === 1 ? '' : 's'}</span></span> : undefined}>
      <p className="subtle result-intro">Results are research scores, not calibrated clinical confidence. Scores from different model families may not be directly comparable.</p>
      {!result && <div className="result-empty"><Mic2 size={28}/><b>Your results will appear here</b><span>Choose a WAV file and at least one available model.</span></div>}
      {result && <>
        <div className="file-meta"><b title={result.filename}>{result.filename}</b><span>{result.source_sample_rate} Hz · {result.channels} channel(s) · {result.input_duration_seconds}s</span></div>
        <details className="result-processing"><summary>Audio processing details</summary><p>{result.handling}</p></details>
        {result.status !== 'success' && <div className="callout">Prediction status: {result.status.replace('_', ' ')}. Review the model results and any failures below.</div>}
        {result.results?.length > 0 && <div className="prediction-grid" aria-label="Prediction results by model">{result.results.map((model: Obj) => {
          const scores = Object.entries(model.scores || {}).sort((a: any, b: any) => b[1] - a[1]);
          const topScore = scores.length ? Number(scores[0][1]) : null;
          return <section className="prediction" key={model.model_id}>
            <div className="prediction-head"><b>{model.model_name}</b><span>{model.inference_ms} ms</span></div>
            <div className="pred-label"><span>Predicted class</span><strong>{model.predicted_class}</strong></div>
            <p className="score-caveat">Top model score: {topScore === null ? 'not reported' : `${(topScore * 100).toFixed(1)}%`} · not calibrated confidence</p>
            {scores.length > 0 && <details className="score-details">
              <summary>View all {scores.length} class scores</summary>
              <div className="scores">{scores.map(([label, value]) => <div className="score" key={label}>
                <div><span>{label}</span><b>{(Number(value) * 100).toFixed(1)}%</b></div>
                <i><em style={{ width: `${Math.max(0, Math.min(Number(value) * 100, 100))}%` }}/></i>
              </div>)}</div>
              <small className="subtle">{model.score_type}</small>
            </details>}
          </section>;
        })}</div>}
        {result.failures?.length > 0 && <div className="failure-list">{result.failures.map((failure: Obj) => <div className="failure" key={failure.model_id}><b>{failure.model_id} failed</b><span>{failure.error}</span></div>)}</div>}
        <button className="secondary" onClick={downloadResult}><Download size={15}/> Download result JSON</button>
      </>}
    </Card>
  </div>;
}
