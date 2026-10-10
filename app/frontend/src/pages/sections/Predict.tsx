import { useObjectUrl } from '../../hooks/useObjectUrl';
import { Download, Mic2, Play, Trash2, UploadCloud } from 'lucide-react';
import { useRef } from 'react';
import { Card, Heatmap, SignalPlot, titleCase } from '../../components/shared';

type Obj = Record<string, any>;

export default function Predict({
  models,
  selected,
  setSelected,
  file,
  setFile,
  annotationFile,
  setAnnotationFile,
  setResult,
  run,
  result,
  loading,
}: any) {
  const available = models.filter((model: Obj) => model.available);
  const audioUrl = useObjectUrl(file);
  const wavInput = useRef<HTMLInputElement>(null);
  const annotationInput = useRef<HTMLInputElement>(null);

  function toggle(id: string) {
    setSelected((current: string[]) =>
      current.includes(id) ? current.filter((value) => value !== id) : [...current, id],
    );
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

  return (
    <div className="predict-layout">
      <div className="predict-controls">
        <Card title="1 · Choose an audio recording">
          <label className="dropzone">
            <input
              ref={wavInput}
              type="file"
              accept="audio/wav,.wav"
              onChange={(event) => {
                setFile(event.target.files?.[0] || null);
                setAnnotationFile(null);
                setResult(null);
                event.currentTarget.value = '';
                if (annotationInput.current) annotationInput.current.value = '';
              }}
            />
            <div className="upload-icon">
              <UploadCloud />
            </div>
            <b>{file ? file.name : 'Choose a WAV recording'}</b>
            <span>
              {file
                ? `${(file.size / 1024 / 1024).toFixed(2)} MB · WAV`
                : 'Click to browse · maximum 20 MB · 120 seconds'}
            </span>
          </label>
          {file && (
            <button
              className="remove-file"
              type="button"
              onClick={() => {
                setFile(null);
                setAnnotationFile(null);
                setResult(null);
                if (wavInput.current) wavInput.current.value = '';
                if (annotationInput.current) annotationInput.current.value = '';
              }}
            >
              <Trash2 size={15} /> Remove WAV
            </button>
          )}
          {file && <audio controls src={audioUrl} />}
          <div className="annotation-upload">
            <span>Cycle annotations (.txt) · optional</span>
            <input
              ref={annotationInput}
              type="file"
              accept=".txt,text/plain"
              onChange={(event) => {
                setAnnotationFile(event.target.files?.[0] || null);
                setResult(null);
                event.currentTarget.value = '';
              }}
            />
            {annotationFile && (
              <span className="attached-file">
                <small title={annotationFile.name}>{annotationFile.name}</small>
                <button
                  type="button"
                  aria-label="Remove annotation file"
                  onClick={() => {
                    setAnnotationFile(null);
                    setResult(null);
                    if (annotationInput.current) annotationInput.current.value = '';
                  }}
                >
                  <Trash2 size={14} /> Remove
                </button>
              </span>
            )}
          </div>
          <p className="small-note">
            Readable WAV recordings up to 20 MB and 120 seconds can be processed; predictions are
            limited to the model’s eight trained labels. Without annotations, the full recording is
            split into consecutive five-second windows. These are not detected respiratory cycles,
            and results on arbitrary audio are exploratory because the models were trained on
            cycles. Add a matching ICBHI-style .txt file when you have cycle timestamps.
          </p>
        </Card>

        <Card
          title="2 · Select models"
          aside={
            <button
              className="text-button"
              onClick={() => setSelected(available.map((model: Obj) => model.model_id))}
            >
              Select all available
            </button>
          }
        >
          <div className="select-list">
            {models.map((model: Obj) => (
              <label
                className={`select-row ${!model.available ? 'disabled' : ''}`}
                key={model.model_id}
              >
                <input
                  type="checkbox"
                  disabled={!model.available}
                  checked={selected.includes(model.model_id)}
                  onChange={() => toggle(model.model_id)}
                />
                <span>
                  <b>{model.display_name}</b>
                  <small>
                    {model.family === 'deep_learning' ? 'Deep learning' : 'Traditional ML'}
                    {model.artifact_version ? ` · ${model.artifact_version}` : ''}
                    {model.artifact_size_bytes
                      ? ` · ${(model.artifact_size_bytes / 1024).toFixed(0)} KB`
                      : ''}{' '}
                    · {model.available ? 'Ready' : model.reason}
                  </small>
                </span>
                <span className={`status-pill ${model.available ? 'ready' : 'off'}`}>
                  {model.available ? 'Ready' : 'Unavailable'}
                </span>
              </label>
            ))}
          </div>
        </Card>
        <button
          className="primary run"
          disabled={!file || !selected.length || loading}
          onClick={run}
        >
          <Play size={16} />
          {loading ? 'Running…' : 'Run prediction'}
          <span>{selected.length} selected</span>
        </button>
      </div>

      <Card
        title={result ? 'Live prediction workflow' : 'Prediction results'}
        aside={
          result?.results?.length ? (
            <span className="results-tools">
              <span className="status-pill ready">
                {result.results.length} model{result.results.length === 1 ? '' : 's'}
              </span>
            </span>
          ) : undefined
        }
      >
        <p className="subtle result-intro">
          Results are research scores, not calibrated clinical confidence. Scores from different
          model families may not be directly comparable.
        </p>
        {!result && (
          <div className="result-empty">
            <Mic2 size={28} />
            <b>Your results will appear here</b>
            <span>
              Choose any WAV recording, optionally add cycle timestamps, then select an available
              model.
            </span>
          </div>
        )}
        {result && (
          <>
            <div className="file-meta">
              <b title={result.filename}>{result.filename}</b>
              <span>
                {result.source_sample_rate} Hz · {result.channels} channel(s) ·{' '}
                {result.input_duration_seconds}s · {result.segment_count ?? result.cycle_count}{' '}
                {result.input_mode === 'annotated_recording' || result.input_mode === 'single_cycle'
                  ? 'cycle(s)'
                  : 'window(s)'}
              </span>
            </div>
            <div className="callout">
              Input handling:{' '}
              {result.input_mode === 'annotated_recording'
                ? 'cycles extracted using the uploaded annotation timestamps'
                : result.input_mode === 'automatic_windows'
                  ? 'complete WAV processed in consecutive five-second windows'
                  : 'legacy single-cycle API input'}
            </div>
            {result.warnings?.length > 0 && (
              <ul className="result-warnings">
                {result.warnings.map((warning: string) => (
                  <li key={warning}>{warning}</li>
                ))}
              </ul>
            )}
            <details className="result-processing">
              <summary>Audio processing details</summary>
              <p>{result.handling}</p>
            </details>
            {result.audio_analysis && (
              <section className="live-audio-analysis">
                <h4>Processing applied to this recording</h4>
                <div className="processing-steps">
                  {result.audio_analysis.processing_steps?.map((step: Obj) => (
                    <div className="processing-step" key={step.name}>
                      <span className={`status-pill ${step.status === 'passed' ? 'ready' : 'off'}`}>
                        {step.status === 'passed' ? 'Passed' : 'Not selected'}
                      </span>
                      <div>
                        <b>{step.name}</b>
                        <small>{step.detail}</small>
                      </div>
                    </div>
                  ))}
                </div>
                <p className="small-note">
                  The source plot covers the full recording after channel downmix. The separate
                  prepared-input plot and feature plots show the first{' '}
                  {result.input_mode === 'automatic_windows' ? 'window' : 'cycle'} (
                  {result.audio_analysis.preview_cycle_number}) at 4 kHz. Traditional models use
                  each complete segment and their selected engineered features. Feature plots are
                  live transforms; training normalization is applied inside each model.
                </p>
                <div className="grid two live-waveforms">
                  <div className="waveform-panel">
                    <b>Source waveform · downmixed</b>
                    <SignalPlot values={result.audio_analysis.source_waveform} />
                  </div>
                  <div className="waveform-panel">
                    <b>Prepared model input · 4 kHz</b>
                    <SignalPlot values={result.audio_analysis.model_input_waveform} />
                  </div>
                </div>
                <details className="result-processing live-features">
                  <summary>View live Log-Mel, MFCC, and Chroma features</summary>
                  <div className="grid two">
                    <div>
                      <b>Log-Mel · 64 bands</b>
                      <Heatmap matrix={result.audio_analysis.logmel} />
                    </div>
                    <div>
                      <b>MFCC · 13 coefficients</b>
                      <Heatmap matrix={result.audio_analysis.mfcc} />
                    </div>
                  </div>
                  <b>Chroma · 12 bands</b>
                  <Heatmap matrix={result.audio_analysis.chroma} />
                  <small className="small-note">
                    These are raw feature views. Neural models use their saved training statistics
                    to normalize their inputs.
                  </small>
                </details>
              </section>
            )}
            {result.status !== 'success' && (
              <div className="callout">
                Prediction status: {result.status.replace('_', ' ')}. Review the model results and
                any failures below.
              </div>
            )}
            {result.results?.length > 0 && (
              <div className="prediction-grid" aria-label="Prediction results by model">
                {result.results.map((model: Obj) => {
                  const scores = Object.entries(model.scores || {}).sort(
                    (a: any, b: any) => b[1] - a[1],
                  );
                  const topScore = scores.length ? Number(scores[0][1]) : null;
                  return (
                    <section className="prediction" key={model.model_id}>
                      <div className="prediction-head">
                        <b>{model.model_name}</b>
                        <span>
                          {model.inference_ms} ms inference
                          <br />
                          {model.explanation_ms ?? '—'} ms XAI
                        </span>
                      </div>
                      <div className="pred-label">
                        <span>Predicted class</span>
                        <strong>{model.predicted_class}</strong>
                      </div>
                      <div className="score-caveat">
                        <span>Top model score</span>
                        <b>
                          {topScore === null ? 'Not reported' : `${(topScore * 100).toFixed(1)}%`}
                        </b>
                        <small>
                          Uncalibrated output score, not a probability of correctness. Use it to
                          compare classes within this model only.
                        </small>
                      </div>
                      {scores.length > 0 && (
                        <details className="score-details">
                          <summary>View all {scores.length} class scores</summary>
                          <div className="scores">
                            {scores.map(([label, value]) => (
                              <div className="score" key={label}>
                                <div>
                                  <span>{label}</span>
                                  <b>{(Number(value) * 100).toFixed(1)}%</b>
                                </div>
                                <i>
                                  <em
                                    style={{
                                      width: `${Math.max(0, Math.min(Number(value) * 100, 100))}%`,
                                    }}
                                  />
                                </i>
                              </div>
                            ))}
                          </div>
                          <small className="subtle">{model.score_type}</small>
                        </details>
                      )}
                      {model.cycle_predictions?.length > 1 && (
                        <details className="cycle-results">
                          <summary>
                            View {model.cycle_predictions.length}{' '}
                            {result.input_mode === 'automatic_windows' ? 'window' : 'cycle'}{' '}
                            predictions
                          </summary>
                          <div>
                            {model.cycle_predictions.map((cycle: Obj) => (
                              <p key={cycle.cycle_number}>
                                <span>
                                  {result.input_mode === 'automatic_windows' ? 'Window' : 'Cycle'}{' '}
                                  {cycle.cycle_number}
                                </span>
                                <b>{cycle.predicted_class}</b>
                              </p>
                            ))}
                          </div>
                        </details>
                      )}
                      <details className="live-explanation">
                        <summary>Explain this model’s prediction</summary>
                        {model.explanation ? (
                          <div className="explanation-content">
                            <p className="small-note">
                              {model.explanation.method}. {model.explanation.scope}. Target
                              class(es):{' '}
                              {(model.explanation.target_classes || []).join(', ') || '—'}.
                            </p>
                            {model.explanation.feature_attributions &&
                              Object.entries(model.explanation.feature_attributions).map(
                                ([feature, matrix]) => (
                                  <div key={feature}>
                                    <b>{titleCase(feature)} saliency</b>
                                    <Heatmap matrix={matrix} />
                                  </div>
                                ),
                              )}
                            {model.explanation.features && (
                              <div className="feature-effects">
                                {model.explanation.features.map((item: Obj) => {
                                  const largest = Math.max(
                                    ...model.explanation.features.map((row: Obj) =>
                                      Math.abs(Number(row.impact)),
                                    ),
                                    1e-12,
                                  );
                                  const magnitude = (Math.abs(Number(item.impact)) / largest) * 100;
                                  return (
                                    <div className="feature-effect" key={item.feature}>
                                      <span title={item.feature}>{titleCase(item.feature)}</span>
                                      <i>
                                        <em
                                          className={
                                            Number(item.impact) >= 0 ? 'supports' : 'opposes'
                                          }
                                          style={{ width: `${magnitude}%` }}
                                        />
                                      </i>
                                      <b>
                                        {Number(item.impact) >= 0 ? '+' : ''}
                                        {Number(item.impact).toFixed(4)}
                                      </b>
                                    </div>
                                  );
                                })}
                                <small className="small-note">
                                  Positive values indicate the feature moved the target score above
                                  its baseline; negative values indicate movement below baseline.
                                  Values are local model score changes, not probabilities unless
                                  explicitly labeled “class probability change.”
                                </small>
                              </div>
                            )}
                            <p className="xai-caveat">
                              This is a model-behavior explanation for the uploaded audio, not
                              evidence of a clinical cause or proof that the prediction is correct.
                            </p>
                          </div>
                        ) : (
                          <p className="small-note">
                            A live explanation could not be generated for this model. The prediction
                            is still available; check the API logs for details.
                          </p>
                        )}
                      </details>
                    </section>
                  );
                })}
              </div>
            )}
            {result.failures?.length > 0 && (
              <div className="failure-list">
                {result.failures.map((failure: Obj) => (
                  <div className="failure" key={failure.model_id}>
                    <b>{failure.model_id} failed</b>
                    <span>{failure.error}</span>
                  </div>
                ))}
              </div>
            )}
            <button className="secondary" onClick={downloadResult}>
              <Download size={15} /> Download result JSON
            </button>
          </>
        )}
      </Card>
    </div>
  );
}
