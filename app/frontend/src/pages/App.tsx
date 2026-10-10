import { useEffect, useState, type ReactNode } from 'react';
import {
  Activity,
  BarChart3,
  BrainCircuit,
  Database,
  FileText,
  Mic2,
  Stethoscope,
  SlidersHorizontal,
  ShieldAlert,
} from 'lucide-react';
import { get, upload } from '../services/api';
import type { ModelInfo, PredictionPayload } from '../types/api';
import Overview from './sections/Overview';
import Eda from './sections/Eda';
import Predict from './sections/Predict';
import ResearchPages from './sections/ResearchPages';

type PageId =
  | 'overview'
  | 'eda'
  | 'features'
  | 'development'
  | 'evaluation'
  | 'artifacts'
  | 'predict';
const nav = [
  ['Overview', 'overview', Activity],
  ['Dataset & EDA', 'eda', Database],
  ['Audio & features', 'features', SlidersHorizontal],
  ['Model development', 'development', BrainCircuit],
  ['Evaluation & XAI', 'evaluation', BarChart3],
  ['Artifacts & validation', 'artifacts', FileText],
  ['Predict audio', 'predict', Mic2],
] as const;
const routes: Partial<Record<PageId, string>> = {
  overview: '/api/overview',
  eda: '/api/eda/summary',
};

export default function App() {
  const [page, setPage] = useState<PageId>('overview');
  const [data, setData] = useState<Record<string, any>>({});
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [file, setFile] = useState<File | null>(null);
  const [annotationFile, setAnnotationFile] = useState<File | null>(null);
  const [selected, setSelected] = useState<string[]>([]);
  const [result, setResult] = useState<PredictionPayload | null>(null);
  const [apiState, setApiState] = useState<'checking' | 'connected' | 'error'>('checking');
  const [modelsLoaded, setModelsLoaded] = useState(false);

  useEffect(() => {
    if (page !== 'predict' || modelsLoaded) return;
    setApiState('checking');
    get('/api/health')
      .then(() => get('/api/models'))
      .then((x) => {
        setModels(x.models);
        setSelected(
          x.models.filter((m: ModelInfo) => m.available).map((m: ModelInfo) => m.model_id),
        );
        setApiState('connected');
        setModelsLoaded(true);
      })
      .catch(() => setApiState('error'));
  }, [page, modelsLoaded]);
  useEffect(() => {
    const route = routes[page];
    if (route) {
      setLoading(true);
      setError('');
      get(route)
        .then(setData)
        .catch((e: Error) => setError(e.message))
        .finally(() => setLoading(false));
      return;
    }
    const groupedRoutes: Partial<Record<PageId, Record<string, string>>> = {
      features: { features: '/api/features/summary', pipeline: '/api/pipeline/summary' },
      development: { development: '/api/development/summary' },
      evaluation: {
        comparison: '/api/evaluation/models',
        patient: '/api/evaluation/patient-level',
        xai: '/api/xai/summary',
      },
      artifacts: { reports: '/api/reports', status: '/api/system/status' },
    };
    const section = groupedRoutes[page];
    if (!section) return;
    setLoading(true);
    setError('');
    Promise.all(Object.entries(section).map(async ([key, url]) => [key, await get(url)] as const))
      .then((entries) => setData(Object.fromEntries(entries)))
      .catch((e: Error) => setError(e.message))
      .finally(() => setLoading(false));
  }, [page]);
  const heading = nav.find((item) => item[1] === page)?.[0] || 'Research workspace';
  async function run() {
    if (!file || !selected.length) return;
    setLoading(true);
    setError('');
    setResult(null);
    try {
      setResult(
        await upload(
          '/api/predict/compare',
          file,
          {
            model_ids: JSON.stringify(selected),
          },
          { annotations: annotationFile },
        ),
      );
    } catch (e: any) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }
  const pageContent = {
    overview: <Overview data={data} go={setPage} />,
    eda: <Eda data={data} />,
    features: <ResearchPages section="features" data={data} />,
    development: <ResearchPages section="development" data={data} />,
    evaluation: <ResearchPages section="evaluation" data={data} />,
    artifacts: <ResearchPages section="artifacts" data={data} />,
    predict: (
      <Predict
        models={models}
        selected={selected}
        setSelected={setSelected}
        file={file}
        setFile={setFile}
        annotationFile={annotationFile}
        setAnnotationFile={setAnnotationFile}
        setResult={setResult}
        run={run}
        result={result}
        loading={loading}
      />
    ),
  } satisfies Record<PageId, ReactNode>;
  return (
    <div className="shell">
      <aside className="sidebar">
        <div className="sidebar-top">
          <div className="brand">
            <div className="brand-icon">
              <Activity size={20} />
            </div>
            <div className="brand-copy">
              <b>RespiraLab</b>
              <small>RESEARCH PLATFORM</small>
            </div>
          </div>
        </div>
        <div className="nav-label">WORKSPACE</div>
        <nav>
          {nav.map(([label, id, Icon]) => (
            <button
              key={id}
              onClick={() => setPage(id)}
              className={page === id ? 'active' : ''}
              title={label}
              aria-label={label}
              aria-current={page === id ? 'page' : undefined}
            >
              <Icon size={18} />
              <span className="nav-text">{label}</span>
              {id === 'predict' && <span className="nav-badge">LIVE</span>}
            </button>
          ))}
        </nav>
      </aside>
      <main>
        <header className="topbar">
          <div className="crumb">
            <span>RespiraLab</span>
            <span className="slash">/</span>
            <b>{heading}</b>
          </div>
          <div className="top-right">
            {page === 'predict' && (
              <span className={`api-state api-${apiState}`}>
                <i /> API{' '}
                {apiState === 'checking'
                  ? 'checking'
                  : apiState === 'connected'
                    ? 'connected'
                    : 'unavailable'}
              </span>
            )}
            {page !== 'predict' && (
              <span className="api-state">
                <i /> Saved research artifacts
              </span>
            )}
            <span className="avatar">RD</span>
          </div>
        </header>
        <div className="content">
          <div className="page-head">
            <div>
              <div className="eyebrow">PDS · DATA MINING · LUNG SOUND ANALYSIS</div>
              <h1>{heading}</h1>
              <p>
                {page === 'predict'
                  ? 'Run saved respiratory sound models against a WAV recording.'
                  : page === 'overview'
                    ? 'A notebook-to-app research record: curated data, signal processing, model experiments, evaluation and saved artifacts.'
                    : `Read saved notebook outputs, figures and evaluation records for ${heading.toLowerCase()}.`}
              </p>
            </div>
            {page === 'predict' && (
              <span className="research-pill">
                <ShieldAlert size={15} /> Research use only
              </span>
            )}
          </div>
          {error && (
            <div className="error" role="alert">
              <ShieldAlert size={17} />
              {error}
            </div>
          )}
          {loading && (
            <div className="loading" role="status">
              <span /> Processing…
            </div>
          )}
          {pageContent[page]}
          <div className="disclaimer">
            <Stethoscope size={15} />
            <span>
              <b>Academic research only.</b> Model outputs are experimental and are not intended for
              clinical use or diagnosis.
            </span>
          </div>
        </div>
      </main>
    </div>
  );
}
