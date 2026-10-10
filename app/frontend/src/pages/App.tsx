import { useEffect, useState, type ReactNode } from 'react';
import { Activity, AudioLines, BarChart3, BrainCircuit, Database, FileText, HeartPulse, Mic2, ShieldAlert, SlidersHorizontal, Stethoscope, Workflow } from 'lucide-react';
import { get, upload } from '../services/api';
import type { ModelInfo, PredictionPayload } from '../types/api';
import Overview from './sections/Overview';
import Eda from './sections/Eda';
import AudioPage from './sections/AudioPage';
import Features from './sections/Features';
import Predict from './sections/Predict';
import Comparison from './sections/Comparison';
import Patient from './sections/Patient';
import Xai from './sections/Xai';
import Reports from './sections/Reports';
import Status from './sections/Status';

type PageId = 'overview'|'eda'|'audio'|'features'|'predict'|'comparison'|'patient'|'xai'|'reports'|'status';
const nav = [['Overview','overview',Activity],['Dataset & EDA','eda',Database],['Audio pipeline','audio',AudioLines],['Features & PCA','features',SlidersHorizontal],['Predict audio','predict',Mic2],['Model comparison','comparison',BarChart3],['Patient analysis','patient',HeartPulse],['Explainability','xai',BrainCircuit],['Reports','reports',FileText],['System status','status',Workflow]] as const;
const routes: Partial<Record<PageId,string>> = { overview:'/api/overview', eda:'/api/eda/summary', features:'/api/features/summary', comparison:'/api/evaluation/models', patient:'/api/evaluation/patient-level', xai:'/api/xai/summary', reports:'/api/reports', status:'/api/system/status' };

export default function App() {
  const [page,setPage]=useState<PageId>('overview');
  const [data,setData]=useState<Record<string,any>>({});
  const [error,setError]=useState('');
  const [loading,setLoading]=useState(false);
  const [models,setModels]=useState<ModelInfo[]>([]);
  const [file,setFile]=useState<File|null>(null);
  const [selected,setSelected]=useState<string[]>([]);
  const [result,setResult]=useState<PredictionPayload|null>(null);
  const [apiState,setApiState]=useState<'checking'|'connected'|'error'>('checking');

  useEffect(()=>{get('/api/health').then(()=>{setApiState('connected');return get('/api/models')}).then(x=>{setModels(x.models);setSelected(x.models.filter((m:ModelInfo)=>m.available).map((m:ModelInfo)=>m.model_id))}).catch(()=>setApiState('error'));},[]);
  useEffect(()=>{const route=routes[page];if(!route)return;setLoading(true);setError('');get(route).then(setData).catch((e:Error)=>setError(e.message)).finally(()=>setLoading(false));},[page]);
  const heading=nav.find(item=>item[1]===page)?.[0]||'Research workspace';
  async function run(){if(!file||!selected.length)return;setLoading(true);setError('');setResult(null);try{setResult(await upload('/api/predict/compare',file,{model_ids:JSON.stringify(selected)}));}catch(e:any){setError(e.message)}finally{setLoading(false)}}
  const pageContent={overview:<Overview data={data} models={models} go={setPage}/>,eda:<Eda data={data}/>,audio:<AudioPage/>,features:<Features data={data}/>,predict:<Predict models={models} selected={selected} setSelected={setSelected} file={file} setFile={setFile} setResult={setResult} run={run} result={result} loading={loading}/>,comparison:<Comparison data={data}/>,patient:<Patient data={data}/>,xai:<Xai data={data}/>,reports:<Reports data={data}/>,status:<Status data={data}/>} satisfies Record<PageId,ReactNode>;
  return <div className="shell">
    <aside className="sidebar">
      <div className="sidebar-top"><div className="brand"><div className="brand-icon"><Activity size={20}/></div><div className="brand-copy"><b>RespiraLab</b><small>RESEARCH PLATFORM</small></div></div></div>
      <div className="nav-label">WORKSPACE</div><nav>{nav.map(([label,id,Icon])=><button key={id} onClick={()=>setPage(id)} className={page===id?'active':''} title={label} aria-label={label} aria-current={page===id?'page':undefined}><Icon size={18}/><span className="nav-text">{label}</span>{id==='predict'&&<span className="nav-badge">LIVE</span>}</button>)}</nav>
      <div className="sidebar-bottom"><div className="local-chip"><span className="pulse"/><span className="local-text">Local environment</span></div></div>
    </aside>
    <main><header className="topbar"><div className="crumb"><span>RespiraLab</span><span className="slash">/</span><b>{heading}</b></div><div className="top-right"><span className={`api-state api-${apiState}`}><i/> API {apiState==='checking'?'checking':apiState==='connected'?'connected':'unavailable'}</span><span className="avatar">RD</span></div></header>
      <div className="content"><div className="page-head"><div><div className="eyebrow">PDS · DATA MINING · LUNG SOUND ANALYSIS</div><h1>{heading}</h1><p>{page==='predict'?'Run saved respiratory sound models against a WAV recording.':`Explore ${heading.toLowerCase()} from the verified research workspace.`}</p></div>{page==='predict'&&<span className="research-pill"><ShieldAlert size={15}/> Research use only</span>}</div>
        {error&&<div className="error" role="alert"><ShieldAlert size={17}/>{error}</div>}{loading&&<div className="loading" role="status"><span/> Processing…</div>}
        {pageContent[page]}<div className="disclaimer"><Stethoscope size={15}/><span><b>Academic research only.</b> Model outputs are experimental and are not intended for clinical use or diagnosis.</span></div>
      </div>
    </main>
  </div>;
}
