import {
  Activity,
  ArrowRight,
  AudioLines,
  AudioWaveform,
  BrainCircuit,
  HeartPulse,
  Mic2,
} from 'lucide-react';
type Obj = Record<string, any>;
import { Card, Chart, MetricTable, Stat } from '../../components/shared';

export default function Overview({ data, go }: any) {
  const models = data.models || [];
  const diseaseRows = data.dataset?.classes || [];
  const totalPatients = diseaseRows.reduce(
    (total: number, row: Obj) => total + Number(row.count || 0),
    0,
  );
  const largestClass = [...diseaseRows].sort(
    (a: Obj, b: Obj) => Number(b.count || 0) - Number(a.count || 0),
  )[0];
  return (
    <>
      <div className="hero">
        <div>
          <div className="hero-tag">
            <span /> RESEARCH DASHBOARD
          </div>
          <h2>
            Understanding respiratory health
            <br />
            through sound.
          </h2>
          <p>
            A research platform for exploring lung-sound data, model evaluation, and explainability.
          </p>
          <button onClick={() => go('predict')} className="primary">
            <Mic2 size={16} /> Analyze an audio sample <span>→</span>
          </button>
        </div>
        <div className="hero-art">
          <div className="ring r1" />
          <div className="ring r2" />
          <div className="lungs">
            <Activity size={67} />
          </div>
          <div className="mini-label">
            <span className="pulse" /> SIGNAL ANALYSIS
          </div>
        </div>
      </div>
      <div className="stats">
        <Stat
          label="Patients"
          value={data.dataset?.patients}
          sub="cleaned dataset"
          icon={HeartPulse}
        />
        <Stat
          label="Recordings"
          value={data.dataset?.recordings}
          sub="unique recordings"
          icon={AudioLines}
        />
        <Stat
          label="Respiratory cycles"
          value={data.dataset?.cycles?.toLocaleString()}
          sub="after data cleaning"
          icon={AudioWaveform}
        />
        <Stat
          label="Model artifacts present"
          value={`${models.filter((m: Obj) => m.artifact_exists === true).length} / ${models.length || 7}`}
          sub="saved artifact inventory"
          icon={BrainCircuit}
        />
      </div>
      <div className="grid two">
        <Card
          title="Disease distribution"
          aside={<span className="card-note">Cleaned patient cohort</span>}
        >
          <Chart data={data.dataset?.classes} />
        </Card>
        <Card title="Saved model artifact inventory · 7 models">
          <div className="model-list">
            {models.map((m: Obj) => (
              <div className="model-row" key={m.model_id}>
                <span className={`status-dot ${m.artifact_exists ? 'good' : 'bad'}`} />
                <span>{m.model_name}</span>
                <small>{m.family === 'deep_learning' ? 'Deep learning' : 'Traditional ML'}</small>
                <b className={m.artifact_exists ? 'text-good' : 'text-bad'}>
                  {m.artifact_exists ? 'Artifact present' : 'Artifact missing'}
                </b>
              </div>
            ))}
          </div>
        </Card>
      </div>
      <div className="grid two">
        <Card title="Project walkthrough · Notebooks 01–18">
          <div className="overview-journey">
            {[
              ['01–03', 'Dataset & EDA', 'eda'],
              ['04–08', 'Audio & features', 'features'],
              ['09–13', 'Model development', 'development'],
              ['14–16', 'Evaluation & XAI', 'evaluation'],
              ['17–18', 'Artifacts & validation', 'artifacts'],
            ].map(([range, label, page]) => (
              <button key={range} onClick={() => go(page)}>
                <small>NOTEBOOKS {range}</small>
                <b>{label}</b>
                <ArrowRight size={15} />
              </button>
            ))}
          </div>
        </Card>
        <Card title="How to read the results">
          <ul className="plain-list overview-notes">
            <li>
              {largestClass
                ? `${largestClass.diagnosis} is the largest diagnosis group (${largestClass.count} of ${totalPatients} patients); overall accuracy can hide weak performance on rare classes.`
                : 'The cohort has uneven disease support; check per-class metrics alongside accuracy.'}
            </li>
            <li>
              Cycle-level and patient-level metrics answer different questions and are shown
              separately.
            </li>
            <li>Model scores are uncalibrated research outputs, not probabilities of diagnosis.</li>
            <li>Only Predict Audio accepts a recording and runs live inference.</li>
          </ul>
        </Card>
      </div>
      <Card
        title="Research highlights"
        aside={
          <span className="card-note">
            Saved cycle-level test results · splits not verified equivalent
          </span>
        }
      >
        <MetricTable rows={data.highlights || []} />
      </Card>
    </>
  );
}
