import { useState } from 'react';
import { BookOpen, CircleHelp, Layers3 } from 'lucide-react';
import {
  CartesianGrid,
  Line,
  LineChart,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import { Card, Chart, Table } from '../../components/shared';
import Comparison from './Comparison';
import Patient from './Patient';
import Xai from './Xai';
import Reports from './Reports';
import Status from './Status';

type Section = 'features' | 'development' | 'evaluation' | 'artifacts';
const papers = [
  {
    title: 'Lung Disease Recognition Methods Using Audio-based Analysis with Machine Learning',
    authors: 'Sabry et al. · Heliyon · 2024',
    doi: 'https://doi.org/10.1016/j.heliyon.2024.e26218',
    relation:
      'Review context for respiratory-audio datasets, signal preparation, hand-crafted features and learning approaches. It is a survey, not the source of this project’s exact 4 kHz processing settings.',
  },
  {
    title:
      'Lung Sound Classification With Multi-Feature Integration Utilizing Lightweight CNN Model',
    authors: 'Wanasinghe et al. · IEEE Access · 2024',
    doi: 'https://doi.org/10.1109/ACCESS.2024.3361943',
    relation:
      'Related to the project’s Mel, MFCC and chroma feature families and lightweight CNN/XAI experiments. The paper’s data, configuration and reported accuracy are not project results.',
  },
  {
    title:
      'Advanced Deep Learning Techniques for Lung Sound Classification: Binary, Multi-Class and Ensemble Approach',
    authors: 'Perera & Pathmakumara · CSECS · 2025',
    doi: 'https://doi.org/10.1109/CSECS64665.2025.11009363',
    relation:
      'Related work for CNN, CNN-LSTM and ensemble approaches. This project evaluates individual model artifacts; it does not implement the paper’s CNN/CNN-LSTM ensemble or claim its results.',
  },
  {
    title: 'Hierarchical Embedded System Based on FPGA for Classification of Respiratory Diseases',
    authors: 'Han et al. · IEEE Access · 2025',
    doi: 'https://doi.org/10.1109/ACCESS.2025.3573162',
    relation:
      'Related work on respiratory-cycle features and classification. Its hierarchical FPGA system and evaluation protocol are not implemented here; its reported metrics are not comparable to this project.',
  },
  {
    title: 'Lung-Sound Respiratory Disease Classification via Multiple-Instance Learning',
    authors: 'Nguyen et al. · IEEE Access · 2026',
    doi: 'https://doi.org/10.1109/ACCESS.2026.3669914',
    relation:
      'Patient-level multiple-instance learning is relevant future-work context. This project aggregates saved cycle outputs and does not implement the paper’s multi-channel MIL method or dataset.',
  },
];

function Tabs({
  items,
  value,
  setValue,
}: {
  items: string[];
  value: string;
  setValue: (x: string) => void;
}) {
  return (
    <div className="research-tabs" role="tablist">
      {items.map((item) => (
        <button
          role="tab"
          aria-selected={value === item}
          className={value === item ? 'selected' : ''}
          key={item}
          onClick={() => setValue(item)}
        >
          {item}
        </button>
      ))}
    </div>
  );
}

function Sources({ compact = false }: { compact?: boolean }) {
  const list = compact ? papers.slice(0, 2) : papers;
  return (
    <Card title={compact ? 'Research context' : 'Research papers and method provenance'}>
      <p className="subtle">
        These papers are cited as related work. A citation identifies context; it does not imply
        that a paper’s dataset, architecture, preprocessing, or reported score was reproduced.
      </p>
      <div className="reference-list">
        {list.map((paper) => (
          <article className="reference" key={paper.doi}>
            <div>
              <BookOpen size={17} />
              <span>
                <b>{paper.title}</b>
                <small>{paper.authors}</small>
              </span>
            </div>
            <p>{paper.relation}</p>
            <a href={paper.doi} target="_blank" rel="noreferrer">
              Open DOI ↗
            </a>
          </article>
        ))}
      </div>
    </Card>
  );
}

function ProcessView({ data }: any) {
  const summary = data.pipeline || {};
  return (
    <>
      <div className="research-flow">
        {[
          [
            '1 · Decode & inspect',
            'Read WAV metadata, validate cycle timestamps and check audio integrity.',
          ],
          [
            '2 · Standardize signal',
            'Downmix to mono, resample to 4 kHz and peak-normalize each annotated cycle.',
          ],
          [
            '3 · Prepare model input',
            'For neural models, pad or truncate each cycle to 5 seconds (20,000 samples).',
          ],
          [
            '4 · Transform features',
            'Create a 64-band Log-Mel view; multi-feature models also use MFCC and chroma inputs.',
          ],
        ].map(([title, body]) => (
          <article className="flow-step" key={title}>
            <span>{title}</span>
            <p>{body}</p>
          </article>
        ))}
      </div>
      <div className="grid two">
        <Card title="Saved data-cleaning checks · Notebook 02">
          <Table rows={summary.cleaning_summary || []} />
        </Card>
        <Card title="Input contract and interpretation">
          <ul className="plain-list">
            <li>
              Dataset audio was collected at multiple native sampling rates; processed cycles use a
              4 kHz target.
            </li>
            <li>
              Cycle labels and patient diagnoses are different targets and must not be conflated.
            </li>
            <li>
              The app does not automatically segment a full recording without cycle annotation
              timestamps.
            </li>
            <li>
              Live upload processing details and transformed signal plots appear only after
              prediction on Predict Audio.
            </li>
          </ul>
        </Card>
      </div>
      <Card title="Notebook 04 · Saved preprocessing example">
        <p className="subtle">
          One annotated cycle from the notebook, shown at each saved processing stage. These are
          static notebook figures; uploaded audio is analyzed separately on Predict Audio.
        </p>
        <div className="preprocess-figures">
          {[
            ['audio-preprocessing-1.png', 'Extracted cycle at its source sample rate'],
            ['audio-preprocessing-2.png', 'Same cycle resampled to 4 kHz'],
            ['audio-preprocessing-3.png', '4 kHz cycle after peak normalization'],
          ].map(([src, caption]) => (
            <figure key={src}>
              <img src={`/research/${src}`} alt={caption} />
              <figcaption>{caption}</figcaption>
            </figure>
          ))}
        </div>
      </Card>
      <Sources compact />
    </>
  );
}

function FeatureAnalysis({ data }: any) {
  const pipeline = data.pipeline || {};
  return (
    <>
      <Card title="Notebook 05 · Cycle-level feature engineering">
        <p className="subtle">
          The saved dataset contains 39 handcrafted features per respiratory cycle, spanning
          time-domain statistics, frequency/spectral descriptors, band-energy summaries, MFCCs, and
          wavelet features. These features support the traditional classifiers and later
          feature-analysis notebooks; neural models consume waveform-derived representations.
        </p>
      </Card>
      <div className="grid two">
        <Card title="Feature statistics · Notebook 06">
          <Table rows={(pipeline.feature_statistics || []).slice(0, 16)} />
        </Card>
        <Card title="High-correlation feature pairs">
          <Table rows={(pipeline.high_correlations || []).slice(0, 16)} />
          <p className="subtle">
            Correlation is a redundancy diagnostic; it is not evidence that a feature causes a
            disease.
          </p>
        </Card>
      </div>
      <Card title="Consensus feature ranking · saved scores">
        <Chart
          data={(data.features?.ranking || []).slice(0, 12)}
          labelKey="feature"
          valueKey="consensus_score"
          allowDecimals
          showLabels={false}
        />
      </Card>
      <div className="grid two">
        <Card title="Consensus feature selection · Notebook 07">
          <Table
            rows={(data.features?.ranking || []).slice(0, 15)}
            columns={['feature', 'consensus_score', 'average_rank', 'rf_importance']}
          />
        </Card>
        <Card title="Selected feature set">
          <div className="chips">
            {(data.features?.selected_features || []).map((x: any) => (
              <span key={x.feature || x}>{x.feature || x}</span>
            ))}
          </div>
          <p className="subtle">
            Saved classical-model feature order: 33 features. The common-cohort audit states that
            selection and correlation pruning were fit using training patients only.
          </p>
        </Card>
      </div>
      <Sources compact />
    </>
  );
}

function PCAView({ data }: any) {
  return (
    <>
      <div className="grid two">
        <Card title="PCA explained variance · Notebook 08">
          <Chart
            data={(data.features?.pca || []).slice(0, 10)}
            labelKey="component"
            valueKey="explained_variance_ratio"
            color="#5499b5"
            allowDecimals
            showLabels={false}
          />
          <Table rows={data.features?.pca || []} />
        </Card>
        <Card title="K-means cluster diagnostics">
          <Chart
            data={data.features?.clusters || []}
            labelKey="k"
            valueKey="silhouette_score"
            color="#d6904b"
            allowDecimals
            showLabels={false}
          />
          <Table rows={data.features?.clusters || []} />
          <p className="callout">
            <CircleHelp size={17} /> Unsupervised clusters are not disease labels. The notebook’s
            low disease/sound agreement indicates that the clusters should not be presented as a
            classifier.
          </p>
        </Card>
      </div>
      <Card title="Feature-analysis sequence">
        <p className="subtle">
          The notebooks move from descriptive statistics and correlations (06), to supervised
          feature ranking and redundancy pruning (07), then unsupervised PCA/K-means exploration
          (08). These are saved analyses, not a live retraining workflow.
        </p>
        <Table rows={data.pipeline?.anova_ranking || []} />
      </Card>
    </>
  );
}

function LossChart({ history }: { history: any[] }) {
  if (!history?.length)
    return <div className="empty">No saved training history for this experiment.</div>;
  const normalized = history.map((row) => ({
    ...row,
    val_loss: row.val_loss ?? row.validation_loss,
  }));
  return (
    <div className="chart history-chart">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={normalized}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="epoch" />
          <YAxis />
          <Tooltip />
          <Legend />
          <Line dataKey="train_loss" name="Training loss" stroke="#248f84" dot={false} />
          <Line dataKey="val_loss" name="Validation loss" stroke="#dd8a37" dot={false} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

function Development({ data }: any) {
  const [family, setFamily] = useState('Neural models');
  const dev = data.development || {};
  const expNames: Record<string, string> = {
    lightweight_mel_cnn: 'Lightweight Mel CNN · Notebook 11',
    multifeature_cnn: 'Multi-Feature CNN · Notebook 12',
    regularized_multifeature_cnn_12b: 'Regularized Multi-Feature CNN 12B · Notebook 12 variant',
    cnn_lstm: 'CNN-LSTM · Notebook 13',
  };
  return (
    <>
      <Tabs items={['Classical ML', 'Neural models']} value={family} setValue={setFamily} />
      {family === 'Classical ML' ? (
        <>
          <Card title="Traditional model experiments · Notebook 09">
            <Table rows={dev.classical_metrics || []} />
            <p className="subtle">
              Logistic Regression, SVM and Random Forest use engineered cycle features. Per-disease
              support and imbalance matter alongside overall accuracy.
            </p>
          </Card>
          <Card title="Per-disease results">
            <Table rows={dev.classical_per_disease || []} />
          </Card>
        </>
      ) : (
        <>
          <div className="callout">
            <Layers3 size={17} />
            <span>
              <b>12 and 12B are two experiments in one model family.</b> 12B is a regularized
              variant of the multi-feature CNN, not a second independent “multi-feature family.”
              Both consume Log-Mel, MFCC and Chroma views.
            </span>
          </div>
          <div className="grid two">
            {Object.entries(expNames).map(([key, title]) => {
              const experiment = dev[key] || {};
              return (
                <Card key={key} title={title}>
                  <p className="small-note">
                    Saved metrics · {experiment.metrics?.[0]?.test_samples ?? '—'} cycle test
                    samples
                  </p>
                  <Table rows={experiment.metrics || []} />
                  <LossChart history={experiment.history || []} />
                </Card>
              );
            })}
          </div>
          <Card title="Neural input preparation · Notebook 10">
            <p className="subtle">
              Mono 4 kHz cycle waveform → fixed 5-second input → Log-Mel representation. The
              multi-feature branches add MFCC and chroma. Normalization statistics are fitted on
              training data and then applied to validation/test data.
            </p>
            <div className="feature-branches">
              <span>Waveform</span>
              <b>→</b>
              <span>Log-Mel</span>
              <span>MFCC</span>
              <span>Chroma</span>
              <b>→</b>
              <span>Model / fusion</span>
            </div>
          </Card>
        </>
      )}
      <Sources />
    </>
  );
}

export default function ResearchPages({ section, data }: { section: Section; data: any }) {
  const [tab, setTab] = useState('Signal preprocessing');
  const [evaluationTab, setEvaluationTab] = useState('Model comparison');
  const [artifactTab, setArtifactTab] = useState('Reports');
  if (section === 'features')
    return (
      <>
        <Tabs
          items={['Signal preprocessing', 'Feature analysis', 'PCA & clustering']}
          value={tab}
          setValue={setTab}
        />
        {tab === 'Signal preprocessing' ? (
          <ProcessView data={data} />
        ) : tab === 'Feature analysis' ? (
          <FeatureAnalysis data={data} />
        ) : (
          <PCAView data={data} />
        )}
      </>
    );
  if (section === 'development') return <Development data={data} />;
  if (section === 'evaluation')
    return (
      <>
        <Tabs
          items={['Model comparison', 'Patient analysis', 'Saved XAI examples']}
          value={evaluationTab}
          setValue={setEvaluationTab}
        />
        {evaluationTab === 'Model comparison' ? (
          <Comparison data={data.comparison || {}} />
        ) : evaluationTab === 'Patient analysis' ? (
          <>
            <Patient data={data.patient || {}} />
            <Card title="Related work: patient-level learning">
              <p className="subtle">
                The 2026 multiple-instance learning paper uses its own multi-channel cohort and
                architecture. This app reports aggregation of its saved cycle predictions; it does
                not implement that MIL model.
              </p>
              <a href={papers[4].doi} target="_blank" rel="noreferrer">
                Nguyen et al., IEEE Access 2026 ↗
              </a>
            </Card>
          </>
        ) : (
          <Xai data={data.xai || {}} />
        )}
      </>
    );
  return (
    <>
      <Tabs
        items={['Reports', 'Validation record', 'Research papers']}
        value={artifactTab}
        setValue={setArtifactTab}
      />
      {artifactTab === 'Reports' ? (
        <>
          <Reports data={data.reports || {}} />
          <Card title="Notebook-to-application map · Notebooks 01–18">
            <div className="notebook-map">
              {[
                ['01', 'Dataset understanding', 'Dataset scope, labels and hierarchy'],
                ['02', 'Data cleaning', 'Integrity checks and cleaned records'],
                ['03', 'EDA', 'Cohort, disease and sound-label distributions'],
                ['04', 'Audio preprocessing', 'Cycle extraction, resampling and normalization'],
                ['05', 'Feature engineering', 'Cycle-level handcrafted features'],
                ['06', 'Feature analysis', 'Statistics, distributions and correlations'],
                ['07', 'Feature selection', 'Ranked and selected feature set'],
                ['08', 'PCA and clustering', 'Unsupervised projections and cluster diagnostics'],
                ['09', 'Classical ML', 'Traditional models on engineered features'],
                [
                  '10',
                  'Deep learning data preparation',
                  'Fixed-length waveform and spectral inputs',
                ],
                ['11', 'CNN baseline', 'Lightweight Log-Mel CNN experiment'],
                ['12', 'Multi-feature CNN', 'Multi-branch feature integration and 12B variant'],
                ['13', 'CNN-LSTM', 'Temporal neural experiment'],
                ['14', 'Patient-level prediction', 'Aggregate cycle outputs per patient'],
                ['15', 'XAI', 'Saved Grad-CAM research examples'],
                ['16', 'Final model comparison', 'Compare saved model experiments and protocols'],
                [
                  '17',
                  'Final artifact generation and validation',
                  'Registry, package and final validation artifacts',
                ],
                ['18', 'Model validation', 'Architecture, inference and prediction checks'],
              ].map(([n, t, detail]) => (
                <div key={n}>
                  <b>
                    Notebook {n} · {t}
                  </b>
                  <span>{detail}</span>
                </div>
              ))}
            </div>
          </Card>
        </>
      ) : artifactTab === 'Validation record' ? (
        <>
          <Status data={data.status || {}} />
          <Sources />
        </>
      ) : (
        <Sources />
      )}
    </>
  );
}
