import { ShieldAlert } from 'lucide-react';
import { Card, MetricTable, Table, titleCase } from '../../components/shared';

export default function Comparison({ data }: any) {
  const common = data.common_patient_split || {};
  return (
    <>
      <Card
        title="Common patient-held-out evaluation · cycle level"
        aside={<span className="card-note">82 training patients · 25 held-out test patients</span>}
      >
        <Table
          rows={common.cycle_level || []}
          columns={[
            'model_id',
            'training_patients',
            'test_patients',
            'test_recordings',
            'test_cycle_support',
            'accuracy',
            'balanced_accuracy',
            'macro_f1',
            'weighted_f1',
            'deployed_artifact_bytes',
            'estimator_predict_ms_per_cycle',
          ]}
        />
        <p className="small-note">
          Classical estimator latency is measured on the audit machine and excludes audio decoding
          and feature extraction. Uploaded-audio inference time appears on the Predict page.
        </p>
      </Card>
      <Card title="Common cohort · per-disease cycle metrics">
        <Table
          rows={common.cycle_per_class || []}
          columns={[
            'model_id',
            'disease',
            'precision',
            'recall',
            'f1',
            'support_cycles',
            'support_patients',
            'support_recordings',
            'metric_status',
          ]}
        />
      </Card>
      <Card title="Common cohort · patient-level majority vote">
        <Table
          rows={common.patient_level || []}
          columns={[
            'model_id',
            'test_patients',
            'accuracy',
            'balanced_accuracy',
            'macro_f1',
            'weighted_f1',
          ]}
        />
        <p className="small-note">
          Each patient's cycle predictions are reduced to one majority-vote class. These are
          patient-level outcomes; cycle-level metrics above weight patients with more recorded
          cycles more heavily.
        </p>
      </Card>
      <Card title="Common cohort · per-disease patient metrics">
        <Table
          rows={common.patient_per_class || []}
          columns={[
            'model_id',
            'disease',
            'precision',
            'recall',
            'f1',
            'support_patients',
            'support_recordings',
            'metric_status',
          ]}
        />
      </Card>
      <Card title="Common cohort · confusion matrices">
        <div className="grid two">
          {Object.entries(common.confusion || {}).map(([name, rows]: any) => (
            <details className="score-details" key={name}>
              <summary>{titleCase(name)}</summary>
              <Table rows={rows} />
            </details>
          ))}
        </div>
      </Card>
      <Card
        title="Cycle-level evaluation"
        aside={
          <span className="card-note">Saved metrics · test split comparability not confirmed</span>
        }
      >
        <MetricTable rows={data.cycle_level || []} />
      </Card>
      <Card title="Per-disease classical ML results">
        <Table rows={data.classical || []} />
      </Card>
      <div className="grid two">
        {Object.entries(data.confusion || {}).map(([name, rows]: any) => (
          <Card
            key={name}
            title={`${titleCase(name)} · ${name === 'cnn_baseline' ? 'matrix from saved test predictions' : 'saved confusion matrix'}`}
          >
            <Table rows={rows} />
          </Card>
        ))}
      </div>
      <p className="callout">
        <ShieldAlert size={17} /> The common-cohort tables use one patient-held-out split. Some
        diseases have no test patients, and cycle-level rows from one patient are correlated. The
        older saved tables below use their original protocols. Scores are not calibrated confidence.
      </p>
    </>
  );
}
