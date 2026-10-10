import { AudioLines, AudioWaveform, CheckCircle2, HeartPulse } from 'lucide-react';
import { Card, Chart, Stat, Table } from '../../components/shared';

export default function Eda({ data }: any) {
  return (
    <>
      <div className="stats">
        <Stat label="Patients" value={data.patient_count} sub="cleaned" icon={HeartPulse} />
        <Stat label="Recordings" value={data.recording_count} sub="cleaned" icon={AudioLines} />
        <Stat
          label="Cycles"
          value={data.cycle_count?.toLocaleString()}
          sub="cleaned"
          icon={AudioWaveform}
        />
        <Stat
          label="Cleaning checks"
          value={data.cleaning_summary?.length || '—'}
          sub="saved summary rows"
          icon={CheckCircle2}
        />
      </div>
      <div className="grid two">
        <Card title="Disease distribution">
          <Chart data={data.disease_distribution} />
        </Card>
        <Card title="Respiratory sound labels">
          <Chart data={data.sound_distribution} labelKey="sound_label" />
        </Card>
      </div>
      <Card title="Data cleaning summary">
        <Table rows={data.cleaning_summary} />
      </Card>
      <p className="subtle">
        Counts come from cleaned artifacts. Source, cleaned, and model-specific filtered counts are
        not interchangeable.
      </p>
    </>
  );
}
