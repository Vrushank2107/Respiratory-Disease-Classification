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
      <div className="grid two">
        <Card title="Disease × respiratory sound · Notebook 03">
          <Table rows={data.disease_sound_counts || []} />
          <p className="subtle">
            Cycle sound labels (Normal, Crackles, Wheezes, Both) are not the same label as the
            patient-level disease diagnosis.
          </p>
        </Card>
        <Card title="Patient recording summary">
          <Table rows={(data.patient_audio_summary || []).slice(0, 20)} />
        </Card>
      </div>
      <Card title="Data cleaning summary">
        <Table rows={data.cleaning_summary} />
      </Card>
      <p className="subtle">
        Notebooks 01–03 document dataset structure, cleaning and exploratory analysis. Counts come
        from saved cleaned artifacts. Source, cleaned, cycle-level and model-specific filtered
        counts are not interchangeable; patient diagnosis and cycle sound labels are separate.
      </p>
    </>
  );
}
