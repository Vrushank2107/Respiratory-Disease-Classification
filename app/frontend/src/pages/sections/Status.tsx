import { Card, Table } from '../../components/shared';
type Obj = Record<string, any>;

export default function Status({ data }: any) {
  return (
    <>
      <Card title="Model artifact and load status">
        <div className="model-list">
          {(data.models || []).map((m: Obj) => (
            <div className="model-row" key={m.model_id}>
              <span className={`status-dot ${m.available ? 'good' : 'bad'}`} />
              <span>{m.display_name}</span>
              <small>{m.artifact_validation}</small>
              <b className={m.available ? 'text-good' : 'text-bad'}>
                {m.available ? 'Inference ready' : m.reason}
              </b>
            </div>
          ))}
        </div>
      </Card>
      <Card title="Current runtime checks">
        <Table
          rows={data.validation_records || []}
          columns={['category', 'item', 'requirement', 'status', 'details']}
        />
      </Card>
      <p className="subtle">
        A software artifact check or successful inference is not scientific or clinical validation.
      </p>
    </>
  );
}
