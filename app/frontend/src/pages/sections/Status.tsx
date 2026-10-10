import { Card, Table } from '../../components/shared';
type Obj = Record<string, any>;

export default function Status({ data }: any) {
  return (
    <>
      <Card title="Saved model artifact registry">
        <div className="model-list">
          {(data.artifact_registry || []).map((m: Obj) => (
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
      <Card title="Saved project validation records · Notebooks 17–18">
        <Table
          rows={data.validation_records || []}
          columns={['category', 'item', 'path', 'requirement', 'status', 'details']}
        />
      </Card>
      <p className="subtle">
        These are saved checks from the project validation artifacts, not a live runtime health
        test. A file/checkpoint check or successful inference is not scientific or clinical
        validation.
      </p>
    </>
  );
}
