import { API } from '../../services/api';
import { Download, FileText } from 'lucide-react';
import { Card } from '../../components/shared';
type Obj = Record<string, any>;

export default function Reports({ data }: any) {
  return (
    <Card title="Research files">
      <div className="report-list">
        {(data.reports || []).map((r: Obj) => (
          <a href={API + r.url} key={r.name} className="report">
            <FileText size={19} />
            <span>
              <b>{r.name}</b>
              <small>Verified project report or CSV artifact</small>
            </span>
            <Download size={16} />
          </a>
        ))}
      </div>
    </Card>
  );
}
