import { API } from '../../services/api';
import { BrainCircuit, ShieldAlert } from 'lucide-react';
import { Card } from '../../components/shared';
type Obj = Record<string, any>;

export default function Xai({ data }: any) {
  return (
    <>
      <Card title="Grad-CAM saved examples">
        <p className="subtle">
          Notebook 15 implements Grad-CAM for the Lightweight Mel CNN. These are precomputed
          research examples, not explanations generated for the current upload. Grad-CAM explains
          influential regions; it does not calculate or validate a confidence score.
        </p>
        <div className="xai-grid">
          {(data.images || []).map((name: string) => (
            <div className="xai-card" key={name}>
              <img src={`${API}/api/xai/image/${encodeURIComponent(name)}`} alt={name} />
              <b>{name.replace('gradcam_', '').replace('.png', '').replaceAll('_', ' ')}</b>
            </div>
          ))}
        </div>
      </Card>
      <Card title="Example metadata">
        <pre>{JSON.stringify(data.metadata || {}, null, 2)}</pre>
      </Card>
      <p className="callout">
        <BrainCircuit size={17} /> Grad-CAM highlights activation regions associated with a model
        output. It does not establish causal evidence or medical correctness.
      </p>
    </>
  );
}
