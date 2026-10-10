import { Activity } from 'lucide-react';
import { Card, Table } from '../../components/shared';

export default function Patient({ data }: any) {
  const d = data.aggregation_diagnostics;
  return (
    <>
      <Card title="Patient-level aggregation">
        <Table rows={data.metrics || []} />
      </Card>
      {d?.predictions_identical && (
        <p className="callout">
          <Activity size={17} />
          <span>
            <b>Why the rows match:</b> majority vote and mean probability selected the same class
            for all {d.test_patients} saved test patients ({d.matching_predicted_classes}/
            {d.test_patients}). That gives identical metrics and confusion matrices. The methods
            aggregate cycle outputs differently; they just agree on the winning class in this test
            set.
          </span>
        </p>
      )}
      <div className="grid matrix-grid">
        <Card title="Majority vote confusion matrix · patients">
          <Table
            rows={data.majority_confusion || []}
            columns={[
              'true_class',
              'Asthma',
              'Bronchiectasis',
              'Bronchiolitis',
              'COPD',
              'Healthy',
              'LRTI',
              'Pneumonia',
              'URTI',
            ]}
          />
        </Card>
        <Card title="Mean probability confusion matrix · patients">
          <Table
            rows={data.mean_confusion || []}
            columns={[
              'true_class',
              'Asthma',
              'Bronchiectasis',
              'Bronchiolitis',
              'COPD',
              'Healthy',
              'LRTI',
              'Pneumonia',
              'URTI',
            ]}
          />
        </Card>
      </div>
      <p className="subtle">
        These saved patient-level results cover 25 test patients and remain separate from
        cycle-level evaluation.
      </p>
    </>
  );
}
