export interface ModelInfo {
  model_id: string;
  display_name: string;
  family: 'traditional_ml' | 'deep_learning';
  available: boolean;
  reason?: string | null;
}

export interface PredictionPayload {
  status: 'success' | 'partial_failure' | 'failure';
  filename: string;
  source_sample_rate: number;
  target_sample_rate: number;
  channels: number;
  input_duration_seconds: number;
  neural_input_duration_seconds: number;
  handling: string;
  results: Array<Record<string, any>>;
  failures: Array<{ model_id: string; error: string }>;
  partial_success: boolean;
  disclaimer: string;
}
