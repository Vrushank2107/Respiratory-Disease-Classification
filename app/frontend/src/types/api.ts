export interface ModelInfo {
  model_id: string;
  display_name: string;
  family: 'traditional_ml' | 'deep_learning';
  available: boolean;
  artifact_version?: string | null;
  artifact_size_bytes?: number | null;
  input_contract?: string;
  reason?: string | null;
}

export interface PredictionPayload {
  status: 'success' | 'partial_failure' | 'failure';
  filename: string;
  source_sample_rate: number;
  target_sample_rate: number;
  channels: number;
  input_duration_seconds: number;
  input_mode: 'annotated_recording' | 'single_cycle' | 'automatic_windows';
  cycle_count: number;
  segment_count: number;
  neural_input_duration_seconds: number;
  handling: string;
  warnings: string[];
  results: Array<Record<string, any>>;
  failures: Array<{ model_id: string; error: string }>;
  partial_success: boolean;
  disclaimer: string;
}
