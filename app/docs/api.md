# API reference

Base URL: `http://localhost:8000`; OpenAPI UI: `/docs`.

| Method | Route | Description |
|---|---|---|
| GET | `/api/health` | API health |
| GET | `/api/models` | Model availability and artifact load status |
| GET | `/api/overview` | Cleaned dataset counts, model status, saved highlights |
| GET | `/api/eda/summary` | Saved EDA and cleaning summaries |
| GET | `/api/features/summary` | Selected feature names, rankings, statistics |
| GET | `/api/clustering/summary` | PCA variance and K-means summaries |
| GET | `/api/evaluation/models` | Cycle-level and class-wise saved evaluations |
| GET | `/api/evaluation/cycle-level` | Cycle-level comparison only |
| GET | `/api/evaluation/patient-level` | Patient aggregation metrics and confusion tables |
| GET | `/api/xai/summary` | Saved Grad-CAM metadata and image names |
| GET | `/api/xai/image/{name}` | Safe PNG allowlist within saved XAI directory |
| GET | `/api/reports` | Allowlisted report names and download URLs |
| GET | `/api/reports/download/{name}` | Download allowlisted final report / CSV |
| GET | `/api/system/status` | Current inference readiness and saved validation rows |
| POST | `/api/audio/preview` | Multipart `file`; returns waveform and raw derived Log-Mel/MFCC/Chroma arrays for preview (not training-normalized) |
| POST | `/api/predict` | Multipart `file` plus one `model_id` |
| POST | `/api/predict/compare` | Multipart `file` plus JSON-array string `model_ids` |

Prediction endpoints accept WAV files up to 20 MiB and 120 seconds by default; the backend variables `MAX_UPLOAD_BYTES` and `MAX_AUDIO_DURATION_SECONDS` can change those limits. A prediction response includes duration and preprocessing details, a top-level `success`, `partial_failure`, or `failure` status, model result rows, per-model failures, partial-success state, and a research-use disclaimer. Only selected traditional estimators are executed, and their shared features are computed once per upload. Neural inputs use the first five seconds (zero-padded when shorter); traditional models use the full resampled recording. Each score map is keyed by class. They are not presented as calibrated confidence. Invalid audio returns a structured 4xx response. Arbitrary filesystem paths are never accepted.
