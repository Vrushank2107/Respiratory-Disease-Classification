# API reference

Base URL: `http://localhost:8000`; OpenAPI UI: `/docs`.

| Method | Route | Description |
|---|---|---|
| GET | `/api/health` | API health |
| GET | `/api/models` | Runtime model availability, artifact version/size, and input contract; fetched when Predict Audio opens |
| GET | `/api/overview` | Cleaned dataset counts, static saved model registry, saved highlights |
| GET | `/api/eda/summary` | Saved EDA and cleaning summaries |
| GET | `/api/features/summary` | Selected feature names, rankings, statistics |
| GET | `/api/pipeline/summary` | Saved preprocessing checks, feature statistics, correlations and feature-ranking inputs |
| GET | `/api/development/summary` | Saved classical metrics, neural metrics and training histories; does not load checkpoints |
| GET | `/api/clustering/summary` | PCA variance and K-means summaries |
| GET | `/api/evaluation/models` | Legacy saved evaluations and the common patient-held-out seven-model comparison |
| GET | `/api/evaluation/cycle-level` | Cycle-level comparison only |
| GET | `/api/evaluation/patient-level` | Patient aggregation metrics and confusion tables |
| GET | `/api/xai/summary` | Saved Grad-CAM metadata and image names |
| GET | `/api/xai/image/{name}` | Safe PNG allowlist within saved XAI directory |
| GET | `/api/reports` | Allowlisted report names and download URLs |
| GET | `/api/reports/download/{name}` | Download allowlisted final and common-split audit reports |
| GET | `/api/system/status` | Static model artifact registry and saved validation rows; does not load checkpoints |
| POST | `/api/audio/preview` | Multipart `file`; returns waveform and raw derived Log-Mel/MFCC/Chroma arrays for preview (not training-normalized) |
| POST | `/api/predict` | Multipart `file`, one `model_id`, and optional cycle `annotations` `.txt`; unannotated WAVs use automatic windows |
| POST | `/api/predict/compare` | Multipart `file`, JSON-array string `model_ids`, and optional cycle `annotations` `.txt`; unannotated WAVs use automatic windows |

Prediction endpoints accept WAV files up to 20 MiB and 120 seconds by default; the backend variables `MAX_UPLOAD_BYTES` and `MAX_AUDIO_DURATION_SECONDS` can change those limits. A matching ICBHI-style `.txt` annotation file is optional; when provided, the API validates its start/end timestamps and extracts the annotated cycles. Any readable WAV without annotations is processed in consecutive five-second windows. These are fixed windows, not detected respiratory cycles. Disease models receive mono, 4 kHz, peak-normalized inputs padded or truncated to five seconds. The selectable Experiment B sound-phenotype model uses the same mono/resampling stage but preserves amplitude and center-crops or symmetrically zero-pads to three seconds, exactly as specified by its saved configuration. It outputs Normal, Crackles, Wheezes, or Both; it does not predict disease diagnoses. Recording-level scores are the unweighted mean of segment scores and are not calibrated probabilities. Responses include a live audio-analysis preview, preprocessing steps, and per-segment predictions. Model-specific explanations are available for the existing disease models; Experiment B does not advertise a live explanation. Inference time and explanation time are reported separately. Invalid audio returns a structured 4xx response. Arbitrary filesystem paths are never accepted.
