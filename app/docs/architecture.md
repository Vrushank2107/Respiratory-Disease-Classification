# Architecture

The Vite/React/TypeScript frontend talks to a local FastAPI backend. Model loading, feature extraction, audio resampling, scoring, and access to saved artifacts occur only in Python. The backend resolves all data and model paths from the repository root derived from its own module path.

```text
Browser (localhost:5173)
  └── documented JSON / multipart HTTP API (localhost:8000)
       ├── research artifact readers (CSV / JSON / allowlisted report downloads)
       ├── audio decode → channel mean downmix → soxr resample to 4 kHz
       ├── neural adapters (strict state-dict loading, cached models)
       └── saved scikit-learn estimators + exact 33-feature order
```

Uploads are limited to 20 MiB and 120 seconds by default (both configurable), must have a `.wav` filename, and must decode to finite, non-silent audio. Temporary files are deleted in a `finally` block. Neural inputs are padded or truncated to five seconds; classical models operate on the entire resampled waveform; selected estimators share one feature extraction pass. Multichannel downmix uses arithmetic channel mean. The user-facing interface separates saved historical evaluation from uploaded-audio inference and keeps patient-level results separate.

Models are loaded lazily and cached. A model is listed ready only after artifacts load and strict checkpoint compatibility succeeds. Scores are labeled as softmax or estimator outputs; calibration has not been established.

Notebook source is treated as read-only. Uploaded audio is neither sent to third-party services nor retained by the app.
