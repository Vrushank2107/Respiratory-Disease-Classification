# Architecture

The Vite/React/TypeScript frontend talks to a local FastAPI backend. Model loading, feature extraction, audio resampling, scoring, and access to saved artifacts occur only in Python. The backend resolves all data and model paths from the repository root derived from its own module path.

The seven top-level frontend sections consolidate the former separate Features, Comparison, Patient, XAI, Reports, and Status pages. Dataset, preprocessing, development, evaluation, and artifact sections are informational: they read saved CSV/JSON/notebook figures and do not load model weights or run inference. The overview and status pages use the saved model registry/validation records. The frontend requests `/api/models` only when Predict Audio opens; that endpoint performs runtime artifact loading checks. A single uploaded file can therefore trigger live prediction and input-specific analysis only in Predict Audio.

```text
Browser (localhost:5173)
  └── documented JSON / multipart HTTP API (localhost:8000)
       ├── research artifact readers (CSV / JSON / allowlisted report downloads)
       ├── audio decode → channel mean downmix → soxr resample to 4 kHz
       ├── neural adapters (strict state-dict loading, cached models)
       └── saved scikit-learn estimators + exact 33-feature order
```

Uploads are limited to 20 MiB and 120 seconds by default (both configurable), must have a `.wav` filename, and must decode to finite audio. Temporary files are deleted in a `finally` block. A full recording may include an optional ICBHI-style `.txt` annotation file; validated timestamps select the annotated intervals. Without annotations, the full WAV is covered by consecutive five-second windows, including a possibly shorter final window. Fixed windows are not respiratory-cycle detection. Audio is downmixed to mono and resampled to 4 kHz. Existing disease models receive peak-normalized inputs padded or truncated to five seconds; Experiment B preserves amplitude and center-crops or symmetrically pads to three seconds. Classical models use peak-normalized full segments and share one feature-extraction pass. Recording scores are the unweighted mean of segment scores. The user-facing interface separates saved historical evaluation from uploaded-audio inference and keeps patient-level results separate.

Models are loaded lazily and cached. A model is listed ready only after artifacts load and strict checkpoint compatibility succeeds. Scores are labeled as softmax or estimator outputs; calibration has not been established.

The live prediction response also includes a downsampled source waveform, the first prepared segment as a neural-input reference, and raw Log-Mel/MFCC/Chroma views. Existing disease model results carry an explanation for the first segment: normalized input-gradient saliency for neural models and single-feature baseline ablation for traditional models. Experiment B currently has no live explanation. All segments are scored, while limiting explanations to one representative segment keeps long uploads responsive. Explanation runtime is separate from estimator/network inference runtime. Saliency maps and feature effects are diagnostic views of model behavior, not causal or clinical evidence.

Notebook source is treated as read-only. Uploaded audio is neither sent to third-party services nor retained by the app.
