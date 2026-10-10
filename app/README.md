# Respiratory Sound Research Platform

A locally run academic dashboard built around existing lung-sound research results and saved models. It does not retrain models and does not modify notebooks 01–18, datasets, or checkpoints. Model predictions are not diagnoses.

## Start on macOS

From the repository root:

```bash
# Terminal 1: existing project environment
source .venv/bin/activate
python -m pip install -r app/backend/requirements.txt
uvicorn app.backend.main:app --reload --host 127.0.0.1 --port 8000

# Terminal 2
cd app/frontend
npm install
npm run dev
```

Open <http://localhost:5173>. Backend API and OpenAPI docs: <http://localhost:8000> and <http://localhost:8000/docs>.

Dependency installation is local to the project venv for Python. The frontend uses the package manifest in `app/frontend`. No package installation is run automatically by the app.

## Validation

```bash
source .venv/bin/activate
python -m pytest app/backend/tests tests/test_final_artifacts.py
cd app/frontend && npm run build
```

## Implemented research methodology

Neural models use saved architectures/checkpoints and training normalizations. Log-Mel settings match Notebook 11; multi-feature settings match Notebook 12 (Log-Mel, MFCC, Chroma); CNN-LSTM follows Notebook 13. Traditional models use the saved 33-feature list and classifier-specific scaler usage in metadata. Historical metrics are loaded from saved outputs, and evaluation units are kept separate. Notebook 15 Grad-CAM output is represented by saved examples only.

See `docs/artifact_mapping.md`, `docs/architecture.md`, and `docs/api.md` for artifact provenance, caveats, and API details.

## Troubleshooting

- If API reports a missing module, activate `.venv` and install `app/backend/requirements.txt`.
- If a model is unavailable, inspect its exact artifact-specific reason in **System status** and compare with `docs/artifact_mapping.md`.
- Upload non-silent WAV files under 20 MiB and 120 seconds by default. Set `MAX_UPLOAD_BYTES` or `MAX_AUDIO_DURATION_SECONDS` in the backend environment to adjust limits. Input is decoded locally, downmixed, resampled where needed, and not persisted. Frontend dependency versions are pinned in `app/frontend/package.json` and its lockfile.
- The Vite development server proxies `/api` to `127.0.0.1:8000`, so localhost and private-LAN browser origins work without cross-origin fetches. For a separately hosted API, set `VITE_API_URL` in `app/frontend/.env.local`; backend limit examples are in `app/backend/.env.example`.
