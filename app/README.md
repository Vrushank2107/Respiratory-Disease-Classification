# RespiraLab Research Dashboard

The web application provides a research interface to saved respiratory-sound results and runs inference on uploaded WAV recordings. It consists of a React/Vite frontend and a FastAPI backend. The app does not retrain the research models, and predictions are not medical diagnoses.

## Live deployment

- **Frontend:** [RespiraLab on Vercel](https://respiratory-disease-classification-seven.vercel.app/)
- **Backend API:** [Render service](https://respiratory-disease-classification.onrender.com/)
- **Health check:** [API health](https://respiratory-disease-classification.onrender.com/api/health)
- **API docs:** [OpenAPI / Swagger UI](https://respiratory-disease-classification.onrender.com/docs)

The backend URL serves the API, so its `/` route may return `404`. Use `/api/health` to check availability. Render free services have resource and availability limits, so a service may be slow after inactivity or unable to handle some workloads.

## Run locally

### Start the API

With Docker Desktop running, start the backend from the repository root:

```bash
docker compose up --build
```

The API is at `http://localhost:8000`; the interactive API docs are at `http://localhost:8000/docs`.

### Start the frontend

In another terminal:

```bash
cd app/frontend
npm ci
npm run dev
```

Open `http://localhost:5173`. Vite proxies `/api` requests to the local API. Stop the backend with `Ctrl+C` or `docker compose down`.

To run the API directly with Python instead of Docker, use the repository root as the working directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r app/backend/requirements.txt
uvicorn app.backend.main:app --reload --host 127.0.0.1 --port 8000
```

## Format the application code

Frontend formatting uses Prettier:

```bash
cd app/frontend
npm run format
```

Check frontend formatting without changing files with `npm run format:check`. Backend formatting uses Ruff; install it with `python -m pip install ruff`, then run `ruff format app/backend` from the repository root.

## Deployment

The frontend is deployed on Vercel from `app/frontend`; the production API origin is configured in `app/frontend/.env.production`. The FastAPI backend is deployed on Render as a Docker web service from the repository root `Dockerfile`.

For another deployment, set `VITE_API_URL` to the backend origin in the frontend build environment and allow that exact frontend origin through the backend's `FRONTEND_ORIGINS` setting. See [Vercel deployment notes](docs/vercel-deployment.md) and the [Cloud Run alternative](docs/cloud-run-deployment.md).

## What the app shows

- **Overview** and cleaned dataset context
- **Dataset & EDA** distributions, patient summaries, and cleaning records from Notebooks 01–03
- **Audio & features** saved preprocessing examples, feature analyses, selection and PCA/clustering from Notebooks 04–08
- **Model development** classical and neural training histories/metrics from Notebooks 09–13
- **Evaluation & XAI** cycle/patient results and saved Grad-CAM examples from Notebooks 14–16
- **Artifacts & validation** downloadable reports, saved validation records, the full Notebook 01–18 map, and related research papers
- **Predict audio** WAV inference and input-specific processing/explanations

The seven pages are **Overview**, **Dataset & EDA**, **Audio & features**, **Model development**, **Evaluation & XAI**, **Artifacts & validation**, and **Predict audio**. The first six are informational and read saved project artifacts; only Predict audio loads models and runs live inference. Model readiness is checked when Predict audio opens. Overview and Status show a lightweight seven-model artifact inventory without loading model weights. Paper references are presented as related work with notes on differences; paper-reported metrics are not presented as project results. See [research paper provenance](docs/research_provenance.md).

The backend loads saved estimators, neural checkpoints, normalizations, and research artifacts from the repository. Raw source audio and preprocessed cycle audio are not required for inference. Uploaded recordings are validated, decoded, downmixed as needed, resampled, and processed temporarily; the API does not retain uploads.

Uploads must be readable WAV files, up to 20 MiB and 120 seconds by default. A matching ICBHI-style timestamp `.txt` file is optional. With annotations, the backend extracts those cycle intervals. Without annotations, it scores the full recording in consecutive five-second windows, including a possibly shorter final window; these are not detected respiratory cycles. The models were trained on respiratory cycles, so scores for arbitrary WAV recordings are exploratory and may not generalize. Configure `MAX_UPLOAD_BYTES` or `MAX_AUDIO_DURATION_SECONDS` to change the limits.

After prediction, the page displays source and prepared-input waveforms in separate panels, live Log-Mel/MFCC/Chroma views, preprocessing steps, all segment predictions, and model-specific explanations for the first segment only. Recording scores are the unweighted mean of segment scores. The highest displayed score is an uncalibrated model output, not a probability of correctness. Uploaded audio is not retained; temporary backend files are deleted after processing.

## Development and references

```bash
# Frontend production build and TypeScript check
cd app/frontend && npm run build

# Backend and artifact tests
python -m pytest app/backend/tests tests/test_final_artifacts.py
```

See [API reference](docs/api.md), [architecture](docs/architecture.md), and [artifact mapping](docs/artifact_mapping.md) for implementation and provenance details.
