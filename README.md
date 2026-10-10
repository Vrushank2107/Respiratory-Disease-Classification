# Respiratory Disease Classification

An academic research project for analyzing respiratory sounds from the ICBHI 2017 lung-sound dataset. It includes the data-preparation and machine-learning notebooks, saved model artifacts and evaluation results, and a web dashboard for exploring the research outputs and predicting from uploaded WAV recordings.

> **Research use only.** Model scores are experimental, are not established as calibrated confidence, and must not be used for medical diagnosis or treatment decisions.

## Live application

- **Dashboard:** [respiratory-disease-classification-seven.vercel.app](https://respiratory-disease-classification-seven.vercel.app/)
- **FastAPI backend:** [respiratory-disease-classification.onrender.com](https://respiratory-disease-classification.onrender.com/)
- **API health:** [/api/health](https://respiratory-disease-classification.onrender.com/api/health)
- **API documentation:** [/docs](https://respiratory-disease-classification.onrender.com/docs)

The dashboard is hosted on Vercel and uses the separately hosted Render API. The API root may return `404`; use `/api/health`, `/api/models`, or `/docs` to check it. Render free services have resource and availability limits, so inference performance may vary.

For the full project history and technical walkthrough, see [the detailed project documentation](docs/PROJECT_DOCUMENTATION.md). It covers notebooks 01–18, data and model artifacts, evaluation results, application pages and API, prediction flow, and deployment.

## What is included

- **Research workflow:** 18 Jupyter notebooks cover dataset understanding, cleaning, EDA, audio processing, feature work, classical and deep learning, patient-level aggregation, explainability, and validation.
- **Saved models:** Logistic Regression, SVM, Random Forest, Lightweight Mel CNN, Multi-Feature CNN, Regularized Multi-Feature CNN (12B), and CNN-LSTM.
- **Dashboard:** dataset and EDA views, audio pipeline and feature summaries, model comparison, patient analysis, saved explainability examples, reports, system status, and WAV prediction.
- **API:** FastAPI reads the saved research artifacts and runs inference; prediction uploads are processed in memory and are not retained.

The cleaned dataset contains **126 patients, 920 WAV recordings, and 6,898 respiratory cycles**. The current dashboard recording card mistakenly counts 34 distinct short recording codes instead of unique WAV files; the detailed documentation explains this known API summary issue. Class counts and evaluation outputs are available in the app and under `data/processed/` and `reports/final_artifacts/`.

## Repository layout

```text
.
├── app/
│   ├── backend/              # FastAPI API and inference services
│   ├── frontend/             # React, TypeScript, and Vite dashboard
│   ├── docs/                 # API, architecture, artifacts, and deployment notes
│   └── artifacts/            # Dashboard artifact documentation
├── data/
│   ├── interim/              # Parsed intermediate tables
│   └── processed/            # Cleaned data, features, predictions, and metrics
├── models/
│   ├── deep_learning/        # PyTorch checkpoints and normalization data
│   └── traditional_ml/       # scikit-learn estimators and metadata
├── notebooks/                # Ordered research and validation notebooks
├── reports/
│   └── final_artifacts/      # Final comparisons, registries, and validation reports
├── Dockerfile                # CPU-only backend container for Render / local Docker
├── docker-compose.yml        # Local backend service
├── requirements.txt          # Full research/notebook environment
└── app/backend/requirements.txt # Backend dependencies
```

The raw ICBHI audio dataset and generated audio-cycle files are excluded from Git because of their size. The saved models, processed tables, and reports used by the dashboard are included. Obtain the raw data from the [ICBHI 2017 Challenge](https://bhichallenge.med.auth.gr/) if you want to rerun the original data-preparation workflow.

## Run the dashboard locally

### Backend with Docker

Install and start Docker Desktop, then run this from the repository root:

```bash
docker compose up --build
```

This starts the API on `http://localhost:8000`. API docs are at `http://localhost:8000/docs`. Stop it with `Ctrl+C` or run `docker compose down` in another terminal.

### Frontend

In a second terminal:

```bash
cd app/frontend
npm ci
npm run dev
```

Open `http://localhost:5173`. During local development, Vite proxies `/api` requests to the local backend. Production builds use the Render API URL in `app/frontend/.env.production`; a `VITE_API_URL` configured in Vercel overrides that default.

### Run the backend directly with Python (optional)

From the repository root, create a virtual environment and install the backend dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r app/backend/requirements.txt
uvicorn app.backend.main:app --reload --host 127.0.0.1 --port 8000
```

Then start the frontend using the commands above.

## Run the research notebooks

The root `requirements.txt` contains the broader research environment, including notebook and analysis dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
jupyter lab
```

Open the notebooks in `notebooks/` and follow their numbered order. Re-running the original data-preparation and training stages requires the raw ICBHI data. The dashboard itself uses the checked-in saved models and results; it does not retrain models.

## Models and interpretation

Traditional machine-learning inference uses the saved scaler, label encoder, selected-feature list, and estimators. Neural inference uses the saved PyTorch checkpoints and training normalizations. The API lists a model as available only when its required artifacts load successfully.

The saved evaluation results are subject to class imbalance and use protocols that may differ between experiments. Check the split-comparability notes in `reports/final_artifacts/final_results_report.md` before comparing model scores. Probability-like outputs are not calibrated confidence estimates.

## Deployment

- **Vercel:** deploy the Vite frontend with the project root set to `app/frontend`. Its production API URL is in `.env.production`.
- **Render:** deploy the FastAPI backend as a Docker web service using the repository root `Dockerfile`; leave Root Directory blank. Set `FRONTEND_ORIGINS` if additional frontend origins need access.
- **Local container:** `docker compose up --build` runs the backend API.

See [Vercel deployment notes](app/docs/vercel-deployment.md), [Cloud Run alternative](app/docs/cloud-run-deployment.md), [API reference](app/docs/api.md), and [architecture](app/docs/architecture.md).

## API routes

The API provides health and model status, overview and research summaries, saved evaluations and report downloads, audio preview, and prediction for one or multiple models. See the live [OpenAPI documentation](https://respiratory-disease-classification.onrender.com/docs) or [API reference](app/docs/api.md) for routes and request formats.

WAV uploads default to a maximum of 20 MiB and 120 seconds. The backend accepts `MAX_UPLOAD_BYTES` and `MAX_AUDIO_DURATION_SECONDS` environment variables to change those limits.

## Technology

- **Research:** Python, Jupyter, pandas, NumPy, SciPy, scikit-learn, PyTorch, librosa, PyWavelets
- **Backend:** FastAPI, Uvicorn, soundfile, soxr
- **Frontend:** React, TypeScript, Vite, Recharts, Lucide
- **Deployment:** Docker, Render, Vercel
