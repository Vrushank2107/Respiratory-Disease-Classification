# Deploy the frontend on Vercel

Deploy only the static Vite frontend to Vercel. The FastAPI backend must run
on a separate Python host because Vercel reports that the backend deployment
bundle is about 5 GB, above the 2 GB limit available to this project.

## Vercel frontend

1. Import the GitHub repository into Vercel.
2. Set **Root Directory** to `app/frontend` and choose **Vite** as the preset.
3. The included `vercel.json` builds with `npm run build` and serves `dist`.
4. Add `VITE_API_URL` for Production and Preview in Vercel Project Settings.
   Set it to the public backend origin, without a trailing slash or `/api`,
   for example `https://your-api.example.com`.
5. Redeploy after changing the variable; Vite embeds it at build time.

The browser sends API requests to `VITE_API_URL + /api/...`. For local
development, leave `VITE_API_URL` unset to use the Vite proxy to
`http://127.0.0.1:8000`.

## Separate FastAPI backend

Choose a Python/container host whose deployment artifact and memory limits can
accommodate the backend's actual package. Configure its build from the
repository and install runtime dependencies with:

```sh
pip install -r app/backend/requirements.txt
```

Run the API from the repository root so the Python package imports and
repository-relative model/data paths resolve:

```sh
uvicorn app.backend.main:app --host 0.0.0.0 --port "$PORT"
```

The backend needs the model checkpoints and the processed data/report files
under `models/`, `data/processed/`, and `reports/final_artifacts/`. The raw
source audio is not needed by the inference API. Confirm the hosting plan's
artifact-size, RAM, disk, execution-time, and upload limits before deploying.

Set backend environment variables:

```text
FRONTEND_ORIGINS=https://your-project.vercel.app
MAX_UPLOAD_BYTES=20971520
MAX_AUDIO_DURATION_SECONDS=120
```

Add exact Vercel Preview origins to `FRONTEND_ORIGINS` if those deployments
should call the API. Multiple allowed origins are comma-separated. No wildcard
is needed.

## Check the deployment

1. Open `https://<backend-host>/api/health` and confirm it returns healthy.
2. Open `/api/models` and confirm model artifacts load successfully.
3. Confirm the Vercel site shows the API as connected.
4. Upload a representative WAV and run inference; the backend accepts WAVs up
   to the configured upload and duration limits.
5. Check backend logs if model loading or artifact paths fail.
