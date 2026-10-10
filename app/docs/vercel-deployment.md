# Deploying the RespiraLab app

## Recommended deployment shape

Deploy `app/frontend` as the Vercel project and host the FastAPI backend on a
Python service that supports the repository's model dependencies and audio
uploads. The frontend sends requests directly to that backend using
`VITE_API_URL`.

This split matters for this repository: the app accepts WAV uploads up to 20
MiB, while Vercel Functions currently cap each request and response at 4.5 MB.
The checked-in research data is also multi-gigabyte, and the inference service
loads Python/scientific/ML dependencies and saved model artifacts. Those are a
poor fit for a frontend deployment and a serverless function bundle.

## Vercel frontend

1. Import the repository into Vercel.
2. Set **Root Directory** to `app/frontend`.
3. Use the Vite framework preset. The included `vercel.json` sets the build
   command to `npm run build` and output directory to `dist`.
4. Add `VITE_API_URL` in Vercel Project Settings for Production and Preview.
   Set it to the public backend origin, for example
   `https://respiralab-api.example.com` (no trailing slash and no `/api`).
   Vite embeds this value at build time, so redeploy after changing it.
5. Deploy. The frontend can be visited at the Vercel URL, but API-backed
   screens need the backend to be online and CORS-configured.

## FastAPI backend

Use a Python host that can build dependencies from the repository and persist
or include the required artifacts. Configure its working directory so the
repository root is importable, install dependencies with
`pip install -r app/backend/requirements.txt`, and start the service with an
ASGI command similar to:

```sh
uvicorn app.backend.main:app --host 0.0.0.0 --port "$PORT"
```

Set these backend environment variables:

```text
FRONTEND_ORIGINS=https://your-project.vercel.app
MAX_UPLOAD_BYTES=20971520
MAX_AUDIO_DURATION_SECONDS=120
```

For Vercel Preview deployments, add the exact preview origins you intend to
use as comma-separated entries in `FRONTEND_ORIGINS`. The API also keeps the
existing localhost and local-network development origins. Do not set a
wildcard origin when credentials or uploaded audio are in use.

The backend runtime must have access to the model checkpoints and the
processed/report artifacts the API reads. Keep those files in the expected
repository-relative locations (`models/`, `data/processed/`, and
`reports/final_artifacts/`) or update the backend's artifact configuration.
Do not upload raw multi-gigabyte source datasets unless your research workflow
needs them at runtime.

## Verify after deployment

1. Open `https://<backend-host>/api/health` and confirm a healthy response.
2. Open `https://<backend-host>/api/models` and confirm expected models appear.
3. Load the Vercel site and confirm its API status shows connected.
4. Upload a representative WAV larger than 4.5 MB and run inference. This
   confirms uploads are reaching the Python host rather than a Vercel Function.
5. Check browser developer tools for CORS failures and check backend logs for
   model loading or missing artifact errors.

## If the backend must also run on Vercel

That requires a separate deployment redesign: the current upload size exceeds
the Vercel Function request limit, and the model/data bundle exceeds ordinary
function packaging expectations. A Vercel-only setup would need external
object storage or chunked upload plus a separately hosted inference worker and
artifact storage; simply adding a Python function would not preserve the
current prediction flow for all accepted WAVs.
