# Vercel multi-service deployment

The repository-root `vercel.json` defines two services:

- `frontend` builds the Vite UI from `app/frontend` and serves all paths other
  than `/api/*`.
- `backend` runs the FastAPI app from the repository root, with the processed
  research artifacts and model checkpoints included in its function bundle.
  The repository root is used so the app's existing package imports and
  repository-relative artifact paths continue to resolve.

The browser already calls `/api/...`, so Vercel's public rewrite routes those
same-origin requests to FastAPI. There is no server-side service-to-service
call in this app; a service binding would not be usable by the static browser
code and is not needed for this routing setup. `VITE_API_URL` should remain
unset for this deployment.

## Vercel dashboard

Import the repository and keep the Vercel project root at the repository root
so it can read the root `vercel.json`. Vercel Services are currently in beta;
confirm the feature is available for the Vercel team/project. The frontend
build uses `npm run build` in `app/frontend`. The backend installs
`app/backend/requirements.txt` and uses `app.backend.main:app` as its ASGI
entrypoint.

## Compatibility limits to resolve

The current FastAPI app accepts uploads up to 20 MiB. Vercel Functions cap
request and response bodies at 4.5 MB, so recordings above that limit will
fail before reaching FastAPI. Keeping the existing 20 MiB upload behavior
requires a separately hosted inference API or an upload redesign. Reducing the
allowed WAV size can make this flow fit Vercel's request cap, but the limit
must be below 4.5 MB to leave room for multipart overhead.

Vercel's standard Python function bundle limit is 500 MB. The API depends on
PyTorch and scientific Python packages; the local environment's installed
PyTorch package is about 590 MB. The Vercel Large Functions option is currently
public beta and supports larger Python bundles when enabled. Confirm that it
is available for this project before relying on this backend deployment.

The API needs `data/processed/`, `models/`, and `reports/final_artifacts/` at
runtime. The include pattern in `vercel.json` packages those directories but
does not include raw source audio that the running API does not read.

## Test locally

Install the Vercel CLI and run `vercel dev -L` from the repository root to
exercise the service routing locally. Then check `/api/health`, `/api/models`,
load the root page, and submit a WAV small enough to fit the function request
limit. A Vercel deployment will not preserve the full local 20 MiB upload
limit.
