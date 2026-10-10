# Vercel frontend deployment

The production React/Vite frontend is hosted on Vercel. The FastAPI backend is a separate Docker web service on Render.

## Current project settings

- **Git repository:** `Vrushank2107/Respiratory-Disease-Classification`
- **Branch:** `master`
- **Root Directory:** `app/frontend`
- **Framework preset:** Vite
- **Build command:** `npm run build`
- **Output directory:** `dist`
- **API origin:** `https://respiratory-disease-classification.onrender.com`

The production API origin is stored in `app/frontend/.env.production` as `VITE_API_URL`. Vite embeds it when it builds the frontend. If `VITE_API_URL` is also set in Vercel Project Settings, that environment variable takes precedence; keep it equal to the Render API origin or remove it to use the checked-in default.

For local development, Vite uses its `/api` proxy to the API at `http://127.0.0.1:8000`; it does not use the production environment file in dev mode.

## Backend and CORS

Deploy the backend separately from the repository root using its `Dockerfile`; see the [Cloud Run alternative notes](cloud-run-deployment.md) for another container host. The backend includes the production Vercel origin in its default CORS allowlist. To allow other frontend origins, set `FRONTEND_ORIGINS` on the backend to a comma-separated list of exact origins, without trailing slashes.

## Why the backend is not deployed as a Vercel function

The attempted Python function build reported a **5,720.12 MB bundle** against a **500 MB function-size limit** for that deployment. The oversized bundle came from the backend's Python ML/audio dependencies. The project therefore keeps the frontend on Vercel and runs the API on Render as a Docker service.
