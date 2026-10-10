# Deploy the API to Google Cloud Run

This deploys the FastAPI backend as a public container service. Keep the React frontend on Vercel and configure it to call the Cloud Run service URL.

The included image installs the official CPU-only PyTorch wheel. This matches the current inference code, which loads checkpoints and tensors on CPU. It also avoids downloading CUDA libraries. Do not enable a Cloud Run GPU for this setup; a GPU would require additional app changes and much larger minimum Cloud Run resources.

## 1. Prepare Google Cloud

1. Create or select a Google Cloud project and attach a billing account. Cloud Run has a monthly free allowance, but usage beyond it and related services can be billed.
2. Enable the **Cloud Run**, **Artifact Registry**, and **Cloud Build** APIs.
3. Install and initialize the Google Cloud CLI, then select the project:

   ```sh
   gcloud auth login
   gcloud config set project YOUR_PROJECT_ID
   ```

## 2. Deploy from the repository root

Run this in the repository root (the directory containing `Dockerfile`). Replace `YOUR_PROJECT_ID` and set the Vercel URL to your actual frontend origin:

```sh
gcloud run deploy respiratory-api \
  --source . \
  --region asia-south1 \
  --allow-unauthenticated \
  --memory 4Gi \
  --cpu 2 \
  --timeout 300 \
  --max 3 \
  --set-env-vars FRONTEND_ORIGINS=https://YOUR-APP.vercel.app
```

Cloud Build builds the Docker image from the included `Dockerfile`; `.dockerignore` keeps raw data, notebooks, the virtual environment, frontend, and Git history out of the build context. Cloud Run supplies the `PORT` variable to the container.

After deployment, copy the service URL printed by `gcloud`, then check:

```text
https://YOUR-CLOUD-RUN-URL/api/health
```

## 3. Connect the Vercel frontend

In **Vercel → Project → Settings → Environment Variables**, set:

```text
VITE_API_URL=https://YOUR-CLOUD-RUN-URL
```

Use the Cloud Run service origin exactly, without a trailing slash. Redeploy the frontend after changing this build-time variable. The `FRONTEND_ORIGINS` value on Cloud Run must exactly match the deployed Vercel origin, also without a trailing slash.

## Notes

- `--allow-unauthenticated` makes this API publicly callable. Set application-level rate limits or authentication before sharing it broadly; public access can consume your quota.
- Cloud Run scales to zero by default. The first request after inactivity may be slower while the container starts and loads model files.
- The initial image includes model files and processed analysis artifacts, but not raw audio. Keep the model files and processed CSVs in the Git repository (or otherwise available to Cloud Build), because the backend needs them at runtime.
- Charges can include usage beyond Cloud Run's free allowance, image storage, builds, and network egress. Check the billing dashboard after deployment.
