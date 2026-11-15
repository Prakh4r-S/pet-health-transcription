#!/usr/bin/env bash
#
# Builds the image from this directory with Cloud Build and deploys it
# to Cloud Run.
#
#   PROJECT_ID=my-project ./deploy.sh
#
# One-time setup this script assumes has been done:
#
#   1. Both secrets exist in Secret Manager:
#        printf '%s' "$VALUE" | gcloud secrets create SERVICE_API_KEY --data-file=-
#        printf '%s' "$VALUE" | gcloud secrets create GEMINI_API_KEY --data-file=-
#   2. The service's runtime service account has
#      roles/secretmanager.secretAccessor on each of them.
#
# service.yaml describes the same configuration declaratively; if you
# change a limit here, change it there too.

set -euo pipefail

PROJECT_ID="${PROJECT_ID:?Set PROJECT_ID to the Google Cloud project to deploy into}"
REGION="${REGION:-us-central1}"
SERVICE="${SERVICE:-pet-health-transcription}"
MAX_INSTANCES="${MAX_INSTANCES:-3}"

# Secret Manager secret names, in case yours differ from the variable names.
SERVICE_API_KEY_SECRET="${SERVICE_API_KEY_SECRET:-SERVICE_API_KEY}"
GEMINI_API_KEY_SECRET="${GEMINI_API_KEY_SECRET:-GEMINI_API_KEY}"

ALLOWED_AUDIO_HOSTS="${ALLOWED_AUDIO_HOSTS:-*.public.blob.vercel-storage.com,*.blob.vercel-storage.com}"
MAX_CONCURRENT_TRANSCRIPTIONS="${MAX_CONCURRENT_TRANSCRIPTIONS:-2}"

# The service checks a bearer token on every request, and its caller (the
# web application) sits outside Google Cloud, so Cloud Run's own IAM
# check is turned off by default. Set ALLOW_UNAUTHENTICATED=false if the
# caller can present a Google identity token instead.
if [[ "${ALLOW_UNAUTHENTICATED:-true}" == "true" ]]; then
  ACCESS_FLAG="--allow-unauthenticated"
else
  ACCESS_FLAG="--no-allow-unauthenticated"
fi

# "^@^" switches the list delimiter to "@", because ALLOWED_AUDIO_HOSTS
# itself contains commas.
ENV_VARS="^@^MAX_CONCURRENT_TRANSCRIPTIONS=${MAX_CONCURRENT_TRANSCRIPTIONS}@ALLOWED_AUDIO_HOSTS=${ALLOWED_AUDIO_HOSTS}"

# The port opens before the Whisper model has loaded. The startup probe
# holds traffic until /ready answers 200 (up to 120 s) and keeps CPU
# allocated while the weights load.
STARTUP_PROBE="httpGet.path=/ready,httpGet.port=8080,periodSeconds=5,timeoutSeconds=3,failureThreshold=24"

gcloud run deploy "${SERVICE}" \
  --project="${PROJECT_ID}" \
  --region="${REGION}" \
  --source=. \
  --port=8080 \
  --cpu=2 \
  --memory=2Gi \
  --concurrency=10 \
  --timeout=300 \
  --cpu-boost \
  --min-instances=0 \
  --max-instances="${MAX_INSTANCES}" \
  --startup-probe="${STARTUP_PROBE}" \
  --set-secrets="SERVICE_API_KEY=${SERVICE_API_KEY_SECRET}:latest,GEMINI_API_KEY=${GEMINI_API_KEY_SECRET}:latest" \
  --update-env-vars="${ENV_VARS}" \
  "${ACCESS_FLAG}"

echo
echo "Deployed. Service URL:"
gcloud run services describe "${SERVICE}" \
  --project="${PROJECT_ID}" --region="${REGION}" --format='value(status.url)'
