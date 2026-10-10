#!/usr/bin/env bash
# One-time Google Cloud setup for deploying this app to Cloud Run from GitHub Actions.
# Run it in Cloud Shell (console.cloud.google.com, ">_" icon) after selecting your project:
#
#   gcloud config set project YOUR_PROJECT_ID
#   bash <(curl -fsSL https://raw.githubusercontent.com/soumyanilain/notebooklm-clone/main/scripts/gcp_setup.sh)
#
# It is safe to run more than once. What it does:
#   1. enables the APIs Cloud Run deployments need
#   2. creates a Docker repository in Artifact Registry
#   3. stores your Groq API key in Secret Manager (you paste it; it is never printed)
#   4. creates a runtime service account (reads the secret) and a deployer service account
#   5. sets up Workload Identity Federation so GitHub Actions can deploy WITHOUT a JSON key
# At the end it prints the three values to add as GitHub Actions secrets.

set -euo pipefail

GITHUB_REPO="${GITHUB_REPO:-soumyanilain/notebooklm-clone}"   # owner/repo allowed to deploy
REGION="${REGION:-us-central1}"
AR_REPO="${AR_REPO:-notebooklm}"
PROJECT_ID="${PROJECT_ID:-$(gcloud config get-value project 2>/dev/null)}"

if [[ -z "${PROJECT_ID}" ]]; then
  echo "No project selected. Run: gcloud config set project YOUR_PROJECT_ID" >&2
  exit 1
fi
PROJECT_NUMBER="$(gcloud projects describe "${PROJECT_ID}" --format='value(projectNumber)')"
DEPLOYER="github-deployer@${PROJECT_ID}.iam.gserviceaccount.com"
RUNTIME="notebooklm-runtime@${PROJECT_ID}.iam.gserviceaccount.com"
POOL="github-pool"
PROVIDER="github-provider"

echo "==> Project ${PROJECT_ID} (${PROJECT_NUMBER}), region ${REGION}, repo ${GITHUB_REPO}"

echo "==> Enabling APIs (takes a minute the first time)"
gcloud services enable run.googleapis.com artifactregistry.googleapis.com secretmanager.googleapis.com \
  iam.googleapis.com iamcredentials.googleapis.com sts.googleapis.com cloudresourcemanager.googleapis.com

echo "==> Artifact Registry repository '${AR_REPO}'"
if ! gcloud artifacts repositories describe "${AR_REPO}" --location="${REGION}" >/dev/null 2>&1; then
  gcloud artifacts repositories create "${AR_REPO}" --repository-format=docker --location="${REGION}" \
    --description="Images for the NotebookLM clone"
fi

echo "==> Service accounts"
gcloud iam service-accounts describe "${RUNTIME}" >/dev/null 2>&1 || \
  gcloud iam service-accounts create notebooklm-runtime --display-name="NotebookLM clone (Cloud Run runtime)"
gcloud iam service-accounts describe "${DEPLOYER}" >/dev/null 2>&1 || \
  gcloud iam service-accounts create github-deployer --display-name="GitHub Actions deployer"

echo "==> Groq API key in Secret Manager"
if ! gcloud secrets describe GROQ_API_KEY >/dev/null 2>&1; then
  read -rsp "Paste your GROQ_API_KEY and press Enter (it will not be shown): " GROQ_KEY; echo
  if [[ -z "${GROQ_KEY}" ]]; then echo "No key entered." >&2; exit 1; fi
  printf "%s" "${GROQ_KEY}" | gcloud secrets create GROQ_API_KEY --replication-policy=automatic --data-file=-
  unset GROQ_KEY
else
  echo "    secret GROQ_API_KEY already exists (add a new version in Secret Manager to change it)"
fi
gcloud secrets add-iam-policy-binding GROQ_API_KEY --member="serviceAccount:${RUNTIME}" \
  --role=roles/secretmanager.secretAccessor --quiet >/dev/null

echo "==> Deployer permissions"
for ROLE in roles/run.admin roles/artifactregistry.writer; do
  gcloud projects add-iam-policy-binding "${PROJECT_ID}" --member="serviceAccount:${DEPLOYER}" \
    --role="${ROLE}" --condition=None --quiet >/dev/null
done
# the deployer may run the service as the runtime account
gcloud iam service-accounts add-iam-policy-binding "${RUNTIME}" --member="serviceAccount:${DEPLOYER}" \
  --role=roles/iam.serviceAccountUser --quiet >/dev/null

echo "==> Workload Identity Federation for GitHub Actions"
if ! gcloud iam workload-identity-pools describe "${POOL}" --location=global >/dev/null 2>&1; then
  gcloud iam workload-identity-pools create "${POOL}" --location=global --display-name="GitHub Actions"
fi
if ! gcloud iam workload-identity-pools providers describe "${PROVIDER}" --location=global \
    --workload-identity-pool="${POOL}" >/dev/null 2>&1; then
  gcloud iam workload-identity-pools providers create-oidc "${PROVIDER}" --location=global \
    --workload-identity-pool="${POOL}" --display-name="GitHub" \
    --issuer-uri="https://token.actions.githubusercontent.com" \
    --attribute-mapping="google.subject=assertion.sub,attribute.repository=assertion.repository" \
    --attribute-condition="assertion.repository=='${GITHUB_REPO}'"
fi
gcloud iam service-accounts add-iam-policy-binding "${DEPLOYER}" --role=roles/iam.workloadIdentityUser \
  --member="principalSet://iam.googleapis.com/projects/${PROJECT_NUMBER}/locations/global/workloadIdentityPools/${POOL}/attribute.repository/${GITHUB_REPO}" \
  --quiet >/dev/null

PROVIDER_NAME="projects/${PROJECT_NUMBER}/locations/global/workloadIdentityPools/${POOL}/providers/${PROVIDER}"

cat <<EOF

=====================================================================
Done. Add these three secrets in GitHub:
  ${GITHUB_REPO} -> Settings -> Secrets and variables -> Actions -> New repository secret

  GCP_PROJECT_ID        ${PROJECT_ID}
  GCP_WIF_PROVIDER      ${PROVIDER_NAME}
  GCP_SERVICE_ACCOUNT   ${DEPLOYER}

Then push to main (or re-run the workflow) to deploy.
Permission changes can take a minute or two to apply; if the first deploy
says "permission denied", wait a moment and re-run the workflow.
=====================================================================
EOF
