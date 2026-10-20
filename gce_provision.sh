#!/usr/bin/env bash
# =============================================================================
# gce_provision.sh — create a GPU VM for training, trying bigger configs first.
#
# A100 (a2-*) capacity is tight in many zones; L4 (g2-*) is far more available
# and more than enough for this ~381k-param model. This tries, in order:
#   1. g2-standard-48  4x L4   48 vCPU  192 GB  ~$3.07/hr  (3 trainings parallel)
#   2. g2-standard-32  2x L4   32 vCPU  128 GB  ~$2.05/hr  (3 trainings parallel)
#   3. g2-standard-16  1x L4   16 vCPU   64 GB  ~$1.02/hr  (3 trainings sequential)
#   4. a2-highgpu-1g   1x A100 12 vCPU   85 GB  ~$3.67/hr  (3 trainings sequential)
#
# Usage:
#   PROJECT=your-project-id bash gce_provision.sh
# =============================================================================
set -uo pipefail

PROJECT="${PROJECT:?set PROJECT to your GCP project id, e.g. PROJECT=my-proj-123}"
VM_NAME="${VM_NAME:-gnn-train}"

ZONES=(
  us-central1-a us-central1-b us-central1-c us-central1-f
  us-east1-b us-west1-b us-west3-b us-west4-b
  europe-west4-a asia-northeast3-a me-west1-a
)

MACHINES=(g2-standard-48 g2-standard-32 g2-standard-16 a2-highgpu-1g)

for MT in "${MACHINES[@]}"; do
  for ZONE in "${ZONES[@]}"; do
    echo ">>> trying $MT in $ZONE ..."
    if gcloud compute instances create "$VM_NAME" \
        --project="$PROJECT" --zone="$ZONE" --machine-type="$MT" \
        --image-family=ubuntu-2204-lts --image-project=ubuntu-os-cloud \
        --boot-disk-size=200GB --boot-disk-type=pd-ssd \
        --maintenance-policy=TERMINATE --scopes=cloud-platform --quiet; then
      echo "Provisioned $MT in $ZONE (VM: $VM_NAME)"
      exit 0
    fi
    echo "    failed (capacity/quota) — next."
  done
done

echo "No GPU capacity in any tried zone. Widen ZONES or try again later."
exit 1
