#!/usr/bin/env bash
# =============================================================================
# gce_provision.sh — create a GPU VM for training, trying the HIGHEST config
# first and falling back to lower ones until one is available.
#
# A100 (a2-*) capacity is tight in many zones; L4 (g2-*) is far more available
# and more than enough for this ~381k-param model. Order (most -> least powerful):
#   a2-highgpu-8g   8x A100  96 vCPU  680 GB  ~$29.39/hr
#   a2-highgpu-4g   4x A100  48 vCPU  340 GB  ~$14.69/hr
#   a2-highgpu-2g   2x A100  24 vCPU  170 GB  ~$7.35/hr
#   g2-standard-96  8x L4    96 vCPU  384 GB  ~$6.15/hr
#   a2-highgpu-1g   1x A100  12 vCPU   85 GB  ~$3.67/hr
#   g2-standard-48  4x L4    48 vCPU  192 GB  ~$3.07/hr
#   g2-standard-32  2x L4    32 vCPU  128 GB  ~$2.05/hr
#   g2-standard-16  1x L4    16 vCPU   64 GB  ~$1.02/hr
#
# The first config that provisions is kept (the highest one available), so if an
# A100 zone has capacity you get the A100; otherwise you get the best L4.
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

# highest -> lowest
MACHINES=(
  a2-highgpu-8g
  a2-highgpu-4g
  a2-highgpu-2g
  g2-standard-96
  a2-highgpu-1g
  g2-standard-48
  g2-standard-32
  g2-standard-16
)

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
