#!/usr/bin/env bash
# Remove the entire Azure footprint (resource group deletion cascades everything).
set -euo pipefail
RG="${AZURE_RESOURCE_GROUP:-rg-payment-integrity}"
echo "Deleting resource group $RG (this removes ALL platform resources)…"
az group delete -n "$RG" --yes --no-wait
echo "Deletion started (runs asynchronously)."
