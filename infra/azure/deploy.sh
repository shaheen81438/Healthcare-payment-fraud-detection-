#!/usr/bin/env bash
# One-time provisioning of the Azure footprint for the payment-integrity platform:
#   resource group, ACR, Postgres Flexible Server, Container Apps env, and 2 apps.
# Safe to re-run: existing resources are updated in place.
#
# Prereqs: azure-cli >= 2.60 logged in (`az login`), subscription selected.
# Cost note: ~$25-40/month (B1ms Postgres + Consumption-plan Container Apps);
# run infra/azure/teardown.sh to remove everything.

set -euo pipefail

LOCATION="${LOCATION:-westus2}"
RG="${AZURE_RESOURCE_GROUP:-rg-payment-integrity}"
ACR="${ACR_NAME:-crpaymentintegrity}"
ENV_NAME="${CONTAINERAPPS_ENV:-cae-payment-integrity}"
PG_SERVER="${PG_SERVER:-pg-payment-integrity}"
PG_ADMIN="${PG_ADMIN:-pi_admin}"
PG_PASSWORD="${PG_PASSWORD:?set PG_PASSWORD env var (strong password) before running}"
API_APP="pi-api"
DASHBOARD_APP="pi-dashboard"

echo ">> resource group"
az group create -n "$RG" -l "$LOCATION" -o none

echo ">> container registry"
az acr create -n "$ACR" -g "$RG" --sku Basic --admin-enabled false -o none

echo ">> postgres flexible server (B1ms burstable)"
az postgres flexible-server create -n "$PG_SERVER" -g "$RG" -l "$LOCATION" \
  --tier Burstable --sku-name Standard_B1ms --storage-size 32 \
  --version 16 --admin-user "$PG_ADMIN" --admin-password "$PG_PASSWORD" \
  --public-access 0.0.0.0 -o none   # 0.0.0.0 = allow Azure services only
az postgres flexible-server db create -g "$RG" -s "$PG_SERVER" -d payment_integrity -o none
az postgres flexible-server db create -g "$RG" -s "$PG_SERVER" -d mlflow -o none

echo ">> container apps environment"
az containerapp env create -n "$ENV_NAME" -g "$RG" -l "$LOCATION" -o none

echo ">> build images in ACR (cloud build; no local docker needed)"
az acr build --registry "$ACR" --image pi-api:boot --target api .
az acr build --registry "$ACR" --image pi-dashboard:boot --target dashboard .

PG_HOST="$PG_SERVER.postgres.database.azure.com"
COMMON_ENV=(
  "POSTGRES_HOST=$PG_HOST" "POSTGRES_PORT=5432" "POSTGRES_USER=$PG_ADMIN"
  "POSTGRES_DB=payment_integrity" "ENV=prod" "LOG_LEVEL=INFO"
)

echo ">> API container app"
az containerapp create -n "$API_APP" -g "$RG" --environment "$ENV_NAME" \
  --image "$ACR.azurecr.io/pi-api:boot" --registry-server "$ACR.azurecr.io" \
  --registry-identity system --system-assigned \
  --target-port 8000 --ingress external --min-replicas 0 --max-replicas 2 \
  --cpu 1.0 --memory 2Gi \
  --secrets "pg-password=$PG_PASSWORD" \
  --env-vars "${COMMON_ENV[@]}" "POSTGRES_PASSWORD=secretref:pg-password" -o none

echo ">> dashboard container app"
az containerapp create -n "$DASHBOARD_APP" -g "$RG" --environment "$ENV_NAME" \
  --image "$ACR.azurecr.io/pi-dashboard:boot" --registry-server "$ACR.azurecr.io" \
  --registry-identity system --system-assigned \
  --target-port 8501 --ingress external --min-replicas 0 --max-replicas 1 \
  --cpu 1.0 --memory 2Gi \
  --secrets "pg-password=$PG_PASSWORD" \
  --env-vars "${COMMON_ENV[@]}" "POSTGRES_PASSWORD=secretref:pg-password" -o none

echo ">> grant the apps pull rights on ACR"
ACR_ID=$(az acr show -n "$ACR" -g "$RG" --query id -o tsv)
for APP in "$API_APP" "$DASHBOARD_APP"; do
  PRINCIPAL=$(az containerapp show -n "$APP" -g "$RG" --query identity.principalId -o tsv)
  az role assignment create --assignee "$PRINCIPAL" --role AcrPull --scope "$ACR_ID" -o none || true
done

echo ">> done"
az containerapp show -n "$API_APP" -g "$RG" --query properties.configuration.ingress.fqdn -o tsv \
  | sed 's/^/API:        https:\/\//'
az containerapp show -n "$DASHBOARD_APP" -g "$RG" --query properties.configuration.ingress.fqdn -o tsv \
  | sed 's/^/Dashboard:  https:\/\//'
