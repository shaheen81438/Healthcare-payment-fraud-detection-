#!/usr/bin/env bash
# Configure Microsoft Entra Easy Auth and group authorization before exposing apps.
set -euo pipefail

RG="${AZURE_RESOURCE_GROUP:-rg-payment-integrity}"
API_APP="${API_APP:-pi-api}"
DASHBOARD_APP="${DASHBOARD_APP:-pi-dashboard}"

: "${ENTRA_TENANT_ID:?Set ENTRA_TENANT_ID}"
: "${ENTRA_ALLOWED_GROUP_ID:?Set ENTRA_ALLOWED_GROUP_ID to the authorized Entra security group object ID}"
: "${API_ENTRA_CLIENT_ID:?Set API_ENTRA_CLIENT_ID for a dedicated single-tenant app registration}"
: "${API_ENTRA_CLIENT_SECRET:?Set API_ENTRA_CLIENT_SECRET from a secret manager}"
: "${DASHBOARD_ENTRA_CLIENT_ID:?Set DASHBOARD_ENTRA_CLIENT_ID for a dedicated single-tenant app registration}"
: "${DASHBOARD_ENTRA_CLIENT_SECRET:?Set DASHBOARD_ENTRA_CLIENT_SECRET from a secret manager}"

if [[ "$API_ENTRA_CLIENT_ID" == "$DASHBOARD_ENTRA_CLIENT_ID" ]]; then
  echo "Use separate Entra app registrations for the dashboard and API." >&2
  exit 2
fi
if [[ ! "$ENTRA_ALLOWED_GROUP_ID" =~ ^[0-9a-fA-F-]{36}$ ]]; then
  echo "ENTRA_ALLOWED_GROUP_ID must be a GUID security group object ID." >&2
  exit 2
fi

configure_app() {
  local app_name="$1"
  local port="$2"
  local client_id="$3"
  local client_secret="$4"
  local unauthenticated_action="$5"
  local sign_in_audience

  sign_in_audience="$(az ad app show --id "$client_id" --query signInAudience --output tsv)"
  if [[ "$sign_in_audience" != "AzureADMyOrg" ]]; then
    echo "Entra registration $client_id must be single-tenant (AzureADMyOrg)." >&2
    return 1
  fi

  # Close an existing external endpoint while the auth policy is being updated.
  az containerapp ingress enable --name "$app_name" --resource-group "$RG" \
    --type internal --target-port "$port" --transport auto --allow-insecure false -o none

  az containerapp secret set --name "$app_name" --resource-group "$RG" \
    --secrets "entra-client-secret=$client_secret" -o none
  az containerapp auth microsoft update --name "$app_name" --resource-group "$RG" \
    --client-id "$client_id" --tenant-id "$ENTRA_TENANT_ID" \
    --client-secret-name entra-client-secret --yes -o none
  az containerapp auth update --name "$app_name" --resource-group "$RG" \
    --enabled true --require-https true --proxy-convention Standard \
    --unauthenticated-client-action "$unauthenticated_action" \
    --redirect-provider AzureActiveDirectory \
    --set "identityProviders.azureActiveDirectory.validation.defaultAuthorizationPolicy.allowedPrincipals.groups=[\"$ENTRA_ALLOWED_GROUP_ID\"]" \
    --yes -o none

  local allowed_group
  allowed_group="$(az containerapp auth show --name "$app_name" --resource-group "$RG" \
    --query 'identityProviders.azureActiveDirectory.validation.defaultAuthorizationPolicy.allowedPrincipals.groups[0]' \
    --output tsv)"
  if [[ "${allowed_group,,}" != "${ENTRA_ALLOWED_GROUP_ID,,}" ]]; then
    echo "Entra group authorization did not verify for $app_name; public ingress remains closed." >&2
    return 1
  fi
  local allowed_group_count
  allowed_group_count="$(az containerapp auth show --name "$app_name" --resource-group "$RG" \
    --query 'length(identityProviders.azureActiveDirectory.validation.defaultAuthorizationPolicy.allowedPrincipals.groups)' \
    --output tsv)"
  if [[ "$allowed_group_count" != "1" ]]; then
    echo "Expected exactly one authorized Entra group for $app_name; public ingress remains closed." >&2
    return 1
  fi

  az ad app update --id "$client_id" --enable-id-token-issuance true \
    --set groupMembershipClaims=SecurityGroup -o none

  # Open external HTTPS ingress only after authentication and group policy verify.
  az containerapp ingress enable --name "$app_name" --resource-group "$RG" \
    --type external --target-port "$port" --transport auto --allow-insecure false -o none

  local fqdn callback_uri
  fqdn="$(az containerapp show --name "$app_name" --resource-group "$RG" \
    --query 'properties.configuration.ingress.fqdn' --output tsv)"
  if [[ -z "$fqdn" || "$fqdn" == "None" ]]; then
    echo "Azure did not return an external hostname for $app_name." >&2
    return 1
  fi
  callback_uri="https://$fqdn/.auth/login/aad/callback"
  # These registrations must be dedicated to this app; this intentionally sets
  # their web redirect URI to the current ACA callback.
  az ad app update --id "$client_id" --web-redirect-uris "$callback_uri" -o none

  local policy_action
  policy_action="$(az containerapp auth show --name "$app_name" --resource-group "$RG" \
    --query 'globalValidation.unauthenticatedClientAction' --output tsv)"
  if [[ "$policy_action" != "$unauthenticated_action" ]]; then
    echo "Unexpected unauthenticated action for $app_name: $policy_action" >&2
    return 1
  fi
  printf '%s: https://%s (Entra group restricted)\n' "$app_name" "$fqdn"
}

# Configure both policies before publishing either application.
configure_app "$API_APP" 8000 "$API_ENTRA_CLIENT_ID" \
  "$API_ENTRA_CLIENT_SECRET" Return401
configure_app "$DASHBOARD_APP" 8501 "$DASHBOARD_ENTRA_CLIENT_ID" \
  "$DASHBOARD_ENTRA_CLIENT_SECRET" RedirectToLoginPage

echo "Both apps require Microsoft Entra authentication and membership in the configured security group."
