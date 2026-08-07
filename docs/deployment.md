# Deployment Guide

Three tiers, smallest to largest. The API and dashboard read their model bundle and
serving stores from the image (baked at build time), and use Postgres when reachable —
so every tier serves predictions even before wiring a database.

## 1. Local (no containers)

```bash
python -m venv .venv && source .venv/bin/activate
make install
make compose-up          # Postgres + MLflow (docker) — or point .env at your own Postgres
make pipeline            # data → features → train → evaluate → score  (~10 min)
make monitor             # Evidently drift run
make api                 # FastAPI on :8000  (docs at /docs)
make dashboard           # Streamlit on :8501
```

## 2. Local containers (compose)

```bash
make pipeline            # produce artifacts once (they get baked into images)
make compose-serve       # builds api + dashboard images; starts postgres, mlflow, api, dashboard
# API http://localhost:8000/docs · dashboard http://localhost:8501 · MLflow http://localhost:5001
```

## 3. Azure Container Apps (production-style)

Topology: **ACR** (images) + **Azure Database for PostgreSQL Flexible Server** (B1ms)
+ **Container Apps** environment with two apps (`pi-api`, `pi-dashboard`,
scale-to-zero) — ~$25–40/month while running.

### One-time provisioning

```bash
az login
export PG_PASSWORD='<strong-password>'
make pipeline                       # train artifacts locally first (baked into images)
bash infra/azure/deploy.sh          # RG + ACR + Postgres + env + both apps; prints URLs
```

### Continuous deployment (GitHub Actions)

`.github/workflows/deploy.yml` redeploys on manual dispatch or a `v*` tag using
**OIDC federated credentials** (no cloud passwords in GitHub):

1. In Entra ID, create an app registration with a federated credential for
   `repo:FaridoonFarahi/healthcare-payment-integrity:ref:refs/heads/main` (and `:ref:refs/tags/*`).
2. Grant its service principal `Contributor` on `rg-payment-integrity` and `AcrPush` on the ACR.
3. Add repo secrets `AZURE_CLIENT_ID`, `AZURE_TENANT_ID`, `AZURE_SUBSCRIPTION_ID`.
4. `git tag v1.0.0 && git push --tags` → build in ACR → update both apps → health smoke.

### Operational notes

- **Model updates** ship as new images (`make pipeline` → tag → deploy): immutable,
  rollback = redeploy previous tag. The MLflow registry remains the experiment system of
  record; images are the serving system of record.
- **Scale-to-zero** keeps the demo nearly free; first request pays a ~15s cold start.
- **Security posture (demo scope):** public ingress, no auth on either app — put Azure
  Entra auth / API keys and network restrictions in front before any real use; the
  dashboard exposes no PHI (all data synthetic) but should still be treated as internal.
- **Postgres bootstrap:** run `scripts/sql/init_db.sql` against the flexible server once
  (`psql "host=<server> user=pi_admin dbname=payment_integrity sslmode=require" -f scripts/sql/init_db.sql`),
  then load scored data with `python scripts/score_claims.py` pointed at it via `.env`.
- **Teardown:** `bash infra/azure/teardown.sh`.

## CI/CD pipeline (`.github/workflows/ci.yml`)

| Stage | Gate |
|---|---|
| quality | ruff format --check · ruff check · mypy (src + app) |
| tests | unit suite with coverage |
| model-gate | tiny-world pipeline train; **fails CI** if PR-AUC/recall floors or rules-beating check regress; then API integration tests + slow suite against those artifacts (Postgres service container) |
| docker | build api + dashboard images; boot API; `/health` smoke |
| deploy (separate workflow) | manual/tag-gated; OIDC login; ACR build; Container Apps update; health smoke |
