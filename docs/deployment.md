# Deployment guide

## 0. Hosted dashboard demo (free)

The investigator dashboard deploys to Streamlit Community Cloud with no infrastructure of its own.
The repo commits an 8 MB sampled bundle (`demo_data/`, built by `scripts/make_demo_data.py`)
that the dashboard loaders fall back to when full artifacts are absent. To deploy, sign in at
share.streamlit.io with GitHub, choose New app, then this repo, branch `main`, entry point
`app/dashboard/Home.py` (pick Python 3.11 under Advanced settings). Rebuild the bundle after any
retrain with `python scripts/make_demo_data.py` and push. The hosted demo has no scoring API,
MLflow, or Postgres; those belong to the tiers below.

There are three tiers, smallest to largest. The API and dashboard read their model bundle and
serving stores from the image (baked at build time) and use Postgres when it is reachable, so
every tier serves predictions even before a database is wired up.

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

Topology: ACR (images), Azure Database for PostgreSQL Flexible Server (B1ms), and a
Container Apps environment with two apps (`pi-api`, `pi-dashboard`, scale-to-zero). Cost is
~$25-40/month while running.

### One-time provisioning

```bash
az login
export PG_PASSWORD='<strong-password>'
make pipeline                       # train artifacts locally first (baked into images)
bash infra/azure/deploy.sh          # RG + ACR + Postgres + env + both apps; prints URLs
```

### Continuous deployment (GitHub Actions)

`.github/workflows/deploy.yml` redeploys on manual dispatch or a `v*` tag using
OIDC federated credentials (no cloud passwords in GitHub):

1. In Entra ID, create an app registration with a federated credential for
   `repo:FaridoonFarahi/healthcare-payment-integrity:ref:refs/heads/main` (and `:ref:refs/tags/*`).
2. Grant its service principal `Contributor` on `rg-payment-integrity` and `AcrPush` on the ACR.
3. Add repo secrets `AZURE_CLIENT_ID`, `AZURE_TENANT_ID`, `AZURE_SUBSCRIPTION_ID`.
4. Run `git tag v1.0.0 && git push --tags`. The workflow builds in ACR, updates both apps, and runs a health smoke test.

### Operational notes

- Model updates ship as new, immutable images (`make pipeline`, then tag, then deploy). To roll
  back, redeploy the previous tag. The MLflow registry stays the system of record for
  experiments; images are the system of record for serving.
- Scale-to-zero keeps the demo nearly free; the first request pays a ~15s cold start.
- Security (demo scope): public ingress and no auth on either app. Put Azure Entra auth or API
  keys and network restrictions in front before any real use. The dashboard exposes no PHI (all
  data is synthetic) but should still be treated as internal.
- Postgres bootstrap: run `scripts/sql/init_db.sql` against the flexible server once
  (`psql "host=<server> user=pi_admin dbname=payment_integrity sslmode=require" -f scripts/sql/init_db.sql`),
  then load scored data with `python scripts/score_claims.py` pointed at it via `.env`.
- Teardown: `bash infra/azure/teardown.sh`.

## CI/CD pipeline (`.github/workflows/ci.yml`)

| Stage | Gate |
|---|---|
| quality | ruff format --check, ruff check, mypy (src + app) |
| tests | unit suite with coverage |
| model-gate | tiny-world pipeline train; **fails CI** if PR-AUC/recall floors or the rules-beating check regress; then API integration tests and the slow suite against those artifacts (Postgres service container) |
| docker | build api + dashboard images; boot API; `/health` smoke |
| deploy (separate workflow) | manual/tag-gated; OIDC login; ACR build; Container Apps update; health smoke |
