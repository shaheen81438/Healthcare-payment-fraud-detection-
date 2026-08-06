# MLflow tracking server with Postgres driver.
# The official image ships without psycopg2, which the Postgres backend store requires.
FROM ghcr.io/mlflow/mlflow:v2.22.5

RUN pip install --no-cache-dir "psycopg2-binary>=2.9,<3"
