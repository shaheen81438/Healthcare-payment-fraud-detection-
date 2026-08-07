# MLflow tracking server with Postgres driver.
# Built from the Python base image + PyPI (the ghcr.io/mlflow/mlflow tags don't
# reliably exist for every patch release), keeping the server in the same 2.x
# line as the client pin in requirements.txt.
FROM python:3.11-slim

RUN pip install --no-cache-dir "mlflow>=2.19,<3" "psycopg2-binary>=2.9,<3"

EXPOSE 5000
