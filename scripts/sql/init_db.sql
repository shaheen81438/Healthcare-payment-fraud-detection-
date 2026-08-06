-- First-boot initialization for the local Postgres container.
-- Creates application schemas in the main database and a separate database for MLflow.
-- NOTE: docker-entrypoint-initdb.d scripts run only when the pgdata volume is first created.

CREATE SCHEMA IF NOT EXISTS raw;      -- ingested source data, as close to source as practical
CREATE SCHEMA IF NOT EXISTS ref;      -- reference/lookup tables (code sets, edit pairs, peer norms inputs)
CREATE SCHEMA IF NOT EXISTS staging;  -- validated + cleaned entities (claims, providers, beneficiaries)
CREATE SCHEMA IF NOT EXISTS labels;   -- generated weak-supervision labels, kept apart from source fields
CREATE SCHEMA IF NOT EXISTS features; -- point-in-time feature tables and frozen peer statistics
CREATE SCHEMA IF NOT EXISTS scoring;  -- model predictions, explanations, reason codes
CREATE SCHEMA IF NOT EXISTS ops;      -- investigator decisions, monitoring runs, alerts
CREATE SCHEMA IF NOT EXISTS meta;     -- dataset/model version bookkeeping

CREATE DATABASE mlflow;
