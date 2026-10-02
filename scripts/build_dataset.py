#!/usr/bin/env python
"""Build the full dataset: reference tables, CMS/LEIE ingestion, synthetic claims,
validation, cleaning, and Postgres/parquet persistence.

Usage:
    python scripts/build_dataset.py [--config configs/data_generation.yaml]
        [--cms-api] [--skip-db]

Safe to rerun; the same config and seed give the same output.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import clean as cleaning
from src.data import validate as validation
from src.data.db import write_df
from src.data.ingest_cms import ingest_cms
from src.data.ingest_leie import ingest_leie
from src.data.reference import load_reference_frames, load_reference_tables
from src.data.synthetic.assemble import build_dataset
from src.data.synthetic.generate import load_generation_config
from src.utils.logging import configure_logging


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/data_generation.yaml")
    parser.add_argument("--cms-api", action="store_true", help="pull CMS data from the live API")
    parser.add_argument("--skip-db", action="store_true", help="write files only, no Postgres")
    args = parser.parse_args()
    configure_logging("INFO")

    cfg = load_generation_config(args.config)
    ref = load_reference_frames()

    # 1) Ingest real reference data.
    if args.skip_db:
        from src.data.ingest_cms import load_snapshot

        cms_df = load_snapshot()
    else:
        load_reference_tables()
        cms_df = ingest_cms(use_snapshot=not args.cms_api)
        ingest_leie(use_fixture=True)

    # 2) Generate the synthetic dataset (claims, with labels kept separate).
    bundle = build_dataset(cfg, cms_df, ref)

    # 3) Persist file artifacts.
    Path("data/raw").mkdir(parents=True, exist_ok=True)
    Path("data/interim").mkdir(parents=True, exist_ok=True)
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    Path("monitoring/reports").mkdir(parents=True, exist_ok=True)
    bundle.raw_claims.to_csv("data/raw/claims_raw.csv.gz", index=False, compression="gzip")
    bundle.claims.to_parquet("data/interim/ground_truth_claims.parquet", index=False)
    bundle.labels.to_parquet("data/processed/labels.parquet", index=False)
    bundle.providers.to_parquet("data/processed/providers.parquet", index=False)
    bundle.beneficiaries.to_parquet("data/processed/beneficiaries.parquet", index=False)

    # 4) Validate raw claims, quarantine failures, write the report.
    raw = pd.read_csv("data/raw/claims_raw.csv.gz", dtype={"hcpcs": str, "pos": str})
    result = validation.validate_claims(raw, ref["pos_codes"], ref["dx_groups"])
    result.quarantined.to_parquet("data/interim/quarantine.parquet", index=False)
    report_path = Path("monitoring/reports/validation_report.json")
    report = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "dataset_version": bundle.dataset_version,
        **result.report,
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if not result.report["within_tolerance"]:
        print(f"FATAL: quarantine share {result.report['quarantine_share']} above tolerance")
        return 1

    # 5) Clean into staging.
    clean_df, clean_report = cleaning.clean_claims(result.passed)
    clean_df.to_parquet("data/processed/claims.parquet", index=False)

    # 6) Load Postgres.
    if not args.skip_db:
        write_df(clean_df, "claims", schema="staging")
        write_df(bundle.providers, "providers", schema="staging")
        write_df(bundle.beneficiaries, "beneficiaries", schema="staging")
        write_df(bundle.labels, "claim_labels", schema="labels")
        meta = pd.DataFrame(
            [
                {
                    "dataset_version": bundle.dataset_version,
                    "created_at": datetime.now(UTC),
                    "rows_claims": len(clean_df),
                    "positive_rate": bundle.stats["positive_rate"],
                    "config_path": args.config,
                }
            ]
        )
        write_df(meta, "dataset_versions", schema="meta", if_exists="append")

    # 7) Summary.
    stats_path = Path("data/interim/generation_stats.json")
    stats_path.write_text(json.dumps(bundle.stats, indent=2, default=str), encoding="utf-8")
    print(
        json.dumps(
            {
                "dataset_version": bundle.dataset_version,
                "claims_clean": len(clean_df),
                "claims_raw": len(bundle.raw_claims),
                "quarantined": result.report["rows_quarantined"],
                "positive_rate": round(bundle.stats["positive_rate"], 4),
                "improper_amount_total": round(bundle.stats["improper_amount_total"], 2),
                "ingestion_dups_removed": clean_report["ingestion_duplicates_removed"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
