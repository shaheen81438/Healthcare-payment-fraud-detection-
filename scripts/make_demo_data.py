#!/usr/bin/env python
"""Build the committed demo_data/ bundle for the hosted Streamlit demo.

Streamlit Community Cloud clones the git repo, which excludes the generated
data/ and models/ artifacts. This script samples the scored outputs into a
small committed bundle the dashboard falls back to when the full artifacts are
absent (see app/dashboard/common.py). Run after `make pipeline`.

Usage: python scripts/make_demo_data.py
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

OUT = Path("demo_data")
EXPLANATION_EXTRA_TOP_N = 5000  # auto-clear claims still browsable in Claim Detail


def main() -> int:
    OUT.mkdir(exist_ok=True)

    predictions = pd.read_parquet("data/processed/predictions.parquet")
    predictions.to_parquet(OUT / "predictions.parquet", index=False)

    # Claims: test window plus the 45-day lookback used by the similar-claims panel.
    claims = pd.read_parquet("data/processed/claims.parquet")
    start = pd.to_datetime(predictions["service_date"]).min() - pd.Timedelta(days=45)
    claims_demo = claims[pd.to_datetime(claims["service_date"]) >= start].copy()
    claims_demo.to_parquet(OUT / "claims.parquet", index=False)

    labels = pd.read_parquet("data/processed/labels.parquet")
    labels[labels["claim_id"].isin(set(claims_demo["claim_id"]))].to_parquet(
        OUT / "labels.parquet", index=False
    )

    # Explanations: everything the queue surfaces + the highest-risk auto-clears.
    explanations = pd.read_parquet("data/processed/explanations.parquet")
    keep_ids = set(
        predictions.loc[predictions["recommended_action"] != "auto_clear", "claim_id"]
    ) | set(
        predictions[predictions["recommended_action"] == "auto_clear"].nlargest(
            EXPLANATION_EXTRA_TOP_N, "risk_score"
        )["claim_id"]
    )
    explanations[explanations["claim_id"].isin(keep_ids)].to_parquet(
        OUT / "explanations.parquet", index=False
    )

    # Frozen peer stats + the threshold policy (read from the bundle metadata).
    for stats_file in Path("models/feature_store").glob("*.parquet"):
        if stats_file.name != "monitoring_reference.parquet":
            shutil.copy(stats_file, OUT / stats_file.name)
    pointer = json.loads(Path("models/latest.json").read_text(encoding="utf-8"))
    meta = json.loads((Path(pointer["path"]) / "metadata.json").read_text(encoding="utf-8"))
    (OUT / "policy.json").write_text(
        json.dumps(meta.get("threshold_policy", {}), indent=2), encoding="utf-8"
    )

    total_mb = sum(f.stat().st_size for f in OUT.iterdir()) / 1e6
    print(
        json.dumps(
            {
                "predictions": len(predictions),
                "claims": len(claims_demo),
                "explanations": int(explanations["claim_id"].isin(keep_ids).sum()),
                "bundle_mb": round(total_mb, 1),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
