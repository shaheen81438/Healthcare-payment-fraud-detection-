"""Backward-looking duplicate detection features.

For every claim, candidate duplicates are earlier claims for the same
(beneficiary, HCPCS) within a lookback window. "Earlier" is defined by the
strict ordering (service_date, submit_date, claim_id), so exact same-day
duplicates flag the *later-submitted* claim — a claim can never be flagged as
a duplicate of the future.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DUP_FEATURES = ["dup_max_sim", "dup_cand_count", "exact_dup_prior"]


def duplicate_features(claims: pd.DataFrame, dup_cfg: dict[str, Any]) -> pd.DataFrame:
    """Compute duplicate-similarity features; returns claim_id + DUP_FEATURES."""
    w = dup_cfg["weights"]
    window = int(dup_cfg["window_days"])
    tol = float(dup_cfg["exact_amount_tolerance"])

    cols = [
        "claim_id",
        "bene_id",
        "provider_id",
        "hcpcs",
        "modifier",
        "units",
        "pos",
        "service_date",
        "submit_date",
        "paid_amount",
    ]
    base = claims[cols].copy()
    # Only blocks with >1 row can contain duplicates — huge cost saver.
    sizes = base.groupby(["bene_id", "hcpcs"])["claim_id"].transform("size")
    block = base[sizes > 1].copy()

    pairs = block.merge(block, on=["bene_id", "hcpcs"], suffixes=("", "_p"))
    # Vectorized strict lexicographic "prior < current" on (service, submit, claim_id).
    sd, sd_p = pairs["service_date"], pairs["service_date_p"]
    sub, sub_p = pairs["submit_date"], pairs["submit_date_p"]
    cid, cid_p = pairs["claim_id"], pairs["claim_id_p"]
    earlier = (
        (sd_p < sd)
        | ((sd_p == sd) & (sub_p < sub))
        | ((sd_p == sd) & (sub_p == sub) & (cid_p < cid))
    ).to_numpy()
    delta_days = (sd - sd_p).dt.days
    pairs = pairs[earlier & (delta_days >= 0) & (delta_days <= window)].copy()

    if pairs.empty:
        out = claims[["claim_id"]].copy()
        out["dup_max_sim"] = 0.0
        out["dup_cand_count"] = 0
        out["exact_dup_prior"] = 0
        return out

    delta = (pairs["service_date"] - pairs["service_date_p"]).dt.days.to_numpy(dtype=float)
    max_paid = np.maximum(
        pairs["paid_amount"].to_numpy(dtype=float),
        pairs["paid_amount_p"].to_numpy(dtype=float),
    )
    amount_rel = np.abs(
        pairs["paid_amount"].to_numpy(dtype=float) - pairs["paid_amount_p"].to_numpy(dtype=float)
    ) / np.maximum(max_paid, 0.01)

    sim = (
        w["same_provider"] * (pairs["provider_id"] == pairs["provider_id_p"]).to_numpy()
        + w["amount_close"] * (amount_rel <= 0.02)
        + w["same_pos"] * (pairs["pos"] == pairs["pos_p"]).to_numpy()
        + w["same_units"] * (pairs["units"] == pairs["units_p"]).to_numpy()
        + w["same_modifier"] * (pairs["modifier"] == pairs["modifier_p"]).to_numpy()
        + w["date_proximity"] * (1.0 - delta / window)
    )
    exact = (
        (pairs["provider_id"] == pairs["provider_id_p"]).to_numpy()
        & (delta == 0)
        & (pairs["units"] == pairs["units_p"]).to_numpy()
        & (amount_rel <= tol)
    )

    pairs = pairs.assign(sim=sim, exact=exact.astype(int))
    agg = pairs.groupby("claim_id").agg(
        dup_max_sim=("sim", "max"),
        dup_cand_count=("sim", "size"),
        exact_dup_prior=("exact", "max"),
    )

    out = claims[["claim_id"]].merge(agg, on="claim_id", how="left")
    out["dup_max_sim"] = out["dup_max_sim"].fillna(0.0).round(4)
    out["dup_cand_count"] = out["dup_cand_count"].fillna(0).astype(int)
    out["exact_dup_prior"] = out["exact_dup_prior"].fillna(0).astype(int)
    logger.info(
        "duplicate_features",
        extra={
            "claims": len(out),
            "with_candidates": int((out["dup_cand_count"] > 0).sum()),
            "exact_dups": int(out["exact_dup_prior"].sum()),
        },
    )
    return out
