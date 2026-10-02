"""Provider-level behavioral features at monthly grain, lagged one month.

A claim in month *m* joins provider features computed only from months up to
*m-1*. The current month never feeds its own features, so they stay
point-in-time correct even for batch scoring at month end.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

PROVIDER_FEATURES = [
    "prov_claims_3m",
    "prov_paid_3m",
    "prov_claims_1m",
    "prov_velocity_1v3",
    "prov_distinct_benes_3m",
    "prov_claims_per_bene_3m",
    "prov_new_bene_share_1m",
    "prov_top_code_share_3m",
    "prov_em_level_mean_3m",
    "prov_high_cost_share_3m",
    "prov_paid_vs_peer_3m",
    "prov_months_active",
]


def build_provider_month_features(
    claims: pd.DataFrame,
    em_levels: pd.DataFrame,
    code_stats: pd.DataFrame,
    provider_month_peer: pd.DataFrame,
    providers: pd.DataFrame,
    cfg: dict[str, Any],
) -> pd.DataFrame:
    """Return one row per (provider_id, month) with trailing features from prior months only."""
    trailing = int(cfg["provider_rolling"]["trailing_months"])
    df = claims.copy()
    df["month_p"] = df["service_date"].dt.to_period("M")
    df["paid_per_unit"] = df["paid_amount"] / df["units"].clip(lower=1)

    em_map = dict(zip(em_levels["hcpcs"], em_levels["level"].astype(int), strict=True))
    df["em_level"] = df["hcpcs"].map(em_map)
    high_cost_thr = dict(
        zip(
            code_stats["hcpcs"],
            code_stats["code_paid_unit_median"].astype(float) * 2.0,
            strict=True,
        )
    )
    df["high_cost"] = df["paid_per_unit"] > df["hcpcs"].map(high_cost_thr).fillna(np.inf)

    # First-ever appearance of a (provider, bene) pair marks a "new" beneficiary.
    df = df.sort_values(["provider_id", "service_date", "claim_id"])
    df["new_bene"] = ~df.duplicated(subset=["provider_id", "bene_id"])

    monthly = (
        df.groupby(["provider_id", "month_p"])
        .agg(
            claims_1m=("claim_id", "size"),
            paid_1m=("paid_amount", "sum"),
            benes_1m=("bene_id", "nunique"),
            new_benes_1m=("new_bene", "sum"),
            em_sum_1m=("em_level", "sum"),
            em_n_1m=("em_level", "count"),
            high_cost_1m=("high_cost", "sum"),
        )
        .reset_index()
    )
    # Top-code share needs per-code monthly counts.
    code_month = df.groupby(["provider_id", "month_p", "hcpcs"]).size().rename("n").reset_index()

    # Dense provider-month grid across the full window so gaps count as zero months.
    all_months = pd.period_range(df["month_p"].min(), df["month_p"].max(), freq="M")
    grid = pd.MultiIndex.from_product(
        [monthly["provider_id"].unique(), all_months], names=["provider_id", "month_p"]
    ).to_frame(index=False)
    monthly = grid.merge(monthly, on=["provider_id", "month_p"], how="left").fillna(0.0)
    monthly = monthly.sort_values(["provider_id", "month_p"]).reset_index(drop=True)

    g = monthly.groupby("provider_id")
    # Trailing sums over the prior `trailing` months (shift(1) excludes the current month).
    for col in ["claims_1m", "paid_1m", "benes_1m", "em_sum_1m", "em_n_1m", "high_cost_1m"]:
        monthly[f"{col}_t"] = g[col].transform(
            lambda s: s.shift(1).rolling(trailing, min_periods=1).sum()
        )
    monthly["claims_prev_1m"] = g["claims_1m"].transform(lambda s: s.shift(1)).fillna(0.0)
    monthly["new_bene_share_1m"] = np.where(
        monthly["claims_prev_1m"] > 0,
        g["new_benes_1m"].transform(lambda s: s.shift(1)).fillna(0.0)
        / monthly["claims_prev_1m"].clip(lower=1),
        0.0,
    )
    monthly["months_active"] = g["claims_1m"].transform(
        lambda s: (s > 0).astype(int).shift(1, fill_value=0).cumsum()
    )

    # Trailing top-code share (max single-code share of prior months' claims).
    code_month = code_month.sort_values(["provider_id", "hcpcs", "month_p"])
    code_month["n_t"] = code_month.groupby(["provider_id", "hcpcs"])["n"].transform(
        lambda s: s.shift(1).rolling(trailing, min_periods=1).sum()
    )
    # Reindexing against the month grid is slow, so compute the share within observed
    # months and merge (missing = 0 means no prior activity for that code).
    top_code = (
        code_month.groupby(["provider_id", "month_p"])["n_t"].max().rename("top_code_n_t")
    ).reset_index()
    monthly = monthly.merge(top_code, on=["provider_id", "month_p"], how="left")
    monthly["top_code_n_t"] = monthly["top_code_n_t"].fillna(0.0)

    out = pd.DataFrame(
        {
            "provider_id": monthly["provider_id"],
            "month": monthly["month_p"].astype(str),
            "prov_claims_3m": monthly["claims_1m_t"],
            "prov_paid_3m": monthly["paid_1m_t"].round(2),
            "prov_claims_1m": monthly["claims_prev_1m"],
            "prov_velocity_1v3": (
                monthly["claims_prev_1m"] / (monthly["claims_1m_t"] / trailing).clip(lower=1)
            ).round(4),
            "prov_distinct_benes_3m": monthly["benes_1m_t"],
            "prov_claims_per_bene_3m": (
                monthly["claims_1m_t"] / monthly["benes_1m_t"].clip(lower=1)
            ).round(4),
            "prov_new_bene_share_1m": monthly["new_bene_share_1m"].round(4),
            "prov_top_code_share_3m": (
                monthly["top_code_n_t"] / monthly["claims_1m_t"].clip(lower=1)
            ).round(4),
            "prov_em_level_mean_3m": (
                monthly["em_sum_1m_t"] / monthly["em_n_1m_t"].clip(lower=1)
            ).round(4),
            "prov_high_cost_share_3m": (
                monthly["high_cost_1m_t"] / monthly["claims_1m_t"].clip(lower=1)
            ).round(4),
            "prov_months_active": monthly["months_active"].astype(int),
        }
    )

    # Peer normalization uses frozen train-window medians.
    out = out.merge(providers[["provider_id", "specialty"]], on="provider_id", how="left")
    out = out.merge(provider_month_peer, on="specialty", how="left")
    out["prov_paid_vs_peer_3m"] = (
        out["prov_paid_3m"]
        / int(cfg["provider_rolling"]["trailing_months"])
        / out["peer_monthly_paid_median"].clip(lower=1.0)
    ).round(4)
    out = out.drop(
        columns=[
            "specialty",
            "peer_monthly_claims_median",
            "peer_monthly_paid_median",
            "peer_monthly_paid_p95",
        ]
    )
    logger.info("provider_month_features", extra={"rows": len(out)})
    return out
