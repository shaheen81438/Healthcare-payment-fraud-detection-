"""Claim-level features: amount norms, timing, coding, and beneficiary history.

All history-dependent features are strictly backward-looking (``closed='left'``
rolling windows, shifted first-occurrence counts). Same-day context features
(counts of other same-day claims) are legitimate in this platform's post-payment
batch setting and are documented as such.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.data.reference import hcpcs_category

logger = logging.getLogger(__name__)

SENTINEL_DAYS = 999.0


def amount_and_coding_features(
    claims: pd.DataFrame,
    ref: dict[str, pd.DataFrame],
    peer: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Peer-normalized amounts, E/M level gaps, dx compatibility, POS typicality, unbundling."""
    df = claims.copy()
    df["paid_per_unit"] = (df["paid_amount"] / df["units"].clip(lower=1)).round(4)

    df = df.merge(
        peer["code_stats"][["hcpcs", "code_paid_unit_median", "code_units_p95"]],
        on="hcpcs",
        how="left",
    )
    df = df.merge(
        peer["spec_code_stats"][["specialty", "hcpcs", "spec_code_paid_unit_median"]],
        on=["specialty", "hcpcs"],
        how="left",
    )
    df = df.merge(peer["pos_stats"][["hcpcs", "code_office_share"]], on="hcpcs", how="left")
    df = df.merge(peer["em_spec_stats"], on="specialty", how="left")

    df["paid_vs_code_median"] = (
        (df["paid_per_unit"] / df["code_paid_unit_median"].clip(lower=0.01)).fillna(1.0).round(4)
    )
    df["paid_vs_spec_code_median"] = (
        (df["paid_per_unit"] / df["spec_code_paid_unit_median"].clip(lower=0.01))
        .fillna(df["paid_vs_code_median"])
        .round(4)
    )
    df["units_vs_code_p95"] = (
        (df["units"] / df["code_units_p95"].clip(lower=1)).fillna(1.0).round(4)
    )

    # Timing
    df["submit_lag_days"] = (df["submit_date"] - df["service_date"]).dt.days.clip(lower=0)
    df["service_days"] = (df["service_end_date"] - df["service_date"]).dt.days.clip(lower=0)
    df["dow"] = df["service_date"].dt.dayofweek
    df["month_num"] = df["service_date"].dt.month

    # Modifiers
    df["mod_25"] = (df["modifier"] == "25").astype(int)
    df["mod_59"] = (df["modifier"] == "59").astype(int)
    df["mod_ltrt"] = df["modifier"].isin(["LT", "RT"]).astype(int)

    # E/M level + upcoding gap vs specialty norm
    em = ref["em_levels"]
    em_map = dict(zip(em["hcpcs"], em["level"].astype(int), strict=True))
    df["em_level"] = df["hcpcs"].map(em_map).fillna(0).astype(int)
    df["em_gap_vs_specialty"] = np.where(
        df["em_level"] > 0, df["em_level"] - df["spec_em_level_mean"].fillna(3.0), 0.0
    ).round(4)

    # Diagnosis-procedure compatibility
    compat = ref["dx_procedure_compat"].copy()
    compat["compat"] = compat["compat"].astype(float)
    df["hcpcs_cat"] = [hcpcs_category(c) for c in df["hcpcs"].astype(str)]
    df = df.merge(
        compat.rename(columns={"hcpcs_category": "hcpcs_cat"}),
        on=["hcpcs_cat", "dx_group"],
        how="left",
    )
    df["dxproc_compat"] = df["compat"].fillna(0.6).round(4)

    # POS typicality
    df["pos_office"] = df["pos"].isin(["11", "12"]).astype(int)
    df["pos_atypicality"] = (
        (df["pos_office"] - df["code_office_share"].fillna(df["pos_office"])).abs().round(4)
    )

    # Unbundling: component billed same bene/provider/day as its comprehensive code,
    # WITHOUT a bypass modifier.
    ptp = ref["ptp_pairs"]
    keys = ["bene_id", "provider_id", "service_date"]
    comp_claims = df[df["hcpcs"].isin(set(ptp["comprehensive_code"]))][[*keys, "hcpcs"]]
    comp_claims = comp_claims.rename(columns={"hcpcs": "comprehensive_code"})
    expected_components = comp_claims.merge(
        ptp[["comprehensive_code", "component_code"]], on="comprehensive_code"
    )[[*keys, "component_code"]].drop_duplicates()
    expected_components["_pair_present"] = 1
    df = df.merge(
        expected_components.rename(columns={"component_code": "hcpcs"}),
        on=[*keys, "hcpcs"],
        how="left",
    )
    df["unbundle_component_no_mod"] = ((df["_pair_present"] == 1) & (df["mod_59"] == 0)).astype(int)

    return df


def beneficiary_history_features(df: pd.DataFrame) -> pd.DataFrame:
    """Backward-looking beneficiary utilization features."""
    df = df.sort_values(["bene_id", "service_date", "claim_id"]).reset_index(drop=True)

    # Same-day context (post-payment batch setting).
    df["bene_same_day_claims"] = (
        df.groupby(["bene_id", "service_date"])["claim_id"].transform("size") - 1
    )
    df["bene_provider_same_day_claims"] = (
        df.groupby(["bene_id", "provider_id", "service_date"])["claim_id"].transform("size") - 1
    )

    # Days since prior activity.
    df["days_since_bene_last_claim"] = (
        df.groupby("bene_id")["service_date"].diff().dt.days.fillna(SENTINEL_DAYS)
    )
    df = df.sort_values(["bene_id", "hcpcs", "service_date", "claim_id"])
    df["days_since_bene_same_hcpcs"] = (
        df.groupby(["bene_id", "hcpcs"])["service_date"].diff().dt.days.fillna(SENTINEL_DAYS)
    )
    df = df.sort_values(["bene_id", "service_date", "claim_id"]).reset_index(drop=True)

    # Strictly-prior 30-day rolling utilization (closed='left' excludes same-day).
    rolled = (
        df.groupby("bene_id")
        .rolling("30D", on="service_date", closed="left")["paid_amount"]
        .agg(["count", "sum"])
        .reset_index(drop=True)
    )
    df["bene_claims_30d"] = rolled["count"].fillna(0.0)
    df["bene_paid_30d"] = rolled["sum"].fillna(0.0).round(2)

    # Distinct providers seen before this claim (first-occurrence cumsum, shifted).
    first_pair = ~df.duplicated(subset=["bene_id", "provider_id"])
    df["bene_distinct_providers_prior"] = (
        first_pair.groupby(df["bene_id"]).cumsum().groupby(df["bene_id"]).shift(1).fillna(0.0)
    )
    return df
