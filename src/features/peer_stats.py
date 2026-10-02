"""Peer-group statistics, fit on the TRAIN window only and frozen as artifacts.

Serving (batch scoring and the API) reads these frozen tables. Peer norms are
never recomputed on the data being scored, because that would leak the future
into the past and move the benchmarks when the data drifts.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

PEER_STATS_TABLES = [
    "code_stats",
    "spec_code_stats",
    "pos_stats",
    "em_spec_stats",
    "provider_month_peer",
]


def fit_peer_stats(
    train_claims: pd.DataFrame, em_levels: pd.DataFrame, cfg: dict[str, Any]
) -> dict[str, pd.DataFrame]:
    """Compute frozen peer statistics from train-window claims."""
    df = train_claims.copy()
    df["paid_per_unit"] = df["paid_amount"] / df["units"].clip(lower=1)

    code_stats = (
        df.groupby("hcpcs")
        .agg(
            code_paid_unit_median=("paid_per_unit", "median"),
            code_paid_unit_p95=("paid_per_unit", lambda s: s.quantile(0.95)),
            code_units_p95=("units", lambda s: s.quantile(0.95)),
            code_units_p99=("units", lambda s: s.quantile(0.99)),
            code_n=("paid_per_unit", "size"),
        )
        .reset_index()
    )
    global_median = float(df["paid_per_unit"].median())
    low_support = code_stats["code_n"] < int(cfg["peer_stats"]["min_code_n"])
    code_stats.loc[low_support, "code_paid_unit_median"] = global_median

    spec_code = (
        df.groupby(["specialty", "hcpcs"])
        .agg(
            spec_code_paid_unit_median=("paid_per_unit", "median"),
            spec_code_n=("paid_per_unit", "size"),
        )
        .reset_index()
    )
    spec_code = spec_code[spec_code["spec_code_n"] >= int(cfg["peer_stats"]["min_spec_code_n"])]

    pos_stats = (
        df.assign(office=df["pos"].isin(["11", "12"]).astype(float))
        .groupby("hcpcs")
        .agg(code_office_share=("office", "mean"), pos_n=("office", "size"))
        .reset_index()
    )

    em_map = dict(zip(em_levels["hcpcs"], em_levels["level"].astype(int), strict=True))
    em = df[df["hcpcs"].isin(em_map)].copy()
    em["em_level"] = em["hcpcs"].map(em_map)
    em_spec_stats = (
        em.groupby("specialty").agg(spec_em_level_mean=("em_level", "mean")).reset_index()
    )

    provider_months = (
        df.assign(month=df["service_date"].dt.to_period("M").astype(str))
        .groupby(["specialty", "provider_id", "month"])
        .agg(claims=("claim_id", "size"), paid=("paid_amount", "sum"))
        .reset_index()
    )
    provider_month_peer = (
        provider_months.groupby("specialty")
        .agg(
            peer_monthly_claims_median=("claims", "median"),
            peer_monthly_paid_median=("paid", "median"),
            peer_monthly_paid_p95=("paid", lambda s: s.quantile(0.95)),
        )
        .reset_index()
    )

    stats = {
        "code_stats": code_stats,
        "spec_code_stats": spec_code,
        "pos_stats": pos_stats,
        "em_spec_stats": em_spec_stats,
        "provider_month_peer": provider_month_peer,
    }
    logger.info(
        "peer_stats_fit",
        extra={name: len(frame) for name, frame in stats.items()},
    )
    return stats


def save_peer_stats(stats: dict[str, pd.DataFrame], out_dir: str | Path) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, frame in stats.items():
        frame.to_parquet(out / f"{name}.parquet", index=False)


def load_peer_stats(in_dir: str | Path) -> dict[str, pd.DataFrame]:
    path = Path(in_dir)
    return {name: pd.read_parquet(path / f"{name}.parquet") for name in PEER_STATS_TABLES}


def peer_stats_fingerprint(stats: dict[str, pd.DataFrame]) -> str:
    """Return a cheap fingerprint so tests can check that the artifacts were not refit."""
    parts = []
    for name in PEER_STATS_TABLES:
        frame = stats[name]
        numeric = frame.select_dtypes(include=[np.number])
        parts.append(f"{name}:{len(frame)}:{round(float(numeric.sum().sum()), 4)}")
    return "|".join(parts)
