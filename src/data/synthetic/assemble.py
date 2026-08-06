"""End-to-end dataset assembly: simulate → inject patterns → label → add raw defects."""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.data.synthetic.calibration import build_calibration
from src.data.synthetic.generate import (
    GenerationContext,
    _build_panels,
    _build_repertoires,
    _sample_beneficiaries,
    _sample_providers,
    add_confounders,
    simulate_base_claims,
)
from src.data.synthetic.patterns import inject_all

logger = logging.getLogger(__name__)

PUBLIC_CLAIM_COLUMNS = [
    "claim_id",
    "bene_id",
    "provider_id",
    "specialty",
    "hcpcs",
    "modifier",
    "units",
    "dx_group",
    "pos",
    "service_date",
    "service_end_date",
    "submit_date",
    "submitted_amount",
    "allowed_amount",
    "paid_amount",
]


@dataclass
class DatasetBundle:
    """Everything the pipeline persists, with labels kept separate from claims."""

    claims: pd.DataFrame  # clean ground-truth claims (no label columns)
    raw_claims: pd.DataFrame  # claims + injected data-quality defects (pipeline input)
    providers: pd.DataFrame
    beneficiaries: pd.DataFrame
    labels: pd.DataFrame  # claim_id, is_improper, patterns, improper_amount
    stats: dict[str, Any]
    dataset_version: str


def build_dataset(
    cfg: dict[str, Any], cms_df: pd.DataFrame, ref: dict[str, pd.DataFrame]
) -> DatasetBundle:
    """Build the full synthetic dataset deterministically from config + seed."""
    rng = np.random.default_rng(int(cfg["seed"]))
    calibrations = build_calibration(cms_df)
    providers = _sample_providers(cfg, calibrations, rng)
    repertoires = _build_repertoires(providers, calibrations, rng)
    beneficiaries = _sample_beneficiaries(cfg, providers, rng)
    panels = _build_panels(providers, len(beneficiaries), rng)

    ctx = GenerationContext(
        cfg=cfg,
        rng=rng,
        providers=providers,
        repertoires=repertoires,
        calibrations=calibrations,
        ref=ref,
        dx_probs={},
        panels=panels,
        bene_ids=beneficiaries["bene_id"].to_numpy(),
    )

    claims = simulate_base_claims(ctx)
    claims = add_confounders(claims, ctx)
    claims, records = inject_all(claims, ctx)

    # Finalize claim IDs in service-date order.
    claims = claims.sort_values(["service_date", "provider_id", "_rowid"]).reset_index(drop=True)
    claims["claim_id"] = "CLM-" + (claims.index + 1).astype(str).str.zfill(8)

    labels = _build_labels(claims, records)
    claims_public = claims[PUBLIC_CLAIM_COLUMNS].copy()
    raw_claims = inject_defects(claims_public, cfg["defects"], rng, ctx.stats)

    providers_public = providers[["provider_id", "specialty", "state"]].copy()
    version = _dataset_version(cfg, len(claims_public))
    ctx.stats["rows_claims"] = len(claims_public)
    ctx.stats["rows_raw"] = len(raw_claims)
    ctx.stats["positive_rate"] = float(labels["is_improper"].mean())
    ctx.stats["improper_amount_total"] = float(labels["improper_amount"].sum())
    logger.info(
        "dataset_built",
        extra={
            "version": version,
            "claims": len(claims_public),
            "positive_rate": round(ctx.stats["positive_rate"], 4),
        },
    )
    return DatasetBundle(
        claims=claims_public,
        raw_claims=raw_claims,
        providers=providers_public,
        beneficiaries=beneficiaries,
        labels=labels,
        stats=ctx.stats,
        dataset_version=version,
    )


def _build_labels(claims: pd.DataFrame, records: list[tuple[int, str, float]]) -> pd.DataFrame:
    """Aggregate (rowid, pattern, improper) records into one labels row per claim."""
    rowid_to_claim = dict(zip(claims["_rowid"], claims["claim_id"], strict=True))
    paid_by_claim = dict(zip(claims["claim_id"], claims["paid_amount"], strict=True))
    if records:
        rec = pd.DataFrame(records, columns=["_rowid", "pattern", "improper_amount"])
        rec["claim_id"] = rec["_rowid"].map(rowid_to_claim)
        agg = (
            rec.groupby("claim_id")
            .agg(
                patterns=("pattern", lambda s: ";".join(sorted(set(s)))),
                improper_amount=("improper_amount", "sum"),
            )
            .reset_index()
        )
    else:  # pragma: no cover - only for degenerate configs
        agg = pd.DataFrame(columns=["patterns", "improper_amount"])

    labels = pd.DataFrame({"claim_id": claims["claim_id"]})
    labels = labels.merge(agg, on="claim_id", how="left")
    labels["is_improper"] = labels["patterns"].notna()
    labels["patterns"] = labels["patterns"].fillna("")
    labels["improper_amount"] = labels["improper_amount"].fillna(0.0)
    # An improper amount can never exceed what was actually paid on the claim.
    paid = labels["claim_id"].map(paid_by_claim).astype(float)
    labels["improper_amount"] = np.round(np.minimum(labels["improper_amount"], paid), 2)
    return labels


def inject_defects(
    claims: pd.DataFrame,
    dcfg: dict[str, Any],
    rng: np.random.Generator,
    stats: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Inject data-quality defects into a copy of the claims (the 'raw' pipeline input).

    Defect classes are applied to disjoint row sets so each remains measurable:
    missing POS, `$`-formatted amount strings, impossible date order, same-ID
    ingestion duplicates, negative amounts, truncated HCPCS codes.
    """
    raw = claims.copy()
    n = len(raw)
    perm = rng.permutation(n)
    cursor = 0

    def take(rate: float) -> np.ndarray:
        nonlocal cursor
        k = int(n * rate)
        sel = perm[cursor : cursor + k]
        cursor += k
        return raw.index.to_numpy()[sel]

    idx_pos = take(float(dcfg["missing_pos_rate"]))
    raw.loc[idx_pos, "pos"] = ""

    idx_str = take(float(dcfg["amount_as_string_rate"]))
    raw["paid_amount"] = raw["paid_amount"].astype(object)
    raw.loc[idx_str, "paid_amount"] = [
        f"${v:,.2f}" for v in claims.loc[idx_str, "paid_amount"].astype(float)
    ]

    idx_date = take(float(dcfg["impossible_date_rate"]))
    raw.loc[idx_date, "submit_date"] = raw.loc[idx_date, "service_date"] - np.array(
        rng.integers(1, 11, size=len(idx_date)), dtype="timedelta64[D]"
    )

    idx_neg = take(float(dcfg["negative_amount_rate"]))
    raw.loc[idx_neg, "paid_amount"] = [
        -abs(float(v)) for v in claims.loc[idx_neg, "paid_amount"].astype(float)
    ]

    idx_hcpcs = take(float(dcfg["invalid_hcpcs_rate"]))
    raw.loc[idx_hcpcs, "hcpcs"] = raw.loc[idx_hcpcs, "hcpcs"].astype(str).str[:4]

    k_dup = int(n * float(dcfg["ingestion_duplicate_rate"]))
    dup_rows = raw.loc[raw.index.to_numpy()[perm[cursor : cursor + k_dup]]].copy()
    raw = pd.concat([raw, dup_rows], ignore_index=True)

    raw = raw.sample(frac=1.0, random_state=int(rng.integers(0, 2**31))).reset_index(drop=True)
    if stats is not None:
        stats["defects"] = {
            "missing_pos": len(idx_pos),
            "amount_as_string": len(idx_str),
            "impossible_dates": len(idx_date),
            "negative_amounts": len(idx_neg),
            "invalid_hcpcs": len(idx_hcpcs),
            "ingestion_duplicates": k_dup,
        }
    return raw


def _dataset_version(cfg: dict[str, Any], n_rows: int) -> str:
    payload = json.dumps(cfg, sort_keys=True, default=str) + f"|rows={n_rows}"
    return "ds-" + hashlib.sha256(payload.encode()).hexdigest()[:12]
