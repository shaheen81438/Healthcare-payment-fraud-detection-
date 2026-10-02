"""Online feature assembly, using the same feature functions as the batch pipeline.

Online features match offline ones because this module reuses the ``src.features``
code against in-memory context stores loaded at startup:

* frozen peer stats (``models/feature_store``),
* the provider-month feature table (lagged, point-in-time),
* recent claim history per beneficiary (for duplicate and utilization features).

Unknown providers or beneficiaries fall back to cold-start defaults and set
``insufficient_provider_history`` so consumers can see the difference.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from app.schemas.claims import ClaimIn
from src.data.reference import load_reference_frames
from src.features.claim_features import (
    amount_and_coding_features,
    beneficiary_history_features,
)
from src.features.dup_similarity import duplicate_features
from src.features.peer_stats import load_peer_stats
from src.features.provider_features import PROVIDER_FEATURES
from src.features.splits import load_features_config

logger = logging.getLogger(__name__)

HISTORY_COLUMNS = [
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


class FeatureService:
    """Assembles model-ready features for API payloads."""

    def __init__(
        self,
        feature_store_dir: str = "models/feature_store",
        history_path: str = "data/processed/claims.parquet",
        provider_month_path: str = "data/processed/provider_month_features.parquet",
    ) -> None:
        self.cfg = load_features_config()
        self.ref = load_reference_frames()
        self.peer = load_peer_stats(Path(feature_store_dir))
        self.provider_month = self._load_provider_month(provider_month_path)
        self.history = self._load_history(history_path)
        logger.info(
            "feature_service_ready",
            extra={
                "history_rows": 0 if self.history is None else len(self.history),
                "provider_months": 0 if self.provider_month is None else len(self.provider_month),
            },
        )

    @staticmethod
    def _load_provider_month(path: str) -> pd.DataFrame | None:
        if not Path(path).exists():
            logger.warning("provider_month_missing", extra={"path": path})
            return None
        pm = pd.read_parquet(path)
        return pm.set_index(["provider_id", "month"]).sort_index()

    @staticmethod
    def _load_history(path: str) -> pd.DataFrame | None:
        if not Path(path).exists():
            logger.warning("claim_history_missing", extra={"path": path})
            return None
        hist = pd.read_parquet(path, columns=HISTORY_COLUMNS)
        return hist.sort_values(["bene_id", "service_date"]).set_index("bene_id", drop=False)

    # ------------------------------------------------------------------ public
    def build(self, claims: list[ClaimIn]) -> pd.DataFrame:
        """Return the feature frame (one row per payload claim, payload order)."""
        payload = pd.DataFrame([c.model_dump() for c in claims])
        for col in ["service_date", "service_end_date", "submit_date"]:
            payload[col] = pd.to_datetime(payload[col])
        payload["units"] = payload["units"].astype(int)

        # Peer-normalized amount/coding features need no history.
        df = amount_and_coding_features(payload, self.ref, self.peer)

        # Beneficiary history and duplicate features: run the batch functions over
        # the stored history for these benes plus the payload, then keep payload rows.
        combined = self._with_history(payload)
        bene_feats = beneficiary_history_features(combined)
        dup_feats = duplicate_features(combined, self.cfg["duplicates"])
        bene_cols = [
            "claim_id",
            "bene_same_day_claims",
            "bene_provider_same_day_claims",
            "days_since_bene_last_claim",
            "days_since_bene_same_hcpcs",
            "bene_claims_30d",
            "bene_paid_30d",
            "bene_distinct_providers_prior",
        ]
        df = df.merge(bene_feats[bene_cols], on="claim_id", how="left")
        df = df.merge(dup_feats, on="claim_id", how="left")

        # Provider-month lookup (lagged features, computed from prior months).
        df["month"] = df["service_date"].dt.to_period("M").astype(str)
        df = self._join_provider_month(df)
        return df.set_index(payload.index)

    # ------------------------------------------------------------------ internals
    def _with_history(self, payload: pd.DataFrame) -> pd.DataFrame:
        if self.history is None:
            return payload.copy()
        benes = payload["bene_id"].unique().tolist()
        present = [b for b in benes if b in self.history.index]
        slices = [self.history.loc[[b]] for b in present]
        history = (
            pd.concat(slices, ignore_index=True)
            if slices
            else pd.DataFrame(columns=HISTORY_COLUMNS)
        )
        # Payload claims replace any stored copy of themselves (idempotent rescoring).
        history = history[~history["claim_id"].isin(set(payload["claim_id"]))]
        return pd.concat([history, payload[HISTORY_COLUMNS]], ignore_index=True)

    def _join_provider_month(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in PROVIDER_FEATURES:
            df[col] = 0.0
        df["insufficient_provider_history"] = True
        if self.provider_month is None:
            return df
        idx = self.provider_month.index
        for i in df.index:
            key = (df.at[i, "provider_id"], df.at[i, "month"])
            if key not in idx:
                # Fall back to the latest month available for the provider.
                pid = df.at[i, "provider_id"]
                try:
                    available = self.provider_month.loc[pid]
                except KeyError:
                    continue
                prior = available[available.index <= df.at[i, "month"]]
                if prior.empty:
                    continue
                row = prior.iloc[-1]
            else:
                row = self.provider_month.loc[key]
            for col in PROVIDER_FEATURES:
                df.at[i, col] = float(row[col])
            df.at[i, "insufficient_provider_history"] = bool(row["prov_months_active"] < 2)
        return df
