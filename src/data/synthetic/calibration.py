"""Calibration of the synthetic generator against real CMS distributions.

For each provider specialty present in the CMS snapshot we derive:
* the HCPCS code mix with service-volume weights and payment levels,
* the facility/office split per code,
* a state distribution,
* a rough per-provider annual volume distribution (lognormal fit, then scaled).

Only aggregates are used, so no synthetic provider copies any single real NPI.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class SpecialtyCalibration:
    """Empirical distributions for one provider specialty."""

    specialty: str
    codes: (
        pd.DataFrame
    )  # hcpcs, hcpcs_desc, drug_ind, facility_share, avg_submitted, avg_allowed, avg_paid, weight
    states: list[str] = field(default_factory=list)
    state_probs: list[float] = field(default_factory=list)
    log_volume_mu: float = 7.5
    log_volume_sigma: float = 0.8


def build_calibration(cms_df: pd.DataFrame) -> dict[str, SpecialtyCalibration]:
    """Build per-specialty calibrations from normalized CMS provider-service rows."""
    calibrations: dict[str, SpecialtyCalibration] = {}
    for specialty, group in cms_df.groupby("provider_type"):
        # Aggregate per code across POS rows: volume-weighted amounts and facility share.
        g = group.copy()
        g["svc"] = g["total_services"].clip(lower=1.0)
        agg = (
            g.assign(
                w_sub=g["avg_submitted"] * g["svc"],
                w_alw=g["avg_allowed"] * g["svc"],
                w_paid=g["avg_paid"] * g["svc"],
                fac_svc=np.where(g["place_of_service"] == "F", g["svc"], 0.0),
            )
            .groupby(["hcpcs", "hcpcs_desc", "drug_ind"], as_index=False)
            .agg(
                weight=("svc", "sum"),
                w_sub=("w_sub", "sum"),
                w_alw=("w_alw", "sum"),
                w_paid=("w_paid", "sum"),
                fac_svc=("fac_svc", "sum"),
            )
        )
        agg["avg_submitted"] = agg["w_sub"] / agg["weight"]
        agg["avg_allowed"] = agg["w_alw"] / agg["weight"]
        agg["avg_paid"] = agg["w_paid"] / agg["weight"]
        agg["facility_share"] = agg["fac_svc"] / agg["weight"]
        codes = agg[
            [
                "hcpcs",
                "hcpcs_desc",
                "drug_ind",
                "facility_share",
                "avg_submitted",
                "avg_allowed",
                "avg_paid",
                "weight",
            ]
        ].reset_index(drop=True)
        # Keep paid <= allowed <= submitted in calibration space.
        codes["avg_allowed"] = np.minimum(codes["avg_allowed"], codes["avg_submitted"])
        codes["avg_paid"] = np.minimum(codes["avg_paid"], codes["avg_allowed"])

        state_counts = group.groupby("state")["npi"].nunique().sort_values(ascending=False)
        states = state_counts.index.tolist()
        probs = (state_counts / state_counts.sum()).tolist()

        # Per-NPI service totals in the snapshot give a (truncated) view of annual volume.
        npi_totals = group.groupby("npi")["total_services"].sum().to_numpy(dtype=float)
        npi_totals = npi_totals[npi_totals > 0]
        if len(npi_totals) >= 5:
            logs = np.log(npi_totals)
            mu, sigma = float(np.mean(logs)), float(np.clip(np.std(logs), 0.4, 1.2))
        else:  # pragma: no cover - tiny snapshot edge case
            mu, sigma = 7.5, 0.8

        calibrations[str(specialty)] = SpecialtyCalibration(
            specialty=str(specialty),
            codes=codes,
            states=states,
            state_probs=probs,
            log_volume_mu=mu,
            log_volume_sigma=sigma,
        )
    return calibrations
