"""Base synthetic claim simulation (before FWA pattern injection).

The simulator draws provider profiles, beneficiary panels, and 24 months of
claim activity calibrated to real CMS specialty/code/payment distributions
(``calibration.py``). It also adds benign confounders (corrected-claim
resubmissions, legitimately expensive claims, properly modified component
billing) so the detectors have to deal with realistic false positives.

Everything is driven by ``configs/data_generation.yaml`` and one seeded
``numpy`` Generator, so the same config and seed always give the same dataset.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from src.data.reference import hcpcs_category
from src.data.synthetic.calibration import SpecialtyCalibration

logger = logging.getLogger(__name__)

SEASONALITY = np.array([1.06, 1.04, 1.02, 1.0, 0.97, 0.95, 0.94, 0.95, 0.98, 1.01, 1.03, 1.05])
WEEKDAY_WEIGHTS = {0: 1.15, 1: 1.2, 2: 1.2, 3: 1.15, 4: 1.0, 5: 0.18, 6: 0.12}
FACILITY_POS = np.array(["21", "22", "23"])
FACILITY_POS_P = np.array([0.5, 0.3, 0.2])
OFFICE_POS = np.array(["11", "12"])
OFFICE_POS_P = np.array([0.92, 0.08])
AGE_BANDS = np.array(["65-69", "70-74", "75-79", "80-84", "85+"])
AGE_BAND_P = np.array([0.28, 0.26, 0.2, 0.14, 0.12])

CLAIM_COLUMNS = [
    "_rowid",
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
class Repertoire:
    """A provider's personal code mix with base amounts (specialty avg × provider effect)."""

    hcpcs: np.ndarray
    probs: np.ndarray
    base_submitted: np.ndarray
    base_allowed: np.ndarray
    base_paid: np.ndarray
    facility_share: np.ndarray
    category: np.ndarray


@dataclass
class GenerationContext:
    """Everything the pattern injectors need, beyond the claims frame itself."""

    cfg: dict[str, Any]
    rng: np.random.Generator
    providers: pd.DataFrame  # includes internal cols: amount_multiplier, monthly_base
    repertoires: dict[str, Repertoire]
    calibrations: dict[str, SpecialtyCalibration]
    ref: dict[str, pd.DataFrame]
    dx_probs: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]]
    panels: dict[str, tuple[np.ndarray, np.ndarray]]  # provider -> (bene indices, probs)
    bene_ids: np.ndarray
    next_rowid: int = 0
    stats: dict[str, Any] = field(default_factory=dict)

    def take_rowids(self, n: int) -> np.ndarray:
        ids = np.arange(self.next_rowid, self.next_rowid + n)
        self.next_rowid += n
        return ids


def load_generation_config(path: str | Path = "configs/data_generation.yaml") -> dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        cfg: dict[str, Any] = yaml.safe_load(fh)
    return cfg


# --------------------------------------------------------------------------
# Entity sampling
# --------------------------------------------------------------------------
def _sample_providers(
    cfg: dict[str, Any], calibrations: dict[str, SpecialtyCalibration], rng: np.random.Generator
) -> pd.DataFrame:
    weights = {s: w for s, w in cfg["specialty_weights"].items() if s in calibrations}
    specs = list(weights)
    probs = np.array([weights[s] for s in specs])
    probs = probs / probs.sum()
    n = int(cfg["n_providers"])
    chosen = rng.choice(specs, size=n, p=probs)
    lo, hi = cfg["annual_claims_clip"]
    rows = []
    for i, spec in enumerate(chosen):
        cal = calibrations[spec]
        state = rng.choice(cal.states, p=np.array(cal.state_probs))
        annual = float(
            np.clip(
                rng.lognormal(cal.log_volume_mu, cal.log_volume_sigma) * cfg["volume_scale"],
                lo,
                hi,
            )
        )
        rows.append(
            {
                "provider_id": f"PRV-{i + 1:05d}",
                "specialty": spec,
                "state": state,
                "amount_multiplier": float(rng.lognormal(0.0, cfg["provider_multiplier_sigma"])),
                "annual_claims": annual,
                "trend_slope": float(rng.normal(0.03, 0.12)),
            }
        )
    return pd.DataFrame(rows)


def _build_repertoires(
    providers: pd.DataFrame,
    calibrations: dict[str, SpecialtyCalibration],
    rng: np.random.Generator,
) -> dict[str, Repertoire]:
    reps: dict[str, Repertoire] = {}
    for row in providers.itertuples():
        cal = calibrations[row.specialty]
        codes = cal.codes
        k = min(len(codes), int(rng.integers(12, 26)))
        w = codes["weight"].to_numpy(dtype=float)
        idx = rng.choice(len(codes), size=k, replace=False, p=w / w.sum())
        sub = codes["avg_submitted"].to_numpy()[idx] * row.amount_multiplier
        alw = codes["avg_allowed"].to_numpy()[idx] * row.amount_multiplier
        paid = codes["avg_paid"].to_numpy()[idx] * row.amount_multiplier
        cw = w[idx]
        reps[row.provider_id] = Repertoire(
            hcpcs=codes["hcpcs"].to_numpy()[idx].astype(str),
            probs=cw / cw.sum(),
            base_submitted=sub,
            base_allowed=np.minimum(alw, sub),
            base_paid=np.minimum(paid, alw),
            facility_share=codes["facility_share"].to_numpy(dtype=float)[idx],
            category=np.array([hcpcs_category(c) for c in codes["hcpcs"].to_numpy()[idx]]),
        )
    return reps


def _sample_beneficiaries(
    cfg: dict[str, Any], providers: pd.DataFrame, rng: np.random.Generator
) -> pd.DataFrame:
    n = int(cfg["n_beneficiaries"])
    states = providers["state"].to_numpy()
    return pd.DataFrame(
        {
            "bene_id": [f"BEN-{i + 1:06d}" for i in range(n)],
            "age_band": rng.choice(AGE_BANDS, size=n, p=AGE_BAND_P),
            "sex": rng.choice(np.array(["F", "M"]), size=n, p=np.array([0.55, 0.45])),
            "state": rng.choice(states, size=n),
        }
    )


def _build_panels(
    providers: pd.DataFrame, n_benes: int, rng: np.random.Generator
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    panels: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for row in providers.itertuples():
        size = int(np.clip(row.annual_claims / 3, 20, 400))
        members = rng.choice(n_benes, size=size, replace=False)
        ranks = np.arange(1, size + 1, dtype=float)
        w = ranks**-0.7
        panels[row.provider_id] = (members, w / w.sum())
    return panels


def _dx_probability_table(
    ref: dict[str, pd.DataFrame],
) -> dict[str, dict[str, tuple[np.ndarray, np.ndarray]]]:
    """Per (category) -> (dx_groups, base probs); specialty affinity applied at lookup."""
    compat = ref["dx_procedure_compat"].copy()
    compat["compat"] = compat["compat"].astype(float)
    table: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = {}
    for cat, grp in compat.groupby("hcpcs_category"):
        table[str(cat)] = {
            "__base__": (grp["dx_group"].to_numpy().astype(str), grp["compat"].to_numpy())
        }
    return table  # affinity handled in _dx_probs_for


def _dx_probs_for(
    cat_table: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]],
    dx_groups: pd.DataFrame,
    cache: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]],
    specialty: str,
    category: str,
) -> tuple[np.ndarray, np.ndarray]:
    key = (specialty, category)
    if key in cache:
        return cache[key]
    groups, base = cat_table.get(category, cat_table["OTHER"])["__base__"]
    affinity = np.ones_like(base)
    typical = dx_groups.set_index("dx_group")["typical_specialties"]
    for i, g in enumerate(groups):
        specs = str(typical.get(g, ""))
        if specialty in specs.split("|"):
            affinity[i] = 3.0
    w = base * affinity
    w = np.where(w <= 0, 1e-6, w)
    probs = w / w.sum()
    cache[key] = (groups, probs)
    return cache[key]


# --------------------------------------------------------------------------
# Base claim simulation
# --------------------------------------------------------------------------
def _month_starts(cfg: dict[str, Any]) -> np.ndarray:
    start = np.datetime64(cfg["start_month"], "M")
    return (start + np.arange(int(cfg["n_months"]))).astype("datetime64[D]")


def _sample_days(month_start: np.datetime64, n: int, rng: np.random.Generator) -> np.ndarray:
    offsets = np.arange(28)
    days = (month_start + offsets).astype("datetime64[D]")
    # Epoch day 0 (1970-01-01) was a Thursday, which is weekday index 3 with Monday=0.
    weekdays = ((days.astype("datetime64[D]").view("int64") + 3) % 7).astype(int)
    w = np.array([WEEKDAY_WEIGHTS[int(d)] for d in weekdays])
    return days[rng.choice(28, size=n, p=w / w.sum())]


def _units_for(category: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    units = np.ones(len(category), dtype=int)
    therapy = category == "THERAPY"
    units[therapy] = rng.integers(1, 5, size=int(therapy.sum()))
    drug = category == "DRUG"
    units[drug] = rng.integers(1, 7, size=int(drug.sum()))
    lab = category == "LAB"
    units[lab] = rng.integers(1, 3, size=int(lab.sum()))
    return units


def _modifiers_for(category: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    mods = np.full(len(category), "", dtype=object)
    r = rng.random(len(category))
    em = np.isin(category, ["EM_OFFICE", "EM_FACILITY"])
    mods[em & (r < 0.04)] = "25"
    th = category == "THERAPY"
    mods[th & (r < 0.02)] = "59"
    proc = category == "PROCEDURE"
    mods[proc & (r < 0.03)] = "LT"
    mods[proc & (r >= 0.03) & (r < 0.06)] = "RT"
    return mods


def simulate_base_claims(ctx: GenerationContext) -> pd.DataFrame:
    """Simulate benign claim activity for every provider over the configured window."""
    cfg, rng = ctx.cfg, ctx.rng
    month_starts = _month_starts(cfg)
    n_months = len(month_starts)
    payer_share = float(cfg["payer_share"])
    sigma = float(cfg["amount_noise_sigma"])
    cat_table = _dx_probability_table(ctx.ref)
    dx_cache: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}
    frames: list[pd.DataFrame] = []

    for prow in ctx.providers.itertuples():
        rep = ctx.repertoires[prow.provider_id]
        members, member_p = ctx.panels[prow.provider_id]
        monthly_base = prow.annual_claims / 12.0
        trend = np.clip(1.0 + prow.trend_slope * (np.arange(n_months) / 12.0), 0.5, 1.7)
        lam = monthly_base * SEASONALITY[np.arange(n_months) % 12] * trend
        counts = rng.poisson(lam)
        total = int(counts.sum())
        if total == 0:
            continue

        month_idx = np.repeat(np.arange(n_months), counts)
        service_dates = np.concatenate(
            [_sample_days(month_starts[m], int(c), rng) for m, c in enumerate(counts) if c > 0]
        )
        code_idx = rng.choice(len(rep.hcpcs), size=total, p=rep.probs)
        bene_idx = members[rng.choice(len(members), size=total, p=member_p)]
        category = rep.category[code_idx]
        units = _units_for(category, rng)

        noise = rng.lognormal(0.0, sigma, size=total)
        submitted = rep.base_submitted[code_idx] * noise * units
        allowed = np.minimum(rep.base_allowed[code_idx] * noise * units, submitted)
        paid = allowed * payer_share * rng.uniform(0.985, 1.0, size=total)

        if prow.specialty == "Clinical Laboratory":
            pos = np.full(total, "81", dtype=object)
        else:
            is_fac = rng.random(total) < rep.facility_share[code_idx]
            pos = np.where(
                is_fac,
                FACILITY_POS[rng.choice(3, size=total, p=FACILITY_POS_P)],
                OFFICE_POS[rng.choice(2, size=total, p=OFFICE_POS_P)],
            ).astype(object)

        dx = np.empty(total, dtype=object)
        for cat in np.unique(category):
            groups, probs = _dx_probs_for(
                cat_table, ctx.ref["dx_groups"], dx_cache, prow.specialty, str(cat)
            )
            mask = category == cat
            dx[mask] = groups[rng.choice(len(groups), size=int(mask.sum()), p=probs)]

        stay = np.where(
            (category == "EM_FACILITY") & np.isin(pos.astype(str), ["21", "31"]),
            rng.integers(0, 4, size=total),
            0,
        )
        submit_lag = np.clip(rng.lognormal(np.log(9), 0.6, size=total), 1, 60).astype(int)

        frames.append(
            pd.DataFrame(
                {
                    "_rowid": ctx.take_rowids(total),
                    "bene_id": ctx.bene_ids[bene_idx],
                    "provider_id": prow.provider_id,
                    "specialty": prow.specialty,
                    "hcpcs": rep.hcpcs[code_idx],
                    "modifier": _modifiers_for(category, rng),
                    "units": units,
                    "dx_group": dx,
                    "pos": pos,
                    "service_date": service_dates,
                    "service_end_date": service_dates + stay.astype("timedelta64[D]"),
                    "submit_date": service_dates + submit_lag.astype("timedelta64[D]"),
                    "submitted_amount": np.round(submitted, 2),
                    "allowed_amount": np.round(allowed, 2),
                    "paid_amount": np.round(paid, 2),
                }
            )
        )
        del month_idx  # month bookkeeping not persisted

    claims = pd.concat(frames, ignore_index=True)[CLAIM_COLUMNS]
    logger.info("base_claims_simulated", extra={"rows": len(claims)})
    return claims


# --------------------------------------------------------------------------
# Benign confounders
# --------------------------------------------------------------------------
def add_confounders(claims: pd.DataFrame, ctx: GenerationContext) -> pd.DataFrame:
    """Label-0 lookalikes: corrected resubmissions, legit high-cost claims, modified components."""
    cfg, rng = ctx.cfg["confounders"], ctx.rng
    n = len(claims)

    # Corrected resubmissions (look like near-duplicates).
    k = int(n * float(cfg["corrected_resubmission_rate"]))
    idx = rng.choice(n, size=k, replace=False)
    resub = claims.iloc[idx].copy()
    resub["_rowid"] = ctx.take_rowids(k)
    resub["submit_date"] = resub["service_date"] + np.array(
        rng.integers(7, 22, size=k), dtype="timedelta64[D]"
    )
    jitter = rng.uniform(0.99, 1.01, size=k)
    for col in ["submitted_amount", "allowed_amount", "paid_amount"]:
        resub[col] = np.round(resub[col].to_numpy() * jitter, 2)
    ctx.stats["confounder_resubmissions"] = k

    # Legitimately expensive claims.
    k2 = int(n * float(cfg["legit_high_cost_rate"]))
    idx2 = rng.choice(n, size=k2, replace=False)
    mult = rng.uniform(2.0, 3.0, size=k2)
    for col in ["submitted_amount", "allowed_amount", "paid_amount"]:
        claims.loc[claims.index[idx2], col] = np.round(claims.iloc[idx2][col].to_numpy() * mult, 2)
    ctx.stats["confounder_high_cost"] = k2

    # Properly modified component billing (modifier 59 present, so the edits allow it).
    ptp = ctx.ref["ptp_pairs"]
    comp_map = dict(zip(ptp["comprehensive_code"], ptp["component_code"], strict=True))
    ratio_map = dict(zip(ptp["comprehensive_code"], ptp["component_amount_ratio"], strict=True))
    eligible = claims[claims["hcpcs"].isin(comp_map)].index.to_numpy()
    k3 = min(len(eligible), int(n * float(cfg["legit_unbundle_with_modifier_rate"])))
    if k3 > 0:
        idx3 = rng.choice(eligible, size=k3, replace=False)
        comp = claims.loc[idx3].copy()
        comp["_rowid"] = ctx.take_rowids(k3)
        comp["hcpcs"] = comp["hcpcs"].map(comp_map)
        ratios = claims.loc[idx3, "hcpcs"].map(ratio_map).astype(float).to_numpy()
        for col in ["submitted_amount", "allowed_amount", "paid_amount"]:
            comp[col] = np.round(comp[col].to_numpy() * ratios, 2)
        comp["modifier"] = "59"
        comp["units"] = 1
        claims = pd.concat([claims, resub, comp], ignore_index=True)
    else:  # pragma: no cover
        claims = pd.concat([claims, resub], ignore_index=True)
    ctx.stats["confounder_modified_components"] = k3
    return claims
