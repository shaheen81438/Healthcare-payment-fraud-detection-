"""FWA pattern injectors (F01–F12).

Each injector mutates or appends claims and returns label records
``(_rowid, pattern, improper_amount)``. Injection changes ONLY the claim fields a
real bad actor could influence — no marker columns, no systematic artifacts
beyond the behavior itself — so labels are recoverable only through the
patterns, never through generation shortcuts. Prevalences and parameters come
from ``configs/data_generation.yaml``; assumptions are documented in
``docs/labeling.md``.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from src.data.synthetic.generate import GenerationContext

logger = logging.getLogger(__name__)

LabelRecord = tuple[int, str, float]
AMOUNT_COLS = ["submitted_amount", "allowed_amount", "paid_amount"]

UPCODE_MAP = {"99212": ["99213", "99214"], "99213": ["99214", "99215"]}


def _pick_providers(ctx: GenerationContext, share: float) -> np.ndarray:
    ids = ctx.providers["provider_id"].to_numpy()
    k = max(1, int(len(ids) * share))
    return ctx.rng.choice(ids, size=k, replace=False)


def _scale_amounts(df: pd.DataFrame, idx: pd.Index, mult: np.ndarray | float) -> None:
    for col in AMOUNT_COLS:
        df.loc[idx, col] = np.round(df.loc[idx, col].to_numpy(dtype=float) * mult, 2)


def _copy_rows(claims: pd.DataFrame, idx: np.ndarray, ctx: GenerationContext) -> pd.DataFrame:
    rows = claims.loc[idx].copy().reset_index(drop=True)
    rows["_rowid"] = ctx.take_rowids(len(rows))
    return rows


# ---------------------------------------------------------------- F01 / F02
def f01_exact_duplicate(
    claims: pd.DataFrame, ctx: GenerationContext, pcfg: dict[str, Any]
) -> tuple[pd.DataFrame, list[LabelRecord]]:
    providers = _pick_providers(ctx, pcfg["provider_share"])
    eligible = claims.index[claims["provider_id"].isin(providers)].to_numpy()
    k = min(len(eligible), int(len(claims) * pcfg["claim_rate"]))
    idx = ctx.rng.choice(eligible, size=k, replace=False)
    dups = _copy_rows(claims, idx, ctx)
    dups["submit_date"] = dups["service_date"] + np.array(
        ctx.rng.integers(3, 31, size=k), dtype="timedelta64[D]"
    )
    records = [
        (int(r), "F01_exact_duplicate", float(p))
        for r, p in zip(dups["_rowid"], dups["paid_amount"], strict=True)
    ]
    return pd.concat([claims, dups], ignore_index=True), records


def f02_near_duplicate(
    claims: pd.DataFrame, ctx: GenerationContext, pcfg: dict[str, Any]
) -> tuple[pd.DataFrame, list[LabelRecord]]:
    rng = ctx.rng
    providers = _pick_providers(ctx, pcfg["provider_share"])
    eligible = claims.index[claims["provider_id"].isin(providers)].to_numpy()
    k = min(len(eligible), int(len(claims) * pcfg["claim_rate"]))
    idx = rng.choice(eligible, size=k, replace=False)
    dups = _copy_rows(claims, idx, ctx)
    dups["submit_date"] = dups["service_date"] + np.array(
        rng.integers(5, 40, size=k), dtype="timedelta64[D]"
    )
    jitter = rng.uniform(0.98, 1.02, size=k)
    for col in AMOUNT_COLS:
        dups[col] = np.round(dups[col].to_numpy(dtype=float) * jitter, 2)
    swap_dx = rng.random(k) < 0.5
    all_groups = ctx.ref["dx_groups"]["dx_group"].to_numpy()
    dups.loc[dups.index[swap_dx], "dx_group"] = rng.choice(all_groups, size=int(swap_dx.sum()))
    add_mod = (rng.random(k) < 0.3) & (dups["modifier"] == "").to_numpy()
    dups.loc[dups.index[add_mod], "modifier"] = "59"
    records = [
        (int(r), "F02_near_duplicate", float(p))
        for r, p in zip(dups["_rowid"], dups["paid_amount"], strict=True)
    ]
    return pd.concat([claims, dups], ignore_index=True), records


# ---------------------------------------------------------------- F03
def f03_rapid_repeat(
    claims: pd.DataFrame, ctx: GenerationContext, pcfg: dict[str, Any]
) -> tuple[pd.DataFrame, list[LabelRecord]]:
    rng = ctx.rng
    providers = _pick_providers(ctx, pcfg["provider_share"])
    records: list[LabelRecord] = []
    new_frames: list[pd.DataFrame] = []
    repeat_ok = claims["hcpcs"].astype(str).str.match(r"^(99|97|1[0-7])")
    for pid in providers:
        pool = claims.index[(claims["provider_id"] == pid) & repeat_ok].to_numpy()
        if len(pool) == 0:
            continue
        n_episodes = int(rng.integers(pcfg["n_episodes"][0], pcfg["n_episodes"][1] + 1))
        seeds = rng.choice(pool, size=min(n_episodes, len(pool)), replace=False)
        for seed in seeds:
            n_rep = int(rng.integers(pcfg["n_repeats"][0], pcfg["n_repeats"][1] + 1))
            reps = _copy_rows(claims, np.repeat(seed, n_rep), ctx)
            gaps = np.cumsum(rng.integers(1, 6, size=n_rep))
            reps["service_date"] = reps["service_date"] + gaps.astype("timedelta64[D]")
            reps["service_end_date"] = reps["service_date"]
            reps["submit_date"] = reps["service_date"] + np.array(
                rng.integers(2, 15, size=n_rep), dtype="timedelta64[D]"
            )
            jitter = rng.uniform(0.99, 1.01, size=n_rep)
            for col in AMOUNT_COLS:
                reps[col] = np.round(reps[col].to_numpy(dtype=float) * jitter, 2)
            new_frames.append(reps)
            records += [
                (int(r), "F03_rapid_repeat", float(p))
                for r, p in zip(reps["_rowid"], reps["paid_amount"], strict=True)
            ]
    if new_frames:
        claims = pd.concat([claims, *new_frames], ignore_index=True)
    return claims, records


# ---------------------------------------------------------------- F04 / F05
def f04_excessive_units(
    claims: pd.DataFrame, ctx: GenerationContext, pcfg: dict[str, Any]
) -> tuple[pd.DataFrame, list[LabelRecord]]:
    rng = ctx.rng
    providers = _pick_providers(ctx, pcfg["provider_share"])
    unit_scalable = claims["hcpcs"].astype(str).str.match(r"^(97|J|8)")
    eligible = claims.index[claims["provider_id"].isin(providers) & unit_scalable].to_numpy()
    k = min(len(eligible), int(len(claims) * pcfg["claim_rate"]))
    if k == 0:
        return claims, []
    idx = rng.choice(eligible, size=k, replace=False)
    mult = rng.integers(pcfg["unit_multiplier"][0], pcfg["unit_multiplier"][1] + 1, size=k)
    old_paid = claims.loc[idx, "paid_amount"].to_numpy(dtype=float)
    claims.loc[idx, "units"] = claims.loc[idx, "units"].to_numpy() * mult
    _scale_amounts(claims, pd.Index(idx), mult.astype(float))
    new_paid = claims.loc[idx, "paid_amount"].to_numpy(dtype=float)
    records = [
        (int(claims.loc[i, "_rowid"]), "F04_excessive_units", float(np_ - o))
        for i, np_, o in zip(idx, new_paid, old_paid, strict=True)
    ]
    return claims, records


def f05_amount_inflation(
    claims: pd.DataFrame, ctx: GenerationContext, pcfg: dict[str, Any]
) -> tuple[pd.DataFrame, list[LabelRecord]]:
    rng = ctx.rng
    providers = _pick_providers(ctx, pcfg["provider_share"])
    records: list[LabelRecord] = []
    for pid in providers:
        pool = claims.index[claims["provider_id"] == pid].to_numpy()
        if len(pool) == 0:
            continue
        affected = rng.choice(pool, size=max(1, int(len(pool) * 0.4)), replace=False)
        m = float(rng.uniform(pcfg["multiplier"][0], pcfg["multiplier"][1]))
        _scale_amounts(claims, pd.Index(affected), m)
        paid = claims.loc[affected, "paid_amount"].to_numpy(dtype=float)
        rowids = claims.loc[affected, "_rowid"].to_numpy()
        records += [
            (int(r), "F05_amount_inflation", float(p - p / m))
            for r, p in zip(rowids, paid, strict=True)
        ]
    return claims, records


# ---------------------------------------------------------------- F06
def f06_upcoding(
    claims: pd.DataFrame, ctx: GenerationContext, pcfg: dict[str, Any]
) -> tuple[pd.DataFrame, list[LabelRecord]]:
    rng = ctx.rng
    em = ctx.ref["em_levels"]
    rel = dict(zip(em["hcpcs"], em["relative_payment"].astype(float), strict=True))
    providers = _pick_providers(ctx, pcfg["provider_share"])
    eligible = claims.index[
        claims["provider_id"].isin(providers) & claims["hcpcs"].isin(list(UPCODE_MAP))
    ].to_numpy()
    if len(eligible) == 0:
        return claims, []
    share = float(rng.uniform(pcfg["shift_share"][0], pcfg["shift_share"][1]))
    idx = rng.choice(eligible, size=max(1, int(len(eligible) * share)), replace=False)
    records: list[LabelRecord] = []
    for i in idx:
        old_code = str(claims.loc[i, "hcpcs"])
        new_code = UPCODE_MAP[old_code][0 if rng.random() < 0.7 else 1]
        ratio = rel[new_code] / rel[old_code]
        old_paid = float(claims.loc[i, "paid_amount"])
        claims.loc[i, "hcpcs"] = new_code
        _scale_amounts(claims, pd.Index([i]), ratio)
        records.append(
            (
                int(claims.loc[i, "_rowid"]),
                "F06_upcoding",
                float(claims.loc[i, "paid_amount"]) - old_paid,
            )
        )
    return claims, records


# ---------------------------------------------------------------- F07
def f07_unbundling(
    claims: pd.DataFrame, ctx: GenerationContext, pcfg: dict[str, Any]
) -> tuple[pd.DataFrame, list[LabelRecord]]:
    rng = ctx.rng
    ptp = ctx.ref["ptp_pairs"]
    comp_map = dict(zip(ptp["comprehensive_code"], ptp["component_code"], strict=True))
    ratio_map = dict(zip(ptp["comprehensive_code"], ptp["component_amount_ratio"], strict=True))
    providers = _pick_providers(ctx, pcfg["provider_share"])
    eligible = claims.index[
        claims["provider_id"].isin(providers) & claims["hcpcs"].isin(list(comp_map))
    ].to_numpy()
    k = min(len(eligible), int(len(claims) * pcfg["claim_rate"]))
    if k == 0:
        return claims, []
    idx = rng.choice(eligible, size=k, replace=False)
    comp = _copy_rows(claims, idx, ctx)
    ratios = claims.loc[idx, "hcpcs"].map(ratio_map).astype(float).to_numpy()
    comp["hcpcs"] = claims.loc[idx, "hcpcs"].map(comp_map).to_numpy()
    for col in AMOUNT_COLS:
        comp[col] = np.round(comp[col].to_numpy(dtype=float) * ratios, 2)
    comp["modifier"] = ""  # the tell: component billed same day WITHOUT a bypass modifier
    comp["units"] = 1
    records = [
        (int(r), "F07_unbundling", float(p))
        for r, p in zip(comp["_rowid"], comp["paid_amount"], strict=True)
    ]
    return pd.concat([claims, comp], ignore_index=True), records


# ---------------------------------------------------------------- F08
def f08_volume_spike(
    claims: pd.DataFrame, ctx: GenerationContext, pcfg: dict[str, Any]
) -> tuple[pd.DataFrame, list[LabelRecord]]:
    rng = ctx.rng
    providers = _pick_providers(ctx, pcfg["provider_share"])
    records: list[LabelRecord] = []
    frames: list[pd.DataFrame] = []
    n_benes = len(ctx.bene_ids)
    for pid in providers:
        pool = claims.index[claims["provider_id"] == pid].to_numpy()
        if len(pool) < 10:
            continue
        n_months = int(ctx.cfg["n_months"])
        lo_m, hi_m = max(2, n_months // 4), n_months - 1
        n_spike_months = int(rng.integers(pcfg["spike_months"][0], pcfg["spike_months"][1] + 1))
        months = rng.choice(
            np.arange(lo_m, hi_m), size=min(n_spike_months, hi_m - lo_m), replace=False
        )
        monthly_avg = max(3, len(pool) // 24)
        for m in months:
            mult = float(rng.uniform(pcfg["multiplier"][0], pcfg["multiplier"][1]))
            n_extra = int(monthly_avg * (mult - 1.0))
            templates = rng.choice(pool, size=n_extra, replace=True)
            extra = _copy_rows(claims, templates, ctx)
            month_start = np.datetime64(ctx.cfg["start_month"], "M") + int(m)
            days = rng.integers(0, 28, size=n_extra)
            extra["service_date"] = month_start.astype("datetime64[D]") + days.astype(
                "timedelta64[D]"
            )
            extra["service_end_date"] = extra["service_date"]
            extra["submit_date"] = extra["service_date"] + np.array(
                rng.integers(2, 20, size=n_extra), dtype="timedelta64[D]"
            )
            new_bene = rng.random(n_extra) < 0.6
            extra.loc[extra.index[new_bene], "bene_id"] = ctx.bene_ids[
                rng.integers(0, n_benes, size=int(new_bene.sum()))
            ]
            frames.append(extra)
            records += [
                (int(r), "F08_volume_spike", float(p))
                for r, p in zip(extra["_rowid"], extra["paid_amount"], strict=True)
            ]
    if frames:
        claims = pd.concat([claims, *frames], ignore_index=True)
    return claims, records


# ---------------------------------------------------------------- F09 / F10
def f09_dx_mismatch(
    claims: pd.DataFrame, ctx: GenerationContext, pcfg: dict[str, Any]
) -> tuple[pd.DataFrame, list[LabelRecord]]:
    rng = ctx.rng
    compat = ctx.ref["dx_procedure_compat"].copy()
    compat["compat"] = compat["compat"].astype(float)
    low: dict[str, np.ndarray] = {
        str(cat): g.loc[g["compat"] <= 0.1, "dx_group"].to_numpy()
        for cat, g in compat.groupby("hcpcs_category")
    }
    from src.data.reference import hcpcs_category

    providers = _pick_providers(ctx, pcfg["provider_share"])
    # E/M visits plausibly accompany any diagnosis; mismatch is meaningful for
    # procedures, therapy, imaging, labs, and drugs.
    cats = pd.Series([hcpcs_category(c) for c in claims["hcpcs"].astype(str)], index=claims.index)
    mismatchable = cats.isin(["PROCEDURE", "THERAPY", "IMAGING", "LAB", "DRUG"])
    eligible = claims.index[claims["provider_id"].isin(providers) & mismatchable].to_numpy()
    k = min(len(eligible), int(len(claims) * pcfg["claim_rate"]))
    if k == 0:
        return claims, []
    idx = rng.choice(eligible, size=k, replace=False)
    records: list[LabelRecord] = []
    for i in idx:
        cat = str(cats.loc[i])
        options = low.get(cat)
        if options is None or len(options) == 0:
            continue
        claims.loc[i, "dx_group"] = rng.choice(options)
        records.append(
            (int(claims.loc[i, "_rowid"]), "F09_dx_mismatch", float(claims.loc[i, "paid_amount"]))
        )
    return claims, records


def f10_pos_anomaly(
    claims: pd.DataFrame, ctx: GenerationContext, pcfg: dict[str, Any]
) -> tuple[pd.DataFrame, list[LabelRecord]]:
    rng = ctx.rng
    providers = _pick_providers(ctx, pcfg["provider_share"])
    facility = claims["pos"].astype(str).isin(["21", "22", "23"])
    eligible = claims.index[claims["provider_id"].isin(providers) & facility].to_numpy()
    k = min(len(eligible), int(len(claims) * pcfg["claim_rate"]))
    if k == 0:
        return claims, []
    idx = rng.choice(eligible, size=k, replace=False)
    claims.loc[idx, "pos"] = np.where(rng.random(k) < 0.8, "11", "12")
    paid = claims.loc[idx, "paid_amount"].to_numpy(dtype=float)
    rowids = claims.loc[idx, "_rowid"].to_numpy()
    records = [
        (int(r), "F10_pos_anomaly", float(0.3 * p)) for r, p in zip(rowids, paid, strict=True)
    ]
    return claims, records


# ---------------------------------------------------------------- F11 / F12
def f11_collusion_cluster(
    claims: pd.DataFrame, ctx: GenerationContext, pcfg: dict[str, Any]
) -> tuple[pd.DataFrame, list[LabelRecord]]:
    rng = ctx.rng
    provider_ids = ctx.providers["provider_id"].to_numpy()
    records: list[LabelRecord] = []
    frames: list[pd.DataFrame] = []
    n_months = int(ctx.cfg["n_months"])
    lo_m, hi_m = max(1, n_months // 4), n_months
    # Scale-aware budget so small test datasets aren't swamped by cluster rows.
    n_clusters = int(pcfg["n_clusters"])
    total_budget = max(30, int(0.005 * len(claims)))
    rows_per_provider_cap = max(10, total_budget // max(1, n_clusters * 3))
    for _ in range(n_clusters):
        n_prov = int(
            rng.integers(pcfg["providers_per_cluster"][0], pcfg["providers_per_cluster"][1] + 1)
        )
        cluster = rng.choice(provider_ids, size=n_prov, replace=False)
        n_shared = int(rng.integers(pcfg["shared_benes"][0], pcfg["shared_benes"][1] + 1))
        shared = ctx.bene_ids[rng.choice(len(ctx.bene_ids), size=n_shared, replace=False)]
        for pid in cluster:
            pool = claims.index[claims["provider_id"] == pid].to_numpy()
            if len(pool) == 0:
                continue
            n_rows = min(int(n_shared * rng.integers(4, 7)), rows_per_provider_cap)
            templates = rng.choice(pool, size=n_rows, replace=True)
            extra = _copy_rows(claims, templates, ctx)
            extra["bene_id"] = rng.choice(shared, size=n_rows)
            month_offsets = rng.integers(lo_m, hi_m, size=n_rows)
            days = rng.integers(0, 28, size=n_rows)
            base_month = np.datetime64(ctx.cfg["start_month"], "M")
            starts = (base_month + month_offsets).astype("datetime64[D]")
            extra["service_date"] = starts + days.astype("timedelta64[D]")
            extra["service_end_date"] = extra["service_date"]
            extra["submit_date"] = extra["service_date"] + np.array(
                rng.integers(2, 20, size=n_rows), dtype="timedelta64[D]"
            )
            frames.append(extra)
            records += [
                (int(r), "F11_collusion_cluster", float(p))
                for r, p in zip(extra["_rowid"], extra["paid_amount"], strict=True)
            ]
    if frames:
        claims = pd.concat([claims, *frames], ignore_index=True)
    return claims, records


def f12_code_concentration(
    claims: pd.DataFrame, ctx: GenerationContext, pcfg: dict[str, Any]
) -> tuple[pd.DataFrame, list[LabelRecord]]:
    rng = ctx.rng
    providers = _pick_providers(ctx, pcfg["provider_share"])
    records: list[LabelRecord] = []
    for pid in providers:
        rep = ctx.repertoires[pid]
        target_idx = int(np.argmax(rep.base_paid))
        target_code = str(rep.hcpcs[target_idx])
        target_amounts = (
            float(rep.base_submitted[target_idx]),
            float(rep.base_allowed[target_idx]),
            float(rep.base_paid[target_idx]),
        )
        pool = claims.index[
            (claims["provider_id"] == pid) & (claims["hcpcs"] != target_code)
        ].to_numpy()
        if len(pool) == 0:
            continue
        share = float(rng.uniform(pcfg["shift_share"][0], pcfg["shift_share"][1]))
        idx = rng.choice(pool, size=max(1, int(len(pool) * share)), replace=False)
        noise = rng.lognormal(0.0, 0.1, size=len(idx))
        claims.loc[idx, "hcpcs"] = target_code
        claims.loc[idx, "units"] = 1
        claims.loc[idx, "submitted_amount"] = np.round(target_amounts[0] * noise, 2)
        claims.loc[idx, "allowed_amount"] = np.round(
            np.minimum(target_amounts[1] * noise, target_amounts[0] * noise), 2
        )
        claims.loc[idx, "paid_amount"] = np.round(target_amounts[2] * noise * 0.99, 2)
        paid = claims.loc[idx, "paid_amount"].to_numpy(dtype=float)
        rowids = claims.loc[idx, "_rowid"].to_numpy()
        records += [
            (int(r), "F12_code_concentration", float(0.25 * p))
            for r, p in zip(rowids, paid, strict=True)
        ]
    return claims, records


# ---------------------------------------------------------------- orchestration
INJECTORS = {
    "F01_exact_duplicate": f01_exact_duplicate,
    "F02_near_duplicate": f02_near_duplicate,
    "F03_rapid_repeat": f03_rapid_repeat,
    "F04_excessive_units": f04_excessive_units,
    "F05_amount_inflation": f05_amount_inflation,
    "F06_upcoding": f06_upcoding,
    "F07_unbundling": f07_unbundling,
    "F08_volume_spike": f08_volume_spike,
    "F09_dx_mismatch": f09_dx_mismatch,
    "F10_pos_anomaly": f10_pos_anomaly,
    "F11_collusion_cluster": f11_collusion_cluster,
    "F12_code_concentration": f12_code_concentration,
}


def inject_all(
    claims: pd.DataFrame, ctx: GenerationContext
) -> tuple[pd.DataFrame, list[LabelRecord]]:
    """Run every configured injector; returns mutated claims + label records."""
    all_records: list[LabelRecord] = []
    for name, injector in INJECTORS.items():
        pcfg = ctx.cfg["patterns"].get(name)
        if not pcfg:
            continue
        before = len(claims)
        claims, records = injector(claims, ctx, pcfg)
        all_records += records
        ctx.stats[f"pattern_{name}"] = {
            "labeled_claims": len(records),
            "rows_added": len(claims) - before,
        }
        logger.info(
            "pattern_injected",
            extra={"pattern": name, "labeled": len(records), "added": len(claims) - before},
        )
    return claims, all_records
