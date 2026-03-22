"""Category 7 — Load Model Problems.

Checks: ld_01, ld_02, ld_03, ld_05.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from ...network import Network

from ._common import issue, get_bus_base_kv, get_active_phases


def check_load_model(net: Network, *, include_capacity_checks: bool = True) -> list[dict[str, Any]]:
    """Run all load model checks."""
    issues: list[dict[str, Any]] = []
    if include_capacity_checks:
        issues.extend(_ld_01_load_exceeds_transformer_capacity(net))
    issues.extend(_ld_02_power_factor_out_of_range(net))
    if include_capacity_checks:
        issues.extend(_ld_03_load_on_wrong_voltage_level(net))
    issues.extend(_ld_05_unbalanced_loading(net))
    return issues


# ---------------------------------------------------------------------------
# ld_01 — Load kVA exceeds transformer capacity
# ---------------------------------------------------------------------------

def _ld_01_load_exceeds_transformer_capacity(net: Network) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    if not net.loads or not net.transformers:
        return issues

    # Sum load MVA per bus
    load_mva_per_bus: dict[str, float] = {}
    for ld in net.loads.values():
        s = math.sqrt(float(ld.mw.sum())**2 + float(ld.mvar.sum())**2)
        load_mva_per_bus[ld.bus_id] = load_mva_per_bus.get(ld.bus_id, 0.0) + s

    # Check each transformer's LV-side loading
    for xf_id, xf in net.transformers.items():
        if xf.mva_rating <= 0:
            continue
        lv_load = load_mva_per_bus.get(xf.to_bus, 0.0)
        if lv_load > xf.mva_rating * 1.5:
            issues.append(issue(
                "high", "load_model", "transformer", xf_id, "mva_rating",
                f"Transformer {xf_id} rated {xf.mva_rating:.4f} MVA but LV bus {xf.to_bus} "
                f"has {lv_load:.4f} MVA of connected load ({lv_load/xf.mva_rating*100:.0f}%).",
                "Verify load allocation or increase transformer rating.",
            ))
    return issues


# ---------------------------------------------------------------------------
# ld_02 — Power factor out of range
# ---------------------------------------------------------------------------

def _ld_02_power_factor_out_of_range(net: Network) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    for ld_id, ld in net.loads.items():
        for p in get_active_phases(ld.phase_type):
            mw = float(ld.mw[p])
            mvar = float(ld.mvar[p])
            s = math.sqrt(mw**2 + mvar**2)
            if s < 1e-12:
                continue
            pf = abs(mw) / s
            if pf < 0.70:
                phase_names = {0: 'A', 1: 'B', 2: 'C'}
                issues.append(issue(
                    "high", "load_model", "load", ld_id, "mw/mvar",
                    f"Load {ld_id} phase {phase_names.get(p, p)} power factor = {pf:.3f} "
                    f"is below 0.70.",
                    "Verify reactive power or add power factor correction.",
                ))
                break  # one issue per load
    return issues


# ---------------------------------------------------------------------------
# ld_03 — Load on wrong voltage level
# ---------------------------------------------------------------------------

def _ld_03_load_on_wrong_voltage_level(net: Network) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    for ld_id, ld in net.loads.items():
        total_p = float(ld.mw.sum())
        vn = get_bus_base_kv(net, ld.bus_id)
        if vn is None:
            continue
        # Small residential load (< 0.001 MW = 1 kW) on a primary bus (> 4 kV)
        if abs(total_p) < 0.001 and vn > 4.0:
            issues.append(issue(
                "medium", "load_model", "load", ld_id, "bus_id/mw",
                f"Load {ld_id} has total p_mw={total_p:.6f} (< 1 kW) on {vn:.2f} kV bus — "
                f"possible secondary load on primary bus.",
                "Verify load is on the correct bus and voltage level.",
            ))
    return issues


# ---------------------------------------------------------------------------
# ld_05 — Unbalanced loading exceeds threshold
# ---------------------------------------------------------------------------

def _ld_05_unbalanced_loading(net: Network) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    if not net.loads:
        return issues

    # Aggregate load MW per bus per phase
    bus_phase_p: dict[str, dict[int, float]] = {}
    for ld in net.loads.values():
        for p in get_active_phases(ld.phase_type):
            mw = float(ld.mw[p])
            if abs(mw) < 1e-12:
                continue
            bus_phase_p.setdefault(ld.bus_id, {}).setdefault(p, 0.0)
            bus_phase_p[ld.bus_id][p] += mw

    for bus_id, phase_dict in bus_phase_p.items():
        if len(phase_dict) < 2:
            continue
        values = list(phase_dict.values())
        avg = sum(values) / len(values)
        if avg < 1e-12:
            continue
        max_v, min_v = max(values), min(values)
        imbalance = (max_v - min_v) / avg
        if imbalance > 0.20:
            phase_names = {0: 'A', 1: 'B', 2: 'C'}
            phase_str = {phase_names.get(k, k): f"{v:.4f}" for k, v in phase_dict.items()}
            issues.append(issue(
                "low", "load_model", "bus", bus_id, "mw",
                f"Bus {bus_id} phase loading imbalance = {imbalance*100:.1f}% "
                f"(phases: {phase_str}).",
                "Redistribute loads across phases to reduce imbalance.",
            ))
    return issues
