"""Category 1 — Wrong Voltage Base.

Checks: vb_01 through vb_05.
"""
from __future__ import annotations

from typing import Any

from ...network import Network
from ...enums import BusType

from ._common import issue, get_bus_base_kv, get_source_buses, build_network_graph


def check_voltage_base(net: Network, *, include_bfs: bool = True) -> list[dict[str, Any]]:
    """Run all voltage base checks."""
    issues: list[dict[str, Any]] = []
    issues.extend(_vb_01_bus_base_kv_missing_or_zero(net))
    issues.extend(_vb_02_trafo_bus_voltage_mismatch(net))
    issues.extend(_vb_03_source_voltage_deviation(net))
    issues.extend(_vb_04_unit_confusion(net))
    if include_bfs:
        issues.extend(_vb_05_feeder_path_voltage_consistency(net))
    return issues


# ---------------------------------------------------------------------------
# vb_01 — Bus base_kv = 0 or missing
# ---------------------------------------------------------------------------

def _vb_01_bus_base_kv_missing_or_zero(net: Network) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    for bus_id, bus in net.buses.items():
        if bus.base_kv <= 0:
            issues.append(issue(
                "critical", "voltage_base", "bus", bus_id, "base_kv",
                f"Bus {bus_id} has base_kv={bus.base_kv} (missing or non-positive).",
                "Set base_kv to the correct nominal line-line voltage in kV.",
            ))
    return issues


# ---------------------------------------------------------------------------
# vb_02 — Transformer endpoints with same base_kv
# ---------------------------------------------------------------------------

def _vb_02_trafo_bus_voltage_mismatch(net: Network) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    for xf_id, xf in net.transformers.items():
        hv_kv = get_bus_base_kv(net, xf.from_bus)
        lv_kv = get_bus_base_kv(net, xf.to_bus)
        if hv_kv is None or lv_kv is None or hv_kv <= 0 or lv_kv <= 0:
            continue
        # A transformer should connect buses with different voltage levels
        if hv_kv == lv_kv:
            issues.append(issue(
                "high", "voltage_base", "transformer", xf_id, "base_kv",
                f"Transformer {xf_id} connects buses with identical base_kv "
                f"({hv_kv:.4f} kV). Expected a voltage transformation.",
                "Verify bus voltage bases or transformer bus connections.",
            ))
    return issues


# ---------------------------------------------------------------------------
# vb_03 — Source voltage deviation from 1.0 pu
# ---------------------------------------------------------------------------

def _vb_03_source_voltage_deviation(net: Network) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    # Check SLACK bus initial voltage magnitudes
    for bus_id, bus in net.buses.items():
        if bus.bus_type != BusType.SLACK:
            continue
        for p in range(3):
            v = bus.ini_v_mag[p]
            if abs(v - 1.0) > 0.10:
                issues.append(issue(
                    "high", "voltage_base", "bus", bus_id, "ini_v_mag",
                    f"SLACK bus {bus_id} phase {p} ini_v_mag={v:.4f} deviates >10% from 1.0 pu.",
                    "Verify source voltage magnitude is intended.",
                ))
                break  # one issue per bus

    # Check generators with voltage setpoints
    for gen_id, gen in net.generators.items():
        if abs(gen.v_set_pu - 1.0) > 0.10:
            issues.append(issue(
                "high", "voltage_base", "generator", gen_id, "v_set_pu",
                f"Generator {gen_id} v_set_pu={gen.v_set_pu:.4f} deviates >10% from 1.0 pu.",
                "Verify generator voltage setpoint.",
            ))
    return issues


# ---------------------------------------------------------------------------
# vb_04 — Likely unit confusion
# ---------------------------------------------------------------------------

def _vb_04_unit_confusion(net: Network) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    for bus_id, bus in net.buses.items():
        vn = bus.base_kv
        if 0 < vn < 0.1:
            issues.append(issue(
                "medium", "voltage_base", "bus", bus_id, "base_kv",
                f"Bus {bus_id} base_kv={vn:.6f} — possibly Volts stored as kV.",
                "Convert to kV if the intended voltage is in the Volt range.",
            ))
        elif vn > 500:
            issues.append(issue(
                "medium", "voltage_base", "bus", bus_id, "base_kv",
                f"Bus {bus_id} base_kv={vn:.1f} — unusually high for distribution.",
                "Verify this is a transmission-level bus or correct the voltage base.",
            ))
    return issues


# ---------------------------------------------------------------------------
# vb_05 — Voltage base inconsistency along feeder path
# ---------------------------------------------------------------------------

def _vb_05_feeder_path_voltage_consistency(net: Network) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    if not net.branches:
        return issues

    # Collect transformer bus-pairs so we can skip them
    trafo_bus_pairs: set[tuple] = set()
    for xf in net.transformers.values():
        pair = (min(xf.from_bus, xf.to_bus), max(xf.from_bus, xf.to_bus))
        trafo_bus_pairs.add(pair)

    # Check each branch: if endpoints have different base_kv without transformer, flag it
    for br_id, br in net.branches.items():
        vn_from = get_bus_base_kv(net, br.from_bus)
        vn_to = get_bus_base_kv(net, br.to_bus)
        if vn_from is None or vn_to is None or vn_from <= 0 or vn_to <= 0:
            continue
        pair_key = (min(br.from_bus, br.to_bus), max(br.from_bus, br.to_bus))
        if pair_key in trafo_bus_pairs:
            continue
        ratio = abs(vn_from - vn_to) / max(vn_from, vn_to)
        if ratio > 0.01:
            issues.append(issue(
                "medium", "voltage_base", "branch", br_id, "from_bus/to_bus",
                f"Branch {br_id} connects buses with different base_kv "
                f"({vn_from:.4f} vs {vn_to:.4f} kV) without an intervening transformer.",
                "Insert a transformer or correct bus voltage bases.",
            ))
    return issues
