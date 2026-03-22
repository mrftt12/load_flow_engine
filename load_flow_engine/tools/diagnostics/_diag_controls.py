"""Category 8 — Regulator / Capacitor / Inverter Control Errors.

Checks: ctrl_01, ctrl_02, ctrl_05.
"""
from __future__ import annotations

from typing import Any

from ...network import Network
from ...enums import BusType

from ._common import issue


def check_controls(net: Network) -> list[dict[str, Any]]:
    """Run all control-element checks."""
    issues: list[dict[str, Any]] = []
    issues.extend(_ctrl_01_references_nonexistent_bus(net))
    issues.extend(_ctrl_02_voltage_setpoint(net))
    issues.extend(_ctrl_05_multiple_shunts_same_bus(net))
    return issues


# ---------------------------------------------------------------------------
# ctrl_01 — Element references non-existent bus
# ---------------------------------------------------------------------------

def _ctrl_01_references_nonexistent_bus(net: Network) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    for sh_id, sh in net.shunts.items():
        if sh.bus_id not in net.buses:
            issues.append(issue(
                "high", "control_error", "shunt", sh_id, "bus_id",
                f"Shunt {sh_id} references bus {sh.bus_id} which does not exist.",
                "Map to a valid bus ID.",
            ))
    for gen_id, gen in net.generators.items():
        if gen.bus_id not in net.buses:
            issues.append(issue(
                "high", "control_error", "generator", gen_id, "bus_id",
                f"Generator {gen_id} references bus {gen.bus_id} which does not exist.",
                "Map to a valid bus ID.",
            ))
    for ld_id, ld in net.loads.items():
        if ld.bus_id not in net.buses:
            issues.append(issue(
                "high", "control_error", "load", ld_id, "bus_id",
                f"Load {ld_id} references bus {ld.bus_id} which does not exist.",
                "Map to a valid bus ID.",
            ))
    return issues


# ---------------------------------------------------------------------------
# ctrl_02 — Voltage setpoint outside normal range
# ---------------------------------------------------------------------------

def _ctrl_02_voltage_setpoint(net: Network) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    # Check generator voltage setpoints
    for gen_id, gen in net.generators.items():
        if gen.v_set_pu < 0.90 or gen.v_set_pu > 1.10:
            issues.append(issue(
                "high", "control_error", "generator", gen_id, "v_set_pu",
                f"Generator {gen_id} voltage setpoint v_set_pu={gen.v_set_pu:.4f} "
                f"is outside [0.90, 1.10] pu.",
                "Verify voltage setpoint is within ANSI/IEEE limits.",
            ))

    # Check SLACK bus initial voltage magnitudes
    for bus_id, bus in net.buses.items():
        if bus.bus_type != BusType.SLACK:
            continue
        for p in range(3):
            v = bus.ini_v_mag[p]
            if v < 0.90 or v > 1.10:
                issues.append(issue(
                    "high", "control_error", "bus", bus_id, "ini_v_mag",
                    f"SLACK bus {bus_id} phase {p} ini_v_mag={v:.4f} "
                    f"is outside [0.90, 1.10] pu.",
                    "Verify source voltage setpoint.",
                ))
                break  # one issue per bus
    return issues


# ---------------------------------------------------------------------------
# ctrl_05 — Multiple shunts on same bus
# ---------------------------------------------------------------------------

def _ctrl_05_multiple_shunts_same_bus(net: Network) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    bus_shunts: dict[str, list[str]] = {}
    for sh_id, sh in net.shunts.items():
        bus_shunts.setdefault(sh.bus_id, []).append(sh_id)

    for bus_id, shunt_ids in bus_shunts.items():
        if len(shunt_ids) > 1:
            desc = ", ".join(shunt_ids)
            issues.append(issue(
                "low", "control_error", "bus", bus_id, "shunts",
                f"Bus {bus_id} has multiple shunts: {desc}.",
                "Verify coordinated control or remove duplicate shunts.",
            ))
    return issues
