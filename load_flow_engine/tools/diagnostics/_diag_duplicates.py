"""Category 9 — Duplicate / Contradictory Equipment.

Checks: dup_02, dup_04, dup_05, dup_06.
"""
from __future__ import annotations

from typing import Any

from ...network import Network

from ._common import issue


def check_duplicates(net: Network) -> list[dict[str, Any]]:
    """Run all duplicate / contradictory equipment checks."""
    issues: list[dict[str, Any]] = []
    issues.extend(_dup_02_parallel_branches_contradictory_impedance(net))
    issues.extend(_dup_04_overlapping_transformers(net))
    issues.extend(_dup_05_duplicate_loads(net))
    issues.extend(_dup_06_name_collisions(net))
    return issues


# ---------------------------------------------------------------------------
# dup_02 — Parallel branches with contradictory impedance
# ---------------------------------------------------------------------------

def _dup_02_parallel_branches_contradictory_impedance(net: Network) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    if not net.branches:
        return issues

    # Group branches by bus pair (order-independent)
    bus_pair_branches: dict[tuple, list[tuple[str, float]]] = {}
    for br_id, br in net.branches.items():
        key = (min(br.from_bus, br.to_bus), max(br.from_bus, br.to_bus))
        bus_pair_branches.setdefault(key, []).append((br_id, br.r1))

    for pair, entries in bus_pair_branches.items():
        if len(entries) < 2:
            continue
        r_values = [r for _, r in entries if r > 0]
        if len(r_values) >= 2:
            r_max, r_min = max(r_values), min(r_values)
            if r_min > 0 and (r_max / r_min) > 1.5:
                br_ids = [bid for bid, _ in entries]
                issues.append(issue(
                    "high", "duplicate_equipment", "branch", ",".join(br_ids),
                    "r1",
                    f"Parallel branches between buses {pair} have resistance differing "
                    f"by >50% (range {r_min:.6f}–{r_max:.6f} pu).",
                    "Verify both branches are intended parallels or correct impedance values.",
                ))
    return issues


# ---------------------------------------------------------------------------
# dup_04 — Overlapping transformers (same bus pair)
# ---------------------------------------------------------------------------

def _dup_04_overlapping_transformers(net: Network) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    if not net.transformers:
        return issues

    seen_pairs: dict[tuple, list[str]] = {}
    for xf_id, xf in net.transformers.items():
        key = (min(xf.from_bus, xf.to_bus), max(xf.from_bus, xf.to_bus))
        seen_pairs.setdefault(key, []).append(xf_id)

    for pair, xf_ids in seen_pairs.items():
        if len(xf_ids) > 1:
            issues.append(issue(
                "medium", "duplicate_equipment", "transformer", ",".join(xf_ids),
                "bus_pair",
                f"Multiple transformers ({len(xf_ids)}) connect the same bus pair {pair}.",
                "Verify parallel transformers are intentional and have consistent ratings.",
            ))
    return issues


# ---------------------------------------------------------------------------
# dup_05 — Duplicate loads on same bus/phase
# ---------------------------------------------------------------------------

def _dup_05_duplicate_loads(net: Network) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    if not net.loads:
        return issues

    bus_phase_loads: dict[tuple, list[str]] = {}
    for ld_id, ld in net.loads.items():
        key = (ld.bus_id, ld.phase_type)
        bus_phase_loads.setdefault(key, []).append(ld_id)

    for (bus_id, phase_type), ld_ids in bus_phase_loads.items():
        if len(ld_ids) > 1:
            issues.append(issue(
                "medium", "duplicate_equipment", "load", ",".join(ld_ids),
                "bus_id/phase_type",
                f"Multiple loads ({len(ld_ids)}) at bus {bus_id}, phase_type {phase_type.name}.",
                "Consolidate into a single load or verify parallel loads are intentional.",
            ))
    return issues


# ---------------------------------------------------------------------------
# dup_06 — Name collisions within element collections
# ---------------------------------------------------------------------------

def _dup_06_name_collisions(net: Network) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    collections = [
        ("bus", net.buses),
        ("branch", net.branches),
        ("transformer", net.transformers),
        ("load", net.loads),
        ("generator", net.generators),
        ("shunt", net.shunts),
        ("switch", net.switches),
    ]
    for coll_name, coll in collections:
        name_count: dict[str, int] = {}
        for elem in coll.values():
            name = getattr(elem, 'name', '')
            if name:
                name_count[name] = name_count.get(name, 0) + 1
        for name, count in name_count.items():
            if count > 1:
                issues.append(issue(
                    "low", "duplicate_equipment", coll_name, name,
                    "name",
                    f"Duplicate name '{name}' found in {coll_name} ({count} instances).",
                    "Use unique names to avoid ambiguity in reports and cross-references.",
                ))
    return issues
