"""Category 4 — Bad Phase Connectivity.

Checks: ph_01, ph_02, ph_04.
"""
from __future__ import annotations

from collections import deque
from typing import Any

from ...network import Network
from ...enums import BusType

from ._common import (
    issue,
    get_source_buses,
    get_active_phases,
    build_network_graph,
)


def check_phase_connectivity(net: Network) -> list[dict[str, Any]]:
    """Run all phase connectivity checks."""
    issues: list[dict[str, Any]] = []
    issues.extend(_ph_01_load_on_unreachable_phase(net))
    issues.extend(_ph_02_phase_mismatch_across_transformer(net))
    issues.extend(_ph_04_load_phase_not_on_bus(net))
    return issues


# ---------------------------------------------------------------------------
# ph_01 — Load on phase not reachable from source (per-phase BFS)
# ---------------------------------------------------------------------------

def _ph_01_load_on_unreachable_phase(net: Network) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    if not net.loads or not net.branches:
        return issues

    # Build per-phase adjacency
    phase_adj: dict[tuple, set[tuple]] = {}

    for br in net.branches.values():
        for p in get_active_phases(br.phase_type):
            n_from = (br.from_bus, p)
            n_to = (br.to_bus, p)
            phase_adj.setdefault(n_from, set()).add(n_to)
            phase_adj.setdefault(n_to, set()).add(n_from)

    for xf in net.transformers.values():
        for p in get_active_phases(xf.phase_type):
            n_from = (xf.from_bus, p)
            n_to = (xf.to_bus, p)
            phase_adj.setdefault(n_from, set()).add(n_to)
            phase_adj.setdefault(n_to, set()).add(n_from)

    # BFS from source buses on all their active phases
    reachable: set[tuple] = set()
    queue: deque[tuple] = deque()

    for bus_id, bus in net.buses.items():
        if bus.bus_type == BusType.SLACK:
            for p in get_active_phases(bus.phase_type):
                seed = (bus_id, p)
                if seed not in reachable:
                    reachable.add(seed)
                    queue.append(seed)

    while queue:
        current = queue.popleft()
        for neighbor in phase_adj.get(current, set()):
            if neighbor not in reachable:
                reachable.add(neighbor)
                queue.append(neighbor)

    # Check loads
    for ld_id, ld in net.loads.items():
        for p in get_active_phases(ld.phase_type):
            if (ld.bus_id, p) not in reachable:
                phase_names = {0: 'A', 1: 'B', 2: 'C'}
                issues.append(issue(
                    "critical", "phase_connectivity", "load", ld_id, "phase_type",
                    f"Load {ld_id} is on phase {phase_names.get(p, p)} at bus {ld.bus_id}, "
                    f"but that phase is not reachable from any source.",
                    "Verify upstream branch phases or correct load phase assignment.",
                ))
    return issues


# ---------------------------------------------------------------------------
# ph_02 — Phase mismatch across transformer
# ---------------------------------------------------------------------------

def _ph_02_phase_mismatch_across_transformer(net: Network) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    for xf_id, xf in net.transformers.items():
        xf_phases = set(get_active_phases(xf.phase_type))
        from_bus = net.buses.get(xf.from_bus)
        to_bus = net.buses.get(xf.to_bus)
        if from_bus is None or to_bus is None:
            continue
        from_phases = set(get_active_phases(from_bus.phase_type))
        to_phases = set(get_active_phases(to_bus.phase_type))

        # Transformer phases should be a subset of both bus phases
        if not xf_phases.issubset(from_phases):
            missing = xf_phases - from_phases
            phase_names = {0: 'A', 1: 'B', 2: 'C'}
            missing_str = [phase_names.get(p, str(p)) for p in sorted(missing)]
            issues.append(issue(
                "high", "phase_connectivity", "transformer", xf_id, "phase_type",
                f"Transformer {xf_id} uses phase(s) {missing_str} not available "
                f"on from_bus {xf.from_bus}.",
                "Correct transformer or bus phase assignments.",
            ))
        if not xf_phases.issubset(to_phases):
            missing = xf_phases - to_phases
            phase_names = {0: 'A', 1: 'B', 2: 'C'}
            missing_str = [phase_names.get(p, str(p)) for p in sorted(missing)]
            issues.append(issue(
                "high", "phase_connectivity", "transformer", xf_id, "phase_type",
                f"Transformer {xf_id} uses phase(s) {missing_str} not available "
                f"on to_bus {xf.to_bus}.",
                "Correct transformer or bus phase assignments.",
            ))
    return issues


# ---------------------------------------------------------------------------
# ph_04 — Load phase not available on bus
# ---------------------------------------------------------------------------

def _ph_04_load_phase_not_on_bus(net: Network) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    for ld_id, ld in net.loads.items():
        bus = net.buses.get(ld.bus_id)
        if bus is None:
            continue
        load_phases = set(get_active_phases(ld.phase_type))
        bus_phases = set(get_active_phases(bus.phase_type))
        if not load_phases.issubset(bus_phases):
            missing = load_phases - bus_phases
            phase_names = {0: 'A', 1: 'B', 2: 'C'}
            missing_str = [phase_names.get(p, str(p)) for p in sorted(missing)]
            issues.append(issue(
                "medium", "phase_connectivity", "load", ld_id, "phase_type",
                f"Load {ld_id} uses phase(s) {missing_str} but bus {ld.bus_id} "
                f"only has phases {sorted(bus_phases)}.",
                "Correct load phase assignment to match available bus phases.",
            ))
    return issues
