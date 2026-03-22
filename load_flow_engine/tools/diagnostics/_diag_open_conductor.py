"""Category 6 — Open Conductor Issues.

Checks: oc_01, oc_02, oc_04.
"""
from __future__ import annotations

from typing import Any

import networkx as nx

from ...network import Network

from ._common import (
    issue,
    get_active_phases,
    build_network_graph,
)


def check_open_conductor(net: Network) -> list[dict[str, Any]]:
    """Run all open conductor checks."""
    issues: list[dict[str, Any]] = []
    issues.extend(_oc_01_branch_phase_incompatible_with_bus(net))
    issues.extend(_oc_02_single_phasing(net))
    issues.extend(_oc_04_open_switch_isolates_loads(net))
    return issues


# ---------------------------------------------------------------------------
# oc_01 — Branch phase type not compatible with endpoint buses
# ---------------------------------------------------------------------------

def _oc_01_branch_phase_incompatible_with_bus(net: Network) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    for br_id, br in net.branches.items():
        br_phases = set(get_active_phases(br.phase_type))
        for bus_id in (br.from_bus, br.to_bus):
            bus = net.buses.get(bus_id)
            if bus is None:
                continue
            bus_phases = set(get_active_phases(bus.phase_type))
            if not br_phases.issubset(bus_phases):
                missing = br_phases - bus_phases
                phase_names = {0: 'A', 1: 'B', 2: 'C'}
                missing_str = [phase_names.get(p, str(p)) for p in sorted(missing)]
                issues.append(issue(
                    "critical", "open_conductor", "branch", br_id, "phase_type",
                    f"Branch {br_id} uses phase(s) {missing_str} not available "
                    f"on bus {bus_id} — possible open conductor.",
                    "Check for broken conductor or incorrect phase assignment.",
                ))
    return issues


# ---------------------------------------------------------------------------
# oc_02 — Single-phasing: 3-phase bus with incomplete phase coverage
# ---------------------------------------------------------------------------

def _oc_02_single_phasing(net: Network) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    from ...enums import PhaseType

    # For each ABC bus, check that connected branches collectively cover all 3 phases
    for bus_id, bus in net.buses.items():
        if bus.phase_type != PhaseType.ABC:
            continue

        served_phases: set[int] = set()
        for br in net.branches.values():
            if br.from_bus == bus_id or br.to_bus == bus_id:
                served_phases.update(get_active_phases(br.phase_type))
        for xf in net.transformers.values():
            if xf.from_bus == bus_id or xf.to_bus == bus_id:
                served_phases.update(get_active_phases(xf.phase_type))

        if not served_phases:
            continue  # isolated bus — caught by grounding checks

        if len(served_phases) == 2:
            missing = {0, 1, 2} - served_phases
            phase_names = {0: 'A', 1: 'B', 2: 'C'}
            missing_str = [phase_names.get(p, str(p)) for p in sorted(missing)]
            issues.append(issue(
                "high", "open_conductor", "bus", bus_id, "phase",
                f"3-phase bus {bus_id} is only served by 2 phases — "
                f"missing phase(s): {missing_str}.",
                "Check upstream branches for open conductor or disconnected phase.",
            ))
    return issues


# ---------------------------------------------------------------------------
# oc_04 — Open switch isolates downstream loads
# ---------------------------------------------------------------------------

def _oc_04_open_switch_isolates_loads(net: Network) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    if not net.switches:
        return issues

    load_buses: set[str] = {ld.bus_id for ld in net.loads.values()}
    if not load_buses:
        return issues

    # Build full graph (all switches treated as closed)
    graph_full = build_network_graph(net)

    for sw_id, sw in net.switches.items():
        if sw.closed:
            continue
        if sw.et != 'b':
            continue

        bus_a = str(sw.bus)
        bus_b = str(sw.element)
        if bus_b not in graph_full:
            continue

        # Check: if this switch were removed, does it strand loads?
        temp_graph = graph_full.copy()
        if temp_graph.has_edge(bus_a, bus_b):
            temp_graph.remove_edge(bus_a, bus_b)
        if bus_b in temp_graph:
            component = nx.node_connected_component(temp_graph, bus_b)
            stranded_loads = component.intersection(load_buses)
            if stranded_loads:
                issues.append(issue(
                    "medium", "open_conductor", "switch", sw_id, "closed",
                    f"Open switch {sw_id} isolates {len(stranded_loads)} bus(es) with loads.",
                    "Verify this is intentional or close the switch to restore service.",
                ))
    return issues
