"""Category 3 — Floating / Weakly Grounded Nodes.

Checks: gnd_01, gnd_04.
"""
from __future__ import annotations

from typing import Any

from ...network import Network

from ._common import (
    issue,
    get_connected_buses,
    bus_has_equipment,
    build_network_graph,
)


def check_grounding(net: Network) -> list[dict[str, Any]]:
    """Run all grounding / floating node checks."""
    issues: list[dict[str, Any]] = []
    issues.extend(_gnd_01_bus_with_no_connections(net))
    issues.extend(_gnd_04_dead_end_buses(net))
    return issues


# ---------------------------------------------------------------------------
# gnd_01 — Bus with no connected elements
# ---------------------------------------------------------------------------

def _gnd_01_bus_with_no_connections(net: Network) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    connected = get_connected_buses(net)
    for bus_id in net.buses:
        if bus_id not in connected:
            issues.append(issue(
                "critical", "grounding", "bus", bus_id, "connectivity",
                f"Bus {bus_id} has no connected elements (completely isolated).",
                "Connect the bus to the network or remove it.",
            ))
    return issues


# ---------------------------------------------------------------------------
# gnd_04 — Degree-1 buses (dead-ends) that aren't loads/gens/sources
# ---------------------------------------------------------------------------

def _gnd_04_dead_end_buses(net: Network) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    graph = build_network_graph(net)
    for node in graph.nodes():
        if graph.degree(node) == 1 and not bus_has_equipment(net, node):
            issues.append(issue(
                "medium", "grounding", "bus", node, "degree",
                f"Bus {node} is a dead-end (degree 1) with no load, generation, or source.",
                "Verify this bus is intentional or connect equipment / remove it.",
            ))
    return issues
