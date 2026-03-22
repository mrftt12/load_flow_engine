"""Category 10 — Bad Topology / Connectivity.

Checks: top_01 through top_06.
"""
from __future__ import annotations

from typing import Any

import networkx as nx

from ...network import Network

from ._common import (
    issue,
    get_source_buses,
    bus_has_equipment,
    build_network_graph,
    bfs_from_sources,
)


def check_topology(net: Network) -> list[dict[str, Any]]:
    """Run all topology / connectivity checks."""
    graph = build_network_graph(net)
    issues: list[dict[str, Any]] = []
    issues.extend(_top_01_unreachable_from_source(net, graph))
    issues.extend(_top_02_parallel_sources(net, graph))
    issues.extend(_top_03_switch_element_type_mismatch(net))
    issues.extend(_top_04_radial_violation(net, graph))
    issues.extend(_top_05_long_radial_path(net, graph))
    issues.extend(_top_06_dead_end_non_leaf_load(net, graph))
    return issues


# ---------------------------------------------------------------------------
# top_01 — Bus not reachable from any source
# ---------------------------------------------------------------------------

def _top_01_unreachable_from_source(net: Network, graph: nx.Graph) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    reachable = bfs_from_sources(net, graph)
    for bus_id in net.buses:
        if bus_id not in reachable:
            issues.append(issue(
                "critical", "topology", "bus", bus_id, "reachability",
                f"Bus {bus_id} is not reachable from any source.",
                "Connect to the network or remove it.",
            ))
    return issues


# ---------------------------------------------------------------------------
# top_02 — Source-to-source path (unintended parallel sources)
# ---------------------------------------------------------------------------

def _top_02_parallel_sources(net: Network, graph: nx.Graph) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    source_buses = list(get_source_buses(net))
    if len(source_buses) < 2:
        return issues

    for i in range(len(source_buses)):
        for j in range(i + 1, len(source_buses)):
            s1, s2 = source_buses[i], source_buses[j]
            if s1 in graph and s2 in graph and nx.has_path(graph, s1, s2):
                issues.append(issue(
                    "high", "topology", "network", f"{s1},{s2}", "parallel_sources",
                    f"Sources at bus {s1} and {s2} are connected — "
                    f"potential parallel source conflict.",
                    "Verify intentional parallel operation or insert an open switch.",
                ))
    return issues


# ---------------------------------------------------------------------------
# top_03 — Switch references wrong element type
# ---------------------------------------------------------------------------

def _top_03_switch_element_type_mismatch(net: Network) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    for sw_id, sw in net.switches.items():
        et = str(sw.et).lower()
        element = str(sw.element)
        if et == 'b' and element not in net.buses:
            issues.append(issue(
                "high", "topology", "switch", sw_id, "element",
                f"Switch {sw_id} type='b' (bus-bus) but element {element} "
                f"is not a valid bus.",
                "Correct switch element reference or change switch type.",
            ))
        elif et == 'l' and element not in net.branches:
            issues.append(issue(
                "high", "topology", "switch", sw_id, "element",
                f"Switch {sw_id} type='l' (line) but element {element} "
                f"is not a valid branch.",
                "Correct switch element reference or change switch type.",
            ))
    return issues


# ---------------------------------------------------------------------------
# top_04 — Radial violation — mesh detected
# ---------------------------------------------------------------------------

def _top_04_radial_violation(net: Network, graph: nx.Graph) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    cycles = nx.cycle_basis(graph)
    for cycle in cycles:
        cycle_edges = []
        for i in range(len(cycle)):
            u, v = cycle[i], cycle[(i + 1) % len(cycle)]
            edge_data = graph.get_edge_data(u, v)
            if edge_data:
                cycle_edges.append(
                    f"{edge_data.get('element', '?')}[{edge_data.get('index', '?')}]"
                )
        issues.append(issue(
            "medium", "topology", "network", ",".join(map(str, cycle[:5])),
            "loop",
            f"Loop detected involving buses {cycle[:5]}"
            f"{'...' if len(cycle) > 5 else ''} "
            f"via {', '.join(cycle_edges[:3])}.",
            "Open one switching point to restore radial topology.",
        ))
    return issues


# ---------------------------------------------------------------------------
# top_05 — Long radial path (voltage drop risk)
# ---------------------------------------------------------------------------

def _top_05_long_radial_path(
    net: Network, graph: nx.Graph, *, max_segments: int = 50,
) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    source_buses = get_source_buses(net)
    if not source_buses:
        return issues

    # Check leaf buses (degree 1)
    for node in graph.nodes():
        if graph.degree(node) != 1:
            continue
        min_path = None
        for src in source_buses:
            if src not in graph:
                continue
            try:
                path = nx.shortest_path(graph, src, node)
                if min_path is None or len(path) < len(min_path):
                    min_path = path
            except nx.NetworkXNoPath:
                continue

        if min_path is None:
            continue

        n_segments = len(min_path) - 1
        if n_segments > max_segments:
            issues.append(issue(
                "medium", "topology", "bus", node, "path_length",
                f"Leaf bus {node} is {n_segments} segments from source (>{max_segments}).",
                "Check for voltage drop issues on long radial paths.",
            ))
    return issues


# ---------------------------------------------------------------------------
# top_06 — Degree-1 buses that aren't leaf loads
# ---------------------------------------------------------------------------

def _top_06_dead_end_non_leaf_load(net: Network, graph: nx.Graph) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    for node in graph.nodes():
        if graph.degree(node) == 1 and not bus_has_equipment(net, node):
            issues.append(issue(
                "low", "topology", "bus", node, "dead_end",
                f"Bus {node} is a dead-end with no load, generation, or source.",
                "Remove unused bus or connect equipment.",
            ))
    return issues
