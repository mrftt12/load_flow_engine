"""Shared helpers for the diagnostics package.

Provides graph utilities and helper functions used by multiple diagnostic
modules.  All functions operate on LFE ``Network`` objects (dict-of-dataclass
architecture) — no pandas dependency.
"""
from __future__ import annotations

from typing import Any, List, Set

import networkx as nx

from ...network import Network
from ...enums import BusType, PhaseType
from ...helpers import _active_phases


# ---------------------------------------------------------------------------
# Issue builder
# ---------------------------------------------------------------------------

def issue(
    severity: str,
    check: str,
    element_type: str,
    element_index: Any,
    field: str,
    message: str,
    suggestion: str,
) -> dict[str, Any]:
    return {
        "severity": severity,
        "check": check,
        "element_type": element_type,
        "element_index": element_index,
        "field": field,
        "message": message,
        "suggestion": suggestion,
    }


# ---------------------------------------------------------------------------
# Bus / element helpers
# ---------------------------------------------------------------------------

def get_bus_base_kv(net: Network, bus_id: str) -> float | None:
    """Return the base_kv for a bus, or None if the bus doesn't exist."""
    bus = net.buses.get(bus_id)
    return bus.base_kv if bus is not None else None


def get_source_buses(net: Network) -> set[str]:
    """Return bus IDs for all SLACK buses."""
    return {bid for bid, b in net.buses.items() if b.bus_type == BusType.SLACK}


def get_connected_buses(net: Network) -> set[str]:
    """Return set of bus IDs that are referenced by at least one element."""
    connected: set[str] = set()
    for br in net.branches.values():
        connected.add(br.from_bus)
        connected.add(br.to_bus)
    for xf in net.transformers.values():
        connected.add(xf.from_bus)
        connected.add(xf.to_bus)
    for ld in net.loads.values():
        connected.add(ld.bus_id)
    for gen in net.generators.values():
        connected.add(gen.bus_id)
    for sh in net.shunts.values():
        connected.add(sh.bus_id)
    for sw in net.switches.values():
        connected.add(str(sw.bus))
        if sw.et == 'b':
            connected.add(str(sw.element))
    return connected


def get_active_phases(phase_type: PhaseType) -> List[int]:
    """Re-export of the core _active_phases helper."""
    return _active_phases(phase_type)


def bus_has_equipment(net: Network, bus_id: str) -> bool:
    """Return True if any load, generator, shunt, or source is at this bus."""
    if net.buses.get(bus_id) and net.buses[bus_id].bus_type == BusType.SLACK:
        return True
    for ld in net.loads.values():
        if ld.bus_id == bus_id:
            return True
    for gen in net.generators.values():
        if gen.bus_id == bus_id:
            return True
    for sh in net.shunts.values():
        if sh.bus_id == bus_id:
            return True
    return False


# ---------------------------------------------------------------------------
# Graph builders
# ---------------------------------------------------------------------------

def build_network_graph(net: Network, per_phase: bool = False) -> nx.Graph:
    """Build a networkx graph from the LFE network.

    If *per_phase* is False, nodes are bus IDs and edges are branch/transformer
    elements.  If *per_phase* is True, nodes are ``(bus_id, phase_index)``
    tuples.
    """
    graph = nx.Graph()

    if per_phase:
        # Add per-phase nodes for every bus
        for bus_id, bus in net.buses.items():
            for p in _active_phases(bus.phase_type):
                graph.add_node((bus_id, p))

        # Branch edges — same active phases on both ends
        for br_id, br in net.branches.items():
            for p in _active_phases(br.phase_type):
                graph.add_edge(
                    (br.from_bus, p), (br.to_bus, p),
                    element="branch", index=br_id,
                )

        # Transformer edges
        for xf_id, xf in net.transformers.items():
            for p in _active_phases(xf.phase_type):
                graph.add_edge(
                    (xf.from_bus, p), (xf.to_bus, p),
                    element="transformer", index=xf_id,
                )
    else:
        graph.add_nodes_from(net.buses.keys())

        for br_id, br in net.branches.items():
            graph.add_edge(br.from_bus, br.to_bus, element="branch", index=br_id)

        for xf_id, xf in net.transformers.items():
            graph.add_edge(xf.from_bus, xf.to_bus, element="transformer", index=xf_id)

        for sw_id, sw in net.switches.items():
            if sw.closed and sw.et == 'b':
                bus_a = str(sw.bus)
                bus_b = str(sw.element)
                if bus_a in net.buses and bus_b in net.buses:
                    graph.add_edge(bus_a, bus_b, element="switch", index=sw_id)

    return graph


def bfs_from_sources(net: Network, graph: nx.Graph | None = None) -> set:
    """Return set of bus IDs (or (bus, phase) tuples) reachable from SLACK buses."""
    if graph is None:
        graph = build_network_graph(net)

    source_buses = get_source_buses(net)
    reachable: set = set()
    for src in source_buses:
        if src in graph:
            reachable.update(nx.node_connected_component(graph, src))
    return reachable
