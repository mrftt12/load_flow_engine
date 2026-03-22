"""
Export an LFE Network to an OpenDSS script.

Usage::

    from load_flow_engine.tools.opendss_adapter import network_to_opendss

    dss_script = network_to_opendss(net)
    dss_script = network_to_opendss(net, filename="my_network.dss")
"""

import math
from typing import Optional

import numpy as np

from ..enums import PhaseType, BusType
from ..network import Network


# ---------------------------------------------------------------------------
# Phase mapping helpers
# ---------------------------------------------------------------------------

_PHASE_LETTERS = {0: "1", 1: "2", 2: "3"}  # LFE index → DSS phase number


def _active_dss_phases(phase_type: PhaseType) -> list[int]:
    """Return DSS phase numbers (1-indexed) for *phase_type*."""
    from ..helpers import _active_phases
    return [p + 1 for p in _active_phases(phase_type)]


def _phase_str(phase_type: PhaseType) -> str:
    return ".".join(str(p) for p in _active_dss_phases(phase_type))


def _bus_dss(bus_id: str, phase_type: PhaseType) -> str:
    return f"{bus_id}.{_phase_str(phase_type)}"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def network_to_opendss(
    net: Network,
    filename: Optional[str] = None,
) -> str:
    """Convert an LFE :class:`Network` to an OpenDSS script string.

    Parameters
    ----------
    net : Network
        A populated (but not necessarily solved) LFE network.
    filename : str, optional
        If provided the DSS script is also written to this file.

    Returns
    -------
    str
        Complete OpenDSS script.
    """
    lines: list[str] = []
    lines.append("! OpenDSS script generated from LFE network")
    lines.append("Clear")
    lines.append("")

    _write_circuit(net, lines)
    _write_branches(net, lines)
    _write_transformers(net, lines)
    _write_loads(net, lines)
    _write_generators(net, lines)
    _write_shunts(net, lines)
    _write_switches(net, lines)
    _write_footer(net, lines)

    script = "\n".join(lines)
    if filename is not None:
        with open(filename, "w") as fh:
            fh.write(script)
    return script


# ---------------------------------------------------------------------------
# Internal writers
# ---------------------------------------------------------------------------

def _write_circuit(net: Network, lines: list[str]) -> None:
    """Write the ``New Circuit`` (VSource) command for the slack bus."""
    slack_bus = None
    for b in net.buses.values():
        if b.bus_type == BusType.SLACK:
            slack_bus = b
            break

    if slack_bus is None:
        # Fall back to first bus
        slack_bus = next(iter(net.buses.values()))

    vn_kv = slack_bus.base_kv
    vm_pu = slack_bus.v_mag[0]
    va_deg = slack_bus.v_ang[0]
    nphases = len(_active_dss_phases(slack_bus.phase_type))

    name = slack_bus.name or slack_bus.id or "lfe_circuit"
    name = name.replace(" ", "_")

    lines.append(
        f"New Circuit.{name} basekv={vn_kv} pu={vm_pu:.6f} "
        f"angle={va_deg:.2f} phases={nphases}"
    )
    lines.append(
        f"~ bus1={_bus_dss(slack_bus.id, slack_bus.phase_type)}"
    )
    lines.append("")


def _write_branches(net: Network, lines: list[str]) -> None:
    """Write ``New Line`` commands for every branch."""
    if not net.branches:
        return

    lines.append("! Lines")
    base_mva = net.study_case.base_mva

    for br in net.branches.values():
        nphases = len(_active_dss_phases(br.phase_type))
        from_bus = net.buses.get(br.from_bus)
        base_kv = from_bus.base_kv if from_bus else 12.47
        z_base = base_kv ** 2 / base_mva  # ohms

        # Convert per-unit to ohms/km (assume 1 km length)
        r1_ohm = br.r1 * z_base
        x1_ohm = br.x1 * z_base
        r0_ohm = br.r0 * z_base
        x0_ohm = br.x0 * z_base

        line_name = (br.name or br.id).replace(" ", "_").replace(".", "_")
        cmd = (
            f"New Line.{line_name} phases={nphases} "
            f"Bus1={_bus_dss(br.from_bus, br.phase_type)} "
            f"Bus2={_bus_dss(br.to_bus, br.phase_type)} "
            f"r1={r1_ohm:.8g} x1={x1_ohm:.8g} "
            f"r0={r0_ohm:.8g} x0={x0_ohm:.8g} "
            f"Length=1 units=km"
        )
        if abs(br.b1) > 1e-15:
            c1_nf = br.b1 / (2.0 * math.pi * 60.0) * 1e9 / z_base
            cmd += f" c1={c1_nf:.8g}"
        lines.append(cmd)

    lines.append("")


def _write_transformers(net: Network, lines: list[str]) -> None:
    """Write ``New Transformer`` commands."""
    if not net.transformers:
        return

    lines.append("! Transformers")
    for xf in net.transformers.values():
        nphases = len(_active_dss_phases(xf.phase_type))
        from_bus = net.buses.get(xf.from_bus)
        to_bus = net.buses.get(xf.to_bus)
        kv_primary = from_bus.base_kv if from_bus else 12.47
        kv_secondary = to_bus.base_kv if to_bus else 4.16
        kva = xf.mva_rating * 1000.0

        # xhl = total leakage reactance %  (on xfmr base)
        z_pct = math.sqrt(xf.r1 ** 2 + xf.x1 ** 2) * 100.0
        r_pct = xf.r1 * 100.0

        conn_map = {
            "wye_grounded": "wye",
            "wye": "wye",
            "delta": "delta",
        }
        conn1 = conn_map.get(xf.conn_primary, "wye")
        conn2 = conn_map.get(xf.conn_secondary, "wye")

        trafo_name = (xf.name or xf.id).replace(" ", "_").replace(".", "_")
        lines.append(
            f"New Transformer.{trafo_name} phases={nphases} windings=2 "
            f"xhl={z_pct:.6g} %R={r_pct:.6g}"
        )
        lines.append(
            f"~ wdg=1 bus={_bus_dss(xf.from_bus, xf.phase_type)} "
            f"kV={kv_primary:.6g} kVA={kva:.6g} conn={conn1} "
            f"tap={xf.tap_primary:.6g}"
        )
        lines.append(
            f"~ wdg=2 bus={_bus_dss(xf.to_bus, xf.phase_type)} "
            f"kV={kv_secondary:.6g} kVA={kva:.6g} conn={conn2} "
            f"tap={xf.tap_secondary:.6g}"
        )

    lines.append("")


def _write_loads(net: Network, lines: list[str]) -> None:
    """Write ``New Load`` commands for every load."""
    if not net.loads:
        return

    lines.append("! Loads")
    from ..helpers import _active_phases

    for ld in net.loads.values():
        bus = net.buses.get(ld.bus_id)
        base_kv = bus.base_kv if bus else 12.47
        phases = _active_phases(ld.phase_type)

        for p in phases:
            ph_letter = "ABC"[p]
            p_kw = ld.mw[p] * 1000.0
            q_kvar = ld.mvar[p] * 1000.0
            if abs(p_kw) < 1e-9 and abs(q_kvar) < 1e-9:
                continue

            dss_phase = p + 1
            vln_kv = base_kv / math.sqrt(3)
            load_name = f"{ld.id}_{ph_letter}".replace(" ", "_").replace(".", "_")
            lines.append(
                f"New Load.{load_name} Bus1={ld.bus_id}.{dss_phase} "
                f"phases=1 kV={vln_kv:.6g} kW={p_kw:.6g} "
                f"kvar={q_kvar:.6g} model=1"
            )

    lines.append("")


def _write_generators(net: Network, lines: list[str]) -> None:
    """Write ``New Generator`` commands."""
    if not net.generators:
        return

    lines.append("! Generators")
    from ..helpers import _active_phases

    for gen in net.generators.values():
        bus = net.buses.get(gen.bus_id)
        base_kv = bus.base_kv if bus else 12.47
        nphases = 3  # generators are typically three-phase

        total_kw = sum(gen.mw) * 1000.0
        gen_name = (gen.name or gen.id).replace(" ", "_").replace(".", "_")
        lines.append(
            f"New Generator.{gen_name} Bus1={gen.bus_id}.1.2.3 "
            f"phases={nphases} kV={base_kv:.6g} "
            f"kW={total_kw:.6g} model=7"
        )

    lines.append("")


def _write_shunts(net: Network, lines: list[str]) -> None:
    """Write ``New Capacitor`` commands for shunt elements."""
    if not net.shunts:
        return

    lines.append("! Shunts")
    from ..helpers import _active_phases

    for sh in net.shunts.values():
        if not sh.closed:
            continue
        bus = net.buses.get(sh.bus_id)
        base_kv = bus.base_kv if bus else 12.47
        phases = _active_phases(sh.phase_type)
        nphases = len(phases)
        total_kvar = sum(sh.q_mvar[p] for p in phases) * 1000.0
        if abs(total_kvar) < 1e-9:
            continue

        cap_name = (sh.name or sh.id).replace(" ", "_").replace(".", "_")
        phase_str = ".".join(str(p + 1) for p in phases)
        lines.append(
            f"New Capacitor.{cap_name} Bus1={sh.bus_id}.{phase_str} "
            f"phases={nphases} kV={base_kv:.6g} kvar={total_kvar:.6g}"
        )

    lines.append("")


def _write_switches(net: Network, lines: list[str]) -> None:
    """Write bus-bus switches as short Line elements."""
    if not net.switches:
        return

    has_content = False
    for sw in net.switches.values():
        if sw.et != "b":
            continue
        if not has_content:
            lines.append("! Switches")
            has_content = True

        r_ohm = max(sw.r_ohm, 1e-3)
        sw_name = (sw.id).replace(" ", "_").replace(".", "_")
        cmd = (
            f"New Line.{sw_name} phases=3 "
            f"Bus1=bus_{sw.bus}.1.2.3 Bus2=bus_{sw.element}.1.2.3 "
            f"switch=y r1={r_ohm:.8g} x1=1e-3 "
            f"r0={r_ohm:.8g} x0=1e-3 Length=0.001 units=km"
        )
        if not sw.closed:
            cmd += " enabled=false"
        lines.append(cmd)

    if has_content:
        lines.append("")


def _write_footer(net: Network, lines: list[str]) -> None:
    """Write voltage base and solve commands."""
    voltages = sorted({round(b.base_kv, 6) for b in net.buses.values()
                       if b.base_kv > 0}, reverse=True)
    vbases = " ".join(str(v) for v in voltages)
    lines.append(f"Set voltagebases=[{vbases}]")
    lines.append("CalcVoltageBases")
    lines.append("Solve")
