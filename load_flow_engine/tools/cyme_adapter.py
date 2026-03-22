"""
Export an LFE Network to a CYME study via the cympy COM API.

Usage::

    from load_flow_engine.tools.cyme_adapter import network_to_cyme

    network_to_cyme(net, circuit_name="MY_CIRCUIT")

Requires the ``cympy`` package (ships with CYME installations).
"""

import logging
import math
from typing import Optional

import numpy as np

from ..enums import PhaseType, BusType
from ..helpers import _active_phases
from ..network import Network

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Phase mapping helpers
# ---------------------------------------------------------------------------
_PHASE_LETTER = {0: "A", 1: "B", 2: "C"}


def _phase_str(phase_type: PhaseType) -> str:
    """Return CYME-style phase string, e.g. ``'ABC'`` or ``'A'``."""
    return "".join(_PHASE_LETTER[p] for p in _active_phases(phase_type))


def _node_id(bus_id: str, circuit_name: str, head_bus: Optional[str]) -> str:
    """Map LFE bus id to a CYME node id.

    The slack (head) bus maps to the circuit head node so that sections
    connect to the source.
    """
    if bus_id == head_bus:
        return circuit_name
    return bus_id


# ---------------------------------------------------------------------------
# Source
# ---------------------------------------------------------------------------

def _create_source(net: Network, circuit_name: str) -> None:
    """Set the CYME equivalent source from the LFE slack bus."""
    import cympy  # type: ignore

    slack_bus = None
    for b in net.buses.values():
        if b.bus_type == BusType.SLACK:
            slack_bus = b
            break
    if slack_bus is None:
        logger.warning("No slack bus found in network")
        return

    kv_ll = slack_bus.base_kv
    vm_pu = slack_bus.v_mag[0]
    kv_ln = kv_ll * vm_pu / math.sqrt(3.0)

    cympy.study.SetValueTopo(
        kv_ll,
        "Sources[0].EquivalentSourceModels[0].EquivalentSource.KVLL",
        circuit_name,
    )
    for i in range(1, 4):
        cympy.study.SetValueTopo(
            kv_ln,
            f"Sources[0].EquivalentSourceModels[0]"
            f".EquivalentSource.OperatingVoltage{i}",
            circuit_name,
        )
    logger.info("Source set: %.4f kVLL, %.4f p.u.", kv_ll, vm_pu)


# ---------------------------------------------------------------------------
# Lines (branches)
# ---------------------------------------------------------------------------

def _create_branches(net: Network, circuit_name: str,
                     head_bus: Optional[str]) -> None:
    """Create CYME unbalanced-line equipment and devices from LFE branches."""
    import cympy  # type: ignore

    if not net.branches:
        return

    base_mva = net.study_case.base_mva

    for br_id, br in net.branches.items():
        from_bus = net.buses.get(br.from_bus)
        base_kv = from_bus.base_kv if from_bus else 12.47
        z_base = base_kv ** 2 / base_mva

        phases = _active_phases(br.phase_type)
        ph_str = _phase_str(br.phase_type)

        # Self-impedance in ohms (assume 1 km)
        r1_ohm = br.r1 * z_base
        x1_ohm = br.x1 * z_base
        r0_ohm = br.r0 * z_base
        x0_ohm = br.x0 * z_base

        eq_id = f"LFE_LINE_{br_id}"
        equip_type = 19  # UnbalancedLine

        check_eq = cympy.eq.GetEquipment(eq_id, equip_type)
        if check_eq is None:
            cympy.eq.Add(eq_id, equip_type)

        eq = cympy.eq.GetEquipment(eq_id, equip_type)
        eq.SetValue(1, "UserDefinedImpedances")

        for p_idx, ph in enumerate("ABC"):
            if p_idx in phases:
                eq.SetValue(r1_ohm, f"SelfResistance{ph}")
                eq.SetValue(x1_ohm, f"SelfReactance{ph}")
                eq.SetValue(br.ampacity[p_idx], f"NominalRating{ph}")
            else:
                eq.SetValue(0.0, f"SelfResistance{ph}")
                eq.SetValue(0.0, f"SelfReactance{ph}")
                eq.SetValue(0.0, f"NominalRating{ph}")

        # Mutual impedance from sequence components: Zm = (Z0 - Z1) / 3
        r_mut = (r0_ohm - r1_ohm) / 3.0
        x_mut = (x0_ohm - x1_ohm) / 3.0
        for pair in ("AB", "BC", "CA"):
            eq.SetValue(r_mut, f"MutualResistance{pair}")
            eq.SetValue(x_mut, f"MutualReactance{pair}")

        for ph in ("A", "B", "C"):
            eq.SetValue(0.0, f"ShuntSusceptance{ph}")
            eq.SetValue(0.0, f"ShuntConductance{ph}")
        for pair in ("AB", "BC", "CA"):
            eq.SetValue(0.0, f"MutualShuntSusceptance{pair}")
            eq.SetValue(0.0, f"MutualShuntConductance{pair}")

        section_id = f"LINE_{br_id}"
        from_node = _node_id(br.from_bus, circuit_name, head_bus)
        to_node = _node_id(br.to_bus, circuit_name, head_bus)
        dev_type = 12  # UnbalancedLine

        try:
            cympy.study.AddSection(
                section_id, circuit_name, section_id, dev_type,
                from_node, to_node,
            )
            section = cympy.study.GetSection(section_id)
            if section is not None:
                section.SetValue(ph_str, "Phase")
            cympy.study.ReplaceDevice(section_id, dev_type, dev_type, eq_id)
            cympy.study.SetValueDevice(1000, "Length", section_id, dev_type)
        except Exception as e:
            logger.warning("Branch %s: %s", br_id, e)

    logger.info("Created %d branches", len(net.branches))


# ---------------------------------------------------------------------------
# Transformers
# ---------------------------------------------------------------------------

def _create_transformers(net: Network, circuit_name: str,
                         head_bus: Optional[str]) -> None:
    """Create CYME transformer equipment and devices from LFE transformers."""
    import cympy  # type: ignore

    if not net.transformers:
        return

    for xf_id, xf in net.transformers.items():
        from_bus = net.buses.get(xf.from_bus)
        to_bus = net.buses.get(xf.to_bus)
        hv_kv = from_bus.base_kv if from_bus else 12.47
        lv_kv = to_bus.base_kv if to_bus else 4.16
        phases = _active_phases(xf.phase_type)
        ph_str = _phase_str(xf.phase_type)

        kva = xf.mva_rating * 1000.0
        z1_pct = math.sqrt(xf.r1 ** 2 + xf.x1 ** 2) * 100.0
        x1r1 = (math.sqrt(max(0, (xf.x1 / xf.r1) ** 2))
                if xf.r1 > 0 else 10.0)

        # Determine connection type
        conn_map = {
            ("wye_grounded", "wye_grounded"): "Yg_Yg",
            ("wye_grounded", "delta"): "Yg_D",
            ("delta", "wye_grounded"): "D_Yg",
            ("delta", "delta"): "D_D",
            ("wye", "wye"): "Y_Y",
        }
        connection = conn_map.get(
            (xf.conn_primary, xf.conn_secondary), "Yg_Yg"
        )

        nphases = len(phases)
        is_by_phase = nphases <= 3 and nphases != 3
        dev_type = 33 if is_by_phase else 1

        eq_id = f"LFE_TX_{xf_id}"
        check_eq = cympy.eq.GetEquipment(
            eq_id, cympy.enums.EquipmentType.Transformer
        )
        if check_eq is None:
            cympy.eq.Add(eq_id, cympy.enums.EquipmentType.Transformer)
        tx_eq = cympy.eq.GetEquipment(
            eq_id, cympy.enums.EquipmentType.Transformer
        )

        if dev_type == 33:
            tx_eq.SetValue("SinglePhase", "TransfoType")
        else:
            tx_eq.SetValue("ThreePhase", "TransfoType")

        tx_eq.SetValue(kva, "NominalRatingKVA")
        tx_eq.SetValue(hv_kv, "PrimaryVoltage")
        tx_eq.SetValue(lv_kv, "SecondaryVoltage")
        tx_eq.SetValue(z1_pct, "PositiveSequenceImpedancePercent")
        tx_eq.SetValue(x1r1, "XRRatio")
        tx_eq.SetValue(connection, "TransformerConnection")

        section_id = f"TX_{xf_id}-XFO"
        from_node = _node_id(xf.from_bus, circuit_name, head_bus)
        to_node = _node_id(xf.to_bus, circuit_name, head_bus)
        dev_name = f"TX_{xf_id}"

        try:
            cympy.study.AddSection(
                section_id, circuit_name, dev_name, dev_type,
                from_node, to_node,
            )
            section = cympy.study.GetSection(section_id)
            if section is not None:
                section.SetValue(ph_str, "Phase")

            if dev_type == 33:
                for ph_idx, ph in enumerate("ABC"):
                    if ph_idx in phases:
                        cympy.study.SetValueDevice(
                            eq_id,
                            f"PhaseTransformerID{ph_idx + 1}",
                            dev_name, dev_type,
                        )
                    else:
                        cympy.study.SetValueDevice(
                            "", f"PhaseTransformerID{ph_idx + 1}",
                            dev_name, dev_type,
                        )
                cympy.study.SetValueDevice(
                    connection, "Connection", dev_name, dev_type
                )
            else:
                cympy.study.SetValueDevice(
                    eq_id, "DeviceID", dev_name, dev_type
                )

            shift = 30 if "D" in connection else 0
            cympy.study.SetValueDevice(
                f"{shift}deg", "PhaseShift", dev_name, dev_type
            )
            cympy.study.SetValueNode(
                lv_kv, "UserDefinedBaseVoltage", to_node
            )
        except Exception as e:
            logger.warning("Transformer %s: %s", xf_id, e)

    logger.info("Created %d transformers", len(net.transformers))


# ---------------------------------------------------------------------------
# Loads
# ---------------------------------------------------------------------------

def _create_loads(net: Network, circuit_name: str,
                  head_bus: Optional[str]) -> None:
    """Create CYME spot loads from LFE loads."""
    import cympy  # type: ignore

    if not net.loads:
        return

    for ld_id, ld in net.loads.items():
        bus = net.buses.get(ld.bus_id)
        phases = _active_phases(ld.phase_type)
        ph_str = _phase_str(ld.phase_type)

        phase_pq: dict[str, dict[str, float]] = {}
        for p in phases:
            ph = "ABC"[p]
            kw = ld.mw[p] * 1000.0
            kvar = ld.mvar[p] * 1000.0
            if abs(kw) > 1e-9 or abs(kvar) > 1e-9:
                phase_pq[ph] = {"kw": kw, "kvar": kvar}

        if not phase_pq:
            continue

        section_id = f"LOAD_{ld_id}-L"
        dev_name = f"LOAD_{ld_id}"
        from_node = _node_id(ld.bus_id, circuit_name, head_bus)

        try:
            cympy.study.AddSection(
                section_id, circuit_name, dev_name,
                cympy.enums.DeviceType.SpotLoad, from_node,
            )
            section = cympy.study.GetSection(section_id)
            if section is not None:
                section.SetValue(ph_str, "Phase")

            spot_load = cympy.study.GetDevice(
                dev_name, cympy.enums.DeviceType.SpotLoad
            )
            spot_load.SetValue(
                "KW_KVAR",
                "CustomerLoads[0].CustomerLoadModels[0].LoadValueType",
            )

            for i, ph in enumerate(sorted(phase_pq.keys())):
                pq = phase_pq[ph]
                prefix = (
                    "CustomerLoads[0].CustomerLoadModels[0]"
                    f".CustomerLoadValues[{i}].LoadValue"
                )
                spot_load.SetValue(pq["kw"], f"{prefix}.KW")
                spot_load.SetValue(pq["kvar"], f"{prefix}.KVAR")

        except Exception as e:
            logger.warning("Load %s: %s", ld_id, e)

    logger.info("Created %d loads", len(net.loads))


# ---------------------------------------------------------------------------
# Generators
# ---------------------------------------------------------------------------

def _create_generators(net: Network, circuit_name: str,
                       head_bus: Optional[str]) -> None:
    """Create CYME ECG devices from LFE generators."""
    import cympy  # type: ignore

    if not net.generators:
        return

    for gen_id, gen in net.generators.items():
        bus = net.buses.get(gen.bus_id)
        base_kv = bus.base_kv if bus else 12.47
        total_kw = sum(gen.mw) * 1000.0
        ph_str = "ABC"

        section_id = f"GEN_{gen_id}-G"
        dev_name = f"GEN_{gen_id}"
        from_node = _node_id(gen.bus_id, circuit_name, head_bus)
        dev_type = cympy.enums.DeviceType.ElectronicConverterGenerator

        try:
            cympy.study.AddSection(
                section_id, circuit_name, dev_name, dev_type, from_node,
            )
            section = cympy.study.GetSection(section_id)
            if section is not None:
                section.SetValue(ph_str, "Phase")

            gen_dev = cympy.study.GetDevice(dev_name, dev_type)
            gen_dev.SetValue("Connected", "ConnectionStatus")
            gen_dev.SetValue(
                total_kw, "GenerationModels.Get(1).ActiveGeneration"
            )
            gen_dev.SetValue(
                -100, "GenerationModels.Get(1).PowerFactor"
            )
            gen_dev.SetValue(
                base_kv,
                "Inverter.ACDCConverterSettings.NominalACVoltage",
            )
        except Exception as e:
            logger.warning("Generator %s: %s", gen_id, e)

    logger.info("Created %d generators", len(net.generators))


# ---------------------------------------------------------------------------
# Shunts
# ---------------------------------------------------------------------------

def _create_shunts(net: Network, circuit_name: str,
                   head_bus: Optional[str]) -> None:
    """Create CYME shunt capacitors from LFE shunts."""
    import cympy  # type: ignore

    if not net.shunts:
        return

    for sh_id, sh in net.shunts.items():
        if not sh.closed:
            continue
        phases = _active_phases(sh.phase_type)
        ph_str = _phase_str(sh.phase_type)

        section_id = f"SHUNT_{sh_id}-CAP"
        dev_name = f"SHUNT_{sh_id}"
        from_node = _node_id(sh.bus_id, circuit_name, head_bus)

        try:
            cympy.study.AddSection(
                section_id, circuit_name, dev_name,
                cympy.enums.DeviceType.ShuntCapacitor, from_node,
            )
            section = cympy.study.GetSection(section_id)
            if section is not None:
                section.SetValue(ph_str, "Phase")

            cap_dev = cympy.study.GetDevice(
                dev_name, cympy.enums.DeviceType.ShuntCapacitor
            )
            kv_ln = sh.vn_kv / math.sqrt(3) if sh.vn_kv > 0 else 0
            cap_dev.SetValue(kv_ln, "KVLN")

            for ph_idx, ph in enumerate("ABC"):
                kvar = sh.q_mvar[ph_idx] * 1000.0 if ph_idx in phases else 0.0
                cap_dev.SetValue(kvar, f"FixedKVAR{ph}")
                cap_dev.SetValue(0.0, f"SwitchedKVAR{ph}")

        except Exception as e:
            logger.warning("Shunt %s: %s", sh_id, e)

    logger.info("Created %d shunts", len(net.shunts))


# ---------------------------------------------------------------------------
# Switches
# ---------------------------------------------------------------------------

def _create_switches(net: Network, circuit_name: str,
                     head_bus: Optional[str]) -> None:
    """Create CYME switches from LFE switch elements."""
    import cympy  # type: ignore

    if not net.switches:
        return

    for sw_id, sw in net.switches.items():
        if sw.et != "b":
            continue

        ph_str = "ABC"
        section_id = f"SW_{sw_id}"
        dev_name = f"SW_{sw_id}"
        from_node = _node_id(str(sw.bus), circuit_name, head_bus)
        to_node = _node_id(str(sw.element), circuit_name, head_bus)

        try:
            cympy.study.AddSection(
                section_id, circuit_name, dev_name,
                cympy.enums.DeviceType.Switch, from_node, to_node,
            )
            section = cympy.study.GetSection(section_id)
            if section is not None:
                section.SetValue(ph_str, "Phase")

            sw_dev = cympy.study.GetDevice(
                dev_name, cympy.enums.DeviceType.Switch
            )
            sw_dev.SetValue(
                ph_str if sw.closed else "None", "ClosedPhase"
            )
        except Exception as e:
            logger.warning("Switch %s: %s", sw_id, e)

    logger.info("Created %d switches", len(net.switches))


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def network_to_cyme(
    net: Network,
    circuit_name: str,
    *,
    sxst_path: Optional[str] = None,
    create_new_study: bool = False,
) -> bool:
    """Convert an LFE :class:`Network` to a CYME network model.

    Parameters
    ----------
    net : Network
        A populated LFE network.
    circuit_name : str
        CYME circuit / feeder name.
    sxst_path : str, optional
        If provided the CYME study is saved to this path.
    create_new_study : bool
        If ``True`` a new empty CYME study is created first.

    Returns
    -------
    bool
        ``True`` on success.
    """
    import cympy  # type: ignore

    if create_new_study:
        cympy.study.New()

    cympy.study.AddNetwork(circuit_name, cympy.enums.NetworkType.Feeder)

    # Determine the slack (head) bus
    head_bus: Optional[str] = None
    for b in net.buses.values():
        if b.bus_type == BusType.SLACK:
            head_bus = b.id
            break

    logger.info(
        "Converting LFE network to CYME circuit: %s (head_bus=%s)",
        circuit_name, head_bus,
    )

    _create_source(net, circuit_name)
    _create_branches(net, circuit_name, head_bus)
    _create_transformers(net, circuit_name, head_bus)
    _create_loads(net, circuit_name, head_bus)
    _create_generators(net, circuit_name, head_bus)
    _create_shunts(net, circuit_name, head_bus)
    _create_switches(net, circuit_name, head_bus)

    if sxst_path:
        try:
            cympy.study.Save(str(sxst_path), True)
            logger.info("Study saved to %s", sxst_path)
        except Exception as e:
            logger.warning("Failed to save study: %s", e)

    logger.info("Conversion complete")
    return True
