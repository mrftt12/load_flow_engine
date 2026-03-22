"""
CIM / CGMES adapter for LFE networks.

Import CIM XML into an LFE :class:`~load_flow_engine.network.Network` and
export a Network back to CIM XML (IEC 61970 CIM100 RDF).

Usage::

    from load_flow_engine.tools.cim_adapter import CIMAdapter

    adapter = CIMAdapter()
    net = adapter.import_cim("feeder.xml")          # CIM → LFE
    adapter.export_cim(net, "feeder_out.xml")        # LFE → CIM
"""

import logging
import math
import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from uuid import NAMESPACE_OID, uuid4, uuid5

import numpy as np

from ..enums import PhaseType, BusType
from ..models import Bus, Branch, Transformer, Load, Generator, StudyCase
from ..network import Network

logger = logging.getLogger(__name__)

CIM_NS = "http://iec.ch/TC57/CIM100#"
RDF_NS = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"

# Namespace prefixes for ElementTree XPath
_NS = {"cim": CIM_NS, "rdf": RDF_NS}


# ---------------------------------------------------------------------------
# Configuration dataclasses
# ---------------------------------------------------------------------------

@dataclass
class CIMExportConfig:
    base_name: str = "lfe_network"
    include_transformers: bool = True
    include_switches: bool = True
    include_generators: bool = True
    include_loads: bool = True
    include_lines: bool = True


@dataclass
class CGMESExportConfig:
    base_name: str = "lfe_network"
    output_dir: str = "."
    profiles: Tuple[str, ...] = ("EQ", "TP", "SSH", "SV")
    include_transformers: bool = True
    include_switches: bool = True
    include_generators: bool = True
    include_loads: bool = True
    include_lines: bool = True
    include_sv: bool = True
    include_ssh: bool = True


# ---------------------------------------------------------------------------
# Adapter class
# ---------------------------------------------------------------------------

class CIMAdapter:
    """Import / export CIM XML to and from LFE networks."""

    def __init__(self, base_mva: float = 100.0) -> None:
        self.base_mva = base_mva

    # -----------------------------------------------------------------------
    # Import  (CIM XML → LFE Network)
    # -----------------------------------------------------------------------

    def import_cim(self, cim_path: str, **kwargs) -> Network:
        """Parse a CIM100 RDF/XML file and return an LFE Network.

        The importer recognises the following CIM classes:

        * ``TopologicalNode`` / ``ConnectivityNode`` → Bus
        * ``ACLineSegment`` → Branch
        * ``PowerTransformer`` → Transformer
        * ``EnergyConsumer`` → Load
        * ``EnergySource`` / ``SynchronousMachine`` → Generator
        """
        if not os.path.exists(cim_path):
            raise FileNotFoundError(f"CIM file not found: {cim_path}")

        tree = ET.parse(cim_path)
        root = tree.getroot()

        sc = StudyCase(base_mva=self.base_mva)
        net = Network(sc)

        # 1. Parse buses from TopologicalNode or ConnectivityNode
        bus_map = self._parse_buses(root, net)

        # 2. Parse terminals (maps equipment mRID → ordered bus list)
        terminal_map = self._parse_terminals(root, bus_map)

        # 3. Parse lines
        self._parse_lines(root, net, terminal_map)

        # 4. Parse transformers
        self._parse_transformers(root, net, terminal_map, bus_map)

        # 5. Parse loads
        self._parse_loads(root, net, terminal_map)

        # 6. Parse generators
        self._parse_generators(root, net, terminal_map)

        return net

    def import_cgmes(
        self,
        file_list: List[str],
        **kwargs,
    ) -> Network:
        """Import a set of CGMES profile files into a single LFE Network.

        All files are parsed and their elements are merged.  The EQ profile
        provides topology and equipment; SSH provides operating values.
        """
        if not file_list:
            raise ValueError("file_list must contain at least one file")

        sc = StudyCase(base_mva=self.base_mva)
        net = Network(sc)
        bus_map: Dict[str, str] = {}
        terminal_map: Dict[str, List[str]] = {}

        for path in _normalize_file_list(file_list):
            tree = ET.parse(path)
            root = tree.getroot()
            bm = self._parse_buses(root, net)
            bus_map.update(bm)
            tm = self._parse_terminals(root, bus_map)
            terminal_map.update(tm)
            self._parse_lines(root, net, terminal_map)
            self._parse_transformers(root, net, terminal_map, bus_map)
            self._parse_loads(root, net, terminal_map)
            self._parse_generators(root, net, terminal_map)

        return net

    # ---- CIM XML parse helpers -------------------------------------------

    @staticmethod
    def _get_mrid(elem: ET.Element) -> str:
        """Extract the mRID from a CIM element."""
        mrid_elem = elem.find(f"{{{CIM_NS}}}IdentifiedObject.mRID")
        if mrid_elem is not None and mrid_elem.text:
            return mrid_elem.text.strip()
        rdf_id = elem.get(f"{{{RDF_NS}}}ID")
        if rdf_id:
            return rdf_id
        rdf_about = elem.get(f"{{{RDF_NS}}}about")
        if rdf_about:
            return rdf_about.lstrip("#")
        return f"_unknown_{uuid4().hex[:8]}"

    @staticmethod
    def _get_name(elem: ET.Element) -> str:
        name_elem = elem.find(f"{{{CIM_NS}}}IdentifiedObject.name")
        return name_elem.text.strip() if name_elem is not None and name_elem.text else ""

    @staticmethod
    def _get_text(elem: ET.Element, tag: str) -> Optional[str]:
        child = elem.find(f"{{{CIM_NS}}}{tag}")
        return child.text.strip() if child is not None and child.text else None

    @staticmethod
    def _get_float(elem: ET.Element, tag: str, default: float = 0.0) -> float:
        text = CIMAdapter._get_text(elem, tag)
        if text is None:
            return default
        try:
            return float(text)
        except ValueError:
            return default

    @staticmethod
    def _get_ref(elem: ET.Element, tag: str) -> Optional[str]:
        child = elem.find(f"{{{CIM_NS}}}{tag}")
        if child is None:
            return None
        ref = child.get(f"{{{RDF_NS}}}resource")
        return ref.lstrip("#") if ref else None

    def _parse_buses(self, root: ET.Element, net: Network) -> Dict[str, str]:
        """Parse TopologicalNode and ConnectivityNode elements into LFE buses.

        Returns a mapping from CIM mRID → LFE bus id.
        """
        bus_map: Dict[str, str] = {}

        # Collect base voltages
        base_voltages: Dict[str, float] = {}
        for bv in root.iter(f"{{{CIM_NS}}}BaseVoltage"):
            bv_id = self._get_mrid(bv)
            nom_v = self._get_float(bv, "BaseVoltage.nominalVoltage", 0.0)
            base_voltages[bv_id] = nom_v / 1000.0 if nom_v > 100 else nom_v

        # Try TopologicalNode first
        topo_nodes = list(root.iter(f"{{{CIM_NS}}}TopologicalNode"))
        if not topo_nodes:
            topo_nodes = list(root.iter(f"{{{CIM_NS}}}ConnectivityNode"))

        for node in topo_nodes:
            mrid = self._get_mrid(node)
            name = self._get_name(node) or mrid
            bus_id = name.replace(" ", "_").replace(".", "_") or mrid

            # Avoid duplicate bus ids
            if bus_id in net.buses:
                bus_id = f"{bus_id}_{mrid[-6:]}"

            bv_ref = self._get_ref(node, "TopologicalNode.BaseVoltage")
            if bv_ref is None:
                bv_ref = self._get_ref(node, "ConnectivityNode.BaseVoltage")
            kv = base_voltages.get(bv_ref, 12.47) if bv_ref else 12.47

            net.add_bus(Bus(id=bus_id, bus_type=BusType.PQ,
                            phase_type=PhaseType.ABC, base_kv=kv, name=name))
            bus_map[mrid] = bus_id

        # Also map ConnectivityNode mRIDs to the same bus when they
        # reference a TopologicalNode that is already in bus_map.
        for cn in root.iter(f"{{{CIM_NS}}}ConnectivityNode"):
            cn_mrid = self._get_mrid(cn)
            if cn_mrid in bus_map:
                continue
            tn_ref = self._get_ref(cn, "ConnectivityNode.TopologicalNode")
            if tn_ref and tn_ref in bus_map:
                bus_map[cn_mrid] = bus_map[tn_ref]
            else:
                # Standalone CN (no TN) — create a bus for it if not present
                cn_name = self._get_name(cn) or cn_mrid
                cn_bus_id = cn_name.replace(" ", "_").replace(".", "_")
                if cn_bus_id not in net.buses:
                    bv_ref = self._get_ref(cn, "ConnectivityNode.BaseVoltage")
                    kv = base_voltages.get(bv_ref, 12.47) if bv_ref else 12.47
                    net.add_bus(Bus(id=cn_bus_id, bus_type=BusType.PQ,
                                    phase_type=PhaseType.ABC, base_kv=kv,
                                    name=cn_name))
                bus_map[cn_mrid] = cn_bus_id

        return bus_map

    def _parse_terminals(self, root: ET.Element,
                         bus_map: Dict[str, str]) -> Dict[str, List[str]]:
        """Build equipment_mrid → [bus_id, ...] from Terminal elements."""
        terminal_map: Dict[str, List[str]] = {}

        for term in root.iter(f"{{{CIM_NS}}}Terminal"):
            equip_ref = self._get_ref(term, "Terminal.ConductingEquipment")
            conn_ref = self._get_ref(term, "Terminal.ConnectivityNode")
            topo_ref = self._get_ref(term, "Terminal.TopologicalNode")
            seq_text = self._get_text(term, "Terminal.sequenceNumber")
            seq = int(seq_text) if seq_text else 1

            node_ref = topo_ref or conn_ref
            if equip_ref is None or node_ref is None:
                continue

            bus_id = bus_map.get(node_ref)
            if bus_id is None:
                continue

            if equip_ref not in terminal_map:
                terminal_map[equip_ref] = []
            # Ensure list is long enough for the sequence number
            while len(terminal_map[equip_ref]) < seq:
                terminal_map[equip_ref].append("")
            terminal_map[equip_ref][seq - 1] = bus_id

        return terminal_map

    def _parse_lines(self, root: ET.Element, net: Network,
                     terminal_map: Dict[str, List[str]]) -> None:
        """Parse ACLineSegment elements into LFE branches."""
        for line_elem in root.iter(f"{{{CIM_NS}}}ACLineSegment"):
            mrid = self._get_mrid(line_elem)
            name = self._get_name(line_elem) or mrid
            buses = terminal_map.get(mrid, [])
            if len(buses) < 2 or not buses[0] or not buses[1]:
                continue

            length_km = self._get_float(line_elem, "Conductor.length", 1.0)
            r_ohm_km = self._get_float(line_elem, "ACLineSegment.r", 0.01)
            x_ohm_km = self._get_float(line_elem, "ACLineSegment.x", 0.01)
            bch = self._get_float(line_elem, "ACLineSegment.bch", 0.0)

            from_bus = net.buses.get(buses[0])
            base_kv = from_bus.base_kv if from_bus else 12.47
            z_base = base_kv ** 2 / self.base_mva

            r1_pu = r_ohm_km * length_km / z_base
            x1_pu = x_ohm_km * length_km / z_base

            # Assume Z0 = 3 * Z1 if not provided
            r0_pu = r1_pu * 3.0
            x0_pu = x1_pu * 3.0

            br_id = name.replace(" ", "_").replace(".", "_") or mrid
            if br_id in net.branches:
                br_id = f"{br_id}_{mrid[-6:]}"

            net.add_branch(Branch(
                id=br_id, from_bus=buses[0], to_bus=buses[1],
                phase_type=PhaseType.ABC, name=name,
                r1=r1_pu, x1=x1_pu, r0=r0_pu, x0=x0_pu,
            ))

    def _parse_transformers(self, root: ET.Element, net: Network,
                            terminal_map: Dict[str, List[str]],
                            bus_map: Dict[str, str]) -> None:
        """Parse PowerTransformer + PowerTransformerEnd into LFE transformers."""
        # Collect ends by transformer mRID
        end_map: Dict[str, List[ET.Element]] = {}
        for end in root.iter(f"{{{CIM_NS}}}PowerTransformerEnd"):
            tr_ref = self._get_ref(end, "PowerTransformerEnd.PowerTransformer")
            if tr_ref:
                end_map.setdefault(tr_ref, []).append(end)

        for tr_elem in root.iter(f"{{{CIM_NS}}}PowerTransformer"):
            mrid = self._get_mrid(tr_elem)
            name = self._get_name(tr_elem) or mrid
            buses = terminal_map.get(mrid, [])
            ends = end_map.get(mrid, [])

            if len(buses) < 2 or not buses[0] or not buses[1]:
                continue

            # Sort ends by ratedU to determine HV/LV
            end_data = []
            for e in ends:
                rated_u = self._get_float(e, "PowerTransformerEnd.ratedU", 0.0)
                rated_s = self._get_float(e, "PowerTransformerEnd.ratedS", 0.0)
                end_num = int(self._get_text(e, "PowerTransformerEnd.endNumber") or "1")
                end_data.append((end_num, rated_u, rated_s))
            end_data.sort(key=lambda x: x[1], reverse=True)

            mva = (end_data[0][2] / 1e6) if end_data and end_data[0][2] > 0 else 1.0

            # Use a default leakage impedance if not available from CIM
            r1 = 0.005
            x1 = 0.06

            xf_id = name.replace(" ", "_").replace(".", "_") or mrid
            if xf_id in net.transformers:
                xf_id = f"{xf_id}_{mrid[-6:]}"

            net.add_transformer(Transformer(
                id=xf_id, from_bus=buses[0], to_bus=buses[1],
                phase_type=PhaseType.ABC, name=name,
                r1=r1, x1=x1, r0=r1, x0=x1,
                mva_rating=mva,
                conn_primary="wye_grounded",
                conn_secondary="wye_grounded",
            ))

    def _parse_loads(self, root: ET.Element, net: Network,
                     terminal_map: Dict[str, List[str]]) -> None:
        """Parse EnergyConsumer elements into LFE loads."""
        for load_elem in root.iter(f"{{{CIM_NS}}}EnergyConsumer"):
            mrid = self._get_mrid(load_elem)
            name = self._get_name(load_elem) or mrid
            buses = terminal_map.get(mrid, [])
            if not buses or not buses[0]:
                continue

            p_w = self._get_float(load_elem, "EnergyConsumer.p", 0.0)
            q_w = self._get_float(load_elem, "EnergyConsumer.q", 0.0)
            # CIM uses watts; LFE uses MW
            p_mw = p_w / 1e6
            q_mvar = q_w / 1e6

            ld_id = name.replace(" ", "_").replace(".", "_") or mrid
            if ld_id in net.loads:
                ld_id = f"{ld_id}_{mrid[-6:]}"

            # Split evenly across three phases
            net.add_load(Load(
                id=ld_id, bus_id=buses[0], phase_type=PhaseType.ABC,
                name=name,
                mw=np.array([p_mw / 3, p_mw / 3, p_mw / 3]),
                mvar=np.array([q_mvar / 3, q_mvar / 3, q_mvar / 3]),
            ))

    def _parse_generators(self, root: ET.Element, net: Network,
                          terminal_map: Dict[str, List[str]]) -> None:
        """Parse EnergySource and SynchronousMachine into LFE generators."""
        for tag in ("EnergySource", "SynchronousMachine"):
            for gen_elem in root.iter(f"{{{CIM_NS}}}{tag}"):
                mrid = self._get_mrid(gen_elem)
                name = self._get_name(gen_elem) or mrid
                buses = terminal_map.get(mrid, [])
                if not buses or not buses[0]:
                    continue

                p_max = self._get_float(gen_elem, "EnergySource.pMax", 0.0)
                p_mw = p_max / 1e6 if abs(p_max) > 1 else p_max

                gen_id = name.replace(" ", "_").replace(".", "_") or mrid
                if gen_id in net.generators:
                    gen_id = f"{gen_id}_{mrid[-6:]}"

                bus = net.buses.get(buses[0])
                if bus is not None:
                    bus.bus_type = BusType.SLACK

                net.add_generator(Generator(
                    id=gen_id, bus_id=buses[0], bus_type=BusType.PV,
                    name=name,
                    mw=np.array([p_mw / 3, p_mw / 3, p_mw / 3]),
                ))

    # -----------------------------------------------------------------------
    # Export  (LFE Network → CIM XML)
    # -----------------------------------------------------------------------

    def export_cim(
        self,
        net: Network,
        cim_path: str,
        *,
        config: Optional[CIMExportConfig] = None,
    ) -> None:
        """Export an LFE Network to a CIM100 RDF/XML file."""
        if config is None:
            config = CIMExportConfig()

        ET.register_namespace("cim", CIM_NS)
        ET.register_namespace("rdf", RDF_NS)
        rdf = ET.Element(f"{{{RDF_NS}}}RDF")

        def _new_id() -> str:
            return f"_{uuid4()}".upper()

        def _add_io(element: ET.Element, name: str,
                     mrid: Optional[str] = None) -> str:
            oid = mrid or _new_id()
            element.set(f"{{{RDF_NS}}}ID", oid)
            ET.SubElement(element, f"{{{CIM_NS}}}IdentifiedObject.mRID").text = oid
            ET.SubElement(element, f"{{{CIM_NS}}}IdentifiedObject.name").text = name
            return oid

        def _stable_id(prefix: str, key: str) -> str:
            return f"_{prefix}_{uuid5(NAMESPACE_OID, key).hex.upper()}"

        # CIM version header
        version = ET.SubElement(rdf, f"{{{CIM_NS}}}IEC61970CIMVersion")
        _add_io(version, "cim_version")
        ET.SubElement(version, f"{{{CIM_NS}}}IEC61970CIMVersion.version").text = "IEC61970CIM100"

        # Feeder container
        feeder = ET.SubElement(rdf, f"{{{CIM_NS}}}Feeder")
        feeder_id = _add_io(feeder, config.base_name)

        # Base voltages
        base_voltage_ids: Dict[float, str] = {}
        for b in net.buses.values():
            kv = b.base_kv
            if kv > 0 and kv not in base_voltage_ids:
                bv = ET.SubElement(rdf, f"{{{CIM_NS}}}BaseVoltage")
                bv_id = _add_io(bv, f"BaseV_{kv:.4f}")
                ET.SubElement(bv, f"{{{CIM_NS}}}BaseVoltage.nominalVoltage").text = f"{kv * 1000:.3f}"
                base_voltage_ids[kv] = bv_id

        # Topological and connectivity nodes for each bus
        topo_ids: Dict[str, str] = {}
        conn_ids: Dict[str, str] = {}

        for b in net.buses.values():
            topo = ET.SubElement(rdf, f"{{{CIM_NS}}}TopologicalNode")
            topo_id = _add_io(topo, b.name or b.id, _stable_id("TN", b.id))
            topo_ids[b.id] = topo_id

            conn = ET.SubElement(rdf, f"{{{CIM_NS}}}ConnectivityNode")
            conn_id = _add_io(conn, b.name or b.id, _stable_id("CN", b.id))
            conn_ids[b.id] = conn_id
            ET.SubElement(conn, f"{{{CIM_NS}}}ConnectivityNode.TopologicalNode").set(
                f"{{{RDF_NS}}}resource", f"#{topo_id}"
            )
            ET.SubElement(conn, f"{{{CIM_NS}}}ConnectivityNode.ConnectivityNodeContainer").set(
                f"{{{RDF_NS}}}resource", f"#{feeder_id}"
            )

        def _add_terminal(equip_id: str, bus_id: str, seq: int) -> str:
            term = ET.SubElement(rdf, f"{{{CIM_NS}}}Terminal")
            term_id = _add_io(term, f"T_{equip_id}_{seq}")
            ET.SubElement(term, f"{{{CIM_NS}}}Terminal.ConductingEquipment").set(
                f"{{{RDF_NS}}}resource", f"#{equip_id}"
            )
            ET.SubElement(term, f"{{{CIM_NS}}}Terminal.ConnectivityNode").set(
                f"{{{RDF_NS}}}resource", f"#{conn_ids[bus_id]}"
            )
            ET.SubElement(term, f"{{{CIM_NS}}}Terminal.sequenceNumber").text = str(seq)
            return term_id

        # Lines (branches)
        if config.include_lines:
            for br in net.branches.values():
                from_bus_obj = net.buses.get(br.from_bus)
                z_base = (from_bus_obj.base_kv ** 2 / net.study_case.base_mva
                          if from_bus_obj else 1.0)

                line = ET.SubElement(rdf, f"{{{CIM_NS}}}ACLineSegment")
                line_id = _add_io(line, br.name or br.id, _stable_id("LN", br.id))
                ET.SubElement(line, f"{{{CIM_NS}}}Conductor.length").text = "1.0"
                ET.SubElement(line, f"{{{CIM_NS}}}ACLineSegment.r").text = f"{br.r1 * z_base:.8g}"
                ET.SubElement(line, f"{{{CIM_NS}}}ACLineSegment.x").text = f"{br.x1 * z_base:.8g}"

                bv_id = base_voltage_ids.get(from_bus_obj.base_kv if from_bus_obj else 0)
                if bv_id:
                    ET.SubElement(line, f"{{{CIM_NS}}}ConductingEquipment.BaseVoltage").set(
                        f"{{{RDF_NS}}}resource", f"#{bv_id}"
                    )
                _add_terminal(line_id, br.from_bus, 1)
                _add_terminal(line_id, br.to_bus, 2)

        # Loads
        if config.include_loads:
            for ld in net.loads.values():
                p_w = sum(ld.mw) * 1e6
                q_w = sum(ld.mvar) * 1e6
                load = ET.SubElement(rdf, f"{{{CIM_NS}}}EnergyConsumer")
                load_id = _add_io(load, ld.name or ld.id, _stable_id("LD", ld.id))
                ET.SubElement(load, f"{{{CIM_NS}}}EnergyConsumer.p").text = f"{p_w:.6f}"
                ET.SubElement(load, f"{{{CIM_NS}}}EnergyConsumer.q").text = f"{q_w:.6f}"
                _add_terminal(load_id, ld.bus_id, 1)

        # Generators
        if config.include_generators:
            for gen in net.generators.values():
                p_w = sum(gen.mw) * 1e6
                source = ET.SubElement(rdf, f"{{{CIM_NS}}}EnergySource")
                source_id = _add_io(source, gen.name or gen.id, _stable_id("GN", gen.id))
                ET.SubElement(source, f"{{{CIM_NS}}}EnergySource.pMax").text = f"{p_w:.6f}"
                _add_terminal(source_id, gen.bus_id, 1)

        # Switches
        if config.include_switches:
            for sw in net.switches.values():
                if sw.et != "b":
                    continue
                sw_elem = ET.SubElement(rdf, f"{{{CIM_NS}}}Switch")
                sw_id = _add_io(sw_elem, sw.id, _stable_id("SW", sw.id))
                ET.SubElement(sw_elem, f"{{{CIM_NS}}}Switch.normalOpen").text = (
                    "false" if sw.closed else "true"
                )
                _add_terminal(sw_id, str(sw.bus), 1)
                _add_terminal(sw_id, str(sw.element), 2)

        # Transformers
        if config.include_transformers:
            for xf in net.transformers.values():
                tr = ET.SubElement(rdf, f"{{{CIM_NS}}}PowerTransformer")
                tr_id = _add_io(tr, xf.name or xf.id, _stable_id("XF", xf.id))

                for end_num, bus_id in enumerate(
                    [xf.from_bus, xf.to_bus], start=1
                ):
                    end = ET.SubElement(rdf, f"{{{CIM_NS}}}PowerTransformerEnd")
                    end_id = _add_io(end, f"{xf.id}_end_{end_num}")
                    ET.SubElement(end, f"{{{CIM_NS}}}PowerTransformerEnd.PowerTransformer").set(
                        f"{{{RDF_NS}}}resource", f"#{tr_id}"
                    )
                    ET.SubElement(end, f"{{{CIM_NS}}}PowerTransformerEnd.endNumber").text = str(end_num)
                    bus_obj = net.buses.get(bus_id)
                    rated_u = bus_obj.base_kv * 1000 if bus_obj else 0
                    rated_s = xf.mva_rating * 1e6
                    ET.SubElement(end, f"{{{CIM_NS}}}PowerTransformerEnd.ratedU").text = f"{rated_u:.3f}"
                    ET.SubElement(end, f"{{{CIM_NS}}}PowerTransformerEnd.ratedS").text = f"{rated_s:.3f}"
                    term_id = _add_terminal(tr_id, bus_id, end_num)
                    ET.SubElement(end, f"{{{CIM_NS}}}PowerTransformerEnd.Terminal").set(
                        f"{{{RDF_NS}}}resource", f"#{term_id}"
                    )

        os.makedirs(os.path.dirname(os.path.abspath(cim_path)), exist_ok=True)
        tree = ET.ElementTree(rdf)
        tree.write(cim_path, encoding="utf-8", xml_declaration=True)

    # -----------------------------------------------------------------------
    # CGMES multi-profile export
    # -----------------------------------------------------------------------

    def export_cgmes(
        self,
        net: Network,
        *,
        config: Optional[CGMESExportConfig] = None,
    ) -> List[str]:
        """Export an LFE Network to a set of CGMES profile XML files.

        Returns a list of file paths written.
        """
        if config is None:
            config = CGMESExportConfig()

        os.makedirs(config.output_dir, exist_ok=True)
        paths: List[str] = []
        profiles = {p.upper() for p in config.profiles}

        if "EQ" in profiles:
            path = os.path.join(config.output_dir, f"{config.base_name}_EQ.xml")
            self.export_cim(net, path, config=CIMExportConfig(
                base_name=config.base_name,
                include_transformers=config.include_transformers,
                include_switches=config.include_switches,
                include_generators=config.include_generators,
                include_loads=config.include_loads,
                include_lines=config.include_lines,
            ))
            paths.append(path)

        if "SSH" in profiles and config.include_ssh:
            path = os.path.join(config.output_dir, f"{config.base_name}_SSH.xml")
            self._export_ssh(net, path, config)
            paths.append(path)

        if "SV" in profiles and config.include_sv:
            path = os.path.join(config.output_dir, f"{config.base_name}_SV.xml")
            self._export_sv(net, path, config)
            paths.append(path)

        return paths

    def _export_ssh(self, net: Network, path: str,
                    config: CGMESExportConfig) -> None:
        """Write a minimal SSH profile with load/generator operating values."""
        ET.register_namespace("cim", CIM_NS)
        ET.register_namespace("rdf", RDF_NS)
        rdf = ET.Element(f"{{{RDF_NS}}}RDF")

        def _add_io(elem: ET.Element, name: str) -> str:
            oid = f"_{uuid4()}".upper()
            elem.set(f"{{{RDF_NS}}}ID", oid)
            ET.SubElement(elem, f"{{{CIM_NS}}}IdentifiedObject.mRID").text = oid
            ET.SubElement(elem, f"{{{CIM_NS}}}IdentifiedObject.name").text = name
            return oid

        model = ET.SubElement(rdf, f"{{{CIM_NS}}}FullModel")
        _add_io(model, f"{config.base_name}_SSH")
        ET.SubElement(model, f"{{{CIM_NS}}}Model.profile").text = "SSH"

        for ld in net.loads.values():
            p_w = sum(ld.mw) * 1e6
            q_w = sum(ld.mvar) * 1e6
            load = ET.SubElement(rdf, f"{{{CIM_NS}}}EnergyConsumer")
            _add_io(load, ld.name or ld.id)
            ET.SubElement(load, f"{{{CIM_NS}}}EnergyConsumer.p").text = f"{p_w:.6f}"
            ET.SubElement(load, f"{{{CIM_NS}}}EnergyConsumer.q").text = f"{q_w:.6f}"

        for gen in net.generators.values():
            p_w = sum(gen.mw) * 1e6
            source = ET.SubElement(rdf, f"{{{CIM_NS}}}EnergySource")
            _add_io(source, gen.name or gen.id)
            ET.SubElement(source, f"{{{CIM_NS}}}EnergySource.p").text = f"{p_w:.6f}"

        tree = ET.ElementTree(rdf)
        tree.write(path, encoding="utf-8", xml_declaration=True)

    def _export_sv(self, net: Network, path: str,
                   config: CGMESExportConfig) -> None:
        """Write a minimal SV profile with bus voltage results."""
        ET.register_namespace("cim", CIM_NS)
        ET.register_namespace("rdf", RDF_NS)
        rdf = ET.Element(f"{{{RDF_NS}}}RDF")

        def _add_io(elem: ET.Element, name: str) -> str:
            oid = f"_{uuid4()}".upper()
            elem.set(f"{{{RDF_NS}}}ID", oid)
            ET.SubElement(elem, f"{{{CIM_NS}}}IdentifiedObject.mRID").text = oid
            ET.SubElement(elem, f"{{{CIM_NS}}}IdentifiedObject.name").text = name
            return oid

        model = ET.SubElement(rdf, f"{{{CIM_NS}}}FullModel")
        _add_io(model, f"{config.base_name}_SV")
        ET.SubElement(model, f"{{{CIM_NS}}}Model.profile").text = "SV"

        for b in net.buses.values():
            sv = ET.SubElement(rdf, f"{{{CIM_NS}}}SvVoltage")
            _add_io(sv, f"SvVoltage_{b.id}")
            topo_id = f"_TN_{uuid5(NAMESPACE_OID, b.id).hex.upper()}"
            ET.SubElement(sv, f"{{{CIM_NS}}}SvVoltage.TopologicalNode").set(
                f"{{{RDF_NS}}}resource", f"#{topo_id}"
            )
            ET.SubElement(sv, f"{{{CIM_NS}}}SvVoltage.v").text = f"{b.v_mag[0]:.6f}"
            ET.SubElement(sv, f"{{{CIM_NS}}}SvVoltage.angle").text = f"{b.v_ang[0]:.6f}"

        tree = ET.ElementTree(rdf)
        tree.write(path, encoding="utf-8", xml_declaration=True)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _normalize_file_list(file_list: List[str]) -> List[str]:
    """Expand directories to individual XML/ZIP files."""
    normalized: List[str] = []
    for path in file_list:
        if not path:
            continue
        if os.path.isdir(path):
            for name in os.listdir(path):
                if name.lower().endswith((".xml", ".zip")):
                    normalized.append(os.path.join(path, name))
        elif os.path.isfile(path):
            normalized.append(path)
    return normalized
