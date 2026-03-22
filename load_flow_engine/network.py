"""
Network topology builder and admittance-matrix assembly.
Mirrors CNetworkReductionEC / CUnbalanceInterface.
"""

import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.sparse import lil_matrix, csc_matrix

from .enums import PhaseType, BusType
from .models import (StudyCase, Bus, Branch, Transformer,
                     Load, Generator, Shunt, Switch)
from .helpers import _active_phases, _seq_to_z_abc, _matrix_invert_3x3


class Network:
    """
    Holds all network elements and assembles the 3N×3N nodal admittance
    matrix Y_abc used by the load-flow solver.

    Workflow (mirrors ETAP's LF3PH_FILL_DATA_STRUCT fill sequence):
        net = Network(study_case)
        net.add_bus(...)
        net.add_branch(...)
        net.add_transformer(...)
        net.add_load(...)        # aggregates MW/MVAr onto bus
        net.add_generator(...)   # sets bus type and generation schedule
        net.build()              # assembles Y_abc
    """

    def __init__(self, study_case: Optional[StudyCase] = None):
        self.study_case:   StudyCase              = study_case or StudyCase()
        self.buses:        Dict[str, Bus]         = {}
        self.branches:     Dict[str, Branch]      = {}
        self.transformers: Dict[str, Transformer] = {}
        self.loads:        Dict[str, Load]        = {}
        self.generators:   Dict[str, Generator]   = {}
        self.shunts:       Dict[str, Shunt]       = {}
        self.switches:     Dict[str, Switch]      = {}

        # Populated by build()
        self._bus_order: List[str]      = []
        self._bus_index: Dict[str, int] = {}
        self.Y_abc:      Optional[np.ndarray] = None   # (3N × 3N) complex

    def __repr__(self) -> str:
        lines = ["This LFE network includes the following parameter tables:"]
        for name, collection in [
            ("bus", self.buses),
            ("branch", self.branches),
            ("transformer", self.transformers),
            ("load", self.loads),
            ("generator", self.generators),
            ("shunt", self.shunts),
            ("switch", self.switches),
        ]:
            count = len(collection)
            if count > 0:
                lines.append(f"  - {name} ({count} elements)")
        if hasattr(self, "res_bus") and self.res_bus is not None:
            lines.append(f"  - res_bus ({len(self.res_bus)} elements)")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Element registration
    # ------------------------------------------------------------------

    def add_bus(self, bus: Bus) -> None:
        self.buses[bus.id] = bus

    def add_branch(self, branch: Branch) -> None:
        self.branches[branch.id] = branch

    def add_transformer(self, xfmr: Transformer) -> None:
        self.transformers[xfmr.id] = xfmr

    def add_load(self, load: Load) -> None:
        """Aggregate load power onto the host bus (mirrors ETAP load fill)."""
        self.loads[load.id] = load
        if load.bus_id in self.buses:
            b      = self.buses[load.bus_id]
            phases = _active_phases(load.phase_type)
            for p in phases:
                b.load_mw[p]   += load.mw[p]
                b.load_mvar[p] += load.mvar[p]

    def add_generator(self, gen: Generator) -> None:
        """Register generator and push scheduled generation onto bus."""
        self.generators[gen.id] = gen
        if gen.bus_id in self.buses:
            b = self.buses[gen.bus_id]
            b.bus_type = gen.bus_type
            for p in range(3):
                b.gen_mw[p] += gen.mw[p]

    def add_shunt(self, shunt: Shunt) -> None:
        """Register shunt and aggregate reactive injection onto bus."""
        self.shunts[shunt.id] = shunt

    def add_switch(self, switch: Switch) -> None:
        """Register switch element."""
        self.switches[switch.id] = switch

    # ------------------------------------------------------------------
    # Y-matrix assembly  (mirrors NetworkReductionEC stamping logic)
    # ------------------------------------------------------------------

    def build(self) -> None:
        """
        Assemble the 3N×3N complex nodal admittance matrix Y_abc.

        For each series element:
          1. Build the 3×3 phase-domain Z_abc from sequence components
             (mirrors _seq_to_z_abc — uses Fortescue transformation)
          2. Reduce to the active-phase sub-matrix for partial-phase branches
          3. Invert Z to obtain the primitive admittance Y (mirrors
             MatrixInvert in NetworkReductionEC.cpp)
          4. Stamp self- and mutual-admittance blocks into Y_abc
        """
        self._bus_order = list(self.buses.keys())
        self._bus_index = {bid: i for i, bid in enumerate(self._bus_order)}
        N = len(self._bus_order)
        self.Y_abc = lil_matrix((3 * N, 3 * N), dtype=complex)

        for br in self.branches.values():
            self._stamp_branch(br)

        for xf in self.transformers.values():
            self._stamp_transformer(xf)

        # Convert to CSC for efficient arithmetic
        self.Y_abc = csc_matrix(self.Y_abc)

    def _primitive_admittance(self, r1: float, x1: float,
                               r0: float, x0: float,
                               phase_type: PhaseType,
                               element_id: str) -> Optional[Tuple[np.ndarray, List[int]]]:
        """
        Build the primitive phase-domain admittance matrix for an element.
        Returns (Y_prim, active_phase_indices) or None if the element is open.
        """
        z1 = complex(r1, x1)
        z0 = complex(r0, x0)

        if abs(z1) < 1e-15 and abs(z0) < 1e-15:
            return None     # open / zero-impedance element

        Z_full = _seq_to_z_abc(z1, z0)

        mask   = _active_phases(phase_type)
        Z_sub  = Z_full[np.ix_(mask, mask)]

        Y_sub  = _matrix_invert_3x3(Z_sub) if len(mask) == 3 else self._invert_small(Z_sub)
        if Y_sub is None:
            warnings.warn(f"Element {element_id}: singular Z matrix — skipping.")
            return None

        return Y_sub, mask

    @staticmethod
    def _invert_small(Z: np.ndarray) -> Optional[np.ndarray]:
        """Invert 1×1 or 2×2 sub-matrix (single- or two-phase branch)."""
        try:
            return np.linalg.inv(Z)
        except np.linalg.LinAlgError:
            return None

    def _stamp(self, fi: int, ti: int, Y_prim: np.ndarray, mask: List[int]) -> None:
        """
        Stamp a branch primitive admittance into Y_abc (sparse lil_matrix).

        Standard nodal admittance stamp (mirrors MatrixByMatrix stamping):
            Y_abc[f,f] += Y_prim   (self-admittance at from-bus)
            Y_abc[t,t] += Y_prim   (self-admittance at to-bus)
            Y_abc[f,t] -= Y_prim   (mutual admittance)
            Y_abc[t,f] -= Y_prim   (mutual admittance, transposed)
        """
        for a, pa in enumerate(mask):
            fr = 3 * fi + pa
            tr = 3 * ti + pa
            for b, pb in enumerate(mask):
                fc = 3 * fi + pb
                tc = 3 * ti + pb
                y = Y_prim[a, b]
                self.Y_abc[fr, fc] += y
                self.Y_abc[tr, tc] += y
                self.Y_abc[fr, tc] -= y
                self.Y_abc[tr, fc] -= y

    def _stamp_branch(self, br: Branch) -> None:
        """
        Stamp branch series admittance and π-model shunt charging into Y_abc.
        """
        result = self._primitive_admittance(br.r1, br.x1, br.r0, br.x0,
                                             br.phase_type, br.id)
        if result is None:
            return
        Y_prim, mask = result

        fi = self._bus_index[br.from_bus]
        ti = self._bus_index[br.to_bus]
        self._stamp(fi, ti, Y_prim, mask)

        # π-model shunt charging  B/2 at each end
        if abs(br.b1) > 1e-15:
            y_half  = 0.5j * br.b1 / len(mask)
            for p in mask:
                self.Y_abc[3 * fi + p, 3 * fi + p] += y_half
                self.Y_abc[3 * ti + p, 3 * ti + p] += y_half

    def _stamp_transformer(self, xf: Transformer) -> None:
        """
        Stamp a two-winding transformer using the off-nominal tap π-equivalent.

        For a transformer with turns ratio  t = tap_primary / tap_secondary:
            Y_ff +=  Y / t²
            Y_tt +=  Y
            Y_ft += −Y / t
            Y_tf += −Y / t
        """
        base_conv = self.study_case.base_mva / max(xf.mva_rating, 1e-6)
        r1_sys = xf.r1 * base_conv
        x1_sys = xf.x1 * base_conv

        # Zero-sequence blocked by delta windings
        if 'delta' in (xf.conn_primary, xf.conn_secondary):
            r0_sys = 1e6    # effectively open
            x0_sys = 1e6
        else:
            r0_sys = xf.r0 * base_conv
            x0_sys = xf.x0 * base_conv

        result = self._primitive_admittance(r1_sys, x1_sys, r0_sys, x0_sys,
                                             xf.phase_type, xf.id)
        if result is None:
            return
        Y_prim, mask = result

        t  = xf.tap_primary / xf.tap_secondary
        fi = self._bus_index[xf.from_bus]
        ti = self._bus_index[xf.to_bus]

        Y_ff = Y_prim / (t ** 2)
        Y_ft = Y_prim / t
        for a, pa in enumerate(mask):
            fr = 3 * fi + pa
            tr = 3 * ti + pa
            for b, pb in enumerate(mask):
                fc = 3 * fi + pb
                tc = 3 * ti + pb
                self.Y_abc[fr, fc] += Y_ff[a, b]
                self.Y_abc[tr, tc] += Y_prim[a, b]
                self.Y_abc[fr, tc] -= Y_ft[a, b]
                self.Y_abc[tr, fc] -= Y_ft[a, b]
