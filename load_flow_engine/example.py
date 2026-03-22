"""
Four-bus radial distribution feeder example.

Topology (12.47 kV primary, 4.16 kV secondary after transformer):

  BUS1 (slack)  ──── XFM1 ────  BUS2  ──── BR1 ────  BUS3  ──── BR2 ────  BUS4
   12.47 kV                    4.16 kV               4.16 kV              4.16 kV
  swing bus              3φ balanced load    3φ+1φ unbalanced load    1φ-A load only

BUS2: 3-phase balanced load  150 kW + j50 kVAr on each phase
BUS3: unbalanced load  300 kW/ph-A, 100 kW/ph-B, 200 kW/ph-C + reactive
BUS4: single-phase A load  400 kW + j200 kVAr on phase A only
"""

import numpy as np

from .enums import PhaseType, BusType
from .constants import _Ai
from .models import Bus, Branch, Transformer, Load, StudyCase
from .network import Network
from .solver import ThreePhaseLoadFlowSolver


def build_example_network() -> Network:
    sc  = StudyCase(max_iterations=50, solution_precision=1e-4, base_mva=10.0)
    net = Network(sc)

    # ---- Buses ----
    net.add_bus(Bus("BUS1", BusType.SLACK, PhaseType.ABC, base_kv=12.47))
    net.add_bus(Bus("BUS2", BusType.PQ,    PhaseType.ABC, base_kv=4.16))
    net.add_bus(Bus("BUS3", BusType.PQ,    PhaseType.ABC, base_kv=4.16))
    net.add_bus(Bus("BUS4", BusType.PQ,    PhaseType.ABC, base_kv=4.16))

    # ---- Transformer BUS1 → BUS2 ----
    # 3 MVA, 12.47/4.16 kV, wye-grounded / wye-grounded
    # Z1 = 1%, Z0 = 1% on transformer base
    net.add_transformer(Transformer(
        id="XFM1", from_bus="BUS1", to_bus="BUS2",
        r1=0.005, x1=0.06, r0=0.005, x0=0.06,
        mva_rating=3.0,
        conn_primary='wye_grounded', conn_secondary='wye_grounded',
    ))

    # ---- Branch BUS2 → BUS3  (3-phase overhead line, 1 km) ----
    # Z1 = 0.306 + j0.627 Ω/km,  Z0 = 0.745 + j1.2 Ω/km
    # Base Z = (4.16²/10) = 1.731 Ω  →  divide to get pu
    Zbase = 4.16**2 / 10.0   # 1.731 Ω
    net.add_branch(Branch(
        id="BR1", from_bus="BUS2", to_bus="BUS3",
        phase_type=PhaseType.ABC,
        r1=0.306/Zbase, x1=0.627/Zbase,
        r0=0.745/Zbase, x0=1.200/Zbase,
        b1=0.0,
        ampacity=np.full(3, 300.0),   # 300 A rated
    ))

    # ---- Branch BUS3 → BUS4  (single-phase A, 0.5 km) ----
    net.add_branch(Branch(
        id="BR2", from_bus="BUS3", to_bus="BUS4",
        phase_type=PhaseType.A,
        r1=(0.306*0.5)/Zbase, x1=(0.627*0.5)/Zbase,
        r0=(0.745*0.5)/Zbase, x0=(1.200*0.5)/Zbase,
        ampacity=np.array([200.0, 0.0, 0.0]),
    ))

    # ---- Loads ----
    # BUS2: balanced 3-phase load  150 kW + j50 kVAr / phase
    net.add_load(Load("LD2", "BUS2", PhaseType.ABC,
                      mw   = np.array([0.15, 0.15, 0.15]),
                      mvar = np.array([0.05, 0.05, 0.05])))

    # BUS3: unbalanced 3-phase load
    net.add_load(Load("LD3", "BUS3", PhaseType.ABC,
                      mw   = np.array([0.30, 0.10, 0.20]),
                      mvar = np.array([0.15, 0.05, 0.10])))

    # BUS4: single-phase A load  400 kW + j200 kVAr
    net.add_load(Load("LD4", "BUS4", PhaseType.A,
                      mw   = np.array([0.40, 0.0, 0.0]),
                      mvar = np.array([0.20, 0.0, 0.0])))

    return net


def run_example() -> None:
    print("\n" + "=" * 72)
    print("  ETAP-Style Three-Phase Unbalanced Load Flow  —  4-Bus Feeder")
    print("=" * 72)

    net    = build_example_network()
    solver = ThreePhaseLoadFlowSolver(net)

    converged = solver.solve()

    if not converged:
        print("WARNING: solver did not converge!")

    solver.print_bus_results()

    br_results = solver.compute_branch_results()
    solver.print_branch_results(br_results)

    # ---- Voltage unbalance at BUS3 ----
    b3 = net.buses["BUS3"]
    V_abc = np.array([
        b3.v_mag[p] * np.exp(1j * np.deg2rad(b3.v_ang[p]))
        for p in range(3)
    ])
    V_012     = _Ai @ V_abc
    vuf_neg   = abs(V_012[2]) / abs(V_012[1]) * 100.0
    vuf_zero  = abs(V_012[0]) / abs(V_012[1]) * 100.0
    print(f"  BUS3 Voltage Unbalance Factor (NEMA neg-seq) : {vuf_neg:.3f} %")
    print(f"  BUS3 Voltage Unbalance Factor (zero-seq)     : {vuf_zero:.3f} %")
    print()
