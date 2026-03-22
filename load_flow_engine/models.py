"""Data classes mirroring ETAP DB accessor structures."""

from dataclasses import dataclass, field
from typing import List

import numpy as np

from .enums import PhaseType, BusType


@dataclass
class StudyCase:
    """
    Mirrors LF3PH_STUDY_CASE_DATA from dataTyp6.h.
    Controls solver parameters.
    """
    max_iterations:     int   = 100     # maximumIteration
    solution_precision: float = 1e-4    # solutionPrecision (MVA mismatch)
    base_mva:           float = 100.0   # system MVA base
    flat_start:         bool  = True    # use 1.0 pu / nominal angles


@dataclass
class Bus:
    """
    Mirrors IBusLF3PHAccessor from IBusLF3PH.h.

    All voltage magnitudes are in per-unit; angles in degrees.
    Powers are in MW and MVAr (converted to pu internally during solve).
    Array index 0=A, 1=B, 2=C throughout.
    """
    id:         str
    bus_type:   BusType   = BusType.PQ
    phase_type: PhaseType = PhaseType.ABC
    base_kv:    float     = 12.47          # m_BasekV
    name:       str       = ''             # MC source name for cross-referencing

    # Solved voltages — m_VMagA/B/C, m_VAngA/B/C
    v_mag: np.ndarray = field(
        default_factory=lambda: np.ones(3))
    v_ang: np.ndarray = field(
        default_factory=lambda: np.array([0.0, -120.0, 120.0]))

    # Initial voltage guess — m_IniVMagA/B/C, m_IniAngA/B/C
    ini_v_mag: np.ndarray = field(
        default_factory=lambda: np.ones(3))
    ini_v_ang: np.ndarray = field(
        default_factory=lambda: np.array([0.0, -120.0, 120.0]))

    # Scheduled generation per phase (MW, MVAr) — m_GenMWA/B/C, m_GenMvarA/B/C
    gen_mw:   np.ndarray = field(default_factory=lambda: np.zeros(3))
    gen_mvar: np.ndarray = field(default_factory=lambda: np.zeros(3))

    # Scheduled load per phase (MW, MVAr) — m_LoadMWA/B/C, m_LoadMvarA/B/C
    load_mw:   np.ndarray = field(default_factory=lambda: np.zeros(3))
    load_mvar: np.ndarray = field(default_factory=lambda: np.zeros(3))

    # Reactive limits (PV buses) — m_MvarMax, m_MvarMin
    mvar_max: float = 9999.0
    mvar_min: float = -9999.0


@dataclass
class Branch:
    """
    Mirrors IConnectLF3PHAccessor from IConnectLF3PH.h.

    Stores positive- and zero-sequence impedances in per-unit.
    The 3×3 phase-domain matrix Z_abc is computed at build time via the
    symmetrical-component transform (same as NetworkReductionEC).
    """
    id:         str
    from_bus:   str
    to_bus:     str
    phase_type: PhaseType = PhaseType.ABC
    name:       str       = ''

    # Positive-sequence series impedance (pu) — m_R, m_X
    r1: float = 0.0
    x1: float = 0.0

    # Zero-sequence series impedance (pu) — m_R0, m_X0
    r0: float = 0.0
    x0: float = 0.0

    # Shunt charging susceptance (pu, positive-sequence total line charging)
    b1: float = 0.0

    # Ampacity per phase (amps) for loading-% calc — m_AmpacityA/B/C
    ampacity: np.ndarray = field(default_factory=lambda: np.full(3, 9999.0))


@dataclass
class Transformer:
    """
    Two-winding transformer.
    Mirrors LF3PH_Xfmr2Data from dataTyp6.h.

    Impedances are in pu on the transformer's own MVA base; they are
    converted to the system base during Network.build().
    """
    id:           str
    from_bus:     str          # primary (HV) bus
    to_bus:       str          # secondary (LV) bus
    phase_type:   PhaseType  = PhaseType.ABC
    name:         str        = ''

    # Leakage impedance (pu on xfmr base) — positive-sequence
    r1: float = 0.0
    x1: float = 0.01

    # Zero-sequence leakage impedance
    r0: float = 0.0
    x0: float = 0.01

    # Transformer MVA rating (used for base conversion)
    mva_rating: float = 1.0

    # Off-nominal tap (pu of nominal ratio) — m_TapPctPrim / m_TapPctSec
    tap_primary:   float = 1.0
    tap_secondary: float = 1.0

    # Winding connections (affects zero-sequence path)
    # Supported: 'wye_grounded', 'wye', 'delta'
    conn_primary:   str = 'wye_grounded'
    conn_secondary: str = 'wye_grounded'

    # HV phases this transformer is connected to (0=A, 1=B, 2=C)
    hv_phases: List[int] = field(default_factory=lambda: [0, 1, 2])


@dataclass
class Load:
    """
    Static load element — mirrors LF3PH_StaticLoadData from dataTyp6.h.
    Per-phase MW and MVAr. add_load() aggregates these onto the bus.
    """
    id:         str
    bus_id:     str
    phase_type: PhaseType = PhaseType.ABC
    name:       str       = ''
    mw:   np.ndarray = field(default_factory=lambda: np.zeros(3))
    mvar: np.ndarray = field(default_factory=lambda: np.zeros(3))


@dataclass
class Generator:
    """
    Synchronous generator — mirrors LF3PH_SynGenData from dataTyp6.h.
    """
    id:       str
    bus_id:   str
    bus_type: BusType = BusType.PV
    name:     str     = ''

    mw:       np.ndarray = field(default_factory=lambda: np.zeros(3))
    v_set_pu: float      = 1.0
    mvar_max: float      = 9999.0
    mvar_min: float      = -9999.0


@dataclass
class Shunt:
    """
    Shunt element — per-phase MW and MVAr (capacitor/reactor banks).
    """
    id:         str
    bus_id:     str
    phase_type: PhaseType = PhaseType.ABC
    name:       str       = ''
    p_mw:  np.ndarray = field(default_factory=lambda: np.zeros(3))
    q_mvar: np.ndarray = field(default_factory=lambda: np.zeros(3))
    vn_kv:  float = 12.47
    closed: bool = True


@dataclass
class Switch:
    """
    Switch element — connects two buses with optional resistance.
    """
    id:      str
    bus:     int = 0
    element: int = 0
    et:      str = 'b'       # 'b' = bus-bus, 'l' = line
    sw_type: str = 'LBS'
    closed:  bool = True
    phase:   int = 0
    r_ohm:   float = 0.0


@dataclass
class BranchResult:
    """
    Post-solve per-branch results.
    Field names mirror LFSumBranchLF3PHAccessor from LFSumBranchLF3PH.h.
    """
    id:       str
    from_bus: str
    to_bus:   str

    # Per-phase current — m_LoadingAmpMagA/B/C, m_LoadingAmpAngA/B/C
    i_mag_abc: np.ndarray = field(default_factory=lambda: np.zeros(3))
    i_ang_abc: np.ndarray = field(default_factory=lambda: np.zeros(3))

    # Per-phase apparent power (MVA) from/to — m_LoadingInMVAA/B/C, m_LoadingOutMVAA/B/C
    mva_from: np.ndarray = field(default_factory=lambda: np.zeros(3))
    mva_to:   np.ndarray = field(default_factory=lambda: np.zeros(3))

    # Per-phase loading percent — m_Loading_A/B/C
    loading_pct: np.ndarray = field(default_factory=lambda: np.zeros(3))

    # Sequence currents — m_LoadingAmpMag0/1/2, m_LoadingAmpAng0/1/2
    i_mag_012: np.ndarray = field(default_factory=lambda: np.zeros(3))
    i_ang_012: np.ndarray = field(default_factory=lambda: np.zeros(3))

    # Unbalance factors — m_CUF2 (negative-seq), m_CUF0 (zero-seq)
    cuf2: float = 0.0
    cuf0: float = 0.0
