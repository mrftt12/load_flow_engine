"""Phase and bus type enumerations matching ETAP definitions."""

from enum import IntEnum


class PhaseType(IntEnum):
    """
    Matches ETAPPhaseType enum in NetworkReductionEC.h.
    Defines which phases a bus or branch participates in.
    """
    ABC = 0   # three-phase
    A   = 5   # single-phase A
    B   = 6   # single-phase B
    C   = 7   # single-phase C
    AB  = 8   # two-phase A-B
    BC  = 9   # two-phase B-C
    CA  = 10  # two-phase C-A


class BusType(IntEnum):
    """Standard power-flow bus types."""
    SLACK = 0   # swing bus — |V| and ∠V fixed per phase
    PV    = 1   # voltage-controlled — P and |V| fixed, Q within limits
    PQ    = 2   # load bus — P and Q fixed per phase
