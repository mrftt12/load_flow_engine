"""Helper functions for impedance transforms and phase mapping."""

from typing import List, Optional

import numpy as np

from .enums import PhaseType


def _active_phases(phase_type: PhaseType) -> List[int]:
    """
    Return the sorted list of active phase indices (0=A, 1=B, 2=C)
    for a given PhaseType, matching ETAPPhaseType handling in
    NetworkReductionEC.cpp.
    """
    mapping = {
        PhaseType.ABC: [0, 1, 2],
        PhaseType.A:   [0],
        PhaseType.B:   [1],
        PhaseType.C:   [2],
        PhaseType.AB:  [0, 1],
        PhaseType.BC:  [1, 2],
        PhaseType.CA:  [0, 2],
    }
    return sorted(mapping.get(phase_type, [0, 1, 2]))


def _seq_to_z_abc(z1: complex, z0: complex) -> np.ndarray:
    """
    Convert positive- and zero-sequence impedances to the 3×3 phase-domain
    impedance matrix for a fully transposed line.

    Mirrors NetworkReductionEC's use of the symmetrical-component transform:
        Z_abc = A · diag(Z0, Z1, Z1) · A⁻¹

    For a transposed line this reduces to the Toeplitz form:
        Zs = (Z0 + 2·Z1) / 3    (self impedance)
        Zm = (Z0 −    Z1) / 3   (mutual impedance)

        Z_abc = [[Zs, Zm, Zm],
                 [Zm, Zs, Zm],
                 [Zm, Zm, Zs]]
    """
    Zs = (z0 + 2.0 * z1) / 3.0
    Zm = (z0       - z1) / 3.0
    return np.array([[Zs, Zm, Zm],
                     [Zm, Zs, Zm],
                     [Zm, Zm, Zs]], dtype=complex)


def _matrix_invert_3x3(Z: np.ndarray) -> Optional[np.ndarray]:
    """
    Invert a 3×3 complex matrix using Gauss-Jordan elimination.
    Mirrors MatrixInvert(DComplexEC mtxZ[3][3]) in NetworkReductionEC.cpp,
    including the same singular-matrix guard (returns None on failure).
    """
    SMALL = 1e-12
    A = Z.astype(complex).copy()
    I = np.eye(3, dtype=complex)

    for i in range(3):
        # Partial pivoting (mirrors ETAP's row-sum fallback)
        if abs(A[i, i]) < SMALL:
            swapped = False
            for k in range(i + 1, 3):
                if abs(A[k, i]) >= SMALL:
                    A[[i, k]] = A[[k, i]]
                    I[[i, k]] = I[[k, i]]
                    swapped = True
                    break
            if not swapped:
                return None   # singular

        pivot = A[i, i]
        A[i] /= pivot
        I[i] /= pivot

        for k in range(3):
            if k != i and abs(A[k, i]) >= SMALL:
                factor = A[k, i]
                A[k] -= factor * A[i]
                I[k] -= factor * I[i]

    return I
