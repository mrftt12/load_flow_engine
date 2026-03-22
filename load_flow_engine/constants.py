"""
Symmetrical-component (Fortescue) transformation matrices.
Mirrors ETAP's use of A-matrix for zero/pos/neg sequence decomposition.
"""

import numpy as np

_a   = np.exp(1j * 2.0 * np.pi / 3.0)          # 1∠120°
_A   = np.array([[1,      1,      1     ],       # Fortescue A
                 [1,      _a**2,  _a    ],
                 [1,      _a,     _a**2 ]], dtype=complex)
_Ai  = np.linalg.inv(_A)                        # A⁻¹
