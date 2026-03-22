"""Category 5 — Incorrect Impedance Data.

Checks: imp_01, imp_04, imp_06.
"""
from __future__ import annotations

import math
from typing import Any

from ...network import Network

from ._common import issue


def check_impedance(net: Network, *, include_matrix_checks: bool = True) -> list[dict[str, Any]]:
    """Run all impedance data checks."""
    issues: list[dict[str, Any]] = []
    issues.extend(_imp_01_zero_or_negative_self_impedance(net))
    if include_matrix_checks:
        issues.extend(_imp_04_zero_sequence_inconsistent(net))
    issues.extend(_imp_06_xr_ratio(net))
    return issues


# ---------------------------------------------------------------------------
# imp_01 — Zero or negative self-impedance
# ---------------------------------------------------------------------------

def _imp_01_zero_or_negative_self_impedance(net: Network) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    for br_id, br in net.branches.items():
        if br.r1 <= 0:
            issues.append(issue(
                "critical", "impedance_data", "branch", br_id, "r1",
                f"Branch {br_id} has non-positive resistance r1={br.r1}.",
                "Set resistance to a positive value based on conductor specifications.",
            ))
        if br.x1 == 0:
            issues.append(issue(
                "critical", "impedance_data", "branch", br_id, "x1",
                f"Branch {br_id} has zero reactance x1={br.x1}.",
                "Set reactance to a non-zero value for physical conductors.",
            ))
    return issues


# ---------------------------------------------------------------------------
# imp_04 — Zero-sequence impedance inconsistent with positive-sequence
# ---------------------------------------------------------------------------

def _imp_04_zero_sequence_inconsistent(net: Network) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    for br_id, br in net.branches.items():
        z1 = math.sqrt(br.r1**2 + br.x1**2)
        z0 = math.sqrt(br.r0**2 + br.x0**2)
        if z1 <= 0:
            continue
        if z0 <= 0:
            # Zero-sequence impedance is zero but positive-sequence isn't
            issues.append(issue(
                "medium", "impedance_data", "branch", br_id, "r0/x0",
                f"Branch {br_id} has zero zero-sequence impedance (r0={br.r0}, x0={br.x0}) "
                f"but non-zero positive-sequence.",
                "Set zero-sequence impedance based on conductor geometry.",
            ))
            continue
        ratio = z0 / z1
        if ratio < 1.0 or ratio > 10.0:
            issues.append(issue(
                "medium", "impedance_data", "branch", br_id, "z0/z1",
                f"Branch {br_id} z0/z1 ratio = {ratio:.2f} is outside typical range [1, 10].",
                "Verify zero-sequence impedance data against conductor geometry.",
            ))
    return issues


# ---------------------------------------------------------------------------
# imp_06 — X/R ratio unusually high or low
# ---------------------------------------------------------------------------

def _imp_06_xr_ratio(net: Network) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    for br_id, br in net.branches.items():
        if br.r1 <= 0 or br.x1 == 0:
            continue
        xr = abs(br.x1) / br.r1
        if xr < 0.2 or xr > 5.0:
            issues.append(issue(
                "low", "impedance_data", "branch", br_id, "x/r",
                f"Branch {br_id} X/R ratio = {xr:.2f} is outside typical distribution "
                f"range [0.2, 5.0].",
                "Verify conductor parameters or check for unusual line characteristics.",
            ))
    return issues
