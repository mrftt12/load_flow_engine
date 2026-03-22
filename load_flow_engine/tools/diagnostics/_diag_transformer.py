"""Category 2 — Transformer Modeling Errors.

Checks: tx_01, tx_03, tx_04, tx_06.
"""
from __future__ import annotations

import math
from typing import Any

from ...network import Network

from ._common import issue


def check_transformers(net: Network) -> list[dict[str, Any]]:
    """Run all transformer modeling checks."""
    issues: list[dict[str, Any]] = []
    issues.extend(_tx_01_zero_or_missing_impedance(net))
    issues.extend(_tx_03_mva_rating_out_of_range(net))
    issues.extend(_tx_04_tap_at_extreme(net))
    issues.extend(_tx_06_xr_ratio(net))
    return issues


# ---------------------------------------------------------------------------
# tx_01 — Zero or missing impedance
# ---------------------------------------------------------------------------

def _tx_01_zero_or_missing_impedance(net: Network) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    for xf_id, xf in net.transformers.items():
        if xf.x1 == 0 and xf.r1 == 0:
            issues.append(issue(
                "critical", "transformer_model", "transformer", xf_id, "r1/x1",
                f"Transformer {xf_id} has zero impedance (r1=0, x1=0).",
                "Set impedance to manufacturer nameplate value.",
            ))
        elif xf.x1 == 0:
            issues.append(issue(
                "critical", "transformer_model", "transformer", xf_id, "x1",
                f"Transformer {xf_id} has x1=0 (zero reactance).",
                "Set x1 to nameplate short-circuit reactance value.",
            ))
    return issues


# ---------------------------------------------------------------------------
# tx_03 — mva_rating out of range
# ---------------------------------------------------------------------------

def _tx_03_mva_rating_out_of_range(net: Network) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    for xf_id, xf in net.transformers.items():
        if xf.mva_rating <= 0:
            issues.append(issue(
                "high", "transformer_model", "transformer", xf_id, "mva_rating",
                f"Transformer {xf_id} mva_rating={xf.mva_rating} is non-positive.",
                "Set mva_rating to the nameplate apparent power rating.",
            ))
        elif xf.mva_rating > 500:
            issues.append(issue(
                "high", "transformer_model", "transformer", xf_id, "mva_rating",
                f"Transformer {xf_id} mva_rating={xf.mva_rating} exceeds 500 MVA "
                f"(unusual for distribution).",
                "Verify rating or check for unit errors.",
            ))
    return issues


# ---------------------------------------------------------------------------
# tx_04 — Tap position at extreme values
# ---------------------------------------------------------------------------

def _tx_04_tap_at_extreme(net: Network) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    for xf_id, xf in net.transformers.items():
        for tap_name, tap_val in [("tap_primary", xf.tap_primary),
                                   ("tap_secondary", xf.tap_secondary)]:
            if abs(tap_val - 1.0) > 0.10:
                issues.append(issue(
                    "medium", "transformer_model", "transformer", xf_id, tap_name,
                    f"Transformer {xf_id} {tap_name}={tap_val:.4f} deviates >10% from 1.0 pu.",
                    "Verify tap setting is intentional.",
                ))
    return issues


# ---------------------------------------------------------------------------
# tx_06 — X/R ratio out of typical range
# ---------------------------------------------------------------------------

def _tx_06_xr_ratio(net: Network) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    for xf_id, xf in net.transformers.items():
        if xf.r1 <= 0 or xf.x1 <= 0:
            continue
        xr = xf.x1 / xf.r1
        if xr < 2.0 or xr > 50.0:
            issues.append(issue(
                "low", "transformer_model", "transformer", xf_id, "x1/r1",
                f"Transformer {xf_id} X/R ratio = {xr:.1f} is outside typical range [2, 50].",
                "Verify impedance data against nameplate test report.",
            ))
    return issues
