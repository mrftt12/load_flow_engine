"""LFE Network Diagnostics Suite.

Usage::

    from load_flow_engine.tools.diagnostics import run_diagnostics

    result = run_diagnostics(net)
    result.issues          # list of issue dicts
    result.recommendations # prioritized corrective actions
    result.summary         # count by severity
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from ...network import Network

from ._diag_voltage_base import check_voltage_base
from ._diag_transformer import check_transformers
from ._diag_grounding import check_grounding
from ._diag_phase import check_phase_connectivity
from ._diag_impedance import check_impedance
from ._diag_open_conductor import check_open_conductor
from ._diag_load_model import check_load_model
from ._diag_controls import check_controls
from ._diag_duplicates import check_duplicates
from ._diag_topology import check_topology


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class ValidationResult:
    """Container for diagnostic results."""
    issues: list[dict[str, Any]] = field(default_factory=list)
    recommendations: list[dict[str, Any]] = field(default_factory=list)
    summary: dict[str, int] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Category registry
# ---------------------------------------------------------------------------

_ALL_CATEGORIES: dict[str, Any] = {
    "voltage_base": check_voltage_base,
    "transformer": check_transformers,
    "grounding": check_grounding,
    "phase": check_phase_connectivity,
    "impedance": check_impedance,
    "open_conductor": check_open_conductor,
    "load_model": check_load_model,
    "controls": check_controls,
    "duplicates": check_duplicates,
    "topology": check_topology,
}

_FAST_CATEGORIES = {"voltage_base", "duplicates", "impedance", "load_model"}

_SEVERITY_ORDER = {"critical": 1, "high": 2, "medium": 3, "low": 4, "info": 5}

_ISSUE_KEYS = [
    "severity", "check", "element_type", "element_index",
    "field", "message", "suggestion",
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_diagnostics(
    net: Network,
    *,
    categories: str | Sequence[str] = "all",
    severity_threshold: str = "info",
    fast_only: bool = False,
) -> ValidationResult:
    """Run diagnostic checks on an LFE network.

    Parameters
    ----------
    net : Network
        The LFE network object.
    categories : str or list of str
        ``"all"`` to run every category, or a list of category names.
    severity_threshold : str
        Lowest severity to include: ``"critical"``, ``"high"``,
        ``"medium"``, ``"low"``, or ``"info"``.
    fast_only : bool
        If True, only run fast checks (duplicates, voltage_base scalars,
        impedance ranges, load model basics).

    Returns
    -------
    ValidationResult
    """
    if categories == "all":
        selected = set(_ALL_CATEGORIES)
    else:
        selected = set(categories)

    if fast_only:
        selected = selected.intersection(_FAST_CATEGORIES)

    # Run checks
    all_issues: list[dict] = []
    for name in _ALL_CATEGORIES:
        if name not in selected:
            continue
        check_fn = _ALL_CATEGORIES[name]
        kwargs: dict[str, Any] = {}
        if name == "voltage_base" and fast_only:
            kwargs["include_bfs"] = False
        elif name == "impedance" and fast_only:
            kwargs["include_matrix_checks"] = False
        elif name == "load_model" and fast_only:
            kwargs["include_capacity_checks"] = False

        try:
            issues = check_fn(net, **kwargs)
            all_issues.extend(issues)
        except Exception as exc:
            all_issues.append({
                "severity": "high",
                "check": f"{name}_error",
                "element_type": "diagnostics",
                "element_index": name,
                "field": "exception",
                "message": f"Diagnostic category '{name}' raised: {exc}",
                "suggestion": "Investigate the error; the network data may be malformed.",
            })

    # Filter by severity threshold
    threshold_rank = _SEVERITY_ORDER.get(severity_threshold, 5)
    filtered = [
        i for i in all_issues
        if _SEVERITY_ORDER.get(i.get("severity", "info"), 99) <= threshold_rank
    ]

    if not filtered:
        return ValidationResult(issues=[], recommendations=[], summary={"info": 0})

    # Summary
    summary: dict[str, int] = {}
    for i in filtered:
        sev = i.get("severity", "info")
        summary[sev] = summary.get(sev, 0) + 1

    # Recommendations
    recommendations = _build_recommendations(filtered)

    return ValidationResult(issues=filtered, recommendations=recommendations, summary=summary)


# ---------------------------------------------------------------------------
# Recommendation builder
# ---------------------------------------------------------------------------

_ACTION_MAP = {
    "voltage_base": "Verify and correct bus nominal voltages (base_kv) and source voltage settings.",
    "transformer_model": "Fix transformer impedance, turns ratio, or tap settings per nameplate data.",
    "grounding": "Ensure neutral grounding paths are continuous and substation ground is defined.",
    "phase_connectivity": "Correct phase assignments and verify per-phase reachability from sources.",
    "impedance_data": "Verify conductor impedance data against standard libraries and check units.",
    "open_conductor": "Investigate broken conductor conditions and restore phase continuity.",
    "load_model": "Correct load power factor, voltage level, and balance across phases.",
    "control_error": "Fix control element references, setpoints, and switching thresholds.",
    "duplicate_equipment": "Remove or merge duplicate elements and resolve contradictory data.",
    "topology": "Repair network connectivity, remove unintended loops, and verify source paths.",
}


def _build_recommendations(issues: list[dict]) -> list[dict]:
    # Group by (check, severity) and count
    grouped: dict[tuple[str, str], int] = {}
    for i in issues:
        key = (i.get("check", ""), i.get("severity", "info"))
        grouped[key] = grouped.get(key, 0) + 1

    rows: list[dict] = []
    for (check, severity), count in sorted(
        grouped.items(),
        key=lambda kv: (_SEVERITY_ORDER.get(kv[0][1], 99), -kv[1]),
    ):
        rows.append({
            "priority": severity,
            "issue_type": check,
            "count": count,
            "recommendation": _ACTION_MAP.get(check, "Investigate and resolve reported issues."),
        })
    return rows
