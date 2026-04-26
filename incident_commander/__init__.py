"""Incident Commander package exports.

The root package stays client-friendly by avoiding eager imports of server modules.
`IncidentCommanderEnvironment` remains available for compatibility via `__getattr__`.
"""

from __future__ import annotations

from incident_commander.client import IncidentCommanderEnv
from incident_commander.models import (
    ActionType,
    CausalEdge,
    DelayedEffect,
    GovernanceConstraint,
    IncidentAction,
    IncidentObservation,
    IncidentState,
    Severity,
    StepResult,
)
from incident_commander.causal_graph import evaluate_causal_faithfulness

__all__ = [
    "ActionType",
    "CausalEdge",
    "DelayedEffect",
    "GovernanceConstraint",
    "IncidentAction",
    "IncidentCommanderEnv",
    "IncidentCommanderEnvironment",
    "IncidentObservation",
    "IncidentState",
    "Severity",
    "StepResult",
    "evaluate_causal_faithfulness",
]


def __getattr__(name: str) -> object:
    if name == "IncidentCommanderEnvironment":
        from incident_commander.server.incident_environment import IncidentCommanderEnvironment

        return IncidentCommanderEnvironment
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
