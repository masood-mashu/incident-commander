"""Core data models for the Incident Commander environment."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from enum import Enum
from typing import Any, Callable
import uuid


class StringEnum(str, Enum):
    """Enum that serializes cleanly to JSON and dicts."""


class IncidentFamily(StringEnum):
    BAD_DEPLOY = "bad_deploy"
    DATABASE_SATURATION = "database_saturation"
    FEATURE_FLAG = "feature_flag"
    THIRD_PARTY = "third_party_failure"
    MULTI_REGION_GOVERNANCE = "multi_region_governance"
    CERTIFICATE_EXPIRY = "certificate_expiry"
    CAPACITY_EXHAUSTION = "capacity_exhaustion"


class ActionType(StringEnum):
    QUERY_TOOL = "query_tool"
    PROPOSE_HYPOTHESIS = "propose_hypothesis"
    EXECUTE_MITIGATION = "execute_mitigation"
    ESCALATE = "escalate"
    UPDATE_STATUS = "update_status"
    CLOSE_INCIDENT = "close_incident"


class Severity(StringEnum):
    SEV1 = "sev1"
    SEV2 = "sev2"
    SEV3 = "sev3"


@dataclass
class ServiceNode:
    name: str
    tier: str
    dependencies: list[str] = field(default_factory=list)
    regions: list[str] = field(default_factory=lambda: ["global"])
    criticality: str = "high"
    cost_per_hour: float = 0.0
    data_residency: str | None = None


@dataclass
class ServiceGraph:
    topology_id: str
    entrypoints: list[str]
    nodes: dict[str, ServiceNode]


@dataclass
class ToolEvidence:
    tool_name: str
    target: str
    content: str
    useful: bool
    noisy: bool = False


@dataclass
class DelayedEffect:
    """A consequence that fires N steps after its trigger."""
    trigger_step: int
    target_service: str
    health_delta: float
    description: str
    source_action: str = ""


@dataclass
class GovernanceConstraint:
    """Policy / budget / legal constraint on mitigations."""
    constraint_type: str  # "cost_budget" | "data_residency" | "change_freeze"
    description: str
    blocked_mitigations: list[str] = field(default_factory=list)
    required_check: str | None = None  # tool query that reveals the constraint


@dataclass
class CausalEdge:
    """One edge in the hidden causal graph for an incident."""
    source: str
    target: str
    mechanism: str  # human-readable causal mechanism
    discoverable_via: str = ""  # which tool query reveals this edge


@dataclass
class IncidentScenario:
    scenario_id: str
    title: str
    family: IncidentFamily
    description: str
    topology_id: str
    initial_alert: str
    root_cause: str
    impacted_services: list[str]
    correct_mitigation: str
    required_escalation: bool = False
    preferred_escalation_target: str = "sre_lead"
    deploy_service: str | None = None
    region: str | None = None
    hypothesis_aliases: list[str] = field(default_factory=list)
    tool_data: dict[str, dict[str, str]] = field(default_factory=dict)
    runbook_notes: list[str] = field(default_factory=list)
    chat_messages: list[str] = field(default_factory=list)
    report_keywords: list[str] = field(default_factory=list)
    # ── v2 upgrade fields ──
    confounders: list[str] = field(default_factory=list)
    disambiguating_tool: str | None = None
    noise_level: float = 0.0
    governance_constraints: list[GovernanceConstraint] = field(default_factory=list)
    causal_edges: list[CausalEdge] = field(default_factory=list)
    delayed_effects_config: list[dict[str, Any]] = field(default_factory=list)
    secondary_mitigation: str | None = None


@dataclass
class IncidentAction:
    action_type: ActionType
    tool_name: str | None = None
    target: str | None = None
    cause: str | None = None
    mitigation: str | None = None
    message: str | None = None
    escalate_to: str | None = None
    report: str | None = None
    severity: Severity | None = None


@dataclass
class ActionResult:
    success: bool
    summary: str
    details: dict[str, Any] = field(default_factory=dict)
    evidence_delta: float = 0.0
    action_quality: float = 0.0
    communication_quality: float = 0.0
    risk_penalty: float = 0.0
    waste_penalty: float = 0.0
    outcome_reward: float = 0.0
    failure_penalty: float = 0.0
    done: bool = False
    stability_delta: float = 0.0
    governance_penalty: float = 0.0


@dataclass
class IncidentObservation:
    summary: str
    visible_alerts: list[str]
    tool_results: list[ToolEvidence]
    stakeholder_messages: list[str]
    hypotheses: list[str]
    service_health: dict[str, float]
    available_actions: list[str]
    step_count: int
    remaining_budget: int
    resolved: bool
    terminal_reason: str | None = None
    reward_breakdown: dict[str, float] = field(default_factory=dict)


@dataclass
class IncidentState:
    episode_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    scenario_id: str = ""
    scenario_family: str = ""
    scenario_title: str = ""
    step_count: int = 0
    max_steps: int = 12
    severity: str = Severity.SEV2.value
    resolved: bool = False
    closed: bool = False
    awaiting_recovery_verification: bool = False
    service_health: dict[str, float] = field(default_factory=dict)
    service_status: dict[str, str] = field(default_factory=dict)
    queried_tools: list[str] = field(default_factory=list)
    evidence: list[ToolEvidence] = field(default_factory=list)
    hypotheses: list[str] = field(default_factory=list)
    escalations: list[str] = field(default_factory=list)
    status_updates: list[str] = field(default_factory=list)
    timeline: list[str] = field(default_factory=list)
    active_policy_flags: list[str] = field(default_factory=list)
    root_cause: str = ""
    correct_mitigation: str = ""
    required_escalation: bool = False
    useful_actions: int = 0
    wasted_actions: int = 0
    harmful_actions: int = 0
    diagnosis_confidence: float = 0.0
    resolution_step: int | None = None
    last_reward_breakdown: dict[str, float] = field(default_factory=dict)
    history: list[dict[str, Any]] = field(default_factory=list)
    # ── v2 upgrade fields ──
    pending_effects: list[DelayedEffect] = field(default_factory=list)
    fired_effects: list[DelayedEffect] = field(default_factory=list)
    governance_violations: list[str] = field(default_factory=list)
    causal_trace: list[dict[str, str]] = field(default_factory=list)
    secondary_resolved: bool = False
    stability_score: float = 1.0


@dataclass
class StepResult:
    observation: IncidentObservation
    reward: float
    done: bool
    info: dict[str, Any] = field(default_factory=dict)


def to_plain_data(value: Any) -> Any:
    """Convert dataclasses and enums to plain Python data."""
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return {key: to_plain_data(val) for key, val in asdict(value).items()}
    if isinstance(value, dict):
        return {key: to_plain_data(val) for key, val in value.items()}
    if isinstance(value, list):
        return [to_plain_data(item) for item in value]
    return value
