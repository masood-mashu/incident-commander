"""Causal Incident Twin — hidden causal graph evaluation.

Each scenario defines a ground-truth causal DAG.  The agent never sees it during
training.  After an episode, the evaluator scores how faithfully the agent's
actions traced the true causal chain (causal faithfulness).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from incident_commander.models import CausalEdge, IncidentState


@dataclass
class CausalFaithfulnessResult:
    """Scores how well the agent's investigation traced the hidden causal graph."""

    total_edges: int
    discovered_edges: int
    faithfulness_score: float  # discovered / total
    correct_causal_actions: int
    spurious_actions: int
    edge_details: list[dict[str, Any]] = field(default_factory=list)


def evaluate_causal_faithfulness(
    causal_edges: list[CausalEdge],
    state: IncidentState,
) -> CausalFaithfulnessResult:
    """Score how many true causal edges the agent discovered through its actions.

    A causal edge is "discovered" if the agent queried the tool that reveals it
    (``edge.discoverable_via``) *or* if the agent's hypothesis or mitigation
    target aligns with nodes on that edge.
    """
    if not causal_edges:
        return CausalFaithfulnessResult(
            total_edges=0,
            discovered_edges=0,
            faithfulness_score=1.0,
            correct_causal_actions=0,
            spurious_actions=0,
        )

    queried_set = set(state.queried_tools)
    hypothesis_set = {h.lower() for h in state.hypotheses}

    # Collect all service names that the agent interacted with via mitigations.
    mitigation_targets: set[str] = set()
    for entry in state.history:
        if entry.get("action") == "execute_mitigation":
            summary = entry.get("summary", "")
            # The summary contains the target service name.
            mitigation_targets.add(summary.lower())

    discovered = 0
    edge_details: list[dict[str, Any]] = []

    for edge in causal_edges:
        found = False

        # Check if the agent queried the tool that reveals this edge.
        if edge.discoverable_via:
            for queried in queried_set:
                if edge.discoverable_via in queried:
                    found = True
                    break

        # Check if the agent hypothesized about the source node.
        if not found and edge.source.lower() in hypothesis_set:
            found = True

        # Check if the agent's hypothesis matches the mechanism.
        if not found:
            for hyp in hypothesis_set:
                if hyp in edge.mechanism.lower() or edge.source.lower() in hyp:
                    found = True
                    break

        if found:
            discovered += 1

        edge_details.append(
            {
                "source": edge.source,
                "target": edge.target,
                "mechanism": edge.mechanism,
                "discoverable_via": edge.discoverable_via,
                "discovered": found,
            }
        )

    # Count correct vs spurious actions.
    causal_nodes = set()
    for edge in causal_edges:
        causal_nodes.add(edge.source.lower())
        causal_nodes.add(edge.target.lower())

    correct_causal = 0
    spurious = 0
    for entry in state.history:
        action_relevant = False
        summary_lower = entry.get("summary", "").lower()
        for node in causal_nodes:
            if node in summary_lower:
                action_relevant = True
                break
        if action_relevant:
            correct_causal += 1
        else:
            spurious += 1

    faithfulness = discovered / len(causal_edges) if causal_edges else 1.0

    return CausalFaithfulnessResult(
        total_edges=len(causal_edges),
        discovered_edges=discovered,
        faithfulness_score=round(faithfulness, 4),
        correct_causal_actions=correct_causal,
        spurious_actions=spurious,
        edge_details=edge_details,
    )
