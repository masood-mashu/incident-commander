"""Reward shaping for the Incident Commander environment.

v2: composable rubric with named dimensions, stability tracking,
governance penalties, and escalating anti-gaming waste multipliers.
"""

from __future__ import annotations

from incident_commander.models import ActionResult


class RewardWeights:
    # ── Named rubric dimensions ──
    # Diagnosis quality: did the agent gather useful evidence?
    alpha = 1.2
    # Mitigation safety: did the agent choose the right fix?
    beta = 1.6
    # Stakeholder trust: was communication clear and timely?
    gamma = 0.8
    # Risk penalty: harmful or unsafe actions
    delta = 1.0
    # Waste penalty: redundant / irrelevant actions
    epsilon = 0.6
    # Outcome: successful resolution bonus
    eta = 2.0
    # Failure: unresolved or timeout penalty
    zeta = 2.2
    # Long-term stability: sustained health post-mitigation
    theta = 0.5
    # Governance penalty: violated policy / budget / compliance
    iota = 1.8


def compute_reward(
    action_result: ActionResult,
    *,
    consecutive_waste_count: int = 0,
) -> tuple[float, dict[str, float]]:
    """Return scalar reward and the individual reward terms.

    Parameters
    ----------
    action_result:
        The result of the agent's action.
    consecutive_waste_count:
        Number of consecutive wasted actions.  Drives an escalating
        waste multiplier (anti-gaming).
    """
    # Anti-gaming: escalating waste penalty for repeated junk actions.
    waste_multiplier = 1.0 + 0.3 * min(consecutive_waste_count, 5)

    breakdown = {
        "diagnostic": RewardWeights.alpha * action_result.evidence_delta,
        "action": RewardWeights.beta * action_result.action_quality,
        "communication": RewardWeights.gamma * action_result.communication_quality,
        "risk_penalty": RewardWeights.delta * action_result.risk_penalty,
        "waste_penalty": RewardWeights.epsilon * action_result.waste_penalty * waste_multiplier,
        "outcome": RewardWeights.eta * action_result.outcome_reward,
        "failure": RewardWeights.zeta * action_result.failure_penalty,
        "stability": RewardWeights.theta * action_result.stability_delta,
        "governance": RewardWeights.iota * action_result.governance_penalty,
    }
    reward = (
        breakdown["diagnostic"]
        + breakdown["action"]
        + breakdown["communication"]
        - breakdown["risk_penalty"]
        - breakdown["waste_penalty"]
        + breakdown["outcome"]
        - breakdown["failure"]
        + breakdown["stability"]
        - breakdown["governance"]
    )
    return reward, breakdown
