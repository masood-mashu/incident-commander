"""Counterfactual evaluator — Decision Quality Delta.

For each step in an episode, fork the environment state, replay alternative
actions, and compare downstream cumulative reward vs the chosen action.
Produces a ``decision_quality_delta.json`` and chart.
"""

from __future__ import annotations

import copy
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from incident_commander.models import (
    ActionType,
    IncidentAction,
    IncidentObservation,
)
from incident_commander.causal_graph import evaluate_causal_faithfulness
from incident_commander.server.incident_environment import IncidentCommanderEnvironment

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "outputs" / "evals"


# ── Alternative action generators ──────────────────────────────────────────────

_ALTERNATIVE_ACTIONS: list[IncidentAction] = [
    IncidentAction(ActionType.QUERY_TOOL, tool_name="metrics", target="global"),
    IncidentAction(ActionType.QUERY_TOOL, tool_name="logs", target="global"),
    IncidentAction(ActionType.PROPOSE_HYPOTHESIS, cause="unknown"),
    IncidentAction(ActionType.ESCALATE, escalate_to="sre_lead", message="Requesting guidance."),
    IncidentAction(ActionType.UPDATE_STATUS, message="Investigating impact."),
    IncidentAction(ActionType.EXECUTE_MITIGATION, mitigation="rollback", target="global"),
]


def rollout_from_state(
    env: IncidentCommanderEnvironment,
    action: IncidentAction,
    remaining_steps: int,
    default_policy: Callable[[IncidentObservation], IncidentAction],
) -> float:
    """Execute one action then follow default_policy for remaining steps.

    Returns cumulative reward from this point onward.
    """
    total = 0.0
    try:
        result = env.step(action)
        total += result.reward
        done = result.done
        obs = result.observation
        for _ in range(remaining_steps - 1):
            if done:
                break
            next_action = default_policy(obs)
            result = env.step(next_action)
            total += result.reward
            done = result.done
            obs = result.observation
    except Exception:
        pass
    return total


def _noop_policy(_obs: IncidentObservation) -> IncidentAction:
    """Fallback policy: just do a harmless status update."""
    return IncidentAction(
        ActionType.UPDATE_STATUS,
        message="Impact under investigation. Cause being analyzed. Action pending. Next step TBD.",
    )


def counterfactual_episode(
    scenario_id: str,
    policy_fn: Callable[[IncidentObservation], IncidentAction],
    max_steps: int = 12,
    top_k: int = 3,
) -> dict[str, Any]:
    """Run one episode with counterfactual evaluation at each step.

    For each step, compare the chosen action's downstream reward against
    top_k alternative actions.
    """
    env = IncidentCommanderEnvironment(max_steps=max_steps)
    obs = env.reset(scenario_id=scenario_id)

    step_analyses: list[dict[str, Any]] = []
    total_chosen_reward = 0.0
    done = False
    step = 0

    while not done and step < max_steps:
        chosen_action = policy_fn(obs)

        # Save env state for counterfactual replay.
        saved_scenario = copy.deepcopy(env._scenario)
        saved_state = copy.deepcopy(env._state)
        saved_waste = env._consecutive_waste

        remaining = max_steps - step

        # Rollout chosen action.
        chosen_reward = rollout_from_state(
            env, chosen_action, remaining, _noop_policy,
        )

        # Restore and try alternatives.
        alt_results: list[dict[str, Any]] = []
        for alt_action in _ALTERNATIVE_ACTIONS[:top_k + 2]:
            alt_env = IncidentCommanderEnvironment(max_steps=max_steps)
            alt_env._scenario = copy.deepcopy(saved_scenario)
            alt_env._state = copy.deepcopy(saved_state)
            alt_env._consecutive_waste = saved_waste

            alt_reward = rollout_from_state(
                alt_env, alt_action, remaining, _noop_policy,
            )
            alt_results.append({
                "action": alt_action.action_type.value,
                "detail": alt_action.tool_name or alt_action.mitigation or alt_action.cause or "",
                "downstream_reward": round(alt_reward, 4),
            })

        # Sort alternatives by reward descending.
        alt_results.sort(key=lambda x: x["downstream_reward"], reverse=True)
        best_alt = alt_results[0]["downstream_reward"] if alt_results else 0.0

        step_analyses.append({
            "step": step + 1,
            "chosen_action": chosen_action.action_type.value,
            "chosen_downstream_reward": round(chosen_reward, 4),
            "best_alternative_reward": round(best_alt, 4),
            "decision_quality_delta": round(chosen_reward - best_alt, 4),
            "alternatives": alt_results[:top_k],
        })

        # Restore state and execute the actual chosen action to advance.
        env._scenario = copy.deepcopy(saved_scenario)
        env._state = copy.deepcopy(saved_state)
        env._consecutive_waste = saved_waste

        result = env.step(chosen_action)
        total_chosen_reward += result.reward
        done = result.done
        obs = result.observation
        step += 1

    # Evaluate causal faithfulness.
    scenario = env._scenario
    state = env._state
    causal_result = None
    if scenario and state and scenario.causal_edges:
        causal_result = evaluate_causal_faithfulness(
            scenario.causal_edges, state,
        )

    return {
        "scenario_id": scenario_id,
        "total_reward": round(total_chosen_reward, 4),
        "steps_taken": step,
        "resolved": state.resolved if state else False,
        "step_analyses": step_analyses,
        "causal_faithfulness": asdict(causal_result) if causal_result else None,
        "stability_score": round(state.stability_score, 4) if state else 1.0,
        "delayed_effects_fired": len(state.fired_effects) if state else 0,
        "governance_violations": list(state.governance_violations) if state else [],
    }


def save_dqd_chart(results: list[dict[str, Any]]) -> None:
    """Create Decision Quality Delta chart across episodes."""
    all_deltas: list[float] = []
    for ep in results:
        for step_data in ep["step_analyses"]:
            all_deltas.append(step_data["decision_quality_delta"])

    if not all_deltas:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Histogram of DQD values.
    colors = ["#22c55e" if d >= 0 else "#ef4444" for d in all_deltas]
    ax1.hist(all_deltas, bins=20, color="#60a5fa", edgecolor="#1e40af", alpha=0.8)
    ax1.axvline(0, color="#ef4444", linewidth=1.5, linestyle="--", label="Break-even")
    ax1.set_title("Decision Quality Delta Distribution", fontsize=12, fontweight="bold")
    ax1.set_xlabel("DQD (chosen − best alternative)")
    ax1.set_ylabel("Frequency")
    ax1.legend()

    # Per-episode mean DQD.
    ep_means = []
    ep_labels = []
    for ep in results:
        deltas = [s["decision_quality_delta"] for s in ep["step_analyses"]]
        ep_means.append(sum(deltas) / len(deltas) if deltas else 0)
        ep_labels.append(ep["scenario_id"][:20])

    bar_colors = ["#22c55e" if m >= 0 else "#ef4444" for m in ep_means]
    ax2.barh(ep_labels, ep_means, color=bar_colors, edgecolor="#334155")
    ax2.axvline(0, color="#64748b", linewidth=1, linestyle="--")
    ax2.set_title("Mean DQD by Scenario", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Mean Decision Quality Delta")

    fig.suptitle("Counterfactual Decision Quality Analysis", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "decision_quality_delta.png", dpi=140)
    plt.close(fig)


def save_causal_chart(results: list[dict[str, Any]]) -> None:
    """Create causal faithfulness vs reward correlation chart."""
    rewards = []
    faithfulness = []
    labels = []
    for ep in results:
        cf = ep.get("causal_faithfulness")
        if cf and cf.get("total_edges", 0) > 0:
            rewards.append(ep["total_reward"])
            faithfulness.append(cf["faithfulness_score"])
            labels.append(ep["scenario_id"][:15])

    if len(rewards) < 2:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    scatter = ax.scatter(faithfulness, rewards, s=80, c="#6366f1", edgecolors="#312e81", alpha=0.8)

    for i, label in enumerate(labels):
        ax.annotate(label, (faithfulness[i], rewards[i]), fontsize=7,
                     xytext=(5, 5), textcoords="offset points")

    ax.set_xlabel("Causal Faithfulness Score", fontsize=11)
    ax.set_ylabel("Episode Total Reward", fontsize=11)
    ax.set_title("Causal Faithfulness vs Reward Correlation", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "causal_faithfulness_correlation.png", dpi=140)
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Import tabular policy
    from examples.minimal_trl_training import TabularPolicy
    POLICY_PATH = OUTPUT_DIR / "tabular_policy.json"
    if not POLICY_PATH.exists():
        raise RuntimeError("Trained policy not found. Run examples/minimal_trl_training.py first.")
    
    trained = TabularPolicy.load(POLICY_PATH)
    trained_policy = lambda obs: trained.greedy_action(obs)

    # Run counterfactual analysis on all base scenarios.
    env = IncidentCommanderEnvironment()
    scenario_ids = env.available_scenarios(split="base")

    # Also include governance flagship if available.
    try:
        gov_ids = env.available_scenarios(split="governance")
        scenario_ids = scenario_ids + gov_ids
    except Exception:
        pass

    results: list[dict[str, Any]] = []
    for sid in scenario_ids:
        print(f"  Counterfactual analysis: {sid}")
        result = counterfactual_episode(sid, trained_policy, top_k=3)
        results.append(result)

    # Save artifacts.
    (OUTPUT_DIR / "decision_quality_delta.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8",
    )

    save_dqd_chart(results)
    save_causal_chart(results)

    # Print summary.
    all_dqd = []
    all_faithfulness = []
    for r in results:
        for s in r["step_analyses"]:
            all_dqd.append(s["decision_quality_delta"])
        cf = r.get("causal_faithfulness")
        if cf and cf.get("total_edges", 0) > 0:
            all_faithfulness.append(cf["faithfulness_score"])

    mean_dqd = sum(all_dqd) / len(all_dqd) if all_dqd else 0
    mean_cf = sum(all_faithfulness) / len(all_faithfulness) if all_faithfulness else 0
    positive_dqd = sum(1 for d in all_dqd if d >= 0) / len(all_dqd) if all_dqd else 0

    print(f"\n  Mean DQD: {mean_dqd:+.4f}")
    print(f"  Positive DQD rate: {positive_dqd:.1%}")
    print(f"  Mean causal faithfulness: {mean_cf:.4f}")
    print(f"  Saved: decision_quality_delta.json, .png, causal_faithfulness_correlation.png")


if __name__ == "__main__":
    main()
