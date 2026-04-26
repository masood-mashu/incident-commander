"""Evaluate random, heuristic, and trained policies with reproducible metrics."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import random
import subprocess
import sys
from typing import Any, Callable, cast

import matplotlib
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from incident_commander.models import ActionType, IncidentAction, IncidentObservation
from incident_commander.server.incident_environment import IncidentCommanderEnvironment

from examples.minimal_trl_training import POLICY_PATH, TabularPolicy

matplotlib.use("Agg")

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "outputs" / "evals"


def detect_family(observation: IncidentObservation) -> str:
    text = " ".join([observation.summary, *observation.visible_alerts]).lower()
    if "deploy" in text or "checkout" in text:
        return "bad_deploy"
    if "db" in text or "database" in text or "saturation" in text:
        return "database_saturation"
    if "flag" in text or "regional" in text or "eu-west" in text:
        return "feature_flag_misconfig"
    if "vendor" in text or "third-party" in text or "dependency" in text:
        return "third_party_failure"
    return "unknown"


def random_policy(_obs: IncidentObservation) -> IncidentAction:
    return IncidentAction(action_type=random.choice(list(ActionType)))


def heuristic_policy(obs: IncidentObservation) -> IncidentAction:
    family = detect_family(obs)
    step = obs.step_count
    script: dict[str, list[IncidentAction]] = {
        "bad_deploy": [
            IncidentAction(ActionType.QUERY_TOOL, tool_name="metrics", target="checkout_service"),
            IncidentAction(ActionType.QUERY_TOOL, tool_name="logs", target="checkout_service"),
            IncidentAction(ActionType.PROPOSE_HYPOTHESIS, cause="bad_deploy"),
            IncidentAction(ActionType.EXECUTE_MITIGATION, mitigation="rollback", target="checkout_service"),
            IncidentAction(
                ActionType.UPDATE_STATUS,
                message="Impact identified. Cause confirmed. Action rollback executed. Next verify recovery.",
            ),
            IncidentAction(
                ActionType.CLOSE_INCIDENT,
                report="Resolved via rollback for bad deploy checkout regression.",
            ),
        ],
        "database_saturation": [
            IncidentAction(ActionType.QUERY_TOOL, tool_name="metrics", target="payments_db"),
            IncidentAction(ActionType.PROPOSE_HYPOTHESIS, cause="database_saturation"),
            IncidentAction(ActionType.ESCALATE, escalate_to="infra_engineer", message="Need DB failover approval."),
            IncidentAction(ActionType.EXECUTE_MITIGATION, mitigation="failover_db", target="payments_db"),
            IncidentAction(
                ActionType.UPDATE_STATUS,
                message="Impact identified. Cause confirmed. Action failover_db executed. Next verify recovery.",
            ),
            IncidentAction(
                ActionType.CLOSE_INCIDENT,
                report="Resolved via failover_db for payments database saturation.",
            ),
        ],
        "feature_flag_misconfig": [
            IncidentAction(ActionType.QUERY_TOOL, tool_name="metrics", target="api_eu"),
            IncidentAction(ActionType.PROPOSE_HYPOTHESIS, cause="feature_flag_misconfig"),
            IncidentAction(ActionType.EXECUTE_MITIGATION, mitigation="disable_flag", target="api_eu"),
            IncidentAction(
                ActionType.UPDATE_STATUS,
                message="Impact identified. Cause confirmed. Action disable_flag executed. Next verify recovery.",
            ),
            IncidentAction(
                ActionType.CLOSE_INCIDENT,
                report="Resolved by disable_flag in eu-west-1 feature flag rollout.",
            ),
        ],
        "third_party_failure": [
            IncidentAction(ActionType.QUERY_TOOL, tool_name="metrics", target="search_vendor"),
            IncidentAction(ActionType.PROPOSE_HYPOTHESIS, cause="third_party_failure"),
            IncidentAction(ActionType.ESCALATE, escalate_to="sre_lead", message="Need traffic shift approval."),
            IncidentAction(ActionType.EXECUTE_MITIGATION, mitigation="traffic_shift", target="global"),
            IncidentAction(
                ActionType.UPDATE_STATUS,
                message="Impact identified. Cause confirmed. Action traffic_shift executed. Next verify recovery.",
            ),
            IncidentAction(
                ActionType.CLOSE_INCIDENT,
                report="Resolved via traffic_shift for third-party vendor dependency degradation.",
            ),
        ],
    }
    plan = script.get(family, [IncidentAction(ActionType.QUERY_TOOL, tool_name="metrics", target="global")])
    return plan[min(step, len(plan) - 1)]


def evaluate_policy(
    name: str,
    policy_fn: Callable[[IncidentObservation], IncidentAction],
    episodes: int = 100,
    seed: int = 11,
    split: str = "test",
) -> dict[str, Any]:
    env = IncidentCommanderEnvironment()
    rewards: list[float] = []
    steps: list[int] = []
    resolved_count = 0

    for ep in range(episodes):
        observation = env.reset(seed=seed + ep, split=split)
        done = False
        total_reward = 0.0
        while not done:
            action = policy_fn(observation)
            result = env.step(action)
            observation = result.observation
            done = result.done
            total_reward += result.reward

        rewards.append(total_reward)
        steps.append(observation.step_count)
        if observation.terminal_reason == "resolved":
            resolved_count += 1

    return {
        "policy": name,
        "episodes": episodes,
        "mean_reward": sum(rewards) / len(rewards),
        "median_reward": sorted(rewards)[len(rewards) // 2],
        "resolved_rate": resolved_count / episodes,
        "mean_steps": sum(steps) / len(steps),
        "split": split,
        "rewards": rewards,
    }


def save_comparison_plot(results: list[dict[str, Any]]) -> None:
    labels = [row["policy"] for row in results]
    means = [row["mean_reward"] for row in results]
    resolved = [row["resolved_rate"] for row in results]

    fig, axes = cast(Any, plt).subplots(1, 2, figsize=(10.5, 4.5))
    fig = cast(Figure, fig)
    ax1, ax2 = cast(tuple[Axes, Axes], axes)
    plot_fig = cast(Any, fig)
    plot_ax1 = cast(Any, ax1)
    plot_ax2 = cast(Any, ax2)

    plot_ax1.bar(labels, means, color=["#64748b", "#0284c7", "#16a34a"])
    plot_ax1.set_title("Mean Reward")
    plot_ax1.set_ylabel("Return")

    plot_ax2.bar(labels, resolved, color=["#64748b", "#0284c7", "#16a34a"])
    plot_ax2.set_title("Resolved Rate")
    plot_ax2.set_ylim(0.0, 1.0)

    plot_fig.suptitle("Policy Comparison on Incident Commander")
    plot_fig.tight_layout()
    plot_fig.savefig(OUTPUT_DIR / "policy_comparison.png", dpi=140)
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not POLICY_PATH.exists():
        raise RuntimeError(
            "Trained policy not found. Run examples/minimal_trl_training.py first."
        )

    trained = TabularPolicy.load(POLICY_PATH)

    def trained_policy(obs: IncidentObservation) -> IncidentAction:
        return trained.greedy_action(obs)

    # ── Standard test-split evaluation ──
    results = [
        evaluate_policy("random", random_policy, split="test"),
        evaluate_policy("heuristic", heuristic_policy, split="test"),
        evaluate_policy("trained", trained_policy, split="test"),
    ]

    save_comparison_plot(results)

    summary_rows = [
        {
            "policy": row["policy"],
            "episodes": row["episodes"],
            "mean_reward": round(row["mean_reward"], 3),
            "median_reward": round(row["median_reward"], 3),
            "resolved_rate": round(row["resolved_rate"], 3),
            "mean_steps": round(row["mean_steps"], 3),
        }
        for row in results
    ]

    (OUTPUT_DIR / "policy_eval_summary.json").write_text(
        json.dumps(summary_rows, indent=2), encoding="utf-8"
    )

    traces = {row["policy"]: row["rewards"] for row in results}
    (OUTPUT_DIR / "policy_eval_rewards.json").write_text(
        json.dumps(traces, indent=2), encoding="utf-8"
    )

    metadata: dict[str, str | int] = {
        "evaluated_split": "test",
        "episodes_per_policy": 100,
        "seed_start": 11,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
    }
    (OUTPUT_DIR / "policy_eval_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    print(json.dumps(summary_rows, indent=2))

    # ── Robustness benchmarks: OOD, Stress, Governance ──
    print("\n-- Robustness Benchmarks --")
    robustness: list[dict[str, Any]] = []

    splits: list[tuple[str, int]] = [
        ("test", 100),
        ("ood", 50),
        ("stress", 50),
        ("governance", 30),
    ]
    policies: list[tuple[str, Callable[[IncidentObservation], IncidentAction]]] = [
        ("heuristic", heuristic_policy),
        ("trained", trained_policy),
    ]

    for split_name, episodes in splits:
        for policy_name, policy_fn in policies:
            try:
                row = evaluate_policy(
                    f"{policy_name}_{split_name}",
                    policy_fn,
                    episodes=episodes,
                    split=split_name,
                )
                robustness.append({
                    "policy": policy_name,
                    "split": split_name,
                    "episodes": episodes,
                    "mean_reward": round(row["mean_reward"], 3),
                    "resolved_rate": round(row["resolved_rate"], 3),
                    "mean_steps": round(row["mean_steps"], 3),
                })
            except Exception as exc:
                print(f"  Skip {policy_name}/{split_name}: {exc}")

    (OUTPUT_DIR / "robustness_summary.json").write_text(
        json.dumps(robustness, indent=2), encoding="utf-8"
    )
    print(json.dumps(robustness, indent=2))
    print(f"\nSaved evaluation artifacts in: {OUTPUT_DIR}")


def _git_commit() -> str:
    try:
        cmd = ["git", "rev-parse", "--short", "HEAD"]
        value = subprocess.check_output(cmd, text=True, cwd=Path(__file__).resolve().parents[1]).strip()
        return value or "unknown"
    except Exception:
        return "unknown"


if __name__ == "__main__":
    main()
