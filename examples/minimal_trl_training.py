"""Train a lightweight policy and emit judge-friendly artifacts.

This script intentionally avoids heavyweight RL dependencies so it can run
end-to-end in constrained hackathon environments while still producing
measurable before/after evidence.
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import random
import subprocess
import sys
from typing import Any, Callable, cast

import matplotlib
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from incident_commander.models import ActionType, IncidentAction, IncidentObservation
from incident_commander.server.incident_environment import IncidentCommanderEnvironment

matplotlib.use("Agg")


OUTPUT_DIR = Path(__file__).resolve().parents[1] / "outputs" / "evals"
POLICY_PATH = OUTPUT_DIR / "tabular_policy.json"
REPO_ROOT = Path(__file__).resolve().parents[1]


def _repo_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except Exception:
        return path.as_posix()


def _detect_family(observation: IncidentObservation) -> str:
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


def _policy_state_key(observation: IncidentObservation) -> str:
    family = _detect_family(observation)
    phase = min(5, observation.step_count // 2)
    evidence_count = min(3, len(observation.tool_results))
    hypothesis_count = min(2, len(observation.hypotheses))
    resolved = int(observation.resolved)
    return f"{family}|p{phase}|e{evidence_count}|h{hypothesis_count}|r{resolved}"


def _action_library() -> list[IncidentAction]:
    return [
        IncidentAction(ActionType.QUERY_TOOL, tool_name="metrics", target="checkout_service"),
        IncidentAction(ActionType.QUERY_TOOL, tool_name="logs", target="checkout_service"),
        IncidentAction(ActionType.PROPOSE_HYPOTHESIS, cause="bad_deploy"),
        IncidentAction(ActionType.EXECUTE_MITIGATION, mitigation="rollback", target="checkout_service"),
        IncidentAction(ActionType.QUERY_TOOL, tool_name="metrics", target="payments_db"),
        IncidentAction(ActionType.PROPOSE_HYPOTHESIS, cause="database_saturation"),
        IncidentAction(ActionType.ESCALATE, escalate_to="infra_engineer", message="Requesting DB failover approval."),
        IncidentAction(ActionType.EXECUTE_MITIGATION, mitigation="failover_db", target="payments_db"),
        IncidentAction(ActionType.QUERY_TOOL, tool_name="metrics", target="api_eu"),
        IncidentAction(ActionType.PROPOSE_HYPOTHESIS, cause="feature_flag_misconfig"),
        IncidentAction(ActionType.EXECUTE_MITIGATION, mitigation="disable_flag", target="api_eu"),
        IncidentAction(ActionType.QUERY_TOOL, tool_name="metrics", target="search_vendor"),
        IncidentAction(ActionType.PROPOSE_HYPOTHESIS, cause="third_party_failure"),
        IncidentAction(ActionType.ESCALATE, escalate_to="sre_lead", message="Requesting traffic shift approval."),
        IncidentAction(ActionType.EXECUTE_MITIGATION, mitigation="traffic_shift", target="global"),
        IncidentAction(
            ActionType.UPDATE_STATUS,
            message="Impact identified. Cause under investigation. Action underway. Next update in 5 minutes.",
        ),
        IncidentAction(
            ActionType.CLOSE_INCIDENT,
            report="Incident resolved with mitigation and stakeholder communication.",
        ),
    ]


def _action_to_plain(action: IncidentAction) -> dict[str, Any]:
    raw = asdict(action)
    return {key: value for key, value in raw.items() if value is not None}


class TabularPolicy:
    def __init__(self, actions: list[IncidentAction], seed: int = 42) -> None:
        self.actions = actions
        self._rng = random.Random(seed)
        self.preferences: dict[str, list[float]] = {}

    def _ensure_state(self, key: str) -> list[float]:
        if key not in self.preferences:
            self.preferences[key] = [0.0 for _ in self.actions]
        return self.preferences[key]

    @staticmethod
    def _softmax(values: list[float]) -> list[float]:
        if not values:
            return []
        peak = max(values)
        exps = [pow(2.718281828, value - peak) for value in values]
        total = sum(exps)
        return [value / total for value in exps]

    def sample_action(self, observation: IncidentObservation) -> tuple[int, IncidentAction, float, str]:
        key = _policy_state_key(observation)
        prefs = self._ensure_state(key)
        probs = self._softmax(prefs)
        roll = self._rng.random()
        cumulative = 0.0
        index = len(probs) - 1
        for i, prob in enumerate(probs):
            cumulative += prob
            if roll <= cumulative:
                index = i
                break
        return index, self.actions[index], max(probs[index], 1e-9), key

    def greedy_action(self, observation: IncidentObservation) -> IncidentAction:
        key = _policy_state_key(observation)
        prefs = self._ensure_state(key)
        best_idx = max(range(len(prefs)), key=lambda idx: prefs[idx])
        return self.actions[best_idx]

    def update_episode(
        self,
        states: list[str],
        action_indices: list[int],
        log_probs: list[float],
        episode_return: float,
        baseline: float,
        learning_rate: float,
    ) -> float:
        advantage = episode_return - baseline
        if not states:
            return 0.0
        for state_key, action_idx in zip(states, action_indices):
            prefs = self._ensure_state(state_key)
            probs = self._softmax(prefs)
            for idx in range(len(prefs)):
                gradient = (1.0 if idx == action_idx else 0.0) - probs[idx]
                prefs[idx] += learning_rate * advantage * gradient
        mean_log_prob = sum(log_probs) / len(log_probs)
        return -advantage * mean_log_prob

    def save(self, path: Path) -> None:
        payload: dict[str, Any] = {
            "preferences": self.preferences,
            "actions": [_action_to_plain(action) for action in self.actions],
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path, seed: int = 42) -> "TabularPolicy":
        payload = json.loads(path.read_text(encoding="utf-8"))
        actions: list[IncidentAction] = []
        for item in payload["actions"]:
            hydrated = dict(item)
            hydrated["action_type"] = ActionType(hydrated["action_type"])
            actions.append(IncidentAction(**hydrated))
        policy = cls(actions=actions, seed=seed)
        policy.preferences = {
            key: [float(value) for value in values]
            for key, values in payload["preferences"].items()
        }
        return policy


def rollout_episode(
    env: IncidentCommanderEnvironment,
    policy: Callable[[IncidentObservation], IncidentAction],
) -> float:
    observation = env.reset()
    done = False
    total_reward = 0.0
    while not done:
        action = policy(observation)
        result = env.step(action)
        observation = result.observation
        done = result.done
        total_reward += result.reward
    return total_reward


def random_policy(_observation: IncidentObservation) -> IncidentAction:
    action_type = random.choice(list(ActionType))
    return IncidentAction(action_type=action_type)


def run_training(
    episodes: int = 300,
    learning_rate: float = 0.08,
    seed: int = 42,
    split: str = "train",
) -> dict[str, Any]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    env = IncidentCommanderEnvironment()
    policy = TabularPolicy(actions=_action_library(), seed=seed)

    episode_rewards: list[float] = []
    losses: list[float] = []
    baseline = 0.0

    for episode in range(episodes):
        observation = env.reset(seed=seed + episode, split=split)
        done = False
        total_reward = 0.0
        states: list[str] = []
        action_indices: list[int] = []
        log_probs: list[float] = []

        while not done:
            action_idx, action, selected_prob, state_key = policy.sample_action(observation)
            step_result = env.step(action)
            states.append(state_key)
            action_indices.append(action_idx)
            log_probs.append(float(math.log(selected_prob)))
            total_reward += step_result.reward
            observation = step_result.observation
            done = step_result.done

        baseline = 0.95 * baseline + 0.05 * total_reward if episode > 0 else total_reward
        loss = policy.update_episode(
            states=states,
            action_indices=action_indices,
            log_probs=log_probs,
            episode_return=total_reward,
            baseline=baseline,
            learning_rate=learning_rate,
        )
        episode_rewards.append(total_reward)
        losses.append(loss)

    policy.save(POLICY_PATH)
    _write_training_plots(episode_rewards, losses)

    summary: dict[str, Any] = {
        "episodes": episodes,
        "split": split,
        "seed": seed,
        "learning_rate": learning_rate,
        "mean_reward": sum(episode_rewards) / len(episode_rewards),
        "best_reward": max(episode_rewards),
        "final_20_mean_reward": sum(episode_rewards[-20:]) / min(20, len(episode_rewards)),
        "policy_path": _repo_relative(POLICY_PATH),
        "reward_curve": _repo_relative(OUTPUT_DIR / "training_reward_curve.png"),
        "loss_curve": _repo_relative(OUTPUT_DIR / "training_loss_curve.png"),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
    }
    (OUTPUT_DIR / "training_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def _moving_average(values: list[float], window: int = 15) -> list[float]:
    result: list[float] = []
    for idx in range(len(values)):
        low = max(0, idx - window + 1)
        segment = values[low : idx + 1]
        result.append(sum(segment) / len(segment))
    return result


def _write_training_plots(rewards: list[float], losses: list[float]) -> None:
    xs = list(range(1, len(rewards) + 1))
    mplt = cast(Any, plt)

    mplt.figure(figsize=(10, 4.5))
    mplt.plot(xs, rewards, alpha=0.35, label="episode reward")
    mplt.plot(xs, _moving_average(rewards), linewidth=2.0, label="moving average (15)")
    mplt.xlabel("Episode")
    mplt.ylabel("Return")
    mplt.title("Incident Commander Training Reward")
    mplt.legend()
    mplt.tight_layout()
    mplt.savefig(OUTPUT_DIR / "training_reward_curve.png", dpi=140)
    mplt.close()

    mplt.figure(figsize=(10, 4.5))
    mplt.plot(xs, losses, linewidth=1.2, color="tab:orange")
    mplt.xlabel("Episode")
    mplt.ylabel("Policy Gradient Loss Proxy")
    mplt.title("Incident Commander Training Loss")
    mplt.tight_layout()
    mplt.savefig(OUTPUT_DIR / "training_loss_curve.png", dpi=140)
    mplt.close()


def _git_commit() -> str:
    try:
        cmd = ["git", "rev-parse", "--short", "HEAD"]
        value = subprocess.check_output(cmd, text=True, cwd=Path(__file__).resolve().parents[1]).strip()
        return value or "unknown"
    except Exception:
        return "unknown"


def main() -> None:
    summary = run_training()
    print(json.dumps(summary, indent=2))
    print(f"Saved artifacts in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
