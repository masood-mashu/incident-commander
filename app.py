"""Gradio Space app for the Incident Commander demo."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import threading
from typing import Any, cast
import uuid

import gradio as gr

from incident_commander.models import ActionType, IncidentAction, IncidentFamily
from incident_commander.server.incident_environment import IncidentCommanderEnvironment

JsonDict = dict[str, Any]
EVAL_SUMMARY_PATH = Path(__file__).parent / "outputs" / "evals" / "policy_eval_summary.json"
if not EVAL_SUMMARY_PATH.exists():
    EVAL_SUMMARY_PATH = Path("outputs/evals/policy_eval_summary.json")

EVAL_METADATA_PATH = Path(__file__).parent / "outputs" / "evals" / "policy_eval_metadata.json"
if not EVAL_METADATA_PATH.exists():
    EVAL_METADATA_PATH = Path("outputs/evals/policy_eval_metadata.json")
_ENV_LOCK = threading.RLock()
_ENV_BY_SESSION: dict[str, IncidentCommanderEnvironment] = {}

# ── helpers ────────────────────────────────────────────────────────────────────

def make_env() -> IncidentCommanderEnvironment:
    return IncidentCommanderEnvironment()


def put_env(session_id: str, env: IncidentCommanderEnvironment) -> None:
    with _ENV_LOCK:
        _ENV_BY_SESSION[session_id] = env


def get_env(session_id: str) -> IncidentCommanderEnvironment | None:
    with _ENV_LOCK:
        return _ENV_BY_SESSION.get(session_id)


def fmt_observation(obs_dict: JsonDict) -> str:
    lines: list[str] = []
    lines.append(f"## {obs_dict.get('summary', '')}\n")

    alerts = obs_dict.get("visible_alerts", [])
    if alerts:
        lines.append("**Alerts**")
        for a in alerts:
            lines.append(f"  • {a}")

    tool_results = obs_dict.get("tool_results", [])
    if tool_results:
        lines.append("\n**Tool Evidence**")
        for t in tool_results:
            useful = "[useful]" if t.get("useful") else "[repeat]"
            lines.append(
                f"- {useful} {t.get('tool_name')} -> {t.get('target')}: {t.get('content', '')[:120]}"
            )

    hypotheses = obs_dict.get("hypotheses", [])
    if hypotheses:
        lines.append("\n**Active Hypotheses**")
        for h in hypotheses:
            lines.append(f"- {h}")

    health = obs_dict.get("service_health", {})
    if health:
        lines.append("\n**Service Health**")
        for svc, val in health.items():
            bar = "OK" if val > 0.8 else ("WARN" if val > 0.4 else "CRIT")
            lines.append(f"- {bar} {svc}: {val:.0%}")

    msgs = obs_dict.get("stakeholder_messages", [])
    if msgs:
        lines.append("\n**Stakeholder Messages**")
        for m in msgs[-3:]:
            lines.append(f"> {m}")

    step = obs_dict.get("step_count", 0)
    budget = obs_dict.get("remaining_budget", 0)
    resolved = obs_dict.get("resolved", False)
    lines.append(f"\nStep {step} | Budget remaining: {budget} | Resolved: {resolved}")

    terminal = obs_dict.get("terminal_reason")
    if terminal:
        lines.append(f"\nTerminal: {terminal}")

    return "\n".join(lines)


def fmt_reward(breakdown: JsonDict, total: float) -> str:
    if not breakdown:
        return ""
    lines = [f"### Step Reward: {total:+.3f}\n"]
    labels = {
        "diagnostic": "🔬 Diagnosis Quality",
        "action": "⚡ Mitigation Safety",
        "communication": "📢 Stakeholder Trust",
        "risk_penalty": "⚠️ Risk Penalty",
        "waste_penalty": "🗑️ Waste Penalty",
        "outcome": "✅ Outcome",
        "failure": "❌ Failure Penalty",
        "stability": "📊 Long-Term Stability",
        "governance": "🏛️ Governance Penalty",
    }
    for k, label in labels.items():
        v = breakdown.get(k, 0.0)
        sign = "+" if v >= 0 else ""
        lines.append(f"- {label}: {sign}{v:.3f}")
    return "\n".join(lines)


def load_results_markdown() -> str:
    if not EVAL_SUMMARY_PATH.exists():
        return (
            "### Latest Benchmark Results\n"
            "No evaluation summary found yet. Run:\n"
            "- `python examples/minimal_trl_training.py`\n"
            "- `python examples/evaluate_policies.py`"
        )

    try:
        parsed: Any = json.loads(EVAL_SUMMARY_PATH.read_text(encoding="utf-8"))
        if not isinstance(parsed, list) or not parsed:
            return "### Latest Benchmark Results\nNo rows found in policy_eval_summary.json."
        rows: list[JsonDict] = []
        for candidate in cast(list[Any], parsed):
            if isinstance(candidate, dict):
                rows.append(cast(JsonDict, candidate))
        if not rows:
            return "### Latest Benchmark Results\nNo valid rows found in policy_eval_summary.json."

        lines = [
            "### Latest Benchmark Results",
            "| Policy | Episodes | Mean Reward | Median Reward | Resolved Rate | Mean Steps |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for row in rows:
            lines.append(
                "| {policy} | {episodes} | {mean_reward} | {median_reward} | {resolved_rate} | {mean_steps} |".format(
                    policy=str(row.get("policy", "n/a")),
                    episodes=str(row.get("episodes", "n/a")),
                    mean_reward=str(row.get("mean_reward", "n/a")),
                    median_reward=str(row.get("median_reward", "n/a")),
                    resolved_rate=str(row.get("resolved_rate", "n/a")),
                    mean_steps=str(row.get("mean_steps", "n/a")),
                )
            )

        if EVAL_METADATA_PATH.exists():
            try:
                parsed_metadata: Any = json.loads(EVAL_METADATA_PATH.read_text(encoding="utf-8"))
                if isinstance(parsed_metadata, dict):
                    metadata = cast(JsonDict, parsed_metadata)
                    lines.append("")
                    lines.append("**Provenance**")
                    lines.append(
                        "- split: {split} | episodes/policy: {episodes} | seed_start: {seed}".format(
                            split=str(metadata.get("evaluated_split", "n/a")),
                            episodes=str(metadata.get("episodes_per_policy", "n/a")),
                            seed=str(metadata.get("seed_start", "n/a")),
                        )
                    )
                    lines.append(
                        "- timestamp_utc: {ts} | git_commit: {commit}".format(
                            ts=str(metadata.get("timestamp_utc", "n/a")),
                            commit=str(metadata.get("git_commit", "n/a")),
                        )
                    )
            except Exception:
                pass
        return "\n".join(lines)
    except Exception as exc:
        return f"### Latest Benchmark Results\nCould not parse summary file: {exc}"


# ── scripted demo agent ────────────────────────────────────────────────────────

DEMO_SCRIPTS = {
    IncidentFamily.BAD_DEPLOY: [
        IncidentAction(ActionType.QUERY_TOOL, tool_name="metrics", target="checkout_service"),
        IncidentAction(ActionType.QUERY_TOOL, tool_name="logs", target="checkout_service"),
        IncidentAction(ActionType.PROPOSE_HYPOTHESIS, cause="bad_deploy"),
        IncidentAction(ActionType.EXECUTE_MITIGATION, mitigation="rollback", target="checkout_service"),
        IncidentAction(
            ActionType.UPDATE_STATUS,
            message="Impact contained. Cause identified. Action taken. Next step is verification.",
        ),
        IncidentAction(
            ActionType.CLOSE_INCIDENT,
            report="Resolved via rollback for checkout_service bad deploy latency regression.",
        ),
    ],
    IncidentFamily.DATABASE_SATURATION: [
        IncidentAction(ActionType.QUERY_TOOL, tool_name="metrics", target="payments_db"),
        IncidentAction(ActionType.PROPOSE_HYPOTHESIS, cause="database_saturation"),
        IncidentAction(
            ActionType.ESCALATE,
            escalate_to="infra_engineer",
            message="Need approval for DB failover mitigation.",
        ),
        IncidentAction(ActionType.EXECUTE_MITIGATION, mitigation="failover_db", target="payments_db"),
        IncidentAction(
            ActionType.UPDATE_STATUS,
            message="Impact contained. Cause database_saturation. Action failover_db. Next verify.",
        ),
        IncidentAction(
            ActionType.CLOSE_INCIDENT,
            report="Resolved with approved failover_db after database saturation in payments_db.",
        ),
    ],
    IncidentFamily.FEATURE_FLAG: [
        IncidentAction(ActionType.QUERY_TOOL, tool_name="metrics", target="api_eu"),
        IncidentAction(ActionType.PROPOSE_HYPOTHESIS, cause="feature_flag_misconfig"),
        IncidentAction(ActionType.EXECUTE_MITIGATION, mitigation="disable_flag", target="api_eu"),
        IncidentAction(
            ActionType.UPDATE_STATUS,
            message="Impact contained. Cause identified. Action disable_flag. Next monitor region.",
        ),
        IncidentAction(
            ActionType.CLOSE_INCIDENT,
            report="Regional issue resolved by disable_flag in eu-west-1 feature flag rollout.",
        ),
    ],
    IncidentFamily.THIRD_PARTY: [
        IncidentAction(ActionType.QUERY_TOOL, tool_name="metrics", target="search_vendor"),
        IncidentAction(ActionType.PROPOSE_HYPOTHESIS, cause="third_party_failure"),
        IncidentAction(
            ActionType.ESCALATE,
            escalate_to="sre_lead",
            message="Need approval to shift eligible traffic to fallback path.",
        ),
        IncidentAction(ActionType.EXECUTE_MITIGATION, mitigation="traffic_shift", target="global"),
        IncidentAction(
            ActionType.UPDATE_STATUS,
            message="Impact contained. Cause third_party_failure. Action traffic shift. Next verify.",
        ),
        IncidentAction(
            ActionType.CLOSE_INCIDENT,
            report="Vendor dependency mitigated via traffic_shift fallback and escalation.",
        ),
    ],
    IncidentFamily.MULTI_REGION_GOVERNANCE: [
        IncidentAction(ActionType.QUERY_TOOL, tool_name="metrics", target="primary_db"),
        IncidentAction(ActionType.QUERY_TOOL, tool_name="compliance", target="global"),
        IncidentAction(ActionType.QUERY_TOOL, tool_name="budget", target="global"),
        IncidentAction(ActionType.PROPOSE_HYPOTHESIS, cause="primary_db_failure"),
        IncidentAction(
            ActionType.ESCALATE,
            escalate_to="infra_engineer",
            message="Need approval for DB failover with compliance clearance.",
        ),
        IncidentAction(ActionType.EXECUTE_MITIGATION, mitigation="failover_db", target="primary_db"),
        IncidentAction(
            ActionType.UPDATE_STATUS,
            message="Impact contained. Cause primary_db_failure. Action failover_db. Next verify compliance.",
        ),
        IncidentAction(
            ActionType.CLOSE_INCIDENT,
            report="Primary DB failover completed with compliance and governance checks for data_residency.",
        ),
    ],
}


def get_script(family_str: str) -> list[IncidentAction]:
    try:
        family = IncidentFamily(family_str)
        return DEMO_SCRIPTS.get(family, list(DEMO_SCRIPTS.values())[0])
    except Exception:
        return list(DEMO_SCRIPTS.values())[0]


# ── state ──────────────────────────────────────────────────────────────────────

class SessionState:
    def __init__(self) -> None:
        self.session_id: str = str(uuid.uuid4())
        self.current_obs: JsonDict | None = None
        self.total_reward: float = 0.0
        self.step_rewards: list[float] = []
        self.script: list[IncidentAction] = []
        self.script_idx: int = 0
        self.family: str | None = None
        self.done: bool = False
        self.log: list[tuple[str, str]] = []


# ── gradio functions ───────────────────────────────────────────────────────────

def start_episode(state: SessionState):
    env = make_env()
    put_env(state.session_id, env)
    observation = env.reset()
    obs_dict: JsonDict = asdict(observation)
    state.current_obs = obs_dict
    state.total_reward = 0.0
    state.step_rewards = []
    state.done = False
    state.log = []

    family = env.state().scenario_family
    state.family = family
    state.script = get_script(family)
    state.script_idx = 0

    obs_text = fmt_observation(obs_dict)
    state.log.append(("New episode started", ""))
    log_text = "\n".join(f"**{a}** {b}" for a, b in state.log)

    return (
        state,
        obs_text,
        "",
        f"Episode started | Incident family: {family}",
        log_text,
        gr.update(interactive=True),
        gr.update(interactive=False),
        reward_chart(state.step_rewards),
    )


def step_agent(state: SessionState):
    env = get_env(state.session_id)
    if state.done or env is None:
        return state, "Episode not started or already done.", "", "", "", gr.update(), gr.update(), reward_chart([])

    if state.script_idx >= len(state.script):
        state.done = True
        return (
            state,
            fmt_observation(state.current_obs or {}),
            "",
            "Script complete. Start a new episode.",
            "\n".join(f"**{a}** {b}" for a, b in state.log),
            gr.update(interactive=False),
            gr.update(interactive=True),
            reward_chart(state.step_rewards),
        )

    action: IncidentAction = state.script[state.script_idx]
    state.script_idx += 1

    # Describe action taken
    action_desc = action.action_type.value
    if action.tool_name:
        action_desc += f" -> {action.tool_name}:{action.target}"
    if action.cause:
        action_desc += f" (hypothesis: {action.cause})"
    if action.mitigation:
        action_desc += f" (mitigation: {action.mitigation})"
    if action.message:
        action_desc += f' "{action.message[:60]}..."'
    if action.report:
        action_desc += " [report submitted]"

    try:
        result = env.step(action)
        obs_dict = asdict(result.observation)
        reward = result.reward
        done = result.done
        breakdown = obs_dict.get("reward_breakdown", {})
    except Exception as exc:
        return (
            state,
            f"Step error: {exc}",
            "",
            "",
            "",
            gr.update(),
            gr.update(),
            reward_chart(state.step_rewards),
        )

    state.current_obs = obs_dict
    state.total_reward += reward
    state.step_rewards.append(reward)
    state.done = done

    state.log.append((f"Step {len(state.step_rewards)}:", action_desc))
    log_text = "\n".join(f"**{a}** {b}" for a, b in state.log)

    obs_text = fmt_observation(obs_dict)
    reward_text = fmt_reward(breakdown, reward)
    total_text = f"Cumulative Reward: {state.total_reward:+.3f}"

    next_btn = gr.update(interactive=not done)
    new_btn = gr.update(interactive=done)

    return (
        state,
        obs_text,
        reward_text,
        total_text,
        log_text,
        next_btn,
        new_btn,
        reward_chart(state.step_rewards),
    )


def reward_chart(rewards: list[float]):
    if not rewards:
        return None
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        fig, ax = plt.subplots(figsize=(7, 3), facecolor="white")  # pyright: ignore[reportUnknownMemberType]
        fig = cast(Any, fig)
        ax = cast(Any, ax)
        ax.set_facecolor("#ffffff")

        steps = list(range(1, len(rewards) + 1))
        cumulative = list(np.cumsum(rewards))

        colors = ["#22c55e" if r >= 0 else "#ef4444" for r in rewards]
        ax.bar(steps, rewards, color=colors, alpha=0.7, label="Step reward", zorder=3)
        ax.plot(steps, cumulative, color="#60a5fa", linewidth=2,
                marker="o", markersize=4, label="Cumulative", zorder=4)
        ax.axhline(0, color="#4b5563", linewidth=0.8, linestyle="--")

        ax.set_xlabel("Step", color="#334155", fontsize=9)
        ax.set_ylabel("Reward", color="#334155", fontsize=9)
        ax.set_title("Episode Reward Curve", color="#0f172a", fontsize=11, pad=8)
        ax.tick_params(colors="#334155", labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor("#cbd5e1")
        ax.legend(fontsize=8, facecolor="#ffffff", edgecolor="#cbd5e1")
        ax.grid(True, color="#e2e8f0", linewidth=0.5, zorder=0)

        plt.tight_layout()
        return fig
    except Exception:
        return None


# -- UI ------------------------------------------------------------------------

CSS = """
/* ── Force readable text in both light and dark mode ── */
.gradio-container {
    font-family: ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}

/* Markdown text */
.gradio-container .prose h1,
.gradio-container .prose h2,
.gradio-container .prose h3 {
    color: var(--body-text-color) !important;
}
.gradio-container .prose p,
.gradio-container .prose li,
.gradio-container .prose strong,
.gradio-container .prose em,
.gradio-container .prose code,
.gradio-container .prose blockquote {
    color: var(--body-text-color) !important;
}

/* Table readability */
.gradio-container .prose table {
    color: var(--body-text-color) !important;
}
.gradio-container .prose th,
.gradio-container .prose td {
    color: var(--body-text-color) !important;
    border-color: var(--border-color-primary) !important;
}
.gradio-container .prose th {
    background: var(--background-fill-secondary) !important;
}

/* Label text */
.label-wrap, .label-wrap span {
    color: var(--body-text-color) !important;
}

footer { display: none !important; }
"""


DQD_PATH = Path(__file__).parent / "outputs" / "evals" / "decision_quality_delta.json"
if not DQD_PATH.exists():
    DQD_PATH = Path("outputs/evals/decision_quality_delta.json")


def load_counterfactual_markdown() -> str:
    """Load counterfactual analysis for judge replay."""
    if not DQD_PATH.exists():
        return (
            "### Counterfactual Analysis\n"
            "Run `python examples/counterfactual_evaluator.py` to generate."
        )
    try:
        data = json.loads(DQD_PATH.read_text(encoding="utf-8"))
        lines = [
            "### 🧪 Counterfactual Decision Quality Analysis\n",
            "| Scenario | Reward | Resolved | DQD Steps | Causal Faithfulness | Stability |",
            "|---|---:|:---:|---:|---:|---:|",
        ]
        for ep in data:
            cf = ep.get("causal_faithfulness") or {}
            cf_score = cf.get("faithfulness_score", "n/a")
            if isinstance(cf_score, (int, float)):
                cf_score = f"{cf_score:.2f}"
            dqd_positive = sum(1 for s in ep["step_analyses"] if s["decision_quality_delta"] >= 0)
            dqd_total = len(ep["step_analyses"])
            lines.append(
                f"| {ep['scenario_id']} | {ep['total_reward']:+.3f} "
                f"| {'✅' if ep['resolved'] else '❌'} "
                f"| {dqd_positive}/{dqd_total} "
                f"| {cf_score} "
                f"| {ep.get('stability_score', 1.0):.2f} |"
            )
        lines.append("")
        lines.append("**Decision Quality Delta (DQD):** positive means the agent chose better than alternatives.")
        lines.append("**Causal Faithfulness:** how well the agent traced the hidden causal graph.")

        # Add step-by-step replay for first episode.
        if data:
            ep = data[0]
            lines.append(f"\n---\n### 🔍 Step-by-Step Replay: {ep['scenario_id']}\n")
            for step in ep["step_analyses"]:
                dqd = step["decision_quality_delta"]
                icon = "✅" if dqd >= 0 else "⚠️"
                lines.append(
                    f"**Step {step['step']}** {icon} Action: `{step['chosen_action']}` "
                    f"| Chosen reward: {step['chosen_downstream_reward']:+.3f} "
                    f"| Best alt: {step['best_alternative_reward']:+.3f} "
                    f"| **DQD: {dqd:+.3f}**"
                )
                for alt in step.get("alternatives", []):
                    lines.append(
                        f"  - Alt: `{alt['action']}` ({alt.get('detail', '')}) → {alt['downstream_reward']:+.3f}"
                    )

        return "\n".join(lines)
    except Exception as exc:
        return f"### Counterfactual Analysis\nError loading: {exc}"


with gr.Blocks(title="Incident Commander — OpenEnv", css=CSS, theme=gr.themes.Soft()) as demo:
    state = gr.State(SessionState())

    gr.Markdown(
        """
# 🚨 Incident Commander
### Long-horizon enterprise outage resolution with delayed consequences & causal reasoning

Built for the **Meta PyTorch OpenEnv Hackathon 2026** | Features: delayed-failure dynamics, governance constraints, causal incident twin
""")

    with gr.Tabs():
        with gr.TabItem("🎮 Interactive Demo"):
            with gr.Row():
                with gr.Column(scale=3):
                    obs_box = gr.Markdown("*Start an episode to begin.*", label="Observation")
                with gr.Column(scale=1):
                    reward_box = gr.Markdown("", label="Step Reward")
                    total_box = gr.Markdown("", label="Cumulative")

            with gr.Row():
                start_btn = gr.Button("▶ New Episode", variant="primary")
                step_btn = gr.Button("⏭ Agent Step", variant="secondary", interactive=False)

            with gr.Row():
                with gr.Column(scale=2):
                    log_box = gr.Markdown("", label="Action Log")
                with gr.Column(scale=3):
                    chart = gr.Plot(label="Reward Curve")

        with gr.TabItem("🏆 Judge Replay"):
            gr.Markdown(
                "One-click counterfactual replay: see how the agent's decisions compare to "
                "alternatives, and how faithfully it traced the hidden causal graph."
            )
            cf_box = gr.Markdown(load_counterfactual_markdown(), label="Counterfactual Analysis")
            refresh_cf_btn = gr.Button("🔄 Refresh Analysis", variant="secondary")
            refresh_cf_btn.click(load_counterfactual_markdown, inputs=[], outputs=[cf_box])

        with gr.TabItem("📊 Results"):
            results_box = gr.Markdown(load_results_markdown(), label="Results")
            refresh_results_btn = gr.Button("Refresh Results", variant="secondary")
            refresh_results_btn.click(load_results_markdown, inputs=[], outputs=[results_box])

    # wire buttons
    start_btn.click(
        start_episode,
        inputs=[state],
        outputs=[state, obs_box, reward_box, total_box, log_box, step_btn, start_btn, chart],
    )
    step_btn.click(
        step_agent,
        inputs=[state],
        outputs=[state, obs_box, reward_box, total_box, log_box, step_btn, start_btn, chart],
    )

    gr.Markdown(
        """
---
Resources: [OpenEnv](https://github.com/meta-pytorch/OpenEnv) | [TRL](https://huggingface.co/docs/trl)
""")

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
