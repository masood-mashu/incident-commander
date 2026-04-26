# pyright: reportGeneralTypeIssues=false, reportUnknownArgumentType=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownParameterType=false
"""Minimal HF TRL GRPO training script for Incident Commander.

This script mirrors the official TRL + OpenEnv environment-training pattern:
it wraps the local environment as a tool-using class for ``environment_factory``,
trains a small instruction model with GRPO, and writes judge-friendly artifacts.

It is intentionally small and conservative so it can serve as a Colab notebook
backing script for hackathon validation.
"""

from __future__ import annotations

import argparse
import ast
import inspect
import json
from pathlib import Path
import re
import traceback
from typing import Any, cast

import matplotlib
import matplotlib.pyplot as _plt
import torch
from datasets import Dataset  # pyright: ignore[reportMissingTypeStubs]
from peft import LoraConfig
from transformers import AutoTokenizer, set_seed
from trl.trainer.grpo_config import GRPOConfig
from trl.trainer.grpo_trainer import GRPOTrainer

from incident_commander.models import ActionType, IncidentAction
from incident_commander.server.incident_environment import IncidentCommanderEnvironment

matplotlib.use("Agg")
plt = cast(Any, _plt)

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "outputs" / "evals"
TRL_OUTPUT_DIR = OUTPUT_DIR / "trl_grpo"

DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
SYSTEM_PROMPT = (
    "You are an incident commander resolving a live enterprise outage. "
    "Use the available tools to gather evidence, form a hypothesis, execute a safe mitigation, "
    "communicate clearly, and only close the incident after recovery is verified. "
    "Be concise and action-oriented."
)


def format_observation(observation: Any) -> str:
    lines = [
        f"Summary: {observation.summary}",
        f"Alerts: {', '.join(observation.visible_alerts) if observation.visible_alerts else 'none'}",
        f"Step: {observation.step_count}",
        f"Budget remaining: {observation.remaining_budget}",
        f"Resolved: {observation.resolved}",
    ]
    if observation.tool_results:
        lines.append("Recent evidence:")
        for item in observation.tool_results[-3:]:
            lines.append(f"- {item.tool_name}:{item.target} -> {item.content[:160]}")
    if observation.hypotheses:
        lines.append(f"Hypotheses: {', '.join(observation.hypotheses[-3:])}")
    if observation.stakeholder_messages:
        lines.append("Stakeholder messages:")
        for message in observation.stakeholder_messages[-2:]:
            lines.append(f"- {message}")
    return "\n".join(lines)


SCENARIO_SPECS: dict[str, dict[str, str]] = {
    "bad_deploy_checkout": {
        "target": "checkout_service",
        "cause": "bad_deploy",
        "mitigation": "rollback",
        "escalate_to": "sre_lead",
    },
    "db_saturation_payments": {
        "target": "payments_db",
        "cause": "database_saturation",
        "mitigation": "failover_db",
        "escalate_to": "infra_engineer",
    },
    "feature_flag_regional": {
        "target": "api_eu",
        "cause": "feature_flag_misconfig",
        "mitigation": "disable_flag",
        "escalate_to": "sre_lead",
    },
    "third_party_search": {
        "target": "search_vendor",
        "cause": "third_party_failure",
        "mitigation": "traffic_shift",
        "escalate_to": "vendor_support",
    },
}


def _to_text(value: object) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        items = cast(list[object], value)
        parts: list[str] = []
        for item in items:
            if isinstance(item, dict):
                content = cast(dict[str, object], item).get("content")
                if isinstance(content, str):
                    parts.append(content)
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(parts)
    if isinstance(value, dict):
        content = cast(dict[str, object], value).get("content")
        if isinstance(content, str):
            return content
    return str(value)


def _parse_steps_from_completion(text: str) -> list[str]:
    raw = text.strip()
    if not raw:
        return []

    fenced_match = re.search(r"```(?:json)?\s*(.*?)\s*```", raw, flags=re.DOTALL | re.IGNORECASE)
    if fenced_match:
        raw = fenced_match.group(1)

    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            steps = cast(dict[str, object], parsed).get("steps")
            if isinstance(steps, list):
                return [str(step).lower() for step in cast(list[object], steps)]
        if isinstance(parsed, list):
            return [str(step).lower() for step in cast(list[object], parsed)]
    except Exception:
        pass

    try:
        parsed_literal = ast.literal_eval(raw)
        if isinstance(parsed_literal, list):
            return [str(step).lower() for step in cast(list[Any], parsed_literal)]
    except Exception:
        pass

    lines = [line.strip("- *\t ").lower() for line in raw.splitlines() if line.strip()]
    if lines:
        return lines[:8]
    return [raw.lower()]


def _simulate_episode_reward(completion_text: str, scenario_id: str, split: str, seed: int) -> float:
    spec = SCENARIO_SPECS.get(scenario_id, SCENARIO_SPECS["bad_deploy_checkout"])
    target = spec["target"]
    cause = spec["cause"]
    mitigation = spec["mitigation"]
    escalate_to = spec["escalate_to"]

    steps = _parse_steps_from_completion(completion_text)
    joined = "\n".join(steps)

    env = IncidentCommanderEnvironment(max_steps=10)
    env.reset(scenario_id=scenario_id, split=split, seed=seed)
    total = 0.0

    actions: list[IncidentAction] = [
        IncidentAction(ActionType.QUERY_TOOL, tool_name="metrics", target=target),
    ]
    if "logs" in joined or "log" in joined:
        actions.append(IncidentAction(ActionType.QUERY_TOOL, tool_name="logs", target=target))

    proposed_cause = cause if cause in joined else "bad_deploy"
    actions.append(IncidentAction(ActionType.PROPOSE_HYPOTHESIS, cause=proposed_cause))

    if mitigation in {"failover_db", "traffic_shift"} and "escalat" in joined:
        actions.append(
            IncidentAction(
                ActionType.ESCALATE,
                escalate_to=escalate_to,
                message="Need approval for high-risk mitigation.",
            )
        )

    chosen_mitigation = mitigation if mitigation in joined else "rollback"
    actions.append(IncidentAction(ActionType.EXECUTE_MITIGATION, mitigation=chosen_mitigation, target=target))

    if "status" in joined or "stakeholder" in joined or "update" in joined:
        actions.append(
            IncidentAction(
                ActionType.UPDATE_STATUS,
                message="Impact, cause, action, and next step communicated.",
            )
        )

    if "close" in joined or "resolved" in joined or "report" in joined:
        actions.append(
            IncidentAction(
                ActionType.CLOSE_INCIDENT,
                report="Incident resolved with validated mitigation and communication.",
            )
        )

    last_terminal_reason: str | None = None
    resolved = False
    for action in actions:
        result = env.step(action)
        total += float(result.reward)
        resolved = bool(result.observation.resolved)
        last_terminal_reason = result.observation.terminal_reason
        if result.done:
            break

    if resolved:
        total += 0.5
    elif last_terminal_reason is not None and last_terminal_reason != "resolved":
        total -= 0.25
    return float(total)


def reward_func(
    prompts: list[Any], completions: list[Any], **metadata: Any
) -> list[float]:
    rewards: list[float] = []
    scenario_ids = cast(list[str], metadata.get("scenario_id", []))
    splits = cast(list[str], metadata.get("split", []))
    seeds = cast(list[int], metadata.get("seed", []))
    for _, completion, sid, run_split, run_seed in zip(prompts, completions, scenario_ids, splits, seeds, strict=True):
        completion_text = _to_text(completion)
        rewards.append(_simulate_episode_reward(completion_text, str(sid), str(run_split), int(run_seed)))
    return rewards


def build_dataset(repeats: int = 24, split: str = "base", seed_start: int = 11) -> Dataset:
    scenarios = [
        "bad_deploy_checkout",
        "db_saturation_payments",
        "feature_flag_regional",
        "third_party_search",
    ]
    prompts: list[list[dict[str, str]]] = []
    scenario_ids: list[str] = []
    objectives: list[str] = []
    splits: list[str] = []
    seeds: list[int] = []

    for idx in range(repeats):
        scenario_id = scenarios[idx % len(scenarios)]
        scenario_ids.append(scenario_id)
        objectives.append(
            "Investigate the outage, use the tools to gather evidence, choose a safe mitigation, "
            "communicate clearly, and close only after recovery."
        )
        splits.append(split)
        seeds.append(seed_start + idx)
        prompts.append(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"Scenario: {scenario_id}. "
                        "Output ONLY JSON in this format: {\"steps\": [\"...\"]}. "
                        "Use concise incident-response steps that mention evidence gathering, "
                        "hypothesis, mitigation, communication, and closure."
                    ),
                },
            ]
        )

    dataset_input: dict[str, list[Any]] = {
        "prompt": prompts,
        "scenario_id": scenario_ids,
        "objective": objectives,
        "split": splits,
        "seed": seeds,
    }
    return Dataset.from_dict(dataset_input)  # pyright: ignore[reportUnknownMemberType]


def make_grpo_config(output_dir: Path, max_steps: int, learning_rate: float) -> GRPOConfig:
    has_cuda = torch.cuda.is_available()
    bf16 = bool(has_cuda and torch.cuda.get_device_capability()[0] >= 8)
    candidate_kwargs: dict[str, Any] = {
        "output_dir": str(output_dir),
        "learning_rate": learning_rate,
        "per_device_train_batch_size": 2,
        "gradient_accumulation_steps": 1,
        "num_generations": 2,
        "max_steps": max_steps,
        "save_strategy": "no",
        "logging_steps": 1,
        "report_to": [],
        "remove_unused_columns": False,
        "max_prompt_length": 1024,
        "max_completion_length": 768,
        "num_train_epochs": 1,
        "bf16": bf16,
        "fp16": bool(has_cuda and not bf16),
    }
    init_fn: Any = getattr(GRPOConfig, "__init__")
    signature = inspect.signature(init_fn)
    supported = {
        key: value for key, value in candidate_kwargs.items() if key in signature.parameters
    }
    return GRPOConfig(**supported)


def save_training_artifacts(log_history: list[dict[str, Any]], output_dir: Path) -> dict[str, Any]:
    rewards: list[float] = []
    losses: list[float] = []
    steps_reward: list[int] = []
    steps_loss: list[int] = []

    for row in log_history:
        reward_value = None
        if "reward" in row:
            reward_value = row["reward"]
        else:
            for key, value in row.items():
                if key.startswith("reward"):
                    reward_value = value
                    break
        if reward_value is not None:
            rewards.append(float(reward_value))
            steps_reward.append(int(row.get("step", len(steps_reward) + 1)))

        if "loss" in row:
            losses.append(float(row["loss"]))
            steps_loss.append(int(row.get("step", len(steps_loss) + 1)))

    if rewards:
        plt.figure(figsize=(9, 4))
        plt.plot(steps_reward, rewards, marker="o", linewidth=1.5)
        plt.xlabel("Training step")
        plt.ylabel("Reward")
        plt.title("Incident Commander TRL GRPO Reward Curve")
        plt.tight_layout()
        plt.savefig(output_dir / "trl_grpo_reward_curve.png", dpi=140)
        plt.close()

    if losses:
        plt.figure(figsize=(9, 4))
        plt.plot(steps_loss, losses, marker="o", linewidth=1.5, color="tab:orange")
        plt.xlabel("Training step")
        plt.ylabel("Loss")
        plt.title("Incident Commander TRL GRPO Loss Curve")
        plt.tight_layout()
        plt.savefig(output_dir / "trl_grpo_loss_curve.png", dpi=140)
        plt.close()

    summary: dict[str, float | int | None] = {
        "reward_points": len(rewards),
        "loss_points": len(losses),
        "final_reward": rewards[-1] if rewards else None,
        "final_loss": losses[-1] if losses else None,
        "best_reward": max(rewards) if rewards else None,
    }
    (output_dir / "trl_grpo_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def run_training(
    model_name: str = DEFAULT_MODEL,
    max_steps: int = 20,
    dataset_repeats: int = 24,
    seed: int = 42,
    learning_rate: float = 5e-6,
) -> dict[str, Any]:
    set_seed(seed)
    TRL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    dataset = build_dataset(repeats=dataset_repeats)

    # Load tokenizer and set a ChatML template with full tool-calling
    # support so TRL's validation and add_response_schema both pass.
    tokenizer = cast(Any, AutoTokenizer.from_pretrained(model_name))  # pyright: ignore[reportUnknownMemberType]
    CHATML_TOOL_TEMPLATE = (
        "{%- for message in messages %}"
        "{%- if message['role'] == 'system' %}"
        "<|im_start|>system\n{{ message['content'] }}<|im_end|>\n"
        "{%- elif message['role'] == 'user' %}"
        "<|im_start|>user\n{{ message['content'] }}<|im_end|>\n"
        "{%- elif message['role'] == 'assistant' %}"
        "<|im_start|>assistant\n"
        "{%- if message.get('tool_calls') %}"
        "{%- for tc in message['tool_calls'] %}"
        '<tool_call>\n{"name": "{{ tc[\"function\"][\"name\"] }}", "arguments": {{ tc[\"function\"][\"arguments\"] }}}\n</tool_call>\n'
        "{%- endfor %}"
        "{%- else %}"
        "{{ message['content'] }}"
        "{%- endif %}"
        "<|im_end|>\n"
        "{%- elif message['role'] == 'tool' %}"
        "<|im_start|>tool\n{{ message['content'] }}<|im_end|>\n"
        "{%- endif %}"
        "{%- endfor %}"
        "{%- if add_generation_prompt %}"
        "<|im_start|>assistant\n"
        "{%- endif %}"
    )
    tokenizer.chat_template = CHATML_TOOL_TEMPLATE

    trainer = GRPOTrainer(
        model=model_name,
        processing_class=tokenizer,
        reward_funcs=cast(Any, reward_func),
        train_dataset=dataset,
        args=make_grpo_config(TRL_OUTPUT_DIR, max_steps=max_steps, learning_rate=learning_rate),
        peft_config=LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        ),
    )

    result = cast(Any, trainer).train()
    summary = save_training_artifacts(list(trainer.state.log_history), TRL_OUTPUT_DIR)
    merged: dict[str, float | int | None | str] = {
        "model_name": model_name,
        "seed": seed,
        "max_steps": max_steps,
        "dataset_repeats": dataset_repeats,
        "train_runtime": getattr(result, "metrics", {}).get("train_runtime"),
        "train_steps_per_second": getattr(result, "metrics", {}).get("train_steps_per_second"),
        **summary,
    }
    (TRL_OUTPUT_DIR / "trl_grpo_run.json").write_text(
        json.dumps(merged, indent=2), encoding="utf-8"
    )
    return merged


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Incident Commander with TRL GRPO.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="HF model id to fine-tune.")
    parser.add_argument("--max-steps", type=int, default=20, help="Number of GRPO optimizer steps.")
    parser.add_argument("--dataset-repeats", type=int, default=24, help="Number of training episodes.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--learning-rate", type=float, default=5e-6, help="Trainer learning rate.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    TRL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    try:
        summary = run_training(
            model_name=args.model,
            max_steps=args.max_steps,
            dataset_repeats=args.dataset_repeats,
            seed=args.seed,
            learning_rate=args.learning_rate,
        )
        print(json.dumps(summary, indent=2))
        print(f"Saved TRL artifacts in: {TRL_OUTPUT_DIR}")
    except Exception as exc:
        error_payload = {
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "traceback": traceback.format_exc(),
        }
        (TRL_OUTPUT_DIR / "trl_grpo_error.json").write_text(
            json.dumps(error_payload, indent=2), encoding="utf-8"
        )
        print(json.dumps(error_payload, indent=2))
        raise


if __name__ == "__main__":
    main()
