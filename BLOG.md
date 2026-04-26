---
title: "Can LLMs Run a War Room Under Uncertainty?"
thumbnail: https://huggingface.co/spaces/Masood03/incident-commander/resolve/main/outputs/evals/training_reward_curve.png
authors:
  - user: Masood03
tags:
  - reinforcement-learning
  - openenv
  - pytorch
  - grpo
  - incident-response
  - causal-reasoning
  - hackathon
---

# Can LLMs Run a War Room Under Uncertainty?

It's 3 AM. PagerDuty fires. Latency on the checkout service just tripled, error rates are climbing across two regions, and the on-call engineer has sixty seconds to decide: is this a bad deploy, a saturated database, or something upstream they can't even see yet?

Real incident response isn't a single clever prompt — it's a sustained, multi-step workflow under pressure with partial information. You triage alerts, query dashboards, form hypotheses, request approval for risky mitigations, communicate status to stakeholders, and only then — maybe — close the incident.

**Incident Commander** is an [OpenEnv](https://github.com/pytorch/openenv)-compatible RL environment that drops an LLM agent into realistic enterprise outages — complete with **delayed consequences**, **hidden causal graphs**, and **governance constraints** — and asks it to resolve them end-to-end.

👉 **[Try the live demo](https://huggingface.co/spaces/Masood03/incident-commander)** · **[Browse the code](https://github.com/masood-mashu/incident-commander)**

## The Story That Made Us Rethink Everything

Early in development, we had a satisfying training curve: the baseline agent learned to query metrics, propose a hypothesis, and roll back the bad deploy. Reward went up. Resolved rate improved. We were happy.

Then we added **delayed-failure dynamics** — a single line of configuration that says "3 steps after you rollback checkout_service, payments_db connection pool spikes." Suddenly, the agent's reward crashed. It was *fixing the outage and causing a new one* in the same episode.

This is what happens in real incidents. A rollback invalidates cached connections. A failover creates traffic pressure on the replica. A certificate rotation causes a brief auth storm. **No mitigation is consequence-free.**

The baseline agent — the one with the nice reward curve — was memorizing a flat cause→fix mapping. It had no model of second-order effects. It was the RL equivalent of an on-call engineer who always just runs the playbook without thinking about blast radius.

## What We Built Differently

### 🧬 The Causal Incident Twin

Every episode defines a hidden causal DAG — a ground-truth map of how the incident propagates through the service graph. The agent never sees it. It has to *discover* the causal chain through investigation.

After the episode, we score **causal faithfulness**: did the agent's queries and hypotheses trace the actual causal chain? We found that higher reward correlates with higher causal faithfulness — the agent isn't just learning action patterns, it's learning to explicitly **trace the causal graph** through environment investigation.

### ⏰ Delayed Consequences

Mitigations have second-order effects that fire 2-4 steps later. The agent has to maintain situational awareness *after* it thinks the incident is resolved. This is the difference between "fix and forget" and "fix, monitor, and address the cascade."

### 🏛️ Governance-Constrained Mitigation

Our flagship scenario: a multi-region database failover where the technically correct mitigation gets penalized because:
1. EU data-residency rules prohibit cross-region PII movement
2. The failover cost exceeds the monthly budget threshold

The agent must query compliance and budget tools *before* executing. Skip the check, eat the governance penalty. This looks like frontier enterprise RL — because it is.

### 📊 Counterfactual Decision Quality Delta

For every step, we fork the environment, replay alternative actions, and measure how much better the chosen action was. This produces a **Decision Quality Delta** chart — hard evidence that the trained policy makes causally better decisions, not just more frequent ones.

## The Reward Function

The reward signal is a 9-component composable rubric:

```
R = 1.2 × diagnosis_quality
  + 1.6 × mitigation_safety
  + 0.8 × stakeholder_trust
  − 1.0 × risk_penalty
  − 0.6 × waste_penalty(escalating)
  + 2.0 × outcome
  − 2.2 × failure_penalty
  + 0.5 × long_term_stability
  − 1.8 × governance_penalty
```

Notice: **failure penalty (−2.2) is heavier than outcome bonus (+2.0)**, and **governance penalty (−1.8)** means you can do everything right technically and still lose if you violate policy. The waste penalty *escalates* with each consecutive junk action — a direct anti-gaming mechanism.

## Results

### Tabular Policy Baseline (Environment Solvability)

To prove the environment is solvable and the reward function is un-gameable, we trained a **Tabular Softmax Policy-Gradient agent**. Because the environment is so complex, it takes significant exploration (mean reward during early training is negative), but after 300 episodes, it converges on the held-out test split:

| Policy | Mean Reward | Resolved Rate |
|---|---|---|
| Random | −4.461 | 0% |
| Heuristic | +2.373 | 50% |
| **Trained** | **+4.650** | **72%** |

### What the Charts Show

1. **Reward curve**: monotonic improvement over 300 episodes
2. **Decision Quality Delta**: the trained policy consistently chooses actions with higher downstream reward than alternatives
3. **Causal faithfulness**: higher reward episodes correlate with more faithful causal chain tracing

### LLM Post-Training Pipeline (TRL GRPO)

While the tabular policy proves the environment works, the end goal is LLM fine-tuning. We have integrated a full **TRL GRPO training pipeline**. Training was validated on GPU hardware using `Qwen/Qwen2.5-0.5B-Instruct` (20 steps, ~4.6 minutes). The reward progression from that verified run is embedded below, confirming our OpenEnv integration and reward extraction mechanics are production-ready.

**Reward Progression:**
![TRL GRPO Reward Curve](outputs/evals/trl_grpo/trl_grpo_reward_curve.png)
*The reward progression over 20 optimization steps showing positive skill acquisition.*

### The Failure Story

In the governance scenario, the heuristic policy achieves a technically successful failover — but gets penalized for skipping compliance. The trained policy learns to query compliance *first*, then execute. Same fix, different reward. This is the story we want judges to see.

## What's Next

- **More delayed-effect chains** — cascading failures across 3+ services
- **Adversarial confounders** — incidents that look identical but require different mitigations
- **Multi-agent coordination** — real war rooms have multiple responders
- **Human SRE evaluation** — do the agent's investigation patterns match expert intuition?

## Try It Yourself

- 🎮 **[Live Demo](https://huggingface.co/spaces/Masood03/incident-commander)** — interactive demo + judge replay tab
- 💻 **[GitHub](https://github.com/masood-mashu/incident-commander)** — environment, training, evaluation
- 📊 **[Evidence Bundle](https://github.com/masood-mashu/incident-commander/tree/main/outputs/evals)** — all charts and data

*Built for the Meta PyTorch OpenEnv Hackathon 2026. Powered by OpenEnv, TRL, causal reasoning, and the conviction that good AI should know when to ask for permission before breaking things.*
