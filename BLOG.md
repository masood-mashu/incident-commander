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

It is 3 AM. PagerDuty fires. Latency on the checkout service has tripled, error rates are spreading across two regions, and the on-call engineer has about a minute to decide whether this is a bad deploy, a saturated database, or something upstream they cannot see yet.

That rhythm is what inspired this project. Real incident response is not a single clever prompt. It is a sequence of imperfect decisions under pressure: triaging alerts, checking metrics and logs, forming hypotheses, asking for approval on risky actions, keeping humans informed, and only then deciding whether the incident is truly over.

**Incident Commander** is an [OpenEnv](https://github.com/pytorch/openenv)-compatible RL environment that drops an LLM agent into realistic enterprise outages — complete with **delayed consequences**, **hidden causal graphs**, and **governance constraints** — and asks it to resolve them end to end.

👉 **[Try the live demo](https://huggingface.co/spaces/Masood03/incident-commander)** · **[Browse the code](https://github.com/masood-mashu/incident-commander)**

## Why We Built It

Early in development, we had the kind of result that feels reassuring at first: the baseline agent learned to query metrics, propose a hypothesis, and roll back a bad deploy. Reward moved up. Resolved rate improved. It looked like progress.

Then we added **delayed-failure dynamics** — the idea that a "correct" mitigation can create a new problem a few steps later. In one scenario, rolling back `checkout_service` fixes latency immediately, but three steps later `payments_db` starts choking on invalidated connections. Suddenly the training curve stopped feeling neat. The agent was not just solving incidents. Sometimes it was solving one and creating another.

That felt much closer to real operations. A rollback can invalidate cached connections. A failover can overload the replica. A certificate rotation can trigger an auth storm. **In real systems, even the right move has a blast radius.**

That was the moment the environment clicked for us. The earlier agent had learned a shallow cause-to-fix mapping. It did not really understand the outage. It was behaving like an inexperienced responder who follows the playbook without thinking through second-order effects.

## What We Built Differently

### 🧬 The Causal Incident Twin

Every episode defines a hidden causal DAG — a ground-truth map of how the incident propagates through the service graph. The agent never sees it directly. It has to *discover* that chain by investigating.

After the episode, we score **causal faithfulness**: did the agent's queries and hypotheses trace the actual causal chain? Higher reward correlates with higher causal faithfulness, which is exactly what we wanted. The agent is not just collecting points by chance; it is being rewarded for investigating the system in a more grounded way.

### ⏰ Delayed Consequences

Mitigations have second-order effects that fire 2-4 steps later. The agent has to stay situationally aware *after* it thinks the incident is resolved. This is the difference between "fix and forget" and "fix, monitor, and handle the cascade."

### 🏛️ Governance-Constrained Mitigation

Our flagship scenario: a multi-region database failover where the technically correct mitigation gets penalized because:
1. EU data-residency rules prohibit cross-region PII movement
2. The failover cost exceeds the monthly budget threshold

The agent must query compliance and budget tools *before* executing. Skip the check, eat the governance penalty. This looks like frontier enterprise RL — because it is.

### 📊 Counterfactual Decision Quality Delta

For every step, we fork the environment, replay alternative actions, and measure how much better the chosen action was. This produces a **Decision Quality Delta** chart — a simple way to show whether the chosen move was actually better than realistic alternatives.

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

Two things matter here. First, **failure penalty (−2.2) is heavier than outcome bonus (+2.0)**, so the environment is not easy to game with reckless actions. Second, **governance penalty (−1.8)** means a technically correct move can still be wrong in context if it violates policy or budget constraints.

## Results

### Tabular Policy Baseline (Environment Solvability)

To prove the environment is solvable and the reward function is difficult to game, we trained a **Tabular Softmax Policy-Gradient agent**. Because the environment is so complex, it takes significant exploration and reward remains noisy, but after 300 episodes the learned policy resolves more held-out incidents than the scripted heuristic:

| Policy | Mean Reward | Resolved Rate |
|---|---|---|
| Random | −4.697 | 0% |
| Heuristic | +2.217 | 50% |
| **Trained** | **+1.199** | **72%** |

### What the Charts Show

1. **Reward curve**: noisy but real learning over 300 episodes
2. **Decision Quality Delta**: the trained policy consistently chooses actions with higher downstream reward than alternatives
3. **Causal faithfulness**: higher reward episodes correlate with more faithful causal chain tracing

### LLM Post-Training Pipeline (TRL GRPO)

While the tabular policy proves the environment works, the long-term goal is LLM fine-tuning. We integrated a full **TRL GRPO training pipeline** around a deliberately small model, `Qwen/Qwen2.5-0.5B-Instruct`, because we wanted something judges could realistically rerun. The committed artifact bundle comes from a verified 20-step GPU run, and the Colab notebook in the repo is configured for a longer rerunnable pass.

**Reward Progression:**
![TRL GRPO Reward Curve](outputs/evals/trl_grpo/trl_grpo_reward_curve.png)
*The reward progression over 20 optimization steps showing positive skill acquisition.*

### What We Think Judges Should Notice

In the governance scenario, the heuristic policy can perform a technically successful failover and still lose because it skipped compliance. That is the behavior we wanted to surface. The environment should reward not just operational competence, but disciplined operational judgment.

In other words, this project is not trying to be a flashy outage simulator. It is trying to be a useful training ground for agents that need to reason through messy professional workflows where the obvious answer is not always the complete answer.

## What's Next

- **More delayed-effect chains** — cascading failures across 3+ services
- **Adversarial confounders** — incidents that look identical but require different mitigations
- **Multi-agent coordination** — real war rooms have multiple responders
- **Human SRE evaluation** — do the agent's investigation patterns match expert intuition?

## Try It Yourself

- 🎮 **[Live Demo](https://huggingface.co/spaces/Masood03/incident-commander)** — interactive demo + judge replay tab
- 💻 **[GitHub](https://github.com/masood-mashu/incident-commander)** — environment, training, evaluation
- 📊 **[Evidence Bundle](https://github.com/masood-mashu/incident-commander/tree/main/outputs/evals)** — all charts and data

*Built for the Meta PyTorch OpenEnv Hackathon 2026. Powered by OpenEnv, TRL, causal reasoning, and a strong preference for agents that investigate before they act.*
