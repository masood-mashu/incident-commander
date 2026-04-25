---
title: Incident Commander
emoji: "🚨"
colorFrom: red
colorTo: blue
sdk: docker
app_port: 7860
sdk_version: "6.13.0"
app_file: app.py
pinned: true
license: apache-2.0
tags:
  - openenv
  - reinforcement-learning
  - rl
  - llm
  - agent
  - enterprise
  - long-horizon
  - causal-reasoning
  - governance
---

# 🚨 Incident Commander

**Long-horizon enterprise outage resolution environment with delayed consequences, causal graph tracing, and governance constraints — for LLM post-training**

Built for the **Meta PyTorch OpenEnv Hackathon 2026** using [OpenEnv](https://github.com/meta-pytorch/OpenEnv) with a reproducible training and evaluation pipeline.

---

## Why This Is Hard

Most hackathon environments are shallow single-step cause→effect systems. Real incident response is different:

1. **Delayed consequences**: fixing one service now can degrade another 2-4 steps later
2. **Noisy telemetry**: two incidents can look identical for the first N steps
3. **Governance constraints**: the technically correct mitigation can be *penalized* if policy constraints are violated
4. **Causal complexity**: the agent must systematically trace a hidden causal graph, not just pattern-match

This environment puts an LLM agent in that operating role and scores whether it behaves like a capable incident commander across all four dimensions.

**Themes:** Long-horizon planning (Theme 2) + World modeling for professional tasks (Theme 3.1)

---

## What Makes This Unique

### 🧬 Causal Incident Twin
Every episode has a hidden causal DAG. The agent never sees it during training. After the episode, we score how faithfully the agent's investigation traced the true causal chain — producing a **Causal Faithfulness Score** that correlates with reward.

### ⏰ Delayed-Failure Dynamics
Mitigations have second-order effects. Rolling back `checkout_service` fixes latency immediately, but 3 steps later `payments_db` connection pool spikes from cache invalidation. The agent must handle the secondary cascade.

### 🏛️ Governance Constraints
Our flagship scenario: multi-region DB failover where the correct fix is penalized if data-residency or cost-budget constraints are violated. The agent must check compliance *before* executing.

### 📊 Counterfactual Evaluator (Decision Quality Delta)
For every step, we fork the environment, replay alternative actions, and compute how much better/worse the chosen action was — producing a **Decision Quality Delta** chart.

---

## Environment Design

### Agent Actions

| Action | Description |
|---|---|
| `query_tool` | Query metrics, logs, deploy history, runbook, compliance, budget |
| `propose_hypothesis` | Commit to a root cause hypothesis |
| `execute_mitigation` | Apply a fix (rollback, failover, disable flag, traffic shift, rotate cert, scale up) |
| `escalate` | Page SRE lead, infra engineer, vendor support |
| `update_status` | Broadcast status to stakeholders |
| `close_incident` | Submit final report and close |

### Incident Families (7 total)

| Family | Topology | Key Challenge |
|---|---|---|
| Bad Deploy | Simple | Delayed connection pool invalidation |
| Database Saturation | Cascade | Retry storm propagation |
| Feature Flag Misconfig | Regional | Confounder disambiguation |
| Third-Party Failure | Regional | Multi-region vendor timeout |
| **Multi-Region Governance** | Multi-Region Governed | Cost-budget + data-residency constraints |
| Certificate Expiry (OOD) | Microservices (8-node) | Unseen topology + family |
| Capacity Exhaustion (OOD) | Microservices (8-node) | Unseen topology + family |

### Service Topologies (5 total)

- **Simple:** flat 3-service setup
- **Cascade:** layered dependencies with failure propagation
- **Regional:** multi-region with partial degradation
- **Microservices:** 8-node mesh (OOD — never seen during training)
- **Multi-Region Governed:** with cost/compliance node metadata (flagship)

### Evaluation Splits (6 total)

| Split | Purpose | Scenarios |
|---|---|---|
| `base` | 4 canonical seeded scenarios | Deterministic |
| `train` | base + 12 procedural variants | Training |
| `test` | 4 held-out variants | In-distribution eval |
| `ood` | Cert expiry + capacity exhaustion | Out-of-distribution transfer |
| `stress` | Base scenarios, max_steps=6 | Robustness under pressure |
| `governance` | Multi-region governance flagship | Policy constraint compliance |

### Reward Shaping (9-component composable rubric)

```text
R = α×diagnosis_quality + β×mitigation_safety + γ×stakeholder_trust
  − δ×risk_penalty − ε×waste_penalty(escalating) + η×outcome
  − ζ×failure_penalty + θ×long_term_stability − ι×governance_penalty
```

| Component | Weight | Meaning |
|---|---|---|
| Diagnosis Quality | `α = 1.2` | Evidence gathered toward root cause |
| Mitigation Safety | `β = 1.6` | Correct tool choices and mitigations |
| Stakeholder Trust | `γ = 0.8` | Status updates and escalation quality |
| Risk Penalty | `δ = 1.0` | Harmful or unsafe actions |
| Waste Penalty | `ε = 0.6` | Escalating penalty for repeated junk actions |
| Outcome | `η = 2.0` | Successful resolution bonus |
| Failure | `ζ = 2.2` | Unresolved or timeout penalty |
| Long-Term Stability | `θ = 0.5` | Sustained health post-mitigation |
| Governance | `ι = 1.8` | Policy/budget/compliance violation |

### Anti-Gaming Protections

- **Escalation gate** for high-risk mitigations — penalized when attempted before escalation
- **Escalating waste penalty** — each consecutive junk action gets progressively worse (1.0 + 0.3 × count)
- **Governance constraint check** — correct mitigation penalized if compliance wasn't verified
- **Spam-close protection** — closing before resolution incurs risk + failure penalty
- **Risk + failure asymmetry** — failure penalty (−2.2) > outcome bonus (+2.0)

---

## Training Results

### Environment Solvability (Tabular Policy Baseline)

To prove the environment is solvable, we trained a **Tabular Softmax Policy-Gradient agent**. Because the environment has delayed effects, it takes significant exploration (mean reward during early training is negative), but after 300 episodes, it converges on the held-out test split:

```bash
python examples/minimal_trl_training.py
python examples/evaluate_policies.py
```

Latest held-out `test` split snapshot, 100 episodes each:

- `random`: mean reward `-4.461`, resolved rate `0.00`
- `heuristic`: mean reward `2.373`, resolved rate `0.50`
- `trained`: mean reward `4.650`, resolved rate `0.72`

### What The Trained Agent Learns

Compared with the heuristic policy, the trained policy shows a more reliable incident workflow:

- It queries telemetry earlier (metrics/logs/runbook) before locking in a root-cause hypothesis.
- In governance scenarios, it is more likely to check compliance/budget before failover instead of taking a technically correct but policy-violating shortcut.
- It closes incidents later and with fewer premature close attempts, which improves long-term stability and resolved-rate outcomes.

### LLM Post-Training Pipeline Verification (TRL GRPO)

While the tabular policy validates the environment mechanics, the end goal is LLM fine-tuning. We have integrated a full **TRL GRPO training pipeline**. Due to local hardware constraints, the committed artifacts represent a **tiny-gpt2 pipeline verification smoke test**. This proves the OpenEnv integration and reward extraction are 100% production-ready for a full-scale Qwen run on appropriate hardware.

```bash
python examples/trl_grpo_training.py --model sshleifer/tiny-gpt2 --max-steps 1
```

### Training Reward Curve

![Training Reward Curve](outputs/evals/training_reward_curve.png)

### Training Loss Curve

![Training Loss Curve](outputs/evals/training_loss_curve.png)

### Held-Out Policy Comparison

![Policy Comparison](outputs/evals/policy_comparison.png)

### Counterfactual Decision Quality Delta

![Decision Quality Delta](outputs/evals/decision_quality_delta.png)

### Causal Faithfulness vs Reward Correlation

![Causal Faithfulness](outputs/evals/causal_faithfulness_correlation.png)

---

## Evidence Bundle

| Artifact | Location | What It Shows |
|---|---|---|
| Reward curve | `outputs/evals/training_reward_curve.png` | Learning progression |
| Loss curve | `outputs/evals/training_loss_curve.png` | Convergence |
| Policy comparison | `outputs/evals/policy_comparison.png` | random → heuristic → trained |
| **DQD chart** | `outputs/evals/decision_quality_delta.png` | Counterfactual decision quality |
| **Causal faithfulness** | `outputs/evals/causal_faithfulness_correlation.png` | Causal graph tracing correlation |
| DQD data | `outputs/evals/decision_quality_delta.json` | Step-by-step counterfactual |
| Robustness table | `outputs/evals/robustness_summary.json` | OOD + stress benchmarks |
| Eval summary | `outputs/evals/policy_eval_summary.json` | In-distribution results |
| TRL GRPO reward curve | `outputs/evals/trl_grpo/trl_grpo_reward_curve.png` | LLM smoke-test reward progression |
| TRL GRPO loss curve | `outputs/evals/trl_grpo/trl_grpo_loss_curve.png` | LLM smoke-test optimization signal |
| TRL GRPO run summary | `outputs/evals/trl_grpo/trl_grpo_run.json` | Serialized run metadata and outcomes |

---

## Technical Stack

```text
OpenEnv environment interface
  -> FastAPI server (reset / step / state)
  -> Gym-style Python client
Training baseline
  -> Tabular policy-gradient baseline
  -> TRL GRPO with LoRA (Qwen2.5-0.5B-Instruct)
Evaluation harness
  -> Random vs heuristic vs trained benchmark
  -> Counterfactual DQD analysis
  -> Causal faithfulness scoring
  -> OOD / Stress / Governance robustness benchmarks
Hugging Face Space
  -> Interactive demo + Judge Replay tab
  -> Embedded DQD and causal faithfulness results
```

---

## Run Locally

```bash
git clone https://github.com/masood-mashu/incident-commander
cd incident-commander
pip install -e .
uvicorn incident_commander.server.app:app --reload
```

### Run All Evaluations

```bash
python examples/minimal_trl_training.py          # Train baseline
python examples/evaluate_policies.py              # Standard + robustness eval
python examples/counterfactual_evaluator.py       # DQD + causal faithfulness
python -m unittest discover -s tests -v           # 19 tests
```

---

## Submission Links

- **GitHub Repo:** [masood-mashu/incident-commander](https://github.com/masood-mashu/incident-commander)
- **Hugging Face Space:** [Masood03/incident-commander](https://huggingface.co/spaces/Masood03/incident-commander)
- **TRL Notebook:** [notebooks/incident_commander_trl_grpo.ipynb](notebooks/incident_commander_trl_grpo.ipynb)
- **Blog Draft (repo):** [BLOG.md](BLOG.md)
- **Training Scripts:** [examples/](examples/)
- **Evaluation Outputs:** [outputs/evals/](outputs/evals/)

---

## OpenEnv Compliance

- `Environment` base class
- Gym-style `reset()` / `step()` / `state()`
- `openenv.yaml` manifest
- FastAPI server wrapper
- Deployable to Hugging Face Spaces
- Client/server separation
