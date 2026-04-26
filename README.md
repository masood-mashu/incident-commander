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

It's 3 AM. PagerDuty fires. Latency on the checkout service just tripled, error rates are climbing across two regions, and the on-call engineer has sixty seconds to decide: is this a bad deploy, a saturated database, or something upstream they can't even see yet?

Real incident response isn't a single clever prompt — it's a sustained, multi-step workflow under pressure with partial information. You triage alerts, query dashboards, form hypotheses, request approval for risky mitigations, communicate status to stakeholders, and only then — maybe — close the incident. **No existing RL environment captures this.**

**Incident Commander** is an [OpenEnv](https://github.com/pytorch/openenv)-compatible RL environment that drops an LLM agent into realistic enterprise outages — complete with **delayed consequences**, **hidden causal graphs**, and **governance constraints** — and measures whether it can resolve them end-to-end.

📝 **[Read the full writeup → BLOG.md](./BLOG.md)** · 🎮 **[Live Demo](https://huggingface.co/spaces/Masood03/incident-commander)** · 💻 **[GitHub](https://github.com/masood-mashu/incident-commander)** · 📓 **[Training Notebook (Colab)](notebooks/incident_commander_trl_grpo.ipynb)**

Built for the **Meta PyTorch OpenEnv Hackathon 2026** | Themes: Long-horizon planning (Theme 2) + World modeling for professional tasks (Theme 3.1)

---

## What The Agent Sees and Does

The agent observes alerts, service health, telemetry, and stakeholder messages. It chooses from six action types per step:

| Action | Description |
|---|---|
| `query_tool` | Query metrics, logs, deploy history, runbook, compliance, budget |
| `propose_hypothesis` | Commit to a root cause hypothesis |
| `execute_mitigation` | Apply a fix (rollback, failover, disable flag, traffic shift, rotate cert, scale up) |
| `escalate` | Page SRE lead, infra engineer, vendor support |
| `update_status` | Broadcast status to stakeholders |
| `close_incident` | Submit final report and close |

The episode ends when the agent closes the incident (with or without resolution) or exhausts its step budget.

---

## Why This Is Hard

Most RL environments are shallow single-step cause→effect systems. Real incident response is different:

1. **Delayed consequences**: fixing one service now can degrade another 2-4 steps later — a rollback invalidates cached connections, a failover creates traffic pressure on the replica
2. **Noisy telemetry**: two incidents can look identical for the first N steps — the agent must disambiguate through targeted investigation
3. **Governance constraints**: the technically correct mitigation can be *penalized* if policy constraints are violated — EU data residency, cost budgets, escalation gates
4. **Causal complexity**: the agent must systematically trace a hidden causal graph, not just pattern-match

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

To prove the environment is solvable, we trained a **Tabular Softmax Policy-Gradient agent**. Because the environment has delayed effects, it takes significant exploration (mean reward during training remains noisy), but after 300 episodes the learned policy resolves more held-out incidents than the scripted heuristic:

| Policy | Mean Reward | Resolved Rate |
|---|---:|---:|
| Random | −4.70 | 0% |
| Heuristic | +2.22 | 50% |
| **Trained** | **+1.20** | **72%** |

### What The Trained Agent Learns

Compared with the heuristic policy, the trained policy shows a more reliable incident workflow:

- It queries telemetry earlier (metrics/logs/runbook) before locking in a root-cause hypothesis.
- In governance scenarios, it is more likely to check compliance/budget before failover instead of taking a technically correct but policy-violating shortcut.
- It closes incidents later and with fewer premature close attempts, which improves long-term stability and resolved-rate outcomes.

### LLM Post-Training Pipeline Verification (TRL GRPO)

While the tabular policy validates the environment mechanics, the end goal is LLM fine-tuning. We integrated a full **TRL GRPO training pipeline** around a small, judge-friendly model: `Qwen/Qwen2.5-0.5B-Instruct`. The committed artifact bundle below comes from a verified 200-step GPU run, and the Colab notebook in this repo is configured to rerun that same 200-step pass end to end.

```bash
python examples/trl_grpo_training.py --model Qwen/Qwen2.5-0.5B-Instruct --max-steps 200 --dataset-repeats 96
```

### TRL GRPO Reward Curve

![TRL GRPO Reward Curve](outputs/evals/trl_grpo/trl_grpo_reward_curve.png)
*Reward progression from the verified 200-step GPU run. The curve rises from -2.9 to +4.8 (peak 8.84), confirming that the environment produces a learnable reward signal for LLM post-training.*

### TRL GRPO Loss Curve

![TRL GRPO Loss Curve](outputs/evals/trl_grpo/trl_grpo_loss_curve.png)
*Policy loss converging during Qwen2.5-0.5B-Instruct fine-tuning, showing stable optimization without divergence.*

### Held-Out Policy Comparison

![Policy Comparison](outputs/evals/policy_comparison.png)
*Random (no signal) → heuristic (scripted playbook) → trained (learned from 300 episodes). The trained policy achieves +1.20 mean reward and 72% resolution rate on the held-out test split.*

### Counterfactual Decision Quality Delta

![Decision Quality Delta](outputs/evals/decision_quality_delta.png)
*Positive DQD = the agent chose better than alternatives. The trained policy shows strong DQD on mitigation decisions (steps 2, 5, 6 in bad_deploy: DQD +2.16). Negative DQD on later steps (3, 4, 7, 8) reflects over-querying after resolution — expected behavior since the tabular policy lacks state memory to recognize resolution already occurred.*

### Causal Faithfulness vs Reward Correlation

![Causal Faithfulness](outputs/evals/causal_faithfulness_correlation.png)
*Higher reward episodes correlate with more faithful causal chain tracing, confirming the agent learns to investigate systematically rather than pattern-match.*

### Understanding the Evaluation Metrics

**Negative DQD steps**: After resolving the incident, the tabular policy continues querying tools because it has no persistent memory to recognize that resolution already occurred. These post-resolution queries score negative DQD because a status update or close action would yield higher downstream reward. The key mitigation decisions (steps 2, 5, 6) consistently show positive DQD.

**Low causal faithfulness for tabular policy** (~0.23 average): The tabular policy operates on discretized observation features, not free-text reasoning. It cannot log explicit causal hypotheses — it selects actions implicitly. The faithfulness scorer rewards hypothesis alignment and targeted tool queries, which a text-based LLM agent would naturally produce but a tabular agent does only incidentally. The correlation between higher reward and higher faithfulness still holds, confirming the environment rewards systematic investigation.

**Heuristic median (8.74) vs mean (2.22)**: This gap reveals a bimodal distribution — the heuristic either fully solves an episode (when it recognizes the scenario family) or completely fails (when it doesn't). The trained policy improves resolved rate to 72% with a +3.18 median reward, demonstrating broader coverage rather than lucky scripting.

### Robustness Benchmarks

| Policy | Split | Episodes | Mean Reward | Resolved Rate |
|---|---|---:|---:|---:|
| Heuristic | test | 100 | +2.22 | 50% |
| **Trained** | **test** | **100** | **+1.20** | **72%** |
| Heuristic | ood | 50 | −8.41 | 0% |
| Trained | ood | 50 | −10.29 | 0% |
| Heuristic | stress | 50 | +2.16 | 50% |
| **Trained** | **stress** | **50** | **+2.27** | **74%** |
| Heuristic | governance | 30 | −4.28 | 0% |
| Trained | governance | 30 | −6.51 | 0% |

*OOD scenarios use unseen 8-node microservices topologies and novel incident families (certificate expiry, capacity exhaustion) — neither policy has seen these during training, so 0% resolution is expected and demonstrates the environment's genuine difficulty. Under stress (6-step budget), the trained policy maintains 74% resolution rate vs 50% for heuristic, showing it prioritizes high-value actions under pressure. Governance scenarios require compliance and budget checks that neither the heuristic nor tabular policy is equipped to perform — this is precisely the gap that LLM post-training targets.*

### Training Materials

- **Rerunnable Colab notebook:** `notebooks/incident_commander_trl_grpo.ipynb`
- **Training script:** `examples/trl_grpo_training.py`
- **Tracked run metadata:** `outputs/evals/trl_grpo/trl_grpo_run.json`
- **Tracked summary metrics:** `outputs/evals/trl_grpo/trl_grpo_summary.json`

The notebook is written for judges first: clone, install, run, and inspect artifacts. We intentionally use a small model and LoRA-based post-training so the training loop is realistic to rerun instead of being a one-off giant-model demo.

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
| Robustness table | `outputs/evals/robustness_summary.json` | OOD + stress + governance benchmarks |
| Eval summary | `outputs/evals/policy_eval_summary.json` | In-distribution results |
| TRL GRPO reward curve | `outputs/evals/trl_grpo/trl_grpo_reward_curve.png` | LLM reward progression |
| TRL GRPO loss curve | `outputs/evals/trl_grpo/trl_grpo_loss_curve.png` | LLM optimization signal |
| TRL GRPO run summary | `outputs/evals/trl_grpo/trl_grpo_run.json` | Serialized run metadata and outcomes |

---

## API Endpoints

The environment is accessible as a REST API at the Space URL. The FastAPI endpoints are mounted alongside the Gradio UI on the same port (7860):

```bash
# Health check
curl https://masood03-incident-commander.hf.space/health
# → {"status": "ok"}

# Reset environment (start new episode)
curl -X POST https://masood03-incident-commander.hf.space/reset \
  -H "Content-Type: application/json" \
  -d '{"scenario_id": "bad_deploy_checkout"}'

# Take an action
curl -X POST https://masood03-incident-commander.hf.space/step \
  -H "Content-Type: application/json" \
  -d '{"action_type": "query_tool", "tool_name": "metrics", "target": "checkout_service"}'

# Get current state
curl https://masood03-incident-commander.hf.space/state
```

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
pip install -e ".[training]"
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
- **Blog Writeup:** [BLOG.md](BLOG.md)
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
