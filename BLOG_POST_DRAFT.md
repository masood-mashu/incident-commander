# Incident Commander

Incident Commander is an OpenEnv-compatible reinforcement learning environment for long-horizon enterprise outage response. Instead of solving a single QA task, the agent has to investigate alerts, gather evidence, form hypotheses, execute safe mitigations, communicate status, and close the incident correctly.

The environment is designed around realistic operational constraints. Observations are partially observable, incidents span multiple service topologies, and some mitigations require escalation approval before they can be executed. We modeled four incident families: bad deploys, database saturation, feature-flag misconfiguration, and third-party dependency failures. To reduce memorization, training uses a procedural train split and evaluation is reported on held-out test scenarios.

For reproducibility, the project includes a lightweight baseline trainer and an evaluation harness. The current benchmark compares random, heuristic, and trained policies on the held-out test split. In the latest run, the trained policy outperforms both baselines on mean reward and resolved-rate, showing that the environment provides a learnable signal while still requiring multi-step reasoning.

The Hugging Face Space exposes the environment through a Gradio demo, including a scripted walkthrough and benchmark summary. The repository also includes generated training curves, policy comparison plots, unit tests, and a notebook for rerunning the pipeline.

Suggested publish links:
- Hugging Face article: replace this file with the final article URL in the README
- Optional short video: add a YouTube link if you record one
