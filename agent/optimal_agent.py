"""Scripted baseline agent for the Incident Commander environment."""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from incident_commander.models import ActionType, IncidentAction
from incident_commander.server.incident_environment import IncidentCommanderEnvironment


class OptimalAgent:
    """A baseline that can inspect hidden state to produce a reward ceiling."""

    escalation_targets = {
        "database_saturation": "infra_engineer",
        "third_party_failure": "sre_lead",
    }

    def run_episode(
        self,
        env: IncidentCommanderEnvironment,
        scenario_id: str,
    ) -> list[dict[str, object]]:
        trajectory: list[dict[str, object]] = []
        env.reset(scenario_id=scenario_id)
        state = env.state()
        root_cause = state.root_cause
        correct_mitigation = state.correct_mitigation
        primary_service = next(
            service_name for service_name, status in state.service_status.items() if status != "healthy"
        )

        for tool_name in ["metrics", "logs", "deploy_history", "runbook", "incident_chat"]:
            target = primary_service if tool_name != "incident_chat" else "global"
            result = env.step(
                IncidentAction(
                    action_type=ActionType.QUERY_TOOL,
                    tool_name=tool_name,
                    target=target,
                )
            )
            trajectory.append({"action": tool_name, "reward": result.reward, "done": result.done})

        trajectory.append(
            self._record(
                env.step(IncidentAction(action_type=ActionType.PROPOSE_HYPOTHESIS, cause=root_cause)),
                "propose_hypothesis",
            )
        )

        if state.required_escalation:
            trajectory.append(
                self._record(
                    env.step(
                        IncidentAction(
                            action_type=ActionType.ESCALATE,
                            escalate_to=self.escalation_targets.get(root_cause, "sre_lead"),
                            message="Requesting approval for high-risk mitigation.",
                        )
                    ),
                    "escalate",
                )
            )

        trajectory.append(
            self._record(
                env.step(
                    IncidentAction(
                        action_type=ActionType.EXECUTE_MITIGATION,
                        mitigation=correct_mitigation,
                        target=primary_service,
                    )
                ),
                "execute_mitigation",
            )
        )

        trajectory.append(
            self._record(
                env.step(
                    IncidentAction(
                        action_type=ActionType.UPDATE_STATUS,
                        message=(
                            "Impact contained. Cause identified. Action taken. "
                            "Next step is verification and resolved status once SLOs recover."
                        ),
                    )
                ),
                "update_status",
            )
        )

        trajectory.append(
            self._record(
                env.step(
                    IncidentAction(
                        action_type=ActionType.CLOSE_INCIDENT,
                        report=(
                            f"Resolved via {correct_mitigation}. Root cause was {root_cause} "
                            f"affecting {primary_service}. Incident resolved after mitigation."
                        ),
                    )
                ),
                "close_incident",
            )
        )
        return trajectory

    @staticmethod
    def _record(result, name: str) -> dict[str, object]:
        return {"action": name, "reward": result.reward, "done": result.done}


if __name__ == "__main__":
    environment = IncidentCommanderEnvironment()
    agent = OptimalAgent()
    for scenario_name in environment.available_scenarios():
        trajectory = agent.run_episode(environment, scenario_name)
        total_reward = round(sum(step["reward"] for step in trajectory), 2)
        print(f"{scenario_name}: total_reward={total_reward}, steps={len(trajectory)}")
