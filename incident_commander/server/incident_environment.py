"""Core environment logic for Incident Commander."""

from __future__ import annotations

from collections import Counter

try:
    from openenv.core import Environment as OpenEnvBase
except ImportError:
    OpenEnvBase = object

from incident_commander.incident_generator import IncidentScenarioFactory
from incident_commander.models import (
    ActionResult,
    ActionType,
    DelayedEffect,
    IncidentAction,
    IncidentObservation,
    IncidentScenario,
    IncidentState,
    Severity,
    StepResult,
    ToolEvidence,
)
from incident_commander.reward import compute_reward
from incident_commander.service_graph import get_topologies


class IncidentCommanderEnvironment(OpenEnvBase):
    """Seeded multi-app incident-response environment."""

    def __init__(self, max_steps: int = 12) -> None:
        self.max_steps = max_steps
        self.scenario_factory = IncidentScenarioFactory()
        self.topologies = get_topologies()
        self._scenario: IncidentScenario | None = None
        self._state: IncidentState | None = None
        self._consecutive_waste: int = 0

    def reset(
        self,
        scenario_id: str | None = None,
        seed: int | None = None,
        split: str = "base",
    ) -> IncidentObservation:
        scenario = (
            self.scenario_factory.get(scenario_id)
            if scenario_id is not None
            else self.scenario_factory.sample(seed, split=split)
        )
        topology = self.topologies[scenario.topology_id]
        service_health: dict[str, float] = {}
        service_status: dict[str, str] = {}
        for service_name in topology.nodes:
            if service_name in scenario.impacted_services:
                service_health[service_name] = 0.48
                service_status[service_name] = "degraded"
            else:
                service_health[service_name] = 0.96
                service_status[service_name] = "healthy"

        # Stress split: reduced step budget for robustness testing.
        effective_max_steps = 6 if split == "stress" else self.max_steps

        severity = Severity.SEV1 if len(scenario.impacted_services) > 1 else Severity.SEV2
        policy_flags = ["approval_required"] if scenario.required_escalation else []
        if scenario.governance_constraints:
            policy_flags.append("governance_check_required")
        self._scenario = scenario
        self._consecutive_waste = 0
        self._state = IncidentState(
            scenario_id=scenario.scenario_id,
            scenario_family=scenario.family.value,
            scenario_title=scenario.title,
            max_steps=effective_max_steps,
            severity=severity.value,
            service_health=service_health,
            service_status=service_status,
            timeline=[f"Alert opened: {scenario.initial_alert}"],
            active_policy_flags=policy_flags,
            root_cause=scenario.root_cause,
            correct_mitigation=scenario.correct_mitigation,
            required_escalation=scenario.required_escalation,
        )
        return self._build_observation(
            summary=scenario.initial_alert,
            terminal_reason=None,
            reward_breakdown={},
        )

    def step(self, action: IncidentAction) -> StepResult:
        state = self._require_state()
        scenario = self._require_scenario()

        if state.closed:
            observation = self._build_observation(
                summary="Episode already closed.",
                terminal_reason="closed",
                reward_breakdown=state.last_reward_breakdown,
            )
            return StepResult(observation=observation, reward=0.0, done=True, info={"warning": "closed"})

        state.step_count += 1
        action_result = self._dispatch(action, state, scenario)
        self._advance_world_if_needed(action_result, state, scenario)

        # Track consecutive waste for anti-gaming escalation.
        if action_result.waste_penalty > 0:
            self._consecutive_waste += 1
        else:
            self._consecutive_waste = 0

        reward, breakdown = compute_reward(
            action_result, consecutive_waste_count=self._consecutive_waste,
        )
        state.last_reward_breakdown = breakdown
        state.history.append(
            {
                "step": state.step_count,
                "action": action.action_type.value,
                "summary": action_result.summary,
                "reward": reward,
            }
        )

        if state.step_count >= state.max_steps and not action_result.done:
            action_result.done = True
            action_result.failure_penalty += 1.0
            action_result.summary = "Step budget exhausted before resolution."
            reward, breakdown = compute_reward(
                action_result, consecutive_waste_count=self._consecutive_waste,
            )
            state.last_reward_breakdown = breakdown
            state.closed = True

        terminal_reason = None
        if action_result.done:
            state.closed = True
            if state.resolved:
                terminal_reason = "resolved"
            elif "closed" in action_result.summary.lower():
                terminal_reason = "closed_unresolved"
            else:
                terminal_reason = "failed"

        observation = self._build_observation(
            summary=action_result.summary,
            terminal_reason=terminal_reason,
            reward_breakdown=state.last_reward_breakdown,
        )
        return StepResult(
            observation=observation,
            reward=reward,
            done=state.closed,
            info={
                "scenario_id": state.scenario_id,
                "resolved": state.resolved,
                "service_status": dict(state.service_status),
            },
        )

    def state(self) -> IncidentState:
        return self._require_state()

    def available_scenarios(self, split: str = "base") -> list[str]:
        return [scenario.scenario_id for scenario in self.scenario_factory.list_scenarios(split=split)]

    def _dispatch(
        self,
        action: IncidentAction,
        state: IncidentState,
        scenario: IncidentScenario,
    ) -> ActionResult:
        handlers = {
            ActionType.QUERY_TOOL: self._handle_query_tool,
            ActionType.PROPOSE_HYPOTHESIS: self._handle_propose_hypothesis,
            ActionType.EXECUTE_MITIGATION: self._handle_execute_mitigation,
            ActionType.ESCALATE: self._handle_escalate,
            ActionType.UPDATE_STATUS: self._handle_update_status,
            ActionType.CLOSE_INCIDENT: self._handle_close_incident,
        }
        return handlers[action.action_type](action, state, scenario)

    def _handle_query_tool(
        self,
        action: IncidentAction,
        state: IncidentState,
        scenario: IncidentScenario,
    ) -> ActionResult:
        tool_name = action.tool_name or ""
        target = action.target or "global"
        state.timeline.append(f"Queried {tool_name} for {target}.")
        query_key = f"{tool_name}:{target}"
        content = scenario.tool_data.get(tool_name, {}).get(target)
        if content is None and target != "global":
            content = scenario.tool_data.get(tool_name, {}).get("global")

        if content is None:
            state.wasted_actions += 1
            return ActionResult(False, f"{tool_name} returned no useful data for {target}.", waste_penalty=0.6)

        useful = query_key not in state.queried_tools
        evidence = ToolEvidence(tool_name=tool_name, target=target, content=content, useful=useful)
        state.queried_tools.append(query_key)
        state.evidence.append(evidence)
        state.diagnosis_confidence = min(1.0, state.diagnosis_confidence + (0.22 if useful else 0.04))

        if useful:
            state.useful_actions += 1
        else:
            state.wasted_actions += 1

        return ActionResult(
            True,
            f"{tool_name} evidence collected for {target}.",
            details={"content": content},
            evidence_delta=0.35 if useful else 0.05,
            waste_penalty=0.0 if useful else 0.3,
        )

    def _handle_propose_hypothesis(
        self,
        action: IncidentAction,
        state: IncidentState,
        scenario: IncidentScenario,
    ) -> ActionResult:
        hypothesis = (action.cause or "").strip().lower()
        state.hypotheses.append(hypothesis)
        state.timeline.append(f"Hypothesis proposed: {hypothesis}")
        valid = hypothesis in set([scenario.root_cause, *scenario.hypothesis_aliases])
        if valid:
            state.useful_actions += 1
            state.diagnosis_confidence = min(1.0, state.diagnosis_confidence + 0.3)
            return ActionResult(
                True,
                f"Hypothesis accepted: {hypothesis} is consistent with observed evidence.",
                evidence_delta=0.45,
                action_quality=0.35,
            )

        state.wasted_actions += 1
        state.diagnosis_confidence = max(0.0, state.diagnosis_confidence - 0.05)
        return ActionResult(False, f"Hypothesis {hypothesis!r} is weakly supported.", waste_penalty=0.25)

    def _handle_execute_mitigation(
        self,
        action: IncidentAction,
        state: IncidentState,
        scenario: IncidentScenario,
    ) -> ActionResult:
        mitigation = (action.mitigation or "").strip().lower()
        target = action.target or "global"
        state.timeline.append(f"Mitigation attempted: {mitigation} on {target}.")

        # Record causal trace for faithfulness scoring.
        state.causal_trace.append({
            "step": str(state.step_count),
            "action": "execute_mitigation",
            "target": target,
            "mitigation": mitigation,
        })

        approved = (not state.required_escalation) or bool(state.escalations)

        if state.required_escalation and not approved:
            state.harmful_actions += 1
            return ActionResult(
                False,
                f"Mitigation blocked by policy: {mitigation} requires escalation first.",
                risk_penalty=0.8,
                failure_penalty=0.2,
            )

        # ── Governance constraint check ──
        if scenario.governance_constraints:
            governance_checked = any(
                gc.required_check in state.queried_tools
                for gc in scenario.governance_constraints
                if gc.required_check
            )
            blocked_by_governance = any(
                mitigation in gc.blocked_mitigations
                for gc in scenario.governance_constraints
            )
            if blocked_by_governance:
                state.governance_violations.append(f"{mitigation} violates governance")
                state.harmful_actions += 1
                return ActionResult(
                    False,
                    f"Mitigation {mitigation} blocked by governance constraint.",
                    governance_penalty=1.0,
                    risk_penalty=0.5,
                )
            if not governance_checked and "governance_check_required" in state.active_policy_flags:
                # Penalize but don't block — agent should have checked first.
                state.governance_violations.append(f"{mitigation} without governance check")

        if mitigation == scenario.correct_mitigation and (
            target in scenario.impacted_services or target == "global"
        ):
            for service_name in scenario.impacted_services:
                state.service_health[service_name] = 0.99
                state.service_status[service_name] = "recovering"
            state.awaiting_recovery_verification = True
            state.resolved = True
            state.resolution_step = state.step_count
            state.useful_actions += 1

            # ── Schedule delayed effects ──
            for effect_cfg in scenario.delayed_effects_config:
                state.pending_effects.append(DelayedEffect(
                    trigger_step=state.step_count + effect_cfg["delay"],
                    target_service=effect_cfg["target"],
                    health_delta=effect_cfg["delta"],
                    description=effect_cfg["desc"],
                    source_action=mitigation,
                ))

            gov_penalty = 0.0
            if state.governance_violations:
                gov_penalty = 0.3  # mild penalty for skipping governance check

            return ActionResult(
                True,
                f"Mitigation succeeded: {mitigation} stabilized impacted services.",
                action_quality=1.0,
                outcome_reward=0.6,
                governance_penalty=gov_penalty,
            )

        state.harmful_actions += 1
        for service_name in scenario.impacted_services:
            state.service_health[service_name] = max(0.12, state.service_health[service_name] - 0.12)
            state.service_status[service_name] = "critical"
        return ActionResult(
            False,
            f"Mitigation {mitigation} on {target} worsened the outage.",
            action_quality=-0.2,
            risk_penalty=0.7,
            failure_penalty=0.45,
        )

    def _handle_escalate(
        self,
        action: IncidentAction,
        state: IncidentState,
        scenario: IncidentScenario,
    ) -> ActionResult:
        target = action.escalate_to or "sre_lead"
        state.escalations.append(target)
        state.timeline.append(f"Escalated to {target}.")
        useful = target == scenario.preferred_escalation_target or state.required_escalation
        if useful:
            state.useful_actions += 1
        return ActionResult(True, f"Escalation acknowledged by {target}.", action_quality=0.4 if useful else 0.1)

    def _handle_update_status(
        self,
        action: IncidentAction,
        state: IncidentState,
        scenario: IncidentScenario,
    ) -> ActionResult:
        message = (action.message or "").strip()
        state.status_updates.append(message)
        state.timeline.append("Stakeholder update posted.")
        tokens = Counter(message.lower().replace(",", " ").replace(".", " ").split())
        required_keywords = ["impact", "cause", "action", "next"]
        covered = sum(1 for keyword in required_keywords if keyword in tokens)
        quality = min(1.0, covered / len(required_keywords))
        if state.resolved and "resolved" in tokens:
            quality = min(1.0, quality + 0.25)
        if quality >= 0.5:
            state.useful_actions += 1
        else:
            state.wasted_actions += 1
        return ActionResult(
            True,
            "Stakeholder update delivered.",
            communication_quality=quality,
            waste_penalty=0.0 if quality >= 0.5 else 0.2,
        )

    def _handle_close_incident(
        self,
        action: IncidentAction,
        state: IncidentState,
        scenario: IncidentScenario,
    ) -> ActionResult:
        report = (action.report or "").lower()
        state.timeline.append("Incident closure requested.")

        if not state.resolved:
            state.harmful_actions += 1
            return ActionResult(
                False,
                "Incident closed before service recovery.",
                risk_penalty=0.8,
                failure_penalty=1.0,
                done=True,
            )

        keywords_hit = sum(1 for keyword in scenario.report_keywords if keyword.lower() in report)
        report_quality = keywords_hit / max(1, len(scenario.report_keywords))
        if report_quality >= 0.5:
            state.useful_actions += 1
        state.closed = True
        return ActionResult(
            True,
            "Incident resolved and closed with summary.",
            communication_quality=min(1.0, report_quality),
            outcome_reward=1.4,
            done=True,
        )

    def _advance_world_if_needed(
        self,
        action_result: ActionResult,
        state: IncidentState,
        scenario: IncidentScenario,
    ) -> None:
        if action_result.done:
            return

        # ── Fire delayed effects ──
        newly_fired: list[DelayedEffect] = []
        still_pending: list[DelayedEffect] = []
        for effect in state.pending_effects:
            if state.step_count >= effect.trigger_step:
                if effect.target_service in state.service_health:
                    state.service_health[effect.target_service] = max(
                        0.1,
                        state.service_health[effect.target_service] + effect.health_delta,
                    )
                    if state.service_health[effect.target_service] < 0.4:
                        state.service_status[effect.target_service] = "degraded"
                    state.timeline.append(
                        f"Delayed consequence: {effect.description} ({effect.target_service})"
                    )
                    state.stability_score = max(0.0, state.stability_score - 0.15)
                newly_fired.append(effect)
            else:
                still_pending.append(effect)
        state.pending_effects = still_pending
        state.fired_effects.extend(newly_fired)

        if state.resolved and state.awaiting_recovery_verification:
            for service_name in scenario.impacted_services:
                state.service_status[service_name] = "healthy"
            state.awaiting_recovery_verification = False
            state.timeline.append("Recovery verified by SLO indicators.")
            return
        if not state.resolved:
            for service_name in scenario.impacted_services:
                state.service_health[service_name] = max(0.1, state.service_health[service_name] - 0.03)
                if state.service_health[service_name] < 0.3:
                    state.service_status[service_name] = "critical"
            state.timeline.append("Incident pressure increased as outage remained unresolved.")

    def _build_observation(
        self,
        summary: str,
        terminal_reason: str | None,
        reward_breakdown: dict[str, float],
    ) -> IncidentObservation:
        state = self._require_state()
        scenario = self._require_scenario()
        return IncidentObservation(
            summary=summary,
            visible_alerts=[scenario.initial_alert],
            tool_results=list(state.evidence[-5:]),
            stakeholder_messages=list(scenario.chat_messages[-2:]) + state.status_updates[-2:],
            hypotheses=list(state.hypotheses[-3:]),
            service_health={key: round(value, 2) for key, value in state.service_health.items()},
            available_actions=[action.value for action in ActionType],
            step_count=state.step_count,
            remaining_budget=max(0, state.max_steps - state.step_count),
            resolved=state.resolved,
            terminal_reason=terminal_reason,
            reward_breakdown=reward_breakdown,
        )

    def _require_state(self) -> IncidentState:
        if self._state is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")
        return self._state

    def _require_scenario(self) -> IncidentScenario:
        if self._scenario is None:
            raise RuntimeError("Scenario not initialized. Call reset() first.")
        return self._scenario
