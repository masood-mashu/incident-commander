"""Unit tests for the Incident Commander environment.

v2: expanded with anti-gaming, governance, delayed effects, causal twin,
and OOD/stress tests.
"""

from __future__ import annotations

import unittest

from incident_commander.models import ActionType, IncidentAction
from incident_commander.causal_graph import evaluate_causal_faithfulness
from incident_commander.server.incident_environment import IncidentCommanderEnvironment


class IncidentCommanderEnvironmentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.env = IncidentCommanderEnvironment(max_steps=12)

    def test_reset_returns_initial_alert(self) -> None:
        observation = self.env.reset(scenario_id="bad_deploy_checkout")
        self.assertIn("Latency alert", observation.summary)
        self.assertFalse(observation.resolved)

    def test_correct_diagnosis_and_rollback_resolves_bad_deploy(self) -> None:
        self.env.reset(scenario_id="bad_deploy_checkout")
        self.env.step(
            IncidentAction(ActionType.QUERY_TOOL, tool_name="metrics", target="checkout_service")
        )
        self.env.step(IncidentAction(ActionType.PROPOSE_HYPOTHESIS, cause="bad_deploy"))
        result = self.env.step(
            IncidentAction(
                ActionType.EXECUTE_MITIGATION,
                mitigation="rollback",
                target="checkout_service",
            )
        )
        self.assertTrue(result.observation.resolved)
        self.assertGreater(result.reward, 0.0)

    def test_policy_aware_escalation_is_required_for_db_failover(self) -> None:
        self.env.reset(scenario_id="db_saturation_payments")
        blocked = self.env.step(
            IncidentAction(
                ActionType.EXECUTE_MITIGATION,
                mitigation="failover_db",
                target="payments_db",
            )
        )
        self.assertLess(blocked.reward, 0.0)
        self.assertFalse(blocked.observation.resolved)

        self.env.step(
            IncidentAction(
                ActionType.ESCALATE,
                escalate_to="infra_engineer",
                message="Need approval for DB failover.",
            )
        )
        success = self.env.step(
            IncidentAction(
                ActionType.EXECUTE_MITIGATION,
                mitigation="failover_db",
                target="payments_db",
            )
        )
        self.assertTrue(success.observation.resolved)

    def test_redundant_tool_queries_are_penalized(self) -> None:
        self.env.reset(scenario_id="feature_flag_regional")
        first = self.env.step(IncidentAction(ActionType.QUERY_TOOL, tool_name="metrics", target="api_eu"))
        second = self.env.step(IncidentAction(ActionType.QUERY_TOOL, tool_name="metrics", target="api_eu"))
        self.assertGreater(first.reward, second.reward)

    def test_wrong_mitigation_worsens_outage(self) -> None:
        self.env.reset(scenario_id="feature_flag_regional")
        before = self.env.state().service_health["api_eu"]
        result = self.env.step(
            IncidentAction(ActionType.EXECUTE_MITIGATION, mitigation="rollback", target="api_eu")
        )
        after = self.env.state().service_health["api_eu"]
        self.assertLess(after, before)
        self.assertLess(result.reward, 0.0)

    def test_final_report_can_close_resolved_incident(self) -> None:
        self.env.reset(scenario_id="third_party_search")
        self.env.step(
            IncidentAction(
                ActionType.ESCALATE,
                escalate_to="sre_lead",
                message="Need approval for traffic shift.",
            )
        )
        self.env.step(
            IncidentAction(
                ActionType.EXECUTE_MITIGATION,
                mitigation="traffic_shift",
                target="global",
            )
        )
        result = self.env.step(
            IncidentAction(
                ActionType.CLOSE_INCIDENT,
                report="Vendor dependency issue mitigated with traffic_shift fallback across global traffic.",
            )
        )
        self.assertTrue(result.done)
        self.assertEqual(result.observation.terminal_reason, "resolved")

    def test_train_and_test_scenario_splits_do_not_overlap(self) -> None:
        train_ids = set(self.env.available_scenarios(split="train"))
        test_ids = set(self.env.available_scenarios(split="test"))
        self.assertTrue(train_ids)
        self.assertTrue(test_ids)
        self.assertTrue(train_ids.isdisjoint(test_ids))

    def test_reset_supports_split_sampling(self) -> None:
        obs = self.env.reset(seed=123, split="test")
        state = self.env.state()
        self.assertIn(state.scenario_id, self.env.available_scenarios(split="test"))
        self.assertFalse(obs.resolved)


# ── v2: Anti-gaming tests ──────────────────────────────────────────────────────


class AntiGamingTests(unittest.TestCase):
    """Test that reward hacking strategies are penalized."""

    def setUp(self) -> None:
        self.env = IncidentCommanderEnvironment(max_steps=12)

    def test_spam_close_without_resolution_is_penalized(self) -> None:
        """Closing before resolving must produce negative reward."""
        self.env.reset(scenario_id="bad_deploy_checkout")
        result = self.env.step(
            IncidentAction(ActionType.CLOSE_INCIDENT, report="Closing immediately.")
        )
        self.assertLess(result.reward, 0.0)
        self.assertTrue(result.done)

    def test_escalating_waste_penalty_for_repeated_junk(self) -> None:
        """Repeating the same useless query should get progressively worse."""
        self.env.reset(scenario_id="bad_deploy_checkout")
        rewards = []
        for _ in range(4):
            r = self.env.step(
                IncidentAction(ActionType.QUERY_TOOL, tool_name="metrics", target="checkout_service")
            )
            rewards.append(r.reward)
        # First query is useful, rest are waste — each worse than the last.
        self.assertGreater(rewards[0], rewards[1])
        self.assertGreater(rewards[1], rewards[2])

    def test_escalate_then_close_without_mitigation_fails(self) -> None:
        """Escalating and immediately closing is not a valid resolution path."""
        self.env.reset(scenario_id="db_saturation_payments")
        self.env.step(IncidentAction(ActionType.ESCALATE, escalate_to="infra_engineer"))
        result = self.env.step(
            IncidentAction(ActionType.CLOSE_INCIDENT, report="Escalated, closing.")
        )
        self.assertLess(result.reward, 0.0)


# ── v2: Delayed effects tests ─────────────────────────────────────────────────


class DelayedEffectsTests(unittest.TestCase):
    """Test that delayed consequences fire correctly."""

    def setUp(self) -> None:
        self.env = IncidentCommanderEnvironment(max_steps=12)

    def test_delayed_effect_fires_after_correct_mitigation(self) -> None:
        """Bad deploy has a delayed effect on payments_db 3 steps after rollback."""
        self.env.reset(scenario_id="bad_deploy_checkout")
        self.env.step(IncidentAction(ActionType.QUERY_TOOL, tool_name="metrics", target="checkout_service"))
        self.env.step(IncidentAction(ActionType.PROPOSE_HYPOTHESIS, cause="bad_deploy"))
        self.env.step(IncidentAction(ActionType.EXECUTE_MITIGATION, mitigation="rollback", target="checkout_service"))

        # Verify pending effects were scheduled.
        state = self.env.state()
        self.assertGreater(len(state.pending_effects), 0)

        # Step forward until the delayed effect fires.
        health_before = state.service_health.get("payments_db", 0.96)
        for _ in range(4):
            self.env.step(
                IncidentAction(ActionType.UPDATE_STATUS, message="Monitoring impact. Cause confirmed. Action rollback. Next verify.")
            )
        state = self.env.state()
        health_after = state.service_health.get("payments_db", 0.96)
        self.assertLess(health_after, health_before)
        self.assertGreater(len(state.fired_effects), 0)


# ── v2: Governance tests ──────────────────────────────────────────────────────


class GovernanceTests(unittest.TestCase):
    """Test governance constraints on the flagship scenario."""

    def setUp(self) -> None:
        self.env = IncidentCommanderEnvironment(max_steps=12)

    def test_governance_scenario_loads(self) -> None:
        """The multi-region governance scenario should exist."""
        scenarios = self.env.available_scenarios(split="governance")
        self.assertIn("multi_region_failover_gov", scenarios)

    def test_governance_policy_flag_is_set(self) -> None:
        """Governance scenarios should have the governance_check_required flag."""
        self.env.reset(scenario_id="multi_region_failover_gov")
        state = self.env.state()
        self.assertIn("governance_check_required", state.active_policy_flags)

    def test_governance_blocked_mitigation_is_penalized(self) -> None:
        """Blocked mitigations receive governance penalty."""
        self.env.reset(scenario_id="multi_region_failover_gov")
        self.env.step(IncidentAction(ActionType.ESCALATE, escalate_to="infra_engineer"))
        result = self.env.step(
            IncidentAction(ActionType.EXECUTE_MITIGATION, mitigation="traffic_shift_to_us", target="global")
        )
        self.assertLess(result.reward, 0.0)
        state = self.env.state()
        self.assertGreater(len(state.governance_violations), 0)


# ── v2: OOD and Stress tests ──────────────────────────────────────────────────


class OODStressTests(unittest.TestCase):
    """Test out-of-distribution and stress splits."""

    def setUp(self) -> None:
        self.env = IncidentCommanderEnvironment(max_steps=12)

    def test_ood_split_has_scenarios(self) -> None:
        ood = self.env.available_scenarios(split="ood")
        self.assertGreater(len(ood), 0)
        for sid in ood:
            self.assertIn("ood", sid)

    def test_stress_split_reduces_budget(self) -> None:
        obs = self.env.reset(seed=1, split="stress")
        state = self.env.state()
        self.assertEqual(state.max_steps, 6)

    def test_ood_scenario_runs_without_error(self) -> None:
        self.env.reset(scenario_id="cert_expiry_ood")
        result = self.env.step(
            IncidentAction(ActionType.QUERY_TOOL, tool_name="logs", target="auth_svc")
        )
        self.assertTrue(result.observation.step_count > 0)


# ── v2: Causal faithfulness tests ─────────────────────────────────────────────


class CausalFaithfulnessTests(unittest.TestCase):
    """Test the causal twin evaluation system."""

    def setUp(self) -> None:
        self.env = IncidentCommanderEnvironment(max_steps=12)

    def test_good_investigation_gets_high_faithfulness(self) -> None:
        """Agent that follows the causal chain should score well."""
        self.env.reset(scenario_id="bad_deploy_checkout")
        self.env.step(IncidentAction(ActionType.QUERY_TOOL, tool_name="deploy_history", target="checkout_service"))
        self.env.step(IncidentAction(ActionType.QUERY_TOOL, tool_name="metrics", target="api_gateway"))
        self.env.step(IncidentAction(ActionType.PROPOSE_HYPOTHESIS, cause="bad_deploy"))
        self.env.step(IncidentAction(ActionType.EXECUTE_MITIGATION, mitigation="rollback", target="checkout_service"))

        scenario = self.env._scenario
        state = self.env.state()
        result = evaluate_causal_faithfulness(scenario.causal_edges, state)

        self.assertGreater(result.faithfulness_score, 0.5)
        self.assertGreater(result.discovered_edges, 0)


if __name__ == "__main__":
    unittest.main()
