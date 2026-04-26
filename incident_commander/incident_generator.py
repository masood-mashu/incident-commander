"""Scenario factory for the Incident Commander MVP."""

from __future__ import annotations

import copy
import random

from incident_commander.models import (
    CausalEdge,
    GovernanceConstraint,
    IncidentFamily,
    IncidentScenario,
)


def build_scenarios() -> list[IncidentScenario]:
    return [
        IncidentScenario(
            scenario_id="bad_deploy_checkout",
            title="Checkout latency after bad deploy",
            family=IncidentFamily.BAD_DEPLOY,
            description="A canary deploy to checkout_service introduced a latency regression.",
            topology_id="simple",
            initial_alert="Latency alert: checkout_service p95 > 2.8s, SLO burn 4.2x",
            root_cause="bad_deploy",
            impacted_services=["checkout_service", "api_gateway"],
            correct_mitigation="rollback",
            deploy_service="checkout_service",
            hypothesis_aliases=["bad_deploy", "deploy_regression", "rollback_needed"],
            tool_data={
                "metrics": {
                    "checkout_service": "p95 latency spiked from 220ms to 2.8s immediately after build 2026.04.20-rc3.",
                    "api_gateway": "Gateway error rate increased due to upstream timeout budget exhaustion.",
                },
                "logs": {
                    "checkout_service": "Repeated warning: serializer timeout after deployment hash 9af31c.",
                },
                "deploy_history": {
                    "checkout_service": "Recent deploy: 14 minutes ago by release-bot. Rollback candidate available.",
                },
                "incident_chat": {
                    "global": "PM: checkout conversion is dropping. Can we confirm if this is tied to the canary?",
                },
                "runbook": {
                    "checkout_service": "Runbook: if regression starts immediately after deploy, rollback is the first safe mitigation.",
                },
            },
            runbook_notes=[
                "Rollback is preferred over restart when regressions align with a fresh deploy.",
            ],
            chat_messages=[
                "SRE lead: confirm blast radius before you rollback.",
                "Support: multiple checkout failures reported in the last 10 minutes.",
            ],
            report_keywords=["rollback", "deploy", "latency", "checkout_service"],
            causal_edges=[
                CausalEdge("bad_deploy", "checkout_service", "deploy introduced serializer regression", "deploy_history:checkout_service"),
                CausalEdge("checkout_service", "api_gateway", "upstream timeout budget exhausted", "metrics:api_gateway"),
            ],
            delayed_effects_config=[
                {"delay": 3, "target": "payments_db", "delta": -0.25, "desc": "Connection pool invalidated by rollback"},
            ],
        ),
        IncidentScenario(
            scenario_id="db_saturation_payments",
            title="Payments DB saturation cascading into timeouts",
            family=IncidentFamily.DATABASE_SATURATION,
            description="payments_db is saturated and application retries are causing cascading timeouts.",
            topology_id="cascade",
            initial_alert="Error budget alert: checkout failures > 18%, DB saturation critical",
            root_cause="database_saturation",
            impacted_services=["payments_db", "checkout_service", "web_frontend"],
            correct_mitigation="failover_db",
            required_escalation=True,
            preferred_escalation_target="infra_engineer",
            hypothesis_aliases=["database_saturation", "db_hotspot", "capacity_exhaustion"],
            tool_data={
                "metrics": {
                    "payments_db": "CPU 97%, connections 99%, disk queue elevated. Retry storm detected from checkout_service.",
                    "checkout_service": "Timeouts rose after DB saturation crossed 90%.",
                },
                "logs": {
                    "payments_db": "Connection pool exhausted. Primary node rejecting new sessions.",
                },
                "deploy_history": {
                    "payments_db": "No deploys in the last 48 hours.",
                },
                "incident_chat": {
                    "global": "Infra on-call: failover is possible but requires explicit handoff because it touches primary storage.",
                },
                "runbook": {
                    "payments_db": "Runbook: if saturation is persistent and retries amplify impact, escalate before DB failover.",
                },
            },
            runbook_notes=["DB failover is risky and needs infra approval first."],
            chat_messages=[
                "Finance ops: payment confirmations are delayed.",
                "SRE lead: do not restart app tier repeatedly if the DB is already saturated.",
            ],
            report_keywords=["failover", "database", "saturation", "payments_db"],
            causal_edges=[
                CausalEdge("database_saturation", "payments_db", "long queries saturated primary", "metrics:payments_db"),
                CausalEdge("payments_db", "checkout_service", "retry storm from connection exhaustion", "logs:payments_db"),
                CausalEdge("checkout_service", "web_frontend", "cascading timeout propagation", "metrics:checkout_service"),
            ],
        ),
        IncidentScenario(
            scenario_id="feature_flag_regional",
            title="Regional outage from feature flag misconfiguration",
            family=IncidentFamily.FEATURE_FLAG,
            description="A feature flag rollout broke request handling in one region.",
            topology_id="regional",
            initial_alert="Regional alert: eu-west-1 error rate > 24% for checkout API",
            root_cause="feature_flag_misconfig",
            impacted_services=["api_eu"],
            correct_mitigation="disable_flag",
            hypothesis_aliases=["feature_flag_misconfig", "flag_rollout", "regional_config_bug"],
            region="eu-west-1",
            tool_data={
                "metrics": {
                    "api_eu": "Error rate isolated to eu-west-1 after new_flag rollout. us-east-1 remains healthy.",
                },
                "logs": {
                    "api_eu": "Missing config key for staged payment experiment 'new_flag_v2'.",
                },
                "deploy_history": {
                    "api_eu": "No binary deploy. Feature flag new_flag_v2 enabled in eu-west-1 11 minutes ago.",
                },
                "incident_chat": {
                    "global": "PM: issue seems regional. Can we disable the experiment without global impact?",
                },
                "runbook": {
                    "api_eu": "Runbook: disable regional experiment flags before attempting broader rollback.",
                },
            },
            runbook_notes=["Prefer disabling the flag instead of rolling back unrelated services."],
            chat_messages=["Customer ops: EU customers are seeing request failures."],
            report_keywords=["feature_flag", "eu-west-1", "disable", "regional"],
            confounders=["bad_deploy"],
            disambiguating_tool="deploy_history:api_eu",
            causal_edges=[
                CausalEdge("feature_flag_misconfig", "api_eu", "missing config key in staged experiment", "api_eu"),
            ],
        ),
        IncidentScenario(
            scenario_id="third_party_search",
            title="Third-party dependency failure with ambiguous telemetry",
            family=IncidentFamily.THIRD_PARTY,
            description="An external search vendor is degrading requests, but symptoms initially resemble app instability.",
            topology_id="regional",
            initial_alert="Global alert: search-backed requests degraded, elevated 502s from vendor edge",
            root_cause="third_party_failure",
            impacted_services=["search_vendor", "api_us", "api_eu"],
            correct_mitigation="traffic_shift",
            required_escalation=True,
            preferred_escalation_target="sre_lead",
            hypothesis_aliases=["third_party_failure", "vendor_outage", "dependency_failure"],
            tool_data={
                "metrics": {
                    "search_vendor": "Dependency latency and 502s rising globally. Internal service saturation normal.",
                    "api_us": "Application CPU normal; upstream vendor requests are timing out.",
                },
                "logs": {
                    "api_us": "Upstream dependency timeout to vendor search endpoint.",
                    "api_eu": "Vendor edge timeout after 1200ms.",
                },
                "deploy_history": {
                    "global_lb": "No deploys or config changes in the last 24 hours.",
                },
                "incident_chat": {
                    "global": "Vendor status page is slow to update. Traffic shift to cached fallback requires incident lead approval.",
                },
                "runbook": {
                    "global_lb": "Runbook: when external dependency fails, escalate and shift eligible traffic to fallback path.",
                },
            },
            runbook_notes=["Traffic shift is allowed after incident lead approval."],
            chat_messages=["PM: users can still browse if cached search results are enabled."],
            report_keywords=["vendor", "traffic_shift", "fallback", "dependency"],
            causal_edges=[
                CausalEdge("third_party_failure", "search_vendor", "external vendor degradation", "metrics:search_vendor"),
                CausalEdge("search_vendor", "api_us", "upstream vendor timeout", "logs:api_us"),
                CausalEdge("search_vendor", "api_eu", "upstream vendor timeout", "logs:api_eu"),
            ],
        ),
    ]


def build_v2_scenarios() -> list[IncidentScenario]:
    """Flagship governance scenario + OOD scenarios for transfer benchmarks."""
    return [
        # ── Flagship: Multi-region failover with governance constraints ──
        IncidentScenario(
            scenario_id="multi_region_failover_gov",
            title="Multi-region failover with cost-budget and data-residency governance",
            family=IncidentFamily.MULTI_REGION_GOVERNANCE,
            description=(
                "Primary DB in us-east-1 is failing over. The technically correct mitigation "
                "(failover to eu-west-1 replica) violates EU data-residency policy unless "
                "a compliance check is performed first. Budget governance also limits which "
                "regions can absorb the traffic."
            ),
            topology_id="multi_region_governed",
            initial_alert="CRITICAL: primary_db us-east-1 unresponsive, replication lag 45s, SLO breach imminent",
            root_cause="primary_db_failure",
            impacted_services=["primary_db", "api_us_east"],
            correct_mitigation="failover_db",
            required_escalation=True,
            preferred_escalation_target="infra_engineer",
            hypothesis_aliases=["primary_db_failure", "db_failure", "replication_lag"],
            tool_data={
                "metrics": {
                    "primary_db": "CPU 100%, disk I/O stalled, all connections rejected since 4 min ago.",
                    "api_us_east": "Error rate 94%, all DB-dependent requests failing.",
                    "replica_db": "Replication lag 45s but replica is healthy. Read-only queries succeeding.",
                },
                "logs": {
                    "primary_db": "FATAL: storage engine crash, WAL corruption detected.",
                    "api_us_east": "Connection refused from primary_db after 12 retry attempts.",
                },
                "deploy_history": {
                    "primary_db": "No deploys in 72 hours. Last maintenance: 14 days ago.",
                },
                "compliance": {
                    "global": "EU data-residency: user PII from eu-west-1 must not be routed to us-east-1 or ap-south-1.",
                    "replica_db": "Replica in eu-west-1 is compliant for EU traffic. Cross-region failover for US traffic requires legal sign-off.",
                },
                "budget": {
                    "global": "Monthly compute budget: $12,400 remaining. Failover to ap-south-1 costs $3,200/day. eu-west-1 costs $2,100/day.",
                },
                "incident_chat": {
                    "global": "Legal: any cross-region data movement must be logged and reviewed. Infra: failover ready but need compliance clearance.",
                },
                "runbook": {
                    "primary_db": "Runbook: on primary failure, check compliance constraints before failover. Escalate to infra + legal.",
                },
            },
            runbook_notes=[
                "Failover is technically straightforward but policy-gated.",
                "Check budget and data-residency before executing.",
            ],
            chat_messages=[
                "CTO: we cannot afford regulatory fines. Check compliance first.",
                "Infra lead: replica is ready, just need the green light.",
            ],
            report_keywords=["failover", "primary_db", "compliance", "governance", "data_residency"],
            governance_constraints=[
                GovernanceConstraint(
                    constraint_type="data_residency",
                    description="EU PII must not leave eu-west-1 region",
                    blocked_mitigations=["traffic_shift_to_us"],
                    required_check="compliance:global",
                ),
                GovernanceConstraint(
                    constraint_type="cost_budget",
                    description="Failover cost must not exceed remaining monthly budget",
                    blocked_mitigations=[],
                    required_check="budget:global",
                ),
            ],
            causal_edges=[
                CausalEdge("primary_db_failure", "primary_db", "storage engine WAL corruption", "primary_db"),
                CausalEdge("primary_db", "api_us_east", "all DB connections rejected", "api_us_east"),
                CausalEdge("api_us_east", "global_lb", "us-east-1 health check failing", "primary_db"),
            ],
            delayed_effects_config=[
                {"delay": 2, "target": "api_eu_west", "delta": -0.15, "desc": "EU traffic surge from US failover"},
                {"delay": 4, "target": "cache_eu", "delta": -0.20, "desc": "Cache pressure from redirected traffic"},
            ],
        ),
        # ── OOD: Certificate expiry (unseen family) ──
        IncidentScenario(
            scenario_id="cert_expiry_ood",
            title="TLS certificate expiry causing cascading auth failures",
            family=IncidentFamily.CERTIFICATE_EXPIRY,
            description="An internal TLS cert expired, causing auth_svc to reject all mTLS connections.",
            topology_id="microservices",
            initial_alert="Auth failure spike: 98% of mTLS handshakes failing on auth_svc",
            root_cause="certificate_expiry",
            impacted_services=["auth_svc", "ingress"],
            correct_mitigation="rotate_cert",
            hypothesis_aliases=["certificate_expiry", "tls_failure", "mtls_rejection"],
            tool_data={
                "metrics": {
                    "auth_svc": "mTLS handshake failures at 98%. CPU and memory normal.",
                    "ingress": "502 rate 87% — all authenticated routes failing.",
                },
                "logs": {
                    "auth_svc": "x509: certificate has expired — not after 2026-04-25T00:00:00Z.",
                },
                "deploy_history": {
                    "auth_svc": "No deploys. Last cert rotation: 365 days ago.",
                },
                "incident_chat": {
                    "global": "Security: cert auto-renewal was disabled after last compliance audit.",
                },
                "runbook": {
                    "auth_svc": "Runbook: on cert expiry, rotate certificate and restart affected pods.",
                },
            },
            runbook_notes=["Certificate rotation requires security team approval."],
            chat_messages=["Users: login is completely broken across all apps."],
            report_keywords=["certificate", "tls", "auth_svc", "rotate"],
            causal_edges=[
                CausalEdge("certificate_expiry", "auth_svc", "expired x509 cert", "logs:auth_svc"),
                CausalEdge("auth_svc", "ingress", "all authenticated routes fail", "metrics:ingress"),
            ],
        ),
        # ── OOD: Capacity exhaustion (unseen family) ──
        IncidentScenario(
            scenario_id="capacity_exhaust_ood",
            title="Inventory DB capacity exhaustion under traffic surge",
            family=IncidentFamily.CAPACITY_EXHAUSTION,
            description="Black Friday traffic surge exhausted inventory_db connection pool and disk.",
            topology_id="microservices",
            initial_alert="Capacity alert: inventory_db disk 99%, connections maxed, order_svc latency 12s",
            root_cause="capacity_exhaustion",
            impacted_services=["inventory_db", "order_svc"],
            correct_mitigation="scale_up",
            required_escalation=True,
            preferred_escalation_target="infra_engineer",
            hypothesis_aliases=["capacity_exhaustion", "disk_full", "connection_pool_exhausted"],
            tool_data={
                "metrics": {
                    "inventory_db": "Disk 99.2%, connections 500/500, query latency 8.4s.",
                    "order_svc": "Timeout rate 76%, all inventory lookups failing.",
                },
                "logs": {
                    "inventory_db": "ERROR: no space left on device. Connection pool exhausted.",
                },
                "deploy_history": {
                    "order_svc": "No deploys. Traffic volume 4.2x normal (Black Friday surge).",
                },
                "incident_chat": {
                    "global": "Infra: we can scale up but need budget approval for additional nodes.",
                },
                "runbook": {
                    "inventory_db": "Runbook: on capacity exhaustion, escalate for emergency scale-up.",
                },
            },
            runbook_notes=["Scale-up requires infra approval and budget sign-off."],
            chat_messages=["Product: order page is completely broken. Revenue impact critical."],
            report_keywords=["capacity", "scale_up", "inventory_db", "disk"],
            causal_edges=[
                CausalEdge("capacity_exhaustion", "inventory_db", "disk and connection pool exhausted", "metrics:inventory_db"),
                CausalEdge("inventory_db", "order_svc", "inventory lookups timing out", "metrics:order_svc"),
            ],
        ),
    ]


class IncidentScenarioFactory:
    """Deterministic sampler over seeded and procedurally varied scenarios."""

    def __init__(self) -> None:
        self._base_scenarios = build_scenarios()
        self._v2_scenarios = build_v2_scenarios()
        self._train_variants = self._build_variants(holdout=False)
        self._test_variants = self._build_variants(holdout=True)
        self._ood_scenarios = [s for s in self._v2_scenarios if s.family in (
            IncidentFamily.CERTIFICATE_EXPIRY, IncidentFamily.CAPACITY_EXHAUSTION,
        )]
        self._governance_scenarios = [s for s in self._v2_scenarios if s.family == IncidentFamily.MULTI_REGION_GOVERNANCE]
        self._scenarios = (
            self._base_scenarios + self._v2_scenarios
            + self._train_variants + self._test_variants
        )
        self._by_id = {scenario.scenario_id: scenario for scenario in self._scenarios}

    def list_scenarios(self, split: str = "base") -> list[IncidentScenario]:
        return list(self._pool(split))

    def get(self, scenario_id: str) -> IncidentScenario:
        return self._by_id[scenario_id]

    def sample(self, seed: int | None = None, split: str = "base") -> IncidentScenario:
        rng = random.Random(seed)
        return rng.choice(self._pool(split))

    def _pool(self, split: str) -> list[IncidentScenario]:
        if split == "base":
            return self._base_scenarios
        if split == "train":
            return self._base_scenarios + self._train_variants
        if split == "test":
            return self._test_variants
        if split == "ood":
            return self._ood_scenarios
        if split == "stress":
            return self._base_scenarios  # same scenarios but env uses reduced budget
        if split == "governance":
            return self._governance_scenarios
        if split == "all":
            return self._scenarios
        raise ValueError(f"Unknown split {split!r}. Expected one of: base, train, test, ood, stress, governance, all.")

    def _build_variants(self, holdout: bool) -> list[IncidentScenario]:
        variants: list[IncidentScenario] = []
        variant_indices = [3] if holdout else [0, 1, 2]
        for base in self._base_scenarios:
            for variant_idx in variant_indices:
                variants.append(self._make_variant(base, variant_idx, holdout))
        return variants

    def _make_variant(
        self,
        base: IncidentScenario,
        variant_idx: int,
        holdout: bool,
    ) -> IncidentScenario:
        alias_map = self._service_aliases(base.family, variant_idx)
        variant = self._apply_aliases(base, alias_map)

        noise_clues = [
            "On-call notes mention a noisy but non-blocking cache warning.",
            "A parallel canary in another service appears healthy and is a likely distractor.",
            "A stale dashboard panel still shows yesterday's incident banner.",
            "Automated triage flagged an unrelated low-severity anomaly in another tier.",
        ]
        chat_noise = [
            "Ops analyst: one synthetic check failed but recovered immediately.",
            "Support: some users report slowness while others report normal performance.",
            "Release manager: no broad rollback planned unless impact spreads.",
            "Incident bot: confidence score dropped due to conflicting low-signal alerts.",
        ]

        suffix = "ho" if holdout else "tr"
        variant.scenario_id = f"{base.scenario_id}_{suffix}{variant_idx + 1}"
        variant.title = f"{variant.title} [variant {variant_idx + 1}]"
        variant.description = f"{variant.description} {noise_clues[variant_idx % len(noise_clues)]}"
        variant.initial_alert = f"{variant.initial_alert} | signal-mix:{variant_idx + 1}"
        variant.chat_messages = list(variant.chat_messages) + [chat_noise[variant_idx % len(chat_noise)]]
        variant.runbook_notes = list(variant.runbook_notes) + [noise_clues[(variant_idx + 1) % len(noise_clues)]]

        # Keep root cause and mitigation semantics unchanged while broadening hypotheses.
        variant.hypothesis_aliases = sorted(
            set(variant.hypothesis_aliases + [variant.root_cause, f"{variant.root_cause}_path_{variant_idx + 1}"])
        )
        return variant

    @staticmethod
    def _service_aliases(family: IncidentFamily, variant_idx: int) -> dict[str, str]:
        pools: dict[IncidentFamily, list[dict[str, str]]] = {
            IncidentFamily.BAD_DEPLOY: [
                {"checkout_service": "checkout_api", "api_gateway": "edge_gateway"},
                {"checkout_service": "order_checkout", "api_gateway": "frontdoor"},
                {"checkout_service": "payments_edge", "api_gateway": "request_router"},
                {"checkout_service": "checkout_core", "api_gateway": "ingress_gateway"},
            ],
            IncidentFamily.DATABASE_SATURATION: [
                {"payments_db": "payments_primary", "checkout_service": "orders_api", "web_frontend": "web_edge"},
                {"payments_db": "ledger_db", "checkout_service": "checkout_api", "web_frontend": "storefront"},
                {"payments_db": "txn_store", "checkout_service": "purchase_api", "web_frontend": "public_web"},
                {"payments_db": "billing_db", "checkout_service": "cart_api", "web_frontend": "shop_front"},
            ],
            IncidentFamily.FEATURE_FLAG: [
                {"api_eu": "payments_api_eu"},
                {"api_eu": "checkout_eu"},
                {"api_eu": "orders_eu"},
                {"api_eu": "api_west_eu"},
            ],
            IncidentFamily.THIRD_PARTY: [
                {"search_vendor": "search_partner", "api_us": "api_america", "api_eu": "api_europe"},
                {"search_vendor": "external_search", "api_us": "api_use1", "api_eu": "api_euw1"},
                {"search_vendor": "vendor_edge", "api_us": "api_north_america", "api_eu": "api_west_europe"},
                {"search_vendor": "dependency_search", "api_us": "api_us_prod", "api_eu": "api_eu_prod"},
            ],
        }
        family_pool = pools[family]
        return family_pool[variant_idx % len(family_pool)]

    def _apply_aliases(self, scenario: IncidentScenario, aliases: dict[str, str]) -> IncidentScenario:
        variant = copy.deepcopy(scenario)

        def replace_text(text: str) -> str:
            updated = text
            for old, new in aliases.items():
                updated = updated.replace(old, new)
            return updated

        variant.description = replace_text(variant.description)
        variant.initial_alert = replace_text(variant.initial_alert)
        # Keep canonical structured identifiers aligned with static topology nodes.
        variant.root_cause = scenario.root_cause
        variant.correct_mitigation = scenario.correct_mitigation
        variant.deploy_service = scenario.deploy_service
        variant.impacted_services = list(scenario.impacted_services)
        variant.hypothesis_aliases = list(scenario.hypothesis_aliases)
        variant.runbook_notes = [replace_text(note) for note in variant.runbook_notes]
        variant.chat_messages = [replace_text(msg) for msg in variant.chat_messages]
        variant.report_keywords = list(scenario.report_keywords)

        remapped_tool_data: dict[str, dict[str, str]] = {}
        for tool_name, entries in variant.tool_data.items():
            remapped_entries: dict[str, str] = {}
            for target, content in entries.items():
                remapped_entries[target] = replace_text(content)
            remapped_tool_data[tool_name] = remapped_entries
        variant.tool_data = remapped_tool_data
        return variant
