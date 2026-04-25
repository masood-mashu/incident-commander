"""Hardcoded service graphs for the MVP environment.

v2: adds microservices (OOD), multi-region-governed (flagship),
and stress (reduced) topologies.
"""

from __future__ import annotations

from incident_commander.models import ServiceGraph, ServiceNode


def build_simple_topology() -> ServiceGraph:
    return ServiceGraph(
        topology_id="simple",
        entrypoints=["api_gateway"],
        nodes={
            "api_gateway": ServiceNode("api_gateway", "edge", ["checkout_service"]),
            "checkout_service": ServiceNode("checkout_service", "application", ["payments_db"]),
            "payments_db": ServiceNode("payments_db", "data", []),
        },
    )


def build_cascade_topology() -> ServiceGraph:
    return ServiceGraph(
        topology_id="cascade",
        entrypoints=["web_frontend"],
        nodes={
            "web_frontend": ServiceNode(
                "web_frontend",
                "edge",
                ["checkout_service", "catalog_service"],
            ),
            "checkout_service": ServiceNode(
                "checkout_service",
                "application",
                ["payments_db", "feature_flag_service"],
            ),
            "catalog_service": ServiceNode("catalog_service", "application", ["feature_flag_service"]),
            "feature_flag_service": ServiceNode("feature_flag_service", "platform", []),
            "payments_db": ServiceNode("payments_db", "data", []),
        },
    )


def build_regional_topology() -> ServiceGraph:
    return ServiceGraph(
        topology_id="regional",
        entrypoints=["global_lb"],
        nodes={
            "global_lb": ServiceNode("global_lb", "edge", ["api_us", "api_eu"], ["global"]),
            "api_us": ServiceNode("api_us", "application", ["search_vendor"], ["us-east-1"]),
            "api_eu": ServiceNode("api_eu", "application", ["search_vendor"], ["eu-west-1"]),
            "search_vendor": ServiceNode("search_vendor", "external", [], ["global"]),
        },
    )


# ── v2 topologies ─────────────────────────────────────────────────────────────


def build_microservices_topology() -> ServiceGraph:
    """OOD topology: 8-node microservices mesh — never seen during training."""
    return ServiceGraph(
        topology_id="microservices",
        entrypoints=["ingress"],
        nodes={
            "ingress": ServiceNode("ingress", "edge", ["auth_svc", "order_svc", "search_svc"]),
            "auth_svc": ServiceNode("auth_svc", "application", ["user_db", "cache_layer"]),
            "order_svc": ServiceNode("order_svc", "application", ["payment_gw", "inventory_db"]),
            "search_svc": ServiceNode("search_svc", "application", ["cache_layer"]),
            "payment_gw": ServiceNode("payment_gw", "application", ["ledger_db"]),
            "cache_layer": ServiceNode("cache_layer", "platform", []),
            "user_db": ServiceNode("user_db", "data", []),
            "inventory_db": ServiceNode("inventory_db", "data", []),
            "ledger_db": ServiceNode("ledger_db", "data", []),
        },
    )


def build_multi_region_governed_topology() -> ServiceGraph:
    """Flagship scenario topology: multi-region with cost / compliance constraints."""
    return ServiceGraph(
        topology_id="multi_region_governed",
        entrypoints=["global_lb"],
        nodes={
            "global_lb": ServiceNode(
                "global_lb", "edge",
                ["api_us_east", "api_eu_west", "api_ap_south"],
                ["global"],
            ),
            "api_us_east": ServiceNode(
                "api_us_east", "application",
                ["primary_db", "cache_us"],
                ["us-east-1"],
                cost_per_hour=120.0,
                data_residency="us",
            ),
            "api_eu_west": ServiceNode(
                "api_eu_west", "application",
                ["replica_db", "cache_eu"],
                ["eu-west-1"],
                cost_per_hour=95.0,
                data_residency="eu",
            ),
            "api_ap_south": ServiceNode(
                "api_ap_south", "application",
                ["replica_db"],
                ["ap-south-1"],
                cost_per_hour=80.0,
                data_residency="ap",
            ),
            "primary_db": ServiceNode(
                "primary_db", "data", [],
                ["us-east-1"],
                criticality="critical",
                cost_per_hour=200.0,
                data_residency="us",
            ),
            "replica_db": ServiceNode(
                "replica_db", "data",
                ["primary_db"],
                ["eu-west-1"],
                cost_per_hour=150.0,
                data_residency="eu",
            ),
            "cache_us": ServiceNode("cache_us", "platform", [], ["us-east-1"]),
            "cache_eu": ServiceNode("cache_eu", "platform", [], ["eu-west-1"]),
        },
    )


def get_topologies() -> dict[str, ServiceGraph]:
    return {
        "simple": build_simple_topology(),
        "cascade": build_cascade_topology(),
        "regional": build_regional_topology(),
        "microservices": build_microservices_topology(),
        "multi_region_governed": build_multi_region_governed_topology(),
    }
