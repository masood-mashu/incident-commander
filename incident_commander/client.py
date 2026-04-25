"""Minimal sync client for interacting with a running Incident Commander server."""

from __future__ import annotations

import json
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Any
from urllib import request

from incident_commander.models import IncidentAction


@dataclass
class IncidentCommanderEnv(AbstractContextManager):
    """Simple HTTP client with reset, step, and state methods."""

    base_url: str

    def __enter__(self) -> "IncidentCommanderEnv":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def sync(self) -> "IncidentCommanderEnv":
        return self

    def reset(self, scenario_id: str | None = None, seed: int | None = None) -> dict[str, Any]:
        payload = {}
        if scenario_id is not None:
            payload["scenario_id"] = scenario_id
        if seed is not None:
            payload["seed"] = seed
        return self._post("/reset", payload)

    def step(self, action: IncidentAction | dict[str, Any]) -> dict[str, Any]:
        payload = action if isinstance(action, dict) else action.__dict__
        return self._post("/step", payload)

    def state(self) -> dict[str, Any]:
        return self._get("/state")

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        data = json.dumps(payload).encode("utf-8")
        req = request.Request(
            self.base_url.rstrip("/") + path,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with request.urlopen(req) as response:
            return json.loads(response.read().decode("utf-8"))

    def _get(self, path: str) -> dict[str, Any]:
        req = request.Request(self.base_url.rstrip("/") + path, method="GET")
        with request.urlopen(req) as response:
            return json.loads(response.read().decode("utf-8"))
