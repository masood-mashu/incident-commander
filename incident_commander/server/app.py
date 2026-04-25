"""FastAPI app wrapper for Incident Commander."""

from __future__ import annotations

import os

from incident_commander.models import ActionType, IncidentAction, to_plain_data
from incident_commander.server.incident_environment import IncidentCommanderEnvironment

try:
    from fastapi import FastAPI
    from fastapi.responses import JSONResponse
except ImportError:  # pragma: no cover
    FastAPI = None
    JSONResponse = None

try:
    from openenv.core.env_server import create_web_interface_app
except ImportError:  # pragma: no cover
    create_web_interface_app = None


def create_app() -> "FastAPI":
    if FastAPI is None:
        raise RuntimeError(
            "FastAPI is not installed. Install project dependencies with `pip install -e .` first."
        )

    env = IncidentCommanderEnvironment()
    if os.getenv("ENABLE_WEB_INTERFACE", "").lower() == "true" and create_web_interface_app:
        return create_web_interface_app(env, IncidentAction, dict)

    app = FastAPI(title="Incident Commander", version="0.1.0")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/reset")
    def reset(payload: dict | None = None) -> JSONResponse:
        payload = payload or {}
        observation = env.reset(payload.get("scenario_id"), payload.get("seed"))
        return JSONResponse({"observation": to_plain_data(observation)})

    @app.post("/step")
    def step(payload: dict) -> JSONResponse:
        action = IncidentAction(
            action_type=ActionType(payload["action_type"]),
            tool_name=payload.get("tool_name"),
            target=payload.get("target"),
            cause=payload.get("cause"),
            mitigation=payload.get("mitigation"),
            message=payload.get("message"),
            escalate_to=payload.get("escalate_to"),
            report=payload.get("report"),
            severity=payload.get("severity"),
        )
        result = env.step(action)
        return JSONResponse(to_plain_data(result))

    @app.get("/state")
    def state() -> JSONResponse:
        return JSONResponse(to_plain_data(env.state()))

    return app


app = create_app() if FastAPI is not None else None
