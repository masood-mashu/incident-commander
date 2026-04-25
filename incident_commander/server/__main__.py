"""CLI entrypoint for running the development server."""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Incident Commander server.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    try:
        import uvicorn
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("uvicorn is not installed. Run `pip install -e .`.") from exc

    uvicorn.run("incident_commander.server.app:app", host=args.host, port=args.port, reload=False)


if __name__ == "__main__":
    main()
