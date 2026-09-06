"""Module-level ASGI app, for deployment.

A hosting platform runs the server with something like

    uvicorn skyguard.api.asgi:app --host 0.0.0.0 --port $PORT

which needs an importable `app` object rather than the `build_app()` factory the
CLI uses. The stream speed comes from `SKYGUARD_SPEED` (default 12).
"""

from __future__ import annotations

from .server import build_app

app = build_app()
