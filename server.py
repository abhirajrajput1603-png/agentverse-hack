"""Portfolio X-Ray: Unified Web Server Entrypoint.

Delegates to `v-ui/server.py` so that both `python3 server.py [port]`
and `python3 v-ui/server.py [port]` launch the production consumer + institutional
Portfolio X-Ray web dashboard.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
V_UI_DIR = ROOT_DIR / "v-ui"

if str(V_UI_DIR) not in sys.path:
    sys.path.insert(0, str(V_UI_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from server import (  # type: ignore  # noqa: E402
    PortfolioXRayHandler,
    DualStackServer,
    run_server,
    parse_portfolio_payload,
    parse_holdings_from_text,
    HTML_DASHBOARD,
)

__all__ = [
    "PortfolioXRayHandler",
    "DualStackServer",
    "run_server",
    "parse_portfolio_payload",
    "parse_holdings_from_text",
    "HTML_DASHBOARD",
]

if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8080
    run_server(port)
