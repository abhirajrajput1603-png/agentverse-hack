"""End-to-End Integration Tests for Unified Portfolio X-Ray Web Server (`v-ui/server.py` & `server.py`).

Spins up `ThreadingHTTPServer` on an ephemeral port and tests:
- `GET /` (HTML dashboard with all 5 tabs & institutional sections)
- `GET /api/presets` (Dhillon's 3 presets + 4 AgentVerse Benchmark Cases)
- `POST /api/parse-portfolio` (Natural language & OCR text parser)
- `POST /api/analyze` (Unified consumer + institutional analysis, full 100% lookthrough conservation, covariance estimators, historical crisis replays, Euler risk contribution, benchmark comparison, circuit breakers, and `EV-*` registry)
- `POST /api/what-if` (Target allocation + execution friction in bps + circuit-breaker resolution)
- `POST /api/sign-off` (Human-in-the-loop sign-off logged to `logs/audit.jsonl`)
- `POST /api/headline` (Headline shock impact auditor)
- `POST /api/chat` (Autonomous tool-calling risk analyst + OutputSafetyFilter + scope guardrail)
- `GET /api/audit-log` (Live append-only audit log inspection)
"""

from __future__ import annotations

import json
import threading
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path
import sys
import pytest

ROOT_DIR = Path(__file__).resolve().parent.parent
V_UI_DIR = ROOT_DIR / "v-ui"
if str(V_UI_DIR) not in sys.path:
    sys.path.insert(0, str(V_UI_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from server import PortfolioXRayHandler  # type: ignore  # noqa: E402


@pytest.fixture(scope="module")
def live_server_url():
    server = ThreadingHTTPServer(("127.0.0.1", 0), PortfolioXRayHandler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{port}"
    server.shutdown()
    server.server_close()


def _http_get(base_url: str, path: str):
    with urllib.request.urlopen(base_url + path, timeout=10) as resp:
        body = resp.read().decode("utf-8")
        ctype = resp.headers.get("Content-Type", "")
        if "application/json" in ctype:
            return resp.status, json.loads(body)
        return resp.status, body


def _http_post(base_url: str, path: str, payload: dict):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        base_url + path,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=15) as resp:
        body = resp.read().decode("utf-8")
        return resp.status, json.loads(body)


def test_get_dashboard_html(live_server_url):
    status, html = _http_get(live_server_url, "/")
    assert status == 200
    assert "Portfolio X-Ray" in html
    assert "tab-overview" in html
    assert "tab-hidden-risks" in html
    assert "subtab-correlation" in html
    assert "tab-what-if" in html
    assert "tab-simulate" in html
    assert "tab-chat" in html
    assert "modal-evidence" in html


def test_get_presets_includes_consumer_and_benchmark_cases(live_server_url):
    status, presets = _http_get(live_server_url, "/api/presets")
    assert status == 200
    expected_keys = {
        "accidental_tech_overweight",
        "balanced_core",
        "all_in_growth",
        "case_01_diversification_illusion",
        "case_02_ai_capex_energy_theme",
        "case_03_rate_shock_duration_trap",
        "case_04_passive_diversified_benchmark",
    }
    assert expected_keys.issubset(set(presets.keys()))


def test_post_parse_portfolio_natural_language(live_server_url):
    status, data = _http_post(
        live_server_url,
        "/api/parse-portfolio",
        {"text": "I have 50 NVDA @ $130, 10 Apple shares, and £2,000 in an S&P 500 ETF"},
    )
    assert status == 200
    tickers = {h["ticker"] for h in data["holdings"]}
    assert {"NVDA", "AAPL", "VOO"}.issubset(tickers)


def test_post_analyze_unified_institutional_and_consumer_telemetry(live_server_url):
    status, data = _http_post(
        live_server_url,
        "/api/analyze",
        {
            "portfolio": {"NVDA": 50, "QQQ": 25, "VOO": 15, "SMH": 26},
            "cov_method": "ledoit_wolf",
            "lookthrough_mode": "full",
            "user_profile": {
                "risk_tolerance": "Moderate",
                "horizon_years": "5-10 years",
                "loss_capacity": "10-20% max",
            },
        },
    )
    assert status == 200
    assert data["portfolio_value"] == pytest.approx(32500.0)
    assert data["covariance_method"] == "ledoit_wolf"

    # Full Look-Through weight conservation invariant: sum(w) == 1.0
    full_lt_sum = sum(item["lookthrough_weight"] for item in data["full_lookthrough_exposure"])
    assert abs(full_lt_sum - 1.0) < 1e-5

    # Risk & Tail Risk metrics present
    rm = data["risk_metrics"]
    assert rm["annualized_volatility"] > 0
    assert rm["var_95_1m"] > 0
    assert rm["cvar_95_1m"] >= rm["var_95_1m"]
    assert "covariance_comparison" in rm
    assert rm["covariance_comparison"]["ledoit_wolf_annualized_vol"] > 0

    # Euler Risk Contribution & Correlation Matrix
    crc = data["correlation_and_risk_contribution"]
    assert crc["ccr_sum_check_passed"] is True
    assert len(crc["contributions"]) == 4
    assert "NVDA" in crc["corr_matrix_3y"]

    # Historical Crisis Window Replays
    replays = data["historical_crisis_replays"]
    assert len(replays) == 4
    scenario_ids = {r["scenario_id"] for r in replays}
    assert "MAR_2020_COVID_CRASH" in scenario_ids
    assert "YEAR_2022_FED_RATE_SHOCK" in scenario_ids

    # Benchmark Comparison vs SPY and 60/40
    bm = data["benchmark_comparison"]
    assert bm["portfolio_beta_vs_spy"] > 1.0
    assert "spy_benchmark" in bm
    assert "balanced_60_40_benchmark" in bm

    # Circuit Breakers & Evidence Registry
    cb = data["circuit_breakers"]
    assert cb["is_safe"] is False
    assert len(cb["breaches"]) >= 1
    assert "EV-CONC-TOP1-WEIGHT" in data["evidences"]
    assert "EV-RISK-ANN-VOL" in data["evidences"]


def test_post_analyze_graceful_fallback_for_custom_private_tickers(live_server_url):
    status, data = _http_post(
        live_server_url,
        "/api/analyze",
        {"portfolio": {"SPACEX": 20, "OPENAI": 15, "NVDA": 10}},
    )
    assert status == 200
    assert data["portfolio_value"] > 0
    assert "SPACEX" in data["risk_metrics"]["unpriced_holdings"]


def test_post_what_if_and_human_sign_off(live_server_url):
    status, wi = _http_post(
        live_server_url,
        "/api/what-if",
        {
            "portfolio": {"NVDA": 50, "QQQ": 25, "VOO": 15, "SMH": 26},
            "target_allocations": {"TLT": 25},
        },
    )
    assert status == 200
    assert "execution_friction" in wi
    assert wi["execution_friction"]["total_friction_cost"] > 0
    assert wi["execution_friction"]["effective_friction_bps"] > 0
    assert "circuit_breaker_comparison" in wi
    assert "EV-WHATIF-FRICTION" in wi["evidences"]

    # Record Human Sign-Off
    so_status, so_data = _http_post(
        live_server_url,
        "/api/sign-off",
        {
            "decision": "APPROVED",
            "scenario_summary": "Reallocate 25% to TLT",
            "persona_name": "Balanced / Moderate",
        },
    )
    assert so_status == 200
    assert so_data["status"] == "recorded"
    assert so_data["decision"] == "APPROVED"


def test_post_headline_and_chat_and_audit_log(live_server_url):
    # Headline Shock Auditor
    hl_status, hl = _http_post(
        live_server_url,
        "/api/headline",
        {
            "portfolio": {"NVDA": 50, "QQQ": 25},
            "headline": "Federal Reserve hikes interest rates by 50 basis points to fight inflation",
        },
    )
    assert hl_status == 200
    assert hl["net_estimated_impact_pct"] != 0

    # Chat with Autonomous Tool Trace & Evidence Chips
    ch_status, ch = _http_post(
        live_server_url,
        "/api/chat",
        {
            "portfolio": {"NVDA": 50, "QQQ": 25, "VOO": 15, "SMH": 26},
            "query": "Why is my portfolio so concentrated in Nvidia?",
        },
    )
    assert ch_status == 200
    assert ch["safety_check"]["passed"] is True
    assert len(ch["cited_evidences"]) >= 3
    assert len(ch["tool_calls"]) >= 3

    # Audit Log Inspection
    al_status, al = _http_get(live_server_url, "/api/audit-log")
    assert al_status == 200
    assert al["count"] > 0
    assert isinstance(al["entries"], list)
