import pytest
import threading
import time
import requests
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "v-ui"))

from server import run_server

PORT = 8085
BASE_URL = f"http://localhost:{PORT}"

@pytest.fixture(scope="module", autouse=True)
def server_setup():
    thread = threading.Thread(target=run_server, args=(PORT,), daemon=True)
    thread.start()
    time.sleep(2)
    yield
    # Thread terminates when pytest exits

def test_get_root():
    res = requests.get(f"{BASE_URL}/")
    assert res.status_code == 200
    assert "Portfolio X-Ray" in res.text

def test_get_presets():
    res = requests.get(f"{BASE_URL}/api/presets")
    assert res.status_code == 200
    data = res.json()
    assert "accidental_tech_overweight" in data

def test_post_analyze():
    payload = {
        "portfolio": {
            "NVDA": 10,
            "QQQ": 5
        },
        "user_profile": {
            "risk_tolerance": "Moderate"
        }
    }
    res = requests.post(f"{BASE_URL}/api/analyze", json=payload)
    assert res.status_code == 200
    data = res.json()
    
    assert "metrics" in data
    assert "institutional_lookthrough_exposure" in data
    assert "historical_stress" in data
    assert "correlation_and_risk" in data
    assert "evidences" in data
    
    # Check that fallback/deterministic metrics are filled
    metrics = data["metrics"]
    assert "annual_volatility" in metrics
    assert "hhi" in metrics

def test_post_what_if():
    payload = {
        "portfolio": {"NVDA": 10},
        "target_allocations": {"NVDA": 50, "CASH": 50}
    }
    res = requests.post(f"{BASE_URL}/api/what-if", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert "friction" in data
    assert "circuit_breakers" in data

def test_post_sign_off():
    payload = {"decision": "approved", "target": {"NVDA": 20}}
    res = requests.post(f"{BASE_URL}/api/sign-off", json=payload)
    assert res.status_code == 200
    assert res.json().get("status") == "recorded"

def test_get_audit_log():
    res = requests.get(f"{BASE_URL}/api/audit-log")
    assert res.status_code == 200
    data = res.json()
    assert "logs" in data
    # Assert at least one sign_off is present because of previous test
    assert any(log.get("event") == "human_sign_off" for log in data["logs"])

def test_post_headline():
    payload = {"portfolio": {"AAPL": 10}, "headline": "Apple announces record iPhone sales."}
    res = requests.post(f"{BASE_URL}/api/headline", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert "impact_score" in data or "net_estimated_impact_pct" in data

def test_post_chat():
    payload = {"portfolio": {"AAPL": 10}, "query": "Is my portfolio risky?"}
    res = requests.post(f"{BASE_URL}/api/chat", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert "response" in data
    assert "safety_check" in data
    assert data["safety_check"]["passed"] == True
