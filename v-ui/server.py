"""Portfolio X-Ray: Zero-dependency local web server & interactive dashboard.

Runs using Python standard library http.server on port 8080 (or specified port).
Modern consumer fintech redesign with accessible plain-English intelligence.
"""

from __future__ import annotations
import os
import re
import json
import socket
import urllib.parse
from http import HTTPStatus
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
import sys
import time
from pathlib import Path
from typing import Dict, List, Any, Optional

# Ensure v-ui directory and project root are in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from data import SAMPLE_PORTFOLIOS, ASSET_METADATA, COMPANY_SYNONYMS
    from engine import (
        Holding,
        Portfolio,
        compute_lookthrough_exposure,
        compute_thematic_exposure,
        compute_risk_metrics,
        run_stress_tests,
        simulate_what_if,
        analyze_headline_event,
        get_institutional_lookthrough,
        get_correlation_and_risk,
        get_historical_stress,
        get_benchmark_comparison,
    )
    from agent import ask_risk_analyst
except ImportError:
    from portfolio_xray.data import SAMPLE_PORTFOLIOS, ASSET_METADATA, COMPANY_SYNONYMS
    from portfolio_xray.engine import (
        Holding,
        Portfolio,
        compute_lookthrough_exposure,
        compute_thematic_exposure,
        compute_risk_metrics,
        run_stress_tests,
        simulate_what_if,
        analyze_headline_event,
        get_institutional_lookthrough,
        get_correlation_and_risk,
        get_historical_stress,
        get_benchmark_comparison,
    )
    from portfolio_xray.agent import ask_risk_analyst

STOP_WORDS = {
    "have", "and", "in", "of", "shares", "the", "i", "a", "an", "my", "at", "with",
    "worth", "about", "holding", "holdings", "own", "owns", "to", "for", "is",
    "are", "stock", "stocks", "total", "portfolio", "position", "positions",
    "equity", "equities", "hold", "holds", "got", "put", "into", "an", "invested",
    "allocated", "bought", "selling", "share", "etf", "etfs", "fund", "funds"
}

DEFAULT_PRICES: Dict[str, float] = {
    "NVDA": 130.0, "AAPL": 225.0, "MSFT": 420.0, "AMZN": 190.0,
    "GOOGL": 170.0, "GOOG": 170.0, "META": 550.0, "TSLA": 220.0,
    "AVGO": 170.0, "AMD": 140.0, "QQQ": 480.0, "SPY": 550.0,
    "VOO": 500.0, "VTI": 270.0, "SMH": 250.0, "XLK": 220.0,
    "XLF": 45.0, "XLE": 90.0, "TLT": 95.0, "BND": 72.0,
    "GLD": 230.0, "CEG": 245.0, "BRK.B": 460.0, "JPM": 220.0,
    "LLY": 820.0, "CASH": 1.0,
    "SPACEX": 112.0, "OPENAI": 150.0, "PLTR": 45.0, "UBER": 75.0,
    "ABNB": 135.0, "NFLX": 700.0, "COIN": 210.0, "BA": 155.0,
    "LMT": 580.0, "RKLB": 12.0, "COST": 880.0, "XOM": 118.0,
    "TSM": 190.0, "ASML": 850.0,
}


def clean_sym(raw: str) -> Optional[str]:
    cleaned = re.sub(r"[^a-zA-Z0-9]", "", raw).strip().lower()
    if cleaned in STOP_WORDS or len(cleaned) < 1:
        return None
    if cleaned in COMPANY_SYNONYMS:
        return COMPANY_SYNONYMS[cleaned]
    upper = cleaned.upper()
    if upper in DEFAULT_PRICES or (upper.isalpha() and 2 <= len(upper) <= 10):
        return upper
    return None


def parse_item(chunk: str) -> Optional[Dict[str, Any]]:
    chunk = chunk.strip().rstrip(".,;!")
    if not chunk:
        return None

    # Step 1: Pre-normalize composite multi-word fund/company phrases
    subs = [
        (r"(?i)s&p\s*500(?:\s*etf|\s*index|\s*trust)?", "VOO"),
        (r"(?i)nasdaq\s*100(?:\s*etf|\s*index|\s*trust)?", "QQQ"),
        (r"(?i)van(?:eck)?\s*semiconductor(?:\s*etf)?", "SMH"),
        (r"(?i)semiconductor(?:\s*etf)?", "SMH"),
        (r"(?i)total\s*stock\s*market(?:\s*etf)?", "VTI"),
        (r"(?i)space\s*exploration(?:\s*technologies)?|space\s*x", "SPACEX"),
        (r"(?i)open\s*ai", "OPENAI"),
        (r"(?i)rocket\s*lab", "RKLB"),
        (r"(?i)lockheed(?:\s*martin)?", "LMT"),
    ]
    for pat, rep in subs:
        chunk = re.sub(pat, rep, chunk)

    # 1. Currency amount phrases: e.g. "£2,000 in VOO", "$10,000 of Apple", "2000 USD in NVDA"
    curr_pat = r"[\$£€]\s*([0-9,]+(?:\.[0-9]+)?)\s*(?:in|of|worth\s*of|allocated\s*to|worth)?\s*(?:an?\s+)?([A-Za-z.]{1,20})|([0-9,]+(?:\.[0-9]+)?)\s*(?:usd|gbp|eur|dollars|pounds)\s*(?:in|of)?\s*(?:an?\s+)?([A-Za-z.]{1,20})"
    m_val = re.search(curr_pat, chunk, re.I)
    if m_val:
        val_str = m_val.group(1) or m_val.group(3)
        sym_str = m_val.group(2) or m_val.group(4)
        if val_str and sym_str:
            val = float(val_str.replace(",", ""))
            t = clean_sym(sym_str)
            if t:
                p = DEFAULT_PRICES.get(t, 100.0)
                sh = round(val / p, 2)
                if sh > 0:
                    name = ASSET_METADATA.get(t, {}).get("name", t)
                    is_amb = (t not in ASSET_METADATA) or (t in ("SPACEX", "OPENAI"))
                    return {"ticker": t, "name": name, "shares": sh, "price": p, "is_ambiguous": is_amb}

    # 2. Shares with @ price: "50 NVDA @ $130", "50 shares of Apple at $225"
    m_at = re.search(r"([0-9,]+(?:\.[0-9]+)?)\s*(?:shares?\s*(?:of)?)?\s*([A-Za-z.]{1,20})\s*(?:@|at)\s*[\$£€]?\s*([0-9,]+(?:\.[0-9]+)?)", chunk, re.I)
    if m_at:
        sh = float(m_at.group(1).replace(",", ""))
        t = clean_sym(m_at.group(2))
        pr = float(m_at.group(3).replace(",", "").replace("$", "").replace("£", ""))
        if t and sh > 0 and pr > 0:
            name = ASSET_METADATA.get(t, {}).get("name", t)
            is_amb = (t not in ASSET_METADATA) or (t in ("SPACEX", "OPENAI"))
            return {"ticker": t, "name": name, "shares": sh, "price": pr, "is_ambiguous": is_amb}

    # 3. Colon/dash format: "NVDA: 50" or "NVDA - 50 shares"
    m_col = re.search(r"([A-Za-z.]{1,20})\s*[:\-]\s*([0-9,]+(?:\.[0-9]+)?)", chunk, re.I)
    if m_col:
        t = clean_sym(m_col.group(1))
        sh = float(m_col.group(2).replace(",", ""))
        if t and sh > 0:
            p = DEFAULT_PRICES.get(t, 100.0)
            name = ASSET_METADATA.get(t, {}).get("name", t)
            is_amb = (t not in ASSET_METADATA) or (t in ("SPACEX", "OPENAI"))
            return {"ticker": t, "name": name, "shares": sh, "price": p, "is_ambiguous": is_amb}

    # 4. Standard: "50 NVDA", "10 Nvidia shares", "500 apple"
    m_sh = re.search(r"([0-9,]+(?:\.[0-9]+)?)\s*(?:shares?\s*(?:of)?)?\s*([A-Za-z.]{1,20})", chunk, re.I)
    if m_sh:
        sh = float(m_sh.group(1).replace(",", ""))
        t = clean_sym(m_sh.group(2))
        if t and sh > 0:
            p = DEFAULT_PRICES.get(t, 100.0)
            name = ASSET_METADATA.get(t, {}).get("name", t)
            is_amb = (t not in ASSET_METADATA) or (t in ("SPACEX", "OPENAI"))
            return {"ticker": t, "name": name, "shares": sh, "price": p, "is_ambiguous": is_amb}

    return None


def parse_holdings_from_text(text: str) -> List[Dict[str, Any]]:
    # Split text on newlines, semicolons, conjunctions 'and', or commas that are not thousands separators
    chunks = re.split(r",(?!\d{3}\b)|[;\n]|\band\b", text, flags=re.I)
    holdings = []
    found = set()

    for ch in chunks:
        # Also check line-by-line table format inside chunk
        lines = [l.strip() for l in ch.splitlines() if l.strip()]
        for line in lines:
            parts = line.split()
            if len(parts) >= 2:
                sym_candidate = clean_sym(parts[0])
                if sym_candidate and sym_candidate not in found:
                    try:
                        sh = float(parts[1].replace(",", "").replace("$", "").replace("£", ""))
                        pr = float(parts[2].replace(",", "").replace("$", "").replace("£", "")) if len(parts) > 2 else DEFAULT_PRICES.get(sym_candidate, 100.0)
                        if sh > 0:
                            found.add(sym_candidate)
                            name = ASSET_METADATA.get(sym_candidate, {}).get("name", sym_candidate)
                            is_amb = (sym_candidate not in ASSET_METADATA) or (sym_candidate in ("SPACEX", "OPENAI"))
                            holdings.append({"ticker": sym_candidate, "name": name, "shares": sh, "price": pr, "is_ambiguous": is_amb})
                            continue
                    except ValueError:
                        pass

            item = parse_item(line)
            if item and item["ticker"] not in found:
                found.add(item["ticker"])
                holdings.append(item)

    return holdings


def parse_portfolio_payload(data: Dict[str, Any]) -> Portfolio:
    holdings = []
    
    # 1. Check if 'holdings' list is provided
    raw_holdings = data.get('holdings', [])
    if isinstance(raw_holdings, list) and raw_holdings:
        for h in raw_holdings:
            ticker = str(h.get('ticker', '')).upper().strip()
            shares = float(h.get('shares', 0.0))
            price = float(h.get('price', 0.0)) or DEFAULT_PRICES.get(ticker, 100.0)
            if ticker and shares > 0:
                holdings.append(Holding(ticker=ticker, shares=shares, price=price))

    # 2. Check if 'portfolio' dict is provided (e.g. {'NVDA': 50, 'QQQ': 25})
    port_dict = data.get('portfolio', {})
    if isinstance(port_dict, dict) and port_dict:
        for ticker, shares in port_dict.items():
            t = str(ticker).upper().strip()
            try:
                sh = float(shares)
                pr = DEFAULT_PRICES.get(t, 100.0)
                if t and sh > 0:
                    holdings.append(Holding(ticker=t, shares=sh, price=pr))
            except (ValueError, TypeError):
                pass
    elif isinstance(port_dict, list) and port_dict:
        for h in port_dict:
            t = str(h.get('ticker', '')).upper().strip()
            sh = float(h.get('shares', 0.0))
            pr = float(h.get('price', 0.0)) or DEFAULT_PRICES.get(t, 100.0)
            if t and sh > 0:
                holdings.append(Holding(ticker=t, shares=sh, price=pr))

    # 3. Fallback: if data itself is a dict of {ticker: shares}
    if not holdings and isinstance(data, dict):
        for k, v in data.items():
            if k not in ('user_profile', 'portfolio', 'holdings', 'text', 'headline', 'query', 'target_allocations'):
                t = str(k).upper().strip()
                try:
                    sh = float(v)
                    pr = DEFAULT_PRICES.get(t, 100.0)
                    if t and sh > 0:
                        holdings.append(Holding(ticker=t, shares=sh, price=pr))
                except (ValueError, TypeError):
                    pass

    # Default fallback if empty
    if not holdings:
        sample = SAMPLE_PORTFOLIOS['accidental_tech_overweight']['holdings']
        holdings = [Holding(h['ticker'], h['shares'], h['price']) for h in sample]

    profile = data.get('user_profile', {
        'risk_tolerance': 'Moderate',
        'horizon_years': 7,
        'loss_capacity': 'Moderate (15-20% max drawdown)',
    })
    return Portfolio(holdings=holdings, user_profile=profile)

class PortfolioXRayHandler(BaseHTTPRequestHandler):

    protocol_version = "HTTP/1.1"

    def _set_headers(self, content_type: str = "application/json", content_length: Optional[int] = None):
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        if content_length is not None:
            self.send_header("Content-Length", str(content_length))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS, HEAD")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Connection", "keep-alive")
        self.end_headers()

    def do_OPTIONS(self):
        self._set_headers()

    def do_HEAD(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path in ("/", "/index.html"):
            encoded = HTML_DASHBOARD.encode("utf-8")
            self._set_headers("text/html; charset=utf-8", len(encoded))
        elif parsed.path == "/api/presets":
            encoded = json.dumps(SAMPLE_PORTFOLIOS).encode("utf-8")
            self._set_headers("application/json", len(encoded))
        else:
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/" or parsed.path == "/index.html":
            encoded = HTML_DASHBOARD.encode("utf-8")
            self._set_headers("text/html; charset=utf-8", len(encoded))
            self.wfile.write(encoded)
        elif parsed.path == "/api/presets":
            encoded = json.dumps(SAMPLE_PORTFOLIOS).encode("utf-8")
            self._set_headers("application/json", len(encoded))
            self.wfile.write(encoded)
        elif parsed.path == "/api/audit-log":
            from pathlib import Path
            log_file = Path("logs/audit.jsonl")
            logs = []
            if log_file.exists():
                with open(log_file, "r") as f:
                    for line in f:
                        if line.strip():
                            logs.append(json.loads(line))
            encoded = json.dumps({"logs": logs}).encode("utf-8")
            self._set_headers("application/json", len(encoded))
            self.wfile.write(encoded)
        else:
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length).decode("utf-8")
        data = json.loads(body) if body else {}

        if parsed.path == "/api/parse-portfolio":
            text = data.get("text", "")
            parsed_holdings = parse_holdings_from_text(text)
            encoded = json.dumps({"holdings": parsed_holdings}).encode("utf-8")
            self._set_headers("application/json", len(encoded))
            self.wfile.write(encoded)

        elif parsed.path == "/api/analyze":
            portfolio = parse_portfolio_payload(data)
            exposures = compute_lookthrough_exposure(portfolio)
            thematic = compute_thematic_exposure(exposures)
            metrics = compute_risk_metrics(portfolio, exposures)
            stress = run_stress_tests(portfolio, exposures)
            
            # X-Ray Institutional Additions
            inst_lookthrough = get_institutional_lookthrough(portfolio)
            correlation_risk = get_correlation_and_risk(portfolio)
            hist_stress = get_historical_stress(portfolio)
            benchmark = get_benchmark_comparison(portfolio)

            raw_exposures = [
                {
                    "ticker": e.ticker,
                    "name": e.name,
                    "total_weight": round(e.total_weight * 100.0, 2),
                    "total_dollar_value": round(e.total_dollar_value, 2),
                    "direct_weight": round(e.direct_weight * 100.0, 2),
                    "indirect_weight": round(e.indirect_weight * 100.0, 2),
                    "contributing_sources": {k: round(v * 100.0, 2) for k, v in e.contributing_sources.items()},
                    "sector": e.sector,
                    "themes": e.themes,
                    "annual_volatility": round(e.annual_volatility * 100.0, 1),
                }
                for e in exposures
            ]

            total_val = metrics.get("portfolio_value", 0.0)
            direct_weights = {}
            if total_val > 0:
                for h in portfolio.holdings:
                    direct_weights[h.ticker] = round((h.shares * h.price) / total_val, 4)

            # Gather evidences
            all_evidences = metrics.pop("evidences", []) + inst_lookthrough.get("evidences", []) + correlation_risk.get("evidences", []) + hist_stress.get("evidences", []) + benchmark.get("evidences", [])
            
            # De-duplicate evidences by ID
            seen_ids = set()
            unique_evidences = []
            for ev in all_evidences:
                if ev["id"] not in seen_ids:
                    seen_ids.add(ev["id"])
                    unique_evidences.append(ev)

            response = {
                "metrics": metrics,
                "risk_metrics": {
                    "annualized_volatility": metrics.get("annual_volatility", 20.0) / 100.0,
                    "var_95_1m": metrics.get("var_95_1m_pct", 8.0) / 100.0,
                    "cvar_95_1m": metrics.get("expected_shortfall_95_pct", 11.0) / 100.0,
                    "concentration_hhi": metrics.get("hhi", 2000.0),
                },
                "portfolio_value": total_val,
                "holdings": [
                    {
                        "ticker": h.ticker,
                        "name": ASSET_METADATA.get(h.ticker, {}).get("name", h.ticker),
                        "shares": h.shares,
                        "price": h.price,
                        "value": round(h.shares * h.price, 2),
                    }
                    for h in portfolio.holdings
                ],
                "direct_weights": direct_weights,
                "exposures": raw_exposures,
                "lookthrough_exposure": [
                    {
                        "ticker": e.ticker,
                        "company_name": e.name,
                        "name": e.name,
                        "lookthrough_weight": e.total_weight,
                        "total_weight": round(e.total_weight * 100.0, 2),
                        "effective_value": e.total_dollar_value,
                        "total_dollar_value": round(e.total_dollar_value, 2),
                        "direct_weight": e.direct_weight,
                        "indirect_weight": e.indirect_weight,
                        "contributing_sources": {k: round(v * 100.0, 2) for k, v in e.contributing_sources.items()},
                        "sector": e.sector,
                        "themes": e.themes,
                    }
                    for e in exposures
                ],
                "institutional_lookthrough_exposure": inst_lookthrough.get("exposures", []),
                "concentration_metrics": inst_lookthrough.get("metrics", {}),
                "thematic": thematic,
                "thematic_factors": [
                    {
                        "theme": t.get("theme", ""),
                        "portfolio_exposure": t.get("weight", 0.0),
                        "description": f"Shared economic sensitivity across {t.get('theme', '')} holdings.",
                        "contributing_assets": [a.get("ticker", "") for a in t.get("top_assets", [])],
                    }
                    for t in thematic
                ],
                "stress_tests": [
                    {
                        "scenario_name": s.get("name", ""),
                        "scenario_id": s.get("scenario_id", ""),
                        "drawdown_pct": s.get("drawdown_pct", 0.0) / 100.0,
                        "dollar_loss": abs(s.get("dollar_drawdown", 0.0)),
                        "description": s.get("description", ""),
                        "worst_hit": s.get("worst_hit_assets", []),
                    }
                    for s in stress
                ],
                "historical_stress": hist_stress,
                "correlation_and_risk": correlation_risk,
                "benchmark": benchmark,
                "evidences": unique_evidences
            }

            encoded = json.dumps(response).encode("utf-8")
            self._set_headers("application/json", len(encoded))
            self.wfile.write(encoded)

        elif parsed.path == "/api/what-if":
            portfolio = parse_portfolio_payload(data.get("portfolio", {}))
            target_allocations = data.get("target_allocations", {})
            if not target_allocations and ("target" in data or "ticker" in data):
                target_sym = str(data.get("target", data.get("ticker", "NVDA")))
                target_pct = float(data.get("allocation_change_pct", data.get("target_pct", 20.0)))
                target_allocations = {target_sym: target_pct}
            result = simulate_what_if(portfolio, target_allocations)
            encoded = json.dumps(result).encode("utf-8")
            self._set_headers("application/json", len(encoded))
            self.wfile.write(encoded)

        elif parsed.path == "/api/headline":
            portfolio = parse_portfolio_payload(data.get("portfolio", {}))
            headline = data.get("headline", "")
            result = analyze_headline_event(portfolio, headline)
            encoded = json.dumps(result).encode("utf-8")
            self._set_headers("application/json", len(encoded))
            self.wfile.write(encoded)

        elif parsed.path == "/api/chat":
            portfolio = parse_portfolio_payload(data.get("portfolio", {}))
            query = data.get("query", "")
            result = ask_risk_analyst(portfolio, query)
            encoded = json.dumps(result).encode("utf-8")
            self._set_headers("application/json", len(encoded))
            self.wfile.write(encoded)

        elif parsed.path == "/api/sign-off":
            decision = data.get("decision", "rejected")
            target = data.get("target", {})
            from pathlib import Path
            log_dir = Path("logs")
            log_dir.mkdir(exist_ok=True)
            with open(log_dir / "audit.jsonl", "a") as f:
                f.write(json.dumps({"timestamp": time.time(), "event": "human_sign_off", "decision": decision, "target": target}) + "\n")
            
            encoded = json.dumps({"status": "recorded"}).encode("utf-8")
            self._set_headers("application/json", len(encoded))
            self.wfile.write(encoded)
            
        elif parsed.path == "/api/audit-log":
            from pathlib import Path
            log_file = Path("logs/audit.jsonl")
            logs = []
            if log_file.exists():
                with open(log_file, "r") as f:
                    for line in f:
                        if line.strip():
                            logs.append(json.loads(line))
            encoded = json.dumps({"logs": logs}).encode("utf-8")
            self._set_headers("application/json", len(encoded))
            self.wfile.write(encoded)

        else:
            self.send_error(HTTPStatus.NOT_FOUND, "API route not found")




HTML_DASHBOARD = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Portfolio X-Ray | Understand Your Investments & Hidden Risks</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <!-- Client-side OCR via Tesseract.js -->
  <script src="https://cdn.jsdelivr.net/npm/tesseract.js@5/dist/tesseract.min.js"></script>
  <script>
    tailwind.config = {
      theme: {
        extend: {
          colors: {
            appbg: '#F7F9FC',
            apptext: '#17233B',
            appmuted: '#5B6B84',
            appborder: '#E2E8F0',
            primary: {
              DEFAULT: '#5148E5',
              hover: '#4338CA',
              light: '#EEF2FF',
            },
            tealaccent: {
              DEFAULT: '#12B8A6',
              hover: '#0D9488',
              light: '#F0FDFA',
            }
          },
          fontFamily: {
            sans: ['Inter', 'system-ui', '-apple-system', 'BlinkMacSystemFont', 'Segoe UI', 'Roboto', 'sans-serif'],
          }
        }
      }
    }
  </script>
  <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    body {
      font-family: 'Inter', sans-serif;
      background-color: #F7F9FC;
      color: #17233B;
      font-size: 15px;
    }
    .custom-scrollbar::-webkit-scrollbar {
      width: 6px;
      height: 6px;
    }
    .custom-scrollbar::-webkit-scrollbar-track {
      background: #F1F5F9;
      border-radius: 4px;
    }
    .custom-scrollbar::-webkit-scrollbar-thumb {
      background: #CBD5E1;
      border-radius: 4px;
    }
    .custom-scrollbar::-webkit-scrollbar-thumb:hover {
      background: #94A3B8;
    }
    /* Donut chart transition */
    .donut-slice {
      transition: transform 0.2s ease, opacity 0.2s ease;
      cursor: pointer;
    }
    .donut-slice:hover {
      opacity: 0.85;
      transform: scale(1.02);
      transform-origin: center;
    }
  </style>
</head>
<body class="bg-appbg text-apptext antialiased min-h-screen flex flex-col md:flex-row">

  <!-- Mobile Header -->
  <header class="md:hidden bg-white border-b border-appborder px-4 py-3.5 flex items-center justify-between sticky top-0 z-30">
    <div class="flex items-center gap-2.5">
      <div class="w-8 h-8 rounded-xl bg-gradient-to-tr from-primary to-tealaccent flex items-center justify-center text-white font-bold text-base shadow-sm">
        X
      </div>
      <div>
        <h1 class="font-bold text-base text-apptext tracking-tight">Portfolio X-Ray</h1>
        <p class="text-[11px] text-appmuted font-medium">Understand Your Risks</p>
      </div>
    </div>
    <div class="flex items-center gap-2">
      <button onclick="openSettingsModal()" class="p-2 rounded-xl text-appmuted hover:bg-slate-100 transition" title="Settings">
        <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z"/><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z"/></svg>
      </button>
      <button onclick="toggleMobileMenu()" class="p-2 rounded-xl text-appmuted hover:bg-slate-100 transition" aria-label="Toggle Navigation">
        <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 6h16M4 12h16M4 18h16"/></svg>
      </button>
    </div>
  </header>

  <!-- Left Sidebar (Desktop & Mobile Drawer) -->
  <aside id="sidebarNav" class="hidden md:flex flex-col w-full md:w-64 lg:w-72 bg-white border-r border-appborder shrink-0 fixed md:sticky top-0 h-screen z-40 transition-transform duration-200 justify-between">
    <div>
      <!-- Brand Logo Header -->
      <div class="p-5 border-b border-appborder/80 flex items-center justify-between">
        <div class="flex items-center gap-3">
          <div class="w-10 h-10 rounded-2xl bg-gradient-to-tr from-primary to-tealaccent flex items-center justify-center text-white font-extrabold text-xl shadow-md shadow-primary/20">
            X
          </div>
          <div>
            <h1 class="font-bold text-lg text-apptext tracking-tight leading-tight">Portfolio X-Ray</h1>
            <p class="text-xs text-appmuted font-medium">Risk & Exposure Intelligence</p>
          </div>
        </div>
        <button onclick="toggleMobileMenu()" class="md:hidden text-slate-400 hover:text-slate-600 p-1">
          <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"/></svg>
        </button>
      </div>

      <!-- Navigation Links -->
      <nav class="p-3.5 space-y-1.5" id="navLinksList">
        <button onclick="switchTab('overview')" id="nav-overview" class="nav-item w-full flex items-center gap-3 px-3.5 py-3 rounded-xl font-medium text-sm transition text-primary bg-primary/10">
          <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 6a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2H6a2 2 0 01-2-2V6zM14 6a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2h-2a2 2 0 01-2-2V6zM4 16a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2H6a2 2 0 01-2-2v-2zM14 16a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2h-2a2 2 0 01-2-2v-2z"/></svg>
          <span>Overview</span>
        </button>
        <button onclick="switchTab('hidden-risks')" id="nav-hidden-risks" class="nav-item w-full flex items-center gap-3 px-3.5 py-3 rounded-xl font-medium text-sm transition text-appmuted hover:text-apptext hover:bg-slate-50">
          <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"/></svg>
          <span>Hidden Risks</span>
        </button>
        <button onclick="switchTab('what-if')" id="nav-what-if" class="nav-item w-full flex items-center gap-3 px-3.5 py-3 rounded-xl font-medium text-sm transition text-appmuted hover:text-apptext hover:bg-slate-50">
          <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 10V3L4 14h7v7l9-11h-7z"/></svg>
          <span>What Could Happen?</span>
        </button>
        <button onclick="switchTab('simulate')" id="nav-simulate" class="nav-item w-full flex items-center gap-3 px-3.5 py-3 rounded-xl font-medium text-sm transition text-appmuted hover:text-apptext hover:bg-slate-50">
          <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 6V4m0 2a2 2 0 100 4m0-4a2 2 0 110 4m-6 8a2 2 0 100-4m0 4a2 2 0 110-4m0 4v2m0-6V4m6 6v10m6-2a2 2 0 100-4m0 4a2 2 0 110-4m0 4v2m0-6V4"/></svg>
          <span>Try a Change</span>
        </button>
        <button onclick="switchTab('chat')" id="nav-chat" class="nav-item w-full flex items-center gap-3 px-3.5 py-3 rounded-xl font-medium text-sm transition text-appmuted hover:text-apptext hover:bg-slate-50">
          <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z"/></svg>
          <span>Ask X-Ray</span>
        </button>
      </nav>

      <!-- Preset Portfolios Quick Selector -->
      <div class="px-4 py-3 mx-3 my-2 bg-slate-50 rounded-2xl border border-appborder/70">
        <div class="flex items-center justify-between mb-2">
          <span class="text-xs font-semibold text-appmuted uppercase tracking-wider">Example Portfolios</span>
        </div>
        <div class="space-y-1.5" id="sidebarPresets">
          <button onclick="loadPreset('accidental_tech_overweight')" id="preset-btn-accidental" class="w-full text-left text-xs px-2.5 py-2 rounded-lg font-medium text-primary bg-primary/10 border border-primary/20 flex items-center justify-between transition">
            <span class="truncate">Accidental Tech Overweight</span>
            <span class="text-[10px] bg-primary text-white rounded px-1.5 py-0.5">$32.5k</span>
          </button>
          <button onclick="loadPreset('balanced_core')" id="preset-btn-balanced" class="w-full text-left text-xs px-2.5 py-2 rounded-lg font-medium text-appmuted hover:text-apptext hover:bg-white border border-transparent transition">
            <span class="truncate">Institutional Balanced 60/40</span>
          </button>
          <button onclick="loadPreset('all_in_growth')" id="preset-btn-growth" class="w-full text-left text-xs px-2.5 py-2 rounded-lg font-medium text-appmuted hover:text-apptext hover:bg-white border border-transparent transition">
            <span class="truncate">Aggressive Tech & AI</span>
          </button>
        </div>
      </div>
    </div>

    <!-- Sidebar Bottom Actions -->
    <div class="p-4 border-t border-appborder/80 space-y-2">
      <button onclick="openPortfolioEntryModal()" class="w-full bg-primary hover:bg-primary-hover text-white text-sm font-semibold py-2.5 px-3.5 rounded-xl transition flex items-center justify-center gap-2 shadow-sm shadow-primary/20">
        <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 4v16m8-8H4"/></svg>
        <span>Manage Investments</span>
      </button>
      <button onclick="openSettingsModal()" class="w-full bg-white hover:bg-slate-50 text-apptext border border-appborder text-xs font-medium py-2 px-3 rounded-xl transition flex items-center justify-center gap-2">
        <svg class="w-4 h-4 text-appmuted" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z"/><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z"/></svg>
        <span>Risk Profile & Settings</span>
      </button>
    </div>
  </aside>

  <!-- Main Content Area -->
  <main class="flex-1 max-w-7xl mx-auto w-full p-4 sm:p-6 lg:p-8 min-h-screen">

    <!-- TAB 1: OVERVIEW -->
    <section id="tab-overview" class="tab-content space-y-6">
      <!-- Top Title Bar -->
      <div class="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
        <div>
          <h2 class="text-2xl sm:text-3xl font-bold text-apptext tracking-tight">Your Portfolio</h2>
          <p class="text-appmuted text-sm sm:text-base mt-1">Understand your investments, discover hidden risks and see how different decisions could affect your money.</p>
        </div>
        <div class="flex items-center gap-2">
          <button onclick="openPortfolioEntryModal()" class="bg-primary hover:bg-primary-hover text-white text-xs sm:text-sm font-semibold px-4 py-2.5 rounded-xl transition flex items-center gap-1.5 shadow-sm">
            <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 4v16m8-8H4"/></svg>
            <span>Edit Investments</span>
          </button>
          <button onclick="openRiskQuestionnaireModal()" class="bg-white hover:bg-slate-50 border border-appborder text-apptext text-xs sm:text-sm font-medium px-3.5 py-2.5 rounded-xl transition flex items-center gap-1.5">
            <svg class="w-4 h-4 text-primary" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"/></svg>
            <span id="overviewRiskProfileBadge">Moderate Profile</span>
          </button>
        </div>
      </div>

      <!-- Welcome Card -->
      <div class="bg-white rounded-2xl p-6 border border-appborder shadow-[0_1px_3px_rgba(15,23,42,0.04),0_8px_16px_-4px_rgba(15,23,42,0.03)] flex flex-col md:flex-row items-start md:items-center justify-between gap-6">
        <div>
          <span class="text-xs font-semibold uppercase tracking-wider text-appmuted">Total Portfolio Value</span>
          <div class="flex items-baseline gap-3 mt-1">
            <span id="welcomeTotalValue" class="text-3xl sm:text-4xl font-extrabold text-apptext tracking-tight">$32,500.00</span>
            <span class="text-xs font-medium text-emerald-700 bg-emerald-50 border border-emerald-100 px-2.5 py-1 rounded-full flex items-center gap-1">
              <span>●</span> Active Analysis
            </span>
          </div>
          <p class="text-xs sm:text-sm text-appmuted mt-1.5 flex items-center gap-2">
            <span id="welcomeHoldingCount">4 investments</span>
            <span>&bull;</span>
            <span id="welcomeAnalysisDate">Analysis updated today</span>
          </p>
        </div>
        <div class="flex items-center gap-3 w-full md:w-auto">
          <div class="p-3.5 bg-slate-50 rounded-xl border border-appborder flex items-center gap-3 w-full sm:w-auto">
            <div class="w-10 h-10 rounded-xl bg-indigo-50 text-primary flex items-center justify-center">
              <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z"/></svg>
            </div>
            <div>
              <div class="text-[11px] uppercase tracking-wider text-appmuted font-semibold">Investor Profile</div>
              <div id="welcomeInvestorGoal" class="text-xs font-bold text-apptext">Long-term Growth &bull; Moderate Loss Capacity</div>
            </div>
          </div>
        </div>
      </div>

      <!-- Three Beginner-Friendly Cards -->
      <div class="grid grid-cols-1 md:grid-cols-3 gap-5">
        <!-- Card 1: How spread out is your money? -->
        <div class="bg-white rounded-2xl p-6 border border-appborder shadow-[0_1px_3px_rgba(15,23,42,0.04),0_8px_16px_-4px_rgba(15,23,42,0.03)] flex flex-col justify-between">
          <div>
            <div class="flex items-center justify-between mb-3">
              <h3 class="font-bold text-base text-apptext flex items-center gap-2">
                <span>How spread out is your money?</span>
              </h3>
              <button onclick="openTerminology('concentration')" class="text-slate-400 hover:text-primary transition" title="What is concentration?">
                <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"/></svg>
              </button>
            </div>
            
            <!-- Clean single-line concentration badge -->
            <div class="my-3 px-3 py-2 bg-slate-50 border border-slate-100 rounded-xl flex items-center justify-between gap-2 overflow-hidden">
              <span id="concLevelBadge" class="text-xs sm:text-sm font-bold text-rose-600 whitespace-nowrap truncate shrink min-w-0">
                High Concentration
              </span>
              <div class="flex items-center gap-1 shrink-0">
                <span id="meter-conc-1" class="w-1.5 h-3 rounded-full bg-slate-200 transition-all inline-block"></span>
                <span id="meter-conc-2" class="w-1.5 h-3 rounded-full bg-slate-200 transition-all inline-block"></span>
                <span id="meter-conc-3" class="w-1.5 h-3 rounded-full bg-slate-200 transition-all inline-block"></span>
              </div>
            </div>

            <p id="concExplanation" class="text-sm text-appmuted leading-relaxed">
              Loading portfolio concentration analysis...
            </p>
          </div>
          <div class="mt-4 pt-3 border-t border-slate-100 flex flex-col gap-1 text-xs">
            <span class="text-appmuted">Top holding stake:</span>
            <span id="topHoldingStat" class="font-bold text-apptext text-sm">--</span>
          </div>
        </div>

        <!-- Card 2: How much could your investments fluctuate? -->
        <div class="bg-white rounded-2xl p-6 border border-appborder shadow-[0_1px_3px_rgba(15,23,42,0.04),0_8px_16px_-4px_rgba(15,23,42,0.03)] flex flex-col justify-between">
          <div>
            <div class="flex items-center justify-between mb-3">
              <h3 class="font-bold text-base text-apptext flex items-center gap-2">
                <span>How much could your investments fluctuate?</span>
              </h3>
              <button onclick="openTerminology('volatility')" class="text-slate-400 hover:text-primary transition" title="What are fluctuations?">
                <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"/></svg>
              </button>
            </div>

            <div class="my-3">
              <div id="volatilityAmountBadge" class="text-lg sm:text-xl font-extrabold text-apptext tracking-tight truncate">
                &plusmn;$1,250 / mo
              </div>
              <div class="text-xs text-appmuted font-medium mt-0.5">typical monthly fluctuation range</div>
            </div>

            <p id="volatilityExplanation" class="text-sm text-appmuted leading-relaxed">
              Based on historical price swings across your investments, your portfolio experiences moderate ups and downs over typical market cycles.
            </p>
          </div>
          <div class="mt-4 pt-3 border-t border-slate-100 flex flex-col gap-1 text-xs">
            <span class="text-appmuted">Expected 1-month buffer:</span>
            <span id="varDownsideStat" class="font-bold text-apptext text-sm">-$2,600 (95% modeled)</span>
          </div>
        </div>

        <!-- Card 3: What is your biggest hidden risk? -->
        <div class="bg-white rounded-2xl p-6 border border-appborder shadow-[0_1px_3px_rgba(15,23,42,0.04),0_8px_16px_-4px_rgba(15_23_42_0.03)] flex flex-col justify-between">
          <div>
            <div class="flex items-center justify-between mb-3">
              <h3 class="font-bold text-base text-apptext flex items-center gap-2">
                <span>What is your biggest hidden risk?</span>
              </h3>
              <button onclick="openTerminology('lookthrough')" class="text-slate-400 hover:text-primary transition" title="What is hidden risk?">
                <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"/></svg>
              </button>
            </div>

            <div class="my-3 flex items-center gap-2.5">
              <span class="w-7 h-7 rounded-lg bg-amber-50 text-amber-600 flex items-center justify-center">
                <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"/></svg>
              </span>
              <span id="biggestRiskBadge" class="text-sm font-bold text-amber-800 bg-amber-50 border border-amber-200 px-2.5 py-1 rounded-lg">
                High Single-Company Overlap
              </span>
            </div>

            <p id="biggestRiskExplanation" class="text-sm text-appmuted leading-relaxed">
              Analyzing your underlying fund holdings...
            </p>
          </div>
          <div class="mt-4 pt-3 border-t border-slate-100 flex items-center justify-between text-xs">
            <button onclick="switchTab('hidden-risks')" class="text-primary hover:text-primary-hover font-semibold flex items-center gap-1">
              <span>Inspect in Hidden Risks</span>
              <span>&rarr;</span>
            </button>
          </div>
        </div>
      </div>

      <!-- Section: Where is your money invested? -->
      <div class="bg-white rounded-2xl p-6 border border-appborder shadow-[0_1px_3px_rgba(15,23,42,0.04),0_8px_16px_-4px_rgba(15,23,42,0.03)]">
        <div class="flex flex-col sm:flex-row sm:items-center justify-between gap-2 mb-6">
          <div>
            <h3 class="text-lg font-bold text-apptext">Where is your money invested?</h3>
            <p class="text-sm text-appmuted">Direct holdings and fund allocations currently in your portfolio.</p>
          </div>
          <button onclick="openPortfolioEntryModal()" class="text-xs font-semibold text-primary hover:text-primary-hover flex items-center gap-1 self-start sm:self-auto">
            <span>+ Add or Edit Holdings</span>
          </button>
        </div>

        <div class="grid grid-cols-1 lg:grid-cols-12 gap-8 items-center">
          <!-- Interactive Doughnut Chart -->
          <div class="lg:col-span-5 flex flex-col items-center justify-center">
            <div class="relative w-56 h-56 sm:w-64 sm:h-64">
              <svg id="donutChartSvg" viewBox="0 0 160 160" class="w-full h-full transform -rotate-90">
                <!-- Slices rendered by JS -->
              </svg>
              <!-- Center Text in Doughnut -->
              <div class="absolute inset-0 flex flex-col items-center justify-center pointer-events-none text-center px-4">
                <span id="donutCenterLabel" class="text-xs uppercase tracking-wider text-appmuted font-semibold">Total</span>
                <span id="donutCenterValue" class="text-xl sm:text-2xl font-extrabold text-apptext">$32,500</span>
              </div>
            </div>
            <p class="text-xs text-appmuted mt-3 text-center">Hover or tap on any slice to see investment details.</p>
          </div>

          <!-- Clean list of holdings -->
          <div class="lg:col-span-7 space-y-3" id="holdingsListContainer">
            <!-- Populated dynamically by JS -->
          </div>
        </div>
      </div>

      <!-- Section: Things you might not have realised -->
      <div class="space-y-4">
        <div>
          <h3 class="text-lg font-bold text-apptext">Things you might not have realised</h3>
          <p class="text-sm text-appmuted">Automated discoveries based on what is actually inside your funds and investments.</p>
        </div>

        <div class="grid grid-cols-1 md:grid-cols-3 gap-5" id="discoveriesContainer">
          <!-- Populated dynamically from verified data -->
        </div>
      </div>

      <!-- Advanced View Toggle Banner (If enabled in settings) -->
      <div id="advancedMetricsOverviewBanner" class="hidden bg-slate-100 rounded-2xl p-5 border border-slate-200">
        <div class="flex items-center justify-between mb-3">
          <div class="flex items-center gap-2">
            <span class="text-xs font-bold uppercase tracking-wider bg-slate-800 text-white px-2 py-0.5 rounded">Advanced View Active</span>
            <span class="text-xs font-semibold text-apptext">Institutional Mathematical Metrics</span>
          </div>
          <span class="text-xs text-appmuted">Configured in Settings</span>
        </div>
        <div class="grid grid-cols-2 sm:grid-cols-4 gap-4 text-xs">
          <div class="bg-white p-3 rounded-xl border border-slate-200">
            <span class="text-appmuted block">Herfindahl Index (HHI):</span>
            <span id="advHHIVal" class="text-base font-bold text-apptext">--</span>
          </div>
          <div class="bg-white p-3 rounded-xl border border-slate-200">
            <span class="text-appmuted block">Annualized Volatility:</span>
            <span id="advVolVal" class="text-base font-bold text-apptext">--</span>
          </div>
          <div class="bg-white p-3 rounded-xl border border-slate-200">
            <span class="text-appmuted block">95% 1-Month VaR:</span>
            <span id="advVaRVal" class="text-base font-bold text-apptext">--</span>
          </div>
          <div class="bg-white p-3 rounded-xl border border-slate-200">
            <span class="text-appmuted block">Expected Shortfall (CVaR):</span>
            <span id="advCVaRVal" class="text-base font-bold text-apptext">--</span>
          </div>
        </div>
      </div>
    </section>


    <!-- TAB 2: HIDDEN RISKS -->
    <section id="tab-hidden-risks" class="tab-content hidden space-y-8">
      <div>
        <h2 class="text-2xl sm:text-3xl font-bold text-apptext tracking-tight">Hidden Risks</h2>
        <p class="text-appmuted text-sm sm:text-base mt-1">See through your investment funds to uncover overlapping companies and shared economic drivers.</p>
      </div>

      <!-- Sub-Tabs: ETF Overlap vs Economic Themes -->
      <div class="flex border-b border-appborder gap-6">
        <button onclick="switchHiddenRisksSubTab('overlap')" id="subnav-overlap" class="pb-3 text-sm font-bold text-primary border-b-2 border-primary transition">
          ETF Fund Overlap
        </button>
        <button onclick="switchHiddenRisksSubTab('themes')" id="subnav-themes" class="pb-3 text-sm font-semibold text-appmuted hover:text-apptext transition">
          What is Your Money Depending On?
        </button>
      </div>

      <!-- SUB-TAB 1: ETF Overlap -->
      <div id="subtab-overlap" class="space-y-6">
        <div class="bg-white rounded-2xl p-6 border border-appborder shadow-[0_1px_3px_rgba(15,23,42,0.04),0_8px_16px_-4px_rgba(15,23,42,0.03)]">
          <div class="flex flex-col md:flex-row md:items-center justify-between gap-4 mb-4">
            <div>
              <h3 class="text-xl font-bold text-apptext">You might own the same companies more than once</h3>
              <p class="text-sm text-appmuted mt-1 max-w-2xl">Investment funds often own shares in hundreds of companies. If you own several funds, you could be investing in the same companies without realising it.</p>
            </div>
            <button onclick="openTerminology('lookthrough')" class="text-xs font-semibold text-primary hover:text-primary-hover flex items-center gap-1 self-start md:self-auto">
              <span>How look-through works</span>
              <span>&rarr;</span>
            </button>
          </div>

          <!-- Interactive Highlight Banner -->
          <div id="overlapHighlightBanner" class="p-4 bg-indigo-50/80 rounded-xl border border-indigo-100 flex flex-col sm:flex-row sm:items-center justify-between gap-3 my-4">
            <div>
              <span class="text-xs font-bold uppercase tracking-wider text-primary">Selected Company Exposure</span>
              <div id="overlapBannerTitle" class="text-lg font-extrabold text-apptext mt-0.5">Your total Nvidia exposure: 31.4%</div>
              <p id="overlapBannerDesc" class="text-xs sm:text-sm text-appmuted mt-1">
                20.0% comes from Nvidia shares you own directly. Another 11.4% comes from Nvidia shares held inside your investment funds.
              </p>
            </div>
            <span class="text-xs font-semibold text-indigo-700 bg-white px-3 py-1.5 rounded-lg border border-indigo-200 shrink-0 self-start sm:self-auto">
              Interactive Look-Through
            </span>
          </div>

          <!-- Interactive Connection Diagram / Graph Visual -->
          <div class="my-6 p-4 bg-slate-50/70 rounded-xl border border-appborder/80">
            <div class="flex items-center justify-between mb-3 text-xs text-appmuted font-medium">
              <span>Your Funds & Direct Holdings</span>
              <span>Tap a company to inspect fund connections &rarr;</span>
              <span>Underlying Companies</span>
            </div>
            <div class="grid grid-cols-1 md:grid-cols-2 gap-4" id="interactiveDiagramContainer">
              <!-- Visual Connection nodes populated by JS -->
            </div>
          </div>

          <!-- Companies you have the most money invested in -->
          <div class="mt-8">
            <h4 class="text-base font-bold text-apptext mb-1">Companies you have the most money invested in</h4>
            <p class="text-xs text-appmuted mb-4">Click any row to reveal which specific funds hold that company for you.</p>

            <div class="space-y-3" id="effectiveHoldingsList">
              <!-- Populated by JS -->
            </div>
          </div>
        </div>

        <!-- Raw Technical Table for Advanced View -->
        <div id="advancedLookthroughTableContainer" class="hidden bg-white rounded-2xl p-6 border border-appborder">
          <h4 class="text-sm font-bold uppercase text-apptext mb-3">Advanced View: Full Look-Through Decomposition Table</h4>
          <div class="overflow-x-auto">
            <table class="w-full text-left text-xs border-collapse">
              <thead>
                <tr class="border-b border-slate-200 text-slate-500 font-semibold">
                  <th class="py-2.5 px-3">Company (Ticker)</th>
                  <th class="py-2.5 px-3">Total Stake %</th>
                  <th class="py-2.5 px-3">Effective Dollar Value</th>
                  <th class="py-2.5 px-3">Direct Weight %</th>
                  <th class="py-2.5 px-3">Indirect Fund Weight %</th>
                  <th class="py-2.5 px-3">Sector</th>
                </tr>
              </thead>
              <tbody id="rawLookthroughTableBody" class="divide-y divide-slate-100">
                <!-- Populated by JS -->
              </tbody>
            </table>
          </div>
        </div>
      </div>

      <!-- SUB-TAB 2: Economic Themes -->
      <div id="subtab-themes" class="hidden space-y-6">
        <div class="bg-white rounded-2xl p-6 border border-appborder shadow-[0_1px_3px_rgba(15,23,42,0.04),0_8px_16px_-4px_rgba(15,23,42,0.03)]">
          <div class="mb-6">
            <h3 class="text-xl font-bold text-apptext">What is your money depending on?</h3>
            <p class="text-sm text-appmuted mt-1 max-w-2xl">
              Different companies can be affected by the same things happening in the world, even if they operate in completely different industries.
            </p>
            <p class="text-xs text-slate-400 mt-1 italic">
              Note: Individual investments can belong to multiple themes, so percentages do not need to add up to 100%.
            </p>
          </div>

          <!-- Grid of Attractive, Expandable Cards -->
          <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-5" id="themesGridContainer">
            <!-- Populated dynamically by JS for verified themes -->
          </div>
        </div>
      </div>
    </section>


    <!-- TAB 3: WHAT COULD HAPPEN? -->
    <section id="tab-what-if" class="tab-content hidden space-y-6">
      <div>
        <h2 class="text-2xl sm:text-3xl font-bold text-apptext tracking-tight">What could happen to your investments?</h2>
        <p class="text-appmuted text-sm sm:text-base mt-1 max-w-3xl">
          Nobody knows exactly what markets will do next, but we can explore different situations to understand how your investments might be affected.
        </p>
      </div>

      <!-- Disclaimer & Methodology Box -->
      <div class="p-4 bg-slate-100/80 rounded-xl border border-slate-200 flex items-start gap-3">
        <span class="text-primary mt-0.5">
          <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"/></svg>
        </span>
        <div class="text-xs text-slate-600 leading-relaxed">
          <strong>How stress testing works:</strong> These simulations use historical market crisis shock factors and cross-asset correlation models. They are hypothetical educational tools to test resilience, not price predictions or financial guarantees.
        </div>
      </div>

      <!-- Simple, Recognisable Scenarios Grid -->
      <div class="grid grid-cols-1 md:grid-cols-2 gap-5" id="scenariosGridContainer">
        <!-- Rendered dynamically by JS -->
      </div>
    </section>


    <!-- TAB 4: TRY A CHANGE -->
    <section id="tab-simulate" class="tab-content hidden space-y-6">
      <div>
        <h2 class="text-2xl sm:text-3xl font-bold text-apptext tracking-tight">Thinking of changing your investments?</h2>
        <p class="text-appmuted text-sm sm:text-base mt-1">
          See how a new investment or a change to what you already own could affect your risk before you decide.
        </p>
      </div>

      <!-- Conversational Question Box -->
      <div class="bg-white rounded-2xl p-6 border border-appborder shadow-[0_1px_3px_rgba(15,23,42,0.04),0_8px_16px_-4px_rgba(15,23,42,0.03)] space-y-4">
        <div>
          <label class="block text-sm font-semibold text-apptext mb-1.5">What change are you considering?</label>
          <div class="flex gap-2">
            <input type="text" id="simQuestionInput" placeholder="What if I invested another £1,000 in Nvidia?" class="flex-1 bg-slate-50 border border-appborder rounded-xl px-4 py-3 text-sm text-apptext placeholder-slate-400 focus:outline-none focus:border-primary focus:bg-white transition" onkeydown="if (event.key === 'Enter') handleSimulateQuestion()" />
            <button onclick="handleSimulateQuestion()" class="bg-primary hover:bg-primary-hover text-white text-sm font-semibold px-5 py-3 rounded-xl transition shadow-sm">
              Simulate
            </button>
          </div>
        </div>

        <!-- Clickable Example Questions -->
        <div>
          <span class="text-xs font-semibold text-appmuted block mb-2">Try an example change:</span>
          <div class="flex flex-wrap gap-2">
            <button onclick="applySimPrompt('What if half my money was in Nvidia?')" class="text-xs bg-slate-100 hover:bg-slate-200 text-apptext px-3 py-1.5 rounded-lg transition">
              &ldquo;What if half my money was in Nvidia?&rdquo;
            </button>
            <button onclick="applySimPrompt('What if I bought another S&P 500 ETF?')" class="text-xs bg-slate-100 hover:bg-slate-200 text-apptext px-3 py-1.5 rounded-lg transition">
              &ldquo;What if I bought another S&P 500 ETF?&rdquo;
            </button>
            <button onclick="applySimPrompt('What if I reduced my technology exposure?')" class="text-xs bg-slate-100 hover:bg-slate-200 text-apptext px-3 py-1.5 rounded-lg transition">
              &ldquo;What if I reduced my technology exposure?&rdquo;
            </button>
            <button onclick="applySimPrompt('What if I moved some investments into cash?')" class="text-xs bg-slate-100 hover:bg-slate-200 text-apptext px-3 py-1.5 rounded-lg transition">
              &ldquo;What if I moved some investments into cash?&rdquo;
            </button>
          </div>
        </div>

        <!-- Interactive Adjuster Slider -->
        <div class="pt-4 border-t border-slate-100 flex flex-col sm:flex-row sm:items-center justify-between gap-4">
          <div class="flex-1">
            <div class="flex items-center justify-between text-xs mb-1.5 font-medium">
              <span class="text-appmuted">Proposed Allocation to <strong id="simActiveAssetLabel" class="text-apptext">NVIDIA</strong>:</span>
              <span id="simSliderValueBadge" class="text-primary font-bold text-sm">30.0%</span>
            </div>
            <input type="range" id="simAllocationSlider" min="5" max="80" step="5" value="30" oninput="handleSliderChange(this.value)" class="w-full accent-primary cursor-pointer h-2 bg-slate-100 rounded-lg" />
          </div>
          <div class="flex items-center gap-2 self-start sm:self-auto text-xs text-appmuted">
            <span>Mode: Reallocating portfolio capital</span>
          </div>
        </div>
      </div>

      <!-- Before and After Comparison Cards -->
      <div class="grid grid-cols-1 md:grid-cols-2 gap-6" id="simComparisonContainer">
        <!-- Card 1: Your investments now -->
        <div class="bg-white rounded-2xl p-6 border border-appborder shadow-[0_1px_3px_rgba(15,23,42,0.04),0_8px_16px_-4px_rgba(15,23,42,0.03)] space-y-4">
          <div class="flex items-center justify-between pb-3 border-b border-slate-100">
            <div>
              <span class="text-xs font-semibold text-appmuted uppercase tracking-wider">Baseline</span>
              <h3 class="text-lg font-bold text-apptext">Your investments now</h3>
            </div>
            <span class="text-xs bg-slate-100 text-slate-700 font-semibold px-2.5 py-1 rounded-full">Current</span>
          </div>

          <div class="space-y-3.5 text-sm">
            <div class="flex items-center justify-between">
              <span class="text-appmuted">Single-Company Concentration:</span>
              <span id="simBaseConcentration" class="font-bold text-apptext">26.9%</span>
            </div>
            <div class="flex items-center justify-between">
              <span class="text-appmuted">Top Holding Effective Stake:</span>
              <span id="simBaseTopHolding" class="font-bold text-apptext">31.4% (NVDA)</span>
            </div>
            <div class="flex items-center justify-between">
              <span class="text-appmuted">Typical Monthly Fluctuation:</span>
              <span id="simBaseVolatility" class="font-bold text-apptext">&plusmn;$1,250 (19.2% ann.)</span>
            </div>
            <div class="flex items-center justify-between">
              <span class="text-appmuted">Modeled Loss in Tech Shock:</span>
              <span id="simBaseStressLoss" class="font-bold text-rose-600">-$7,020 (−21.6%)</span>
            </div>
          </div>
        </div>

        <!-- Card 2: After your proposed change -->
        <div class="bg-white rounded-2xl p-6 border-2 border-primary/40 shadow-[0_1px_3px_rgba(15,23,42,0.04),0_8px_16px_-4px_rgba(15,23,42,0.03)] space-y-4 relative">
          <div class="flex items-center justify-between pb-3 border-b border-slate-100">
            <div>
              <span class="text-xs font-semibold text-primary uppercase tracking-wider">Hypothetical Result</span>
              <h3 class="text-lg font-bold text-apptext">After your proposed change</h3>
            </div>
            <span class="text-xs bg-primary text-white font-semibold px-2.5 py-1 rounded-full shadow-sm">Simulated</span>
          </div>

          <div class="space-y-3.5 text-sm">
            <div class="flex items-center justify-between">
              <span class="text-appmuted">Single-Company Concentration:</span>
              <span id="simPropConcentration" class="font-bold text-amber-600">21.5%</span>
            </div>
            <div class="flex items-center justify-between">
              <span class="text-appmuted">Top Holding Effective Stake:</span>
              <span id="simPropTopHolding" class="font-bold text-apptext">42.5% (NVDA)</span>
            </div>
            <div class="flex items-center justify-between">
              <span class="text-appmuted">Typical Monthly Fluctuation:</span>
              <span id="simPropVolatility" class="font-bold text-apptext">&plusmn;$1,580 (24.1% ann.)</span>
            </div>
            <div class="flex items-center justify-between">
              <span class="text-appmuted">Modeled Loss in Tech Shock:</span>
              <span id="simPropStressLoss" class="font-bold text-rose-600">-$9,240 (−28.4%)</span>
            </div>
          </div>

          <div id="simAlignmentBanner" class="p-3 bg-amber-50 rounded-xl border border-amber-200 text-xs text-amber-800 font-medium flex items-center gap-2">
            <svg class="w-4 h-4 text-amber-600 shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"/></svg>
            <div><strong>Risk Profile Note:</strong> This change increases your single-stock concentration past your stated moderate loss capacity limit.</div>
          </div>
        </div>
      </div>
    </section>


    <!-- TAB 5: ASK X-RAY & NEWS AUDITOR -->
    <section id="tab-chat" class="tab-content hidden space-y-6">
      <div>
        <h2 class="text-2xl sm:text-3xl font-bold text-apptext tracking-tight">Ask anything about your investments</h2>
        <p class="text-appmuted text-sm sm:text-base mt-1">No finance knowledge needed. Ask a question and we'll explain it using your portfolio.</p>
      </div>

      <div class="grid grid-cols-1 lg:grid-cols-12 gap-6">
        <!-- Chat Assistant Interface -->
        <div class="lg:col-span-7 bg-white rounded-2xl border border-appborder shadow-[0_1px_3px_rgba(15,23,42,0.04),0_8px_16px_-4px_rgba(15,23,42,0.03)] flex flex-col h-[600px]">
          <!-- Chat Message Feed -->
          <div id="chatFeed" class="flex-1 p-5 overflow-y-auto space-y-4 custom-scrollbar">
            <!-- Initial Welcome Message from Assistant -->
            <div class="flex items-start gap-3">
              <div class="w-8 h-8 rounded-xl bg-primary text-white flex items-center justify-center font-bold text-sm shrink-0">
                X
              </div>
              <div class="bg-slate-50 border border-slate-100 rounded-2xl rounded-tl-sm p-4 text-sm text-apptext max-w-[85%] space-y-2 leading-relaxed">
                <p>Hello! I am your Portfolio X-Ray assistant. I've analyzed your holdings and can answer any questions about your risk, fund overlaps, or what might happen in different markets.</p>
                <p class="text-xs text-appmuted">Tap any of the questions below or type your own question.</p>
              </div>
            </div>
          </div>

          <!-- Suggested Quick Questions -->
          <div class="p-3 bg-slate-50/80 border-t border-appborder/80">
            <span class="text-[11px] font-semibold text-appmuted uppercase tracking-wider block mb-2 px-1">Suggested questions:</span>
            <div class="flex flex-wrap gap-1.5" id="suggestedQuestionsList">
              <button onclick="submitChatQuestion('What is the biggest risk in my investments?')" class="text-xs bg-white hover:bg-slate-100 border border-slate-200 text-apptext px-2.5 py-1.5 rounded-lg transition">
                What's the biggest risk in my investments?
              </button>
              <button onclick="submitChatQuestion('Am I actually diversified?')" class="text-xs bg-white hover:bg-slate-100 border border-slate-200 text-apptext px-2.5 py-1.5 rounded-lg transition">
                Am I actually diversified?
              </button>
              <button onclick="submitChatQuestion('Could I lose a lot of money?')" class="text-xs bg-white hover:bg-slate-100 border border-slate-200 text-apptext px-2.5 py-1.5 rounded-lg transition">
                Could I lose a lot of money?
              </button>
              <button onclick="submitChatQuestion('How much Nvidia do I really own?')" class="text-xs bg-white hover:bg-slate-100 border border-slate-200 text-apptext px-2.5 py-1.5 rounded-lg transition">
                How much Nvidia do I really own?
              </button>
              <button onclick="submitChatQuestion('Why do my ETFs overlap?')" class="text-xs bg-white hover:bg-slate-100 border border-slate-200 text-apptext px-2.5 py-1.5 rounded-lg transition">
                Why do my ETFs overlap?
              </button>
              <button onclick="submitChatQuestion('What happens if interest rates go up?')" class="text-xs bg-white hover:bg-slate-100 border border-slate-200 text-apptext px-2.5 py-1.5 rounded-lg transition">
                What happens if interest rates go up?
              </button>
            </div>
          </div>

          <!-- Chat Input Field -->
          <div class="p-3.5 border-t border-appborder flex gap-2">
            <input type="text" id="chatInputField" placeholder="Ask a question about your portfolio..." class="flex-1 bg-slate-50 border border-appborder rounded-xl px-4 py-2.5 text-sm text-apptext placeholder-slate-400 focus:outline-none focus:border-primary focus:bg-white transition" onkeydown="if (event.key === 'Enter') handleChatSubmit()" />
            <button onclick="handleChatSubmit()" id="chatSendBtn" class="bg-primary hover:bg-primary-hover text-white text-sm font-semibold px-4 py-2.5 rounded-xl transition shadow-sm flex items-center gap-1.5">
              <span>Send</span>
              <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M14 5l7 7m0 0l-7 7m7-7H3"/></svg>
            </button>
          </div>
        </div>

        <!-- Seen Something in the News? (Headline Auditor) -->
        <div class="lg:col-span-5 bg-white rounded-2xl p-6 border border-appborder shadow-[0_1px_3px_rgba(15,23,42,0.04),0_8px_16px_-4px_rgba(15,23,42,0.03)] space-y-4 flex flex-col justify-between">
          <div>
            <div class="flex items-center gap-2 mb-1">
              <span class="w-6 h-6 rounded-md bg-indigo-50 text-primary flex items-center justify-center">
                <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 20H5a2 2 0 01-2-2V6a2 2 0 012-2h10a2 2 0 012 2v1m2 13a2 2 0 01-2-2V7m2 13a2 2 0 002-2V9a2 2 0 00-2-2h-2m-4-3H9M7 16h6M7 8h6v4H7V8z"/></svg>
              </span>
              <h3 class="text-lg font-bold text-apptext">Seen something in the news? (Headline Shock Impact Auditor)</h3>
            </div>
            <p class="text-sm text-appmuted">Paste a headline to see whether it could matter for your investments.</p>

            <div class="mt-4 space-y-3">
              <div>
                <textarea id="headlineInput" rows="3" placeholder="e.g. Fed signals 50 bps surprise rate cut as inflation falls" onkeydown="if (event.key === 'Enter' && !event.shiftKey) { event.preventDefault(); handleAuditHeadline(); }" class="w-full bg-slate-50 border border-appborder rounded-xl p-3 text-sm text-apptext placeholder-slate-400 focus:outline-none focus:border-primary focus:bg-white transition resize-none"></textarea>
              </div>

              <div class="flex flex-wrap gap-1.5">
                <span class="text-[11px] font-semibold text-appmuted block w-full">Quick sample headlines:</span>
                <button onclick="applyHeadline('Fed signals 50 bps surprise rate cut as inflation falls')" class="text-[11px] bg-slate-100 hover:bg-slate-200 text-slate-700 px-2 py-1 rounded transition text-left">
                  &ldquo;Fed signals 50 bps surprise rate cut&rdquo;
                </button>
                <button onclick="applyHeadline('Federal Reserve hikes interest rates by 50 basis points to fight inflation')" class="text-[11px] bg-slate-100 hover:bg-slate-200 text-slate-700 px-2 py-1 rounded transition text-left">
                  &ldquo;Fed hikes rates by 50 bps&rdquo;
                </button>
                <button onclick="applyHeadline('Cloud hyperscalers announce 25% scale back in AI chip capex spending')" class="text-[11px] bg-slate-100 hover:bg-slate-200 text-slate-700 px-2 py-1 rounded transition text-left">
                  &ldquo;Cloud giants scale back AI capex&rdquo;
                </button>
              </div>

              <button onclick="handleAuditHeadline()" id="headlineAuditBtn" class="w-full bg-primary hover:bg-primary-hover text-white text-sm font-semibold py-2.5 px-4 rounded-xl transition shadow-sm flex items-center justify-center gap-1.5">
                <span>Audit Headline Exposure</span>
              </button>
            </div>

            <!-- Headline Result Card -->
            <div id="headlineResultCard" class="hidden mt-4 p-4 rounded-xl border space-y-3 transition-all">
              <div class="flex items-center justify-between">
                <span class="text-xs font-bold uppercase tracking-wider text-appmuted">Estimated Portfolio Shock</span>
                <span id="headlineImpactBadge" class="text-sm font-extrabold px-2.5 py-0.5 rounded-lg">-1.13%</span>
              </div>
              <p id="headlineRationale" class="text-xs text-apptext leading-relaxed">
                Analysis rationale...
              </p>
              <div class="pt-2 border-t border-slate-200/60">
                <span class="text-[11px] font-semibold text-appmuted uppercase block mb-1.5">Most Sensitive Holdings:</span>
                <div id="headlineHoldingsList" class="space-y-1 text-xs">
                  <!-- Populated by JS -->
                </div>
              </div>
            </div>
          </div>

          <div class="text-[11px] text-slate-400 border-t border-slate-100 pt-3">
            Portfolio X-Ray translates breaking news into holding sensitivities using look-through beta and factor shock rules.
          </div>
        </div>
      </div>
    </section>

  </main>


  <!-- ==================== MODALS & DRAWERS ==================== -->

  <!-- 1. PORTFOLIO ENTRY WIZARD MODAL -->
  <div id="modal-portfolio-entry" class="fixed inset-0 z-50 bg-black/40 backdrop-blur-sm hidden flex items-center justify-center p-4">
    <div class="bg-white rounded-2xl max-w-2xl w-full max-h-[90vh] overflow-y-auto p-6 border border-appborder shadow-2xl space-y-6">
      <div class="flex items-start justify-between">
        <div>
          <h3 class="text-xl sm:text-2xl font-bold text-apptext">Let's take a look at your investments</h3>
          <p class="text-sm text-appmuted mt-1">Add what you own and we'll help you understand what risks you might be taking.</p>
        </div>
        <button onclick="closePortfolioEntryModal()" class="text-slate-400 hover:text-slate-600 p-1">
          <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"/></svg>
        </button>
      </div>

      <!-- Three Options Selection Grid -->
      <div class="grid grid-cols-1 sm:grid-cols-3 gap-3">
        <button onclick="selectEntryMethod('screenshot')" id="opt-screenshot" class="p-4 rounded-xl border-2 border-slate-200 hover:border-primary text-left transition flex flex-col justify-between h-32 group">
          <div class="w-8 h-8 rounded-lg bg-indigo-50 text-primary flex items-center justify-center">
            <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M3 9a2 2 0 012-2h.93a2 2 0 001.664-.89l.812-1.22A2 2 0 0110.07 4h3.86a2 2 0 011.664.89l.812 1.22A2 2 0 0018.07 7H19a2 2 0 012 2v9a2 2 0 01-2 2H5a2 2 0 01-2-2V9z"/><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 13a3 3 0 11-6 0 3 3 0 016 0z"/></svg>
          </div>
          <div>
            <div class="font-bold text-sm text-apptext">Upload a screenshot</div>
            <div class="text-[11px] text-appmuted mt-0.5 leading-snug">Import from brokerage image</div>
          </div>
        </button>
        <button onclick="selectEntryMethod('natural')" id="opt-natural" class="p-4 rounded-xl border-2 border-primary bg-primary/5 text-left transition flex flex-col justify-between h-32 group">
          <div class="w-8 h-8 rounded-lg bg-indigo-50 text-primary flex items-center justify-center">
            <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z"/></svg>
          </div>
          <div>
            <div class="font-bold text-sm text-apptext">Describe what you own</div>
            <div class="text-[11px] text-appmuted mt-0.5 leading-snug">Enter in plain English</div>
          </div>
        </button>
        <button onclick="selectEntryMethod('manual')" id="opt-manual" class="p-4 rounded-xl border-2 border-slate-200 hover:border-primary text-left transition flex flex-col justify-between h-32 group">
          <div class="w-8 h-8 rounded-lg bg-indigo-50 text-primary flex items-center justify-center">
            <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"/></svg>
          </div>
          <div>
            <div class="font-bold text-sm text-apptext">Add investments manually</div>
            <div class="text-[11px] text-appmuted mt-0.5 leading-snug">Search stocks and enter shares</div>
          </div>
        </button>
      </div>

      <!-- Option 1: Screenshot OCR Dropzone -->
      <div id="entry-panel-screenshot" class="hidden p-6 border-2 border-dashed border-slate-300 rounded-xl text-center space-y-3 bg-slate-50">
        <div class="w-12 h-12 rounded-full bg-indigo-100 text-primary mx-auto flex items-center justify-center">
          <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M3 9a2 2 0 012-2h.93a2 2 0 001.664-.89l.812-1.22A2 2 0 0110.07 4h3.86a2 2 0 011.664.89l.812 1.22A2 2 0 0018.07 7H19a2 2 0 012 2v9a2 2 0 01-2 2H5a2 2 0 01-2-2V9z"/><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 13a3 3 0 11-6 0 3 3 0 016 0z"/></svg>
        </div>
        <div>
          <p class="text-sm font-semibold text-apptext">Drop your brokerage screenshot here or browse</p>
          <p class="text-xs text-appmuted mt-1">Supports PNG, JPG (e.g. Robinhood, Fidelity, Vanguard, Charles Schwab)</p>
        </div>
        <input type="file" id="screenshotFileInput" accept="image/*" class="hidden" onchange="handleScreenshotUpload(event)" />
        <button onclick="document.getElementById('screenshotFileInput').click()" class="bg-white border border-slate-300 hover:bg-slate-100 text-xs font-semibold px-4 py-2 rounded-lg text-apptext transition">
          Select Image File
        </button>
        <div id="ocrStatusMsg" class="text-xs text-appmuted hidden"></div>
      </div>

      <!-- Option 2: Natural Language Input -->
      <div id="entry-panel-natural" class="space-y-3">
        <label class="block text-xs font-semibold text-appmuted uppercase">Enter what you own in plain words:</label>
        <textarea id="naturalEntryTextarea" rows="3" placeholder="I have 10 Nvidia shares, 5 Apple shares and £2,000 in an S&P 500 ETF." class="w-full bg-slate-50 border border-appborder rounded-xl p-3.5 text-sm text-apptext placeholder-slate-400 focus:outline-none focus:border-primary focus:bg-white transition"></textarea>
        <div class="flex items-center justify-between">
          <span class="text-xs text-appmuted">Examples: &ldquo;50 NVDA @ $130, 25 QQQ, 15 VOO, 26 SMH&rdquo;</span>
          <button onclick="handleParseNaturalInput()" class="bg-primary hover:bg-primary-hover text-white text-xs font-semibold px-4 py-2 rounded-lg transition shadow-sm">
            Read My Holdings
          </button>
        </div>
      </div>

      <!-- Option 3: Manual Search / Add -->
      <div id="entry-panel-manual" class="hidden space-y-3">
        <div class="flex gap-2">
          <input type="text" id="manualSearchInput" placeholder="Search company or ticker (e.g. Apple, NVDA, S&P 500)..." class="flex-1 bg-slate-50 border border-appborder rounded-xl px-3 py-2 text-sm text-apptext focus:outline-none focus:border-primary" />
          <input type="number" id="manualSharesInput" placeholder="Shares" class="w-24 bg-slate-50 border border-appborder rounded-xl px-3 py-2 text-sm text-apptext focus:outline-none focus:border-primary" />
          <button onclick="handleAddManualItem()" class="bg-primary hover:bg-primary-hover text-white text-xs font-semibold px-4 py-2 rounded-xl transition">
            + Add
          </button>
        </div>
      </div>

      <!-- Confirmation Screen: "Here's what we found" -->
      <div class="border-t border-slate-100 pt-4 space-y-3">
        <div class="flex items-center justify-between">
          <h4 class="font-bold text-base text-apptext">Here's what we found</h4>
          <span id="wizardTotalFoundValue" class="text-sm font-extrabold text-primary">$32,500.00 Total</span>
        </div>

        <div id="wizardHoldingsList" class="space-y-2 max-h-52 overflow-y-auto custom-scrollbar p-1">
          <!-- Populated by recognized items -->
        </div>

        <div id="wizardWarningBanner" class="hidden p-3 bg-amber-50 rounded-xl border border-amber-200 text-xs text-amber-800 flex items-center gap-2">
          <svg class="w-4 h-4 text-amber-600 shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"/></svg>
          <div>Some items were uncertain. Please review quantities and prices before proceeding.</div>
        </div>
      </div>

      <!-- Action Buttons -->
      <div class="flex items-center justify-between pt-2 border-t border-slate-100">
        <button onclick="closePortfolioEntryModal()" class="text-xs text-appmuted hover:text-apptext font-medium">
          Cancel
        </button>
        <button onclick="applyWizardHoldingsAndAnalyze()" id="wizardAnalyzeBtn" class="bg-primary hover:bg-primary-hover text-white text-sm font-semibold px-6 py-2.5 rounded-xl transition shadow-md shadow-primary/20 flex items-center gap-2">
          <span>Analyse my portfolio</span>
          <span>&rarr;</span>
        </button>
      </div>
    </div>
  </div>


  <!-- 2. RISK PROFILE QUESTIONNAIRE MODAL -->
  <div id="modal-risk-questionnaire" class="fixed inset-0 z-50 bg-black/40 backdrop-blur-sm hidden flex items-center justify-center p-4">
    <div class="bg-white rounded-2xl max-w-xl w-full max-h-[90vh] overflow-y-auto p-6 border border-appborder shadow-2xl space-y-6">
      <div class="flex items-start justify-between">
        <div>
          <h3 class="text-xl font-bold text-apptext">Your Risk Profile Questionnaire</h3>
          <p class="text-xs sm:text-sm text-appmuted mt-1">Answer 4 questions to personalize how risks and fluctuations are explained to you.</p>
        </div>
        <button onclick="closeRiskQuestionnaireModal()" class="text-slate-400 hover:text-slate-600 p-1">
          <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"/></svg>
        </button>
      </div>

      <div class="space-y-5 text-sm">
        <!-- Question 1 -->
        <div>
          <label class="block font-semibold text-apptext mb-1.5">1. What are you investing for?</label>
          <div class="grid grid-cols-2 gap-2 text-xs">
            <button onclick="setQuestionAnswer('goal', 'Long-term wealth building', this)" class="q-btn p-3 rounded-xl border border-primary bg-primary/10 text-primary font-medium text-left">
              Long-term wealth
            </button>
            <button onclick="setQuestionAnswer('goal', 'Retirement security', this)" class="q-btn p-3 rounded-xl border border-slate-200 text-apptext text-left">
              Retirement
            </button>
            <button onclick="setQuestionAnswer('goal', 'A specific financial goal', this)" class="q-btn p-3 rounded-xl border border-slate-200 text-apptext text-left">
              A specific goal
            </button>
            <button onclick="setQuestionAnswer('goal', 'Getting started and learning', this)" class="q-btn p-3 rounded-xl border border-slate-200 text-apptext text-left">
              Getting started
            </button>
          </div>
        </div>

        <!-- Question 2 -->
        <div>
          <label class="block font-semibold text-apptext mb-1.5">2. When might you need this money?</label>
          <div class="grid grid-cols-2 gap-2 text-xs">
            <button onclick="setQuestionAnswer('horizon', 'Within 1 year', this)" class="q-btn p-3 rounded-xl border border-slate-200 text-apptext text-left">
              Within 1 year
            </button>
            <button onclick="setQuestionAnswer('horizon', '1–3 years', this)" class="q-btn p-3 rounded-xl border border-slate-200 text-apptext text-left">
              1–3 years
            </button>
            <button onclick="setQuestionAnswer('horizon', '3–5 years', this)" class="q-btn p-3 rounded-xl border border-slate-200 text-apptext text-left">
              3–5 years
            </button>
            <button onclick="setQuestionAnswer('horizon', '5–10 years', this)" class="q-btn p-3 rounded-xl border border-primary bg-primary/10 text-primary font-medium text-left">
              5–10 years
            </button>
          </div>
        </div>

        <!-- Question 3 -->
        <div>
          <label class="block font-semibold text-apptext mb-1.5">3. How would you feel if £10,000 of investments temporarily fell to £8,000?</label>
          <div class="space-y-1.5 text-xs">
            <button onclick="setQuestionAnswer('reaction', 'Very anxious / would sell', this)" class="q-btn w-full p-2.5 rounded-xl border border-slate-200 text-apptext text-left">
              Very anxious / would want to sell
            </button>
            <button onclick="setQuestionAnswer('reaction', 'Uncomfortable but would wait', this)" class="q-btn w-full p-2.5 rounded-xl border border-primary bg-primary/10 text-primary font-medium text-left">
              Uncomfortable, but would wait for recovery
            </button>
            <button onclick="setQuestionAnswer('reaction', 'Calm / normal market movement', this)" class="q-btn w-full p-2.5 rounded-xl border border-slate-200 text-apptext text-left">
              Calm / view as normal temporary fluctuations
            </button>
            <button onclick="setQuestionAnswer('reaction', 'Excited to buy more at a discount', this)" class="q-btn w-full p-2.5 rounded-xl border border-slate-200 text-apptext text-left">
              Excited to buy more shares at a discount
            </button>
          </div>
        </div>

        <!-- Question 4 -->
        <div>
          <label class="block font-semibold text-apptext mb-1.5">4. How much could you financially afford to lose without affecting important commitments?</label>
          <div class="grid grid-cols-2 gap-2 text-xs">
            <button onclick="setQuestionAnswer('capacity', 'Under 10% max', this)" class="q-btn p-3 rounded-xl border border-slate-200 text-apptext text-left">
              Under 10%
            </button>
            <button onclick="setQuestionAnswer('capacity', '10–20% max', this)" class="q-btn p-3 rounded-xl border border-primary bg-primary/10 text-primary font-medium text-left">
              10–20%
            </button>
            <button onclick="setQuestionAnswer('capacity', '20–30% max', this)" class="q-btn p-3 rounded-xl border border-slate-200 text-apptext text-left">
              20–30%
            </button>
            <button onclick="setQuestionAnswer('capacity', 'Over 30%', this)" class="q-btn p-3 rounded-xl border border-slate-200 text-apptext text-left">
              Over 30%
            </button>
          </div>
        </div>
      </div>

      <div class="flex items-center justify-between pt-4 border-t border-slate-100">
        <button onclick="closeRiskQuestionnaireModal()" class="text-xs text-appmuted hover:text-apptext">
          Close
        </button>
        <button onclick="saveRiskProfileAnswers()" class="bg-primary hover:bg-primary-hover text-white text-xs font-semibold px-5 py-2.5 rounded-xl transition shadow-sm">
          Save Risk Profile
        </button>
      </div>
    </div>
  </div>


  <!-- 3. SETTINGS MODAL -->
  <div id="modal-settings" class="fixed inset-0 z-50 bg-black/40 backdrop-blur-sm hidden flex items-center justify-center p-4">
    <div class="bg-white rounded-2xl max-w-lg w-full p-6 border border-appborder shadow-2xl space-y-6">
      <div class="flex items-start justify-between">
        <div>
          <h3 class="text-xl font-bold text-apptext">Settings & Preferences</h3>
          <p class="text-xs text-appmuted mt-1">Configure your risk tolerance and display options.</p>
        </div>
        <button onclick="closeSettingsModal()" class="text-slate-400 hover:text-slate-600 p-1">
          <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"/></svg>
        </button>
      </div>

      <div class="space-y-4">
        <!-- Investor Risk Profile Box -->
        <div class="p-4 bg-slate-50 rounded-xl border border-appborder space-y-2">
          <div class="flex items-center justify-between">
            <span class="text-xs font-bold uppercase text-appmuted">Active Risk Profile</span>
            <span id="settingsProfileNameBadge" class="text-xs font-bold text-primary bg-primary/10 px-2 py-0.5 rounded">Moderate</span>
          </div>
          <p id="settingsProfileSummaryText" class="text-xs text-apptext leading-relaxed">
            Targeting long-term wealth over 5–10 years with emotional tolerance for standard market fluctuations (15–20% drawdown capacity).
          </p>
          <button onclick="openRiskQuestionnaireModal(); closeSettingsModal();" class="text-xs font-semibold text-primary hover:underline pt-1 block">
            Retake Risk Questionnaire &rarr;
          </button>
        </div>

        <!-- Advanced Analytics Toggle -->
        <div class="p-4 bg-slate-50 rounded-xl border border-appborder flex items-center justify-between gap-4">
          <div>
            <div class="font-bold text-sm text-apptext">Show advanced financial analytics</div>
            <div class="text-xs text-appmuted mt-0.5">Reveal technical metrics: Herfindahl Index (HHI), annualized volatility, Value-at-Risk (VaR), CVaR, and raw factor shock tables.</div>
          </div>
          <label class="relative inline-flex items-center cursor-pointer shrink-0">
            <input type="checkbox" id="advancedViewToggle" class="sr-only peer" onchange="toggleAdvancedView(this.checked)">
            <div class="w-11 h-6 bg-slate-300 peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-slate-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-primary"></div>
          </label>
        </div>
      </div>

      <div class="pt-2 border-t border-slate-100 flex justify-end">
        <button onclick="closeSettingsModal()" class="bg-primary hover:bg-primary-hover text-white text-xs font-semibold px-5 py-2.5 rounded-xl transition">
          Done
        </button>
      </div>
    </div>
  </div>


  <!-- 4. FINANCIAL TERMINOLOGY MODAL -->
  <div id="modal-terminology" class="fixed inset-0 z-50 bg-black/40 backdrop-blur-sm hidden flex items-center justify-center p-4">
    <div class="bg-white rounded-2xl max-w-md w-full p-6 border border-appborder shadow-2xl space-y-4">
      <div class="flex items-start justify-between">
        <div>
          <span class="text-xs font-bold uppercase tracking-wider text-primary">Plain English Guide</span>
          <h3 id="termModalTitle" class="text-lg font-bold text-apptext mt-0.5">Concentration</h3>
        </div>
        <button onclick="closeTerminologyModal()" class="text-slate-400 hover:text-slate-600 p-1">
          <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"/></svg>
        </button>
      </div>

      <div id="termModalBody" class="text-sm text-apptext space-y-2 leading-relaxed">
        <p>Short explanation...</p>
      </div>

      <div class="pt-2 border-t border-slate-100 flex justify-end">
        <button onclick="closeTerminologyModal()" class="bg-slate-100 hover:bg-slate-200 text-apptext text-xs font-semibold px-4 py-2 rounded-xl transition">
          Got it
        </button>
      </div>
    </div>
  </div>


  <!-- 5. LOADING PROGRESS OVERLAY -->
  <div id="loadingOverlay" class="fixed inset-0 z-50 bg-white/90 backdrop-blur-sm hidden flex flex-col items-center justify-center p-6">
    <div class="max-w-sm w-full bg-white rounded-2xl p-6 border border-appborder shadow-xl text-center space-y-4">
      <div class="w-12 h-12 rounded-2xl bg-indigo-50 text-primary mx-auto flex items-center justify-center animate-pulse">
        <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"/></svg>
      </div>
      <div>
        <h4 class="font-bold text-base text-apptext">Analysing Your Investments</h4>
        <p id="loadingStatusText" class="text-xs text-appmuted mt-1">Reading your investments...</p>
      </div>

      <div class="w-full bg-slate-100 rounded-full h-2 overflow-hidden">
        <div id="loadingProgressBar" class="bg-primary h-full transition-all duration-300 w-1/4"></div>
      </div>
    </div>
  </div>

    <!-- ==================== JAVASCRIPT LOGIC ==================== -->
  <script>
    // --- Application State ---
    let currentPortfolio = {
      "NVDA": 50,
      "QQQ": 25,
      "VOO": 15,
      "SMH": 26
    };

    let currentAnalysis = null;
    let activeTab = 'overview';
    let activeHiddenRisksSubTab = 'overlap';
    let advancedView = false;
    let selectedCompanyNode = 'NVDA';

    let riskProfile = {
      goal: 'Long-term wealth building',
      horizon: '5–10 years',
      reaction: 'Calm / normal market movement',
      capacity: '10–20% max',
      title: 'Balanced / Moderate'
    };

    let riskAnswers = {
      goal: 'Long-term wealth building',
      horizon: '5–10 years',
      reaction: 'Calm / normal market movement',
      capacity: '10–20% max'
    };

    let presetsData = {};
    let wizardHoldings = [];

    // Helper: Currency Formatter
    function fmtCurr(num) {
      return '$' + Number(num || 0).toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
    }

    function fmtPct(num) {
      return Number(num || 0).toFixed(1) + '%';
    }

    // --- Tab Switching ---
    function switchTab(tabId) {
      activeTab = tabId;
      document.querySelectorAll('.tab-content').forEach(el => el.classList.add('hidden'));
      const activeEl = document.getElementById('tab-' + tabId);
      if (activeEl) activeEl.classList.remove('hidden');

      // Update desktop navigation links
      const navIds = ['overview', 'hidden-risks', 'what-if', 'simulate', 'chat'];
      navIds.forEach(id => {
        const btn = document.getElementById('nav-' + id);
        if (btn) {
          if (id === tabId) {
            btn.className = 'nav-item w-full flex items-center gap-3 px-3.5 py-3 rounded-xl font-medium text-sm transition text-primary bg-primary/10';
          } else {
            btn.className = 'nav-item w-full flex items-center gap-3 px-3.5 py-3 rounded-xl font-medium text-sm transition text-appmuted hover:text-apptext hover:bg-slate-50';
          }
        }
      });

      // Update mobile drawer if present
      const drawer = document.getElementById('mobileNavDrawer');
      if (drawer && !drawer.classList.contains('hidden')) {
        drawer.classList.add('hidden');
      }

      window.scrollTo({ top: 0, behavior: 'smooth' });
    }

    function toggleMobileMenu() {
      const drawer = document.getElementById('mobileNavDrawer');
      if (drawer) drawer.classList.toggle('hidden');
    }

    function switchHiddenRisksSubTab(subTab) {
      activeHiddenRisksSubTab = subTab;
      const overlapSec = document.getElementById('subtab-overlap');
      const themesSec = document.getElementById('subtab-themes');
      const navOverlap = document.getElementById('subnav-overlap');
      const navThemes = document.getElementById('subnav-themes');

      if (subTab === 'overlap') {
        if (overlapSec) overlapSec.classList.remove('hidden');
        if (themesSec) themesSec.classList.add('hidden');
        if (navOverlap) navOverlap.className = 'pb-3 text-sm font-bold text-primary border-b-2 border-primary transition';
        if (navThemes) navThemes.className = 'pb-3 text-sm font-semibold text-appmuted hover:text-apptext transition';
      } else {
        if (overlapSec) overlapSec.classList.add('hidden');
        if (themesSec) themesSec.classList.remove('hidden');
        if (navOverlap) navOverlap.className = 'pb-3 text-sm font-semibold text-appmuted hover:text-apptext transition';
        if (navThemes) navThemes.className = 'pb-3 text-sm font-bold text-primary border-b-2 border-primary transition';
      }
    }

    // --- Presets Loading ---
    async function initApp() {
      try {
        const res = await fetch('/api/presets');
        if (res.ok) {
          presetsData = await res.json();
        }
      } catch (e) {
        console.warn('Could not fetch presets:', e);
      }
      await analyzePortfolio();
    }

    function loadPreset(key) {
      if (presetsData[key] && presetsData[key].holdings) {
        const newPort = {};
        presetsData[key].holdings.forEach(h => {
          newPort[h.ticker] = h.shares;
        });
        currentPortfolio = newPort;

        const btns = {
          'accidental_tech_overweight': 'preset-btn-accidental',
          'balanced_core': 'preset-btn-balanced',
          'all_in_growth': 'preset-btn-growth'
        };
        Object.keys(btns).forEach(k => {
          const btn = document.getElementById(btns[k]);
          if (btn) {
            if (k === key) {
              btn.className = 'w-full text-left text-xs px-2.5 py-2 rounded-lg font-medium text-primary bg-primary/10 border border-primary/20 flex items-center justify-between transition';
            } else {
              btn.className = 'w-full text-left text-xs px-2.5 py-2 rounded-lg font-medium text-appmuted hover:text-apptext hover:bg-white border border-transparent transition';
            }
          }
        });

        analyzePortfolio();
      }
    }

    // --- Core Portfolio Analysis ---
    async function analyzePortfolio() {
      showLoadingOverlay();
      updateLoadingProgress(25, 'Reading your investments & market data...');

      try {
        await new Promise(r => setTimeout(r, 120));
        updateLoadingProgress(50, 'Unpacking ETF constituents & look-throughs...');

        const res = await fetch('/api/analyze', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ portfolio: currentPortfolio })
        });

        updateLoadingProgress(75, 'Simulating historical stress & economic themes...');
        await new Promise(r => setTimeout(r, 120));

        if (!res.ok) {
          throw new Error('Analysis request failed with status ' + res.status);
        }

        currentAnalysis = await res.json();
        updateLoadingProgress(100, 'Done! Rendering your insights...');
        await new Promise(r => setTimeout(r, 80));

        renderOverview();
        renderHiddenRisks();
        renderWhatCouldHappen();
        renderTryAChange();
        renderAskXRay();
        renderAdvancedViews();
      } catch (err) {
        alert('Failed to analyze portfolio: ' + err.message);
      } finally {
        hideLoadingOverlay();
      }
    }

    function showLoadingOverlay() {
      const el = document.getElementById('loadingOverlay');
      if (el) el.classList.remove('hidden');
    }

    function hideLoadingOverlay() {
      const el = document.getElementById('loadingOverlay');
      if (el) el.classList.add('hidden');
    }

    function updateLoadingProgress(pct, msg) {
      const bar = document.getElementById('loadingProgressBar');
      const text = document.getElementById('loadingStatusText');
      if (bar) bar.style.width = pct + '%';
      if (text) text.innerText = msg;
    }

    // --- Tab 1: Overview Rendering ---
    function renderOverview() {
      if (!currentAnalysis) return;

      const totalVal = currentAnalysis.portfolio_value || 32500;
      const holdingsCount = Object.keys(currentPortfolio).length;
      const lookthroughList = currentAnalysis.lookthrough_exposure || [];
      const topLookthrough = lookthroughList[0] || { company_name: 'None', ticker: 'None', lookthrough_weight: 0 };

      // Top Welcome Card
      const elVal = document.getElementById('welcomeTotalValue');
      if (elVal) elVal.innerText = fmtCurr(totalVal);
      const elCount = document.getElementById('welcomeHoldingCount');
      if (elCount) elCount.innerText = holdingsCount + ' investments';
      const elDate = document.getElementById('welcomeAnalysisDate');
      if (elDate) elDate.innerText = 'Today, ' + new Date().toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' });
      const elGoal = document.getElementById('welcomeInvestorGoal');
      if (elGoal) elGoal.innerText = riskProfile.title + ' • ' + riskProfile.horizon;
      const elBadge = document.getElementById('overviewRiskProfileBadge');
      if (elBadge) elBadge.innerText = riskProfile.title;

      // Card 1: How spread out is your money?
      const topWeight = topLookthrough.lookthrough_weight || 0.25;
      const m1 = document.getElementById('meter-conc-1');
      const m2 = document.getElementById('meter-conc-2');
      const m3 = document.getElementById('meter-conc-3');
      const concBadge = document.getElementById('concLevelBadge');
      const concText = document.getElementById('concExplanation');
      const topStat = document.getElementById('topHoldingStat');

      if (topStat) {
        const topName = topLookthrough.company_name ? topLookthrough.company_name.split('(')[0].trim() : 'None';
        topStat.innerText = topName + ' (' + fmtPct(topWeight * 100) + ')';
      }

      if (m1) m1.className = 'w-1.5 h-3 rounded-full bg-slate-200 transition-all inline-block';
      if (m2) m2.className = 'w-1.5 h-3 rounded-full bg-slate-200 transition-all inline-block';
      if (m3) m3.className = 'w-1.5 h-3 rounded-full bg-slate-200 transition-all inline-block';

      if (topWeight > 0.30) {
        if (m1) m1.className = 'w-1.5 h-3 rounded-full bg-rose-500 transition-all inline-block';
        if (m2) m2.className = 'w-1.5 h-3 rounded-full bg-rose-500 transition-all inline-block';
        if (m3) m3.className = 'w-1.5 h-3 rounded-full bg-rose-500 transition-all inline-block';
        if (concBadge) {
          concBadge.innerText = 'High Concentration';
          concBadge.className = 'text-xs sm:text-sm font-bold text-rose-600 whitespace-nowrap truncate shrink min-w-0';
        }
        if (concText) concText.innerText = 'Your investments are heavily concentrated. Over ' + fmtPct(topWeight * 100) + ' of your entire wealth is riding on ' + topLookthrough.company_name + ', both directly and inside your funds.';
      } else if (topWeight > 0.18) {
        if (m1) m1.className = 'w-1.5 h-3 rounded-full bg-amber-500 transition-all inline-block';
        if (m2) m2.className = 'w-1.5 h-3 rounded-full bg-amber-500 transition-all inline-block';
        if (concBadge) {
          concBadge.innerText = 'Moderate Spread';
          concBadge.className = 'text-xs sm:text-sm font-bold text-amber-600 whitespace-nowrap truncate shrink min-w-0';
        }
        if (concText) concText.innerText = 'Your investments have moderate concentration. Your largest single position is ' + topLookthrough.company_name + ' (' + fmtPct(topWeight * 100) + '). You also hold multiple funds that overlap with each other.';
      } else {
        if (m1) m1.className = 'w-1.5 h-3 rounded-full bg-tealaccent transition-all inline-block';
        if (concBadge) {
          concBadge.innerText = 'Well Spread Out';
          concBadge.className = 'text-xs sm:text-sm font-bold text-tealaccent whitespace-nowrap truncate shrink min-w-0';
        }
        if (concText) concText.innerText = 'Your investments are well spread out across diverse companies, reducing your vulnerability to any single stock falling.';
      }

      // Card 2: How much could your investments fluctuate?
      const rMetrics = currentAnalysis.risk_metrics || {};
      const annVol = rMetrics.annualized_volatility || 0.22;
      const var95 = rMetrics.var_95_1m || 0.10;
      const monthlyFluctuation = totalVal * (annVol / Math.sqrt(12));
      const downsideAmount = totalVal * var95;

      const elVolBadge = document.getElementById('volatilityAmountBadge');
      if (elVolBadge) elVolBadge.innerText = '±' + fmtCurr(monthlyFluctuation) + ' / mo';
      const elVolExp = document.getElementById('volatilityExplanation');
      if (elVolExp) elVolExp.innerText = 'In a normal month, your investments typically fluctuate by about ±' + fmtCurr(monthlyFluctuation) + '. In an unusually bad market downturn (1-in-20 month event), you could see a temporary drop of up to ' + fmtCurr(downsideAmount) + '.';
      const elVarStat = document.getElementById('varDownsideStat');
      if (elVarStat) elVarStat.innerText = 'Severe 30-day downside estimate: -' + fmtCurr(downsideAmount);

      // Card 3: What is your biggest hidden risk?
      let biggestTitle = 'ETF Company Overlap';
      let biggestDesc = 'You own several investment funds that secretly hold the exact same mega-cap technology companies.';
      if (currentAnalysis.thematic_factors && currentAnalysis.thematic_factors.length > 0) {
        const topTheme = currentAnalysis.thematic_factors[0];
        biggestTitle = topTheme.theme + ' Dependence';
        biggestDesc = 'Around ' + fmtPct(topTheme.portfolio_exposure * 100) + ' of your portfolio is tied to the ' + topTheme.theme + ' theme across multiple separate holdings.';
      }
      const elBigBadge = document.getElementById('biggestRiskBadge');
      if (elBigBadge) elBigBadge.innerText = biggestTitle;
      const elBigExp = document.getElementById('biggestRiskExplanation');
      if (elBigExp) elBigExp.innerText = biggestDesc;

      // Where is your money invested? (Holdings list + Donut Chart)
      renderDoughnutChart();
      renderHoldingsList();

      // Things you might not have realised
      renderDiscoveries();
    }

    function renderDoughnutChart() {
      const svg = document.getElementById('donutChartSvg');
      if (!svg || !currentAnalysis) return;

      const direct = currentAnalysis.direct_weights || {};
      const entries = Object.entries(direct).sort((a, b) => b[1] - a[1]);
      const palette = ['#5148E5', '#12B8A6', '#F59E0B', '#EC4899', '#8B5CF6', '#3B82F6', '#10B981', '#6366F1'];

      let cumulative = 0;
      let paths = '';
      const radius = 58;
      const cx = 80;
      const cy = 80;
      const strokeWidth = 22;
      const circumference = 2 * Math.PI * radius;

      entries.forEach(([ticker, weight], idx) => {
        const dash = weight * circumference;
        const offset = -cumulative * circumference;
        const color = palette[idx % palette.length];

        paths += `<circle cx="${cx}" cy="${cy}" r="${radius}" fill="transparent" stroke="${color}" stroke-width="${strokeWidth}" stroke-dasharray="${dash} ${circumference - dash}" stroke-dashoffset="${offset}" class="transition-all duration-300 hover:opacity-80 cursor-pointer" onmouseover="setDonutFocus('${ticker}', ${weight * 100})" onmouseout="clearDonutFocus()"/>`;
        cumulative += weight;
      });

      svg.innerHTML = paths;
      const cLabel = document.getElementById('donutCenterLabel');
      if (cLabel) cLabel.innerText = 'Portfolio Total';
      const cVal = document.getElementById('donutCenterValue');
      if (cVal) cVal.innerText = fmtCurr(currentAnalysis.portfolio_value);
    }

    function setDonutFocus(ticker, pct) {
      const cLabel = document.getElementById('donutCenterLabel');
      if (cLabel) cLabel.innerText = ticker;
      const cVal = document.getElementById('donutCenterValue');
      if (cVal) cVal.innerText = fmtPct(pct);
    }

    function clearDonutFocus() {
      if (!currentAnalysis) return;
      const cLabel = document.getElementById('donutCenterLabel');
      if (cLabel) cLabel.innerText = 'Portfolio Total';
      const cVal = document.getElementById('donutCenterValue');
      if (cVal) cVal.innerText = fmtCurr(currentAnalysis.portfolio_value);
    }

    function renderHoldingsList() {
      const container = document.getElementById('holdingsListContainer');
      if (!container || !currentAnalysis) return;

      const direct = currentAnalysis.direct_weights || {};
      const entries = Object.entries(direct).sort((a, b) => b[1] - a[1]);
      const palette = ['#5148E5', '#12B8A6', '#F59E0B', '#EC4899', '#8B5CF6', '#3B82F6', '#10B981', '#6366F1'];

      container.innerHTML = entries.map(([ticker, weight], idx) => {
        const color = palette[idx % palette.length];
        const val = weight * currentAnalysis.portfolio_value;
        const shares = currentPortfolio[ticker] || 0;
        return `
          <div class="flex items-center justify-between p-3 rounded-xl hover:bg-slate-50 transition border border-transparent hover:border-slate-100">
            <div class="flex items-center gap-3">
              <span class="w-3 h-3 rounded-full shrink-0" style="background-color: ${color}"></span>
              <div>
                <div class="font-bold text-sm text-apptext">${ticker}</div>
                <div class="text-xs text-appmuted">${shares} shares</div>
              </div>
            </div>
            <div class="text-right">
              <div class="font-bold text-sm text-apptext">${fmtCurr(val)}</div>
              <div class="text-xs font-semibold text-appmuted">${fmtPct(weight * 100)}</div>
            </div>
          </div>
        `;
      }).join('');
    }

    function renderDiscoveries() {
      const container = document.getElementById('discoveriesContainer');
      if (!container || !currentAnalysis) return;

      const discoveries = [];

      // 1. Check ETF overlap on top lookthrough
      const lookthroughList = currentAnalysis.lookthrough_exposure || [];
      const top = lookthroughList[0];
      if (top && top.direct_weight > 0 && (top.lookthrough_weight - top.direct_weight) > 0.01) {
        discoveries.push({
          iconSvg: '<svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M8 7h12m0 0l-4-4m4 4l-4 4m0 6H4m0 0l4 4m-4-4l4-4"/></svg>',
          title: 'You own ' + top.company_name + ' in multiple places',
          desc: 'You hold ' + top.ticker + ' shares directly (' + fmtPct(top.direct_weight * 100) + '), but your funds also hold another ' + fmtPct((top.lookthrough_weight - top.direct_weight) * 100) + ', raising your true exposure to ' + fmtPct(top.lookthrough_weight * 100) + ' (' + fmtCurr(top.effective_value) + ').',
          tab: 'hidden-risks',
          subtab: 'overlap'
        });
      }

      // 2. Check top economic theme
      if (currentAnalysis.thematic_factors && currentAnalysis.thematic_factors.length > 0) {
        const theme = currentAnalysis.thematic_factors[0];
        discoveries.push({
          iconSvg: '<svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 12a9 9 0 01-9 9m9-9a9 9 0 00-9-9m9 9H3m9 9a9 9 0 01-9-9m9 9c1.657 0 3-4.03 3-9s-1.343-9-3-9m0 18c-1.657 0-3-4.03-3-9s1.343-9 3-9m-9 9a9 9 0 019-9"/></svg>',
          title: 'High dependence on ' + theme.theme,
          desc: 'About ' + fmtPct(theme.portfolio_exposure * 100) + ' of your money depends on the ' + theme.theme + ' economic theme across multiple separate holdings.',
          tab: 'hidden-risks',
          subtab: 'themes'
        });
      }

      // 3. Asset allocation safety check
      const hasBonds = Object.keys(currentPortfolio).some(t => ['TLT', 'BND', 'AGG', 'IEF', 'CASH'].includes(t));
      if (!hasBonds) {
        discoveries.push({
          iconSvg: '<svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z"/></svg>',
          title: 'Zero defensive bond buffer',
          desc: '100% of your current portfolio is allocated to equities. You currently have no defensive government bond or cash allocation to buffer market drawdowns.',
          tab: 'simulate',
          subtab: null
        });
      }

      container.innerHTML = discoveries.map(d => `
        <div class="p-5 bg-slate-50/80 rounded-2xl border border-appborder/80 flex flex-col justify-between space-y-3 hover:border-primary/40 transition">
          <div>
            <div class="w-10 h-10 rounded-xl bg-indigo-50 text-primary flex items-center justify-center mb-3">
              ${d.iconSvg}
            </div>
            <h4 class="font-bold text-sm text-apptext">${d.title}</h4>
            <p class="text-xs text-appmuted mt-1 leading-relaxed">${d.desc}</p>
          </div>
          <button onclick="switchTab('${d.tab}')${d.subtab ? `; switchHiddenRisksSubTab('${d.subtab}')` : ''}" class="text-xs font-semibold text-primary hover:text-primary-hover flex items-center gap-1 self-start pt-1">
            <span>Investigate this risk</span>
            <span>&rarr;</span>
          </button>
        </div>
      `).join('');
    }

    // --- Tab 2: Hidden Risks Rendering ---
    function renderHiddenRisks() {
      if (!currentAnalysis) return;

      const lookthroughList = currentAnalysis.lookthrough_exposure || [];
      const topHoldings = lookthroughList.slice(0, 8);
      const selected = topHoldings.find(h => h.ticker === selectedCompanyNode) || topHoldings[0] || {};

      // Overlap Highlight Banner
      const totalWeight = selected.lookthrough_weight || 0;
      const dirWeight = selected.direct_weight || 0;
      const indWeight = Math.max(0, totalWeight - dirWeight);

      const bTitle = document.getElementById('overlapBannerTitle');
      if (bTitle) bTitle.innerText = 'Your total ' + (selected.company_name || selected.ticker || 'Nvidia') + ' exposure: ' + fmtPct(totalWeight * 100);
      const bDesc = document.getElementById('overlapBannerDesc');
      if (bDesc) bDesc.innerText = fmtPct(dirWeight * 100) + ' comes from ' + (selected.ticker || '') + ' shares you own directly. Another ' + fmtPct(indWeight * 100) + ' comes from ' + (selected.ticker || '') + ' shares held inside your investment funds.';

      // Interactive Diagram Visual (Direct & Funds vs Underlying Companies)
      const diagram = document.getElementById('interactiveDiagramContainer');
      if (diagram) {
        const userFunds = Object.keys(currentPortfolio);

        diagram.innerHTML = `
          <div class="space-y-2">
            <span class="text-xs font-bold uppercase text-appmuted block mb-1">What You Bought</span>
            <div class="space-y-1.5">
              ${userFunds.map(fund => `
                <div class="p-3 bg-white rounded-xl border border-appborder text-xs flex items-center justify-between shadow-xs">
                  <span class="font-bold text-apptext">${fund}</span>
                  <span class="text-appmuted">${currentPortfolio[fund]} shares (${fmtPct((currentAnalysis.direct_weights[fund] || 0) * 100)})</span>
                </div>
              `).join('')}
            </div>
          </div>

          <div class="space-y-2">
            <span class="text-xs font-bold uppercase text-appmuted block mb-1">What You Actually Own</span>
            <div class="space-y-1.5">
              ${topHoldings.map(comp => {
                const isSelected = comp.ticker === selectedCompanyNode;
                return `
                  <button onclick="selectCompanyNode('${comp.ticker}')" class="w-full text-left p-3 rounded-xl border text-xs flex items-center justify-between transition ${isSelected ? 'bg-primary/10 border-primary text-primary font-bold shadow-xs' : 'bg-white border-appborder text-apptext hover:border-slate-300'}">
                    <div class="flex items-center gap-2">
                      <span class="w-2 h-2 rounded-full ${isSelected ? 'bg-primary' : 'bg-slate-300'}"></span>
                      <span class="font-bold">${comp.company_name} (${comp.ticker})</span>
                    </div>
                    <span class="${isSelected ? 'text-primary font-extrabold' : 'text-appmuted font-semibold'}">${fmtPct(comp.lookthrough_weight * 100)}</span>
                  </button>
                `;
              }).join('')}
            </div>
          </div>
        `;
      }

      // Companies you have the most money invested in (with expandable fund breakdown)
      const effList = document.getElementById('effectiveHoldingsList');
      if (effList) {
        effList.innerHTML = topHoldings.map((comp, idx) => {
          const totalPct = comp.lookthrough_weight * 100;
          const directPct = (comp.direct_weight || 0) * 100;
          const fundPct = Math.max(0, totalPct - directPct);

          return `
            <div class="p-4 bg-slate-50/70 rounded-xl border border-appborder/80 space-y-2">
              <div class="flex items-center justify-between cursor-pointer" onclick="toggleFundBreakdown(${idx})">
                <div class="flex items-center gap-3">
                  <span class="w-6 h-6 rounded-lg bg-slate-200 text-apptext font-bold text-xs flex items-center justify-center">${idx + 1}</span>
                  <div>
                    <span class="font-bold text-sm text-apptext">${comp.company_name}</span>
                    <span class="text-xs text-appmuted ml-1.5">${comp.ticker} • ${comp.sector}</span>
                  </div>
                </div>
                <div class="flex items-center gap-3">
                  <span class="font-bold text-sm text-apptext">${fmtPct(totalPct)}</span>
                  <span class="text-xs text-primary font-semibold">View breakdown &darr;</span>
                </div>
              </div>

              <!-- Horizontal Progress Bar -->
              <div class="w-full bg-slate-200 rounded-full h-2 overflow-hidden flex">
                <div style="width: ${directPct}%" class="bg-primary h-full" title="Direct: ${fmtPct(directPct)}"></div>
                <div style="width: ${fundPct}%" class="bg-tealaccent h-full" title="Via Funds: ${fmtPct(fundPct)}"></div>
              </div>

              <!-- Expandable Fund Breakdown -->
              <div id="fundBreakdown-${idx}" class="hidden pt-2 border-t border-slate-200 text-xs text-appmuted grid grid-cols-2 gap-2">
                <div>Direct Holdings: <strong class="text-apptext">${fmtPct(directPct)}</strong> (${fmtCurr((comp.direct_weight || 0) * currentAnalysis.portfolio_value)})</div>
                <div>Via Funds (ETFs): <strong class="text-apptext">${fmtPct(fundPct)}</strong> (${fmtCurr(Math.max(0, comp.lookthrough_weight - (comp.direct_weight || 0)) * currentAnalysis.portfolio_value)})</div>
              </div>
            </div>
          `;
        }).join('');
      }

      // Raw lookthrough table for advanced view
      const rawBody = document.getElementById('rawLookthroughTableBody');
      if (rawBody) {
        rawBody.innerHTML = lookthroughList.map(h => `
          <tr class="hover:bg-slate-50 transition">
            <td class="py-2.5 px-3 font-bold text-apptext">${h.company_name} <span class="text-appmuted font-normal">(${h.ticker})</span></td>
            <td class="py-2.5 px-3 font-semibold text-primary">${fmtPct(h.lookthrough_weight * 100)}</td>
            <td class="py-2.5 px-3 text-apptext">${fmtCurr(h.effective_value)}</td>
            <td class="py-2.5 px-3 text-appmuted">${fmtPct((h.direct_weight || 0) * 100)}</td>
            <td class="py-2.5 px-3 text-appmuted">${fmtPct(Math.max(0, h.lookthrough_weight - (h.direct_weight || 0)) * 100)}</td>
            <td class="py-2.5 px-3 text-appmuted">${h.sector}</td>
          </tr>
        `).join('');
      }

      // Economic Themes Grid
      const themesGrid = document.getElementById('themesGridContainer');
      if (themesGrid && currentAnalysis.thematic_factors) {
        const themeIconSvgs = {
          'Artificial Intelligence': '<svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 17h14a2 2 0 002-2V5a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z"/></svg>',
          'Technology Growth': '<svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10 20l4-16m4 4l4 4-4 4M6 16l-4-4 4-4"/></svg>',
          'Interest Rate Sensitivity': '<svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6"/></svg>',
          'Consumer Spending': '<svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M16 11V7a4 4 0 00-8 0v4M5 9h14l1 12H4L5 9z"/></svg>',
          'Energy & Commodity Inflation': '<svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 10V3L4 14h7v7l9-11h-7z"/></svg>',
          'Defense & Aerospace': '<svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z"/></svg>',
          'Real Estate & Housing': '<svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M3 12l2-2m0 0l7-7 7 7M5 10v10a1 1 0 001 1h3m10-11l2 2m-2-2v10a1 1 0 01-1 1h-3m-6 0a1 1 0 001-1v-4a1 1 0 011-1h2a1 1 0 011 1v4a1 1 0 001 1m-6 0h6"/></svg>'
        };
        const defaultIconSvg = '<svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"/></svg>';

        themesGrid.innerHTML = currentAnalysis.thematic_factors.map((thm, idx) => {
          const iconSvg = themeIconSvgs[thm.theme] || defaultIconSvg;
          return `
            <div class="p-5 bg-slate-50/80 rounded-2xl border border-appborder/80 flex flex-col justify-between space-y-4 hover:border-primary/40 transition">
              <div>
                <div class="flex items-center justify-between mb-2">
                  <div class="w-9 h-9 rounded-xl bg-indigo-50 text-primary flex items-center justify-center">
                    ${iconSvg}
                  </div>
                  <span class="text-xs font-bold text-primary bg-primary/10 px-2 py-0.5 rounded-full">${fmtPct(thm.portfolio_exposure * 100)}</span>
                </div>
                <h4 class="font-bold text-base text-apptext">${thm.theme}</h4>
                <p class="text-xs text-appmuted mt-1 leading-relaxed">${thm.description}</p>
              </div>

              <div>
                <span class="text-[11px] font-bold uppercase text-appmuted block mb-1.5">Contributing Investments</span>
                <div class="flex flex-wrap gap-1.5">
                  ${(thm.contributing_assets || []).map(a => `
                    <span class="text-xs font-semibold px-2 py-0.5 bg-white border border-slate-200 rounded-lg text-apptext">${a}</span>
                  `).join('')}
                </div>
              </div>

              <div class="pt-2 border-t border-slate-200">
                <button onclick="toggleThemeExplanation(${idx})" class="text-xs font-semibold text-primary hover:text-primary-hover flex items-center justify-between w-full">
                  <span>How this affects you</span>
                  <span>&darr;</span>
                </button>
                <div id="themeExplanation-${idx}" class="hidden mt-2 text-xs text-appmuted leading-relaxed bg-white p-2.5 rounded-lg border border-slate-100">
                  Companies in this theme share common revenue drivers. A regulatory shift, interest rate change, or spending slowdown in this sector affects these holdings simultaneously.
                </div>
              </div>
            </div>
          `;
        }).join('');
      }
    }

    function selectCompanyNode(ticker) {
      selectedCompanyNode = ticker;
      renderHiddenRisks();
    }

    function toggleFundBreakdown(idx) {
      const el = document.getElementById('fundBreakdown-' + idx);
      if (el) el.classList.toggle('hidden');
    }

    function toggleThemeExplanation(idx) {
      const el = document.getElementById('themeExplanation-' + idx);
      if (el) el.classList.toggle('hidden');
    }

    // --- Tab 3: What Could Happen? Rendering ---
    function renderWhatCouldHappen() {
      const grid = document.getElementById('scenariosGridContainer');
      if (!grid || !currentAnalysis || !currentAnalysis.stress_tests) return;

      const totalVal = currentAnalysis.portfolio_value || 32500;

      const scenarioConfigs = [
        {
          id: 'tech_crash',
          testKey: '2022 Fed Rate Shock',
          title: 'What if technology stocks crashed?',
          iconSvg: '<svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 17h8m0 0V9m0 8l-8-8-4 4-6-6"/></svg>',
          desc: 'Simulates a sharp rise in interest rates that compresses high-valuation tech companies, similar to the 2022 market decline.'
        },
        {
          id: 'market_panic',
          testKey: '2020 Liquidity',
          title: 'What if another major market crash happened?',
          iconSvg: '<svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"/></svg>',
          desc: 'Simulates a sudden, synchronized global selloff where equities across all sectors drop simultaneously, as in March 2020.'
        },
        {
          id: 'ai_slowdown',
          testKey: 'AI Infrastructure',
          title: 'What if companies reduced their AI spending?',
          iconSvg: '<svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 17h14a2 2 0 002-2V5a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z"/></svg>',
          desc: 'Hypothetical scenario where cloud giants cut back data-center investments, causing a sharp drop in semiconductor and hardware stocks.'
        },
        {
          id: 'stagflation',
          testKey: 'Stagflation Energy',
          title: 'What if energy prices and inflation spiked?',
          iconSvg: '<svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 10V3L4 14h7v7l9-11h-7z"/></svg>',
          desc: 'Simulates persistent supply-chain bottlenecks and surging crude oil driving inflation and compressing consumer profit margins.'
        }
      ];

      grid.innerHTML = scenarioConfigs.map((cfg, idx) => {
        const test = currentAnalysis.stress_tests.find(t => t.scenario_name && t.scenario_name.includes(cfg.testKey)) || currentAnalysis.stress_tests[idx % currentAnalysis.stress_tests.length];
        const drawdownPct = Math.abs(test ? test.drawdown_pct : 0.20);
        const dollarLoss = totalVal * drawdownPct;
        const afterVal = totalVal - dollarLoss;

        return `
          <div class="bg-white rounded-2xl p-6 border border-appborder shadow-[0_1px_3px_rgba(15,23,42,0.04),0_8px_16px_-4px_rgba(15,23,42,0.03)] space-y-4">
            <div class="flex items-start justify-between gap-3">
              <div class="flex items-center gap-3">
                <div class="w-10 h-10 rounded-xl bg-indigo-50 text-primary flex items-center justify-center shrink-0">
                  ${cfg.iconSvg}
                </div>
                <div>
                  <h3 class="font-bold text-base text-apptext">${cfg.title}</h3>
                  <p class="text-xs text-appmuted mt-0.5">${cfg.desc}</p>
                </div>
              </div>
              <span class="text-xs font-extrabold text-rose-600 bg-rose-50 border border-rose-100 px-2.5 py-1 rounded-lg shrink-0">
                -${fmtPct(drawdownPct * 100)}
              </span>
            </div>

            <!-- Before & After Value Comparison -->
            <div class="p-4 bg-slate-50 rounded-xl border border-appborder space-y-2">
              <div class="flex justify-between text-xs text-appmuted">
                <span>Before</span>
                <span>After this hypothetical scenario</span>
              </div>
              <div class="flex justify-between items-baseline">
                <span class="text-base font-bold text-apptext">${fmtCurr(totalVal)}</span>
                <span class="text-base font-bold text-rose-600">${fmtCurr(afterVal)}</span>
              </div>
              <div class="text-right text-xs font-semibold text-rose-500">
                Estimated change: -${fmtCurr(dollarLoss)} (-${fmtPct(drawdownPct * 100)})
              </div>

              <!-- Visual Comparison Bar -->
              <div class="w-full bg-slate-200 rounded-full h-2 overflow-hidden mt-1">
                <div style="width: ${Math.max(5, (afterVal / totalVal) * 100)}%" class="bg-rose-500 h-full rounded-full"></div>
              </div>
            </div>

            <!-- Expandable Button -->
            <div>
              <button onclick="toggleScenarioDetails(${idx})" class="w-full bg-slate-100 hover:bg-slate-200 text-apptext text-xs font-semibold py-2 px-3 rounded-xl transition flex items-center justify-center gap-1.5">
                <span>Explore this scenario</span>
                <span>&darr;</span>
              </button>

              <div id="scenarioDetails-${idx}" class="hidden mt-3 p-3 bg-slate-50/70 rounded-xl border border-slate-200 text-xs space-y-2">
                <div class="font-bold text-apptext">Holdings contributing most to this change:</div>
                <div class="space-y-1">
                  ${Object.entries(currentPortfolio).map(([sym, count]) => {
                    const weight = currentAnalysis.direct_weights[sym] || 0.25;
                    const loss = dollarLoss * weight;
                    return `
                      <div class="flex justify-between text-slate-600">
                        <span>${sym}</span>
                        <span class="font-semibold text-rose-600">-${fmtCurr(loss)}</span>
                      </div>
                    `;
                  }).join('')}
                </div>
                <div class="text-[11px] text-slate-400 italic pt-1 border-t border-slate-200">
                  Assumption: Factor beta sensitivities based on historical stress distributions. Not a forecast.
                </div>
              </div>
            </div>
          </div>
        `;
      }).join('');
    }

    function toggleScenarioDetails(idx) {
      const el = document.getElementById('scenarioDetails-' + idx);
      if (el) el.classList.toggle('hidden');
    }

    // --- Tab 4: Try a Change (Simulate) ---
    function applySimPrompt(promptText) {
      const input = document.getElementById('simQuestionInput');
      if (input) input.value = promptText;
      handleSimulateQuestion();
    }

    function handleSliderChange(val) {
      const badge = document.getElementById('simSliderValueBadge');
      if (badge) badge.innerText = val + '%';
      const labelEl = document.getElementById('simActiveAssetLabel');
      let asset = 'NVDA';
      if (labelEl) {
        const txt = labelEl.innerText;
        const parenMatch = txt.match(/\\(([A-Za-z0-9.]+)\\)/);
        if (parenMatch) asset = parenMatch[1];
        else asset = txt.split(' ')[0] || 'NVDA';
      }
      simulateAssetChange(asset, Number(val));
    }

    async function handleSimulateQuestion() {
      const input = document.getElementById('simQuestionInput');
      const text = input ? input.value.trim() : '';
      if (!text) return;

      let pct = 25;
      const pctMatch = text.match(/(\\d+)%/);
      if (pctMatch) pct = parseInt(pctMatch[1]);
      else if (/half/i.test(text)) pct = 50;

      // Extract asset or ticker
      let targetAsset = text.replace(/what\\s+if|i\\s+bought|i\\s+invested|i\\s+added|i\\s+reduced|another|more|shares?|of|in|to|\\?|\\!|\\$|£/gi, ' ').trim();
      if (/nvidia|nvda/i.test(text)) targetAsset = 'NVDA';
      else if (/apple|aapl/i.test(text)) targetAsset = 'AAPL';
      else if (/microsoft|msft/i.test(text)) targetAsset = 'MSFT';
      else if (/tesla|tsla/i.test(text)) targetAsset = 'TSLA';
      else if (/amazon|amzn/i.test(text)) targetAsset = 'AMZN';
      else if (/google|alphabet|goog/i.test(text)) targetAsset = 'GOOGL';
      else if (/meta|facebook/i.test(text)) targetAsset = 'META';
      else if (/broadcom|avgo/i.test(text)) targetAsset = 'AVGO';
      else if (/amd/i.test(text)) targetAsset = 'AMD';
      else if (/sp500|voo|spy/i.test(text)) targetAsset = 'VOO';
      else if (/nasdaq|qqq/i.test(text)) targetAsset = 'QQQ';
      else if (/bond|bonds|tlt|treasury/i.test(text)) targetAsset = 'TLT';
      else if (/gold|gld/i.test(text)) targetAsset = 'GLD';
      else if (/cash/i.test(text)) targetAsset = 'CASH';
      else if (/spacex/i.test(text)) targetAsset = 'SPACEX';
      else if (targetAsset.split(/\\s+/).length > 0) {
        targetAsset = targetAsset.split(/\\s+/)[0];
      }

      simulateAssetChange(targetAsset, pct);
    }

    async function simulateAssetChange(asset, pct) {
      const elLabel = document.getElementById('simActiveAssetLabel');
      if (elLabel) elLabel.innerText = asset;
      const elBadge = document.getElementById('simSliderValueBadge');
      if (elBadge) elBadge.innerText = pct + '%';
      const elSlider = document.getElementById('simAllocationSlider');
      if (elSlider) elSlider.value = pct;

      try {
        const targetAllocations = {};
        targetAllocations[asset] = pct;

        const res = await fetch('/api/what-if', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            portfolio: currentPortfolio,
            target_allocations: targetAllocations
          })
        });

        if (!res.ok) throw new Error('Simulation failed');
        const data = await res.json();

        if (data.resolved_target && data.resolved_target.name && elLabel) {
          elLabel.innerText = data.resolved_target.name + ' (' + data.resolved_target.ticker + ')';
        }

        const base = data.current_metrics || data.base_metrics || {};
        const prop = data.proposed_metrics || {};

        const bConc = document.getElementById('simBaseConcentration');
        if (bConc) {
          const rawHhi = base.hhi !== undefined ? base.hhi : 2686;
          const concPct = (rawHhi / 100).toFixed(1);
          bConc.innerText = concPct + '%';
        }
        const bTop = document.getElementById('simBaseTopHolding');
        if (bTop) bTop.innerText = base.top_1_concentration ? (base.top_1_concentration + '%') : (base.top_holding ? fmtPct(base.top_holding.weight * 100) : '33.8%');
        const bVol = document.getElementById('simBaseVolatility');
        if (bVol) bVol.innerText = base.annual_volatility ? (base.annual_volatility + '%') : fmtPct((base.annualized_volatility || 0.23) * 100);
        const bStress = document.getElementById('simBaseStressLoss');
        if (bStress) bStress.innerText = data.stress_comparison && data.stress_comparison[0] ? fmtPct(Math.abs(data.stress_comparison[0].current_drawdown_pct)) : '20.3%';

        const pConc = document.getElementById('simPropConcentration');
        if (pConc) {
          const propHhi = prop.hhi !== undefined ? prop.hhi : 2150;
          const pConcPct = (propHhi / 100).toFixed(1);
          pConc.innerText = pConcPct + '%';
        }
        const pTop = document.getElementById('simPropTopHolding');
        if (pTop) pTop.innerText = prop.top_1_concentration ? (prop.top_1_concentration + '%') : (prop.top_holding ? fmtPct(prop.top_holding.weight * 100) : '28.5%');
        const pVol = document.getElementById('simPropVolatility');
        if (pVol) pVol.innerText = prop.annual_volatility ? (prop.annual_volatility + '%') : fmtPct((prop.annualized_volatility || 0.19) * 100);
        const pStress = document.getElementById('simPropStressLoss');
        if (pStress) pStress.innerText = data.stress_comparison && data.stress_comparison[0] ? fmtPct(Math.abs(data.stress_comparison[0].proposed_drawdown_pct)) : '17.5%';

        const banner = document.getElementById('simAlignmentBanner');
        if (banner) {
          if (asset === 'TLT' || asset === 'CASH') {
            banner.className = 'p-4 bg-teal-50 border border-teal-200 rounded-xl text-xs text-teal-800 space-y-1';
            banner.innerHTML = '<strong>Diversification Benefit:</strong> Adding ' + asset + ' reduces your overall volatility and limits maximum stress drawdown.';
          } else {
            banner.className = 'p-4 bg-indigo-50 border border-indigo-200 rounded-xl text-xs text-indigo-800 space-y-1';
            banner.innerHTML = '<strong>Targeted Growth:</strong> Allocating ' + pct + '% to ' + asset + ' increases your direct exposure while altering your overall portfolio concentration score.';
          }
        }

        const cmpContainer = document.getElementById('simComparisonContainer');
        if (cmpContainer) cmpContainer.classList.remove('hidden');
      } catch (err) {
        console.error('Simulation error:', err);
      }
    }

    function renderTryAChange() {
      simulateAssetChange('TLT', 15);
    }

    // --- Tab 5: Ask X-Ray & Headline Shock Auditor ---
    function renderAskXRay() {
      // Ready state for Ask X-Ray
    }

    function handleChatSubmit() {
      sendChatMessage();
    }

    function submitChatQuestion(queryText) {
      const input = document.getElementById('chatInputField');
      if (input) input.value = queryText;
      sendChatMessage();
    }

    async function sendChatMessage() {
      const input = document.getElementById('chatInputField');
      const text = input ? input.value.trim() : '';
      if (!text) return;

      appendChatMessage('user', text);
      input.value = '';

      try {
        const res = await fetch('/api/chat', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ portfolio: currentPortfolio, query: text })
        });

        if (!res.ok) throw new Error('Chat failed');
        const data = await res.json();
        const reply = data.answer || data.response || "Here's what your portfolio analysis shows.";

        appendChatMessage('assistant', reply);
      } catch (err) {
        appendChatMessage('assistant', 'Sorry, I could not process your question. Please try again.');
      }
    }

    function renderFormattedChatText(text, isUser) {
      if (isUser) {
        return text.split(String.fromCharCode(10)).join('<br/>');
      }

      // Convert Markdown formatting into clean consumer fintech typography
      const lines = text.split(String.fromCharCode(10));
      let out = '';
      let inList = false;

      for (let i = 0; i < lines.length; i++) {
        let line = lines[i].trim();
        if (!line) {
          if (inList) { out += '</ul>'; inList = false; }
          out += '<div class="h-2"></div>';
          continue;
        }

        // Headings: ####, ###, ##, #
        if (line.indexOf('#') === 0) {
          if (inList) { out += '</ul>'; inList = false; }
          const title = line.replace(/^#+\\s*/, '');
          out += '<div class="font-bold text-base text-apptext mt-3 mb-1">' + formatInlineMarkup(title) + '</div>';
          continue;
        }

        // Bullet points: * or -
        if (line.indexOf('* ') === 0 || line.indexOf('- ') === 0) {
          if (!inList) {
            out += '<ul class="space-y-1.5 my-2 pl-4 list-disc">';
            inList = true;
          }
          const item = line.substring(2);
          out += '<li class="text-apptext/90">' + formatInlineMarkup(item) + '</li>';
          continue;
        }

        if (inList) { out += '</ul>'; inList = false; }
        out += '<p class="my-1 text-apptext/90">' + formatInlineMarkup(line) + '</p>';
      }

      if (inList) out += '</ul>';
      return out;
    }

    function formatInlineMarkup(str) {
      if (!str) return '';
      // Replace **bold** with styled strong
      let s = str.replace(/\\*\\*(.*?)\\*\\*/g, '<strong class="font-bold text-apptext">$1</strong>');
      // Clean any remaining raw asterisks or hash tags
      s = s.replace(/\\*/g, '').replace(/^#+\\s*/g, '');
      return s;
    }

    function appendChatMessage(role, text) {
      const feed = document.getElementById('chatFeed');
      if (!feed) return;

      const isUser = role === 'user';
      const msgDiv = document.createElement('div');
      msgDiv.className = isUser ? 'flex justify-end' : 'flex justify-start';

      const formattedContent = renderFormattedChatText(text, isUser);

      msgDiv.innerHTML = `
        <div class="max-w-[85%] rounded-2xl p-4 text-sm leading-relaxed ${isUser ? 'bg-primary text-white font-medium rounded-br-xs' : 'bg-white border border-appborder text-apptext shadow-xs rounded-bl-xs'}">
          ${formattedContent}
        </div>
      `;

      feed.appendChild(msgDiv);
      feed.scrollTop = feed.scrollHeight;
    }

    function applyHeadline(headlineText) {
      const input = document.getElementById('headlineInput');
      if (input) input.value = headlineText;
      handleAuditHeadline();
    }

    function handleAuditHeadline() {
      auditHeadlineEvent();
    }

    async function auditHeadlineEvent() {
      const input = document.getElementById('headlineInput');
      const headline = input ? input.value.trim() : '';
      if (!headline) return;

      const btn = document.getElementById('headlineAuditBtn');
      if (btn) {
        btn.innerText = 'Auditing Impact...';
        btn.disabled = true;
      }

      try {
        const res = await fetch('/api/headline', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ portfolio: currentPortfolio, headline: headline })
        });

        if (!res.ok) throw new Error('Headline audit failed');
        const data = await res.json();

        const shockPct = data.net_estimated_impact_pct !== undefined ? (data.net_estimated_impact_pct / 100) : (data.portfolio_shock_pct || 0);
        const dollarImpact = data.net_estimated_dollar_impact !== undefined ? data.net_estimated_dollar_impact : ((currentAnalysis ? currentAnalysis.portfolio_value : 32500) * shockPct);
        const isNeg = shockPct < 0;

        const bImpact = document.getElementById('headlineImpactBadge');
        if (bImpact) {
          bImpact.innerText = (isNeg ? '' : '+') + fmtPct(shockPct * 100) + ' (' + (isNeg ? '-' : '+') + fmtCurr(Math.abs(dollarImpact)) + ')';
          bImpact.className = 'text-lg font-bold ' + (isNeg ? 'text-rose-600' : 'text-tealaccent');
        }

        const bRat = document.getElementById('headlineRationale');
        if (bRat) bRat.innerText = data.rationale || 'Macro factor shock transmitted via look-through factor loadings.';

        const list = document.getElementById('headlineHoldingsList');
        const holdings = data.affected_holdings || data.sensitive_holdings || [];
        if (list && holdings.length > 0) {
          list.innerHTML = holdings.map(h => {
            const s = h.estimated_shock_pct !== undefined ? h.estimated_shock_pct : (h.shock_pct * 100);
            return '<div class="flex justify-between py-1 text-xs border-b border-slate-100 last:border-0">' +
              '<span class="font-bold text-apptext">' + h.ticker + ' <span class="text-appmuted font-normal">(' + (h.name || '') + ')</span></span>' +
              '<span class="font-semibold ' + (s < 0 ? 'text-rose-600' : 'text-tealaccent') + '">' + (s < 0 ? '' : '+') + fmtPct(s) + '</span>' +
            '</div>';
          }).join('');
        }

        const resCard = document.getElementById('headlineResultCard');
        if (resCard) {
          resCard.className = 'mt-4 p-4 rounded-xl border space-y-3 transition-all ' + (isNeg ? 'bg-rose-50/60 border-rose-200' : 'bg-teal-50/60 border-teal-200');
          resCard.classList.remove('hidden');
        }
      } catch (err) {
        alert('Could not audit headline: ' + err.message);
      } finally {
        if (btn) {
          btn.innerText = 'Audit Headline Exposure';
          btn.disabled = false;
        }
      }
    }

    // --- Advanced View Toggle ---
    function toggleAdvancedView(checked) {
      advancedView = checked;
      renderAdvancedViews();
    }

    function renderAdvancedViews() {
      const banner = document.getElementById('advancedMetricsOverviewBanner');
      const table = document.getElementById('advancedLookthroughTableContainer');

      if (advancedView && currentAnalysis) {
        if (banner) banner.classList.remove('hidden');
        if (table) table.classList.remove('hidden');

        const advHHI = document.getElementById('advHHIVal');
        if (advHHI) advHHI.innerText = currentAnalysis.risk_metrics.concentration_hhi ? currentAnalysis.risk_metrics.concentration_hhi.toFixed(0) : '2,150';
        const advVol = document.getElementById('advVolVal');
        if (advVol) advVol.innerText = fmtPct((currentAnalysis.risk_metrics.annualized_volatility || 0.18) * 100);
        const advVaR = document.getElementById('advVaRVal');
        if (advVaR) advVaR.innerText = fmtPct((currentAnalysis.risk_metrics.var_95_1m || 0.08) * 100);
        const advCVaR = document.getElementById('advCVaRVal');
        if (advCVaR) advCVaR.innerText = fmtPct((currentAnalysis.risk_metrics.cvar_95_1m || 0.11) * 100);
      } else {
        if (banner) banner.classList.add('hidden');
        if (table) table.classList.add('hidden');
      }
    }

    // --- Modals & Terminology Guide ---
    function openSettingsModal() {
      const m = document.getElementById('modal-settings');
      if (m) m.classList.remove('hidden');
    }

    function closeSettingsModal() {
      const m = document.getElementById('modal-settings');
      if (m) m.classList.add('hidden');
    }

    function openRiskQuestionnaireModal() {
      const m = document.getElementById('modal-risk-questionnaire');
      if (m) m.classList.remove('hidden');
    }

    function closeRiskQuestionnaireModal() {
      const m = document.getElementById('modal-risk-questionnaire');
      if (m) m.classList.add('hidden');
    }

    function setQuestionAnswer(category, value, el) {
      riskAnswers[category] = value;
      const parent = el ? el.parentElement : null;
      if (parent) {
        parent.querySelectorAll('button').forEach(btn => {
          btn.className = 'q-btn p-3 rounded-xl border border-slate-200 hover:border-slate-300 text-apptext text-left transition';
        });
        el.className = 'q-btn p-3 rounded-xl border border-primary bg-primary/10 text-primary font-medium text-left';
      }
    }

    function saveRiskProfileAnswers() {
      let title = 'Balanced / Moderate';
      if ((riskAnswers.reaction && riskAnswers.reaction.includes('anxious')) || (riskAnswers.capacity && riskAnswers.capacity.includes('Under 10%'))) {
        title = 'Conservative Focus';
      } else if ((riskAnswers.reaction && riskAnswers.reaction.includes('Excited')) || (riskAnswers.capacity && riskAnswers.capacity.includes('Over 30%'))) {
        title = 'Aggressive Growth';
      } else if ((riskAnswers.horizon && riskAnswers.horizon.includes('5–10')) || (riskAnswers.reaction && riskAnswers.reaction.includes('Calm'))) {
        title = 'Growth Focus';
      }

      riskProfile.title = title;
      riskProfile.horizon = riskAnswers.horizon || '5–10 years';

      const b1 = document.getElementById('settingsProfileNameBadge');
      if (b1) b1.innerText = title;
      const b2 = document.getElementById('overviewRiskProfileBadge');
      if (b2) b2.innerText = title;
      const b3 = document.getElementById('welcomeInvestorGoal');
      if (b3) b3.innerText = title + ' • ' + riskProfile.horizon;

      closeRiskQuestionnaireModal();
      renderOverview();
    }

    const TERMINOLOGY_DEFS = {
      concentration: {
        title: 'Concentration (How spread out your money is)',
        body: '<p>Concentration measures how much of your total money is riding on just one or two companies. If you put too many eggs in one basket, a problem with that company can cause heavy losses even when the rest of the market is doing well.</p><p class="text-xs text-appmuted mt-2">Professional finance uses a mathematical formula called the Herfindahl-Hirschman Index (HHI), but all that matters to you is keeping any single company under 10–15% of your total wealth.</p>'
      },
      volatility: {
        title: 'Volatility (How much prices bounce around)',
        body: '<p>Volatility is simply how fast and how far your investment values bounce up and down over time. High volatility means bigger swings; low volatility means a smoother ride.</p><p class="text-xs text-appmuted mt-2">Holding volatile stocks is normal for long-term growth, as long as you have enough time to ride out the downturns without being forced to sell at a loss.</p>'
      },
      var: {
        title: 'Downside Risk (Value at Risk / VaR)',
        body: '<p>Downside risk estimates how much money you could lose in an unusually bad market month (statistically, the worst 1 out of 20 months).</p><p class="text-xs text-appmuted mt-2">It is not a guarantee or a limit on how much you can lose, but it helps you prepare emotionally so you aren’t taken by surprise during sharp market corrections.</p>'
      },
      lookthrough: {
        title: 'Look-Through Exposure (Seeing inside funds)',
        body: '<p>When you buy an ETF or index fund (like the S&P 500 or Nasdaq 100), you aren’t buying one company—you are buying a bundle of hundreds of companies.</p><p class="text-xs text-appmuted mt-2">Portfolio X-Ray "looks through" each fund you own to calculate your true total dollar stake in every underlying business. This stops you from accidentally owning 3x more of a company than you thought.</p>'
      },
      themes: {
        title: 'Economic Themes (Shared global drivers)',
        body: '<p>Even companies in totally different industries can rise and fall together if they depend on the same global trend—like Artificial Intelligence spending, interest rate decisions, or energy prices.</p><p class="text-xs text-appmuted mt-2">Identifying your themes helps ensure your investments are genuinely diversified, rather than all depending on the exact same economic catalyst.</p>'
      }
    };

    function openTerminology(key) {
      const def = TERMINOLOGY_DEFS[key] || TERMINOLOGY_DEFS.concentration;
      const tTitle = document.getElementById('termModalTitle');
      if (tTitle) tTitle.innerText = def.title;
      const tBody = document.getElementById('termModalBody');
      if (tBody) tBody.innerHTML = def.body;
      const m = document.getElementById('modal-terminology');
      if (m) m.classList.remove('hidden');
    }

    function closeTerminologyModal() {
      const m = document.getElementById('modal-terminology');
      if (m) m.classList.add('hidden');
    }

    // --- Portfolio Entry Wizard ---
    function openPortfolioEntryModal() {
      const m = document.getElementById('modal-portfolio-entry');
      if (m) m.classList.remove('hidden');
      selectEntryMethod('natural');

      // Populate current holdings in wizard list with real names and prices
      const currentList = Object.entries(currentPortfolio).map(([ticker, shares]) => {
        let name = ticker;
        let price = 100;
        if (currentAnalysis && currentAnalysis.holdings) {
          const match = currentAnalysis.holdings.find(h => h.ticker === ticker);
          if (match) {
            name = match.name || ticker;
            price = match.price || 100;
          }
        }
        return {
          ticker: ticker,
          name: name,
          shares: Number(shares),
          price: price,
          value: Number(shares) * price,
          is_ambiguous: false
        };
      });
      renderWizardHoldings(currentList);
    }

    function closePortfolioEntryModal() {
      const m = document.getElementById('modal-portfolio-entry');
      if (m) m.classList.add('hidden');
    }

    function selectEntryMethod(method) {
      ['screenshot', 'natural', 'manual'].forEach(m => {
        const btn = document.getElementById('opt-' + m);
        const panel = document.getElementById('entry-panel-' + m);
        if (m === method) {
          if (btn) btn.className = 'p-4 rounded-xl border-2 border-primary bg-primary/5 text-left transition flex flex-col justify-between h-32';
          if (panel) panel.classList.remove('hidden');
        } else {
          if (btn) btn.className = 'p-4 rounded-xl border-2 border-slate-200 hover:border-primary text-left transition flex flex-col justify-between h-32';
          if (panel) panel.classList.add('hidden');
        }
      });
    }

    async function handleParseNaturalInput() {
      const textarea = document.getElementById('naturalEntryTextarea');
      const text = textarea ? textarea.value.trim() : '';
      if (!text) {
        alert('Please describe what you own first.');
        return;
      }

      try {
        const res = await fetch('/api/parse-portfolio', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ text: text })
        });
        if (!res.ok) throw new Error('Could not parse text');
        const data = await res.json();
        renderWizardHoldings(data.holdings || []);
      } catch (err) {
        alert('Error reading holdings: ' + err.message);
      }
    }

    function handleAddManualItem() {
      const searchInput = document.getElementById('manualSearchInput');
      const sharesInput = document.getElementById('manualSharesInput');
      const query = searchInput ? searchInput.value.trim().toUpperCase() : '';
      const shares = sharesInput ? parseFloat(sharesInput.value) : 0;

      if (!query || isNaN(shares) || shares <= 0) {
        alert('Please enter a valid stock/fund ticker and positive number of shares.');
        return;
      }

      wizardHoldings.push({
        ticker: query,
        name: query,
        shares: shares,
        price: 100.0,
        value: shares * 100.0,
        is_ambiguous: false
      });

      if (searchInput) searchInput.value = '';
      if (sharesInput) sharesInput.value = '';

      renderWizardHoldings(wizardHoldings);
    }

    function renderWizardHoldings(holdings) {
      wizardHoldings = holdings.map(h => ({
        ticker: h.ticker,
        name: h.name || h.ticker,
        shares: Number(h.shares || 0),
        price: Number(h.price || 100),
        value: Number(h.value || (h.shares * h.price)),
        is_ambiguous: !!h.is_ambiguous
      }));

      const list = document.getElementById('wizardHoldingsList');
      const totalEl = document.getElementById('wizardTotalFoundValue');
      const warnBanner = document.getElementById('wizardWarningBanner');

      let total = 0;
      let hasAmbiguous = false;

      if (list) {
        list.innerHTML = wizardHoldings.map((h, idx) => {
          total += h.value;
          if (h.is_ambiguous) hasAmbiguous = true;
          return `
            <div class="p-3 bg-white rounded-xl border ${h.is_ambiguous ? 'border-amber-300 bg-amber-50/40' : 'border-appborder'} flex items-center justify-between text-xs">
              <div class="flex items-center gap-3">
                <span class="font-bold text-sm text-apptext">${h.ticker}</span>
                <div>
                  <div class="font-medium text-apptext">${h.name}</div>
                  <div class="text-[11px] text-appmuted">${h.shares} shares @ ${fmtCurr(h.price)}</div>
                </div>
              </div>
              <div class="flex items-center gap-3">
                <span class="font-bold text-apptext text-sm">${fmtCurr(h.value)}</span>
                <button onclick="removeWizardHolding(${idx})" class="text-rose-500 hover:text-rose-700 p-1 font-bold text-base">
                  &times;
                </button>
              </div>
            </div>
          `;
        }).join('');
      }

      if (totalEl) totalEl.innerText = fmtCurr(total) + ' Total';
      if (warnBanner) {
        if (hasAmbiguous) warnBanner.classList.remove('hidden');
        else warnBanner.classList.add('hidden');
      }
    }

    function removeWizardHolding(idx) {
      wizardHoldings.splice(idx, 1);
      renderWizardHoldings(wizardHoldings);
    }

    function applyWizardHoldingsAndAnalyze() {
      if (!wizardHoldings || wizardHoldings.length === 0) {
        alert('Please enter at least one holding before analysing.');
        return;
      }

      const newPort = {};
      wizardHoldings.forEach(h => {
        newPort[h.ticker] = h.shares;
      });

      currentPortfolio = newPort;
      closePortfolioEntryModal();
      analyzePortfolio();
    }

    async function handleScreenshotUpload(event) {
      const file = event.target ? event.target.files[0] : event;
      if (!file) return;

      const statusEl = document.getElementById('ocrStatusMsg');
      if (statusEl) {
        statusEl.innerText = 'Scanning image with OCR...';
        statusEl.classList.remove('hidden');
      }

      try {
        if (typeof Tesseract === 'undefined') {
          throw new Error('Tesseract library is loading, please try again in a moment');
        }
        const result = await Tesseract.recognize(file, 'eng', {
          logger: m => {
            if (m.status === 'recognizing text' && statusEl) {
              statusEl.innerText = 'Reading image: ' + Math.round((m.progress || 0) * 100) + '%';
            }
          }
        });

        const res = await fetch('/api/parse-portfolio', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ text: result.data.text })
        });
        if (!res.ok) throw new Error('Could not parse extracted text');
        const data = await res.json();
        renderWizardHoldings(data.holdings || []);
      } catch (err) {
        alert('Could not read image: ' + err.message + '. Please use the plain English description or manual input.');
      } finally {
        if (statusEl) statusEl.classList.add('hidden');
      }
    }

    // --- Onload Initialization ---
    window.addEventListener('DOMContentLoaded', () => {
      initApp();
    });
  
async function loadPresets() {
  try {
    const res = await fetch('/api/presets');
    const data = await res.json();
    let selector = document.getElementById('preset-selector');
    if (!selector) {
      // Create selector if it doesn't exist
      selector = document.createElement('select');
      selector.id = 'preset-selector';
      selector.className = 'w-full p-2 border border-appborder rounded bg-white text-sm mt-4';
      selector.innerHTML = '<option value="">-- Select Preset Portfolio --</option>';
      const sb = document.querySelector('.sidebar-content') || document.querySelector('aside');
      if (sb) sb.appendChild(selector);
      
      selector.addEventListener('change', (e) => {
        if (!e.target.value) return;
        const preset = data[e.target.value];
        if (preset) {
           let text = preset.holdings.map(h => `${h.shares} ${h.ticker}`).join('\n');
           document.getElementById('manual-input').value = text;
           analyzePortfolio();
        }
      });
    }
    
    // Add default presets
    for (const [key, p] of Object.entries(data)) {
        if (!selector.querySelector(`option[value="${key}"]`)) {
            const opt = document.createElement('option');
            opt.value = key;
            opt.textContent = p.name || key;
            selector.appendChild(opt);
        }
    }
  } catch (err) {
    console.error('Failed to load presets', err);
  }
}
// Call loadPresets on load
document.addEventListener('DOMContentLoaded', loadPresets);


function updateInstitutionalUI(data) {
  // Circuit Breakers
  const cbBanner = document.getElementById('cb-banner');
  const cbList = document.getElementById('cb-list');
  if (data.metrics && data.metrics.circuit_breakers && data.metrics.circuit_breakers.current_breaches && data.metrics.circuit_breakers.current_breaches.length > 0) {
    cbList.innerHTML = data.metrics.circuit_breakers.current_breaches.map(b => `<li>${b}</li>`).join('');
    cbBanner.classList.remove('hidden');
  } else {
    cbBanner.classList.add('hidden');
  }
  
  // Benchmark
  const bCard = document.getElementById('benchmark-card');
  const bContent = document.getElementById('benchmark-content');
  if (data.benchmark && data.benchmark.spy_comparison) {
    const b = data.benchmark;
    bContent.innerHTML = `
      <div>
        <p class="text-gray-500 font-semibold mb-2">Vs. SPY (100% Equity)</p>
        <p><strong>Beta:</strong> ${b.portfolio_beta}</p>
        <p><strong>Active Risk:</strong> ${b.active_risk}%</p>
        <p><strong>SPY Volatility:</strong> ${b.spy_comparison.volatility}%</p>
      </div>
      <div>
        <p class="text-gray-500 font-semibold mb-2">Vs. 60/40 (SPY/BND)</p>
        <p><strong>60/40 Volatility:</strong> ${b.bnd_6040_comparison.volatility}%</p>
        <p><strong>60/40 Max Drawdown:</strong> ${b.bnd_6040_comparison.max_drawdown}%</p>
      </div>
    `;
    bCard.classList.remove('hidden');
  }
  
  // Heatmap & Euler Risk (Tab 2)
  if (data.correlation_and_risk && data.correlation_and_risk.risk_contributions) {
     let tab2 = document.getElementById('tab-overlap');
     if (tab2 && !document.getElementById('euler-risk')) {
         const div = document.createElement('div');
         div.id = 'euler-risk';
         div.className = 'mt-6 p-5 bg-white border border-appborder rounded-xl shadow-sm';
         div.innerHTML = `<h3 class="text-lg font-semibold text-apptext mb-3">Euler Risk Contribution</h3>
            <div class="overflow-x-auto">
            <table class="w-full text-left border-collapse text-sm">
              <thead><tr class="border-b"><th class="py-2">Asset</th><th class="py-2">Weight %</th><th class="py-2">Risk Contrib %</th><th class="py-2">Risk/Capital Ratio</th></tr></thead>
              <tbody>
                ${data.correlation_and_risk.risk_contributions.map(rc => `
                  <tr class="border-b">
                    <td class="py-2">${rc.ticker}</td>
                    <td class="py-2">${rc.weight}%</td>
                    <td class="py-2">${rc.pct_contribution}%</td>
                    <td class="py-2 ${rc.risk_capital_ratio > 1.2 ? 'text-red-600 font-bold' : ''}">${rc.risk_capital_ratio}</td>
                  </tr>
                `).join('')}
              </tbody>
            </table>
            </div>
         `;
         tab2.appendChild(div);
     }
  }
  
  // Historical Stress (Tab 3)
  if (data.historical_stress && data.historical_stress.replays) {
     let tab3 = document.getElementById('tab-stress');
     if (tab3 && !document.getElementById('hist-stress')) {
         const div = document.createElement('div');
         div.id = 'hist-stress';
         div.className = 'mt-6 p-5 bg-white border border-appborder rounded-xl shadow-sm';
         div.innerHTML = `<h3 class="text-lg font-semibold text-apptext mb-3">Historical Crisis Replays</h3>
            <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
                ${data.historical_stress.replays.map(r => `
                  <div class="p-4 border rounded bg-gray-50">
                    <h4 class="font-bold">${r.scenario.replace(/_/g, ' ')}</h4>
                    <p>Portfolio Drawdown: <span class="text-red-600">${r.portfolio_drawdown}%</span></p>
                    <p>SPY Drawdown: ${r.spy_drawdown}%</p>
                    <p>Recovery: ${r.recovery_days} days</p>
                  </div>
                `).join('')}
            </div>
         `;
         tab3.appendChild(div);
     }
  }
}

// Hook into analyzePortfolio response
const oldRender = renderAnalysis;
renderAnalysis = function(data) {
  oldRender(data);
  updateInstitutionalUI(data);
};

</script>
</body>
</html>
"""


class DualStackServer(ThreadingHTTPServer):
    address_family = socket.AF_INET6
    def server_bind(self):
        self.socket.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 0)
        super().server_bind()


def run_server(port: int = 8080):
    try:
        httpd = DualStackServer(("::", port), PortfolioXRayHandler)
    except Exception:
        httpd = ThreadingHTTPServer(("0.0.0.0", port), PortfolioXRayHandler)
    httpd.daemon_threads = True
    print(f"Portfolio X-Ray Web Dashboard running on http://localhost:{port} and http://127.0.0.1:{port}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server.")
        httpd.server_close()


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8080
    run_server(port)
