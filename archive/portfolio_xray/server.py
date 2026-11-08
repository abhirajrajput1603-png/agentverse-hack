"""Portfolio X-Ray: Zero-dependency local web server & interactive dashboard.

Runs using Python standard library http.server on port 8080 (or specified port).
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
from pathlib import Path
from typing import Dict, List, Any, Optional

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

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
)
from portfolio_xray.agent import ask_risk_analyst

STOP_WORDS = {
    "have", "and", "in", "of", "shares", "the", "i", "a", "my", "at", "with",
    "worth", "about", "holding", "holdings", "own", "owns", "to", "for", "is",
    "are", "stock", "stocks", "total", "portfolio", "position", "positions",
    "equity", "equities", "hold", "holds", "got", "put", "into"
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
    cleaned = re.sub(r"[^a-zA-Z0-9.]", "", raw).strip().lower()
    if cleaned in STOP_WORDS or len(cleaned) < 1:
        return None
    if cleaned in COMPANY_SYNONYMS:
        return COMPANY_SYNONYMS[cleaned]
    upper = cleaned.upper()
    if upper in DEFAULT_PRICES or (upper.isalpha() and 2 <= len(upper) <= 10):
        return upper
    return None


def parse_holdings_from_text(text: str) -> List[Dict[str, Any]]:
    holdings = []
    found = set()

    # 1. Dollar values: e.g. "$10,000 in NVDA"
    for m in re.finditer(r"\$([0-9,]+(?:\.[0-9]+)?)\s*(?:in|of|worth of)?\s*([A-Za-z.]{1,20})", text, re.I):
        val = float(m.group(1).replace(",", ""))
        t = clean_sym(m.group(2))
        if t and t not in found:
            p = DEFAULT_PRICES.get(t, 100.0)
            sh = round(val / p, 2)
            if sh > 0:
                found.add(t)
                name = ASSET_METADATA.get(t, {}).get("name", t)
                holdings.append({"ticker": t, "name": name, "shares": sh, "price": p})

    # 2. Table / line-by-line format: "NVDA 50 $130"
    for line in text.split("\n"):
        parts = line.strip().split()
        if len(parts) >= 2:
            t = clean_sym(parts[0])
            if t and t not in found:
                try:
                    sh = float(parts[1].replace(",", "").replace("$", ""))
                    pr = float(parts[2].replace(",", "").replace("$", "")) if len(parts) > 2 else DEFAULT_PRICES.get(t, 100.0)
                    if sh > 0:
                        found.add(t)
                        name = ASSET_METADATA.get(t, {}).get("name", t)
                        holdings.append({"ticker": t, "name": name, "shares": sh, "price": pr})
                        continue
                except ValueError:
                    pass

    # 3. Phrases: "50 shares of NVDA at $130" or "NVDA: 50"
    patterns = [
        r"([0-9,]+(?:\.[0-9]+)?)\s*(?:shares?\s*(?:of)?)?\s*([A-Za-z.]{1,20})(?:\s*(?:@|at|[\$])\s*[\$]?([0-9,]+(?:\.[0-9]+)?))?",
        r"([A-Za-z.]{1,20})\s*[:\-]?\s*([0-9,]+(?:\.[0-9]+)?)\s*shares?(?:\s*(?:@|at|[\$])\s*[\$]?([0-9,]+(?:\.[0-9]+)?))?"
    ]
    for pat in patterns:
        for m in re.finditer(pat, text, re.I):
            if pat == patterns[1]:
                sym_str, sh_str = m.group(1), m.group(2)
            else:
                sh_str, sym_str = m.group(1), m.group(2)
            pr_str = m.group(3) if len(m.groups()) >= 3 else None
            t = clean_sym(sym_str)
            if t and t not in found:
                try:
                    sh = float(sh_str.replace(",", ""))
                    pr = float(pr_str.replace(",", "")) if pr_str else DEFAULT_PRICES.get(t, 100.0)
                    if sh > 0:
                        found.add(t)
                        name = ASSET_METADATA.get(t, {}).get("name", t)
                        holdings.append({"ticker": t, "name": name, "shares": sh, "price": pr})
                except ValueError:
                    pass

    return holdings


def parse_portfolio_payload(data: Dict[str, Any]) -> Portfolio:
    raw_holdings = data.get("holdings", [])
    holdings = []
    for h in raw_holdings:
        ticker = str(h.get("ticker", "")).upper().strip()
        shares = float(h.get("shares", 0.0))
        price = float(h.get("price", 0.0))
        if ticker and shares > 0 and price > 0:
            holdings.append(Holding(ticker=ticker, shares=shares, price=price))

    profile = data.get("user_profile", {
        "risk_tolerance": "Moderate",
        "horizon_years": 7,
        "loss_capacity": "Moderate (15-20% max drawdown)",
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

            response = {
                "metrics": metrics,
                "exposures": [
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
                ],
                "thematic": [
                    {
                        "theme": t["theme"],
                        "weight": round(t["weight"] * 100.0, 2),
                        "top_assets": [{"ticker": a["ticker"], "weight": round(a["weight"] * 100.0, 2)} for a in t["top_assets"]],
                    }
                    for t in thematic
                ],
                "stress_tests": stress,
            }
            encoded = json.dumps(response).encode("utf-8")
            self._set_headers("application/json", len(encoded))
            self.wfile.write(encoded)

        elif parsed.path == "/api/what-if":
            portfolio = parse_portfolio_payload(data.get("portfolio", {}))
            target_allocations = data.get("target_allocations", {})
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

        else:
            self.send_error(HTTPStatus.NOT_FOUND, "API route not found")


HTML_DASHBOARD = """<!DOCTYPE html>
<html lang="en" class="dark">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Portfolio X-Ray | AI-Powered Portfolio Risk Intelligence</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <!-- Client-side OCR via Tesseract.js -->
  <script src="https://cdn.jsdelivr.net/npm/tesseract.js@5/dist/tesseract.min.js"></script>
  <script>
    tailwind.config = {
      darkMode: 'class',
      theme: {
        extend: {
          colors: {
            brand: { 50: '#eef2ff', 500: '#6366f1', 600: '#4f46e5', 700: '#4338ca' },
            slate: { 850: '#151e2e', 900: '#0f172a', 950: '#080d1a' }
          }
        }
      }
    }
  </script>
  <style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700&family=Inter:wght@300;400;500;600;700&display=swap');
    body { font-family: 'Inter', sans-serif; }
    .font-mono { font-family: 'JetBrains Mono', monospace; }

    /* Segmented Slider Scale Styles */
    .gauge-container {
      position: relative;
      width: 100%;
      height: 10px;
      margin-top: 10px;
      margin-bottom: 4px;
    }
    .gauge-track {
      display: flex;
      gap: 3.5px;
      width: 100%;
      height: 7.5px;
    }
    .gauge-segment {
      flex: 1;
      height: 100%;
      border-radius: 9999px;
      border: 1px solid transparent;
      position: relative;
      overflow: hidden;
    }
    .gauge-green {
      border-color: rgba(16, 185, 129, 0.45);
      background: repeating-linear-gradient(
        -45deg,
        rgba(16, 185, 129, 0.12),
        rgba(16, 185, 129, 0.12) 3px,
        rgba(16, 185, 129, 0.38) 3px,
        rgba(16, 185, 129, 0.38) 6px
      );
    }
    .gauge-yellow {
      border-color: rgba(245, 158, 11, 0.45);
      background: repeating-linear-gradient(
        -45deg,
        rgba(245, 158, 11, 0.12),
        rgba(245, 158, 11, 0.12) 3px,
        rgba(245, 158, 11, 0.38) 3px,
        rgba(245, 158, 11, 0.38) 6px
      );
    }
    .gauge-red {
      border-color: rgba(244, 63, 94, 0.45);
      background: repeating-linear-gradient(
        -45deg,
        rgba(244, 63, 94, 0.12),
        rgba(244, 63, 94, 0.12) 3px,
        rgba(244, 63, 94, 0.38) 3px,
        rgba(244, 63, 94, 0.38) 6px
      );
    }
    .gauge-slider-marker {
      position: absolute;
      top: -3.5px;
      width: 4px;
      height: 14.5px;
      background-color: #ffffff;
      border-radius: 9999px;
      box-shadow: 0 0 8px 1.5px rgba(255, 255, 255, 0.95), 0 2px 4px rgba(0, 0, 0, 0.8);
      transform: translateX(-50%);
      transition: left 0.5s cubic-bezier(0.4, 0, 0.2, 1);
      pointer-events: none;
      z-index: 10;
    }
  </style>
</head>
<body class="bg-slate-950 text-slate-100 min-h-screen flex flex-col selection:bg-brand-500 selection:text-white">

  <!-- Top Navbar -->
  <header class="border-b border-slate-800 bg-slate-900/90 backdrop-blur sticky top-0 z-40 px-6 py-3.5 flex items-center justify-between">
    <div class="flex items-center gap-3">
      <div class="w-9 h-9 rounded-lg bg-gradient-to-tr from-brand-600 to-cyan-500 flex items-center justify-center font-bold text-white shadow-lg shadow-brand-500/20">
        XR
      </div>
      <div>
        <div class="flex items-center gap-2">
          <span class="font-bold text-lg tracking-tight">Portfolio X-Ray</span>
          <span class="text-xs px-2 py-0.5 rounded-full bg-brand-500/20 text-brand-300 font-mono font-medium border border-brand-500/30">Personalized Risk Intelligence</span>
        </div>
        <p class="text-xs text-slate-400">Autonomous Portfolio Risk Intelligence & Look-Through Audit</p>
      </div>
    </div>
    <div class="flex items-center gap-4">
      <div class="text-right hidden sm:block">
        <span class="text-xs text-slate-400 block font-mono">AgentVerse Finance Track</span>
        <span class="text-xs text-emerald-400 flex items-center gap-1.5 justify-end">
          <span class="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse"></span> Risk Engine Ready
        </span>
      </div>
    </div>
  </header>

  <!-- Main Container -->
  <main class="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 py-6 space-y-6">

    <!-- Personalized Portfolio Builder & Custom Input Section -->
    <div class="bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-lg space-y-4">
      <div class="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3 border-b border-slate-800 pb-3">
        <div>
          <div class="flex items-center gap-2">
            <span class="inline-flex items-center justify-center w-6 h-6 rounded-md bg-brand-500/20 text-brand-400 font-bold text-xs">1</span>
            <h2 class="text-base font-bold text-white tracking-tight">Your Portfolio</h2>
            <span class="text-[11px] px-2 py-0.5 rounded-full bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 font-mono">Personalized</span>
          </div>
          <p class="text-xs text-slate-400 mt-0.5">Describe your investments in natural language, upload a screenshot from your broker, or edit holdings directly.</p>
        </div>
        
        <!-- Investor Risk Profile Slider (5 Sub-Sections: Low to High) -->
        <div class="bg-slate-950 border border-slate-800 rounded-xl px-4 py-2.5 self-stretch sm:self-auto min-w-[280px] sm:min-w-[340px]">
          <div class="flex items-center justify-between gap-2 mb-1.5">
            <span class="text-xs text-slate-400 font-mono font-medium">Risk Profile:</span>
            <span id="riskProfileBadge" class="text-[11px] font-semibold px-2 py-0.5 rounded-full bg-amber-500/20 text-amber-300 border border-amber-500/30 font-sans transition">
              Moderate (Max 15-20% drawdown)
            </span>
          </div>
          <!-- 5-Step Range Slider -->
          <div class="relative flex flex-col gap-1.5">
            <input 
              type="range" 
              id="riskProfileSlider" 
              min="1" 
              max="5" 
              step="1" 
              value="3" 
              oninput="handleRiskSliderChange(this.value)" 
              onchange="handleRiskSliderChange(this.value)" 
              class="w-full h-2 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-brand-500 focus:outline-none"
            />
            <!-- 5 Sub-Section Tick Labels (Low to High) -->
            <div class="flex justify-between items-center text-[10px] font-mono text-slate-400 px-0.5 select-none">
              <span class="cursor-pointer hover:text-emerald-400 transition" onclick="setRiskSlider(1)" title="Very Low: Capital Preservation">Very Low</span>
              <span class="cursor-pointer hover:text-teal-400 transition" onclick="setRiskSlider(2)" title="Low: Conservative Income">Low</span>
              <span class="cursor-pointer hover:text-amber-300 transition" onclick="setRiskSlider(3)" title="Moderate: Balanced Growth">Moderate</span>
              <span class="cursor-pointer hover:text-orange-400 transition" onclick="setRiskSlider(4)" title="High: Growth Appreciation">High</span>
              <span class="cursor-pointer hover:text-rose-400 transition" onclick="setRiskSlider(5)" title="Very High: Aggressive / Speculative">Very High</span>
            </div>
          </div>
        </div>
      </div>

      <!-- Input Options: Prompt Text vs Image Upload -->
      <div class="grid grid-cols-1 lg:grid-cols-12 gap-4">
        <!-- Left Column: Prompt Input -->
        <div class="lg:col-span-7 flex flex-col space-y-2">
          <div class="flex items-center justify-between">
            <label class="text-xs font-semibold text-slate-300 flex items-center gap-1.5">
              <span>✍️ Describe in Plain English or Paste Brokerage Table</span>
            </label>
            <span class="text-[10px] text-slate-500 font-mono hidden sm:inline">e.g. "50 NVDA @ $130, 25 QQQ @ $480, $5,000 Apple"</span>
          </div>
          <textarea id="portfolioPromptInput" rows="3" onkeydown="if ((event.metaKey || event.ctrlKey) && event.key === 'Enter') handleParsePrompt()" class="w-full bg-slate-950 border border-slate-800 rounded-xl p-3 text-xs text-slate-200 placeholder-slate-600 focus:outline-none focus:border-brand-500 font-mono transition resize-none" placeholder="Type holdings: e.g. 'I hold 50 shares of Nvidia, 25 shares of QQQ, and 15 shares of Apple' or paste rows from Robinhood/Fidelity..."></textarea>
          <div class="flex flex-wrap items-center justify-between gap-2 pt-1">
            <div class="flex items-center gap-2">
              <button id="parsePromptBtn" onclick="handleParsePrompt()" class="bg-brand-600 hover:bg-brand-500 text-white text-xs font-semibold px-4 py-2 rounded-lg transition shadow-md shadow-brand-600/20 flex items-center gap-1.5 cursor-pointer">
                <span>⚡ Parse & Update Holdings</span>
              </button>
              <button onclick="clearHoldings()" class="text-slate-400 hover:text-rose-400 text-xs px-2.5 py-1.5 transition cursor-pointer">
                Clear
              </button>
            </div>
            <!-- Quick Presets -->
            <div class="flex items-center gap-1.5">
              <span class="text-[11px] text-slate-500">Demo presets:</span>
              <div id="presetButtons" class="flex gap-1.5"></div>
            </div>
          </div>
        </div>

        <!-- Right Column: Screenshot Upload -->
        <div class="lg:col-span-5 flex flex-col space-y-2">
          <label class="text-xs font-semibold text-slate-300 flex items-center gap-1.5">
            <span>📷 Or Upload Portfolio Screenshot</span>
          </label>
          <div id="dropZone" onclick="document.getElementById('fileInput').click()" ondragover="handleDragOver(event)" ondragleave="handleDragLeave(event)" ondrop="handleDrop(event)" class="border-2 border-dashed border-slate-800 hover:border-brand-500/60 bg-slate-950/60 rounded-xl p-4 flex flex-col items-center justify-center text-center cursor-pointer transition h-[115px] group">
            <input type="file" id="fileInput" accept="image/*" class="hidden" onchange="handleImageUpload(event)" />
            <div id="dropZoneContent" class="space-y-1">
              <div class="text-xl group-hover:scale-110 transition duration-200">📸</div>
              <div class="text-xs text-slate-300 font-medium">Click or drag screenshot here</div>
              <div class="text-[10px] text-slate-500">Robinhood, Fidelity, Schwab, Trading212, Vanguard</div>
            </div>
            <div id="ocrLoading" class="hidden flex flex-col items-center space-y-1.5">
              <div class="w-5 h-5 border-2 border-brand-500 border-t-transparent rounded-full animate-spin"></div>
              <span class="text-xs text-brand-300 font-medium" id="ocrStatusText">Scanning image with OCR...</span>
            </div>
          </div>
          <div id="ocrSuccessBanner" class="hidden text-[11px] text-emerald-400 bg-emerald-500/10 border border-emerald-500/20 px-2.5 py-1 rounded-lg"></div>
        </div>
      </div>

      <!-- Current Holdings Editor Table -->
      <div id="holdingsEditorCard" class="bg-slate-950/80 border border-slate-800/80 rounded-xl p-3.5 space-y-3">
        <div class="flex items-center justify-between">
          <div class="flex items-center gap-2">
            <span class="text-xs font-bold text-slate-300 uppercase tracking-wider">Active Holdings Editor</span>
            <span id="holdingCountBadge" class="text-[10px] px-2 py-0.5 rounded-full bg-slate-800 text-slate-300 font-mono">0 assets</span>
          </div>
          <div class="flex items-center gap-2">
            <button onclick="addHoldingRow()" class="text-xs text-brand-300 hover:text-white bg-brand-500/20 hover:bg-brand-500/30 px-3 py-1.5 rounded-lg border border-brand-500/40 transition flex items-center gap-1 cursor-pointer font-medium">
              <span>+ Add Asset</span>
            </button>
            <button onclick="runAnalysisFromEditor()" class="text-xs font-bold text-emerald-200 bg-emerald-600 hover:bg-emerald-500 px-4 py-1.5 rounded-lg transition flex items-center gap-1.5 shadow-md shadow-emerald-600/20 cursor-pointer">
              <span>🚀 Recalculate Risk Analytics</span>
            </button>
          </div>
        </div>
        
        <div class="overflow-x-auto max-h-56 overflow-y-auto">
          <table class="w-full text-left border-collapse text-xs">
            <thead>
              <tr class="border-b border-slate-800 text-[11px] font-mono text-slate-500">
                <th class="py-2 px-3">Stock Name</th>
                <th class="py-2 px-3">Shares</th>
                <th class="py-2 px-3">Share Price ($)</th>
                <th class="py-2 px-3">Market Value ($)</th>
                <th class="py-2 px-3 text-right">Action</th>
              </tr>
            </thead>
            <tbody id="holdingsEditorBody">
              <!-- Dynamically populated rows -->
            </tbody>
          </table>
        </div>
      </div>
    </div>

    <!-- Metrics Stat Cards with Slider Scales -->
    <div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4" id="statCards">
      <!-- Total Capital Card -->
      <div class="bg-slate-900 border border-slate-800 rounded-xl p-4 flex flex-col justify-between">
        <div>
          <span class="text-xs text-slate-400 font-mono flex items-center gap-1">
            Total Capital <button type="button" onclick="openInfoModal('capital')" class="inline-flex items-center justify-center w-4 h-4 rounded-full bg-slate-800 hover:bg-brand-500/30 text-slate-400 hover:text-brand-300 transition text-[11px] font-sans font-bold cursor-pointer border border-slate-700/60 hover:border-brand-500/50" title="Click for explanation">ⓘ</button>
          </span>
          <div class="text-2xl font-bold font-mono text-white mt-1" id="statVal">$0.00</div>
        </div>
        <div class="mt-4 pt-2 border-t border-slate-800/60">
          <span class="text-xs text-slate-500 font-mono block" id="statHoldingSummary">Live Book Value</span>
        </div>
      </div>

      <!-- Concentration Card with Slider Scale -->
      <div class="bg-slate-900 border border-slate-800 rounded-xl p-4 flex flex-col justify-between">
        <div>
          <span class="text-xs text-slate-400 font-mono flex items-center gap-1">
            Concentration (HHI) <button type="button" onclick="openInfoModal('hhi')" class="inline-flex items-center justify-center w-4 h-4 rounded-full bg-slate-800 hover:bg-brand-500/30 text-slate-400 hover:text-brand-300 transition text-[11px] font-sans font-bold cursor-pointer border border-slate-700/60 hover:border-brand-500/50" title="Click for explanation">ⓘ</button>
          </span>
          <div class="flex items-baseline mt-1 gap-2">
            <span class="text-2xl font-bold font-mono text-white" id="statHHI">0.0</span>
            <span class="text-xs font-semibold px-2 py-0.5 rounded-full" id="statTierBadge">Calculating...</span>
          </div>
        </div>
        <div class="mt-3">
          <div class="gauge-container">
            <div class="gauge-track">
              <div class="gauge-segment gauge-green" title="Diversified (HHI &lt; 1,500)"></div>
              <div class="gauge-segment gauge-yellow" title="Moderate (HHI 1,500 - 2,500)"></div>
              <div class="gauge-segment gauge-red" title="High Concentration (HHI &gt; 2,500)"></div>
            </div>
            <div class="gauge-slider-marker" id="statHHIMarker" style="left: 33%;"></div>
          </div>
          <div class="flex justify-between text-[10px] text-slate-500 font-mono mt-1 px-0.5">
            <span>Low</span>
            <span>Moderate</span>
            <span>High</span>
          </div>
        </div>
      </div>

      <!-- Annualized Volatility Card with Slider Scale -->
      <div class="bg-slate-900 border border-slate-800 rounded-xl p-4 flex flex-col justify-between">
        <div>
          <span class="text-xs text-slate-400 font-mono flex items-center gap-1">
            Annualized Volatility <button type="button" onclick="openInfoModal('volatility')" class="inline-flex items-center justify-center w-4 h-4 rounded-full bg-slate-800 hover:bg-brand-500/30 text-slate-400 hover:text-brand-300 transition text-[11px] font-sans font-bold cursor-pointer border border-slate-700/60 hover:border-brand-500/50" title="Click for explanation">ⓘ</button>
          </span>
          <div class="flex items-baseline mt-1 gap-2">
            <span class="text-2xl font-bold font-mono text-white" id="statVol">0.0%</span>
            <span class="text-xs font-semibold px-2 py-0.5 rounded-full" id="statVolTierBadge">Moderate</span>
          </div>
        </div>
        <div class="mt-3">
          <div class="gauge-container">
            <div class="gauge-track">
              <div class="gauge-segment gauge-green" title="Low Volatility (&lt; 14%)"></div>
              <div class="gauge-segment gauge-yellow" title="Moderate Volatility (14% - 22%)"></div>
              <div class="gauge-segment gauge-red" title="High Volatility (&gt; 22%)"></div>
            </div>
            <div class="gauge-slider-marker" id="statVolMarker" style="left: 50%;"></div>
          </div>
          <div class="flex justify-between text-[10px] text-slate-500 font-mono mt-1 px-0.5">
            <span>&lt;14%</span>
            <span>14-22%</span>
            <span>&gt;22%</span>
          </div>
        </div>
      </div>

      <!-- 95% 1-Month VaR Card with Slider Scale -->
      <div class="bg-slate-900 border border-slate-800 rounded-xl p-4 flex flex-col justify-between">
        <div>
          <span class="text-xs text-slate-400 font-mono flex items-center gap-1">
            95% 1-Month VaR <button type="button" onclick="openInfoModal('var')" class="inline-flex items-center justify-center w-4 h-4 rounded-full bg-slate-800 hover:bg-brand-500/30 text-slate-400 hover:text-brand-300 transition text-[11px] font-sans font-bold cursor-pointer border border-slate-700/60 hover:border-brand-500/50" title="Click for explanation">ⓘ</button>
          </span>
          <div class="flex items-baseline mt-1 gap-2">
            <span class="text-2xl font-bold font-mono text-white" id="statVaRPct">0.0%</span>
            <span class="text-xs font-semibold px-2 py-0.5 rounded-full" id="statVaRTierBadge">High</span>
          </div>
          <span class="text-xs text-rose-300/80 font-mono block mt-0.5" id="statVaR">$0.00 max monthly loss</span>
        </div>
        <div class="mt-3">
          <div class="gauge-container">
            <div class="gauge-track">
              <div class="gauge-segment gauge-green" title="Low Risk (&lt; 5%)"></div>
              <div class="gauge-segment gauge-yellow" title="Moderate Risk (5% - 10%)"></div>
              <div class="gauge-segment gauge-red" title="High Risk (&gt; 10%)"></div>
            </div>
            <div class="gauge-slider-marker" id="statVaRMarker" style="left: 65%;"></div>
          </div>
          <div class="flex justify-between text-[10px] text-slate-500 font-mono mt-1 px-0.5">
            <span>&lt;5%</span>
            <span>5-10%</span>
            <span>&gt;10%</span>
          </div>
        </div>
      </div>
    </div>

    <!-- Look-Through & Thematic Grid -->
    <div class="grid grid-cols-1 lg:grid-cols-12 gap-6">

      <!-- Left Column: Look-Through Decomposition -->
      <div class="lg:col-span-8 bg-slate-900 border border-slate-800 rounded-xl p-5 space-y-4">
        <div class="flex items-center justify-between border-b border-slate-800 pb-3">
          <div>
            <h3 class="font-bold text-base text-white">Effective Look-Through Exposure</h3>
            <p class="text-xs text-slate-400">Unpacks direct shares and indirect ETF overlapping holdings into true underlying ownership.</p>
          </div>
          <span class="text-xs font-mono text-slate-400 bg-slate-800 px-2.5 py-1 rounded-md">Ranked by Total Weight</span>
        </div>

        <div class="overflow-x-auto">
          <table class="w-full text-left border-collapse text-xs">
            <thead>
              <tr class="border-b border-slate-800 text-[11px] font-mono text-slate-400">
                <th class="py-2 px-3">Underlying Asset</th>
                <th class="py-2 px-3">Total Effective Exposure</th>
                <th class="py-2 px-3">Direct vs Indirect Split</th>
                <th class="py-2 px-3">Overlap Origins</th>
              </tr>
            </thead>
            <tbody id="exposureTableBody" class="divide-y divide-slate-800/60 font-mono">
              <!-- Populated by JS -->
            </tbody>
          </table>
        </div>
      </div>

      <!-- Right Column: Thematic & Macro Sensitivities -->
      <div class="lg:col-span-4 bg-slate-900 border border-slate-800 rounded-xl p-5 space-y-4">
        <div class="border-b border-slate-800 pb-3">
          <h3 class="font-bold text-base text-white">Thematic Factor Exposure</h3>
          <p class="text-xs text-slate-400">Underlying economic drivers shared across holdings (≥ 5% exposure).</p>
        </div>

        <div class="space-y-3.5" id="thematicBars">
          <!-- Populated by JS -->
        </div>
      </div>
    </div>

    <!-- Stress Testing Grid -->
    <div class="bg-slate-900 border border-slate-800 rounded-xl p-5 space-y-4">
      <div class="border-b border-slate-800 pb-3 flex items-center justify-between">
        <div>
          <h3 class="font-bold text-base text-white">Historical Crisis Stress Simulations</h3>
          <p class="text-xs text-slate-400">Dynamic factor stress scenarios tailored to your portfolio's exposure profile.</p>
        </div>
        <span class="text-xs font-mono text-brand-300 bg-brand-500/10 border border-brand-500/30 px-2.5 py-1 rounded-md">Dynamic Portfolio Factor Shocks</span>
      </div>
      <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4" id="stressCards">
        <!-- Populated by JS -->
      </div>
    </div>

    <!-- Two Column: What-If Simulator & Headline Auditor -->
    <div class="grid grid-cols-1 lg:grid-cols-12 gap-6">

      <!-- What-If Simulator -->
      <div class="lg:col-span-6 bg-slate-900 border border-slate-800 rounded-xl p-5 space-y-4">
        <div class="border-b border-slate-800 pb-3">
          <div class="flex items-center gap-2">
            <span class="text-brand-400 text-sm font-bold">🔮</span>
            <h3 class="font-bold text-base text-white">"What-If" Decision Simulator</h3>
          </div>
          <p class="text-xs text-slate-400">Test pre-trade allocation shifts before executing.</p>
        </div>

        <div class="space-y-3">
          <div class="grid grid-cols-2 gap-3">
            <div>
              <div class="flex items-center justify-between mb-1">
                <label class="text-xs text-slate-400 block">Ticker or Stock Name</label>
                <span id="whatIfDetected" class="text-[10px] text-brand-400 font-mono font-medium hidden"></span>
              </div>
              <input type="text" id="whatIfTicker" value="NVIDIA" placeholder="e.g. Apple, NVDA, Tesla, SpaceX, QQQ..." oninput="updateWhatIfDetection(this.value)" class="bg-slate-800 border border-slate-700 text-white text-xs rounded-lg px-3 py-2 w-full font-mono focus:ring-1 focus:ring-brand-500 focus:outline-none" />
            </div>
            <div>
              <label class="text-xs text-slate-400 block mb-1">Target Allocation (%)</label>
              <input type="number" id="whatIfPct" value="50" min="0" max="100" class="bg-slate-800 border border-slate-700 text-white text-xs rounded-lg px-3 py-2 w-full font-mono focus:ring-1 focus:ring-brand-500 focus:outline-none" />
            </div>
          </div>
          <button onclick="handleWhatIf()" class="w-full bg-brand-600 hover:bg-brand-500 text-white text-xs font-semibold py-2 px-4 rounded-lg transition shadow-md shadow-brand-500/20">
            Simulate Allocation Shift
          </button>
        </div>

        <!-- What-If Output Box -->
        <div id="whatIfResultBox" class="hidden bg-slate-950 border border-slate-800 rounded-lg p-3.5 space-y-2.5 font-mono text-xs">
          <!-- Populated by JS -->
        </div>
      </div>

      <!-- Headline Shock Auditor -->
      <div class="lg:col-span-6 bg-slate-900 border border-slate-800 rounded-xl p-5 space-y-4">
        <div class="border-b border-slate-800 pb-3">
          <div class="flex items-center gap-2">
            <span class="text-brand-400 text-sm font-bold">📰</span>
            <h3 class="font-bold text-base text-white">Headline Shock Impact Auditor</h3>
          </div>
          <p class="text-xs text-slate-400">Type any breaking economic news to inspect holding sensitivity.</p>
        </div>

        <div class="space-y-3">
          <div>
            <label class="text-xs text-slate-400 block mb-1">Breaking Headline / Macro Event</label>
            <input type="text" id="headlineInput" value="Fed signals 50 bps surprise rate hike as inflation spikes" onkeydown="if (event.key === 'Enter') handleHeadlineAudit()" class="bg-slate-800 border border-slate-700 text-white text-xs rounded-lg px-3 py-2 w-full focus:ring-1 focus:ring-brand-500 focus:outline-none" />
          </div>
          <button id="headlineAuditBtn" onclick="handleHeadlineAudit()" class="w-full bg-slate-800 hover:bg-slate-750 border border-slate-700 text-slate-200 text-xs font-semibold py-2 px-4 rounded-lg transition cursor-pointer">
            Audit Headline Exposure
          </button>
        </div>

        <!-- Headline Output Box -->
        <div id="headlineResultBox" class="hidden bg-slate-950 border border-slate-800 rounded-lg p-3.5 space-y-2.5 font-mono text-xs">
          <!-- Populated by JS -->
        </div>
      </div>

    </div>

    <!-- AI Risk Analyst Chat -->
    <div class="bg-slate-900 border border-slate-800 rounded-xl p-5 space-y-4">
      <div class="border-b border-slate-800 pb-3 flex items-center justify-between">
        <div>
          <div class="flex items-center gap-2">
            <span class="w-2.5 h-2.5 rounded-full bg-brand-500 animate-pulse"></span>
            <h3 class="font-bold text-base text-white">Conversational AI Risk Assistant</h3>
          </div>
          <p class="text-xs text-slate-400">Ask questions about hidden vulnerabilities, hypothetical moves, or risk concepts.</p>
        </div>
        <span class="text-[11px] font-mono text-slate-500 bg-slate-800/80 px-2 py-1 rounded">Traceable Reasoning</span>
      </div>

      <!-- Chat History -->
      <div id="chatMessages" class="bg-slate-950 border border-slate-800 rounded-xl p-4 min-h-[180px] max-h-[320px] overflow-y-auto space-y-3 text-xs">
        <div class="bg-slate-900 border border-slate-800 p-3 rounded-lg text-slate-300">
          <span class="font-bold text-brand-400 block mb-1">Portfolio X-Ray Analyst:</span>
          Hello! I have analyzed your portfolio. Ask me anything about your look-through overlaps, concentration risks, or how specific market events could affect your capital.
        </div>
      </div>

      <!-- Chat Input -->
      <form onsubmit="handleChatSubmit(event)" class="flex gap-2">
        <input type="text" id="chatInput" placeholder="Ask e.g. 'What is my single biggest vulnerability?' or 'How much NVDA do I really own?'..." class="flex-1 bg-slate-800 border border-slate-700 text-white text-xs rounded-lg px-3.5 py-2.5 focus:ring-1 focus:ring-brand-500 focus:outline-none font-sans" />
        <button type="submit" class="bg-brand-600 hover:bg-brand-500 text-white text-xs font-semibold px-5 py-2.5 rounded-lg transition shadow-md shadow-brand-500/20">
          Send
        </button>
      </form>
    </div>

    <!-- Info Explanation Modal -->
    <div id="infoModalBackdrop" class="fixed inset-0 z-50 bg-black/75 backdrop-blur-sm hidden items-center justify-center p-4 transition-all duration-200" onclick="closeInfoModalOnBackdrop(event)">
      <div class="bg-slate-900 border border-slate-700/80 rounded-2xl max-w-lg w-full p-6 shadow-2xl relative text-left" onclick="event.stopPropagation()">
        <div class="flex items-start justify-between pb-3 border-b border-slate-800">
          <div class="flex items-center gap-3">
            <div class="w-9 h-9 rounded-xl bg-brand-500/20 border border-brand-500/40 flex items-center justify-center text-brand-300 font-bold text-base shadow-sm">ⓘ</div>
            <div>
              <h3 id="infoModalTitle" class="text-base font-bold text-white leading-tight">Metric Explanation</h3>
              <p id="infoModalSubtitle" class="text-xs text-slate-400 mt-0.5">Quantitative definition & institutional standard</p>
            </div>
          </div>
          <button type="button" onclick="closeInfoModal()" class="text-slate-400 hover:text-white bg-slate-800 hover:bg-slate-700 w-7 h-7 rounded-lg flex items-center justify-center transition text-sm font-bold cursor-pointer" title="Close (Esc)">✕</button>
        </div>

        <div class="mt-4 space-y-4 text-xs text-slate-300">
          <div id="infoModalBody" class="leading-relaxed space-y-2">
            <!-- Dynamic content -->
          </div>

          <div id="infoModalScale" class="bg-slate-950/80 border border-slate-800/80 rounded-xl p-3.5 space-y-2">
            <!-- Benchmarks & tiers -->
          </div>
        </div>

        <div class="mt-5 pt-3 border-t border-slate-800 flex justify-end">
          <button type="button" onclick="closeInfoModal()" class="bg-brand-600 hover:bg-brand-500 text-white px-5 py-2 rounded-lg text-xs font-semibold transition cursor-pointer shadow-md shadow-brand-500/20">
            Got it
          </button>
        </div>
      </div>
    </div>

  </main>

  <!-- Footer -->
  <footer class="border-t border-slate-800 bg-slate-900/60 py-4 px-6 text-center text-xs text-slate-500 font-mono mt-8">
    Portfolio X-Ray &bull; Institutional Portfolio Risk Engine &bull; Pure Standard Library Architecture &bull; AgentVerse 2026
  </footer>

  <!-- Frontend Logic -->
  <script>
    let currentPortfolio = null;
    let presets = {};

    const STOCK_NAMES = {
      "NVDA": "NVIDIA Corporation",
      "AAPL": "Apple Inc.",
      "MSFT": "Microsoft Corporation",
      "AMZN": "Amazon.com Inc.",
      "GOOGL": "Alphabet Inc. (Google Class A)",
      "GOOG": "Alphabet Inc. (Google Class C)",
      "META": "Meta Platforms Inc.",
      "TSLA": "Tesla Inc.",
      "AVGO": "Broadcom Inc.",
      "AMD": "Advanced Micro Devices Inc.",
      "TSM": "Taiwan Semiconductor (TSMC)",
      "ASML": "ASML Holding N.V.",
      "CEG": "Constellation Energy Corp.",
      "JPM": "JPMorgan Chase & Co.",
      "LLY": "Eli Lilly and Company",
      "XOM": "Exxon Mobil Corp.",
      "COST": "Costco Wholesale Corp.",
      "SPACEX": "Space Exploration Technologies Corp. (SpaceX)",
      "OPENAI": "OpenAI Inc.",
      "PLTR": "Palantir Technologies Inc.",
      "UBER": "Uber Technologies Inc.",
      "ABNB": "Airbnb Inc.",
      "NFLX": "Netflix Inc.",
      "COIN": "Coinbase Global Inc.",
      "BA": "The Boeing Company",
      "LMT": "Lockheed Martin Corporation",
      "RKLB": "Rocket Lab USA Inc.",
      "QQQ": "Invesco QQQ Trust (Nasdaq-100)",
      "SPY": "SPDR S&P 500 ETF Trust",
      "VOO": "Vanguard S&P 500 ETF",
      "VTI": "Vanguard Total Stock Market ETF",
      "SMH": "VanEck Semiconductor ETF",
      "XLK": "Technology Select Sector SPDR Fund",
      "XLF": "Financial Select Sector SPDR Fund",
      "XLE": "Energy Select Sector SPDR Fund",
      "TLT": "iShares 20+ Year Treasury Bond ETF",
      "BND": "Vanguard Total Bond Market ETF",
      "GLD": "SPDR Gold Shares ETF",
      "CASH": "USD Cash / Money Market"
    };

    const STOCK_SYNONYMS = {
      "nvidia": "NVDA", "nvda": "NVDA", "nia": "NVDA", "nvida": "NVDA",
      "apple": "AAPL", "aapl": "AAPL", "apple inc": "AAPL",
      "microsoft": "MSFT", "msft": "MSFT", "microsoft corp": "MSFT",
      "amazon": "AMZN", "amzn": "AMZN", "amazon.com": "AMZN",
      "google": "GOOGL", "googl": "GOOGL", "goog": "GOOG", "alphabet": "GOOGL",
      "meta": "META", "facebook": "META",
      "tesla": "TSLA", "tsla": "TSLA", "tesla inc": "TSLA", "tesla motors": "TSLA",
      "broadcom": "AVGO", "avgo": "AVGO",
      "amd": "AMD",
      "spacex": "SPACEX", "space x": "SPACEX", "starlink": "SPACEX", "space exploration technologies": "SPACEX",
      "openai": "OPENAI", "open ai": "OPENAI",
      "palantir": "PLTR", "pltr": "PLTR",
      "uber": "UBER", "airbnb": "ABNB", "abnb": "ABNB",
      "netflix": "NFLX", "nflx": "NFLX", "coinbase": "COIN", "coin": "COIN",
      "boeing": "BA", "ba": "BA", "lockheed": "LMT", "lmt": "LMT",
      "rocket lab": "RKLB", "rocketlab": "RKLB", "rklb": "RKLB",
      "invesco": "QQQ", "qqq": "QQQ", "nasdaq": "QQQ",
      "spy": "SPY", "spdr": "SPY", "ps": "SPY", "s&p": "SPY", "s&p 500": "SPY",
      "voo": "VOO", "veo": "VOO", "vanguard": "VOO", "vti": "VTI",
      "smh": "SMH", "semiconductor": "SMH", "semiconductors": "SMH",
      "tlt": "TLT", "treasury": "TLT", "bnd": "BND", "bonds": "BND",
      "gld": "GLD", "gold": "GLD", "cash": "CASH", "usd": "CASH",
      "xle": "XLE", "energy": "XLE", "xlf": "XLF", "financials": "XLF", "xlk": "XLK", "tech": "XLK",
      "ceg": "CEG", "berkshire": "BRK.B", "brk": "BRK.B", "brk.b": "BRK.B", "brie": "BRK.B",
      "jpmorgan": "JPM", "jpm": "JPM", "lilly": "LLY", "lly": "LLY",
      "costco": "COST", "cost": "COST", "exxon": "XOM", "xom": "XOM",
      "tsmc": "TSM", "taiwan semi": "TSM", "asml": "ASML"
    };

    const DEFAULT_PRICES_MAP = {
      "NVDA": 130.0, "AAPL": 225.0, "MSFT": 420.0, "AMZN": 190.0,
      "GOOGL": 170.0, "GOOG": 170.0, "META": 550.0, "TSLA": 220.0,
      "AVGO": 170.0, "AMD": 140.0, "QQQ": 480.0, "SPY": 550.0,
      "VOO": 500.0, "VTI": 270.0, "SMH": 250.0, "XLK": 220.0,
      "XLF": 45.0, "XLE": 90.0, "TLT": 95.0, "BND": 72.0,
      "GLD": 230.0, "CEG": 245.0, "BRK.B": 460.0, "JPM": 220.0,
      "LLY": 820.0, "CASH": 1.0, "SPACEX": 112.0, "OPENAI": 150.0,
      "PLTR": 45.0, "UBER": 75.0, "ABNB": 135.0, "NFLX": 700.0,
      "COIN": 210.0, "BA": 155.0, "LMT": 580.0, "RKLB": 12.0,
      "TSM": 190.0, "ASML": 850.0, "COST": 880.0, "XOM": 118.0
    };

    function resolveStock(input) {
      if (!input) return { ticker: "CUSTOM", name: "Custom Holding", defaultPrice: 100.0 };
      const raw = input.trim();
      const upper = raw.toUpperCase();
      const lower = raw.toLowerCase();

      // Exact ticker match
      if (STOCK_NAMES[upper]) {
        return { ticker: upper, name: STOCK_NAMES[upper], defaultPrice: DEFAULT_PRICES_MAP[upper] || 100.0 };
      }
      // Exact synonym match
      if (STOCK_SYNONYMS[lower]) {
        const sym = STOCK_SYNONYMS[lower];
        return { ticker: sym, name: STOCK_NAMES[sym] || sym, defaultPrice: DEFAULT_PRICES_MAP[sym] || 100.0 };
      }
      // Check individual tokens against synonyms
      for (const tok of lower.split(' ')) {
        const cleanTok = tok.replace(/[^a-z0-9.]/g, '');
        if (cleanTok && STOCK_SYNONYMS[cleanTok]) {
          const sym = STOCK_SYNONYMS[cleanTok];
          return { ticker: sym, name: STOCK_NAMES[sym] || sym, defaultPrice: DEFAULT_PRICES_MAP[sym] || 100.0 };
        }
      }
      // Full name or substring match in known stock names
      for (const [sym, fullName] of Object.entries(STOCK_NAMES)) {
        if (fullName.toLowerCase().includes(lower) || lower.includes(fullName.toLowerCase())) {
          return { ticker: sym, name: fullName, defaultPrice: DEFAULT_PRICES_MAP[sym] || 100.0 };
        }
      }
      // Fallback: use raw input as name, uppercase as ticker
      const fallbackTicker = (upper.replace(/[^A-Z0-9]/g, '').slice(0, 8)) || "CUSTOM";
      return { ticker: fallbackTicker, name: raw, defaultPrice: DEFAULT_PRICES_MAP[fallbackTicker] || 100.0 };
    }

    const RISK_PROFILES = {
      1: {
        tolerance: "Very Low",
        label: "Very Low (Max 5-8% drawdown)",
        loss_capacity: "Capital Preservation (Max 5-8% drawdown)",
        badgeClass: "bg-emerald-500/20 text-emerald-300 border border-emerald-500/30",
        horizon_years: 2,
      },
      2: {
        tolerance: "Low",
        label: "Low (Max 10-12% drawdown)",
        loss_capacity: "Conservative (Max 10-12% drawdown)",
        badgeClass: "bg-teal-500/20 text-teal-300 border border-teal-500/30",
        horizon_years: 4,
      },
      3: {
        tolerance: "Moderate",
        label: "Moderate (Max 15-20% drawdown)",
        loss_capacity: "Moderate (Max 15-20% drawdown)",
        badgeClass: "bg-amber-500/20 text-amber-300 border border-amber-500/30",
        horizon_years: 7,
      },
      4: {
        tolerance: "High",
        label: "High (Max 25-30% drawdown)",
        loss_capacity: "High Growth (Max 25-30% drawdown)",
        badgeClass: "bg-orange-500/20 text-orange-300 border border-orange-500/30",
        horizon_years: 10,
      },
      5: {
        tolerance: "Very High",
        label: "Very High (Max 35%+ drawdown)",
        loss_capacity: "Aggressive Speculative (Max 35%+ drawdown)",
        badgeClass: "bg-rose-500/20 text-rose-300 border border-rose-500/30",
        horizon_years: 15,
      }
    };

    function getRiskLevelFromTolerance(tolerance) {
      if (!tolerance) return 3;
      const t = String(tolerance).toLowerCase();
      if (t.includes("very low") || t.includes("preservation")) return 1;
      if (t.includes("low") || t.includes("conservative")) return 2;
      if (t.includes("very high") || t.includes("aggressive")) return 5;
      if (t.includes("high") || t.includes("growth")) return 4;
      if (t.includes("moderate") || t.includes("balanced")) return 3;
      return 3;
    }

    function setRiskSlider(val) {
      const slider = document.getElementById('riskProfileSlider');
      if (slider) {
        slider.value = val;
      }
      handleRiskSliderChange(val, true);
    }

    function handleRiskSliderChange(val, triggerAnalysis = true) {
      const level = parseInt(val) || 3;
      const profile = RISK_PROFILES[level] || RISK_PROFILES[3];
      const badge = document.getElementById('riskProfileBadge');
      if (badge) {
        badge.innerText = profile.label;
        badge.className = "text-[11px] font-semibold px-2 py-0.5 rounded-full font-sans transition " + profile.badgeClass;
      }
      if (currentPortfolio) {
        currentPortfolio.user_profile = currentPortfolio.user_profile || {};
        currentPortfolio.user_profile.risk_tolerance = profile.tolerance;
        currentPortfolio.user_profile.loss_capacity = profile.loss_capacity;
        currentPortfolio.user_profile.horizon_years = profile.horizon_years;
        if (triggerAnalysis) {
          runAnalysis();
        }
      }
    }

    const METRIC_DEFINITIONS = {
      hhi: {
        title: "Concentration — Herfindahl-Hirschman Index (HHI)",
        subtitle: "Regulatory Standard for Single-Stock & Look-Through Concentration",
        body: `
          <p>The <strong>Herfindahl-Hirschman Index (HHI)</strong> is the institutional gold standard used by the U.S. Department of Justice, the Federal Reserve, and top risk desks to measure asset concentration.</p>
          <p>In <strong>Portfolio X-Ray</strong>, HHI is computed <em>after full look-through exposure</em>: if you own an individual stock and also hold it inside QQQ or SPY, your total combined effective exposure is determined before calculating HHI:</p>
          <div class="bg-slate-950 p-2.5 rounded-lg border border-slate-800 font-mono text-[11px] text-brand-300">
            HHI = &Sigma; (Weight<sub>i</sub> &times; 100)<sup>2</sup>
          </div>
          <p class="text-slate-400 text-[11px]">A perfectly balanced 100-stock portfolio has an HHI of 100. A portfolio concentrated 100% in a single company has an HHI of 10,000.</p>
        `,
        scale: `
          <span class="text-[11px] font-bold uppercase tracking-wider text-slate-400 block mb-1.5">Institutional Benchmark Tiers:</span>
          <div class="space-y-1.5 text-[11px]">
            <div class="flex items-center justify-between p-2 rounded-lg bg-emerald-500/10 border border-emerald-500/20 text-emerald-300">
              <span class="font-semibold">&lt; 1,500: Diversified</span>
              <span class="text-slate-400 font-mono">Healthy balance</span>
            </div>
            <div class="flex items-center justify-between p-2 rounded-lg bg-amber-500/10 border border-amber-500/20 text-amber-300">
              <span class="font-semibold">1,500 – 2,500: Moderate</span>
              <span class="text-slate-400 font-mono">Top holdings dominate</span>
            </div>
            <div class="flex items-center justify-between p-2 rounded-lg bg-rose-500/10 border border-rose-500/20 text-rose-300">
              <span class="font-semibold">&gt; 2,500: High Concentration</span>
              <span class="text-slate-400 font-mono">Acute single-stock risk</span>
            </div>
          </div>
        `
      },
      capital: {
        title: "Total Portfolio Capital",
        subtitle: "Live Aggregate Market Value",
        body: `
          <p>The total current dollar value of your entire investment portfolio across all individual equities, ETFs, index funds, and private/custom holdings.</p>
          <p class="text-slate-400 text-[11px]">All portfolio percentage weights, stress-test loss projections, and Value-at-Risk dollar calculations are evaluated against this active capital balance.</p>
        `,
        scale: `
          <div class="text-[11px] text-slate-300 flex items-start gap-2">
            <span class="text-base leading-none">💡</span>
            <div>
              <span class="font-semibold text-white">Dynamic Recalculation:</span>
              <p class="text-slate-400 mt-0.5">You can adjust share quantities or prices directly in the <em>Active Holdings Editor</em> table above and click <em>Recalculate Risk Analytics</em> to inspect live changes.</p>
            </div>
          </div>
        `
      },
      volatility: {
        title: "Annualized Volatility (Risk)",
        subtitle: "1-Year Historical Cross-Asset Standard Deviation",
        body: `
          <p><strong>Annualized Volatility</strong> measures the dispersion and severity of return swings over a standard 252-day trading year.</p>
          <p>Higher volatility indicates wider price fluctuations and heightened drawdowns during turbulent market cycles. Portfolio X-Ray computes this using constituent volatilities and cross-asset correlation matrices.</p>
        `,
        scale: `
          <span class="text-[11px] font-bold uppercase tracking-wider text-slate-400 block mb-1.5">Volatility Scale & Regimes:</span>
          <div class="space-y-1.5 text-[11px]">
            <div class="flex items-center justify-between p-2 rounded-lg bg-emerald-500/10 border border-emerald-500/20 text-emerald-300">
              <span class="font-semibold">&lt; 14%: Low / Defensive</span>
              <span class="text-slate-400 font-mono">Bonds & Defensives</span>
            </div>
            <div class="flex items-center justify-between p-2 rounded-lg bg-amber-500/10 border border-amber-500/20 text-amber-300">
              <span class="font-semibold">14% – 22%: Moderate</span>
              <span class="text-slate-400 font-mono">Standard S&P 500 (~16%)</span>
            </div>
            <div class="flex items-center justify-between p-2 rounded-lg bg-rose-500/10 border border-rose-500/20 text-rose-300">
              <span class="font-semibold">&gt; 22%: High Volatility</span>
              <span class="text-slate-400 font-mono">High-Beta & Tech Growth</span>
            </div>
          </div>
        `
      },
      var: {
        title: "95% 1-Month Value at Risk (VaR)",
        subtitle: "Parametric Downside Tail Risk Exposure",
        body: `
          <p><strong>Value at Risk (VaR)</strong> estimates the potential loss in portfolio value over a 1-month horizon at a 95% statistical confidence level.</p>
          <p>This means that in <strong>19 out of 20 months</strong> (95% of the time), your monthly drawdown will stay within this number. In <strong>1 out of 20 months</strong> (5% tail risk probability), losses are expected to exceed this threshold.</p>
        `,
        scale: `
          <span class="text-[11px] font-bold uppercase tracking-wider text-slate-400 block mb-1.5">Tail Risk Tiers:</span>
          <div class="space-y-1.5 text-[11px]">
            <div class="flex items-center justify-between p-2 rounded-lg bg-emerald-500/10 border border-emerald-500/20 text-emerald-300">
              <span class="font-semibold">&lt; 5%: Low Tail Risk</span>
              <span class="text-slate-400 font-mono">Capital Preservation</span>
            </div>
            <div class="flex items-center justify-between p-2 rounded-lg bg-amber-500/10 border border-amber-500/20 text-amber-300">
              <span class="font-semibold">5% – 10%: Moderate Tail Risk</span>
              <span class="text-slate-400 font-mono">Balanced Equities</span>
            </div>
            <div class="flex items-center justify-between p-2 rounded-lg bg-rose-500/10 border border-rose-500/20 text-rose-300">
              <span class="font-semibold">&gt; 10%: High Tail Risk</span>
              <span class="text-slate-400 font-mono">Substantial Vulnerability</span>
            </div>
          </div>
        `
      }
    };

    function openInfoModal(key) {
      const data = METRIC_DEFINITIONS[key];
      if (!data) return;
      document.getElementById('infoModalTitle').textContent = data.title;
      document.getElementById('infoModalSubtitle').textContent = data.subtitle;
      document.getElementById('infoModalBody').innerHTML = data.body;
      document.getElementById('infoModalScale').innerHTML = data.scale;
      const modal = document.getElementById('infoModalBackdrop');
      if (modal) {
        modal.classList.remove('hidden');
        modal.classList.add('flex');
      }
    }

    function closeInfoModal() {
      const modal = document.getElementById('infoModalBackdrop');
      if (modal) {
        modal.classList.add('hidden');
        modal.classList.remove('flex');
      }
    }

    function closeInfoModalOnBackdrop(e) {
      if (e.target.id === 'infoModalBackdrop') {
        closeInfoModal();
      }
    }

    window.addEventListener('keydown', (e) => {
      if (e.key === 'Escape') {
        closeInfoModal();
      }
    });

    async function init() {
      try {
        const res = await fetch('/api/presets');
        presets = await res.json();
        renderPresetButtons();
        // Load default preset
        loadPreset('accidental_tech_overweight');
        const defaultWhatIf = document.getElementById('whatIfTicker');
        if (defaultWhatIf) updateWhatIfDetection(defaultWhatIf.value);
      } catch (e) {
        console.error("Init failed:", e);
      }
    }

    function renderPresetButtons() {
      const container = document.getElementById('presetButtons');
      container.innerHTML = '';
      for (const [key, p] of Object.entries(presets)) {
        const btn = document.createElement('button');
        btn.className = "text-[11px] px-2.5 py-1 rounded-md border font-medium transition cursor-pointer " +
          (key === 'accidental_tech_overweight' ? "bg-brand-600/20 border-brand-500/40 text-brand-300" : "bg-slate-800 border-slate-700 text-slate-400 hover:text-slate-200 hover:bg-slate-750");
        btn.innerText = p.name.split(' (')[0];
        btn.onclick = () => loadPreset(key);
        btn.id = "preset-" + key;
        container.appendChild(btn);
      }
    }

    function loadPreset(key) {
      if (!presets[key]) return;
      currentPortfolio = JSON.parse(JSON.stringify(presets[key]));
      const riskLevel = getRiskLevelFromTolerance(currentPortfolio.user_profile && currentPortfolio.user_profile.risk_tolerance);
      setRiskSlider(riskLevel);

      // Update button styles
      for (const k of Object.keys(presets)) {
        const el = document.getElementById('preset-' + k);
        if (el) {
          if (k === key) {
            el.className = "text-[11px] px-2.5 py-1 rounded-md border font-medium transition bg-brand-600/20 border-brand-500/40 text-brand-300";
          } else {
            el.className = "text-[11px] px-2.5 py-1 rounded-md border font-medium transition bg-slate-800 border-slate-700 text-slate-400 hover:text-slate-200 hover:bg-slate-750";
          }
        }
      }

      // Populate prompt text representation with readable stock names
      const holdingsSummary = (currentPortfolio.holdings || []).map(h => {
        const info = resolveStock(h.ticker);
        h.name = h.name || info.name;
        return `${h.shares} ${info.name || h.ticker} @ $${h.price}`;
      }).join(', ');
      document.getElementById('portfolioPromptInput').value = holdingsSummary;

      renderHoldingsEditor();
      runAnalysis();
    }

    function renderHoldingsEditor() {
      const tb = document.getElementById('holdingsEditorBody');
      if (!tb || !currentPortfolio) return;
      tb.innerHTML = '';

      const holdings = currentPortfolio.holdings || [];
      document.getElementById('holdingCountBadge').innerText = `${holdings.length} assets`;

      holdings.forEach((h, idx) => {
        const tr = document.createElement('tr');
        tr.className = "border-b border-slate-850 hover:bg-slate-900/40 transition";
        const val = (h.shares * h.price).toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2});
        const stockInfo = resolveStock(h.name || h.ticker);
        const displayName = h.name || stockInfo.name || h.ticker;
        const displayTicker = stockInfo.ticker || h.ticker;

        tr.innerHTML = `
          <td class="py-1.5 px-3">
            <div class="flex items-center gap-2">
              <input type="text" value="${displayName}" onchange="updateHoldingField(${idx}, 'stock_name', this.value)" class="bg-slate-950 border border-slate-800 rounded px-2.5 py-1 text-white font-medium w-52 sm:w-64 text-xs focus:border-brand-500 focus:outline-none" placeholder="Stock Name (e.g. Tesla, Apple)" />
              <span class="text-[10px] font-mono px-1.5 py-0.5 rounded bg-slate-850 text-slate-400 border border-slate-800">${displayTicker}</span>
            </div>
          </td>
          <td class="py-1.5 px-3">
            <input type="number" step="any" min="0" value="${h.shares}" onchange="updateHoldingField(${idx}, 'shares', this.value)" class="bg-slate-950 border border-slate-800 rounded px-2 py-1 text-slate-200 font-mono w-20 text-xs focus:border-brand-500 focus:outline-none" />
          </td>
          <td class="py-1.5 px-3">
            <div class="flex items-center gap-1 font-mono text-slate-400">
              <span>$</span>
              <input type="number" step="any" min="0" value="${h.price}" onchange="updateHoldingField(${idx}, 'price', this.value)" class="bg-slate-950 border border-slate-800 rounded px-2 py-1 text-slate-200 font-mono w-20 text-xs focus:border-brand-500 focus:outline-none" />
            </div>
          </td>
          <td class="py-1.5 px-3 font-mono font-semibold text-white">
            $${val}
          </td>
          <td class="py-1.5 px-3 text-right">
            <button onclick="removeHoldingRow(${idx})" class="text-rose-400 hover:text-rose-300 text-xs px-2 py-1 rounded hover:bg-rose-500/10 transition cursor-pointer" title="Remove asset">
              ✕
            </button>
          </td>
        `;
        tb.appendChild(tr);
      });
    }

    function updateHoldingField(idx, field, val) {
      if (!currentPortfolio || !currentPortfolio.holdings[idx]) return;
      if (field === 'stock_name' || field === 'ticker') {
        const resolved = resolveStock(val);
        currentPortfolio.holdings[idx].ticker = resolved.ticker;
        currentPortfolio.holdings[idx].name = resolved.name;
        if (resolved.defaultPrice && (!currentPortfolio.holdings[idx].price || currentPortfolio.holdings[idx].price === 100.0 || currentPortfolio.holdings[idx].price === 0)) {
          currentPortfolio.holdings[idx].price = resolved.defaultPrice;
        }
      } else if (field === 'shares') {
        currentPortfolio.holdings[idx].shares = parseFloat(val) || 0;
      } else if (field === 'price') {
        currentPortfolio.holdings[idx].price = parseFloat(val) || 0;
      }
      renderHoldingsEditor();
      runAnalysis();
    }

    function addHoldingRow() {
      if (!currentPortfolio) {
        currentPortfolio = { name: "Personalized Portfolio", holdings: [], user_profile: { risk_tolerance: "Moderate" } };
      }
      currentPortfolio.holdings.push({ ticker: "AAPL", name: "Apple Inc.", shares: 10, price: 225.0 });
      renderHoldingsEditor();
      runAnalysis();
    }

    function removeHoldingRow(idx) {
      if (!currentPortfolio || !currentPortfolio.holdings) return;
      currentPortfolio.holdings.splice(idx, 1);
      renderHoldingsEditor();
      runAnalysis();
    }

    function clearHoldings() {
      if (!currentPortfolio) return;
      currentPortfolio.holdings = [];
      document.getElementById('portfolioPromptInput').value = '';
      renderHoldingsEditor();
      runAnalysis();
    }

    function runAnalysisFromEditor() {
      renderHoldingsEditor();
      runAnalysis();
    }

    async function handleParsePrompt(customText) {
      const inputEl = document.getElementById('portfolioPromptInput');
      const text = customText || (inputEl ? inputEl.value.trim() : '');
      if (!text) {
        alert("Please enter your portfolio holdings in the prompt box first (e.g. '500 apple and 20 spacex').");
        return;
      }

      const btn = document.getElementById('parsePromptBtn');
      const originalHtml = btn ? btn.innerHTML : '';
      if (btn) {
        btn.innerHTML = `<span>⏳ Parsing...</span>`;
        btn.disabled = true;
      }

      try {
        const res = await fetch('/api/parse-portfolio', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ text: text })
        });
        const data = await res.json();
        if (data.holdings && data.holdings.length > 0) {
          if (!currentPortfolio) {
            currentPortfolio = { name: "Personalized Portfolio", holdings: [], user_profile: { risk_tolerance: "Moderate" } };
          }
          currentPortfolio.name = "Personalized Portfolio";
          currentPortfolio.holdings = data.holdings.map(h => {
            const info = resolveStock(h.ticker || h.name);
            return {
              ticker: info.ticker,
              name: h.name || info.name,
              shares: h.shares,
              price: h.price || info.defaultPrice || 100.0
            };
          });
          renderHoldingsEditor();
          await runAnalysis();
          if (btn) {
            btn.innerHTML = `<span>✓ Holdings Updated!</span>`;
            setTimeout(() => {
              if (btn) {
                btn.innerHTML = originalHtml;
                btn.disabled = false;
              }
            }, 1800);
          }
        } else {
          alert("Could not detect ticker symbols. Try writing e.g.: '50 NVDA, 25 QQQ, 15 AAPL' or '500 apple and 20 spacex'");
          if (btn) { btn.innerHTML = originalHtml; btn.disabled = false; }
        }
      } catch (e) {
        console.error("Parse error:", e);
        alert("Could not parse portfolio holdings. Please try again.");
        if (btn) { btn.innerHTML = originalHtml; btn.disabled = false; }
      }
    }

    function handleDragOver(e) {
      e.preventDefault();
      document.getElementById('dropZone').classList.add('border-brand-500');
    }

    function handleDragLeave(e) {
      e.preventDefault();
      document.getElementById('dropZone').classList.remove('border-brand-500');
    }

    function handleDrop(e) {
      e.preventDefault();
      document.getElementById('dropZone').classList.remove('border-brand-500');
      if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
        processImageFile(e.dataTransfer.files[0]);
      }
    }

    async function handleImageUpload(e) {
      const file = e.target.files ? e.target.files[0] : null;
      if (!file) return;
      processImageFile(file);
    }

    async function processImageFile(file) {
      const dropContent = document.getElementById('dropZoneContent');
      const loading = document.getElementById('ocrLoading');
      const statusText = document.getElementById('ocrStatusText');
      const successBanner = document.getElementById('ocrSuccessBanner');

      dropContent.classList.add('hidden');
      loading.classList.remove('hidden');
      statusText.innerText = "Scanning image with OCR...";

      try {
        if (typeof Tesseract !== 'undefined') {
          const worker = await Tesseract.createWorker('eng');
          const ret = await worker.recognize(file);
          await worker.terminate();
          const text = ret.data.text;

          loading.classList.add('hidden');
          dropContent.classList.remove('hidden');

          document.getElementById('portfolioPromptInput').value = text;
          successBanner.classList.remove('hidden');
          successBanner.innerText = `✓ Extracted text from screenshot. Parsing holdings...`;

          await handleParsePrompt(text);
        } else {
          loading.classList.add('hidden');
          dropContent.classList.remove('hidden');
          alert("OCR engine is loading. Please enter your holdings in the prompt box.");
        }
      } catch (err) {
        console.error("OCR failed:", err);
        loading.classList.add('hidden');
        dropContent.classList.remove('hidden');
        alert("Could not scan image. Please enter your holdings in the text prompt box.");
      }
    }

    function updateProfile() {
      const slider = document.getElementById('riskProfileSlider');
      if (slider) {
        handleRiskSliderChange(slider.value, true);
      }
    }

    async function runAnalysis() {
      if (!currentPortfolio) return;
      try {
        const res = await fetch('/api/analyze', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(currentPortfolio)
        });
        const data = await res.json();
        renderAnalysis(data);
      } catch (e) {
        console.error("Analysis error:", e);
      }
    }

    function renderAnalysis(data) {
      const m = data.metrics;
      document.getElementById('statVal').innerText = '$' + m.portfolio_value.toLocaleString(undefined, {minimumFractionDigits: 2});
      const holdingsCount = (currentPortfolio && currentPortfolio.holdings) ? currentPortfolio.holdings.length : 0;
      const holdSummary = document.getElementById('statHoldingSummary');
      if (holdSummary) {
        holdSummary.innerText = `${holdingsCount} active asset${holdingsCount === 1 ? '' : 's'} • 100% invested`;
      }

      // 1. Concentration (HHI) Slider Scale
      document.getElementById('statHHI').innerText = m.hhi.toFixed(1);
      const tierBadge = document.getElementById('statTierBadge');
      let hhiPct = 0;
      if (m.hhi <= 1500) {
        hhiPct = (m.hhi / 1500) * 33.3;
        tierBadge.innerText = "Diversified";
        tierBadge.className = "text-xs font-semibold px-2 py-0.5 rounded-full bg-emerald-500/20 text-emerald-400 border border-emerald-500/30";
        document.getElementById('statHHI').className = "text-2xl font-bold font-mono text-emerald-400";
      } else if (m.hhi <= 2500) {
        hhiPct = 33.3 + ((m.hhi - 1500) / 1000) * 33.3;
        tierBadge.innerText = "Moderate";
        tierBadge.className = "text-xs font-semibold px-2 py-0.5 rounded-full bg-amber-500/20 text-amber-400 border border-amber-500/30";
        document.getElementById('statHHI').className = "text-2xl font-bold font-mono text-amber-400";
      } else {
        hhiPct = 66.6 + (Math.min(m.hhi - 2500, 2500) / 2500) * 33.4;
        tierBadge.innerText = "High Concentration";
        tierBadge.className = "text-xs font-semibold px-2 py-0.5 rounded-full bg-rose-500/20 text-rose-400 border border-rose-500/30";
        document.getElementById('statHHI').className = "text-2xl font-bold font-mono text-rose-400";
      }
      hhiPct = Math.max(3, Math.min(97, hhiPct));
      const hhiMarker = document.getElementById('statHHIMarker');
      if (hhiMarker) hhiMarker.style.left = hhiPct + '%';

      // 2. Annualized Volatility Slider Scale
      document.getElementById('statVol').innerText = m.annual_volatility.toFixed(1) + '%';
      const volBadge = document.getElementById('statVolTierBadge');
      const vol = m.annual_volatility;
      let volPct = 0;
      if (vol <= 14) {
        volPct = (vol / 14) * 33.3;
        volBadge.innerText = "Low / Defensive";
        volBadge.className = "text-xs font-semibold px-2 py-0.5 rounded-full bg-emerald-500/20 text-emerald-400 border border-emerald-500/30";
        document.getElementById('statVol').className = "text-2xl font-bold font-mono text-emerald-400";
      } else if (vol <= 22) {
        volPct = 33.3 + ((vol - 14) / 8) * 33.3;
        volBadge.innerText = "Moderate";
        volBadge.className = "text-xs font-semibold px-2 py-0.5 rounded-full bg-amber-500/20 text-amber-400 border border-amber-500/30";
        document.getElementById('statVol').className = "text-2xl font-bold font-mono text-amber-400";
      } else {
        volPct = 66.6 + (Math.min(vol - 22, 18) / 18) * 33.4;
        volBadge.innerText = "High Beta";
        volBadge.className = "text-xs font-semibold px-2 py-0.5 rounded-full bg-rose-500/20 text-rose-400 border border-rose-500/30";
        document.getElementById('statVol').className = "text-2xl font-bold font-mono text-rose-400";
      }
      volPct = Math.max(3, Math.min(97, volPct));
      const volMarker = document.getElementById('statVolMarker');
      if (volMarker) volMarker.style.left = volPct + '%';

      // 3. 95% 1-Month VaR Slider Scale
      const vp = m.var_95_1m_pct;
      document.getElementById('statVaRPct').innerText = vp.toFixed(1) + '%';
      document.getElementById('statVaR').innerText = '-$' + m.var_95_1m_dollar.toLocaleString(undefined, {minimumFractionDigits: 2}) + ' max monthly loss';
      const varBadge = document.getElementById('statVaRTierBadge');
      let varPct = 0;
      if (vp <= 5) {
        varPct = (vp / 5) * 33.3;
        varBadge.innerText = "Low Risk";
        varBadge.className = "text-xs font-semibold px-2 py-0.5 rounded-full bg-emerald-500/20 text-emerald-400 border border-emerald-500/30";
        document.getElementById('statVaRPct').className = "text-2xl font-bold font-mono text-emerald-400";
      } else if (vp <= 10) {
        varPct = 33.3 + ((vp - 5) / 5) * 33.3;
        varBadge.innerText = "Moderate Risk";
        varBadge.className = "text-xs font-semibold px-2 py-0.5 rounded-full bg-amber-500/20 text-amber-400 border border-amber-500/30";
        document.getElementById('statVaRPct').className = "text-2xl font-bold font-mono text-amber-400";
      } else {
        varPct = 66.6 + (Math.min(vp - 10, 10) / 10) * 33.4;
        varBadge.innerText = "High Risk";
        varBadge.className = "text-xs font-semibold px-2 py-0.5 rounded-full bg-rose-500/20 text-rose-400 border border-rose-500/30";
        document.getElementById('statVaRPct').className = "text-2xl font-bold font-mono text-rose-400";
      }
      varPct = Math.max(3, Math.min(97, varPct));
      const varMarker = document.getElementById('statVaRMarker');
      if (varMarker) varMarker.style.left = varPct + '%';

      // Render Look-Through Exposure Table
      const tb = document.getElementById('exposureTableBody');
      tb.innerHTML = '';
      data.exposures.forEach(exp => {
        const tr = document.createElement('tr');
        tr.className = "hover:bg-slate-850/50 transition";
        
        let overlapBadge = '<span class="text-slate-500 text-[11px]">Direct Only</span>';
        if (exp.indirect_weight > 0) {
          const sources = Object.entries(exp.contributing_sources).map(([k, v]) => `${k} (+${v.toFixed(1)}%)`).join(', ');
          overlapBadge = `<span class="text-amber-400 text-[11px] font-medium" title="Sourced from ${sources}">Overlap from ${sources}</span>`;
        } else if (['QQQ', 'SPY', 'VOO', 'SMH', 'XLK', 'XLE', 'XLF', 'TLT', 'BND', 'GLD'].includes(exp.ticker)) {
          overlapBadge = '<span class="text-brand-400 text-[11px]">Index Basket (Intact)</span>';
        }

        tr.innerHTML = `
          <td class="py-2.5 px-3">
            <span class="font-bold text-white block">${exp.ticker}</span>
            <span class="text-[11px] text-slate-400 font-sans">${exp.name}</span>
          </td>
          <td class="py-2.5 px-3">
            <span class="font-bold text-white">${exp.total_weight.toFixed(1)}%</span>
            <span class="text-[11px] text-slate-400 block">$${exp.total_dollar_value.toLocaleString(undefined, {minimumFractionDigits: 2})}</span>
          </td>
          <td class="py-2.5 px-3">
            <div class="flex items-center gap-1.5">
              <span class="text-brand-300 font-semibold">${exp.direct_weight.toFixed(1)}% dir</span>
              <span class="text-slate-500">/</span>
              <span class="text-amber-400 font-semibold">${exp.indirect_weight.toFixed(1)}% ind</span>
            </div>
            <div class="w-24 bg-slate-800 h-1.5 rounded-full overflow-hidden mt-1 flex">
              <div style="width: ${exp.direct_weight}%" class="bg-brand-500"></div>
              <div style="width: ${exp.indirect_weight}%" class="bg-amber-400"></div>
            </div>
          </td>
          <td class="py-2.5 px-3">${overlapBadge}</td>
        `;
        tb.appendChild(tr);
      });

      // Render Thematic Bars
      const thContainer = document.getElementById('thematicBars');
      thContainer.innerHTML = '';
      data.thematic.forEach(th => {
        if (th.weight < 5.0) return;
        const div = document.createElement('div');
        div.className = "space-y-1.5";
        div.innerHTML = `
          <div class="flex justify-between text-xs">
            <span class="font-medium text-slate-300">${th.theme}</span>
            <span class="font-mono font-bold text-brand-300">${th.weight.toFixed(1)}%</span>
          </div>
          <div class="w-full bg-slate-800 h-2 rounded-full overflow-hidden">
            <div class="bg-gradient-to-r from-brand-600 to-cyan-500 h-full rounded-full transition-all duration-500" style="width: ${th.weight}%"></div>
          </div>
          <div class="flex gap-2 text-[10px] text-slate-500 font-mono">
            ${th.top_assets.map(a => `<span>${a.ticker} (${a.weight.toFixed(1)}%)</span>`).join('')}
          </div>
        `;
        thContainer.appendChild(div);
      });

      // Render Stress Cards
      const sc = document.getElementById('stressCards');
      sc.innerHTML = '';
      data.stress_tests.forEach(st => {
        const div = document.createElement('div');
        div.className = "bg-slate-950 border border-slate-800/80 rounded-xl p-3.5 space-y-2 hover:border-slate-700 transition flex flex-col justify-between";
        const badgeHtml = st.category_badge
          ? `<span class="text-[10px] font-mono px-1.5 py-0.5 rounded bg-brand-500/15 border border-brand-500/30 text-brand-300 font-semibold">${st.category_badge}</span>`
          : '';
        const drawdownSign = st.drawdown_pct > 0 ? '+' : '';
        const dollarSign = st.dollar_drawdown >= 0 ? '+$' : '-$';
        const pctColor = st.drawdown_pct < -20 ? 'text-rose-400' : (st.drawdown_pct < 0 ? 'text-amber-400' : 'text-emerald-400');
        const dollarColor = st.dollar_drawdown < 0 ? 'text-rose-300/80' : 'text-emerald-300/80';
        div.innerHTML = `
          <div>
            <div class="flex items-start justify-between gap-2 mb-1.5">
              <div class="flex items-center gap-1.5 flex-wrap">
                ${badgeHtml}
                <span class="font-bold text-xs text-white leading-snug">${st.name}</span>
              </div>
              <span class="font-mono font-bold text-xs shrink-0 ${pctColor}">${drawdownSign}${st.drawdown_pct.toFixed(1)}%</span>
            </div>
            <span class="text-xs ${dollarColor} font-mono block">${dollarSign}${Math.abs(st.dollar_drawdown).toLocaleString(undefined, {minimumFractionDigits: 2})}</span>
          </div>
          <p class="text-[11px] text-slate-400 leading-relaxed font-sans pt-1 border-t border-slate-900">${st.description}</p>
        `;
        sc.appendChild(div);
      });
    }

    function updateWhatIfDetection(val) {
      const badge = document.getElementById('whatIfDetected');
      if (!badge) return;
      if (!val || !val.trim()) {
        badge.classList.add('hidden');
        return;
      }
      const info = resolveStock(val.trim());
      if (info && info.ticker) {
        badge.innerText = `✨ AI: ${info.name || info.ticker} (${info.ticker})`;
        badge.classList.remove('hidden');
      } else {
        badge.classList.add('hidden');
      }
    }

    async function handleWhatIf() {
      const rawInput = document.getElementById('whatIfTicker').value.trim();
      const pct = parseFloat(document.getElementById('whatIfPct').value) || 0;
      if (!rawInput || !currentPortfolio) return;

      const resolved = resolveStock(rawInput);
      const ticker = resolved.ticker;
      const assetName = resolved.name || STOCK_NAMES[ticker] || ticker;

      updateWhatIfDetection(rawInput);

      const targetAlloc = {};
      targetAlloc[ticker] = pct;
      targetAlloc['SPY'] = Math.max(0, 100 - pct);

      try {
        const res = await fetch('/api/what-if', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ portfolio: currentPortfolio, target_allocations: targetAlloc })
        });
        const data = await res.json();
        const box = document.getElementById('whatIfResultBox');
        box.classList.remove('hidden');

        const deltaVol = data.metrics_delta.volatility_change;
        const deltaHHI = data.metrics_delta.hhi_change;
        const targetTitle = (data.resolved_target && data.resolved_target.name) 
          ? `${data.resolved_target.name} (${data.resolved_target.ticker})` 
          : `${assetName} (${ticker})`;

        box.innerHTML = `
          <div class="border-b border-slate-800 pb-2">
            <div class="flex items-center justify-between mb-1">
              <span class="text-slate-400 text-[11px]">Proposed Single-Asset Concentration</span>
              <span class="text-[10px] text-brand-400 font-mono font-medium">${targetTitle}</span>
            </div>
            <span class="text-base font-bold text-white">${data.proposed_metrics.top_1_concentration}% (Tier: ${data.proposed_metrics.concentration_tier})</span>
            <span class="text-[11px] text-slate-500 block">HHI delta: ${deltaHHI > 0 ? '+' : ''}${deltaHHI.toFixed(1)}</span>
          </div>
          <div class="border-b border-slate-800 pb-2">
            <span class="text-slate-400 block text-[11px]">Annualized Volatility Shift</span>
            <span class="text-sm font-bold ${deltaVol > 0 ? 'text-rose-400' : 'text-emerald-400'}">
              ${data.current_metrics.annual_volatility}% &rarr; ${data.proposed_metrics.annual_volatility}% (${deltaVol > 0 ? '+' : ''}${deltaVol.toFixed(1)}%)
            </span>
          </div>
          <div>
            <span class="text-slate-400 block text-[11px]">Loss Capacity Check:</span>
            <span class="text-[11px] ${data.profile_warnings.length > 0 ? 'text-amber-400' : 'text-emerald-400'}">
              ${data.profile_warnings.length > 0 ? '⚠️ ' + data.profile_warnings.join('<br/>') : '✅ Within stated loss capacity bounds.'}
            </span>
          </div>
        `;
      } catch (e) {
        console.error("What-If failed:", e);
      }
    }

    async function handleHeadlineAudit() {
      const hl = document.getElementById('headlineInput').value.trim();
      if (!hl || !currentPortfolio) return;

      try {
        const res = await fetch('/api/headline', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ portfolio: currentPortfolio, headline: hl })
        });
        const data = await res.json();
        const box = document.getElementById('headlineResultBox');
        box.classList.remove('hidden');

        let holdingsHtml = data.affected_holdings.map(h => `
          <div class="flex justify-between items-center py-1 border-b border-slate-800/60 text-xs">
            <span><strong class="text-white">${h.name || h.ticker}</strong> <span class="text-slate-400 font-mono text-[10px]">(${h.ticker})</span> (${h.portfolio_weight}%)</span>
            <span class="${h.estimated_shock_pct < 0 ? 'text-rose-400' : 'text-emerald-400'} font-semibold font-mono">
              ${h.estimated_shock_pct > 0 ? '+' : ''}${h.estimated_shock_pct}% ($${h.estimated_dollar_impact.toLocaleString(undefined, {minimumFractionDigits: 2})})
            </span>
          </div>
        `).join('');

        box.innerHTML = `
          <div class="border-b border-slate-800 pb-2">
            <span class="text-slate-400 block text-[11px]">Net Portfolio Estimated Shock</span>
            <div class="text-lg font-bold font-mono ${data.net_estimated_impact_pct < 0 ? 'text-rose-400' : 'text-emerald-400'}">
              ${data.net_estimated_impact_pct > 0 ? '+' : ''}${data.net_estimated_impact_pct}% ($${data.net_estimated_dollar_impact.toLocaleString(undefined, {minimumFractionDigits: 2})})
            </div>
            <p class="text-slate-400 text-[11px] font-sans mt-1 leading-relaxed">${data.rationale}</p>
          </div>
          <div>
            <span class="text-slate-400 block text-[11px] mb-1">Most Sensitive Holdings:</span>
            ${holdingsHtml}
          </div>
        `;
      } catch (e) {
        console.error("Headline audit failed:", e);
      }
    }

    async function handleChatSubmit(e) {
      e.preventDefault();
      const input = document.getElementById('chatInput');
      const query = input.value.trim();
      if (!query || !currentPortfolio) return;

      input.value = '';
      const chatContainer = document.getElementById('chatMessages');

      // Append User message
      const userDiv = document.createElement('div');
      userDiv.className = "bg-slate-850 border border-slate-800 p-3 rounded-lg text-slate-200 self-end ml-8";
      userDiv.innerHTML = `<span class="font-bold text-slate-400 block mb-1">You:</span> ${escapeHtml(query)}`;
      chatContainer.appendChild(userDiv);

      // Loading placeholder
      const loadingDiv = document.createElement('div');
      loadingDiv.className = "bg-slate-900 border border-slate-800 p-3 rounded-lg text-slate-400 italic";
      loadingDiv.id = "chatLoading";
      loadingDiv.innerText = "Analyzing portfolio exposures and quantitative risk metrics...";
      chatContainer.appendChild(loadingDiv);
      chatContainer.scrollTop = chatContainer.scrollHeight;

      try {
        const res = await fetch('/api/chat', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ portfolio: currentPortfolio, query: query })
        });
        const data = await res.json();
        
        loadingDiv.remove();

        const agentDiv = document.createElement('div');
        agentDiv.className = "bg-slate-900 border border-slate-800 p-3.5 rounded-lg text-slate-200 space-y-2";
        agentDiv.innerHTML = `
          <div class="flex items-center justify-between border-b border-slate-800 pb-1.5 mb-1.5">
            <span class="font-bold text-brand-400">Portfolio X-Ray Analyst</span>
            <span class="text-[10px] text-slate-500 font-mono">${data.source || 'risk-analyst'}</span>
          </div>
          <div class="prose prose-invert prose-xs max-w-none leading-relaxed">${formatMarkdown(data.response)}</div>
        `;
        chatContainer.appendChild(agentDiv);
        chatContainer.scrollTop = chatContainer.scrollHeight;
      } catch (err) {
        loadingDiv.innerText = "Error analyzing question. Please verify connection.";
      }
    }

    function escapeHtml(text) {
      return text.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
    }

    function formatMarkdown(md) {
      if (!md) return '';
      return md
        .replace(/### (.*?)\\n/g, '<h4 class="font-bold text-white text-sm mt-2 mb-1">$1</h4>')
        .replace(/\\*\\*(.*?)\\*\\*/g, '<strong class="text-white font-semibold">$1</strong>')
        .replace(/\\* (.*?)\\n/g, '<div class="flex items-start gap-1.5 my-1"><span class="text-brand-400">&bull;</span><span>$1</span></div>')
        .replace(/\\n/g, '<br/>');
    }

    window.onload = init;
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
