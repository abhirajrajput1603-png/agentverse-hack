"""Portfolio X-Ray: Zero-dependency local web server & interactive dashboard.

Runs using Python standard library http.server on port 8080 (or specified port).
"""

from __future__ import annotations
import json
import urllib.parse
from http import HTTPStatus
from http.server import HTTPServer, BaseHTTPRequestHandler
import sys
from pathlib import Path
from typing import Dict, Any

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from portfolio_xray.data import SAMPLE_PORTFOLIOS
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

    def _set_headers(self, content_type: str = "application/json"):
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_OPTIONS(self):
        self._set_headers()

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/" or parsed.path == "/index.html":
            self._set_headers("text/html; charset=utf-8")
            self.wfile.write(HTML_DASHBOARD.encode("utf-8"))
        elif parsed.path == "/api/presets":
            self._set_headers("application/json")
            self.wfile.write(json.dumps(SAMPLE_PORTFOLIOS).encode("utf-8"))
        else:
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length).decode("utf-8")
        data = json.loads(body) if body else {}

        if parsed.path == "/api/analyze":
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
            self._set_headers("application/json")
            self.wfile.write(json.dumps(response).encode("utf-8"))

        elif parsed.path == "/api/what-if":
            portfolio = parse_portfolio_payload(data.get("portfolio", {}))
            target_allocations = data.get("target_allocations", {})
            result = simulate_what_if(portfolio, target_allocations)
            self._set_headers("application/json")
            self.wfile.write(json.dumps(result).encode("utf-8"))

        elif parsed.path == "/api/headline":
            portfolio = parse_portfolio_payload(data.get("portfolio", {}))
            headline = data.get("headline", "")
            result = analyze_headline_event(portfolio, headline)
            self._set_headers("application/json")
            self.wfile.write(json.dumps(result).encode("utf-8"))

        elif parsed.path == "/api/chat":
            portfolio = parse_portfolio_payload(data.get("portfolio", {}))
            query = data.get("query", "")
            result = ask_risk_analyst(portfolio, query)
            self._set_headers("application/json")
            self.wfile.write(json.dumps(result).encode("utf-8"))

        else:
            self.send_error(HTTPStatus.NOT_FOUND, "API route not found")


HTML_DASHBOARD = """<!DOCTYPE html>
<html lang="en" class="dark">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Portfolio X-Ray | AI-Powered Portfolio Risk Intelligence</title>
  <script src="https://cdn.tailwindcss.com"></script>
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
          <span class="text-xs px-2 py-0.5 rounded-full bg-brand-500/20 text-brand-300 font-mono font-medium border border-brand-500/30">v0 Demo</span>
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

    <!-- Preset Portfolio & Profile Bar -->
    <div class="bg-slate-900 border border-slate-800 rounded-xl p-4 shadow-sm flex flex-col md:flex-row items-start md:items-center justify-between gap-4">
      <div class="flex-1">
        <label class="text-xs font-semibold text-slate-400 uppercase tracking-wider block mb-1">Load Portfolio Preset</label>
        <div class="flex flex-wrap gap-2" id="presetButtons">
          <!-- Populated by JS -->
        </div>
      </div>
      <div class="flex items-center gap-3 self-end md:self-auto">
        <div class="text-xs">
          <span class="text-slate-400 block font-mono">Investor Profile</span>
          <select id="riskProfileSelect" class="bg-slate-800 border border-slate-700 text-slate-200 text-xs rounded-lg px-2.5 py-1.5 focus:ring-1 focus:ring-brand-500" onchange="updateProfile()">
            <option value="Conservative">Conservative (Max 10% drawdown)</option>
            <option value="Moderate" selected>Moderate (Max 15-20% drawdown)</option>
            <option value="Aggressive">Aggressive (Max 30%+ drawdown)</option>
          </select>
        </div>
      </div>
    </div>

    <!-- Metrics Stat Cards -->
    <div class="grid grid-cols-2 md:grid-cols-4 gap-4" id="statCards">
      <div class="bg-slate-900 border border-slate-800 rounded-xl p-4">
        <span class="text-xs text-slate-400 font-mono">Total Capital</span>
        <div class="text-2xl font-bold font-mono text-white mt-1" id="statVal">$0.00</div>
        <span class="text-xs text-slate-500 mt-1 block">Live Book Value</span>
      </div>
      <div class="bg-slate-900 border border-slate-800 rounded-xl p-4">
        <span class="text-xs text-slate-400 font-mono">Concentration (HHI)</span>
        <div class="text-2xl font-bold font-mono text-amber-400 mt-1" id="statHHI">0.0</div>
        <span class="text-xs text-amber-300/80 mt-1 block font-medium" id="statTier">Calculating...</span>
      </div>
      <div class="bg-slate-900 border border-slate-800 rounded-xl p-4">
        <span class="text-xs text-slate-400 font-mono">Annualized Volatility</span>
        <div class="text-2xl font-bold font-mono text-cyan-400 mt-1" id="statVol">0.0%</div>
        <span class="text-xs text-slate-400 mt-1 block">Cross-Asset Risk (1-Yr)</span>
      </div>
      <div class="bg-slate-900 border border-slate-800 rounded-xl p-4">
        <span class="text-xs text-slate-400 font-mono">95% 1-Month VaR</span>
        <div class="text-2xl font-bold font-mono text-rose-400 mt-1" id="statVaR">$0.00</div>
        <span class="text-xs text-rose-400/80 mt-1 block font-mono" id="statVaRPct">(0.0%)</span>
      </div>
    </div>

    <!-- Grid: Look-Through Decomposition & Thematic Exposures -->
    <div class="grid grid-cols-1 lg:grid-cols-12 gap-6">

      <!-- Look-Through Holdings Table (7 cols) -->
      <div class="lg:col-span-7 bg-slate-900 border border-slate-800 rounded-xl p-5 shadow-sm space-y-4">
        <div class="flex items-center justify-between">
          <div>
            <h2 class="text-base font-bold text-white flex items-center gap-2">
              <span>Look-Through Exposure Audit</span>
              <span class="text-xs font-normal text-slate-400">(True Ownership)</span>
            </h2>
            <p class="text-xs text-slate-400">Decomposes direct stocks and overlapping ETF baskets</p>
          </div>
          <span class="text-xs px-2.5 py-1 rounded bg-brand-500/10 text-brand-400 font-mono border border-brand-500/20">Direct + Indirect</span>
        </div>

        <div class="overflow-x-auto">
          <table class="w-full text-left text-xs">
            <thead class="text-slate-400 font-mono uppercase bg-slate-850/80 border-b border-slate-800">
              <tr>
                <th class="py-2.5 px-3">Asset</th>
                <th class="py-2.5 px-3">Total Exposure</th>
                <th class="py-2.5 px-3">Direct vs Indirect</th>
                <th class="py-2.5 px-3">Overlap Origin</th>
              </tr>
            </thead>
            <tbody id="exposureTableBody" class="divide-y divide-slate-800 font-mono">
              <!-- Populated by JS -->
            </tbody>
          </table>
        </div>
      </div>

      <!-- Thematic & Economic Drivers (5 cols) -->
      <div class="lg:col-span-5 bg-slate-900 border border-slate-800 rounded-xl p-5 shadow-sm space-y-4">
        <div>
          <h2 class="text-base font-bold text-white">Economic & Thematic Sensitivities</h2>
          <p class="text-xs text-slate-400">Shared macro drivers across seemingly different industries</p>
        </div>
        <div id="thematicBars" class="space-y-3.5">
          <!-- Populated by JS -->
        </div>
      </div>

    </div>

    <!-- Historical Stress Scenarios -->
    <div class="bg-slate-900 border border-slate-800 rounded-xl p-5 shadow-sm space-y-4">
      <div class="flex items-center justify-between">
        <div>
          <h2 class="text-base font-bold text-white">Historical & Macro Stress Simulations</h2>
          <p class="text-xs text-slate-400">What would happen to this exact portfolio under historical crisis events?</p>
        </div>
        <span class="text-xs text-slate-400 font-mono">Non-Predictive Scenarios</span>
      </div>
      <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4" id="stressCards">
        <!-- Populated by JS -->
      </div>
    </div>

    <!-- Interactive Tools: What-If Simulator & Headline Analyzer -->
    <div class="grid grid-cols-1 lg:grid-cols-12 gap-6">

      <!-- What-If Simulator (6 cols) -->
      <div class="lg:col-span-6 bg-slate-900 border border-slate-800 rounded-xl p-5 shadow-sm space-y-4">
        <div class="flex items-center justify-between">
          <div>
            <h2 class="text-base font-bold text-white">"What-If" Decision Simulator</h2>
            <p class="text-xs text-slate-400">Test trade decisions before acting on them</p>
          </div>
          <span class="text-xs px-2 py-0.5 rounded bg-cyan-500/10 text-cyan-400 font-mono border border-cyan-500/20">Pre-Trade Risk</span>
        </div>

        <div class="space-y-3">
          <div class="flex gap-2">
            <button onclick="runQuickWhatIf('NVDA', 50)" class="text-xs px-2.5 py-1.5 rounded bg-slate-800 hover:bg-slate-700 text-slate-300 border border-slate-700 font-mono">
              + 50% into NVDA
            </button>
            <button onclick="runQuickWhatIf('TLT', 30)" class="text-xs px-2.5 py-1.5 rounded bg-slate-800 hover:bg-slate-700 text-slate-300 border border-slate-700 font-mono">
              + 30% into Treasuries (TLT)
            </button>
            <button onclick="runQuickWhatIf('CASH', 25)" class="text-xs px-2.5 py-1.5 rounded bg-slate-800 hover:bg-slate-700 text-slate-300 border border-slate-700 font-mono">
              + 25% into Cash
            </button>
          </div>

          <div id="whatIfResultBox" class="bg-slate-850 border border-slate-800 rounded-lg p-4 text-xs font-mono space-y-3 hidden">
            <!-- Populated by JS -->
          </div>
        </div>
      </div>

      <!-- Headline & News Analyzer (6 cols) -->
      <div class="lg:col-span-6 bg-slate-900 border border-slate-800 rounded-xl p-5 shadow-sm space-y-4">
        <div>
          <h2 class="text-base font-bold text-white">Headline & Event Shock Impact</h2>
          <p class="text-xs text-slate-400">See how macro news relates specifically to your holdings</p>
        </div>

        <div class="space-y-2">
          <div class="flex gap-2">
            <input type="text" id="headlineInput" class="flex-1 bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-xs text-white placeholder-slate-500 focus:ring-1 focus:ring-brand-500" placeholder="e.g. Fed unexpectedly raises rates by 50 bps...">
            <button onclick="analyzeHeadline()" class="bg-brand-600 hover:bg-brand-500 text-white font-medium text-xs px-4 py-2 rounded-lg transition">Audit Shock</button>
          </div>
          <div class="flex flex-wrap gap-1.5">
            <button onclick="setHeadline('Federal Reserve signals higher rates for longer due to sticky inflation')" class="text-[11px] text-slate-400 hover:text-white bg-slate-800 px-2 py-1 rounded">Fed Rate Hike</button>
            <button onclick="setHeadline('Hyperscalers announce major slowdown in AI data center capex spending')" class="text-[11px] text-slate-400 hover:text-white bg-slate-800 px-2 py-1 rounded">AI Capex Pause</button>
            <button onclick="setHeadline('Middle East crude oil supply disruption drives Brent over $115')" class="text-[11px] text-slate-400 hover:text-white bg-slate-800 px-2 py-1 rounded">Oil Spike</button>
          </div>
        </div>

        <div id="headlineResultBox" class="bg-slate-850 border border-slate-800 rounded-lg p-4 text-xs font-mono space-y-2 hidden">
          <!-- Populated by JS -->
        </div>
      </div>

    </div>

    <!-- Conversational AI Risk Assistant -->
    <div class="bg-slate-900 border border-slate-800 rounded-xl p-5 shadow-sm space-y-4">
      <div class="flex items-center justify-between">
        <div class="flex items-center gap-3">
          <div class="w-8 h-8 rounded-full bg-brand-500/20 text-brand-400 flex items-center justify-center font-bold text-xs border border-brand-500/30">
            AI
          </div>
          <div>
            <h2 class="text-base font-bold text-white">Conversational AI Risk Assistant</h2>
            <p class="text-xs text-slate-400">Ask questions about hidden vulnerabilities, hypothetical moves, or risk concepts</p>
          </div>
        </div>
        <span class="text-xs px-2 py-1 rounded bg-slate-800 text-slate-400 font-mono">Traceable Reasoning</span>
      </div>

      <div id="chatMessages" class="bg-slate-950 border border-slate-800 rounded-xl p-4 min-h-[160px] max-h-[380px] overflow-y-auto space-y-3 text-xs leading-relaxed">
        <div class="bg-slate-900 border border-slate-800 p-3 rounded-lg text-slate-300">
          <span class="font-bold text-brand-400 block mb-1">Portfolio X-Ray Assistant:</span>
          Welcome! I have audited your current holdings. Feel free to ask me questions like:
          <ul class="list-disc list-inside mt-1 text-slate-400 space-y-0.5">
            <li>"What is my real total exposure to Nvidia across all my ETFs?"</li>
            <li>"What happens if I put 50% of my money into Nvidia?"</li>
            <li>"Explain why my portfolio is vulnerable to an interest rate hike."</li>
          </ul>
        </div>
      </div>

      <form onsubmit="handleChatSubmit(event)" class="flex gap-2">
        <input type="text" id="chatInput" class="flex-1 bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-xs text-white placeholder-slate-500 focus:ring-1 focus:ring-brand-500" placeholder="Ask your portfolio risk analyst...">
        <button type="submit" id="chatSendBtn" class="bg-brand-600 hover:bg-brand-500 text-white font-medium text-xs px-5 py-2 rounded-lg transition flex items-center gap-1.5">
          <span>Ask Analyst</span>
        </button>
      </form>
    </div>

  </main>

  <footer class="border-t border-slate-800 py-4 px-6 text-center text-xs text-slate-500 font-mono">
    Portfolio X-Ray &bull; Educational Risk Intelligence Platform &bull; Non-advisory educational prototype for AgentVerse 2026
  </footer>

  <script>
    let currentPortfolio = null;
    let presets = {};

    async function init() {
      try {
        const res = await fetch('/api/presets');
        presets = await res.json();
        renderPresetButtons();
        // Load default preset
        loadPreset('accidental_tech_overweight');
      } catch (e) {
        console.error("Init failed:", e);
      }
    }

    function renderPresetButtons() {
      const container = document.getElementById('presetButtons');
      container.innerHTML = '';
      for (const [key, p] of Object.entries(presets)) {
        const btn = document.createElement('button');
        btn.className = "text-xs px-3 py-1.5 rounded-lg border font-medium transition " +
          (key === 'accidental_tech_overweight' ? "bg-brand-600/20 border-brand-500/40 text-brand-300" : "bg-slate-800 border-slate-700 text-slate-300 hover:bg-slate-750");
        btn.innerText = p.name.split(' (')[0];
        btn.onclick = () => loadPreset(key);
        btn.id = "preset-" + key;
        container.appendChild(btn);
      }
    }

    function loadPreset(key) {
      if (!presets[key]) return;
      currentPortfolio = JSON.parse(JSON.stringify(presets[key]));
      document.getElementById('riskProfileSelect').value = currentPortfolio.user_profile.risk_tolerance;

      // Update button styles
      for (const k of Object.keys(presets)) {
        const el = document.getElementById('preset-' + k);
        if (el) {
          if (k === key) {
            el.className = "text-xs px-3 py-1.5 rounded-lg border font-medium transition bg-brand-600/20 border-brand-500/40 text-brand-300";
          } else {
            el.className = "text-xs px-3 py-1.5 rounded-lg border font-medium transition bg-slate-800 border-slate-700 text-slate-300 hover:bg-slate-750";
          }
        }
      }

      runAnalysis();
    }

    function updateProfile() {
      if (!currentPortfolio) return;
      currentPortfolio.user_profile.risk_tolerance = document.getElementById('riskProfileSelect').value;
      runAnalysis();
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
      document.getElementById('statHHI').innerText = m.hhi.toFixed(1);
      document.getElementById('statTier').innerText = m.concentration_tier;
      document.getElementById('statVol').innerText = m.annual_volatility.toFixed(1) + '%';
      document.getElementById('statVaR').innerText = '$' + m.var_95_1m_dollar.toLocaleString(undefined, {minimumFractionDigits: 2});
      document.getElementById('statVaRPct').innerText = '(' + m.var_95_1m_pct.toFixed(1) + '% of capital)';

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
      data.thematic.slice(0, 5).forEach(th => {
        const topTickers = th.top_assets.map(a => a.ticker).join(', ');
        const div = document.createElement('div');
        div.className = "space-y-1";
        div.innerHTML = `
          <div class="flex justify-between items-center text-xs">
            <span class="font-medium text-slate-200">${th.theme}</span>
            <span class="font-mono font-bold text-brand-300">${th.weight.toFixed(1)}%</span>
          </div>
          <div class="w-full bg-slate-800 h-2 rounded-full overflow-hidden">
            <div class="bg-gradient-to-r from-brand-500 to-cyan-400 h-full rounded-full" style="width: ${Math.min(100, th.weight)}%"></div>
          </div>
          <span class="text-[10px] text-slate-400 block font-mono">Top drivers: ${topTickers}</span>
        `;
        thContainer.appendChild(div);
      });

      // Render Stress Test Cards
      const sc = document.getElementById('stressCards');
      sc.innerHTML = '';
      data.stress_tests.forEach(st => {
        const div = document.createElement('div');
        div.className = "bg-slate-850 border border-slate-800 rounded-xl p-4 flex flex-col justify-between";
        div.innerHTML = `
          <div>
            <span class="text-xs font-bold text-white block mb-1">${st.name}</span>
            <p class="text-[11px] text-slate-400 leading-relaxed mb-3">${st.description}</p>
          </div>
          <div class="pt-2 border-t border-slate-800/80">
            <span class="text-[11px] text-slate-400 block font-mono">Estimated Drawdown</span>
            <div class="text-xl font-bold font-mono ${st.drawdown_pct < 0 ? 'text-rose-400' : 'text-emerald-400'}">
              ${st.drawdown_pct.toFixed(1)}%
            </div>
            <span class="text-[11px] font-mono text-slate-400">$${st.dollar_drawdown.toLocaleString(undefined, {minimumFractionDigits: 2})}</span>
          </div>
        `;
        sc.appendChild(div);
      });
    }

    async function runQuickWhatIf(ticker, pct) {
      if (!currentPortfolio) return;
      const target = {};
      target[ticker] = pct;
      target["SPY"] = 100 - pct;

      try {
        const res = await fetch('/api/what-if', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ portfolio: currentPortfolio, target_allocations: target })
        });
        const sim = await res.json();
        
        const box = document.getElementById('whatIfResultBox');
        box.classList.remove('hidden');

        let warningHtml = '';
        if (sim.profile_warnings.length > 0) {
          warningHtml = `<div class="p-2.5 rounded bg-rose-500/10 border border-rose-500/30 text-rose-300 font-sans text-xs">
            <strong class="font-bold">⚠️ Risk Profile Alert:</strong> ${sim.profile_warnings.join(' ')}
          </div>`;
        } else {
          warningHtml = `<div class="p-2.5 rounded bg-emerald-500/10 border border-emerald-500/30 text-emerald-300 font-sans text-xs">
            <strong class="font-bold">✅ Profile Aligned:</strong> Fits within your stated risk tolerance bounds.
          </div>`;
        }

        box.innerHTML = `
          <div class="flex justify-between items-center border-b border-slate-800 pb-2">
            <span class="font-bold text-white text-sm">Simulation: ${pct}% Target in ${ticker}</span>
            <span class="text-[11px] text-slate-400">Pre-Trade Comparison</span>
          </div>
          <div class="grid grid-cols-2 gap-3 text-xs">
            <div>
              <span class="text-slate-400 block">Annual Volatility:</span>
              <span class="text-slate-300">${sim.current_metrics.annual_volatility}% &rarr; <strong class="text-white">${sim.proposed_metrics.annual_volatility}%</strong> (${sim.metrics_delta.volatility_change > 0 ? '+' : ''}${sim.metrics_delta.volatility_change}%)</span>
            </div>
            <div>
              <span class="text-slate-400 block">Top 1 Concentration:</span>
              <span class="text-slate-300">${sim.current_metrics.top_1_concentration}% &rarr; <strong class="text-white">${sim.proposed_metrics.top_1_concentration}%</strong></span>
            </div>
            <div>
              <span class="text-slate-400 block">Concentration Tier:</span>
              <span class="text-amber-400 font-semibold">${sim.proposed_metrics.concentration_tier}</span>
            </div>
            <div>
              <span class="text-slate-400 block">95% 1-M VaR:</span>
              <span class="text-rose-400">$${sim.proposed_metrics.var_95_1m_dollar.toLocaleString()}</span>
            </div>
          </div>
          ${warningHtml}
        `;
      } catch (e) {
        console.error("What-if failed:", e);
      }
    }

    function setHeadline(text) {
      document.getElementById('headlineInput').value = text;
      analyzeHeadline();
    }

    async function analyzeHeadline() {
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
            <span><strong class="text-white">${h.ticker}</strong> (${h.portfolio_weight}%)</span>
            <span class="${h.estimated_shock_pct < 0 ? 'text-rose-400' : 'text-emerald-400'} font-semibold">
              ${h.estimated_shock_pct > 0 ? '+' : ''}${h.estimated_shock_pct}% ($${h.estimated_dollar_impact.toLocaleString()})
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


def run_server(port: int = 8080):
    server_address = ("", port)
    httpd = HTTPServer(server_address, PortfolioXRayHandler)
    print(f"Portfolio X-Ray Web Dashboard running on http://localhost:{port}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server.")
        httpd.server_close()


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8080
    run_server(port)
