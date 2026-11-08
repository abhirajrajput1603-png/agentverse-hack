# Portfolio X-Ray: Institutional & Consumer Portfolio Risk Intelligence

> **Autonomous portfolio risk management, look-through decomposition, 15-year historical stress testing, and safety-gated scenario intelligence designed for everyday investors.**
> Built for the **UCL AgentVerse 2026 Finance Track** (*Jane Street · Citadel · Marshall Wace · BlackRock · Airwallex · Stripe*).

---

## 💡 The Core Problem: The "Diversification Illusion"

Most retail investors know which stocks and ETFs they bought, but not what they are **actually betting on**.

An investor holding Nvidia (`NVDA`) alongside popular index and sector ETFs (`QQQ`, `VOO`, `SMH`) may believe they are safely diversified across 4 separate investments. In reality, **Portfolio X-Ray's look-through decomposition** reveals:
- **34.7% single-issuer exposure to Nvidia alone** (`[EV-LT-NVDA]`), breaching a moderate investor's 20% single-stock circuit breaker (`[EV-CONC-TOP1-WEIGHT]`).
- **Over 59% thematic concentration** in *AI Data Center & Power Capex* across seemingly distinct holdings (`[EV-THEME-AI_DATA_CENTER_AND_POWER_CAPEX]`).
- **Disproportionate Euler Risk Contribution:** A holding with 20% of capital can generate over 33% of total portfolio volatility (`[EV-CORR-TOP-RISK-DRIVER]`).

---

## 🚀 Quick Start

### 1. Launch the Unified Consumer + Institutional Web Dashboard (`v-ui/server.py`)
```bash
python3 v-ui/server.py 8080
# or from root:
python3 server.py 8080
```
Open **[http://localhost:8080](http://localhost:8080)** in your browser.

### 2. Optional: Launch the Secondary Streamlit Institutional Terminal (`app.py`)
```bash
python3 -m streamlit run app.py --server.port 8501
```

### 3. Run the Full Automated Test Suite (82 Unit, Integration & Adversarial Tests)
```bash
python3 -m pytest -v
```

### 4. Run the Full Benchmark, Holdout Ablation & Out-of-Sample VaR Evaluation
```bash
python3 eval/run_eval.py
```

---

## 🏛️ Unified System Architecture

Portfolio X-Ray combines a **clean, beginner-friendly consumer fintech web dashboard (`v-ui/`)** with a **strictly verified institutional quantitative & safety engine (`xray/`)** connected via `xray/bridge.py`:

```text
├── server.py                      # Root entrypoint delegating to v-ui/server.py
├── v-ui/                          # Consumer Fintech Web Application & AI Assistant
│   ├── server.py                  # HTTP Server + 5-Tab Interactive Web Dashboard (Overview, Hidden Risks, What Could Happen?, Try a Change, Ask X-Ray)
│   ├── agent.py                   # Autonomous Tool-Calling AI Risk Analyst + Scope Guardrail + OutputSafetyFilter
│   ├── engine.py                  # Parametric & Headline Shock Engine + Custom Ticker Fallback
│   ├── data.py                    # Consumer Presets, 4 AgentVerse Benchmark Cases & ETF Metadata
│   └── cli.py                     # CLI Portfolio Risk Auditor
├── xray/                          # Institutional Quantitative & Safety Architecture
│   ├── bridge.py                  # Unified Bridge connecting v-ui/ and xray/ engines
│   ├── evidence.py                # Traceable Evidence (EV-*) & EvidenceRegistry (Zero LLM Mental Math)
│   ├── profile.py                 # UserRiskProfile mapped from 4-Question Risk Questionnaire
│   ├── data_loader.py             # 15-Year Real Market Price Matrix & SEC N-PORT ETF Loader
│   ├── engine/
│   │   ├── lookthrough.py         # Full 100% Weight-Conserving Look-Through + Dual-Share-Class Merger (GOOGL/GOOG, BRK.B/BRK-B)
│   │   ├── concentration.py       # Direct vs. Look-Through HHI, Effective N (1/HHI), Top-1/Top-3/Top-5
│   │   ├── themes.py              # Non-Additive Macroeconomic Theme Clustering
│   │   ├── risk.py                # 3Y Annualized Vol across Sample, Ledoit-Wolf Shrinkage & EWMA (λ=0.94) + Max Drawdown
│   │   ├── stress.py              # Historical 95%/99% VaR & ES with 1,000-Sample Bootstrap CIs + 4 Real Crisis Replays
│   │   ├── correlation.py         # 1Y/3Y Pairwise Correlation Matrices + Euler Risk Decomposition (MCR, CCR, %CR)
│   │   ├── benchmark.py           # Side-by-Side Comparison vs. SPY (100% Equity) & 60/40 SPY/BND + Active Tilts
│   │   └── whatif.py              # What-If Simulator with Realistic Trade Friction (Spread, Commission, Slippage in bps)
│   ├── safety/
│   │   ├── limits.py              # Hard Circuit Breakers (CONCENTRATION_HARD_LIMIT, VOLATILITY_REGIME_ALERT, DATA_INTEGRITY_HALT)
│   │   ├── output_filter.py       # OutputSafetyFilter (Blocks Buy/Sell Directives, Price Targets & Ungrounded Numbers)
│   │   └── audit.py               # Append-Only Immutable Audit Trail (logs/audit.jsonl)
│   └── agent/
│       └── investigator.py        # Autonomous AgentToolRunner (Capped at 8 Tool Calls) & Citation Validator
├── data/
│   ├── prices.csv                 # 15-Year Historical Daily Market Prices (3,970 Trading Days, 2010–2026)
│   ├── etf_constituents.json      # Authoritative ETF Constituent Weights + Residual Buckets Summing to 1.0000
│   ├── themes.json                # Macroeconomic Theme Definitions
│   ├── issuer_map.json            # Dual-Share-Class Issuer Mapping
│   ├── synthetic_benchmark_portfolios.json
│   └── holdout/                   # 10 Strictly Held-Out Evaluation Portfolios
├── eval/
│   ├── run_eval.py                # Full Benchmark, 4-System Holdout Ablation & Out-of-Sample VaR Backtest
│   ├── xray_bench_adapter.py      # Official AgentVerse finance/bench.py Adapter (10/10 PASSED)
│   └── results/                   # Saved JSON & Markdown Evaluation Reports
└── tests/                         # 82 Automated Unit, Integration, Holdout, Server & Adversarial Safety Tests
```

---

## 🔒 Non-Negotiable Safety & Mathematical Invariants

1. **Jane Street Math Invariant (Zero LLM Mental Math):** Every quantitative figure displayed in the UI or cited by the AI is computed deterministically in Python and backed by a clickable `[EV-*]` proof chip showing its exact formula, inputs, and data sources.
2. **Exact Weight & Risk Conservation:**
   - Full look-through decomposition strictly conserves portfolio weights (`|sum(w) - 1.0| < 1e-7`).
   - Euler Component Contributions to Risk strictly sum to portfolio volatility (`|sum(CCR_i) - sigma_p| < 1e-6`).
   - Tail risk metrics strictly satisfy `ES_99 >= VaR_99 >= VaR_95` and `ES_95 >= VaR_95`.
3. **Hard Circuit Breakers & Output Safety Filter:**
   - Every portfolio analysis, What-If simulation, human sign-off, and AI chat response is gated by `SafetyLimitsEngine`, `OutputSafetyFilter` (100% block rate across 18 adversarial prompt-injection, buy/sell, price-prediction, and out-of-scope test cases), and logged to `logs/audit.jsonl`.
