# Portfolio X-Ray: AI-Powered Portfolio Risk Intelligence

> **Autonomous portfolio risk management, look-through decomposition, and scenario intelligence designed for retail investors.**
> Developed for the **AgentVerse 2026 Finance Track** (Trustworthy AI Agents, Traceable Reasoning, Risk Controls).

---

## 💡 The Core Problem

Most retail investors know what stocks and ETFs they own, but not what they are **actually betting on**. 

An investor holding Nvidia alongside broad tech and semiconductor funds (like QQQ, VOO, and SMH) may believe they are well-diversified across four different assets. In reality, **look-through decomposition** reveals they hold over **28–32% effective exposure in Nvidia alone**, with nearly 50% tied to AI data center infrastructure capex.

**Portfolio X-Ray** autonomously investigates an investor's holdings to expose:
1. **Direct & Indirect Look-Through Exposure**: Pinpointing hidden overlapping ownership across ETF baskets.
2. **Economic & Thematic Sensitivities**: Identifying common economic drivers (AI Infrastructure Capex, Interest Rate Duration, Energy shocks) shared by seemingly unrelated investments.
3. **Statistical & Quantitative Risk Analytics**: Herfindahl-Hirschman Index (HHI) concentration, annualized cross-asset volatility, 95% 1-Month Value at Risk (VaR), and Expected Shortfall (CVaR).
4. **Historical Crisis Stress Simulations**: Measuring drawdown vulnerability under the 2022 Fed Rate Shock, 2020 Covid Liquidity Shock, AI Capex Reality Check, and Energy Stagflation.
5. **Interactive "What-If" Decision Simulator**: Auditing hypothetical allocation shifts (e.g. *"What if I put 50% into NVDA?"*) against the investor's stated risk tolerance and loss capacity.
6. **Headline & Macro Shock Impact**: Translating breaking economic news into estimated portfolio impact and identifying which specific holdings bear the brunt of the shock.
7. **Conversational AI Risk Analyst**: An evidence-backed, factual assistant powered by Gemini with deterministic offline fallback.

---

## 🚀 Quick Start (v0 Demo)

The v0 demo uses Python 3.9+ standard library with **zero external dependencies**.

### 1. Launch the Interactive Web Dashboard
```bash
python3 portfolio_xray/server.py 8080
```
Open **[http://localhost:8080](http://localhost:8080)** in your browser.

* Features available in the web dashboard:
  * **Preloaded Portfolios:** "The Accidental Tech Concentration", "Institutional Balanced 60/40", and "Aggressive Tech & AI Moonshot".
  * **Live Look-Through Table:** Shows direct vs. indirect weights and exact ETF overlap origins.
  * **Thematic Factor Gauges:** Visualizes primary macroeconomic sensitivities.
  * **Stress Test Grid:** Displays simulated dollar and percentage drawdowns.
  * **"What-If" Simulation Widget:** Test pre-trade allocation changes and see instant volatility/concentration deltas.
  * **Headline Shock Auditor:** Type any breaking news headline to inspect portfolio vulnerability.
  * **AI Risk Chat:** Ask questions directly to the Portfolio X-Ray agent.

### 2. Run CLI Audit
```bash
python3 portfolio_xray/cli.py
```

### 3. Run Unit Tests
```bash
python3 -m unittest discover -s tests -v
```

---

## 🏛️ Architecture & Verification

```text
portfolio_xray/
├── data.py          # ETF look-through weights (QQQ, SPY, VOO, SMH, XLK, TLT, etc.), asset catalog, & stress shock models
├── engine.py        # Deterministic quantitative risk engine (Look-through, HHI, Volatility, VaR, Stress, What-If)
├── agent.py         # AI Risk Analyst (Gemini API integration with deterministic offline fallback)
├── server.py        # Local zero-dependency HTTP server & Tailwind-powered institutional dark dashboard
└── cli.py           # Command-line portfolio audit interface
tests/
└── test_engine.py   # Unit test suite verifying calculations and counterexamples (e.g. all-cash safety)
```

### Engineering & Trustworthiness Rules Followed:
* **Separation of Math and Language:** All calculations (look-through weights, volatility, VaR, stress tests) are calculated in pure, deterministic functions testable without an LLM. The AI explains and interprets the numbers—it does not do the arithmetic.
* **Deterministic Fallback:** If internet is disconnected or API quota is reached, the platform falls back to an offline rule-based risk analyst that generates complete structured audits.
* **Counterexamples Tested:** Verified against counterexamples (e.g., an all-cash portfolio yields ~0% volatility, negligible VaR, and zero stress drawdown).
* **Cost & Security Boundary:** Operates using local standard libraries with no paid API or data subscriptions required.
