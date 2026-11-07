# Portfolio X-Ray: Benchmark & Holdout Datasets

This folder contains the official testing datasets for the **Portfolio X-Ray** agent, structured according to the **AgentVerse 2026 Finance Track** judging criteria.

## Files Overview

1. `etf_constituents.json`:
   - Authentic holding weights for top retail funds (QQQ, SPY, SMH, VTI, ARKK).
   - Sourced from SEC N-PORT quarterly filings and official fund provider daily disclosures.
   - Used by the deterministic look-through engine to calculate true underlying exposure.

2. `synthetic_benchmark_portfolios.json`:
   - 4 benchmark retail portfolios representing distinct risk traps:
     - **Case 1 (The Diversification Illusion):** Overlapping tech holdings masking 34.7% true Nvidia concentration.
     - **Case 2 (The AI Capex Theme):** Utilities, hardware, and cloud clustered around AI datacenter power.
     - **Case 3 (Rate Shock Duration Trap):** ARKK + TLT co-drawdown under unexpected rate hikes.
     - **Case 4 (Passive Benchmark):** Clean, globally diversified baseline.
   - Includes ground-truth mathematical answers for automated agent validation.

3. `historical_shock_data.json`:
   - Daily price and return series during two major unseen historical shock windows:
     - **August 2024 Tech & Yen Carry Flash Crash**
     - **March 2023 Silicon Valley Bank (SVB) Liquidity Shock**
   - Used to compute empirical Value at Risk (VaR) and stress-test what-if reallocations.

## License & Compliance (Hackathon Slide 17)
- Price data: Public exchange close records.
- ETF constituents: Public SEC regulatory disclosures.
- Synthetic portfolios: Created specifically for academic benchmarking at UCL AgentVerse 2026.
