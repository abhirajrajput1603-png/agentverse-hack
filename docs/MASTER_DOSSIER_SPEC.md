# PORTFOLIO X-RAY: FOUR-PAGE MASTER PROJECT SPECIFICATION & HACKATHON DOSSIER
*Agentverse 2026 — Finance Track: "Agents you would let near your own money"*  
Canonical Self-Contained Summary (Parts I–XXV / Sections 1–107) · British English Standard

---

# PAGE 1 OF 4: HACKATHON CONTEXT, PRODUCT PHILOSOPHY & THE THREE EXPOSURE DIMENSIONS
Synthesising Parts I–V (Sections 1–15): Competition Mandate, Non-Predictive Guardrails, Personalisation & Look-Through Ownership

	*Executive Proposition:* "Most investors know what they own. Far fewer understand what they are actually exposed to. You think your portfolio is diversified—our AI shows you what you're really betting on."

---

### 01. Agentverse 2026 Finance Track & Core Competition Strategy
•  *Theme & Central Challenge:* Developed for *Agentverse 2026 (Finance Track)* under the theme "Agents you would let near your own money" and the central challenge "Help people make better money decisions" without hiding the underlying reasoning or risk.
•  *Domain & Audience Bridge:* Addresses retail portfolio risk management, financial inclusion, and investment decision quality—making institutional-grade quantitative risk analysis understandable to everyday investors while remaining mathematically rigorous enough to withstand scrutiny from hedge fund founders, quantitative traders, and technical judges (noting Jane Street is a supporting organisation, though exact judge backgrounds are unconfirmed).
•  *Scoring Dynamics:* *Traceable reasoning (25%)* and *Testing on unseen data (25%)* constitute *50% of the total score, alongside **Relevance and impact (20%), **Risk controls and oversight (15%), and **Ambition and technical quality (15%)*. There is no leaderboard; transparent methodology, honest benchmark comparisons, and explicitly stated uncertainty score higher than exaggerated claims.
•  *Three Official Deliverables:*
  1. *Public Code Repository:* Reproducible implementation with a comprehensive README covering installation, architecture, data requirements, and evaluation methodology.
  2. *Pitch Deck:* The primary scoring artefact evaluated for every team—must include actual test results on unseen data, benchmark comparisons, architecture, risk controls, and limitations.
  3. *Live Pitch (Top 3 Teams):* Up to 4 minutes presenting + 2 minutes Q&A, requiring a tight, self-evident end-to-end live demonstration.
•  *Data & Competition Constraints:* Synthetic portfolios and test cases are explicitly allowed provided they are clearly labelled as synthetic, their construction methodology is explained, and synthetic backtests are never presented as real-world trading performance. Never use private financial records or real money execution. Permitted data sources include the Agentverse finance kit, SEC EDGAR, Companies House, FRED, Bank of England, ECB Data Portal, ONS, published ETF issuer disclosures, Binance Vision, and Polymarket public APIs.

---

### 02. Product Identity, Strict Boundaries & Regulatory Posture
•  *What Portfolio X-Ray IS:* An autonomous, educational portfolio risk-analysis and decision-support platform for everyday investors holding shares, ETFs, index trackers, mutual funds/investment trusts, bonds, REITs, commodities, exchange-traded crypto, or listed derivatives across ISAs, pensions, or general brokerage accounts.
•  *What Portfolio X-Ray is NOT:*
  * Not an automated trading system, broker, or portfolio rebalancer.
  * Not a stock-picking engine, alpha signal generator, or market-timing predictor.
  * Not a black-box recommendation tool or generic financial chatbot with an LLM bolted onto a static tracker.
•  *Non-Predictive & Non-Prescriptive Guardrails:* The platform never claims an asset will rise or fall ("Nvidia is going to drop") and never issues buy/sell trade instructions ("Sell £5,000 of Apple and buy £5,000 of gilts"). Instead, it quantifies how existing holdings or user-proposed scenarios alter single-issuer concentration, factor exposure, volatility, and tail risk—leaving 100% of decisions to the human user.
•  *Regulatory Awareness (UK FCA Context):* Kept strictly simulation-only and human-controlled for the hackathon, designed to stay on the educational risk-analysis side rather than crossing into regulated personal investment advice.

---

### 03. User Personas, Onboarding & Personalised Risk Interpretation
•  *Five Core User Personas:*
  1. Beginner Investor: Recently started, concentrated in popular tech shares, and heavily influenced by news headlines or social media.
  2. Passive ETF Investor: Holds multiple index trackers believing they are diversified, unaware that their funds hold the exact same top companies.
  3. Long-Term Pension Investor: Has a multi-decade horizon and can rationally accept higher volatility, but wants clarity on concentration and tail drawdowns.
  4. Shorter-Horizon Saver: Expects to withdraw funds in a few years (e.g., for a house deposit) and may not realise their equity allocation risks a sharp loss right before withdrawal.
  5. More Experienced Self-Directed Investor: Understands finance basics and wants correlation matrices, factor betas, and stress testing without building spreadsheets.
•  *Onboarding Profile Dimensions:* Captures:
  * Investment Context: Account type, base currency, value, objective, horizon, liquidity needs.
  * Financial Risk Capacity: Capital/percentage loss capacity before affecting important financial commitments.
  * Psychological Risk Tolerance: Volatility comfort, maximum tolerable drawdown.
  * Experience & Preferences: Familiarity with derivatives/leverage, simple vs. advanced explanation depth.
•  *Critical Rule — Account Wrapper Neutrality:* Holding a Stocks & Shares ISA or a pension does *not* automatically imply conservative or aggressive risk tolerance.
•  *Personalise Interpretation, Never Facts:* Underlying mathematical metrics remain invariant across users, while their interpretation adapts. A 35% historical crisis drawdown is flagged as a critical misalignment for a user with a 15% loss tolerance and a 3-year horizon, whereas it may align with a 25-year growth investor's profile. Never use arbitrary black-box scores such as "Risk Score: 82/100" or fixed Sharpe ratio rules as suitability thresholds.

---

### 04. Portfolio Ingestion, Asset Universe & Persistent Longitudinal State
•  *Extensible Asset Universe:* Long-term architecture supports all publicly traded instruments where reliable data and risk models exist. For the *Hackathon MVP, implementation prioritises accurate, verified coverage of **publicly traded equities and ETFs*, while explicitly reporting partial or unsupported coverage for complex instruments rather than fabricating metrics.
•  *Multimodal Screenshot & Manual Ingestion:* Extracts company/instrument names, tickers, ISIN/CUSIP identifiers, quantities, market values, weights, currencies, and cash balances into structured schemas.
•  *Human-in-the-Loop Verification:* Displays extracted positions for mandatory user confirmation—flagging low-confidence OCR, ambiguous tickers across exchanges, duplicate entries, missing quantities, or totals that fail to reconcile.
•  *Persistent State & Risk Drift Tracking:* Stores confirmed holdings snapshots, timestamps, risk profiles, acknowledged risks, and prior assessments. Distinguishes passive risk drift (e.g., tech exposure rising from 35% to 60% due to market appreciation) from active trades. Direct brokerage API sync is deferred to the commercial roadmap.

---

### 05. The Tripartite Exposure Framework (Direct, Economic & Statistical)
The platform strictly separates three distinct forms of portfolio exposure rather than conflating them:
 1. *Direct & Look-Through Ownership Exposure:* Aggregates actual underlying economic ownership across direct stock positions and multiple ETFs/funds (accounting for dual share classes, parent/subsidiary structures, and disclosing exact constituent coverage percentages and effective dates):
   $$\text{Weight}_{\text{look-through}}(i) = w_{\text{direct}}(i) + \sum_{f \in \text{Funds}} \left( w_{\text{fund}}(f) \times w_{\text{constituent}}(f, i) \right)$$
   Canonical Example: $10\%$ direct Nvidia $+ 20\%$ in an ETF holding $8\%$ Nvidia $= 10\% + (20\% \times 8\%) = 11.6\%$ total Nvidia ownership.
 2. *Economic & Thematic Exposure:* Identifies cross-industry sensitivity to shared macroeconomic or value-chain drivers (e.g., AI Infrastructure Capex spanning chip designers, foundries, cloud providers, data-centre REITs, power utilities, and networking equipment; Interest-Rate Sensitivity; Energy; Inflation; USD Strength; Chinese Growth). Themes are non-exclusive (one company can belong to multiple themes, so percentages need not sum to 100%) and must be supported by filings, revenue segments, or supply-chain evidence.
 3. *Statistical Exposure:* Empirically measured return behaviour from market data (pairwise/rolling correlations, covariance, factor betas, principal components, and stress-period co-movement). Surfaces both unrelated industries that move together statistically and businesses in the same economic theme whose share prices do not correlate strongly.

---
---

# PAGE 2 OF 4: DETERMINISTIC QUANTITATIVE RISK ENGINE & MATHEMATICAL SPECIFICATION
Synthesising Part VI & Part XXI (Sections 16–30, 87–95): Concentration, Volatility, Tail Risk, Regressions & Multi-Asset Models

	*Core Engineering Principle:* The LLM must never improvise financial arithmetic. Conventional, unit-tested financial code computes all metrics; the AI agent selects investigations, validates data completeness, and translates verified numbers into plain English.

---

### 06. Portfolio Composition, Concentration & Effective Diversification
•  *Multi-Dimensional Allocation Breakdowns:* Computes position values/weights, cash allocation, asset class, sector, industry, country/region, listing currency vs. underlying economic currency exposure, market capitalisation, growth/value style, and gross/net/leveraged exposure.
•  *Weight & Issuer Concentration Metrics:*
  * *Herfindahl-Hirschman Index (HHI):*
    $$\text{HHI} = \sum_{i=1}^N w_i^2$$
    (computed on both direct holdings and look-through issuer weights).
  * *Effective Number of Holdings:*
    $$N_{\text{eff}} = \frac{1}{\sum_{i=1}^N w_i^2}$$
    contrasted against Top-3, Top-5, and Top-10 concentration ratios.
  * *Correlation-Adjusted Independent Bets:* Explains why holding 10 overlapping ETFs or 20 highly correlated tech stocks yields a high nominal ticker count but a low effective number of independent risk sources.

---

### 07. Returns, Volatility, Covariance Shrinkage & Risk Attribution
•  *Return Conventions & Reconstruction Integrity:* Supports simple and logarithmic returns across daily, weekly, monthly, cumulative, and annualised geometric (CAGR) horizons using corporate-action and total-return adjusted price series. Strictly separates retrospective reconstruction of current weights from realised historical account performance to eliminate look-ahead and survivorship bias.
•  *Portfolio Volatility & Covariance Estimation:*
  $$\sigma_p = \sqrt{\mathbf{w}^T \boldsymbol{\Sigma} \mathbf{w}}$$
  Supports Historical Sample Covariance, Exponentially Weighted Moving Average (EWMA) Covariance, and *Ledoit-Wolf Shrinkage Covariance* to stabilise estimates when the number of holdings is large relative to the lookback window. Respects asset-specific trading-day conventions (e.g., 252 trading days for equities vs. 365 continuous days for crypto).
•  *Marginal & Component Risk Attribution (Capital Weight vs. Risk Weight):*
  $$\text{MCR}_i = \frac{(\boldsymbol{\Sigma} \mathbf{w})_i}{\sigma_p}, \qquad \text{CCR}_i = w_i \cdot \text{MCR}_i, \qquad \sum_{i=1}^N \text{CCR}_i = \sigma_p$$
  Reveals when a seemingly modest capital allocation (e.g., 10% weight) accounts for a disproportionately large share (e.g., 30%+) of total portfolio volatility.

---

### 08. Tail-Risk Analytics: Value at Risk (VaR), Expected Shortfall & Drawdowns
•  *Value at Risk (VaR):* Supports Historical Simulation VaR, Parametric Variance-Covariance VaR, Monte Carlo VaR, and Stress VaR at *95% and 99% confidence levels* over *1-day, 1-week, and 1-month horizons*. Always communicates VaR as a model-based loss threshold—not a maximum possible loss.
•  *Expected Shortfall (Conditional VaR / ES):*
  $$\text{ES}_{\alpha} = \mathbb{E}\left[ L \;\middle|\; L \ge \text{VaR}_{\alpha} \right]$$
  Estimates the average portfolio loss during the worst $5\%$ or $1\%$ of outcomes, providing far clearer tail-risk education for retail users than VaR alone, while flagging sample-size uncertainty in small windows.
•  *Drawdown Analysis:* Computes Historical Maximum Drawdown ($\text{MDD} = \max_{t} \frac{\text{Peak}_t - \text{Value}_t}{\text{Peak}_t}$), current drawdown from prior peak, underwater duration, and recovery time across crisis periods.
•  *Descriptive Risk-Adjusted Ratios:* Computes Sharpe Ratio ($\frac{R_p - R_f}{\sigma_p}$), Sortino Ratio (downside deviation), Calmar Ratio (annualised return / MDD), Information Ratio, and Tracking Error as secondary diagnostics for advanced users.

---

### 09. Factor Regressions, Correlation Dynamics, Stress Scenarios & Multi-Asset Models
•  *Correlation & Regime Analytics:* Generates pairwise correlation matrices, rolling correlations (3M, 1Y, 3Y), hierarchical correlation clustering, and conditional stress-period correlations (demonstrating how correlations spike during market sell-offs or inflation shocks).
•  *Single- & Multi-Factor Regressions + PCA:*
  $$R_{i,t} - R_{f,t} = \alpha_i + \beta_{i,\text{MKT}}\text{MKT}_t + \beta_{i,\text{SECTOR}}\text{SECTOR}_t + \beta_{i,\text{RATE}}\Delta\text{Yield}_t + \beta_{i,\text{FX}}\Delta\text{FX}_t + \varepsilon_{i,t}$$
  Estimates rolling betas, $R^2$, residual volatility, collinearity diagnostics, and out-of-sample stability against broad equity markets, tech/sector indices, value/growth, size, momentum, interest rates, inflation surprises, credit spreads, oil, and FX. Principal Component Analysis (PCA) identifies latent statistical drivers without inventing unsupported narratives.
•  *Historical Crisis Replays & Hypothetical Shock Propagation:*
  * Historical Replays: Reconstructs portfolio performance across the 2008 GFC, March 2020 COVID-19 crash, and 2022 inflation/rate-hike shock—explicitly flagging missing instrument histories and clearly labelled index proxies.
  * Hypothetical Shocks: Propagates custom shocks (e.g., Tech equities $-20\%$, US 10Y yields $+100\text{ bps}$, GBP/USD $+10\%$, Oil $-25\%$, single holding $-30\%$) via direct weights and factor sensitivities.
•  *Asset-Class-Specific Risk Models:*
  * Bonds: Modified/effective duration, convexity, YTM, DV01, credit-spread sensitivity, maturity profile.
  * Options & Derivatives: Delta ($\Delta$), Gamma ($\Gamma$), Vega ($\mathcal{V}$), Theta ($\Theta$), Rho ($\rho$), implied volatility, contract multipliers, delta-equivalent notional exposure, and nonlinear payoff stress tests.
  * Leveraged & Inverse ETFs: Accounts for daily reset mechanics and volatility decay/path dependence over multi-day horizons.

---
---

# PAGE 3 OF 4: AUTONOMOUS AGENT ARCHITECTURE, USER EXPERIENCE & SAFETY GUARDRAILS
Synthesising Parts VII–XIII (Sections 31–63): 10-Stage Investigation Loop, What-If Chat, Event Intelligence & Hard Risk Controls

	*Why Genuine Agency Matters:* A static dashboard with an LLM summary fails the hackathon's technical ambition criterion. Portfolio X-Ray's agent independently formulates risk hypotheses, selects verification tools, cross-examines evidence, and prioritises material vulnerabilities without waiting for the user to ask.

---

### 10. The 10-Stage Autonomous Investigation Workflow & Logical Architecture
When a user uploads or updates a portfolio (e.g., Nvidia, Microsoft, S&P 500 ETF, Nasdaq-100 ETF, Semiconductor ETF), the agent autonomously executes:
 1. *Ingest:* Extracts structured holdings from screenshots or manual entries.
 2. *Resolve:* Disambiguates tickers, exchanges, share classes, and currencies with human confirmation.
 3. *Retrieve:* Pulls historical adjusted prices, corporate actions, and published ETF constituent weights.
 4. *Analyse:* Runs the deterministic quantitative engine for baseline weights, look-through overlap, and volatility.
 5. *Identify Anomalies:* Detects unusual single-company overlap, disproportionate risk contributions, or unexpected correlation clusters.
 6. *Investigate:* Autonomously launches deeper tool calls (e.g., checking whether other holdings share revenue dependency on AI infrastructure capex or sensitivity to discount-rate hikes).
 7. *Verify:* Cross-checks whether proposed relationships are backed by fund disclosures, corporate filings, and statistical return evidence.
 8. *Contextualise:* Evaluates verified risks against the user’s stated investment horizon, objective, and loss capacity.
 9. *Explain:* Synthesises plain-English conclusions linked to an inspectable evidence trail.
10. *Monitor:* Saves the timestamped assessment snapshot to track longitudinal drift and re-evaluate when markets or holdings shift.

*Logical System Components:* Portfolio Ingestion Service, Instrument Resolver, Market-Data & ETF Constituent Service, Deterministic Quantitative Risk Engine, Exposure Investigation Agent, Event Intelligence Component, Personalisation Engine, Conversational Orchestrator, Evidence & Validation Layer, and Persistent Monitoring Store.

---

### 11. User-Facing Dashboard, Visualisations & Plain-English Explainability
•  *Seven Signature Visualisations:*
  1. Multi-Dimension Allocation Charts: Asset class, sector, geography, listing vs. economic currency, and underlying issuer.
  2. Look-Through Exposure Flow Diagram: Visualises how direct positions and multiple ETFs converge onto the same underlying companies.
  3. Correlation Network Graph: Displays holdings as nodes connected by empirical return correlations, visually distinguishing statistical links from shared economic themes.
  4. Correlation Heatmap: Full matrix view with selectable estimation windows (3M, 1Y, 3Y, Stress) for advanced users.
  5. Capital Weight vs. Volatility Contribution Chart: Side-by-side comparison showing which holdings drive total portfolio risk ($\text{CCR}_i / \sigma_p$).
  6. Non-Additive Theme Concentration Diagram: Shows exposure to overlapping themes (AI capex, interest rates, consumer spending) without double-counting portfolio value.
  7. Scenario & Crisis Contribution Waterfall: Breaks down estimated gains/losses by holding and factor during historical or hypothetical shocks.
•  *Plain-English Translation & Expandable Provenance:* Every technical metric is translated into everyday language (e.g., explaining what $\beta = 1.38$ or $97.5\%\text{ 1-Day VaR} = \text{£1,850}$ means, why tail losses can exceed VaR, and what the model does not know), with 1-click expansion to inspect raw formulas, timestamps, and data coverage. Distinguishes trading near a 52-week price high from fundamental overvaluation.

---

### 12. Conversational AI Risk Assistant, What-If Simulator & Headline Sanity Check
•  *Impulsive Decision Sounding Board ("50% into Nvidia" Workflow):*
  * Clarifies whether the user means 50% of total post-trade portfolio value or 50% of available cash.
  * Computes existing direct + indirect ETF Nvidia exposure and simulates the proposed allocation.
  * Displays before-and-after single-stock concentration, HHI, portfolio volatility, VaR, Expected Shortfall, and historical crisis drawdowns.
  * Compares the simulated downside against the user's stated loss tolerance and horizon—without predicting Nvidia's future share price.
•  *Mathematical Precision Across Trade Types:* Distinguishes adding new cash, reallocating existing capital, partial position sales, and buying funds with overlapping constituents.
•  *User-Directed Risk-Reduction Scenarios:* When a user has 75% exposure to one theme despite low risk tolerance, the agent explains the risk and lets the user test reducing exposure to 60%, 50%, or 40%—never prescribing specific shares to buy or sell.
•  *Headline & News Deconstruction:* Users paste breaking headlines (e.g., "Bank of England unexpectedly raises interest rates" or "Nvidia announces major AI partnership"). The agent verifies the event timestamp, maps affected holdings and themes, inspects historical factor sensitivity, and explains whether the news actually changes portfolio risk—curbing impulsive FOMO or panic selling.

---

### 13. Event-Driven Risk Calendar, Preference Memory & Hard Safety Controls
•  *Portfolio-Aware Event Calendar:* Filters central bank rate decisions, inflation/employment prints, and company earnings specifically to the user's material exposures. Incorporates observable market pricing (rate futures/OIS, options-implied moves, Polymarket probabilities) and historical event studies (highlighting that market reactions depend on surprises relative to expectations) without forecasting outcomes.
•  *Smart Preference Memory:* Remembers when a user understands and acknowledges a risk (e.g., accepting 30% tech exposure) to avoid repetitive alerts, while automatically re-evaluating if exposure drifts to 60%, volatility jumps, or their horizon changes.
•  *Enforced Code-Level Guardrails & Security:*
  * Zero live trade execution, zero brokerage credential access, and zero movement of real money.
  * Mandatory human confirmation for low-confidence OCR or ambiguous tickers.
  * *Prompt-Injection Isolation:* External news articles, web pages, and corporate filings are treated strictly as untrusted data unable to override system instructions, alter tool permissions, or fabricate numbers.
  * Privacy by design: redacts unnecessary account numbers and uses synthetic portfolios for public demonstrations.

---
---

# PAGE 4 OF 4: UNSEEN DATA EVALUATION, JUDGING ALIGNMENT, MVP SCOPE & PITCH STRATEGY
Synthesising Parts XIV–XXV (Sections 64–107): Benchmark Suite, Baseline Comparisons, Acceptance Tests & Live Demo Script

	*Winning the Judging Rubric:* With 50% of the score tied to Traceable Reasoning (25%) and Testing on Unseen Data (25%), the evaluation harness is built as a core deliverable alongside the application—proving measurable agentic value over static dashboards and generic LLMs.

---

### 14. Comprehensive Evaluation Framework & Unseen Data Benchmarks
•  *Held-Out Benchmark Portfolio Dataset:* Combines real historical market data and clearly labelled, documented synthetic portfolios held out from prompt/system tuning—spanning diversified global equities, single-stock concentrations, multi-ETF hidden overlap, cross-industry AI infrastructure chains, hidden duration/interest-rate sensitivity, high statistical correlation without obvious thematic ties, shared themes with low empirical correlation, multi-currency holdings, leveraged ETFs, and cropped/ambiguous screenshots.
•  *Four-Pillar Verification Suite:*
  1. Portfolio Extraction Testing: Ground-truth evaluation of ticker, quantity, value, weight, and currency extraction accuracy, plus ambiguity detection rate and total reconciliation on difficult screenshots.
  2. Quantitative Engine Verification: Automated unit tests (node:test / pytest) comparing look-through weights, volatility, Ledoit-Wolf covariance, betas, VaR, Expected Shortfall, and drawdowns against trusted reference implementations within strict numerical tolerances.
  3. Hidden Exposure Precision & Recall: Measures precision, recall, and false-positive rates in detecting hidden concentrations, explicitly testing whether the system distinguishes direct ownership overlap, economic themes, and statistical correlations.
  4. Unseen Historical & Walk-Forward Testing: Point-in-time out-of-sample evaluation (walk-forward windows and held-out market regimes with zero look-ahead leakage), testing VaR exceedance/coverage rates, ES calibration, and adversarial hallucination resistance on misleading headlines.

| System Evaluated | Look-Through & Risk Math | Adaptive Follow-Up Investigation | Numerical Hallucination Rate | Hidden Cross-Theme Recall |
| :--- | :--- | :--- | :--- | :--- |
| *Baseline A: Traditional Dashboard* | Static ticker/sector weights only | None (User must inspect manually) | 0% (Deterministic) | Low — misses ETF & macro overlap |
| *Baseline B: Generic LLM* | Improvised LLM arithmetic (Unreliable) | Unstructured text speculation | High — fabricates weights/betas | Medium — conflates theme & correlation |
| *Baseline C: Fixed Pipeline* | Deterministic predefined sequence | None (Rigid rule script) | 0% (Deterministic) | Moderate — misses novel anomalies |
| *Portfolio X-Ray (Proposed)* | *Deterministic Quant Engine + Look-Through* | *10-Stage Autonomous Tool Orchestration* | *0% (Strict Code Delegation)* | *High — verified multi-modal discovery* |

---

### 15. Direct Mapping to Agentverse 2026 Judging Criteria (100% Weight Breakdown)

| Judging Criterion | Weight | Official Hackathon Requirement | Portfolio X-Ray Concrete Deliverable & Proof |
| :--- | :--- | :--- | :--- |
| *Traceable Reasoning* | *25%* | Every decision traced from raw data to output; math separated from AI reasoning. | Interactive Evidence Trail Drawer linking every claim to raw OCR, security resolution, ETF holdings dates, formulas, and deterministic code logs. |
| *Testing on Unseen Data* | *25%* | Meaningful evaluation on unseen data, realistic costs, benchmarks, and stated uncertainty. | Held-out portfolio suite across walk-forward market windows; full ablation metrics (precision, recall, latency, tool cost, confidence intervals) vs. Baselines A, B & C. |
| *Relevance & Impact* | *20%* | Clearly defined user and a specific costly/harmful problem addressed in the working product. | Protects everyday retail investors from accidental ETF/thematic concentration and impulsive headline-driven trading; validated via before/after user comprehension tests. |
| *Risk Controls & Oversight* | *15%* | Limits enforced in code, actions logged, and a human control mechanism. | Non-executing architecture, mandatory human confirmation on OCR/ambiguous tickers, prompt-injection sandboxing, audit logs, and refusal to fabricate missing data. |
| *Ambition & Technical Quality* | *15%* | Technically substantial, well-engineered solution beyond conventional software. | Autonomous 10-stage investigative agent orchestrating ETF look-through decomposition, factor regressions, crisis stress replays, and conversational what-if simulations. |

---

### 16. Prioritised Hackathon MVP Roadmap & 10 End-to-End Acceptance Tests
•  *Priority 0 (Core Hackathon Must-Haves):* Functional web UI, screenshot OCR + manual portfolio entry/editing with human confirmation, persistent snapshot storage, user risk-profile onboarding, equity & ETF price/constituent retrieval, ETF look-through calculator, direct/economic/statistical exposure analysis, HHI/volatility/drawdown/VaR/ES engine, autonomous risk investigation, expandable evidence trail, conversational what-if simulator (before/after risk delta), and reproducible benchmark harness.
•  *Priority 1 (Strong Differentiators):* Rolling correlations, multi-factor regressions, marginal/component risk contributions (MCR/CCR), historical crisis stress testing, headline sanity-check analyser, preference memory, and portfolio-relevant event calendar.
•  *Priority 2 & Commercial Horizon:* Comprehensive bond duration/DV01 and option Greeks models, continuous background alerting, and direct brokerage integrations (Open Banking / Plaid / SnapTrade) for B2C subscription or B2B wealth-platform licensing.
•  *10 Mandatory MVP Acceptance Tests (Tests A–J):*
  * *Test A:* Upload & reconcile screenshot weights.
  * *Test B:* Compute exact direct + ETF look-through overlap.
  * *Test C:* Produce verified volatility, correlation, drawdown & tail-risk math.
  * *Test D:* Autonomously discover unprompted hidden exposure.
  * *Test E:* Render inspectable evidence trail per finding.
  * *Test F:* Personalise risk interpretation by profile without altering math.
  * *Test G:* Simulate before/after hypothetical investment.
  * *Test H:* Save & compare historical snapshots.
  * *Test I:* Flag ambiguous/missing data safely without fabrication.
  * *Test J:* Run reproducible unseen-data benchmark vs. baselines.
