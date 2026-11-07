# Portfolio X-Ray: Agent Criteria & Technical Invariants
*(The "Jane Street Standard" Quality & Risk Control Specification)*

---

## 1. System Scope & Boundaries (What the Agent Can & Cannot Do)

### Mandatory Capabilities:
* **Multi-Asset Ingestion:** Must ingest and parse arbitrary portfolio inputs (from screenshots or manual entries) containing equities, ETFs, mutual funds, and cash without failing on missing metadata.
* **Underlying Look-Through Decomposition:** Break down ETFs and mutual funds into constituent stocks so the user sees true economic exposure rather than just surface-level ticker names.
* **Thematic & Economic Linkage:** Detect shared dependencies across seemingly unrelated companies (e.g. chip manufacturers, utilities, and data centre operators all exposed to AI infrastructure capex).
* **What-If Scenario Stress-Testing:** Evaluate hypothetical allocation adjustments or macroeconomic events against historical volatility and factor drawdowns.

### Strict Prohibitions:
* ❌ **No Stock Price Predictions:** The agent must never predict price targets or forecast market direction.
* ❌ **No Buy/Sell Trading Signals:** The agent must never recommend specific stock purchases or issue trading calls.
* ❌ **No Autonomous Execution:** The agent must never execute transactions, place orders, or move funds automatically.
* ❌ **No Black-Box Advice:** The agent must never give an opinion on a portfolio without presenting the exact underlying constituent breakdown and calculation proof.

---

## 2. Quantitative Accuracy & Math Separation (Jane Street Standard)

### Zero Mental Math in the Language Model:
* All mathematical arithmetic—including portfolio percentage weights, aggregate concentration, Value at Risk (VaR), Expected Shortfall, and beta—**must be calculated by deterministic mathematical functions/code**, never generated as probabilistic text by an LLM prompt.
* The language model's role is strictly limited to extracting unstructured data and providing qualitative explanations of the economic linkages.

### Traceable Lineage & Evidence Trail:
* Every exposure warning must trace back to verifiable data: showing direct holdings, each fund holding that asset, the fund's portfolio weight, and the asset's percentage inside the fund.

### Conservation of Exposure:
* The mathematical engine must guarantee total weight conservation ($\sum w_i = 1.0$) with zero phantom allocations or truncation drift.

---

## 3. Testing on Unseen Data & Real-World Friction

### Holdout Validation:
* The risk model must be evaluated on unseen historical shocks not used during prompt development (e.g., the August 2024 tech volatility spike, or the 2022 rapid rate hike cycle).

### Mandatory Friction Modeling:
When evaluating "what-if" reallocations, the system must deduct real-world transaction friction:
* Broker bid-ask spreads
* Trading commissions and transaction fees
* ETF management expense ratio drag
* Price slippage on larger reallocations

### Stated Uncertainty & Confidence Bounds:
* Statistical risk metrics must explicitly state confidence intervals and error bounds (e.g., 95% 1-day Historical VaR with stated sample size) rather than absolute, overconfident figures.
* The agent must explicitly flag when asset data is missing, lagged, or incomplete rather than guessing.

### Baseline Benchmark Comparison:
* The agent must compare the user's hidden portfolio concentration against a standard passive baseline (such as an unmanaged 60/40 benchmark or MSCI World index).

---

## 4. Safety Controls & Operational Invariants

### Hardcoded Limits in Code:
* Safety thresholds (e.g., maximum single-asset look-through limit, maximum sector exposure limit) must be enforced through programmatically verifiable assertions in application code, never delegated to soft prompt suggestions.

### Live Circuit Breaker:
* If a user-proposed reallocation or an incoming market shock breaches an exposure safety limit, the agent must trigger a deterministic halt, record an immutable audit log, and display the exact threshold that was violated.

### Mandatory Human Sign-Off:
* Any proposed portfolio adjustment must stage a pre-flight risk summary and require explicit human confirmation.

---

## 5. Core User Scenario & Failure Mode Handling

### Personalized Suitability:
* Risk must be interpreted through the user's personal investment horizon and loss tolerance (e.g. saving for a house deposit over 3 years vs. a 30-year retirement fund).

### The Retail Diversification Trap (Must Be Tested):
* The agent must detect the classic retail error: an investor holding multiple funds (e.g. S&P 500 ETF + Tech ETF + direct stock) believes they are diversified, but a proposed reallocation creates an extreme single-asset concentration cliff that puts their financial milestone at risk.
