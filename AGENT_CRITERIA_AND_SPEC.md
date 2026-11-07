# Portfolio X-Ray: Agent Acceptance Criteria & System Constraints

This document defines the **functional requirements, evaluation standards, and operational constraints** for the Portfolio X-Ray agent. Use this as the definition of done (acceptance criteria) for the agent.

---

## 1. System Scope & Boundary Criteria

### What the Agent MUST Do
- [ ] **Acceptance Criterion 1.1 (Multi-Asset Ingestion):** Ingest and parse arbitrary user holdings across equities, ETFs, mutual funds, and cash (from screenshots or manual entries) without failing on missing metadata.
- [ ] **Acceptance Criterion 1.2 (Underlying Look-Through):** Resolve indirect exposures by decomposing ETF/fund holdings into their constituent companies so the user sees true aggregate exposure rather than superficial ticker counts.
- [ ] **Acceptance Criterion 1.3 (Thematic Sensitivity Mapping):** Identify shared economic vulnerabilities across disparate assets (e.g., semiconductor manufacturers, power utilities, and data centre operators all tied to AI infrastructure capex).
- [ ] **Acceptance Criterion 1.4 (Scenario Stress-Testing):** Evaluate user-proposed allocation changes or macroeconomic events against historical volatility and factor drawdowns.

### What the Agent MUST NOT Do (Strict Prohibitions)
- [ ] **Constraint 1.1 (No Price Speculation):** The agent must not predict future stock prices, set price targets, or generate buy/sell trading signals.
- [ ] **Constraint 1.2 (No Autonomous Execution):** The agent must never place orders, move funds, or execute reallocations without explicit human confirmation.
- [ ] **Constraint 1.3 (No Black-Box Advice):** The agent must never give a blanket "good" or "bad" portfolio verdict without showing the underlying mathematical exposure breakdown and data sources.

---

## 2. Quantitative Accuracy & Reasoning Criteria

### Separation of Math & Language
- [ ] **Acceptance Criterion 2.1 (Deterministic Calculations):** All quantitative values (weights, portfolio concentration, Value at Risk, drawdown percentages, and tracking error) must be computed by deterministic mathematical routines, never generated as probabilistic text by a language model.
- [ ] **Acceptance Criterion 2.2 (Traceable Evidence Lineage):** Every reported risk exposure must be accompanied by an auditable breakdown showing each contributing fund, its holding weight, and its constituent percentage.
- [ ] **Acceptance Criterion 2.3 (Conservation of Exposure):** Total calculated direct and indirect portfolio weights must strictly conserve total portfolio value without phantom allocations or floating-point truncation drift.

---

## 3. Evaluation & Unseen Data Testing Criteria

### Out-of-Sample Validation
- [ ] **Acceptance Criterion 3.1 (Holdout Testing):** The agent's risk assessment and stress-testing logic must be evaluated on holdout historical market shocks that were not part of initial prompt testing (e.g., the August 2024 tech volatility spike or the 2022 rapid rate hike cycle).
- [ ] **Acceptance Criterion 3.2 (Friction Modeling):** All hypothetical reallocations and what-if scenarios must account for real-world transaction friction:
  - Broker bid-ask spread
  - Trading commission and transaction fees
  - ETF management expense ratio drag
  - Price slippage on larger reallocations
- [ ] **Acceptance Criterion 3.3 (Benchmark Comparison):** The agent's concentration findings must be measured against a standard passive baseline (such as an unmanaged 60/40 benchmark or MSCI World index).
- [ ] **Acceptance Criterion 3.4 (Stated Uncertainty):** Statistical risk metrics (e.g., Value at Risk) must state their confidence levels (e.g., 95% 1-day horizon) and error bounds rather than presenting single, absolute figures.

---

## 4. Risk Controls & Safety Invariants

### Code-Enforced Guardrails
- [ ] **Acceptance Criterion 4.1 (Hard Concentration Invariants):** The system must enforce explicit, programmatically verifiable limits (e.g., maximum single-asset look-through exposure and maximum sector concentration). These must be evaluated by code logic, not prompt suggestions.
- [ ] **Acceptance Criterion 4.2 (Live Circuit Breaker):** When a user's proposed reallocation or an extreme market shock breaches a safety threshold:
  - The system must trigger an immediate operational halt.
  - An immutable audit record of the breach must be logged.
  - The agent must visibly explain the specific mathematical threshold that was violated.
- [ ] **Acceptance Criterion 4.3 (Human Approval Gate):** High-stakes decisions must stage a pre-flight summary and require explicit user sign-off before proceeding.

---

## 5. Target User Context & Scenario Criteria

### Persona & Acute Failure Handling
- [ ] **Acceptance Criterion 5.1 (Personalized Suitability):** Risk evaluations must be interpreted in the context of a specific user profile (investment horizon, liquidity needs, and loss tolerance) rather than an arbitrary universal threshold.
- [ ] **Acceptance Criterion 5.2 (The Critical Failure Test):** The agent must successfully detect and prevent the classic retail diversification trap:
  - *The Scenario:* A user holding multiple funds proposes adding a large position in a popular stock, believing their existing portfolio is safe.
  - *The Requirement:* The agent must detect the existing hidden overlap across the funds, show that the true combined exposure exceeds safe risk limits, and demonstrate how an adverse economic shock would jeopardize the user's specific financial milestone.

---

## 6. Pre-Submission Quality Checklist

Before submitting the agent, ensure it passes all of the following:
- [x] All mathematical metrics trace to verifiable constituent weights and formulas.
- [x] The agent behaves sensibly and states its limitations when presented with incomplete or missing asset data.
- [x] The circuit breaker triggers deterministically when exposure limits are exceeded.
- [x] Friction costs (spreads and fees) are visibly deducted from simulated trades.
- [x] All credentials and API keys are loaded via environment variables with zero secrets committed to the repository.
