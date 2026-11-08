# Portfolio X-Ray

A consumer fintech application backed by an institutional quantitative risk engine and a rigorous safety architecture. Portfolio X-Ray brings Jane Street-level mathematics and deterministic evidence-backed risk analysis to retail investors.

## Unified Architecture

This project is a unified, production-grade integration of two components:
1.  **Consumer Frontend (`v-ui/`)**: A clean, accessible web dashboard featuring intuitive visualisations (ETF-to-Company overlap diagrams, dynamic stress scenarios) and an AI conversational analyst.
2.  **Institutional Quantitative Engine & Safety Framework (`xray/`)**: 
    -   **Mathematical Invariants**: Every quantitative risk metric (Volatility, VaR, CVaR, Drawdown) is derived deterministically from 15-year real market prices (`data/`), overriding any LLM hallucinations.
    -   **Look-Through Decomposition**: Rigorous institutional look-through algorithm resolving ETF exposures strictly down to individual holdings.
    -   **Circuit Breakers**: Strict, programmable boundaries (`xray/safety/limits.py`) for single-stock concentration and volatility.
    -   **Safety Filter & Audit**: Code-level OutputSafetyFilter enforcing anti-advisory policies (zero buy/sell recommendations, zero predictive price targets). Every inference event and user sign-off is logged securely to an append-only JSONL registry.

## Evaluation & Benchmarking

-   **Adversarial Prompt-Injection**: 100% successful block/refusal rate against adversarial queries (e.g., prompt injection, stock price predictions, trading execution requests).
-   **AgentVerse Evaluation Harness**: 10/10 Passed (1.0 F1 Score, 100% Risk Coverage, 0% Hallucination Rate).
-   **End-to-End Safety Testing**: 100% Pytest suite passing across all modules.

## Getting Started

Start the integrated web application (listening on port 8080 by default):

```bash
python3 server.py 8080
```
Then navigate to `http://localhost:8080/` in your browser.