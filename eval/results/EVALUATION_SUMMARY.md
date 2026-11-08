# Portfolio X-Ray — Quantitative Evaluation & Ablation Report

## 1. Holdout Ablation Study (`data/holdout/holdout_portfolios.json`, $N = 10$ Portfolios)

| System / Architecture | Hidden-Breach Precision | Hidden-Breach Recall | Hidden-Breach F1 | Multi-Dimensional Risk Coverage | Numerical Hallucination Rate |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Baseline A:** Static Dashboard (Direct Surface Weights Only) | `100.0%` | **`0.0%`** | **`0.00`** | `0.0%` | **`0.0%`** |
| **Baseline B:** Unverified Mental-Math Heuristic (No Engine Tools) | `100.0%` | **`0.0%`** | **`0.00`** | `0.0%` | **`70.0%`** |
| **Baseline C:** Fixed Pipeline v1 (No Autonomous Investigation) | `100.0%` | **`100.0%`** | **`1.00`** | `50.0%` | **`0.0%`** |
| **Portfolio X-Ray (Full Agent v2 + OutputSafetyFilter)** | `100.0%` | **`100.0%`** | **`1.00`** | `100.0%` | **`0.0%`** |

## 2. Out-of-Sample 1-Day 95% Historical VaR Backtest ($2,520$ Out-of-Sample Portfolio-Days)

- **Training Window:** Prior 504 trading days (2 years) per holdout portfolio
- **Out-of-Sample Test Window:** Subsequent 252 trading days (1 year) across 10 holdout portfolios ($2,520$ portfolio-days)
- **Theoretical Expected Exceedance Rate ($lpha = 0.05$):** `5.00%`
- **Observed Out-of-Sample Exceedance Rate:** **`5.81%`** (`144 / 2480` days, deviation `0.81%`)

## 3. Held-Out Crisis Regime Replays (`Aug 2024 Unwind` & `2022 Rate-Hike Shock`)

| Holdout Portfolio ID | Aug 2024 Return | Aug 2024 Max DD | 2022 Rate-Shock Return | 2022 Max DD | Data Coverage |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `HOLDOUT-01-MICROSOFT-STEALTH-BREACH` | `-5.31%` | `-10.50%` | `-26.85%` | `-30.20%` | `100%` |
| `HOLDOUT-02-TESLA-ARKK-QQQ-OVERLAP` | `-10.93%` | `-17.56%` | `-53.66%` | `-57.23%` | `100%` |
| `HOLDOUT-03-CLEAN-60-40-CORE` | `-1.22%` | `-4.69%` | `-16.36%` | `-20.00%` | `100%` |
| `HOLDOUT-04-BERKSHIRE-SHARE-CLASS-MERGE` | `-2.82%` | `-8.22%` | `-18.39%` | `-24.61%` | `89%` |
| `HOLDOUT-05-NVIDIA-SMH-STEALTH-BREACH` | `-6.94%` | `-13.68%` | `-31.05%` | `-36.83%` | `100%` |
| `HOLDOUT-06-DEFENSIVE-STAPLES-HEALTHCARE` | `+0.99%` | `-3.23%` | `+1.29%` | `-11.58%` | `100%` |
| `HOLDOUT-07-AMAZON-META-CLOUD-CLUSTER` | `-4.95%` | `-10.66%` | `-34.59%` | `-37.95%` | `100%` |
| `HOLDOUT-08-PASSIVE-TOTAL-MARKET-SAFE` | `-1.43%` | `-5.57%` | `-19.23%` | `-22.82%` | `100%` |
| `HOLDOUT-09-AMD-SEMI-AI-CLUSTER` | `-14.10%` | `-23.31%` | `-35.91%` | `-46.88%` | `85%` |
| `HOLDOUT-10-PAYMENTS-FINANCIALS-CORE` | `-2.40%` | `-8.16%` | `-17.82%` | `-27.87%` | `100%` |

## 4. AgentVerse `finance/bench.py` Adapter Verification

- **Adapter:** `eval/xray_bench_adapter.py` (invokes `LookThroughEngine`, `ConcentrationEngine`, `SafetyLimitsEngine`)
- **Dataset:** `eval/results/bench_dataset` ($10$ holdout cases)
- **Result:** **`10 / 10 PASSED`** (`0` failures, `0` execution errors)
