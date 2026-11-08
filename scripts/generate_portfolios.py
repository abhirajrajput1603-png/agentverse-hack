from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from xray.data_loader import DataLoader
from xray.engine.concentration import ConcentrationEngine
from xray.engine.lookthrough import LookThroughEngine
from xray.engine.risk import RiskEngine
from xray.engine.themes import ThemeEngine
from xray.safety.limits import SafetyLimitsEngine


RAW_PORTFOLIO_SPECS: List[Dict[str, Any]] = [
    # --- 5 DEV EXPANDED PORTFOLIOS (DEV-05 to DEV-09) ---
    {
        "split": "dev",
        "case_id": "DEV-05-ALPHABET-DUAL-CLASS-TRAP",
        "name": "The Alphabet Dual Share-Class & Index Overlap",
        "persona": "Maya, product manager splitting equity across GOOG, GOOGL, and QQQ thinking each is a separate 12% position.",
        "stated_goal": "Diversified large-cap growth with a 20% single-company cap",
        "max_risk_tolerance_single_stock": 0.20,
        "holdings": [
            {"ticker": "GOOG", "name": "Alphabet Class C", "allocation_gbp": 12000.0, "weight": 0.12},
            {"ticker": "GOOGL", "name": "Alphabet Class A", "allocation_gbp": 10000.0, "weight": 0.10},
            {"ticker": "QQQ", "name": "Invesco QQQ Trust", "allocation_gbp": 48000.0, "weight": 0.48},
            {"ticker": "SPY", "name": "SPDR S&P 500 ETF", "allocation_gbp": 30000.0, "weight": 0.30},
        ],
    },
    {
        "split": "dev",
        "case_id": "DEV-06-APPLE-TRIPLE-ETF-STACK",
        "name": "The Apple Triple-Wrapper Stack",
        "persona": "Oliver, retail saver holding 14% AAPL directly alongside QQQ, SPY, and VTI.",
        "stated_goal": "Balanced US index growth under an 18% single-stock limit",
        "max_risk_tolerance_single_stock": 0.18,
        "holdings": [
            {"ticker": "AAPL", "name": "Apple Inc", "allocation_gbp": 14000.0, "weight": 0.14},
            {"ticker": "QQQ", "name": "Invesco QQQ Trust", "allocation_gbp": 46000.0, "weight": 0.46},
            {"ticker": "SPY", "name": "SPDR S&P 500 ETF", "allocation_gbp": 25000.0, "weight": 0.25},
            {"ticker": "VTI", "name": "Vanguard Total Stock Market ETF", "allocation_gbp": 15000.0, "weight": 0.15},
        ],
    },
    {
        "split": "dev",
        "case_id": "DEV-07-CONSERVATIVE-BOND-CASH-BUFFER",
        "name": "The Defensive Income & Cash Baseline",
        "persona": "Eleanor, near-retiree prioritising capital stability with bonds, broad market ETFs, and cash.",
        "stated_goal": "Low-volatility capital preservation",
        "max_risk_tolerance_single_stock": 0.10,
        "holdings": [
            {"ticker": "BND", "name": "Vanguard Total Bond Market ETF", "allocation_gbp": 40000.0, "weight": 0.40},
            {"ticker": "SPY", "name": "SPDR S&P 500 ETF", "allocation_gbp": 35000.0, "weight": 0.35},
            {"ticker": "CASH_GBP", "name": "Sterling Cash Buffer", "allocation_gbp": 25000.0, "weight": 0.25},
        ],
    },
    {
        "split": "dev",
        "case_id": "DEV-08-SEMI-FOUNDRY-OVERLAP",
        "name": "The Taiwan Semi & Broadcom Hidden Concentration",
        "persona": "Arjun, hardware engineer investing in SMH, QQQ, and AVGO.",
        "stated_goal": "Technology growth with a 15% single-issuer limit",
        "max_risk_tolerance_single_stock": 0.15,
        "holdings": [
            {"ticker": "AVGO", "name": "Broadcom Inc", "allocation_gbp": 11000.0, "weight": 0.11},
            {"ticker": "SMH", "name": "VanEck Semiconductor ETF", "allocation_gbp": 50000.0, "weight": 0.50},
            {"ticker": "QQQ", "name": "Invesco QQQ Trust", "allocation_gbp": 39000.0, "weight": 0.39},
        ],
    },
    {
        "split": "dev",
        "case_id": "DEV-09-BLUECHIP-DIVIDEND-BALANCE",
        "name": "The Diversified Multi-Sector Core",
        "persona": "Hannah, long-horizon ISA investor combining healthcare, consumer staples, financials, and VTI.",
        "stated_goal": "Steady long-term compounding with low single-stock risk",
        "max_risk_tolerance_single_stock": 0.15,
        "holdings": [
            {"ticker": "VTI", "name": "Vanguard Total Stock Market ETF", "allocation_gbp": 50000.0, "weight": 0.50},
            {"ticker": "JPM", "name": "JPMorgan Chase & Co", "allocation_gbp": 10000.0, "weight": 0.10},
            {"ticker": "UNH", "name": "UnitedHealth Group", "allocation_gbp": 10000.0, "weight": 0.10},
            {"ticker": "PG", "name": "Procter & Gamble", "allocation_gbp": 10000.0, "weight": 0.10},
            {"ticker": "JNJ", "name": "Johnson & Johnson", "allocation_gbp": 10000.0, "weight": 0.10},
            {"ticker": "CASH_GBP", "name": "Sterling Cash", "allocation_gbp": 10000.0, "weight": 0.10},
        ],
    },
    # --- 10 HOLDOUT PORTFOLIOS (HOLDOUT-01 to HOLDOUT-10) ---
    {
        "split": "holdout",
        "case_id": "HOLDOUT-01-MICROSOFT-STEALTH-BREACH",
        "name": "Stealth Microsoft Breach Across Three Index Funds",
        "persona": "Chloe, holding 14% MSFT directly plus QQQ, SPY, and VTI under an 18% single-stock limit.",
        "stated_goal": "Moderate tech-tilted growth within an 18% single-stock cap",
        "max_risk_tolerance_single_stock": 0.18,
        "holdings": [
            {"ticker": "MSFT", "name": "Microsoft Corp", "allocation_gbp": 14000.0, "weight": 0.14},
            {"ticker": "QQQ", "name": "Invesco QQQ Trust", "allocation_gbp": 40000.0, "weight": 0.40},
            {"ticker": "SPY", "name": "SPDR S&P 500 ETF", "allocation_gbp": 30000.0, "weight": 0.30},
            {"ticker": "VTI", "name": "Vanguard Total Stock Market ETF", "allocation_gbp": 16000.0, "weight": 0.16},
        ],
    },
    {
        "split": "holdout",
        "case_id": "HOLDOUT-02-TESLA-ARKK-QQQ-OVERLAP",
        "name": "Hidden Tesla Concentration via ARKK and QQQ",
        "persona": "Lucas, holding 13% TSLA directly alongside 45% ARKK and 42% QQQ with an 18% single-stock limit.",
        "stated_goal": "High-growth innovation within an 18% single-issuer limit",
        "max_risk_tolerance_single_stock": 0.18,
        "holdings": [
            {"ticker": "TSLA", "name": "Tesla Inc", "allocation_gbp": 13000.0, "weight": 0.13},
            {"ticker": "ARKK", "name": "ARK Innovation ETF", "allocation_gbp": 45000.0, "weight": 0.45},
            {"ticker": "QQQ", "name": "Invesco QQQ Trust", "allocation_gbp": 42000.0, "weight": 0.42},
        ],
    },
    {
        "split": "holdout",
        "case_id": "HOLDOUT-03-CLEAN-60-40-CORE",
        "name": "Clean Institutional 60/40 SPY & BND",
        "persona": "Arthur, balanced retirement investor holding 60% SPY, 30% BND, and 10% Cash.",
        "stated_goal": "Balanced retirement income and growth",
        "max_risk_tolerance_single_stock": 0.10,
        "holdings": [
            {"ticker": "SPY", "name": "SPDR S&P 500 ETF", "allocation_gbp": 60000.0, "weight": 0.60},
            {"ticker": "BND", "name": "Vanguard Total Bond Market ETF", "allocation_gbp": 30000.0, "weight": 0.30},
            {"ticker": "CASH_GBP", "name": "Sterling Cash", "allocation_gbp": 10000.0, "weight": 0.10},
        ],
    },
    {
        "split": "holdout",
        "case_id": "HOLDOUT-04-BERKSHIRE-SHARE-CLASS-MERGE",
        "name": "Berkshire Dual Ticker Aggregation Trap",
        "persona": "grace, value investor holding 11% BRK.B and 8% BRK-B alongside 50% SPY and 31% VTI.",
        "stated_goal": "Value-oriented compounding with a 18% single-issuer cap",
        "max_risk_tolerance_single_stock": 0.18,
        "holdings": [
            {"ticker": "BRK.B", "name": "Berkshire Hathaway B (Dot Ticker)", "allocation_gbp": 11000.0, "weight": 0.11},
            {"ticker": "BRK-B", "name": "Berkshire Hathaway B (Hyphen Ticker)", "allocation_gbp": 8000.0, "weight": 0.08},
            {"ticker": "SPY", "name": "SPDR S&P 500 ETF", "allocation_gbp": 50000.0, "weight": 0.50},
            {"ticker": "VTI", "name": "Vanguard Total Stock Market ETF", "allocation_gbp": 31000.0, "weight": 0.31},
        ],
    },
    {
        "split": "holdout",
        "case_id": "HOLDOUT-05-NVIDIA-SMH-STEALTH-BREACH",
        "name": "Stealth Nvidia Breach Under 15% Policy Limit",
        "persona": "Noah, holding only 9% NVDA directly but 35% SMH and 30% QQQ plus 26% BND.",
        "stated_goal": "Growth and income with strict 15% single-stock ceiling",
        "max_risk_tolerance_single_stock": 0.15,
        "holdings": [
            {"ticker": "NVDA", "name": "Nvidia Corp", "allocation_gbp": 9000.0, "weight": 0.09},
            {"ticker": "SMH", "name": "VanEck Semiconductor ETF", "allocation_gbp": 35000.0, "weight": 0.35},
            {"ticker": "QQQ", "name": "Invesco QQQ Trust", "allocation_gbp": 30000.0, "weight": 0.30},
            {"ticker": "BND", "name": "Vanguard Total Bond Market ETF", "allocation_gbp": 26000.0, "weight": 0.26},
        ],
    },
    {
        "split": "holdout",
        "case_id": "HOLDOUT-06-DEFENSIVE-STAPLES-HEALTHCARE",
        "name": "Defensive Staples, Healthcare & Cash Portfolio",
        "persona": "Clara, conservative investor holding WMT, COST, LLY, JNJ, PG, and 30% Cash.",
        "stated_goal": "Low-beta defensive equity with 20% single-stock cap",
        "max_risk_tolerance_single_stock": 0.20,
        "holdings": [
            {"ticker": "WMT", "name": "Walmart Inc", "allocation_gbp": 14000.0, "weight": 0.14},
            {"ticker": "COST", "name": "Costco Wholesale", "allocation_gbp": 14000.0, "weight": 0.14},
            {"ticker": "LLY", "name": "Eli Lilly & Co", "allocation_gbp": 14000.0, "weight": 0.14},
            {"ticker": "JNJ", "name": "Johnson & Johnson", "allocation_gbp": 14000.0, "weight": 0.14},
            {"ticker": "PG", "name": "Procter & Gamble", "allocation_gbp": 14000.0, "weight": 0.14},
            {"ticker": "CASH_GBP", "name": "Sterling Cash", "allocation_gbp": 30000.0, "weight": 0.30},
        ],
    },
    {
        "split": "holdout",
        "case_id": "HOLDOUT-07-AMAZON-META-CLOUD-CLUSTER",
        "name": "Stealth Amazon Breach & Cloud Hyperscaler Cluster",
        "persona": "Ethan, cloud architect holding 13% AMZN, 10% META, 45% QQQ, and 32% SPY under a 16% single-stock cap.",
        "stated_goal": "Cloud growth within a 16% single-stock limit",
        "max_risk_tolerance_single_stock": 0.16,
        "holdings": [
            {"ticker": "AMZN", "name": "Amazon.com Inc", "allocation_gbp": 13000.0, "weight": 0.13},
            {"ticker": "META", "name": "Meta Platforms Inc", "allocation_gbp": 10000.0, "weight": 0.10},
            {"ticker": "QQQ", "name": "Invesco QQQ Trust", "allocation_gbp": 45000.0, "weight": 0.45},
            {"ticker": "SPY", "name": "SPDR S&P 500 ETF", "allocation_gbp": 32000.0, "weight": 0.32},
        ],
    },
    {
        "split": "holdout",
        "case_id": "HOLDOUT-08-PASSIVE-TOTAL-MARKET-SAFE",
        "name": "Pure Total Market & Treasury Barbell Within Limits",
        "persona": "Sophie, index investor holding 45% VTI, 25% SPY, 15% TLT, and 15% Cash.",
        "stated_goal": "Long-term index growth with 10% single-company cap",
        "max_risk_tolerance_single_stock": 0.10,
        "holdings": [
            {"ticker": "VTI", "name": "Vanguard Total Stock Market ETF", "allocation_gbp": 45000.0, "weight": 0.45},
            {"ticker": "SPY", "name": "SPDR S&P 500 ETF", "allocation_gbp": 25000.0, "weight": 0.25},
            {"ticker": "TLT", "name": "iShares 20+ Year Treasury Bond ETF", "allocation_gbp": 15000.0, "weight": 0.15},
            {"ticker": "CASH_GBP", "name": "Sterling Cash", "allocation_gbp": 15000.0, "weight": 0.15},
        ],
    },
    {
        "split": "holdout",
        "case_id": "HOLDOUT-09-AMD-SEMI-AI-CLUSTER",
        "name": "AI Semiconductor & Power Cluster with Unpriced VST",
        "persona": "Daniel, investing across AMD, AVGO, VST, and SMH with a 20% single-stock cap.",
        "stated_goal": "Next-gen AI hardware and power infrastructure",
        "max_risk_tolerance_single_stock": 0.20,
        "holdings": [
            {"ticker": "AMD", "name": "Advanced Micro Devices", "allocation_gbp": 15000.0, "weight": 0.15},
            {"ticker": "AVGO", "name": "Broadcom Inc", "allocation_gbp": 15000.0, "weight": 0.15},
            {"ticker": "VST", "name": "Vistra Corp", "allocation_gbp": 15000.0, "weight": 0.15},
            {"ticker": "SMH", "name": "VanEck Semiconductor ETF", "allocation_gbp": 55000.0, "weight": 0.55},
        ],
    },
    {
        "split": "holdout",
        "case_id": "HOLDOUT-10-PAYMENTS-FINANCIALS-CORE",
        "name": "Global Payments, Financials & Index Core",
        "persona": "Zoe, fintech analyst holding V, MA, JPM, BAC, and 50% VTI under a 15% single-stock limit.",
        "stated_goal": "Financial sector tilt without breaching 15% single-stock cap",
        "max_risk_tolerance_single_stock": 0.15,
        "holdings": [
            {"ticker": "V", "name": "Visa Inc", "allocation_gbp": 13000.0, "weight": 0.13},
            {"ticker": "MA", "name": "Mastercard Inc", "allocation_gbp": 12000.0, "weight": 0.12},
            {"ticker": "JPM", "name": "JPMorgan Chase & Co", "allocation_gbp": 13000.0, "weight": 0.13},
            {"ticker": "BAC", "name": "Bank of America Corp", "allocation_gbp": 12000.0, "weight": 0.12},
            {"ticker": "VTI", "name": "Vanguard Total Stock Market ETF", "allocation_gbp": 50000.0, "weight": 0.50},
        ],
    },
]


def build_labelled_portfolios() -> None:
    loader = DataLoader()
    lt_eng = LookThroughEngine(loader)
    conc_eng = ConcentrationEngine()
    risk_eng = RiskEngine(loader)
    th_eng = ThemeEngine(loader)
    safety_eng = SafetyLimitsEngine()

    dev_cases: List[Dict[str, Any]] = []
    holdout_cases: List[Dict[str, Any]] = []

    for spec in RAW_PORTFOLIO_SPECS:
        holdings = spec["holdings"]
        limit = float(spec["max_risk_tolerance_single_stock"])

        lt_res = lt_eng.analyze(holdings)
        conc_res = conc_eng.analyze(lt_res)
        risk_res = risk_eng.analyze(holdings)
        th_res = th_eng.analyze(lt_res)
        safety_res = safety_eng.evaluate_portfolio(
            lt_result=lt_res,
            concentration=conc_res,
            user_max_single_stock_limit=limit,
            persona_name=spec["persona"],
        )

        top_named = lt_res.sorted_exposures(exclude_residual_and_cash=True)
        top_issuer = top_named[0].issuer if top_named else "NONE"
        top_weight = round(top_named[0].total_weight, 6) if top_named else 0.0
        other_w = round(lt_res.get_issuer_weight("OTHER_CONSTITUENTS"), 6)

        # Surface (direct ticker) breaches vs True Look-Through breaches
        surface_breaches = sorted(
            [
                h["ticker"]
                for h in holdings
                if h["ticker"] not in ("CASH", "CASH_GBP", "CASH_USD", "QQQ", "SPY", "SMH", "VTI", "ARKK", "BND", "TLT")
                and float(h["weight"]) > limit + 1e-6
            ]
        )
        lookthrough_breaches = sorted(
            [
                exp.issuer
                for exp in top_named
                if not exp.is_fund_undisclosed and exp.total_weight > limit + 1e-6
            ]
        )
        # Hidden breaches: issuers that breach on look-through/issuer-mapping but NOT on raw surface ticker weights
        hidden_breaches = sorted(list(set(lookthrough_breaches) - set(surface_breaches)))

        cb_eval = safety_res.evaluations[0]
        record = {
            "case_id": spec["case_id"],
            "name": spec["name"],
            "persona": spec["persona"],
            "stated_goal": spec["stated_goal"],
            "max_risk_tolerance_single_stock": limit,
            "holdings": holdings,
            "ground_truth_lookthrough": {
                "computed_top_issuer": top_issuer,
                "computed_top_issuer_weight": top_weight,
                "computed_unresolved_other_constituents": other_w,
                "computed_total_weight": round(lt_res.total_weight, 6),
                "computed_direct_n_eff": round(conc_res.direct.effective_n, 4),
                "computed_lookthrough_n_eff": round(conc_res.lookthrough.effective_n, 4),
                "computed_circuit_breaker_status": cb_eval.status,
                "computed_surface_breaches": surface_breaches,
                "computed_lookthrough_breaches": lookthrough_breaches,
                "computed_hidden_breaches": hidden_breaches,
                "computed_dominant_theme": th_res.dominant_theme.theme_id if th_res.dominant_theme else "NONE",
                "computed_dominant_theme_weight": round(th_res.dominant_theme.total_weight, 6) if th_res.dominant_theme else 0.0,
                "computed_annualized_volatility": risk_res.annualized_volatility,
                "computed_var_95": risk_res.historical_var_95,
                "computed_max_drawdown": risk_res.max_drawdown,
                "computed_unpriced_holdings": risk_res.unpriced_holdings,
            },
        }

        if spec["split"] == "dev":
            dev_cases.append(record)
        else:
            holdout_cases.append(record)

    data_dir = Path(__file__).resolve().parent.parent / "data"
    holdout_dir = data_dir / "holdout"
    holdout_dir.mkdir(parents=True, exist_ok=True)

    dev_path = data_dir / "dev_expanded_portfolios.json"
    holdout_path = holdout_dir / "holdout_portfolios.json"

    with open(dev_path, "w", encoding="utf-8") as f:
        json.dump(dev_cases, f, indent=2)

    with open(holdout_path, "w", encoding="utf-8") as f:
        json.dump(holdout_cases, f, indent=2)

    print(f"Generated {len(dev_cases)} dev portfolios -> {dev_path}")
    print(f"Generated {len(holdout_cases)} holdout portfolios -> {holdout_path}")


if __name__ == "__main__":
    build_labelled_portfolios()
