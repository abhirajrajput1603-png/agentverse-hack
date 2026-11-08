"""Unit tests for Portfolio X-Ray core quantitative risk engine."""

import pytest
import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

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
from portfolio_xray.agent import ask_risk_analyst, SCOPE_REFUSAL_MESSAGE


class TestPortfolioXRayEngine(unittest.TestCase):

    def setUp(self):
        # Accidental Tech Overweight sample portfolio
        # $6,500 NVDA direct (20%)
        # $12,000 QQQ (36.9%)
        # $7,500 VOO (23.1%)
        # $6,500 SMH (20.0%)
        # Total Value: $32,500
        self.portfolio = Portfolio(
            holdings=[
                Holding(ticker="NVDA", shares=50, price=130.0),    # $6,500
                Holding(ticker="QQQ", shares=25, price=480.0),     # $12,000
                Holding(ticker="VOO", shares=15, price=500.0),     # $7,500
                Holding(ticker="SMH", shares=26, price=250.0),     # $6,500
            ],
            user_profile={
                "risk_tolerance": "Moderate",
                "horizon_years": 7,
                "loss_capacity": "Moderate (15-20% max drawdown)",
            }
        )

    def test_lookthrough_decomposition(self):
        exposures = compute_lookthrough_exposure(self.portfolio)
        self.assertTrue(len(exposures) > 0)

        # Find NVDA exposure
        nvda_exp = next((e for e in exposures if e.ticker == "NVDA"), None)
        self.assertIsNotNone(nvda_exp)

        # Direct NVDA is 20% ($6,500 / $32,500)
        self.assertAlmostEqual(nvda_exp.direct_weight, 0.20, places=2)

        # Indirect NVDA comes from QQQ (~8.6%), VOO (~6.2%), and SMH (~20.8%)
        self.assertTrue(nvda_exp.indirect_weight > 0.05)
        
        # Effective NVDA weight must be direct + indirect (~28-32%)
        self.assertAlmostEqual(nvda_exp.total_weight, nvda_exp.direct_weight + nvda_exp.indirect_weight, places=4)
        self.assertTrue(nvda_exp.total_weight > 0.25)

        # Verify contributing sources exist
        self.assertIn("QQQ", nvda_exp.contributing_sources)
        self.assertIn("SMH", nvda_exp.contributing_sources)
        self.assertIn("VOO", nvda_exp.contributing_sources)

        # Verify that non-individually invested components (e.g. AAPL, MSFT) are NOT separated from indexes
        exp_tickers = {e.ticker for e in exposures}
        self.assertNotIn("AAPL", exp_tickers)
        self.assertNotIn("MSFT", exp_tickers)
        self.assertNotIn("_OTHER_QQQ", exp_tickers)
        # Indexes remain intact with their remaining weights
        self.assertIn("QQQ", exp_tickers)
        self.assertIn("VOO", exp_tickers)
        self.assertIn("SMH", exp_tickers)
        # Total weights must sum to 100%
        self.assertAlmostEqual(sum(e.total_weight for e in exposures), 1.0, places=4)

    def test_pure_index_not_broken_down(self):
        """When an investor holds only an index/ETF, it must NOT be broken down into individual stocks."""
        qqq_portfolio = Portfolio(
            holdings=[Holding(ticker="QQQ", shares=100, price=480.0)],
        )
        exposures = compute_lookthrough_exposure(qqq_portfolio)
        self.assertEqual(len(exposures), 1)
        self.assertEqual(exposures[0].ticker, "QQQ")
        self.assertAlmostEqual(exposures[0].total_weight, 1.0, places=4)
        self.assertEqual(exposures[0].direct_weight, 1.0)
        self.assertEqual(exposures[0].indirect_weight, 0.0)

    def test_index_only_breaks_down_individually_invested_components(self):
        """QQQ exposure only extracts constituents the user has directly invested in."""
        portfolio = Portfolio(
            holdings=[
                Holding(ticker="QQQ", shares=100, price=480.0),   # $48,000 (80%)
                Holding(ticker="TSLA", shares=50, price=240.0),   # $12,000 (20%)
            ]
        )
        exposures = compute_lookthrough_exposure(portfolio)
        exp_dict = {e.ticker: e for e in exposures}

        # Only TSLA and QQQ should exist in exposures
        self.assertEqual(set(exp_dict.keys()), {"TSLA", "QQQ"})

        # TSLA is in QQQ with ~2.9% weight
        # Indirect TSLA = 80% * 0.029 = 2.32%
        # Direct TSLA = 20%
        # Total TSLA = 22.32%
        self.assertAlmostEqual(exp_dict["TSLA"].direct_weight, 0.20, places=3)
        self.assertTrue(exp_dict["TSLA"].indirect_weight > 0.02)
        self.assertIn("QQQ", exp_dict["TSLA"].contributing_sources)

        # Remaining QQQ = 80% * (1 - 0.029) = 77.68%
        self.assertAlmostEqual(exp_dict["QQQ"].total_weight, 0.80 - exp_dict["TSLA"].indirect_weight, places=4)

        # Non-invested components (e.g. NVDA, AAPL) must not appear
        self.assertNotIn("NVDA", exp_dict)
        self.assertNotIn("AAPL", exp_dict)

        # Total weights must equal 100%
        self.assertAlmostEqual(sum(e.total_weight for e in exposures), 1.0, places=4)

    def test_thematic_exposure_removes_under_5_percent(self):
        """Verifies that any themes with exposure strictly below 5% are omitted from thematic breakdown."""
        # Create a portfolio with an asset having tiny 2% exposure
        # e.g., TSLA at 2% has themes 'EV & Clean Tech', 'Autonomous Systems'
        # NVDA at 98% has themes 'AI Infrastructure & Capex', 'Semiconductor Supply Chain', 'High Growth Tech'
        portfolio = Portfolio(
            holdings=[
                Holding(ticker="NVDA", shares=98, price=100.0),  # 98% ($9,800)
                Holding(ticker="TSLA", shares=2, price=100.0),   # 2% ($200)
            ]
        )
        exposures = compute_lookthrough_exposure(portfolio)
        thematics = compute_thematic_exposure(exposures)

        theme_names = [t["theme"] for t in thematics]
        
        # NVDA themes (98%) must be present
        self.assertIn("AI Infrastructure & Capex", theme_names)
        self.assertIn("Semiconductor Supply Chain", theme_names)

        # TSLA unique themes (2% exposure) must be filtered out (< 5%)
        self.assertNotIn("EV & Clean Tech", theme_names)
        self.assertNotIn("Autonomous Systems", theme_names)

        # Verify every returned theme has weight >= 0.05
        for t in thematics:
            self.assertGreaterEqual(t["weight"], 0.05)

    def test_risk_metrics_and_concentration(self):
        exposures = compute_lookthrough_exposure(self.portfolio)
        metrics = compute_risk_metrics(self.portfolio, exposures)

        self.assertEqual(metrics["portfolio_value"], 32500.0)
        self.assertTrue(metrics["hhi"] > 1000)
        self.assertTrue(metrics["annual_volatility"] > 0)
        self.assertTrue(metrics["var_95_1m_dollar"] > 0)
        self.assertTrue(metrics["expected_shortfall_95_dollar"] > metrics["var_95_1m_dollar"])

    def test_stress_testing(self):
        exposures = compute_lookthrough_exposure(self.portfolio)
        stress_results = run_stress_tests(self.portfolio, exposures)

        self.assertEqual(len(stress_results), 4)
        for s in stress_results:
            self.assertIn("scenario_id", s)
            self.assertIn("drawdown_pct", s)
            self.assertIn("dollar_drawdown", s)
            self.assertIn("worst_hit_assets", s)

    def test_dynamic_stress_scenarios_ai_exposure(self):
        """Verifies that high AI / semiconductor portfolios receive AI-related stress scenarios."""
        ai_portfolio = Portfolio(
            holdings=[
                Holding(ticker="NVDA", shares=100, price=130.0),
                Holding(ticker="SMH", shares=50, price=250.0),
            ]
        )
        exposures = compute_lookthrough_exposure(ai_portfolio)
        stress = run_stress_tests(ai_portfolio, exposures)
        
        scenario_ids = [s["scenario_id"] for s in stress]
        # Must feature prominent AI and semiconductor shock scenarios
        self.assertIn("ai_capex_pause_2024", scenario_ids)
        self.assertIn("taiwan_semiconductor_blockade", scenario_ids)
        # Verify category badges
        ai_categories = [s.get("category") for s in stress]
        self.assertTrue(ai_categories.count("ai") >= 2)

    def test_dynamic_stress_scenarios_fig_rates_exposure(self):
        """Verifies that FIG (financial institutions group / banking) portfolios receive rates and banking stress scenarios."""
        fig_portfolio = Portfolio(
            holdings=[
                Holding(ticker="JPM", shares=100, price=220.0),
                Holding(ticker="XLF", shares=300, price=45.0),
            ]
        )
        exposures = compute_lookthrough_exposure(fig_portfolio)
        stress = run_stress_tests(fig_portfolio, exposures)

        scenario_ids = [s["scenario_id"] for s in stress]
        # Must feature banking run and yield curve rate shocks
        self.assertIn("regional_bank_liquidity_run", scenario_ids)
        self.assertIn("yield_curve_inversion_nim_crunch", scenario_ids)
        fig_categories = [s.get("category") for s in stress]
        self.assertTrue(fig_categories.count("fig") >= 2)

    def test_dynamic_stress_scenarios_aerospace_defense(self):
        """Verifies that aerospace & defense portfolios (e.g. SpaceX, Lockheed, Boeing) receive space/defense shocks."""
        defense_portfolio = Portfolio(
            holdings=[
                Holding(ticker="SPACEX", shares=50, price=112.0),
                Holding(ticker="LMT", shares=20, price=580.0),
                Holding(ticker="BA", shares=30, price=155.0),
            ]
        )
        exposures = compute_lookthrough_exposure(defense_portfolio)
        stress = run_stress_tests(defense_portfolio, exposures)

        scenario_ids = [s["scenario_id"] for s in stress]
        self.assertTrue(
            "space_launch_anomaly_grounding" in scenario_ids or
            "defense_budget_sequestration" in scenario_ids
        )
        categories = [s.get("category") for s in stress]
        self.assertTrue("defense" in categories)

    def test_what_if_simulation_breach(self):
        # Moderate investor putting 60% into NVDA should trigger warnings
        sim = simulate_what_if(self.portfolio, {"NVDA": 60.0, "SPY": 40.0})

        self.assertIn("metrics_delta", sim)
        self.assertIn("profile_warnings", sim)
        self.assertFalse(sim["is_aligned_with_profile"])
        self.assertTrue(any("concentration" in w.lower() for w in sim["profile_warnings"]))

    def test_what_if_simulation_with_stock_names(self):
        """Verifies that what-if simulation resolves stock names (e.g. 'apple', 'Nvidia', 'SpaceX') to canonical tickers."""
        sim = simulate_what_if(self.portfolio, {"apple": 50.0, "SPY": 50.0})
        self.assertEqual(sim["resolved_target"]["ticker"], "AAPL")
        self.assertEqual(sim["resolved_target"]["name"], "Apple Inc.")
        self.assertIn("proposed_metrics", sim)

        sim_spacex = simulate_what_if(self.portfolio, {"SpaceX": 40.0, "SPY": 60.0})
        self.assertEqual(sim_spacex["resolved_target"]["ticker"], "SPACEX")

    def test_counterexample_all_cash_portfolio(self):
        """Counterexample test: An all-cash portfolio must not trigger false volatility or drawdowns."""
        cash_portfolio = Portfolio(
            holdings=[Holding(ticker="CASH", shares=10000, price=1.0)],
            user_profile={"risk_tolerance": "Conservative", "horizon_years": 1, "loss_capacity": "Low"}
        )
        exposures = compute_lookthrough_exposure(cash_portfolio)
        metrics = compute_risk_metrics(cash_portfolio, exposures)
        stress = run_stress_tests(cash_portfolio, exposures)

        self.assertEqual(metrics["portfolio_value"], 10000.0)
        self.assertAlmostEqual(metrics["annual_volatility"], 0.5, delta=0.2) # ~0% vol
        self.assertTrue(metrics["var_95_1m_dollar"] < 50.0) # minimal VaR

        # In stress tests, cash is protected (drawdown >= 0%)
        for s in stress:
            self.assertTrue(s["drawdown_pct"] >= 0.0)

    def test_headline_event_analyzer(self):
        # 1. Rate hike test -> negative impact on tech portfolio
        res_hike = analyze_headline_event(self.portfolio, "Federal Reserve hikes interest rates by 50 basis points to fight inflation")
        self.assertIn("affected_holdings", res_hike)
        self.assertTrue(res_hike["net_estimated_impact_pct"] < 0)
        self.assertTrue(len(res_hike["affected_holdings"]) > 0)

        # 2. Rate cut test -> positive impact on tech portfolio
        res_cut = analyze_headline_event(self.portfolio, "Fed signals 50 bps surprise rate cut as inflation falls")
        self.assertTrue(res_cut["net_estimated_impact_pct"] > 0)
        self.assertIn("Rate cuts and cooling inflation", res_cut["rationale"])

        # 3. Direct company mention test (SpaceX)
        space_portfolio = Portfolio(holdings=[Holding(ticker="SPACEX", shares=100, price=180.0)])
        res_space = analyze_headline_event(space_portfolio, "SpaceX achieves historic Starship launch, Pentagon signs $10B contract")
        self.assertTrue(res_space["net_estimated_impact_pct"] > 0)
        self.assertEqual(res_space["affected_holdings"][0]["ticker"], "SPACEX")
        self.assertTrue(res_space["affected_holdings"][0]["estimated_shock_pct"] > 0)

        # 4. Oil crash test -> energy negative, consumer positive
        oil_portfolio = Portfolio(holdings=[Holding(ticker="XOM", shares=100, price=115.0)])
        res_oil = analyze_headline_event(oil_portfolio, "Crude oil plunges 12% as OPEC increases production")
        self.assertTrue(res_oil["net_estimated_impact_pct"] < 0)

    def test_out_of_scope_query_refusal(self):
        """Verifies that non-financial queries (e.g. cooking, trivia) are strictly refused."""
        off_topic_queries = [
            "how do i make eggs",
            "what is the recipe for chocolate cake?",
            "who was the first president of the United States?",
            "write python code to reverse a linked list",
            "tell me a funny bedtime story",
        ]
        for q in off_topic_queries:
            res = ask_risk_analyst(self.portfolio, q)
            self.assertEqual(res["source"], "scope-guardrail")
            self.assertIn(SCOPE_REFUSAL_MESSAGE, res["response"])

    def test_in_scope_finance_queries_accepted(self):
        """Verifies that legitimate portfolio, ticker, and risk queries are accepted and answered."""
        valid_queries = [
            "how much NVDA do I really own?",
            "is my portfolio safe?",
            "how do I make money with these stocks?",
            "what happens if interest rates rise?",
            "what if I allocate 50% to NVDA?",
        ]
        for q in valid_queries:
            res = ask_risk_analyst(self.portfolio, q)
            self.assertNotEqual(res["source"], "scope-guardrail")
            self.assertTrue(len(res["response"]) > 50)


if __name__ == "__main__":
    unittest.main()
