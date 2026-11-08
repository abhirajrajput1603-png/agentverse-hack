import json
import os
import pandas as pd
import streamlit as st

from xray.agent.investigator import PortfolioInvestigator
from xray.data_loader import DataLoader
from xray.engine.benchmark import BenchmarkEngine
from xray.engine.concentration import ConcentrationEngine
from xray.engine.correlation import CorrelationAndRiskContributionEngine
from xray.engine.lookthrough import LookThroughEngine
from xray.engine.risk import RiskEngine
from xray.engine.stress import StressAndTailEngine
from xray.engine.whatif import WhatIfSimulator
from xray.profile import UserRiskProfile
from xray.safety.audit import AuditLogger
from xray.safety.limits import SafetyLimitsEngine


st.set_page_config(
    page_title="Portfolio X-Ray | Risk Intelligence v2",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .metric-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 12px;
    }
    .breach-badge {
        background-color: #fee2e2;
        color: #991b1b;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 700;
        display: inline-block;
    }
    .pass-badge {
        background-color: #dcfce7;
        color: #166534;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 700;
        display: inline-block;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_services():
    loader = DataLoader()
    audit_logger = AuditLogger()
    lt_engine = LookThroughEngine(loader)
    conc_engine = ConcentrationEngine()
    risk_engine = RiskEngine(loader)
    corr_engine = CorrelationAndRiskContributionEngine(loader)
    stress_engine = StressAndTailEngine(loader)
    bench_engine = BenchmarkEngine(loader)
    safety_engine = SafetyLimitsEngine(audit_logger=audit_logger)
    whatif_sim = WhatIfSimulator(data_loader=loader, audit_logger=audit_logger)
    investigator = PortfolioInvestigator()
    return (
        loader,
        audit_logger,
        lt_engine,
        conc_engine,
        risk_engine,
        corr_engine,
        stress_engine,
        bench_engine,
        safety_engine,
        whatif_sim,
        investigator,
    )


(
    loader,
    audit_logger,
    lt_engine,
    conc_engine,
    risk_engine,
    corr_engine,
    stress_engine,
    bench_engine,
    safety_engine,
    whatif_sim,
    investigator,
) = get_services()

synthetic_cases = loader.load_synthetic_portfolios()

# ---------------------------------------------------------------------------
# Sidebar: Portfolio Source, Risk Profile Form & Holdings Editor
# ---------------------------------------------------------------------------
st.sidebar.title("🔬 Portfolio X-Ray")
st.sidebar.caption("Autonomous Risk Intelligence · AgentVerse 2026")

case_options = [f"{c['case_id']}: {c['name']}" for c in synthetic_cases] + ["Custom Portfolio"]
selected_case_name = st.sidebar.selectbox("1. Select Portfolio Source", case_options)

active_persona = "Everyday Retail Investor"
active_goal = "Capital preservation with moderate growth"
active_limit = 0.20
default_holdings = []

if selected_case_name != "Custom Portfolio":
    case_idx = case_options.index(selected_case_name)
    case_data = synthetic_cases[case_idx]
    active_persona = case_data.get("persona", active_persona)
    active_goal = case_data.get("stated_goal", active_goal)
    active_limit = case_data.get("max_risk_tolerance_single_stock", 0.20)
    default_holdings = [
        {
            "ticker": h["ticker"],
            "weight": float(h["weight"]),
            "allocation_gbp": float(h.get("allocation_gbp", 0.0)),
        }
        for h in case_data["holdings"]
    ]
else:
    default_holdings = [
        {"ticker": "NVDA", "weight": 0.25, "allocation_gbp": 12500.0},
        {"ticker": "QQQ", "weight": 0.35, "allocation_gbp": 17500.0},
        {"ticker": "SMH", "weight": 0.25, "allocation_gbp": 12500.0},
        {"ticker": "SPY", "weight": 0.15, "allocation_gbp": 7500.0},
    ]

st.sidebar.markdown("---")
st.sidebar.subheader("2. Investor Risk Profile")
persona_input = st.sidebar.text_area("Persona Description", active_persona, height=68)
goal_input = st.sidebar.text_input("Stated Objective", active_goal)

col_p1, col_p2 = st.sidebar.columns(2)
with col_p1:
    horizon_years = st.number_input("Horizon (Yrs)", min_value=1, max_value=40, value=3, step=1)
with col_p2:
    max_loss_pct = st.number_input("Max Loss Tol (%)", min_value=5.0, max_value=80.0, value=15.0, step=5.0)

experience_level = st.sidebar.selectbox(
    "Experience Level",
    ["Beginner", "Intermediate", "Advanced"],
    index=0,
)
limit_input = (
    st.sidebar.slider(
        "Single-Stock Look-Through Limit (%)",
        min_value=5,
        max_value=50,
        value=int(active_limit * 100),
        step=1,
    )
    / 100.0
)

user_profile = UserRiskProfile(
    horizon_years=int(horizon_years),
    max_tolerable_loss_pct=float(max_loss_pct),
    objective=goal_input,
    experience_level=experience_level,
    single_stock_limit=limit_input,
)
st.session_state["user_risk_profile"] = user_profile

st.sidebar.markdown("---")
st.sidebar.subheader("3. Confirm Holdings")
holdings_df = pd.DataFrame(default_holdings)
edited_df = st.sidebar.data_editor(
    holdings_df,
    num_rows="dynamic",
    use_container_width=True,
    column_config={
        "ticker": st.column_config.TextColumn("Ticker", required=True),
        "weight": st.column_config.NumberColumn("Weight (0-1)", min_value=0.0, max_value=1.0, format="%.4f"),
        "allocation_gbp": st.column_config.NumberColumn("Value (£)", min_value=0.0, format="£%.2f"),
    },
)

confirm_button = st.sidebar.button("Confirm Holdings & Run X-Ray", type="primary", use_container_width=True)

# ---------------------------------------------------------------------------
# Main Dashboard Header & Quantitative Execution
# ---------------------------------------------------------------------------
st.title("Portfolio X-Ray: True Exposure & Risk Intelligence")
st.caption(
    "“You think your portfolio is diversified. We show you what you're really betting on.” · "
    "Zero LLM Mental Math · Full Traceable Evidence"
)

raw_holdings = [
    h for h in edited_df.to_dict("records") if h.get("ticker") and float(h.get("weight", 0.0)) > 0
]
if not raw_holdings:
    st.warning("Please enter at least one holding with a positive weight.")
    st.stop()

total_w = sum(float(h.get("weight", 0.0)) for h in raw_holdings)
if abs(total_w - 1.0) > 1e-4:
    st.sidebar.warning(f"Weights sum to {total_w:.4f}. Normalising to 1.0.")
    for h in raw_holdings:
        h["weight"] = float(h.get("weight", 0.0)) / total_w if total_w > 0 else 0.0

with st.spinner("Running deterministic quantitative engines..."):
    try:
        lt_result = lt_engine.analyze(raw_holdings)
        conc_analysis = conc_engine.analyze(lt_result)
        risk_metrics = risk_engine.analyze(raw_holdings)
        corr_result = corr_engine.analyze(raw_holdings)
        stress_result = stress_engine.analyze(raw_holdings, n_bootstrap=300)
        bench_result = bench_engine.analyze(raw_holdings)
        safety_result = safety_engine.evaluate_portfolio(
            lt_result=lt_result,
            concentration=conc_analysis,
            user_max_single_stock_limit=limit_input,
            persona_name=persona_input,
        )
    except Exception as exc:
        st.error(f"Quantitative engine error: {exc}")
        st.stop()

# 1. Circuit Breaker Banner
primary_eval = safety_result.evaluations[0] if safety_result.evaluations else None
if primary_eval:
    if primary_eval.status == "BREACH":
        st.error(
            f"🚨 **SAFETY CIRCUIT BREAKER TRIGGERED ({primary_eval.rule_name})** — "
            f"{primary_eval.message} *(Proof ID: `{primary_eval.evidence_id}`)*"
        )
    else:
        st.success(
            f"✅ **SAFETY CIRCUIT BREAKER PASSED ({primary_eval.rule_name})** — "
            f"{primary_eval.message} *(Proof ID: `{primary_eval.evidence_id}`)*"
        )

# 2. Executive KPI Row
kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
top_stock = conc_analysis.lookthrough.top_issuers[0] if conc_analysis.lookthrough.top_issuers else None
stock_name = top_stock["issuer"] if top_stock else "N/A"
stock_w = (top_stock["weight"] * 100) if top_stock else 0.0

with kpi1:
    st.metric(
        label="Top Look-Through Issuer",
        value=f"{stock_name} ({stock_w:.2f}%)",
        delta=f"{(stock_w - limit_input * 100):+.2f}% vs Policy Limit",
        delta_color="inverse",
    )
with kpi2:
    st.metric(
        label="Effective Holdings (N_eff)",
        value=f"{conc_analysis.lookthrough.effective_n:.2f}",
        delta=f"Direct N_eff: {conc_analysis.direct.effective_n:.2f}",
        delta_color="off",
        help="N_eff = 1 / HHI across named look-through issuers.",
    )
with kpi3:
    st.metric(
        label="3Y Annualised Volatility",
        value=f"{risk_metrics.annualized_volatility * 100:.2f}%",
        delta=f"Beta vs SPY: {bench_result.portfolio_beta_vs_spy:.2f}x",
        delta_color="off",
    )
with kpi4:
    st.metric(
        label="1-Day 95% VaR / ES",
        value=f"{risk_metrics.historical_var_95 * 100:.2f}%",
        delta=f"95% ES: {stress_result.tail_risk.es_95 * 100:.2f}%",
        delta_color="off",
    )
with kpi5:
    st.metric(
        label="3-Year Max Drawdown",
        value=f"{risk_metrics.max_drawdown * 100:.2f}%",
        delta=f"SPY Max DD: {bench_result.spy_benchmark.max_drawdown * 100:.2f}%",
        delta_color="off",
    )

if risk_metrics.unpriced_holdings:
    st.warning(
        f"⚠️ **Data Coverage Caveat:** Holdings `{risk_metrics.unpriced_holdings}` lack historical price series "
        f"in the market matrix. Risk & stress metrics are computed on the `{risk_metrics.priced_weight_coverage*100:.1f}%` priced portfolio coverage."
    )

st.markdown("---")

# ---------------------------------------------------------------------------
# 6 Interactive Dashboard v2 Tabs
# ---------------------------------------------------------------------------
(
    tab_lt,
    tab_corr,
    tab_stress,
    tab_bench,
    tab_whatif,
    tab_agent,
) = st.tabs(
    [
        "🔍 1. Look-Through Exposure",
        "⚖️ 2. Risk Contribution & Correlation",
        "📉 3. Tail Risk & Stress Replays",
        "📊 4. Benchmark Comparison",
        "🧪 5. What-If Simulator",
        "🤖 6. AI Investigator & Audit Log",
    ]
)

# --- TAB 1: LOOK-THROUGH EXPOSURE ---
with tab_lt:
    st.subheader("Where Your Money Really Goes: Surface vs. True Look-Through")
    c_left, c_right = st.columns(2)

    with c_left:
        st.markdown("**Surface Allocation (What You Think You Hold)**")
        surf_df = pd.DataFrame(
            [
                {"Ticker": h.ticker, "Surface Weight (%)": round(h.weight * 100, 2)}
                for h in lt_result.holdings
            ]
        ).set_index("Ticker")
        st.bar_chart(surf_df)

    with c_right:
        st.markdown("**True Look-Through Issuers (Top 10 Underlying Exposures)**")
        lt_top_df = pd.DataFrame(
            [
                {"Issuer": e.issuer, "Look-Through Weight (%)": round(e.total_weight * 100, 2)}
                for e in lt_result.sorted_exposures(exclude_residual_and_cash=False)[:10]
            ]
        ).set_index("Issuer")
        st.bar_chart(lt_top_df)

    st.markdown("#### Complete Issuer Decomposition Table")
    lt_rows = []
    for exp in lt_result.sorted_exposures(exclude_residual_and_cash=False):
        sources_str = ", ".join(
            [f"{src}: {w * 100:.2f}%" for src, w in exp.breakdown_by_source.items()]
        )
        lt_rows.append(
            {
                "Issuer": exp.issuer,
                "Total Look-Through (%)": f"{exp.total_weight * 100:.2f}%",
                "Direct Weight (%)": f"{exp.direct_weight * 100:.2f}%",
                "Indirect via ETFs (%)": f"{exp.indirect_weight * 100:.2f}%",
                "Source Breakdown": sources_str,
            }
        )
    st.dataframe(pd.DataFrame(lt_rows), use_container_width=True, height=340)

# --- TAB 2: RISK CONTRIBUTION & CORRELATION ---
with tab_corr:
    st.subheader("Capital Weight vs. Euler Risk Contribution (MCR & CCR)")
    st.caption(
        f"Euler Homogeneity Check: ∑ CCR_i = {corr_result.ccr_sum * 100:.2f}% == "
        f"Portfolio Volatility σ_p = {corr_result.portfolio_volatility * 100:.2f}% "
        f"({'PASSED ✅' if corr_result.ccr_sum_check_passed else 'FAILED ❌'})"
    )

    rc_col1, rc_col2 = st.columns([3, 2])
    with rc_col1:
        rc_rows = []
        for c in corr_result.contributions:
            rc_rows.append(
                {
                    "Ticker": c.ticker,
                    "Capital Weight (%)": round(c.capital_weight * 100, 2),
                    "Risk Share %CR (%)": round(c.pct_risk_contribution * 100, 2),
                    "MCR (%)": round(c.mcr * 100, 2),
                    "CCR (%)": round(c.ccr * 100, 2),
                    "Risk / Capital Ratio": f"{c.risk_to_capital_ratio:.2f}x",
                }
            )
        st.dataframe(pd.DataFrame(rc_rows), use_container_width=True)

    with rc_col2:
        if corr_result.contributions:
            chart_rc_df = pd.DataFrame(
                [
                    {
                        "Ticker": c.ticker,
                        "Capital Weight (%)": round(c.capital_weight * 100, 2),
                        "Risk Contribution (%)": round(c.pct_risk_contribution * 100, 2),
                    }
                    for c in corr_result.contributions
                ]
            ).set_index("Ticker")
            st.bar_chart(chart_rc_df)

    st.markdown("---")
    st.subheader("Pairwise Return Correlation Matrix (3-Year vs. 1-Year)")
    corr_m1, corr_m2 = st.columns(2)
    with corr_m1:
        st.markdown(
            f"**3-Year Correlation Matrix** *(Avg Pairwise: `{corr_result.avg_pairwise_corr_3y:.4f}`, "
            f"Sample: `{corr_result.sample_size_3y}` days)*"
        )
        if corr_result.corr_matrix_3y:
            df_c3 = pd.DataFrame(corr_result.corr_matrix_3y)
            st.dataframe(df_c3.style.background_gradient(cmap="RdYlGn_r", vmin=-1.0, vmax=1.0).format("{:.3f}"), use_container_width=True)
    with corr_m2:
        st.markdown(
            f"**1-Year Correlation Matrix** *(Avg Pairwise: `{corr_result.avg_pairwise_corr_1y:.4f}`, "
            f"Sample: `{corr_result.sample_size_1y}` days)*"
        )
        if corr_result.corr_matrix_1y:
            df_c1 = pd.DataFrame(corr_result.corr_matrix_1y)
            st.dataframe(df_c1.style.background_gradient(cmap="RdYlGn_r", vmin=-1.0, vmax=1.0).format("{:.3f}"), use_container_width=True)

# --- TAB 3: TAIL RISK & HISTORICAL STRESS REPLAYS ---
with tab_stress:
    st.subheader("Tail Risk & Bootstrap Confidence Intervals (3-Year Daily Returns)")
    tr = stress_result.tail_risk
    t_col1, t_col2, t_col3, t_col4 = st.columns(4)
    t_col1.metric(
        "1-Day 95% Historical VaR",
        f"{tr.var_95 * 100:.2f}%",
        f"95% CI: [{tr.var_95_ci_low * 100:.2f}%, {tr.var_95_ci_high * 100:.2f}%]",
        delta_color="off",
    )
    t_col2.metric(
        "1-Day 95% Expected Shortfall (ES)",
        f"{tr.es_95 * 100:.2f}%",
        f"95% CI: [{tr.es_95_ci_low * 100:.2f}%, {tr.es_95_ci_high * 100:.2f}%]",
        delta_color="off",
    )
    t_col3.metric(
        "1-Day 99% Historical VaR",
        f"{tr.var_99 * 100:.2f}%",
        f"Tail Days (95% / 99%): {tr.tail_sample_days_95} / {tr.tail_sample_days_99}",
        delta_color="off",
    )
    t_col4.metric(
        "1-Day 99% Expected Shortfall (ES)",
        f"{tr.es_99 * 100:.2f}%",
        f"Sample Size: {tr.total_sample_days} trading days",
        delta_color="off",
    )

    st.markdown("---")
    st.subheader("Historical Crisis Stress Replays (Portfolio vs. SPY)")
    stress_rows = []
    stress_chart_data = []
    for scen_id, sc in stress_result.scenarios.items():
        stress_rows.append(
            {
                "Crisis Scenario": sc.name,
                "Window": f"{sc.start_date} → {sc.end_date}",
                "Portfolio Return (%)": f"{sc.portfolio_cumulative_return * 100:+.2f}%",
                "Portfolio Max DD (%)": f"{sc.portfolio_max_drawdown * 100:.2f}%",
                "Worst 1-Day Drop (%)": f"{sc.worst_single_day_return * 100:.2f}%",
                "SPY Return (%)": f"{sc.spy_cumulative_return * 100:+.2f}%",
                "Coverage (%)": f"{sc.coverage_weight * 100:.1f}%",
                "Missing History": ", ".join(sc.missing_history_holdings) if sc.missing_history_holdings else "None",
            }
        )
        stress_chart_data.append(
            {
                "Scenario": sc.name,
                "Portfolio Return (%)": round(sc.portfolio_cumulative_return * 100, 2),
                "SPY Benchmark (%)": round(sc.spy_cumulative_return * 100, 2),
            }
        )

    st.bar_chart(pd.DataFrame(stress_chart_data).set_index("Scenario"))
    st.dataframe(pd.DataFrame(stress_rows), use_container_width=True)

# --- TAB 4: BENCHMARK COMPARISON ---
with tab_bench:
    st.subheader("Institutional Benchmark Comparison: SPY & 60/40 SPY/BND")
    b_rows = []
    for prof in (
        bench_result.user_metrics,
        bench_result.spy_benchmark,
        bench_result.balanced_60_40_benchmark,
    ):
        b_rows.append(
            {
                "Profile": prof.name,
                "Description": prof.label,
                "Top-1 Issuer": f"{prof.top_1_issuer} ({prof.top_1_weight * 100:.2f}%)",
                "Top-5 Weight (%)": f"{prof.top_5_weight * 100:.2f}%",
                "Look-Through N_eff": f"{prof.lookthrough_n_eff:.2f}",
                "Annualised Vol (%)": f"{prof.annualized_volatility * 100:.2f}%",
                "1-Day 95% VaR (%)": f"{prof.historical_var_95 * 100:.2f}%",
                "1-Day 95% ES (%)": f"{prof.expected_shortfall_95 * 100:.2f}%",
                "3Y Max Drawdown (%)": f"{prof.max_drawdown * 100:.2f}%",
            }
        )
    st.dataframe(pd.DataFrame(b_rows), use_container_width=True)

    st.markdown("#### Active Look-Through Concentration Tilts vs. SPY (US Large-Cap Proxy)")
    tilt_rows = []
    for t in bench_result.active_tilts_vs_spy[:10]:
        tilt_rows.append(
            {
                "Issuer": t.issuer,
                "User Look-Through (%)": f"{t.user_weight * 100:.2f}%",
                "SPY Weight (%)": f"{t.spy_weight * 100:.2f}%",
                "Active Tilt (%)": f"{t.active_tilt * 100:+.2f}%",
                "Multiple of SPY": f"{t.multiple_of_spy:.2f}x" if t.multiple_of_spy is not None else "Not in SPY",
            }
        )
    st.dataframe(pd.DataFrame(tilt_rows), use_container_width=True)

# --- TAB 5: WHAT-IF SIMULATOR & SIGN-OFF ---
with tab_whatif:
    st.subheader("What-If Scenario Simulator with Institutional Friction & Circuit Breaker")
    st.caption(
        "Simulates portfolio changes, deducts bid-ask spread, ticket commission (£4.99), and quadratic market-impact slippage, "
        "and re-evaluates safety circuit breakers. Nothing is ever executed."
    )

    sim_mode = st.radio(
        "Select Scenario Type",
        ["Add New Cash to Ticker", "Rebalance / Rotate Position (Sell A → Buy B)"],
        horizontal=True,
    )

    all_known_tickers = sorted(list(loader.load_prices().columns))
    current_tickers = [h["ticker"] for h in raw_holdings]

    sim_comparison = None
    sim_error = None

    if sim_mode == "Add New Cash to Ticker":
        wc1, wc2 = st.columns(2)
        with wc1:
            default_idx = all_known_tickers.index("NVDA") if "NVDA" in all_known_tickers else 0
            target_ticker = st.selectbox("Target Ticker to Add", all_known_tickers, index=default_idx)
        with wc2:
            cash_to_add = st.number_input("Additional Cash (£)", min_value=100.0, max_value=1000000.0, value=15000.0, step=1000.0)

        try:
            sim_comparison = whatif_sim.simulate_add_cash(
                current_holdings=raw_holdings,
                target_ticker=target_ticker,
                additional_cash_gbp=float(cash_to_add),
                persona_name=persona_input,
                user_max_single_stock_limit=limit_input,
            )
        except Exception as e:
            sim_error = str(e)
    else:
        wc1, wc2, wc3 = st.columns(3)
        with wc1:
            sell_ticker = st.selectbox("Sell Existing Holding", current_tickers)
        with wc2:
            sell_amount = st.number_input("Amount to Sell (£)", min_value=100.0, max_value=500000.0, value=5000.0, step=500.0)
        with wc3:
            default_buy_idx = all_known_tickers.index("TLT") if "TLT" in all_known_tickers else 0
            buy_ticker = st.selectbox("Rotate Proceeds Into", all_known_tickers, index=default_buy_idx)

        try:
            sim_comparison = whatif_sim.simulate_rebalance(
                current_holdings=raw_holdings,
                sell_ticker=sell_ticker,
                sell_gbp=float(sell_amount),
                buy_ticker=buy_ticker,
                persona_name=persona_input,
                user_max_single_stock_limit=limit_input,
            )
        except Exception as e:
            sim_error = str(e)

    if sim_error:
        st.error(f"Scenario error: {sim_error}")
    elif sim_comparison:
        st.info(f"**Scenario:** {sim_comparison.scenario_description}")

        # Post-simulation circuit breaker banner
        after_eval = sim_comparison.after_safety.evaluations[0] if sim_comparison.after_safety.evaluations else None
        if after_eval:
            if after_eval.status == "BREACH":
                st.error(
                    f"🛑 **CIRCUIT BREAKER HALT ON SIMULATED PORTFOLIO:** {after_eval.message} "
                    f"*(Violated Limit: {after_eval.threshold * 100:.1f}%, Excess: +{after_eval.excess * 100:.2f}%)*"
                )
            else:
                st.success(f"✅ **SIMULATED PORTFOLIO PASSES SAFETY LIMITS:** {after_eval.message}")

        # Before vs After Comparison Table
        b_top = sim_comparison.before_concentration.lookthrough.top_issuers[0]
        a_top = sim_comparison.after_concentration.lookthrough.top_issuers[0]
        comp_df = pd.DataFrame(
            [
                {
                    "Metric": "Portfolio Value (£)",
                    "Before": f"£{sim_comparison.before_portfolio_value_gbp:,.2f}",
                    "After (Net of Friction)": f"£{sim_comparison.after_portfolio_value_gbp:,.2f}",
                    "Delta": f"£{sim_comparison.after_portfolio_value_gbp - sim_comparison.before_portfolio_value_gbp:+,.2f}",
                },
                {
                    "Metric": "Top-1 Look-Through Issuer",
                    "Before": f"{b_top['issuer']} ({b_top['weight']*100:.2f}%)",
                    "After (Net of Friction)": f"{a_top['issuer']} ({a_top['weight']*100:.2f}%)",
                    "Delta": f"{(a_top['weight'] - b_top['weight'])*100:+.2f}%",
                },
                {
                    "Metric": "Look-Through HHI",
                    "Before": f"{sim_comparison.before_concentration.lookthrough.hhi:.4f}",
                    "After (Net of Friction)": f"{sim_comparison.after_concentration.lookthrough.hhi:.4f}",
                    "Delta": f"{sim_comparison.after_concentration.lookthrough.hhi - sim_comparison.before_concentration.lookthrough.hhi:+.4f}",
                },
                {
                    "Metric": "Effective Holdings (N_eff)",
                    "Before": f"{sim_comparison.before_concentration.lookthrough.effective_n:.2f}",
                    "After (Net of Friction)": f"{sim_comparison.after_concentration.lookthrough.effective_n:.2f}",
                    "Delta": f"{sim_comparison.after_concentration.lookthrough.effective_n - sim_comparison.before_concentration.lookthrough.effective_n:+.2f}",
                },
                {
                    "Metric": "Annualised Volatility (3Y)",
                    "Before": f"{sim_comparison.before_risk.annualized_volatility*100:.2f}%",
                    "After (Net of Friction)": f"{sim_comparison.after_risk.annualized_volatility*100:.2f}%",
                    "Delta": f"{(sim_comparison.after_risk.annualized_volatility - sim_comparison.before_risk.annualized_volatility)*100:+.2f}%",
                },
                {
                    "Metric": "1-Day 95% Historical VaR",
                    "Before": f"{sim_comparison.before_risk.historical_var_95*100:.2f}%",
                    "After (Net of Friction)": f"{sim_comparison.after_risk.historical_var_95*100:.2f}%",
                    "Delta": f"{(sim_comparison.after_risk.historical_var_95 - sim_comparison.before_risk.historical_var_95)*100:+.2f}%",
                },
                {
                    "Metric": "3-Year Max Drawdown",
                    "Before": f"{sim_comparison.before_risk.max_drawdown*100:.2f}%",
                    "After (Net of Friction)": f"{sim_comparison.after_risk.max_drawdown*100:.2f}%",
                    "Delta": f"{(sim_comparison.after_risk.max_drawdown - sim_comparison.before_risk.max_drawdown)*100:+.2f}%",
                },
            ]
        )
        st.dataframe(comp_df, use_container_width=True)

        # Friction Breakdown Table
        st.markdown("#### Transaction Friction Breakdown (`config/friction.json`)")
        f_rows = [
            {
                "Leg": f"{tf.trade_type} {tf.ticker}",
                "Gross Trade (£)": f"£{tf.trade_value_gbp:,.2f}",
                "Bid-Ask Spread (£)": f"£{tf.spread_cost_gbp:.2f}",
                "Commission (£)": f"£{tf.commission_gbp:.2f}",
                "Quadratic Slippage (£)": f"£{tf.slippage_gbp:.2f}",
                "Total Friction (£)": f"£{tf.total_friction_gbp:.2f} ({tf.friction_bps:.1f} bps)",
            }
            for tf in sim_comparison.trade_frictions
        ]
        st.dataframe(pd.DataFrame(f_rows), use_container_width=True)

        # Human-in-the-loop explicit sign-off
        st.markdown("#### Mandatory Human-in-the-Loop Scenario Sign-Off")
        ack_checked = st.checkbox(
            "I understand the simulated concentration/risk impact and friction costs; record this scenario for my review (no trades are ever executed)."
        )
        if st.button("Record Scenario Sign-Off", disabled=not ack_checked):
            audit_logger.log_event(
                event_type="USER_WHATIF_SCENARIO_SIGNOFF",
                details={
                    "persona": persona_input,
                    "scenario": sim_comparison.scenario_description,
                    "after_circuit_breaker_safe": sim_comparison.after_safety.is_safe,
                    "total_friction_gbp": sim_comparison.total_friction_gbp,
                },
            )
            st.success("Scenario sign-off recorded to tamper-evident audit log (`logs/audit.jsonl`).")

# --- TAB 6: AI INVESTIGATOR (AGENT V2) & AUDIT LOG ---
with tab_agent:
    st.subheader("Autonomous AI Risk Investigator (Agent v2)")
    use_live_llm = st.toggle(
        "Use Live Gemini 3.8 Flash Synthesis (uncheck for instant deterministic synthesis)",
        value=bool(os.environ.get("GEMINI_API_KEY")),
    )

    with st.spinner("Running 6-tool autonomous investigation loop..."):
        investigation_report = investigator.investigate_autonomous(
            holdings=raw_holdings,
            user_goal=goal_input,
            persona_name=persona_input,
            user_max_single_stock_limit=limit_input,
            user_profile=user_profile,
            data_loader=loader,
            audit_logger=audit_logger,
            use_llm=use_live_llm,
        )

    st.markdown(
        f"**Autonomous Tool Trace (`{investigation_report.tool_calls_count} / 8` max tool calls executed):**"
    )
    trace_rows = [
        {
            "Step": t.step,
            "Tool Invoked": t.tool_name,
            "Investigation Rationale": t.reason,
            "Evidence Objects Produced": len(t.evidence_ids_produced),
        }
        for t in investigation_report.tool_trace
    ]
    st.dataframe(pd.DataFrame(trace_rows), use_container_width=True)

    st.markdown("#### Verified Risk Findings")
    for idx, finding in enumerate(investigation_report.findings):
        icon = "🚨" if finding.severity == "CRITICAL" else ("⚠️" if finding.severity == "WARNING" else "ℹ️")
        with st.expander(f"{icon} **[{finding.severity}] {finding.title}**", expanded=(idx < 2)):
            st.write(finding.explanation)
            st.markdown("**Traceable Mathematical Evidence Citations:**")
            for cid in finding.evidence_ids:
                matching_ev = next((e for e in investigation_report.raw_evidence if e.id == cid), None)
                if matching_ev:
                    st.code(
                        f"Evidence ID : {matching_ev.id}\n"
                        f"Metric      : {matching_ev.metric}\n"
                        f"Value       : {matching_ev.value}\n"
                        f"Formula     : {matching_ev.formula}\n"
                        f"Data Source : {', '.join(matching_ev.data_sources)}\n"
                        f"Caveats     : {', '.join(matching_ev.caveats) if matching_ev.caveats else 'None'}",
                        language="yaml",
                    )

    st.markdown("---")
    st.subheader("Tamper-Evident Audit Log (`logs/audit.jsonl`)")
    recent_logs = audit_logger.get_recent_entries(limit=25)
    if recent_logs:
        st.dataframe(pd.DataFrame(recent_logs[::-1]), use_container_width=True, height=280)

    with st.expander("🔍 View Complete Evidence Registry JSON"):
        st.json([e.to_dict() for e in investigation_report.raw_evidence])
