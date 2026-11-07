import json
import os
import pandas as pd
import streamlit as st

from xray.data_loader import DataLoader
from xray.engine.lookthrough import LookThroughEngine
from xray.engine.concentration import ConcentrationEngine
from xray.engine.risk import RiskEngine
from xray.safety.limits import SafetyLimitsEngine
from xray.agent.investigator import PortfolioInvestigator


st.set_page_config(
    page_title="Portfolio X-Ray | Risk Intelligence",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Institutional CSS styling
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
    lt_engine = LookThroughEngine(loader)
    conc_engine = ConcentrationEngine()
    risk_engine = RiskEngine(loader)
    safety_engine = SafetyLimitsEngine()
    investigator = PortfolioInvestigator()
    return loader, lt_engine, conc_engine, risk_engine, safety_engine, investigator


loader, lt_engine, conc_engine, risk_engine, safety_engine, investigator = get_services()
synthetic_cases = loader.load_synthetic_portfolios()

# Sidebar: Configuration & Portfolio Input
st.sidebar.title("🔬 Portfolio X-Ray")
st.sidebar.caption("AI-Powered Risk Intelligence · AgentVerse 2026")

case_options = ["Custom Portfolio"] + [f"{c['case_id']}: {c['name']}" for c in synthetic_cases]
selected_case_name = st.sidebar.selectbox("Select Portfolio Source", case_options)

active_persona = "Everyday Investor"
active_goal = "Capital preservation & growth"
active_limit = 0.20
default_holdings = []

if selected_case_name != "Custom Portfolio":
    case_idx = case_options.index(selected_case_name) - 1
    case_data = synthetic_cases[case_idx]
    active_persona = case_data.get("persona", active_persona)
    active_goal = case_data.get("stated_goal", active_goal)
    active_limit = case_data.get("max_risk_tolerance_single_stock", 0.20)
    default_holdings = [
        {"ticker": h["ticker"], "weight": float(h["weight"]), "allocation_gbp": float(h.get("allocation_gbp", 0.0))}
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
st.sidebar.subheader("Investor Profile")
persona_input = st.sidebar.text_area("Persona", active_persona, height=80)
goal_input = st.sidebar.text_input("Stated Goal", active_goal)
limit_input = st.sidebar.slider(
    "Single-Stock Risk Limit (%)",
    min_value=5,
    max_value=50,
    value=int(active_limit * 100),
    step=1,
) / 100.0

st.sidebar.markdown("---")
st.sidebar.subheader("Portfolio Allocations")

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

confirm_button = st.sidebar.button("Run Portfolio X-Ray", type="primary", use_container_width=True)

# Main Dashboard Layout
st.title("Portfolio X-Ray: True Exposure & Risk Analysis")
st.write(
    "*“Most investors know what they own. Far fewer understand what they are actually exposed to.”*"
)

# Convert edited data to holdings list
raw_holdings = edited_df.to_dict("records")

# Normalize weights if needed
total_w = sum(h.get("weight", 0.0) for h in raw_holdings)
if abs(total_w - 1.0) > 1e-4:
    st.sidebar.warning(f"Weights sum to {total_w:.4f}. Normalising to 1.0.")
    for h in raw_holdings:
        h["weight"] = h.get("weight", 0.0) / total_w if total_w > 0 else 0.0

# Execution Pipeline
with st.spinner("Executing quantitative engines (Look-Through, Concentration, 3-Year Risk)..."):
    try:
        lt_result = lt_engine.analyze(raw_holdings)
        conc_analysis = conc_engine.analyze(lt_result)
        risk_metrics = risk_engine.analyze(raw_holdings)
        safety_result = safety_engine.evaluate_portfolio(
            lt_result=lt_result,
            concentration=conc_analysis,
            user_max_single_stock_limit=limit_input,
            persona_name=persona_input,
        )
    except Exception as e:
        st.error(f"Error in quantitative calculation: {str(e)}")
        st.stop()

# 1. Circuit Breaker Header Banner
st.markdown("### Safety Circuit Breakers")
primary_eval = safety_result.evaluations[0] if safety_result.evaluations else None

if primary_eval:
    if primary_eval.status == "BREACH":
        st.error(
            f"🚨 **CIRCUIT BREAKER TRIGGERED: {primary_eval.rule_name}**\n\n"
            f"{primary_eval.message}\n\n"
            f"*(Traceable Proof: `{primary_eval.evidence_id}`)*"
        )
    else:
        st.success(
            f"✅ **POLICY COMPLIANT: {primary_eval.rule_name}**\n\n"
            f"{primary_eval.message}\n\n"
            f"*(Traceable Proof: `{primary_eval.evidence_id}`)*"
        )

# 2. Executive Metrics Row
st.markdown("### Key Portfolio Indicators")
col1, col2, col3, col4 = st.columns(4)

with col1:
    top_stock = conc_analysis.lookthrough.top_issuers[0] if conc_analysis.lookthrough.top_issuers else None
    stock_name = top_stock["issuer"] if top_stock else "N/A"
    stock_w = (top_stock["weight"] * 100) if top_stock else 0.0
    st.metric(
        label="Top Look-Through Issuer",
        value=f"{stock_name} ({stock_w:.1f}%)",
        delta=f"{(stock_w - limit_input*100):+.1f}% vs Limit" if stock_w > limit_input*100 else "Within limit",
        delta_color="inverse",
    )

with col2:
    st.metric(
        label="Effective Holdings (N_eff)",
        value=f"{conc_analysis.lookthrough.effective_n:.1f}",
        delta=f"{conc_analysis.lookthrough.effective_n - conc_analysis.direct.effective_n:+.1f} vs Direct ({conc_analysis.direct.effective_n:.1f})",
        help="N_eff = 1 / HHI. Measures effective independent holdings.",
    )

with col3:
    st.metric(
        label="3-Year Volatility (Annualised)",
        value=f"{risk_metrics.annualized_volatility * 100:.1f}%",
        help="Sample covariance annualised volatility.",
    )

with col4:
    st.metric(
        label="1-Day 95% Historical VaR",
        value=f"{risk_metrics.historical_var_95 * 100:.2f}%",
        delta=f"Max Drawdown: {risk_metrics.max_drawdown * 100:.1f}%",
        delta_color="off",
        help="Empirical 5th percentile daily loss.",
    )

# 3. Look-Through Exposure vs Surface Table
st.markdown("---")
col_table, col_pie = st.columns([3, 2])

with col_table:
    st.subheader("Look-Through Issuer Decomposition")
    
    table_rows = []
    for exp in lt_result.sorted_exposures(exclude_residual_and_cash=False):
        sources_str = ", ".join([f"{k}: {v*100:.1f}%" for k, v in exp.breakdown_by_source.items()])
        table_rows.append({
            "Issuer": exp.issuer,
            "Total Weight": f"{exp.total_weight * 100:.2f}%",
            "Direct Weight": f"{exp.direct_weight * 100:.2f}%",
            "Indirect Weight": f"{exp.indirect_weight * 100:.2f}%",
            "Decomposition Sources": sources_str,
        })
    st.dataframe(pd.DataFrame(table_rows), use_container_width=True, height=320)

with col_pie:
    st.subheader("Top Underlying Concentrations")
    top_chart_data = []
    for exp in lt_result.sorted_exposures(exclude_residual_and_cash=False)[:6]:
        top_chart_data.append({"Issuer": exp.issuer, "Weight": exp.total_weight})
    
    chart_df = pd.DataFrame(top_chart_data)
    st.bar_chart(chart_df.set_index("Issuer"))

# 4. AI Risk Investigator Findings
st.markdown("---")
st.subheader("AI Risk Investigator Analysis")
st.caption("Autonomously reasoned by Gemini 3.8 Flash using pre-computed quantitative evidence.")

# Run investigator agent
with st.spinner("AI Investigator analyzing evidence registry..."):
    investigation_report = investigator.investigate(
        user_goal=goal_input,
        persona_name=persona_input,
        lt_result=lt_result,
        concentration=conc_analysis,
        risk=risk_metrics,
        safety=safety_result,
    )

for idx, finding in enumerate(investigation_report.findings):
    icon = "🚨" if finding.severity == "CRITICAL" else ("⚠️" if finding.severity == "WARNING" else "ℹ️")
    with st.expander(f"{icon} **[{finding.severity}] {finding.title}**", expanded=(idx == 0)):
        st.write(finding.explanation)
        
        # Display expandable mathematical evidence proofs
        st.markdown("**Quantitative Evidence Citations:**")
        for cid in finding.evidence_ids:
            matching_ev = next((e for e in investigation_report.raw_evidence if e.id == cid), None)
            if matching_ev:
                st.code(
                    f"Evidence ID: {matching_ev.id}\n"
                    f"Metric      : {matching_ev.metric}\n"
                    f"Value       : {matching_ev.value}\n"
                    f"Formula     : {matching_ev.formula}\n"
                    f"Data Source : {', '.join(matching_ev.data_sources)}\n"
                    f"Caveats     : {', '.join(matching_ev.caveats) if matching_ev.caveats else 'None'}",
                    language="yaml",
                )

# 5. Full Mathematical Proof Registry (Footer)
with st.expander("🔍 View Complete Evidence Registry (Jane Street Audit Standard)"):
    ev_dicts = [e.to_dict() for e in investigation_report.raw_evidence]
    st.json(ev_dicts)
