"""Asset metadata, ETF constituent look-through tables, and scenario shock models.

All data is bundled for offline, deterministic risk analysis.
"""

from __future__ import annotations
from typing import Dict, List, Any

# ETF constituent look-through weights (Top holdings normalized)
# Real-world approximate top holdings breakdown
ETF_LOOKTHROUGH: Dict[str, Dict[str, float]] = {
    "QQQ": {
        "NVDA": 0.086,
        "AAPL": 0.088,
        "MSFT": 0.081,
        "AMZN": 0.052,
        "META": 0.049,
        "AVGO": 0.046,
        "GOOGL": 0.048,
        "TSLA": 0.029,
        "COST": 0.025,
        "AMD": 0.021,
        "_OTHER_QQQ": 0.475,
    },
    "SPY": {
        "MSFT": 0.068,
        "AAPL": 0.065,
        "NVDA": 0.062,
        "AMZN": 0.038,
        "META": 0.025,
        "GOOGL": 0.024,
        "BRK.B": 0.018,
        "TSLA": 0.016,
        "JPM": 0.014,
        "LLY": 0.013,
        "_OTHER_SPY": 0.657,
    },
    "VOO": {
        "MSFT": 0.068,
        "AAPL": 0.065,
        "NVDA": 0.062,
        "AMZN": 0.038,
        "META": 0.025,
        "GOOGL": 0.024,
        "BRK.B": 0.018,
        "TSLA": 0.016,
        "JPM": 0.014,
        "LLY": 0.013,
        "_OTHER_VOO": 0.657,
    },
    "SMH": {
        "NVDA": 0.208,
        "TSM": 0.125,
        "AVGO": 0.076,
        "ASML": 0.049,
        "AMD": 0.045,
        "QCOM": 0.043,
        "TXN": 0.041,
        "AMAT": 0.039,
        "LRCX": 0.036,
        "MU": 0.032,
        "_OTHER_SMH": 0.306,
    },
    "XLK": {
        "AAPL": 0.155,
        "NVDA": 0.152,
        "MSFT": 0.148,
        "AVGO": 0.052,
        "CRM": 0.032,
        "AMD": 0.028,
        "ADBE": 0.026,
        "ACN": 0.024,
        "CSCO": 0.023,
        "INTC": 0.020,
        "_OTHER_XLK": 0.340,
    },
    "XLE": {
        "XOM": 0.224,
        "CVX": 0.148,
        "COP": 0.082,
        "EOG": 0.048,
        "SLB": 0.046,
        "MPC": 0.044,
        "PSX": 0.039,
        "VLO": 0.034,
        "_OTHER_XLE": 0.335,
    },
    "XLF": {
        "BRK.B": 0.128,
        "JPM": 0.098,
        "V": 0.082,
        "MA": 0.072,
        "BAC": 0.048,
        "WFC": 0.038,
        "GS": 0.032,
        "MS": 0.028,
        "_OTHER_XLF": 0.474,
    },
    "TLT": {
        "US_TREASURY_20Y": 1.0,
    },
    "BND": {
        "US_TREASURY_INTERMEDIATE": 0.65,
        "US_CORP_BOND": 0.25,
        "US_MORTGAGE_BOND": 0.10,
    },
    "GLD": {
        "PHYSICAL_GOLD": 1.0,
    },
}

# Asset Master Catalog: Sector, Asset Class, Annualized Volatility, and Thematic Exposures
ASSET_METADATA: Dict[str, Dict[str, Any]] = {
    "NVDA": {
        "name": "NVIDIA Corporation",
        "asset_class": "Equity",
        "sector": "Information Technology",
        "industry": "Semiconductors",
        "annual_volatility": 0.48,  # 48% annualized volatility
        "themes": ["AI Infrastructure & Capex", "Semiconductor Supply Chain", "High Growth Tech"],
        "rate_sensitivity": -0.65,  # Moderate-high multiple compression under rate hikes
    },
    "MSFT": {
        "name": "Microsoft Corporation",
        "asset_class": "Equity",
        "sector": "Information Technology",
        "industry": "Systems Software & Cloud",
        "annual_volatility": 0.24,
        "themes": ["AI Infrastructure & Capex", "Cloud Computing", "Mega-Cap Tech"],
        "rate_sensitivity": -0.35,
    },
    "AAPL": {
        "name": "Apple Inc.",
        "asset_class": "Equity",
        "sector": "Information Technology",
        "industry": "Technology Hardware",
        "annual_volatility": 0.22,
        "themes": ["Consumer Hardware & Ecosystem", "Mega-Cap Tech"],
        "rate_sensitivity": -0.30,
    },
    "AMZN": {
        "name": "Amazon.com Inc.",
        "asset_class": "Equity",
        "sector": "Consumer Discretionary",
        "industry": "Broadline Retail & Cloud",
        "annual_volatility": 0.29,
        "themes": ["Cloud Computing", "AI Infrastructure & Capex", "Consumer Discretionary"],
        "rate_sensitivity": -0.45,
    },
    "GOOGL": {
        "name": "Alphabet Inc.",
        "asset_class": "Equity",
        "sector": "Communication Services",
        "industry": "Interactive Media & AI",
        "annual_volatility": 0.26,
        "themes": ["AI Infrastructure & Capex", "Digital Advertising", "Mega-Cap Tech"],
        "rate_sensitivity": -0.35,
    },
    "META": {
        "name": "Meta Platforms Inc.",
        "asset_class": "Equity",
        "sector": "Communication Services",
        "industry": "Social Media & AI",
        "annual_volatility": 0.34,
        "themes": ["AI Infrastructure & Capex", "Digital Advertising", "High Growth Tech"],
        "rate_sensitivity": -0.45,
    },
    "TSLA": {
        "name": "Tesla Inc.",
        "asset_class": "Equity",
        "sector": "Consumer Discretionary",
        "industry": "Automotive & Energy Storage",
        "annual_volatility": 0.52,
        "themes": ["EV & Clean Tech", "High Growth Tech", "Autonomous Systems"],
        "rate_sensitivity": -0.75,
    },
    "AVGO": {
        "name": "Broadcom Inc.",
        "asset_class": "Equity",
        "sector": "Information Technology",
        "industry": "Semiconductors & Software",
        "annual_volatility": 0.38,
        "themes": ["AI Infrastructure & Capex", "Semiconductor Supply Chain"],
        "rate_sensitivity": -0.50,
    },
    "TSM": {
        "name": "Taiwan Semiconductor Manufacturing Co.",
        "asset_class": "Equity",
        "sector": "Information Technology",
        "industry": "Semiconductor Foundry",
        "annual_volatility": 0.36,
        "themes": ["Semiconductor Supply Chain", "AI Infrastructure & Capex", "Geopolitical Taiwan Risk"],
        "rate_sensitivity": -0.40,
    },
    "ASML": {
        "name": "ASML Holding N.V.",
        "asset_class": "Equity",
        "sector": "Information Technology",
        "industry": "Semiconductor Equipment",
        "annual_volatility": 0.33,
        "themes": ["Semiconductor Supply Chain", "AI Infrastructure & Capex", "Geopolitical Trade Controls"],
        "rate_sensitivity": -0.45,
    },
    "AMD": {
        "name": "Advanced Micro Devices Inc.",
        "asset_class": "Equity",
        "sector": "Information Technology",
        "industry": "Semiconductors",
        "annual_volatility": 0.44,
        "themes": ["AI Infrastructure & Capex", "Semiconductor Supply Chain"],
        "rate_sensitivity": -0.60,
    },
    "CEG": {
        "name": "Constellation Energy Corp.",
        "asset_class": "Equity",
        "sector": "Utilities",
        "industry": "Nuclear Power & Clean Energy",
        "annual_volatility": 0.35,
        "themes": ["AI Infrastructure & Capex", "Nuclear Energy & Power Demand"],
        "rate_sensitivity": -0.25,
    },
    "JPM": {
        "name": "JPMorgan Chase & Co.",
        "asset_class": "Equity",
        "sector": "Financials",
        "industry": "Diversified Banking",
        "annual_volatility": 0.20,
        "themes": ["Banking & Credit", "Interest Rate Sensitive (Yield Curve)"],
        "rate_sensitivity": 0.15,  # Banks often gain net interest income on modest rate rises
    },
    "LLY": {
        "name": "Eli Lilly and Company",
        "asset_class": "Equity",
        "sector": "Health Care",
        "industry": "Pharmaceuticals",
        "annual_volatility": 0.25,
        "themes": ["Healthcare & GLP-1 Therapeutics", "Defensive Growth"],
        "rate_sensitivity": -0.20,
    },
    "XOM": {
        "name": "Exxon Mobil Corp.",
        "asset_class": "Equity",
        "sector": "Energy",
        "industry": "Integrated Oil & Gas",
        "annual_volatility": 0.23,
        "themes": ["Energy & Commodities", "Inflation Hedge"],
        "rate_sensitivity": 0.20,
    },
    "COST": {
        "name": "Costco Wholesale Corp.",
        "asset_class": "Equity",
        "sector": "Consumer Staples",
        "industry": "Consumer Staples Merchandise",
        "annual_volatility": 0.18,
        "themes": ["Defensive Consumer", "Inflation Resilient"],
        "rate_sensitivity": -0.15,
    },
    "US_TREASURY_20Y": {
        "name": "US Treasury 20+ Year Bond",
        "asset_class": "Fixed Income",
        "sector": "Government Debt",
        "industry": "Sovereign Bonds",
        "annual_volatility": 0.16,
        "themes": ["Defensive Safe Haven", "Interest Rate Duration"],
        "rate_sensitivity": -1.80,  # High duration vulnerability to rate rises
    },
    "US_TREASURY_INTERMEDIATE": {
        "name": "US Treasury Intermediate Bond",
        "asset_class": "Fixed Income",
        "sector": "Government Debt",
        "industry": "Sovereign Bonds",
        "annual_volatility": 0.08,
        "themes": ["Defensive Safe Haven", "Moderate Duration"],
        "rate_sensitivity": -0.65,
    },
    "US_CORP_BOND": {
        "name": "US Investment Grade Corporate Bond",
        "asset_class": "Fixed Income",
        "sector": "Corporate Debt",
        "industry": "Fixed Income Credit",
        "annual_volatility": 0.09,
        "themes": ["Credit Spread Exposure", "Moderate Duration"],
        "rate_sensitivity": -0.70,
    },
    "US_MORTGAGE_BOND": {
        "name": "US Mortgage-Backed Securities",
        "asset_class": "Fixed Income",
        "sector": "Securitized Debt",
        "industry": "Mortgage Backed",
        "annual_volatility": 0.07,
        "themes": ["Mortgage Rate Prepayment", "Moderate Duration"],
        "rate_sensitivity": -0.60,
    },
    "PHYSICAL_GOLD": {
        "name": "Physical Gold Bullion",
        "asset_class": "Commodity",
        "sector": "Precious Metals",
        "industry": "Hard Assets",
        "annual_volatility": 0.15,
        "themes": ["Inflation Hedge", "Geopolitical Safe Haven"],
        "rate_sensitivity": -0.30,
    },
    "CASH": {
        "name": "USD Cash / Money Market",
        "asset_class": "Cash",
        "sector": "Cash Equivalents",
        "industry": "Cash",
        "annual_volatility": 0.005,
        "themes": ["Capital Preservation", "Liquidity"],
        "rate_sensitivity": 0.10,
    },
}

# Fallback generator for unknown stocks or diversified ETF baskets
def get_asset_info(ticker: str) -> Dict[str, Any]:
    ticker = ticker.upper().strip()
    if ticker in ASSET_METADATA:
        return ASSET_METADATA[ticker]
    if ticker.startswith("_OTHER_"):
        etf_name = ticker.replace("_OTHER_", "")
        return {
            "name": f"Remaining constituents of {etf_name}",
            "asset_class": "Equity",
            "sector": "Diversified Index Blend",
            "industry": "Diversified Holdings",
            "annual_volatility": 0.18,
            "themes": ["Broad Market Exposure"],
            "rate_sensitivity": -0.30,
        }
    # Generic fallback
    return {
        "name": f"{ticker} Common Stock",
        "asset_class": "Equity",
        "sector": "General Market",
        "industry": "Public Company",
        "annual_volatility": 0.30,
        "themes": ["Broad Equity Market"],
        "rate_sensitivity": -0.40,
    }

# Historical and Macro Stress Test Scenarios
STRESS_SCENARIOS = [
    {
        "id": "rate_shock_2022",
        "name": "2022 Fed Rate Shock & Tech Multiple Compression",
        "description": "Rapid interest rate increases caused high-multiple growth equities and long-duration bonds to collapse, while energy and commodities surged.",
        "shocks": {
            "High Growth Tech": -0.36,
            "AI Infrastructure & Capex": -0.32,
            "Semiconductor Supply Chain": -0.34,
            "Interest Rate Duration": -0.31,
            "Consumer Discretionary": -0.25,
            "Broad Market Exposure": -0.19,
            "Banking & Credit": -0.11,
            "Defensive Consumer": -0.05,
            "Healthcare & GLP-1 Therapeutics": -0.04,
            "Energy & Commodities": 0.45,
            "Capital Preservation": 0.015,
        },
        "default_equity_shock": -0.18,
    },
    {
        "id": "covid_liquidity_shock_2020",
        "name": "2020 Liquidity & Pandemic Selloff",
        "description": "Sudden, synchronized global market liquidation across equities, corporate credit, and commodities, with immediate flight to cash and sovereign Treasuries.",
        "shocks": {
            "High Growth Tech": -0.22,
            "AI Infrastructure & Capex": -0.20,
            "Semiconductor Supply Chain": -0.26,
            "Energy & Commodities": -0.52,
            "Banking & Credit": -0.38,
            "Consumer Discretionary": -0.31,
            "Defensive Safe Haven": 0.12,
            "Broad Market Exposure": -0.34,
            "Defensive Consumer": -0.12,
            "Capital Preservation": 0.0,
        },
        "default_equity_shock": -0.32,
    },
    {
        "id": "ai_capex_pause_2024",
        "name": "AI Infrastructure Capex Reality Check",
        "description": "Hypothetical scenario where hyperscaler cloud giants scale back AI data center spending, triggering a severe sharp derating in semiconductors and hardware.",
        "shocks": {
            "AI Infrastructure & Capex": -0.28,
            "Semiconductor Supply Chain": -0.33,
            "High Growth Tech": -0.20,
            "Nuclear Energy & Power Demand": -0.22,
            "Cloud Computing": -0.14,
            "Defensive Consumer": 0.04,
            "Banking & Credit": 0.02,
            "Energy & Commodities": -0.03,
            "Defensive Safe Haven": 0.05,
            "Capital Preservation": 0.0,
        },
        "default_equity_shock": -0.06,
    },
    {
        "id": "energy_stagflation_spike",
        "name": "Stagflation Energy & Supply Chain Shock",
        "description": "Middle-East oil embargo / geopolitical disruption driving crude oil over $120/bbl alongside persistent inflation and consumer margin compression.",
        "shocks": {
            "Energy & Commodities": 0.35,
            "Inflation Hedge": 0.18,
            "Consumer Discretionary": -0.22,
            "Semiconductor Supply Chain": -0.15,
            "High Growth Tech": -0.18,
            "Interest Rate Duration": -0.12,
            "Defensive Consumer": -0.06,
            "Capital Preservation": 0.01,
        },
        "default_equity_shock": -0.12,
    },
]

# Sample preloaded portfolios for instant testing
SAMPLE_PORTFOLIOS = {
    "accidental_tech_overweight": {
        "name": "The 'Accidental' Tech Concentration (Overlapping Risk)",
        "description": "Investor holds Nvidia plus QQQ, VOO, and Semiconductor ETF, believing they are broadly diversified across 4 different tickers.",
        "user_profile": {
            "risk_tolerance": "Moderate",
            "horizon_years": 7,
            "loss_capacity": "Moderate (Max 15-20% drawdown tolerated)",
        },
        "holdings": [
            {"ticker": "NVDA", "shares": 50, "price": 130.0},     # $6,500 (~20.0%)
            {"ticker": "QQQ", "shares": 25, "price": 480.0},      # $12,000 (~36.9%)
            {"ticker": "VOO", "shares": 15, "price": 500.0},      # $7,500 (~23.1%)
            {"ticker": "SMH", "shares": 26, "price": 250.0},      # $6,500 (~20.0%)
        ],
    },
    "balanced_core": {
        "name": "Institutional Balanced 60/40 Portfolio",
        "description": "Traditional core equity, intermediate and long-term treasuries, and gold buffer.",
        "user_profile": {
            "risk_tolerance": "Moderate",
            "horizon_years": 10,
            "loss_capacity": "Moderate (Max 12-15% drawdown tolerated)",
        },
        "holdings": [
            {"ticker": "SPY", "shares": 40, "price": 550.0},      # $22,000 (55.0%)
            {"ticker": "BND", "shares": 150, "price": 72.0},      # $10,800 (27.0%)
            {"ticker": "TLT", "shares": 40, "price": 95.0},       # $3,800 (9.5%)
            {"ticker": "GLD", "shares": 15, "price": 230.0},      # $3,450 (8.5%)
        ],
    },
    "all_in_growth": {
        "name": "Aggressive Tech & AI Moonshot",
        "description": "Heavy direct concentration in high-beta AI names and leveraged semiconductor supply chain.",
        "user_profile": {
            "risk_tolerance": "Aggressive",
            "horizon_years": 5,
            "loss_capacity": "High (can withstand 30%+ drawdown)",
        },
        "holdings": [
            {"ticker": "NVDA", "shares": 80, "price": 130.0},     # $10,400 (41.6%)
            {"ticker": "TSLA", "shares": 25, "price": 220.0},     # $5,500 (22.0%)
            {"ticker": "AMD", "shares": 30, "price": 140.0},      # $4,200 (16.8%)
            {"ticker": "CEG", "shares": 20, "price": 245.0},      # $4,900 (19.6%)
        ],
    },
}
