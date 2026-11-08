"""Portfolio X-Ray: AI-Powered Portfolio Risk Intelligence.

Autonomous risk investigation, look-through exposure decomposition,
and scenario testing for retail and professional investors.
"""

from portfolio_xray.engine import (
    Holding,
    Portfolio,
    EffectiveExposure,
    compute_lookthrough_exposure,
    compute_thematic_exposure,
    compute_risk_metrics,
    run_stress_tests,
    select_stress_scenarios,
    simulate_what_if,
    analyze_headline_event,
)

__version__ = "0.1.0"
