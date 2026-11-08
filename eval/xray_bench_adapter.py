from __future__ import annotations

import json
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from xray.data_loader import DataLoader
from xray.engine.concentration import ConcentrationEngine
from xray.engine.lookthrough import LookThroughEngine
from xray.safety.limits import SafetyLimitsEngine


def main() -> None:
    """
    AgentVerse finance/bench.py streaming JSONL adapter for Portfolio X-Ray.
    Calls the exact deterministic engines used by the application (LookThroughEngine,
    ConcentrationEngine, SafetyLimitsEngine).
    Reads one JSON request per line from stdin and flushes one JSON response per line to stdout.
    """
    loader = DataLoader()
    lt_engine = LookThroughEngine(loader)
    conc_engine = ConcentrationEngine()
    safety_engine = SafetyLimitsEngine()

    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue
        req = json.loads(line)
        req_id = req.get("request_id", req.get("case_id", "unknown"))
        inp = req.get("input", {})

        holdings = inp.get("holdings", [])
        limit = float(inp.get("max_risk_tolerance_single_stock", 0.20))

        try:
            lt_res = lt_engine.analyze(holdings)
            conc_res = conc_engine.analyze(lt_res)
            safety_res = safety_engine.evaluate_portfolio(
                lt_result=lt_res,
                concentration=conc_res,
                user_max_single_stock_limit=limit,
            )

            top_named = lt_res.sorted_exposures(exclude_residual_and_cash=True)
            top_issuer = top_named[0].issuer if top_named else "NONE"

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
            hidden_breaches = sorted(list(set(lookthrough_breaches) - set(surface_breaches)))

            resp = {
                "request_id": req_id,
                "output": {
                    "top_issuer": top_issuer,
                    "circuit_breaker_status": safety_res.evaluations[0].status,
                    "lookthrough_breaches": lookthrough_breaches,
                    "hidden_breaches": hidden_breaches,
                },
            }
        except Exception as exc:
            print(f"Adapter error on {req_id}: {exc}", file=sys.stderr)
            resp = {"request_id": req_id, "output": {"error": "invalid_input"}}

        sys.stdout.write(json.dumps(resp) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
