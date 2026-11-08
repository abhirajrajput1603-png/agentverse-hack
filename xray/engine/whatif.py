from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from xray.data_loader import DataLoader
from xray.engine.concentration import ConcentrationAnalysis, ConcentrationEngine
from xray.engine.lookthrough import LookThroughEngine, LookThroughResult
from xray.engine.risk import RiskEngine, RiskMetrics
from xray.evidence import Evidence
from xray.safety.audit import AuditLogger
from xray.safety.limits import SafetyCheckResult, SafetyLimitsEngine


CONFIG_DIR = Path(__file__).resolve().parent.parent.parent / "config"
FRICTION_CONFIG_PATH = CONFIG_DIR / "friction.json"


@dataclass
class TransactionFriction:
    ticker: str
    trade_type: str                   # "BUY", "SELL", "REALLOCATE"
    trade_value_gbp: float
    spread_cost_gbp: float
    commission_gbp: float
    slippage_gbp: float
    total_friction_gbp: float
    friction_bps: float


@dataclass
class WhatIfComparison:
    scenario_description: str
    trade_frictions: List[TransactionFriction]
    total_friction_gbp: float
    before_portfolio_value_gbp: float
    after_portfolio_value_gbp: float
    
    # Quantitative states
    before_lt: LookThroughResult
    after_lt: LookThroughResult
    before_concentration: ConcentrationAnalysis
    after_concentration: ConcentrationAnalysis
    before_risk: RiskMetrics
    after_risk: RiskMetrics
    
    # Safety & circuit breaker evaluation on new portfolio
    after_safety: SafetyCheckResult
    evidence_list: List[Evidence] = field(default_factory=list)


class WhatIfSimulator:
    """
    Deterministic What-If Scenario Simulator with realistic transaction friction.
    Simulates:
      - Adding new cash to a ticker
      - Reallocating from Ticker A to Ticker B
      - Trimming / selling positions
    Deducts realistic market frictions:
      - Bid-ask spread (bps)
      - Fixed ticket commissions (£)
      - Non-linear quadratic market impact / slippage growing with trade size
    Enforces strict non-negotiable rule:
      Re-evaluates circuit breaker on the simulated portfolio.
      On BREACH: halts, writes to audit log, and explains the violation.
      Zero autonomous trade execution.
    """

    def __init__(
        self,
        data_loader: Optional[DataLoader] = None,
        friction_config_path: Optional[Path] = None,
        audit_logger: Optional[AuditLogger] = None,
    ) -> None:
        self.loader = data_loader or DataLoader()
        self.friction_config_path = friction_config_path or FRICTION_CONFIG_PATH
        self.friction_config = self._load_friction_config()
        self.audit_logger = audit_logger or AuditLogger()

        self.lt_engine = LookThroughEngine(self.loader)
        self.conc_engine = ConcentrationEngine()
        self.risk_engine = RiskEngine(self.loader)
        self.safety_engine = SafetyLimitsEngine(audit_logger=self.audit_logger)

    def _load_friction_config(self) -> Dict[str, float]:
        if not self.friction_config_path.exists():
            return {
                "stock_bid_ask_spread_bps": 5.0,
                "etf_bid_ask_spread_bps": 2.0,
                "fixed_commission_gbp": 4.99,
                "quadratic_slippage_coefficient": 0.0000005,
                "annual_etf_expense_drag_bps": 20.0,
            }
        with open(self.friction_config_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def compute_friction(self, ticker: str, trade_type: str, trade_value_gbp: float) -> TransactionFriction:
        """
        Calculates exact deterministic trading friction:
        Friction = Spread Cost + Fixed Commission + Quadratic Slippage
        """
        if trade_value_gbp <= 0.0:
            return TransactionFriction(
                ticker=ticker,
                trade_type=trade_type,
                trade_value_gbp=0.0,
                spread_cost_gbp=0.0,
                commission_gbp=0.0,
                slippage_gbp=0.0,
                total_friction_gbp=0.0,
                friction_bps=0.0,
            )

        # Spread bps: ETFs have tighter spreads than individual equities
        is_etf = ticker in self.loader.load_etf_constituents()
        spread_bps = (
            self.friction_config.get("etf_bid_ask_spread_bps", 2.0)
            if is_etf
            else self.friction_config.get("stock_bid_ask_spread_bps", 5.0)
        )
        spread_cost = trade_value_gbp * (spread_bps / 10000.0)

        # Fixed ticket charge
        commission = self.friction_config.get("fixed_commission_gbp", 4.99)

        # Quadratic market impact: Slippage = coeff * trade_value^2
        coeff = self.friction_config.get("quadratic_slippage_coefficient", 0.0000005)
        slippage = coeff * (trade_value_gbp ** 2)

        total_friction = spread_cost + commission + slippage
        friction_bps = (total_friction / trade_value_gbp) * 10000.0

        return TransactionFriction(
            ticker=ticker,
            trade_type=trade_type,
            trade_value_gbp=round(trade_value_gbp, 2),
            spread_cost_gbp=round(spread_cost, 2),
            commission_gbp=round(commission, 2),
            slippage_gbp=round(slippage, 2),
            total_friction_gbp=round(total_friction, 2),
            friction_bps=round(friction_bps, 2),
        )

    def simulate_add_cash(
        self,
        current_holdings: List[Dict[str, Any]],
        target_ticker: str,
        additional_cash_gbp: float,
        persona_name: str,
        user_max_single_stock_limit: Optional[float] = None,
    ) -> WhatIfComparison:
        """
        Simulates deploying new cash into a specified ticker.
        Deducts friction and recomputes all mathematical dimensions.
        """
        evidence_list: List[Evidence] = []
        
        # 1. Base Portfolio
        before_lt = self.lt_engine.analyze(current_holdings)
        before_conc = self.conc_engine.analyze(before_lt)
        before_risk = self.risk_engine.analyze(current_holdings)
        
        # Total base portfolio value
        base_gbp = sum(float(h.get("allocation_gbp", 0.0)) for h in current_holdings)
        if base_gbp == 0.0:
            # Assume £10,000 nominal if unstated
            base_gbp = 10000.0
            for h in current_holdings:
                h["allocation_gbp"] = float(h.get("weight", 0.0)) * base_gbp

        # 2. Compute friction on new purchase
        friction = self.compute_friction(target_ticker, "BUY", additional_cash_gbp)
        net_cash_invested = additional_cash_gbp - friction.total_friction_gbp

        # 3. Formulate new allocation
        new_holdings_dict: Dict[str, float] = {
            h["ticker"]: float(h.get("allocation_gbp", 0.0)) for h in current_holdings
        }
        new_holdings_dict[target_ticker] = new_holdings_dict.get(target_ticker, 0.0) + net_cash_invested

        after_total_gbp = sum(new_holdings_dict.values())
        new_holdings = [
            {
                "ticker": t,
                "allocation_gbp": round(amt, 2),
                "weight": amt / after_total_gbp,
            }
            for t, amt in new_holdings_dict.items()
        ]

        # 4. Recompute quantitative engines on simulated portfolio
        after_lt = self.lt_engine.analyze(new_holdings)
        after_conc = self.conc_engine.analyze(after_lt)
        after_risk = self.risk_engine.analyze(new_holdings)

        # 5. Safety Circuit Breaker Evaluation on Simulated Portfolio
        after_safety = self.safety_engine.evaluate_portfolio(
            lt_result=after_lt,
            concentration=after_conc,
            user_max_single_stock_limit=user_max_single_stock_limit,
            persona_name=f"{persona_name} (What-If Simulation)",
        )

        scenario_desc = (
            f"Add £{additional_cash_gbp:,.2f} new cash into {target_ticker} "
            f"(Net: £{net_cash_invested:,.2f} after £{friction.total_friction_gbp:.2f} friction)"
        )

        # Audit log what-if simulation
        self.audit_logger.log_event(
            event_type="WHAT_IF_SCENARIO_SIMULATION",
            details={
                "scenario": scenario_desc,
                "persona": persona_name,
                "target_ticker": target_ticker,
                "cash_added_gbp": additional_cash_gbp,
                "friction_gbp": friction.total_friction_gbp,
                "after_circuit_breaker_safe": after_safety.is_safe,
            },
        )

        # Evidence of scenario friction
        ev_friction = Evidence(
            id=f"EV-WHATIF-FRICTION-{target_ticker}",
            metric="transaction_friction_deduction",
            value=friction.total_friction_gbp,
            formula="spread_cost + fixed_commission + quadratic_slippage",
            inputs={
                "trade_value_gbp": additional_cash_gbp,
                "spread_bps": friction.friction_bps,
                "commission_gbp": friction.commission_gbp,
                "slippage_gbp": friction.slippage_gbp,
            },
            data_sources=["config/friction.json"],
            sample_size=None,
            caveats=["Theoretical institutional execution model."],
        )
        evidence_list.append(ev_friction)

        return WhatIfComparison(
            scenario_description=scenario_desc,
            trade_frictions=[friction],
            total_friction_gbp=friction.total_friction_gbp,
            before_portfolio_value_gbp=base_gbp,
            after_portfolio_value_gbp=after_total_gbp,
            before_lt=before_lt,
            after_lt=after_lt,
            before_concentration=before_conc,
            after_concentration=after_conc,
            before_risk=before_risk,
            after_risk=after_risk,
            after_safety=after_safety,
            evidence_list=evidence_list,
        )

    def simulate_rebalance(
        self,
        current_holdings: List[Dict[str, Any]],
        sell_ticker: str,
        sell_gbp: float,
        buy_ticker: str,
        persona_name: str,
        user_max_single_stock_limit: Optional[float] = None,
    ) -> WhatIfComparison:
        """
        Simulates selling a position and rotating into another ticker.
        Deducts two-legged friction (sell friction + buy friction) and evaluates circuit breakers.
        """
        evidence_list: List[Evidence] = []

        before_lt = self.lt_engine.analyze(current_holdings)
        before_conc = self.conc_engine.analyze(before_lt)
        before_risk = self.risk_engine.analyze(current_holdings)

        base_gbp = sum(float(h.get("allocation_gbp", 0.0)) for h in current_holdings)
        if base_gbp == 0.0:
            base_gbp = 10000.0
            for h in current_holdings:
                h["allocation_gbp"] = float(h.get("weight", 0.0)) * base_gbp

        # Find current allocation of sell_ticker
        current_holdings_dict: Dict[str, float] = {
            h["ticker"]: float(h.get("allocation_gbp", 0.0)) for h in current_holdings
        }
        current_sell_amt = current_holdings_dict.get(sell_ticker, 0.0)
        if sell_gbp > current_sell_amt:
            raise ValueError(
                f"Cannot sell £{sell_gbp:,.2f} of {sell_ticker}: current holding is £{current_sell_amt:,.2f}"
            )

        # 1. Leg 1: Sell Friction
        sell_friction = self.compute_friction(sell_ticker, "SELL", sell_gbp)
        net_sell_proceeds = sell_gbp - sell_friction.total_friction_gbp

        # 2. Leg 2: Buy Friction
        buy_friction = self.compute_friction(buy_ticker, "BUY", net_sell_proceeds)
        net_buy_invested = net_sell_proceeds - buy_friction.total_friction_gbp

        total_friction = round(sell_friction.total_friction_gbp + buy_friction.total_friction_gbp, 2)

        # 3. Update holdings dict
        new_holdings_dict = dict(current_holdings_dict)
        new_holdings_dict[sell_ticker] -= sell_gbp
        if new_holdings_dict[sell_ticker] <= 1e-4:
            del new_holdings_dict[sell_ticker]
        new_holdings_dict[buy_ticker] = new_holdings_dict.get(buy_ticker, 0.0) + net_buy_invested

        after_total_gbp = sum(new_holdings_dict.values())
        new_holdings = [
            {
                "ticker": t,
                "allocation_gbp": round(amt, 2),
                "weight": amt / after_total_gbp,
            }
            for t, amt in new_holdings_dict.items()
        ]

        after_lt = self.lt_engine.analyze(new_holdings)
        after_conc = self.conc_engine.analyze(after_lt)
        after_risk = self.risk_engine.analyze(new_holdings)

        after_safety = self.safety_engine.evaluate_portfolio(
            lt_result=after_lt,
            concentration=after_conc,
            user_max_single_stock_limit=user_max_single_stock_limit,
            persona_name=f"{persona_name} (Rebalance Simulation)",
        )

        scenario_desc = (
            f"Rotate £{sell_gbp:,.2f} from {sell_ticker} into {buy_ticker} "
            f"(Total two-leg friction: £{total_friction:.2f})"
        )

        self.audit_logger.log_event(
            event_type="WHAT_IF_REBALANCE_SIMULATION",
            details={
                "scenario": scenario_desc,
                "persona": persona_name,
                "sell_ticker": sell_ticker,
                "sell_gbp": sell_gbp,
                "buy_ticker": buy_ticker,
                "total_friction_gbp": total_friction,
                "after_circuit_breaker_safe": after_safety.is_safe,
            },
        )

        return WhatIfComparison(
            scenario_description=scenario_desc,
            trade_frictions=[sell_friction, buy_friction],
            total_friction_gbp=total_friction,
            before_portfolio_value_gbp=base_gbp,
            after_portfolio_value_gbp=after_total_gbp,
            before_lt=before_lt,
            after_lt=after_lt,
            before_concentration=before_conc,
            after_concentration=after_conc,
            before_risk=before_risk,
            after_risk=after_risk,
            after_safety=after_safety,
            evidence_list=evidence_list,
        )
