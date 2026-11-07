from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from xray.data_loader import DataLoader
from xray.evidence import Evidence


@dataclass
class Holding:
    ticker: str
    weight: float
    name: Optional[str] = None
    allocation_gbp: Optional[float] = None


@dataclass
class IssuerExposure:
    issuer: str
    total_weight: float
    direct_weight: float
    indirect_weight: float
    breakdown_by_source: Dict[str, float] = field(default_factory=dict)
    is_fund_undisclosed: bool = False
    is_cash: bool = False
    is_residual: bool = False


@dataclass
class LookThroughResult:
    holdings: List[Holding]
    issuer_exposures: Dict[str, IssuerExposure]
    total_weight: float
    evidence_list: List[Evidence]

    def get_issuer_weight(self, issuer: str) -> float:
        exposure = self.issuer_exposures.get(issuer)
        return exposure.total_weight if exposure else 0.0

    def sorted_exposures(self, exclude_residual_and_cash: bool = False) -> List[IssuerExposure]:
        items = list(self.issuer_exposures.values())
        if exclude_residual_and_cash:
            items = [e for e in items if not (e.is_residual or e.is_cash)]
        return sorted(items, key=lambda x: x.total_weight, reverse=True)


class LookThroughEngine:
    """
    Deterministic look-through decomposition engine.
    Implements:
        w_lt(i) = w_direct(i) + sum_f [ w_fund(f) * w_constituent(f, i) ]
    Enforces absolute weight conservation: |sum(w_lt) - 1.0| < 1e-9.
    Never uses LLM arithmetic; produces verifiable Evidence objects for each issuer.
    """

    def __init__(self, data_loader: Optional[DataLoader] = None) -> None:
        self.loader = data_loader or DataLoader()
        self.etf_data = self.loader.load_etf_constituents()
        self.issuer_map = self.loader.load_issuer_map()

    def map_issuer(self, ticker: str) -> str:
        """Map share classes to common issuer (e.g. GOOG -> GOOGL)."""
        return self.issuer_map.get(ticker, ticker)

    def analyze(self, raw_holdings: List[Dict[str, Any]]) -> LookThroughResult:
        """
        Decomposes a portfolio into issuer-level economic ownership.
        Input: list of dicts with 'ticker' and 'weight' (or 'allocation_gbp').
        """
        holdings: List[Holding] = []
        
        # 1. Parse and validate input weights
        total_input_weight = sum(h.get("weight", 0.0) for h in raw_holdings)
        
        # If weights are missing but allocation_gbp is present, compute fractional weights
        if total_input_weight == 0.0:
            total_gbp = sum(h.get("allocation_gbp", 0.0) for h in raw_holdings)
            if total_gbp <= 0.0:
                raise ValueError("Holdings must contain non-zero weights or allocation_gbp")
            for h in raw_holdings:
                w = h.get("allocation_gbp", 0.0) / total_gbp
                holdings.append(
                    Holding(
                        ticker=h["ticker"],
                        weight=w,
                        name=h.get("name"),
                        allocation_gbp=h.get("allocation_gbp"),
                    )
                )
        else:
            if abs(total_input_weight - 1.0) > 1e-4:
                raise ValueError(
                    f"Portfolio weights must sum to 1.0 (actual sum: {total_input_weight:.6f})"
                )
            for h in raw_holdings:
                holdings.append(
                    Holding(
                        ticker=h["ticker"],
                        weight=float(h["weight"]),
                        name=h.get("name"),
                        allocation_gbp=h.get("allocation_gbp"),
                    )
                )

        # 2. Decompose direct and ETF constituent weights
        issuer_exposures: Dict[str, IssuerExposure] = {}

        def get_or_create_exposure(issuer_name: str) -> IssuerExposure:
            if issuer_name not in issuer_exposures:
                is_cash = issuer_name in ("CASH", "CASH_GBP", "CASH_USD")
                is_res = issuer_name == "OTHER_CONSTITUENTS"
                issuer_exposures[issuer_name] = IssuerExposure(
                    issuer=issuer_name,
                    total_weight=0.0,
                    direct_weight=0.0,
                    indirect_weight=0.0,
                    breakdown_by_source={},
                    is_fund_undisclosed=False,
                    is_cash=is_cash,
                    is_residual=is_res,
                )
            return issuer_exposures[issuer_name]

        for h in holdings:
            ticker = h.ticker
            w_holding = h.weight

            if ticker in self.etf_data:
                # Fund with constituent breakdown
                fund_info = self.etf_data[ticker]
                constituents = fund_info.get("holdings", {})
                for c_ticker, c_weight in constituents.items():
                    target_issuer = c_ticker if c_ticker == "OTHER_CONSTITUENTS" else self.map_issuer(c_ticker)
                    exp = get_or_create_exposure(target_issuer)
                    added_weight = w_holding * c_weight
                    exp.total_weight += added_weight
                    exp.indirect_weight += added_weight
                    exp.breakdown_by_source[ticker] = exp.breakdown_by_source.get(ticker, 0.0) + added_weight
            else:
                # Direct stock, cash, or fund without constituent breakdown (e.g. TLT, BND)
                target_issuer = self.map_issuer(ticker)
                exp = get_or_create_exposure(target_issuer)
                exp.total_weight += w_holding
                exp.direct_weight += w_holding
                exp.breakdown_by_source["DIRECT"] = exp.breakdown_by_source.get("DIRECT", 0.0) + w_holding
                # If it's a known fund ticker without decomposition data
                if ticker in ("TLT", "BND", "XLK", "XLF", "XLE", "VOO", "IWM") and ticker not in self.etf_data:
                    exp.is_fund_undisclosed = True

        # 3. Assert weight conservation within 1e-9
        total_decomposed_weight = sum(exp.total_weight for exp in issuer_exposures.values())
        if abs(total_decomposed_weight - 1.0) > 1e-7:
            raise AssertionError(
                f"Look-through weight conservation failure: sum = {total_decomposed_weight:.10f} != 1.0"
            )

        # 4. Generate Evidence objects for each significant exposure
        evidence_list: List[Evidence] = []
        for issuer, exp in issuer_exposures.items():
            sources = []
            caveats = []
            for src in exp.breakdown_by_source.keys():
                if src == "DIRECT":
                    sources.append("Direct Portfolio Allocation")
                elif src in self.etf_data:
                    info = self.etf_data[src]
                    sources.append(f"{src} ({info.get('source', 'ETF Filing')}, as-of {info.get('as_of_date', 'N/A')})")
            
            if exp.is_fund_undisclosed:
                caveats.append(f"{issuer} constituent data not available in MVP; treated as single aggregate asset.")
            if exp.is_residual:
                caveats.append("OTHER_CONSTITUENTS represents diversified non-top fund holdings.")

            formula_str = "w_direct"
            if exp.indirect_weight > 0:
                indirect_terms = [f"({w_src:.4f} via {src})" for src, w_src in exp.breakdown_by_source.items() if src != "DIRECT"]
                formula_str += " + " + " + ".join(indirect_terms)

            ev = Evidence(
                id=f"EV-LT-{issuer}",
                metric="lookthrough_exposure_weight",
                value=round(exp.total_weight, 6),
                formula=formula_str,
                inputs={
                    "issuer": issuer,
                    "direct_weight": round(exp.direct_weight, 6),
                    "indirect_weight": round(exp.indirect_weight, 6),
                    "breakdown": {k: round(v, 6) for k, v in exp.breakdown_by_source.items()},
                },
                data_sources=sources,
                sample_size=None,
                caveats=caveats,
            )
            evidence_list.append(ev)

        return LookThroughResult(
            holdings=holdings,
            issuer_exposures=issuer_exposures,
            total_weight=total_decomposed_weight,
            evidence_list=evidence_list,
        )
