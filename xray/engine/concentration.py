from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from xray.engine.lookthrough import LookThroughResult, IssuerExposure
from xray.evidence import Evidence


@dataclass
class ConcentrationProfile:
    # Portfolio level metrics
    hhi: float
    effective_n: float
    top_1_weight: float
    top_3_weight: float
    top_5_weight: float
    top_10_weight: float
    top_issuers: List[Dict[str, Any]]
    total_issuers_count: int


@dataclass
class ConcentrationAnalysis:
    direct: ConcentrationProfile
    lookthrough: ConcentrationProfile
    evidence_list: List[Evidence]


class ConcentrationEngine:
    """
    Deterministic concentration risk engine.
    Computes:
      - Herfindahl-Hirschman Index (HHI) = sum(w_i^2)
      - Effective number of holdings N_eff = 1 / HHI
      - Top-1, Top-3, Top-5, Top-10 aggregate weights
    Evaluated across:
      1. Direct surface holdings
      2. True underlying look-through holdings
    Adheres to Jane Street math invariants: LLM does zero arithmetic; produces Evidence objects.
    """

    @staticmethod
    def _compute_profile(
        weights_dict: Dict[str, float],
        exclude_residual_and_cash: bool = False,
    ) -> ConcentrationProfile:
        """Computes HHI, N_eff, and top-N metrics for a given dictionary of weights."""
        if not weights_dict:
            return ConcentrationProfile(
                hhi=0.0,
                effective_n=0.0,
                top_1_weight=0.0,
                top_3_weight=0.0,
                top_5_weight=0.0,
                top_10_weight=0.0,
                top_issuers=[],
                total_issuers_count=0,
            )

        # Filter out cash and residual if requested for single-issuer concentration
        filtered_items = []
        for name, w in weights_dict.items():
            if exclude_residual_and_cash and (name == "OTHER_CONSTITUENTS" or name.startswith("CASH")):
                continue
            filtered_items.append((name, w))

        # Sort descending by weight
        sorted_items = sorted(filtered_items, key=lambda x: x[1], reverse=True)

        # HHI across all evaluated weights
        hhi = sum(w**2 for _, w in sorted_items)
        neff = 1.0 / hhi if hhi > 0 else 0.0

        top_1 = sorted_items[0][1] if len(sorted_items) >= 1 else 0.0
        top_3 = sum(w for _, w in sorted_items[:3])
        top_5 = sum(w for _, w in sorted_items[:5])
        top_10 = sum(w for _, w in sorted_items[:10])

        top_issuers = [{"issuer": name, "weight": round(w, 6)} for name, w in sorted_items[:10]]

        return ConcentrationProfile(
            hhi=round(hhi, 6),
            effective_n=round(neff, 2),
            top_1_weight=round(top_1, 6),
            top_3_weight=round(top_3, 6),
            top_5_weight=round(top_5, 6),
            top_10_weight=round(top_10, 6),
            top_issuers=top_issuers,
            total_issuers_count=len(sorted_items),
        )

    def analyze(self, lt_result: LookThroughResult) -> ConcentrationAnalysis:
        """
        Runs concentration analysis comparing direct surface allocations
        against look-through economic exposures.
        """
        evidence_list: List[Evidence] = []

        # 1. Direct profile
        direct_weights = {h.ticker: h.weight for h in lt_result.holdings}
        direct_profile = self._compute_profile(direct_weights, exclude_residual_and_cash=False)

        # Direct Evidence
        evidence_list.append(
            Evidence(
                id="EV-CONC-DIRECT-HHI",
                metric="herfindahl_hirschman_index_direct",
                value=direct_profile.hhi,
                formula="HHI_direct = sum(w_direct^2)",
                inputs={"direct_weights": {k: round(v, 4) for k, v in direct_weights.items()}},
                data_sources=["User Portfolio Allocation"],
                sample_size=len(direct_weights),
                caveats=["Computed on surface tickers prior to look-through ETF unpacking."],
            )
        )
        evidence_list.append(
            Evidence(
                id="EV-CONC-DIRECT-NEFF",
                metric="effective_number_holdings_direct",
                value=direct_profile.effective_n,
                formula="N_eff_direct = 1 / HHI_direct",
                inputs={"hhi_direct": direct_profile.hhi},
                data_sources=["User Portfolio Allocation"],
                sample_size=len(direct_weights),
                caveats=[],
            )
        )

        # 2. Look-through profile (excluding residual and cash for single-issuer exposure metrics)
        lt_weights_all = {k: exp.total_weight for k, exp in lt_result.issuer_exposures.items()}
        lt_profile = self._compute_profile(lt_weights_all, exclude_residual_and_cash=True)

        # Also compute full look-through HHI including residual
        hhi_lt_full = sum(exp.total_weight**2 for exp in lt_result.issuer_exposures.values())
        neff_lt_full = 1.0 / hhi_lt_full if hhi_lt_full > 0 else 0.0

        evidence_list.append(
            Evidence(
                id="EV-CONC-LT-HHI",
                metric="herfindahl_hirschman_index_lookthrough",
                value=round(hhi_lt_full, 6),
                formula="HHI_lt = sum(w_lt^2)",
                inputs={"lookthrough_weights": {k: round(v, 4) for k, v in lt_weights_all.items()}},
                data_sources=["ETF Disclosures & Direct Allocations"],
                sample_size=len(lt_weights_all),
                caveats=["Includes ETF residual holdings basket."],
            )
        )
        evidence_list.append(
            Evidence(
                id="EV-CONC-LT-NEFF",
                metric="effective_number_holdings_lookthrough",
                value=round(neff_lt_full, 2),
                formula="N_eff_lt = 1 / HHI_lt",
                inputs={"hhi_lookthrough": round(hhi_lt_full, 6)},
                data_sources=["ETF Disclosures & Direct Allocations"],
                sample_size=len(lt_weights_all),
                caveats=[],
            )
        )
        evidence_list.append(
            Evidence(
                id="EV-CONC-TOP1-WEIGHT",
                metric="top_single_issuer_lookthrough_weight",
                value=lt_profile.top_1_weight,
                formula="max(w_lt_issuer) [excluding cash/residual]",
                inputs={"top_issuer": lt_profile.top_issuers[0] if lt_profile.top_issuers else None},
                data_sources=["Decomposed Look-Through Weights"],
                sample_size=lt_profile.total_issuers_count,
                caveats=["Excludes OTHER_CONSTITUENTS and cash balances."],
            )
        )
        evidence_list.append(
            Evidence(
                id="EV-CONC-TOP5-WEIGHT",
                metric="top_5_issuers_lookthrough_weight",
                value=lt_profile.top_5_weight,
                formula="sum(top_5(w_lt_issuers)) [excluding cash/residual]",
                inputs={"top_5_issuers": lt_profile.top_issuers[:5]},
                data_sources=["Decomposed Look-Through Weights"],
                sample_size=lt_profile.total_issuers_count,
                caveats=["Excludes OTHER_CONSTITUENTS and cash balances."],
            )
        )

        return ConcentrationAnalysis(
            direct=direct_profile,
            lookthrough=lt_profile,
            evidence_list=evidence_list,
        )
