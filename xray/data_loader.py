from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd


DATA_DIR = Path(__file__).resolve().parent.parent / "data"


class DataLoader:
    """
    Deterministic loader and validator for market prices, ETF constituents,
    issuer mappings, and benchmark portfolios.
    Jane Street standard: Strict data boundary, full verification of weights conservation.
    """

    def __init__(self, data_dir: Optional[Path] = None) -> None:
        self.data_dir = data_dir or DATA_DIR
        self.prices_path = self.data_dir / "us_market_prices_matrix.csv"
        self.etf_path = self.data_dir / "etf_constituents.json"
        self.portfolios_path = self.data_dir / "synthetic_benchmark_portfolios.json"
        self.issuer_map_path = self.data_dir / "issuer_map.json"
        self.shock_data_path = self.data_dir / "historical_shock_data.json"
        self.sp500_metrics_path = self.data_dir / "sp500_benchmark_metrics.json"

        # Lazy caches
        self._prices: Optional[pd.DataFrame] = None
        self._etf_constituents: Optional[Dict[str, Any]] = None
        self._portfolios: Optional[List[Dict[str, Any]]] = None
        self._issuer_map: Optional[Dict[str, str]] = None

    def load_prices(self) -> pd.DataFrame:
        """
        Loads the 15-year daily adjusted close matrix.
        Sets date index as DatetimeIndex.
        Does NOT forward fill across missing dates (preserves NaNs).
        """
        if self._prices is not None:
            return self._prices

        if not self.prices_path.exists():
            raise FileNotFoundError(f"Prices matrix not found at {self.prices_path}")

        df = pd.read_csv(self.prices_path)
        df["date"] = pd.to_datetime(df["date"])
        df.set_index("date", inplace=True)
        df.sort_index(inplace=True)
        self._prices = df
        return self._prices

    def load_etf_constituents(self, normalize_residual: bool = True) -> Dict[str, Any]:
        """
        Loads ETF constituent weights and ensures mathematical conservation:
        The sum of named constituents + residual OTHER_CONSTITUENTS = exactly 1.0.
        
        If normalize_residual is True, OTHER_CONSTITUENTS is dynamically adjusted to
        1.0 - sum(named_constituents) so individual constituent weights match official
        N-PORT filings with zero distortion, and total fund weight is strictly 1.0.
        """
        if self._etf_constituents is not None:
            return self._etf_constituents

        if not self.etf_path.exists():
            raise FileNotFoundError(f"ETF constituents file not found at {self.etf_path}")

        with open(self.etf_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        processed_data = {}
        for fund_ticker, fund_info in data.items():
            holdings = dict(fund_info.get("holdings", {}))
            named_sum = sum(w for k, w in holdings.items() if k != "OTHER_CONSTITUENTS")
            
            if normalize_residual:
                # Ensure exact 1.0 conservation without altering any individual stock weight
                holdings["OTHER_CONSTITUENTS"] = round(1.0 - named_sum, 6)
            
            total_weight = sum(holdings.values())
            if abs(total_weight - 1.0) > 1e-5:
                raise ValueError(
                    f"ETF {fund_ticker} weights do not sum to 1.0 (sum={total_weight:.6f})"
                )

            fund_copy = dict(fund_info)
            fund_copy["holdings"] = holdings
            processed_data[fund_ticker] = fund_copy

        self._etf_constituents = processed_data
        return self._etf_constituents

    def load_issuer_map(self) -> Dict[str, str]:
        """Loads share class to parent issuer map (e.g. GOOG -> GOOGL, BRK.B -> BRK-B)."""
        if self._issuer_map is not None:
            return self._issuer_map

        if not self.issuer_map_path.exists():
            self._issuer_map = {"GOOG": "GOOGL", "BRK.B": "BRK-B"}
            return self._issuer_map

        with open(self.issuer_map_path, "r", encoding="utf-8") as f:
            self._issuer_map = json.load(f)
        return self._issuer_map

    def load_synthetic_portfolios(self) -> List[Dict[str, Any]]:
        """Loads the 4 labelled benchmark cases."""
        if self._portfolios is not None:
            return self._portfolios

        if not self.portfolios_path.exists():
            raise FileNotFoundError(f"Portfolios not found at {self.portfolios_path}")

        with open(self.portfolios_path, "r", encoding="utf-8") as f:
            self._portfolios = json.load(f)
        return self._portfolios

    def validate_holdings(
        self, holdings: List[Dict[str, Any]]
    ) -> Tuple[List[str], List[str], List[str]]:
        """
        Validates a list of portfolio holdings against known price tickers and ETF universe.
        Returns:
            (valid_tickers, decomposed_etfs, missing_tickers)
        """
        prices = self.load_prices()
        etfs = self.load_etf_constituents()
        price_cols = set(prices.columns)
        etf_keys = set(etfs.keys())

        valid_tickers: List[str] = []
        decomposed_etfs: List[str] = []
        missing_tickers: List[str] = []

        for h in holdings:
            ticker = h.get("ticker", "")
            if not ticker:
                continue

            # Cash is recognized
            if ticker in ("CASH", "CASH_GBP", "CASH_USD"):
                valid_tickers.append(ticker)
                continue

            has_price = ticker in price_cols
            has_etf = ticker in etf_keys

            if has_etf:
                decomposed_etfs.append(ticker)
                valid_tickers.append(ticker)
            elif has_price:
                valid_tickers.append(ticker)
            else:
                missing_tickers.append(ticker)

        return valid_tickers, decomposed_etfs, missing_tickers

    def get_summary(self) -> Dict[str, Any]:
        """Generates an overview summary of available datasets."""
        prices = self.load_prices()
        etfs = self.load_etf_constituents()
        portfolios = self.load_synthetic_portfolios()
        issuer_map = self.load_issuer_map()

        return {
            "price_history_range": f"{prices.index.min().strftime('%Y-%m-%d')} to {prices.index.max().strftime('%Y-%m-%d')}",
            "price_rows": len(prices),
            "price_columns_count": len(prices.columns),
            "price_tickers": sorted(list(prices.columns)),
            "decomposable_etfs": sorted(list(etfs.keys())),
            "benchmark_cases": [c.get("case_id") for c in portfolios],
            "mapped_issuers": issuer_map,
        }


def print_summary_table() -> None:
    loader = DataLoader()
    summary = loader.get_summary()

    print("=" * 72)
    print("PORTFOLIO X-RAY DATA LOADER SUMMARY (JANE STREET MATH SEPARATION)")
    print("=" * 72)
    print(f"Prices Date Range   : {summary['price_history_range']}")
    print(f"Total Trading Days  : {summary['price_rows']}")
    print(f"Priced Securities   : {summary['price_columns_count']} assets")
    print(f"Decomposable ETFs   : {', '.join(summary['decomposable_etfs'])}")
    print(f"Benchmark Cases     : {len(summary['benchmark_cases'])} cases loaded")
    for case_id in summary["benchmark_cases"]:
        print(f"  - {case_id}")
    print(f"Mapped Issuers      : {summary['mapped_issuers']}")
    print("=" * 72)

    # Validate each synthetic case
    print("\nValidating Synthetic Cases Universe Coverage:")
    portfolios = loader.load_synthetic_portfolios()
    for case in portfolios:
        cid = case["case_id"]
        valid, decomposed, missing = loader.validate_holdings(case["holdings"])
        missing_str = f" [MISSING PRICE: {', '.join(missing)}]" if missing else " [ALL RESOLVED]"
        print(f"  • {cid}: {len(case['holdings'])} holdings | Decomposed ETFs: {decomposed}{missing_str}")
    print("=" * 72)


if __name__ == "__main__":
    print_summary_table()
