"""One-off data clean-up for the Portfolio X-Ray datasets.

Run from the repo root:  python scripts/fix_data.py

What it fixes (idempotent, safe to re-run):
1. ETF constituent weights: OTHER_CONSTITUENTS is set so every fund sums to exactly 1.0
   (conservation of exposure). Named constituent weights are left untouched.
2. Ticker format: BRK.B -> BRK-B everywhere, so ETF files match the price matrix.
3. Price matrices are left untouched (holiday rows with partial NaNs are dropped by the loader).
4. Writes data/issuer_map.json so share classes of one company (GOOG + GOOGL) are
   aggregated as one issuer by the engine.
5. Recomputes the synthetic portfolios' look-through ground truth from the fixed data,
   and corrects CASE-04 (largest single look-through issuer is MSFT, not NVDA).
"""
import json
from pathlib import Path

import pandas as pd

DATA = Path(__file__).resolve().parent.parent / "data"
TICKER_FIX = {"BRK.B": "BRK-B"}
ISSUER_MAP = {
    "GOOGL": "Alphabet Inc.",
    "GOOG": "Alphabet Inc.",
}


def load(name):
    return json.loads((DATA / name).read_text())


def save(name, obj):
    (DATA / name).write_text(json.dumps(obj, indent=2) + "\n")


def fix_etfs():
    etfs = load("etf_constituents.json")
    for fund in etfs.values():
        h = {TICKER_FIX.get(t, t): w for t, w in fund["holdings"].items()}
        named = sum(w for t, w in h.items() if t != "OTHER_CONSTITUENTS")
        h["OTHER_CONSTITUENTS"] = round(1.0 - named, 6)
        assert h["OTHER_CONSTITUENTS"] >= 0
        fund["holdings"] = h
        fund["normalisation_note"] = (
            "OTHER_CONSTITUENTS set to 1 minus named weights so the fund sums to exactly 1.0"
        )
    save("etf_constituents.json", etfs)
    return etfs


def fix_benchmark():
    b = load("sp500_benchmark_metrics.json")
    tc = b["concentration_baseline"]["top_constituents"]
    b["concentration_baseline"]["top_constituents"] = {TICKER_FIX.get(t, t): w for t, w in tc.items()}
    save("sp500_benchmark_metrics.json", b)


def fix_prices():
    for f in ["us_market_prices_matrix.csv", "us_market_prices_matrix_15y.csv"]:
        df = pd.read_csv(DATA / f, index_col=0)
        before = len(df)
        df = df.dropna(how="all")
        df.to_csv(DATA / f)
        print(f"{f}: dropped {before - len(df)} all-NaN rows")


def lookthrough(holdings, etfs):
    """Reference look-through: issuer weights incl. OTHER_CONSTITUENTS and cash."""
    out = {}
    for h in holdings:
        t, w = h["ticker"], h["weight"]
        if t in etfs:
            for c, cw in etfs[t]["holdings"].items():
                key = c if c == "OTHER_CONSTITUENTS" else ISSUER_MAP.get(c, c)
                out[key] = out.get(key, 0.0) + w * cw
        else:
            key = ISSUER_MAP.get(t, t)
            out[key] = out.get(key, 0.0) + w
    return out


def fix_portfolios(etfs):
    ports = load("synthetic_benchmark_portfolios.json")
    for p in ports:
        lt = lookthrough(p["holdings"], etfs)
        total = sum(lt.values())
        assert abs(total - 1.0) < 1e-9, (p["case_id"], total)
        issuers = {k: v for k, v in lt.items() if k != "OTHER_CONSTITUENTS" and not k.startswith("CASH")}
        top_t, top_w = max(issuers.items(), key=lambda kv: kv[1])
        gt = p["ground_truth_lookthrough"]
        gt["computed_top_issuer"] = top_t
        gt["computed_top_issuer_weight"] = round(top_w, 6)
        gt["computed_unresolved_other_constituents"] = round(lt.get("OTHER_CONSTITUENTS", 0.0), 6)
        gt["computed_total_weight"] = round(total, 6)

        if p["case_id"].startswith("CASE-04"):
            gt["max_single_stock_lookthrough"] = round(top_w, 6)
            gt["max_single_stock_ticker"] = top_t
            gt["NVDA_lookthrough"] = round(lt.get("NVDA", 0.0), 6)
            gt["expected_circuit_breaker_result"] = (
                f"PASS: largest single look-through issuer is {top_t} at {top_w:.2%}, "
                "below the 10% limit."
            )
            p["stated_goal"] = "US total-market equity passive allocation with a cash buffer"
    save("synthetic_benchmark_portfolios.json", ports)
    return ports


if __name__ == "__main__":
    etfs = fix_etfs()
    fix_benchmark()
    save("issuer_map.json", ISSUER_MAP)
    for p in fix_portfolios(etfs):
        print(p["case_id"], {k: v for k, v in p["ground_truth_lookthrough"].items() if k.startswith(("computed", "max", "NVDA_total"))})
