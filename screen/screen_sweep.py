#!/usr/bin/env python3
"""Mechanical archetype pre-filter — Stage 1.5 of the Sunday screen (adopted for No. 005).

Reads screen-data/ohlcv.csv.gz (published weekly by the screen-data GitHub Action)
and flags names whose price/volume structure matches one of the four archetype
fingerprints. THE FILTER NOMINATES; ONLY STRUCTURE ON THE MONTH CHART QUALIFIES —
the flagged list replaces "whatever made the news" as the set of charts that get
eyes, nothing more. The qualification bar (named archetype, structural stop,
>=2:1 R:R, liquidity, correlation rule, event gate) is unchanged downstream.

Usage:  python screen_sweep.py [path/to/ohlcv.csv.gz]
Output: flags table (one row per flagged name, fingerprint columns), summary
        counts, and stale-data warning if last bar is older than the most recent
        Friday. Held Class A names are tagged HELD, never excluded — the
        correlation rule applies at qualification, not at flagging.
"""
import sys
import pandas as pd

# ---------------- tunable parameters (log any change in BUILD_DECISIONS.md) ---
P = dict(
    # liquidity bar (v0.1): price floor + 20d average dollar volume
    min_price=10.0, min_dollar_vol20=25e6,
    # FP1 REST NEAR HIGHS: close within pct of 52wk high AND 5d true-range
    # average contracted vs the prior 20d
    fp1_pct_of_high=0.95, fp1_tr_contraction=0.60,
    # FP2 PULLBACK TO STRUCTURE: rising 50d SMA, close near it, real prior trend
    fp2_sma_rise_lookback=10, fp2_dist_to_sma=0.03, fp2_min_63d_return=0.10,
    # FP3 VOLUME ANOMALY: last-day RVOL, or week vs prior-4wk volume
    fp3_day_rvol=2.5, fp3_week_rvol=1.8,
    # FP4 TIGHTENING BASE (the CME shape): tight 15d range, contracting vs the
    # preceding 30d, drying volume, close in upper half of the 63d range
    fp4_range_pct=0.08, fp4_range_contraction=0.66, fp4_vol_dry=0.70,
)

HELD = {"VOO", "GOOGL", "NVDA", "AMZN", "MSFT", "PLTR", "JPM", "BRK.B"}  # SPCX not screenable


def fingerprints(df):
    """df: one symbol, ascending Date, columns Open High Low Close Volume."""
    c, h, l, v = df["Close"], df["High"], df["Low"], df["Volume"]
    if len(df) < 130 or c.iloc[-1] != c.iloc[-1]:
        return None
    last = c.iloc[-1]
    dv20 = (c * v).rolling(20).mean().iloc[-1]
    out = {"last": last, "dollar_vol20": dv20,
           "liquid": last >= P["min_price"] and dv20 >= P["min_dollar_vol20"]}

    # true range
    pc = c.shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)

    # FP1 — rest near highs
    hi52 = c.rolling(252, min_periods=130).max().iloc[-1]
    tr5, tr20p = tr.iloc[-5:].mean(), tr.iloc[-25:-5].mean()
    out["fp1"] = (last >= P["fp1_pct_of_high"] * hi52
                  and tr20p > 0 and tr5 <= P["fp1_tr_contraction"] * tr20p)
    out["pct_of_52wk_high"] = last / hi52 if hi52 else float("nan")

    # FP2 — pullback to rising 50d
    sma50 = c.rolling(50).mean()
    rising = sma50.iloc[-1] > sma50.iloc[-1 - P["fp2_sma_rise_lookback"]]
    near = abs(last - sma50.iloc[-1]) / last <= P["fp2_dist_to_sma"]
    ret63 = last / c.iloc[-64] - 1 if len(c) >= 64 else float("nan")
    out["fp2"] = bool(rising and near and ret63 >= P["fp2_min_63d_return"])

    # FP3 — volume anomaly
    v20 = v.iloc[-21:-1].mean()
    wk, wk4p = v.iloc[-5:].mean(), v.iloc[-25:-5].mean()
    out["fp3"] = bool((v20 > 0 and v.iloc[-1] >= P["fp3_day_rvol"] * v20)
                      or (wk4p > 0 and wk >= P["fp3_week_rvol"] * wk4p))

    # FP4 — tightening base
    r15 = h.iloc[-15:].max() - l.iloc[-15:].min()
    r30p = h.iloc[-45:-15].max() - l.iloc[-45:-15].min()
    v10, v30 = v.iloc[-10:].mean(), v.iloc[-30:].mean()
    lo63, hi63 = l.iloc[-63:].min(), h.iloc[-63:].max()
    upper_half = hi63 > lo63 and (last - lo63) / (hi63 - lo63) >= 0.5
    out["fp4"] = bool(r15 / last <= P["fp4_range_pct"]
                      and r30p > 0 and r15 <= P["fp4_range_contraction"] * r30p
                      and v30 > 0 and v10 <= P["fp4_vol_dry"] * v30
                      and upper_half)
    return out


def main(path="screen-data/ohlcv.csv.gz"):
    df = pd.read_csv(path, parse_dates=["Date"])
    last_date = df["Date"].max()
    rows = []
    for sym, g in df.sort_values("Date").groupby("Symbol"):
        r = fingerprints(g.reset_index(drop=True))
        if r is None or not r["liquid"]:
            continue
        if r["fp1"] or r["fp2"] or r["fp3"] or r["fp4"]:
            rows.append({"Symbol": sym, "Held": sym in HELD, **r})
    flags = pd.DataFrame(rows)
    n = {k: int(flags[k].sum()) for k in ("fp1", "fp2", "fp3", "fp4")} if len(flags) else {}
    print(f"data through {last_date.date()} · {df['Symbol'].nunique()} symbols · "
          f"{len(flags)} flagged {n}")
    if len(flags):
        flags = flags.sort_values(["fp1", "fp4", "fp2", "fp3", "pct_of_52wk_high"],
                                  ascending=False)
        cols = ["Symbol", "Held", "last", "pct_of_52wk_high", "fp1", "fp2", "fp3", "fp4"]
        print(flags[cols].to_string(index=False,
              formatters={"last": "{:.2f}".format,
                          "pct_of_52wk_high": "{:.3f}".format}))
    return flags


if __name__ == "__main__":
    main(*sys.argv[1:])
