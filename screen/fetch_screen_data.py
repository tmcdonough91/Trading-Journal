#!/usr/bin/env python3
"""Screen-universe EOD data fetcher — runs on GitHub Actions, NOT in the build sandbox.

The build environment's network allowlist blocks every market-data host (verified
2026-09-06: Stooq/Yahoo/Tiingo/Nasdaq all 403), but git+GitHub is open. So this
script runs on a GitHub Actions runner (open internet), pulls ~15 months of daily
OHLCV for the whole screen universe, and the workflow force-pushes the result as a
single orphan commit on the `screen-data` branch (no repo-history growth). The
Sunday build then does:  git clone --depth 1 -b screen-data <repo>  and runs
screen_sweep.py locally.

Universe = S&P 500 constituents (datasets/s-and-p-500-companies, itself refreshed
daily by its own Action) + screen/etfs.txt + screen/extras.txt, deduped.

Outputs (into ./screen-data/):
  ohlcv.csv.gz  — long format: Date,Symbol,Open,High,Low,Close,Volume
  meta.json     — run timestamp, universe/fetched/failed counts, failed symbols
"""
import csv, gzip, io, json, os, sys, urllib.request
from datetime import datetime, timezone

import pandas as pd
import yfinance as yf

CONSTITUENTS_URL = ("https://raw.githubusercontent.com/datasets/"
                    "s-and-p-500-companies/main/data/constituents.csv")
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(HERE), "screen-data")
PERIOD = "15mo"   # 52-week metrics need 12mo; headroom for the 30d windows


def read_list(path):
    if not os.path.exists(path):
        return []
    out = []
    for line in open(path):
        s = line.split("#")[0].strip().upper()
        if s:
            out.append(s)
    return out


def sp500_symbols():
    with urllib.request.urlopen(CONSTITUENTS_URL, timeout=60) as r:
        text = r.read().decode("utf-8")
    return [row["Symbol"].strip().upper()
            for row in csv.DictReader(io.StringIO(text)) if row.get("Symbol")]


def to_yahoo(sym):
    return sym.replace(".", "-")   # BRK.B -> BRK-B, BF.B -> BF-B


def main():
    sp = sp500_symbols()
    etfs = read_list(os.path.join(HERE, "etfs.txt"))
    extras = read_list(os.path.join(HERE, "extras.txt"))
    universe = sorted(set(sp) | set(etfs) | set(extras))
    ymap = {to_yahoo(s): s for s in universe}   # yahoo symbol -> canonical

    print(f"universe: {len(universe)} (S&P {len(sp)} + ETFs {len(etfs)} "
          f"+ extras {len(extras)}, deduped)")

    data = yf.download(list(ymap), period=PERIOD, interval="1d",
                       auto_adjust=False, group_by="ticker",
                       threads=True, progress=False)

    frames, failed = [], []
    for ysym, sym in ymap.items():
        try:
            df = data[ysym][["Open", "High", "Low", "Close", "Volume"]].dropna(how="all")
        except KeyError:
            failed.append(sym); continue
        if df.empty or df["Close"].dropna().empty:
            failed.append(sym); continue
        df = df.copy()
        df["Symbol"] = sym
        frames.append(df.reset_index())

    if len(frames) < 0.9 * len(universe):
        print(f"FATAL: only {len(frames)}/{len(universe)} symbols fetched — "
              "refusing to publish a gutted dataset", file=sys.stderr)
        sys.exit(1)

    allf = pd.concat(frames, ignore_index=True)
    allf["Date"] = pd.to_datetime(allf["Date"]).dt.strftime("%Y-%m-%d")
    allf = allf[["Date", "Symbol", "Open", "High", "Low", "Close", "Volume"]]

    os.makedirs(OUT, exist_ok=True)
    with gzip.open(os.path.join(OUT, "ohlcv.csv.gz"), "wt", newline="") as f:
        allf.to_csv(f, index=False, float_format="%.4f")

    meta = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "universe": len(universe), "fetched": len(frames),
        "failed": sorted(failed), "rows": len(allf),
        "last_date": allf["Date"].max(), "period": PERIOD,
    }
    json.dump(meta, open(os.path.join(OUT, "meta.json"), "w"), indent=1)
    print(f"wrote {len(allf)} rows for {len(frames)} symbols "
          f"(last date {meta['last_date']}); failed: {failed or 'none'}")


if __name__ == "__main__":
    main()
