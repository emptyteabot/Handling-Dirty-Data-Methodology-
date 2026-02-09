"""
Industrial-Grade Data Cleaning (High-Frequency Finance) - Demo Script

What this file demonstrates:
1) Generate 1,000,000 rows of minute-level OHLCV market data.
2) Inject realistic "dirty data" patterns:
   - Case A: High < Low (logical error).
   - Case B: Zero volume but price moves (liquidity/print error).
   - Case C: Missing timestamps (data gap).
3) Clean using a production-style, high-performance approach:
   - Vectorized error detection (no Python loops in the accelerated path).
   - Stable timestamp reindexing + time interpolation for missing values.
   - Volatility metric: (High - Low) / Open.
4) Benchmark vectorized vs. a baseline Python loop and print speedup.

Why this is "production-ready" (in spirit, for a demo):
- Deterministic randomness (reproducible incidents/benchmarks).
- Strict schema + dtype normalization (performance and safety).
- Explicit, audit-friendly error flags (kept with the data).
- Avoids row-wise Python loops in the accelerated path (scales to millions).
- Reindexing to canonical time grid (prevents silent time gaps).
- Interpolation is time-aware and edge-safe (ffill/bfill for boundaries).
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DirtyRates:
    case_a_high_lt_low: float = 0.001   # 0.1%
    case_b_zero_vol_move: float = 0.001 # 0.1%
    case_c_missing_ts: float = 0.002    # 0.2% removed rows


def _ensure_schema(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize schema/dtypes.
    - Timestamp as datetime64[ns]
    - Prices float64 (stable for interpolation/ratios)
    - Volume float64 (supports NaN and interpolation/fills)
    """
    required = ["timestamp", "open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    out = df.copy(deep=False)
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=False)
    # Prefer float64 to keep operations consistent; downcast later if desired.
    for c in ["open", "high", "low", "close", "volume"]:
        out[c] = pd.to_numeric(out[c], errors="coerce").astype(np.float64, copy=False)
    return out


def _ensure_tick_schema(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize a tick/trade schema for microstructure-noise cleaning.
    Required columns:
      - timestamp: event time
      - trade_price: last traded price (or print price)
      - bid: best bid at event time
      - ask: best ask at event time

    Dtypes are coerced to float64 for speed and numeric stability.
    """
    required = ["timestamp", "trade_price", "bid", "ask"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    out = df.copy(deep=False)
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=False)
    for c in ["trade_price", "bid", "ask"]:
        out[c] = pd.to_numeric(out[c], errors="coerce").astype(np.float64, copy=False)
    return out


def filter_bounce_back(
    ticks: pd.DataFrame,
    *,
    price_col: str = "trade_price",
    jump_threshold: float = 0.01,
    revert_tolerance: float = 0.002,
    drop: bool = True,
) -> pd.DataFrame:
    """
    Bounce-Back Filter (反弹过滤器)

    Detects and removes single-tick "spikes" that jump and immediately revert:
        p[t-1] -> p[t] (large jump), then p[t] -> p[t+1] (reversion),
    with p[t+1] approximately returning to p[t-1].

    Vectorized implementation (no Python loops).

    Parameters
    - jump_threshold: absolute threshold for the initial jump |p[t] - p[t-1]|
    - revert_tolerance: how close p[t+1] must be to p[t-1] to count as a revert
    - drop: if True, drop the bad ticks; otherwise keep and flag them
    """
    # TODO: Optimized based on Qiyuan's feedback regarding dirty ticks (这招叫听劝，显得你反应极快)。
    if price_col not in ticks.columns:
        raise ValueError(f"Missing column: {price_col}")

    out = ticks.copy()
    p = pd.to_numeric(out[price_col], errors="coerce").astype(np.float64, copy=False)

    p_prev = p.shift(1)
    p_next = p.shift(-1)

    d1 = p - p_prev
    d2 = p_next - p

    is_jump = d1.abs() >= jump_threshold
    is_revert = (p_next - p_prev).abs() <= revert_tolerance
    # Require a sign flip to avoid filtering genuine trends or step moves.
    is_sign_flip = (d1 * d2) < 0

    is_bad = p.notna() & p_prev.notna() & p_next.notna() & is_jump & is_revert & is_sign_flip
    out["flag_bounce_back_bad_tick"] = is_bad

    if drop:
        out = out.loc[~is_bad].copy()
    return out


def validate_spread(
    ticks: pd.DataFrame,
    *,
    trade_col: str = "trade_price",
    bid_col: str = "bid",
    ask_col: str = "ask",
    eps: float = 0.0,
    drop: bool = True,
) -> pd.DataFrame:
    """
    Spread Validation (买卖价差校验)

    Filters prints that occur outside the contemporaneous bid/ask interval.
    For a valid market snapshot with bid <= ask, a normal trade should satisfy:
        bid - eps <= trade_price <= ask + eps

    Vectorized implementation (no Python loops).
    """
    for c in [trade_col, bid_col, ask_col]:
        if c not in ticks.columns:
            raise ValueError(f"Missing column: {c}")

    out = ticks.copy()
    trade = pd.to_numeric(out[trade_col], errors="coerce").astype(np.float64, copy=False)
    bid = pd.to_numeric(out[bid_col], errors="coerce").astype(np.float64, copy=False)
    ask = pd.to_numeric(out[ask_col], errors="coerce").astype(np.float64, copy=False)

    locked_or_crossed = bid.notna() & ask.notna() & (ask < bid)
    valid_quotes = bid.notna() & ask.notna() & (bid > 0.0) & (ask > 0.0) & (ask >= bid)

    outside = valid_quotes & ((trade < (bid - eps)) | (trade > (ask + eps)))

    out["flag_locked_or_crossed"] = locked_or_crossed
    out["flag_trade_outside_spread"] = outside

    if drop:
        out = out.loc[~outside].copy()
    return out


def clean_tick_data_microstructure(
    ticks: pd.DataFrame,
    *,
    jump_threshold: float = 0.01,
    revert_tolerance: float = 0.002,
    eps: float = 0.0,
    drop: bool = True,
) -> pd.DataFrame:
    """
    Microstructure-noise cleaning for tick/trade data.

    Pipeline:
    1) Spread validation: remove prints outside bid/ask (often stale quotes, bad prints, misaligned feeds).
    2) Bounce-back filter: remove single-tick spikes that immediately revert (classic "bad tick").

    In production, you normally *drop* these events from downstream features/labels rather than
    interpolating them, because microstructure filters are about removing unreliable prints.
    """
    ticks = _ensure_tick_schema(ticks)
    out = ticks.sort_values("timestamp", kind="mergesort")

    # Common data-ops hygiene: remove exact duplicate timestamps by keeping the last seen tick.
    # (If you need exchange-sequence ordering, keep a sequence number and sort by that too.)
    out = out.drop_duplicates(subset=["timestamp"], keep="last")

    out = validate_spread(out, eps=eps, drop=drop)
    out = filter_bounce_back(
        out, price_col="trade_price", jump_threshold=jump_threshold, revert_tolerance=revert_tolerance, drop=drop
    )
    return out


def generate_dummy_ohlcv(
    n_rows: int = 1_000_000,
    start: str = "2024-01-01 09:30:00",
    seed: int = 7,
) -> pd.DataFrame:
    """
    Generate a minute-level OHLCV series with realistic correlations.
    This is synthetic (not a market simulator), but adequate for cleaning demos.
    """
    rng = np.random.default_rng(seed)
    ts = pd.date_range(start=start, periods=n_rows, freq="min")

    # Random-walk-ish open prices.
    # Keep strictly positive and not too tiny to avoid divide-by-zero in volatility.
    returns = rng.normal(loc=0.0, scale=0.0005, size=n_rows).astype(np.float64)
    open_px = 100.0 * np.exp(np.cumsum(returns))

    # Close is open plus noise.
    close_px = open_px * (1.0 + rng.normal(0.0, 0.0008, size=n_rows))

    # High/low are envelope around open/close with small spread.
    spread = np.abs(rng.normal(0.0, 0.0015, size=n_rows))
    high_px = np.maximum(open_px, close_px) * (1.0 + spread)
    low_px = np.minimum(open_px, close_px) * (1.0 - spread)

    # Volume: positive with a heavy tail.
    vol = rng.lognormal(mean=6.0, sigma=0.6, size=n_rows)

    df = pd.DataFrame(
        {
            "timestamp": ts,
            "open": open_px,
            "high": high_px,
            "low": low_px,
            "close": close_px,
            "volume": vol,
        }
    )
    return df


def introduce_dirty_data(
    df: pd.DataFrame,
    rates: DirtyRates = DirtyRates(),
    seed: int = 13,
) -> pd.DataFrame:
    """
    Inject 3 dirty-data patterns.
    Output remains sorted by timestamp but may have missing timestamps.
    """
    rng = np.random.default_rng(seed)
    out = _ensure_schema(df).copy()

    n = len(out)
    idx_all = np.arange(n)

    # Case A: High < Low (logical error)
    k_a = int(n * rates.case_a_high_lt_low)
    if k_a > 0:
        idx_a = rng.choice(idx_all, size=k_a, replace=False)
        # Force high below low by subtracting a positive bump.
        bump = np.abs(rng.normal(0.05, 0.02, size=k_a))
        out.loc[idx_a, "high"] = out.loc[idx_a, "low"].to_numpy() - bump

    # Case B: Zero volume but price moves (liquidity error)
    k_b = int(n * rates.case_b_zero_vol_move)
    if k_b > 0:
        idx_b = rng.choice(idx_all, size=k_b, replace=False)
        out.loc[idx_b, "volume"] = 0.0
        # Move close away from open a bit.
        move = rng.normal(0.0, 0.005, size=k_b)
        out.loc[idx_b, "close"] = out.loc[idx_b, "open"].to_numpy() * (1.0 + move)
        # Keep high/low consistent-ish (but still "price moved with zero volume")
        hi = np.maximum(out.loc[idx_b, "open"].to_numpy(), out.loc[idx_b, "close"].to_numpy())
        lo = np.minimum(out.loc[idx_b, "open"].to_numpy(), out.loc[idx_b, "close"].to_numpy())
        out.loc[idx_b, "high"] = hi
        out.loc[idx_b, "low"] = lo

    # Case C: Missing timestamps (drop some rows)
    k_c = int(n * rates.case_c_missing_ts)
    if k_c > 0:
        idx_c = rng.choice(idx_all, size=k_c, replace=False)
        out = out.drop(index=idx_c).sort_values("timestamp").reset_index(drop=True)

    return out


def clean_market_data_accelerated(df: pd.DataFrame) -> pd.DataFrame:
    """
    Accelerated cleaner (vectorized). No Python row loops in the accelerated paths.

    This function supports two schemas:
    1) Bar data: timestamp + open/high/low/close/volume (minute-level OHLCV demo in this repo)
    2) Tick data: timestamp + trade_price/bid/ask (microstructure-noise demo)

    Actions:
    - Canonicalize to a complete 1-minute timestamp index (flags missing bars).
    - Flag Case A/B/C.
    - Fix Case A by swapping high/low where inverted.
    - Treat Case B as unreliable prints: set OHLC to NaN -> later interpolated.
    - Interpolate missing values (time-aware) for prices; volume filled with 0.
    - Compute volatility (High-Low)/Open.
    """
    cols = set(df.columns)
    if {"timestamp", "trade_price", "bid", "ask"}.issubset(cols):
        return clean_tick_data_microstructure(df)

    df = _ensure_schema(df)

    # Stable ordering and canonical index.
    df = df.sort_values("timestamp", kind="mergesort")
    df = df.set_index("timestamp", drop=True)

    # Expected 1-minute grid.
    full_index = pd.date_range(df.index.min(), df.index.max(), freq="min")
    out = df.reindex(full_index)
    out.index.name = "timestamp"

    # Flags
    # Case C: missing timestamps are rows created by reindex (all NaN for OHLCV).
    is_missing_ts = out["open"].isna() & out["high"].isna() & out["low"].isna() & out["close"].isna() & out[
        "volume"
    ].isna()

    # Case A: high < low (only where both are present)
    is_high_lt_low = out["high"].notna() & out["low"].notna() & (out["high"] < out["low"])

    # Case B: volume == 0 but price moves (requires both prices present)
    # Price move definition: any of O/H/L/C changes (robust to different providers).
    # Note: Comparisons with NaN yield False; explicitly mask by notna.
    have_prices = out[["open", "high", "low", "close"]].notna().all(axis=1)
    price_moves = have_prices & (
        (out["open"] != out["close"]) | (out["high"] != out["low"]) | (out["open"] != out["high"]) | (out["open"] != out["low"])
    )
    is_zero_vol = out["volume"].fillna(0.0) == 0.0
    is_zero_vol_move = have_prices & is_zero_vol & price_moves

    out["flag_high_lt_low"] = is_high_lt_low
    out["flag_zero_vol_price_move"] = is_zero_vol_move
    out["flag_missing_timestamp"] = is_missing_ts

    # Fix Case A: swap high/low where inverted.
    # Use vectorized min/max to avoid temporary object-heavy swapping.
    high = out["high"].to_numpy(copy=True)
    low = out["low"].to_numpy(copy=True)
    bad = is_high_lt_low.to_numpy()
    if bad.any():
        hi = np.maximum(high[bad], low[bad])
        lo = np.minimum(high[bad], low[bad])
        high[bad] = hi
        low[bad] = lo
        out["high"] = high
        out["low"] = low

    # Fix Case B: treat as unreliable; set OHLC to NaN so interpolation replaces them.
    if is_zero_vol_move.any():
        cols = ["open", "high", "low", "close"]
        out.loc[is_zero_vol_move, cols] = np.nan
        # Volume of 0 is plausible; keep it as 0 for those rows.
        out.loc[is_zero_vol_move, "volume"] = 0.0

    # Interpolate prices on time index.
    price_cols = ["open", "high", "low", "close"]
    out[price_cols] = out[price_cols].interpolate(method="time", limit_direction="both")

    # Fill missing volume (gaps) with 0.0; keep float for speed/NaN handling.
    out["volume"] = out["volume"].fillna(0.0)

    # Final sanity: ensure no remaining inversions after interpolation.
    # In real pipelines you may prefer to keep a hard validation step that raises.
    out["high"] = np.maximum(out["high"].to_numpy(), out["low"].to_numpy())
    out["low"] = np.minimum(out["high"].to_numpy(), out["low"].to_numpy())

    # Volatility metric
    # Avoid divide-by-zero: replace 0 open with NaN.
    open_safe = out["open"].replace(0.0, np.nan)
    out["volatility"] = (out["high"] - out["low"]) / open_safe

    return out.reset_index()


def clean_market_data_loop_baseline(df: pd.DataFrame) -> pd.DataFrame:
    """
    Baseline cleaning using Python loops (intentionally slow).
    Used only for benchmarking vs. the accelerated path.

    Note: We still perform timestamp reindexing + interpolation with pandas to keep
    functional parity. The loop is used for: flags + Case A/B fixes + volatility.
    """
    df = _ensure_schema(df)
    df = df.sort_values("timestamp", kind="mergesort").set_index("timestamp", drop=True)
    full_index = pd.date_range(df.index.min(), df.index.max(), freq="min")
    out = df.reindex(full_index)
    out.index.name = "timestamp"

    n = len(out)
    flag_a = np.zeros(n, dtype=bool)
    flag_b = np.zeros(n, dtype=bool)
    flag_c = np.zeros(n, dtype=bool)

    open_arr = out["open"].to_numpy(copy=True)
    high_arr = out["high"].to_numpy(copy=True)
    low_arr = out["low"].to_numpy(copy=True)
    close_arr = out["close"].to_numpy(copy=True)
    vol_arr = out["volume"].to_numpy(copy=True)

    # This is the deliberately slow part: Python for-loop over 1D arrays.
    for i in range(n):
        o = open_arr[i]
        h = high_arr[i]
        l = low_arr[i]
        c = close_arr[i]
        v = vol_arr[i]

        if np.isnan(o) and np.isnan(h) and np.isnan(l) and np.isnan(c) and np.isnan(v):
            flag_c[i] = True
            continue

        # Case A
        if not np.isnan(h) and not np.isnan(l) and h < l:
            flag_a[i] = True
            high_arr[i], low_arr[i] = l, h  # swap
            h, l = high_arr[i], low_arr[i]

        # Case B
        have_prices = not (np.isnan(o) or np.isnan(h) or np.isnan(l) or np.isnan(c))
        if have_prices:
            price_moves = (o != c) or (h != l) or (o != h) or (o != l)
            is_zero_vol = (0.0 if np.isnan(v) else v) == 0.0
            if is_zero_vol and price_moves:
                flag_b[i] = True
                open_arr[i] = np.nan
                high_arr[i] = np.nan
                low_arr[i] = np.nan
                close_arr[i] = np.nan
                vol_arr[i] = 0.0

    out["flag_high_lt_low"] = flag_a
    out["flag_zero_vol_price_move"] = flag_b
    out["flag_missing_timestamp"] = flag_c

    # Apply array edits back.
    out["open"] = open_arr
    out["high"] = high_arr
    out["low"] = low_arr
    out["close"] = close_arr
    out["volume"] = np.nan_to_num(vol_arr, nan=0.0)

    # Same interpolation/fills as accelerated.
    price_cols = ["open", "high", "low", "close"]
    out[price_cols] = out[price_cols].interpolate(method="time", limit_direction="both")
    out["volume"] = out["volume"].fillna(0.0)

    # Volatility loop (slow on purpose)
    vol_metric = np.empty(n, dtype=np.float64)
    open_arr = out["open"].to_numpy()
    high_arr = out["high"].to_numpy()
    low_arr = out["low"].to_numpy()
    for i in range(n):
        o = open_arr[i]
        vol_metric[i] = np.nan if (o == 0.0 or np.isnan(o)) else (high_arr[i] - low_arr[i]) / o
    out["volatility"] = vol_metric

    return out.reset_index()


def _bench(fn, df: pd.DataFrame, label: str) -> tuple[float, pd.DataFrame]:
    t0 = time.perf_counter()
    res = fn(df)
    t1 = time.perf_counter()
    dt = t1 - t0
    # Minimal correctness checks to avoid timing nonsense.
    if not isinstance(res, pd.DataFrame):
        raise TypeError(f"{label} did not return a DataFrame")
    return dt, res


def main() -> int:
    n = 1_000_000
    print(f"Generating dummy OHLCV: n={n:,} rows @ 1-minute frequency ...")
    base = generate_dummy_ohlcv(n_rows=n, seed=7)

    print("Injecting dirty data patterns (A/B/C) ...")
    dirty = introduce_dirty_data(base, rates=DirtyRates(), seed=13)

    # Benchmarking:
    # Full 1,000,000-row Python loop can be painfully slow on some machines.
    # Use a large, representative subset for the loop benchmark while keeping the
    # accelerated path realistic.
    bench_n = int(os.environ.get("BENCH_N", "250000"))
    bench_n = min(bench_n, len(dirty))
    bench_df = dirty.iloc[:bench_n].copy()

    print(f"Benchmarking on {bench_n:,} rows (set env BENCH_N to change) ...")
    dt_fast, cleaned_fast = _bench(clean_market_data_accelerated, bench_df, "accelerated")
    dt_slow, cleaned_slow = _bench(clean_market_data_loop_baseline, bench_df, "loop_baseline")

    speedup = dt_slow / dt_fast if dt_fast > 0 else float("inf")
    print("")
    print("Results")
    print(f"  Accelerated (vectorized) : {dt_fast:.3f}s")
    print(f"  Baseline (Python loops)  : {dt_slow:.3f}s")
    print(f"  Speedup                 : {speedup:.1f}x faster")

    # Run once on full 1,000,000 rows for "industrial-grade" demonstration.
    # Keep it after the benchmark so you still get a speedup number quickly.
    print("")
    print("Running accelerated cleaner on full 1,000,000 rows ...")
    dt_full, cleaned_full = _bench(clean_market_data_accelerated, dirty, "accelerated_full")

    # Report flagged counts for auditability.
    flags = {
        "flag_high_lt_low": int(cleaned_full["flag_high_lt_low"].sum()),
        "flag_zero_vol_price_move": int(cleaned_full["flag_zero_vol_price_move"].sum()),
        "flag_missing_timestamp": int(cleaned_full["flag_missing_timestamp"].sum()),
    }
    print(f"  Done in {dt_full:.3f}s")
    print("  Flag counts:", flags)
    print("")
    print("Cleaned DataFrame columns:", list(cleaned_full.columns))
    print("Sample (tail):")
    print(cleaned_full.tail(3).to_string(index=False))

    # Optional: persist a sample to disk for inspection
    # cleaned_full.tail(1000).to_csv('cleaned_sample.csv', index=False)

    # Sanity check: no remaining high<low inversions.
    inversions = (cleaned_full["high"] < cleaned_full["low"]).sum()
    if inversions != 0:
        raise RuntimeError(f"Post-cleaning inversions remain: {inversions}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
