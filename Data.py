"""
data.py
=======
Data layer for the futures/FX backtesting framework.

Responsibilities
-----------------
1. Provide a `MarketData` container with a strict schema so every downstream
   module (signals, costs, risk, backtest engine) can rely on consistent
   columns regardless of the data source.
2. `load_ohlcv_csv()` — load real futures/FX bar data from disk (CSV/parquet).
3. `generate_synthetic_futures_data()` — generate a realistic synthetic
   FX/futures tick-aggregated bar series (with bid/ask spread, volume, and a
   latent informed-order-flow process) purely for demonstrating and unit
   testing the framework end-to-end when no live feed is connected.

Schema (all instruments normalized to this):
    timestamp (index, tz-aware UTC)
    open, high, low, close      -> trade/mid price
    bid, ask                    -> quoted touch prices
    volume                      -> contracts/lots traded in the bar
    signed_volume               -> proxy for buyer- vs seller-initiated flow
                                    (+ = net buy pressure, - = net sell)

Design notes
------------
- No look-ahead: every derived column here is computable strictly from
  information available *up to and including* that bar's close. Signal
  modules are responsible for additionally lagging by one bar before use
  in the backtest engine (enforced there, not here).
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass


@dataclass
class InstrumentSpec:
    """Contract specification needed to convert price moves into P&L."""
    symbol: str
    tick_size: float          # minimum price increment
    tick_value: float         # $ value of one tick move, per contract/lot
    point_value: float        # $ value of one full price unit move, per contract
    avg_spread_ticks: float   # typical quoted spread, in ticks
    contract_multiplier: float = 1.0
    margin_per_contract: float = 1000.0


# A few realistic presets (approximate, illustrative — not live exchange specs)
INSTRUMENTS = {
    "ES": InstrumentSpec("ES", tick_size=0.25, tick_value=12.50, point_value=50.0,
                         avg_spread_ticks=1.0, margin_per_contract=13200.0),
    "6E": InstrumentSpec("6E", tick_size=0.00005, tick_value=6.25, point_value=125000.0,
                         avg_spread_ticks=1.0, margin_per_contract=2500.0),
    "EURUSD": InstrumentSpec("EURUSD", tick_size=0.00001, tick_value=1.0, point_value=100000.0,
                              avg_spread_ticks=1.2, margin_per_contract=500.0),
}


def generate_synthetic_futures_data(
    n_bars: int = 20_000,
    bar_seconds: int = 30,
    start: str = "2025-01-02 00:00:00",
    instrument: str = "ES",
    seed: int = 42,
    annual_vol: float = 0.16,
    informed_flow_prob: float = 0.15,
    informed_flow_strength: float = 3.0,
) -> pd.DataFrame:
    """
    Generate a synthetic sub-minute bar series with microstructure texture:
    a GARCH-lite volatility process, a latent 'informed trader' regime that
    injects transient order-flow imbalance (the signal we'll try to detect),
    and a bid/ask spread that widens with realized volatility.

    Returns a DataFrame indexed by timestamp with the schema documented
    in the module docstring.
    """
    rng = np.random.default_rng(seed)
    spec = INSTRUMENTS[instrument]

    dt = bar_seconds / (6.5 * 3600)  # fraction of a trading day per bar (RTH-equivalent)
    sigma_bar = annual_vol * np.sqrt(dt / 252)

    # --- Stochastic volatility (GARCH(1,1)-lite) ---
    omega, alpha, beta = sigma_bar**2 * 0.05, 0.10, 0.85
    var = np.empty(n_bars)
    var[0] = sigma_bar**2
    shocks = rng.standard_normal(n_bars)
    for t in range(1, n_bars):
        var[t] = omega + alpha * (shocks[t - 1] ** 2) * var[t - 1] + beta * var[t - 1]
    sigma_t = np.sqrt(var)

    # --- Latent informed-flow regime (this is what our microstructure
    #     signals are trying to detect) ---
    informed_on = rng.random(n_bars) < informed_flow_prob
    informed_direction = rng.choice([-1, 1], size=n_bars)
    informed_impulse = informed_on * informed_direction * informed_flow_strength * sigma_t

    # --- Price path: base noise + informed drift component ---
    base_returns = shocks * sigma_t
    total_returns = base_returns + informed_impulse * sigma_t * 0.5
    log_price = np.log(4500.0 if instrument == "ES" else 1.08) + np.cumsum(total_returns)
    close = np.exp(log_price)

    # OHLC from a Brownian-bridge-style intra-bar path
    intra_noise = rng.standard_normal((n_bars, 2)) * sigma_t[:, None] * 0.5
    open_ = np.roll(close, 1)
    open_[0] = close[0]
    high = np.maximum(open_, close) + np.abs(intra_noise[:, 0]) * close
    low = np.minimum(open_, close) - np.abs(intra_noise[:, 1]) * close

    # --- Volume: baseline lognormal, elevated when informed flow is active ---
    base_vol = rng.lognormal(mean=np.log(150), sigma=0.5, size=n_bars)
    volume = base_vol * (1.0 + informed_on * 1.8)
    volume = np.round(volume).astype(int)

    # --- Signed volume proxy: correlated with the informed impulse plus noise ---
    flow_noise = rng.standard_normal(n_bars) * 0.4
    signed_ratio = np.tanh(informed_impulse * 2.0 + flow_noise)
    signed_volume = signed_ratio * volume

    # --- Spread widens with vol (liquidity thinning under stress) ---
    spread_ticks = spec.avg_spread_ticks * (1.0 + 3.0 * (sigma_t / sigma_t.mean() - 1).clip(min=0))
    spread = spread_ticks * spec.tick_size
    bid = close - spread / 2
    ask = close + spread / 2

    idx = pd.date_range(start=start, periods=n_bars, freq=f"{bar_seconds}s", tz="UTC")
    df = pd.DataFrame(
        {
            "open": open_, "high": high, "low": low, "close": close,
            "bid": bid, "ask": ask,
            "volume": volume, "signed_volume": signed_volume,
            "realized_sigma": sigma_t,  # exposed for diagnostics only, not a tradeable feature
        },
        index=idx,
    )
    df.index.name = "timestamp"
    return df


def load_ohlcv_csv(path: str, tz: str = "UTC") -> pd.DataFrame:
    """
    Load real OHLCV(+bid/ask/volume) data from CSV/parquet.
    Expects at minimum: timestamp, open, high, low, close, volume.
    bid/ask/signed_volume are optional; if absent they are estimated
    (bid/ask from close +/- half a synthetic spread; signed_volume from
    the tick-rule: sign(close - prior close) * volume) — but note this is
    a *weaker* proxy than real quote/trade data and should be replaced
    with venue data whenever available.
    """
    if path.endswith(".parquet"):
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path, parse_dates=["timestamp"])
    df = df.set_index("timestamp").sort_index()
    if df.index.tz is None:
        df.index = df.index.tz_localize(tz)

    if "bid" not in df.columns or "ask" not in df.columns:
        est_spread = (df["high"] - df["low"]).rolling(20, min_periods=1).mean() * 0.05
        df["bid"] = df["close"] - est_spread / 2
        df["ask"] = df["close"] + est_spread / 2

    if "signed_volume" not in df.columns:
        tick_sign = np.sign(df["close"].diff().fillna(0.0))
        tick_sign = tick_sign.replace(0, np.nan).ffill().fillna(0.0)
        df["signed_volume"] = tick_sign * df["volume"]

    required = {"open", "high", "low", "close", "bid", "ask", "volume", "signed_volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Loaded data missing required columns: {missing}")
    return df
