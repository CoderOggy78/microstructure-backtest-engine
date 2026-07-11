"""
signals.py
==========
Market microstructure signal library.

All signals here are computed as strictly causal rolling functions of past
bars only. IMPORTANT — none of these are pre-lagged; the backtest engine
(backtest.py) is responsible for shifting the final combined signal forward
by exactly one bar before it can affect any order, which is the single
choke point where look-ahead bias is prevented. Keeping that logic in one
place (rather than scattered across every signal) makes it auditable.

Signals implemented
--------------------
1. Order Flow Imbalance (OFI)   — rolling sum of signed volume, normalized
                                   by rolling volume. Captures short-horizon
                                   buy/sell pressure, the workhorse HFT signal.
2. VPIN proxy                    — Volume-Synchronized Probability of
                                   Informed Trading (Easley/Lopez de Prado
                                   style), estimated from equal-volume buckets
                                   of signed flow. High VPIN = flow is
                                   unusually one-sided = elevated informed-
                                   trading probability = expect continuation
                                   or imminent volatility.
3. Kyle's lambda                 — rolling regression of price change on
                                   signed order flow; estimates the market's
                                   current price-impact-per-unit-flow
                                   ("illiquidity"). Rising lambda = thinner
                                   liquidity = signals should be sized down.
4. Combined alpha                — z-scored blend of OFI and VPIN, the
                                   tradeable signal fed to the backtest engine.
"""

from __future__ import annotations
import numpy as np
import pandas as pd


def order_flow_imbalance(signed_volume: pd.Series, volume: pd.Series, lookback: int = 20) -> pd.Series:
    num = signed_volume.rolling(lookback, min_periods=lookback).sum()
    den = volume.rolling(lookback, min_periods=lookback).sum().clip(lower=1)
    return (num / den).rename("ofi")


def vpin_proxy(signed_volume: pd.Series, volume: pd.Series, bucket_volume: float | None = None,
               n_buckets_lookback: int = 50) -> pd.Series:
    """
    Approximate VPIN using a rolling-window analogue of equal-volume bucketing:
    instead of resampling into true volume buckets (which changes the index),
    we compute |sum(signed_volume)| / sum(volume) over a rolling window sized
    to contain roughly `n_buckets_lookback` bars — this trades a bit of
    theoretical purity for a result aligned to the same timestamp index as
    every other signal, which is what the vectorized engine needs.
    """
    num = signed_volume.rolling(n_buckets_lookback, min_periods=n_buckets_lookback).sum().abs()
    den = volume.rolling(n_buckets_lookback, min_periods=n_buckets_lookback).sum().clip(lower=1)
    return (num / den).rename("vpin")


def kyle_lambda(price: pd.Series, signed_volume: pd.Series, lookback: int = 50) -> pd.Series:
    """
    Rolling OLS slope of price change (dP) on signed order flow (dV):
        dP_t = lambda * signed_volume_t + eps_t
    Implemented via rolling covariance/variance (closed-form OLS slope),
    which is exact and far faster than looping a regression per bar.
    """
    dp = price.diff()
    cov = dp.rolling(lookback, min_periods=lookback).cov(signed_volume)
    var = signed_volume.rolling(lookback, min_periods=lookback).var().clip(lower=1e-12)
    return (cov / var).rename("kyle_lambda")


def combined_alpha(df: pd.DataFrame, ofi_lookback: int = 20, vpin_lookback: int = 50,
                    z_lookback: int = 500, ofi_weight: float = 0.6, vpin_weight: float = 0.4) -> pd.DataFrame:
    """
    Build the final tradeable alpha:
      1. Compute OFI and VPIN.
      2. Z-score each over a rolling window (so the signal is regime-adaptive
         and comparable across changing volatility/liquidity conditions).
      3. VPIN signals *conviction/magnitude of an imbalance*, not direction,
         so we sign it by the concurrent OFI direction before blending.
      4. Blend into a single continuous alpha in roughly [-1, 1] (via tanh),
         which the backtest engine converts into a target position.

    Returns a DataFrame with the raw components plus the final 'alpha' column.
    All of it is causal as of the bar's close; the engine lags it by 1 bar.
    """
    ofi = order_flow_imbalance(df["signed_volume"], df["volume"], ofi_lookback)
    vpin = vpin_proxy(df["signed_volume"], df["volume"], n_buckets_lookback=vpin_lookback)

    ofi_z = (ofi - ofi.rolling(z_lookback, min_periods=50).mean()) / ofi.rolling(z_lookback, min_periods=50).std()
    vpin_z = (vpin - vpin.rolling(z_lookback, min_periods=50).mean()) / vpin.rolling(z_lookback, min_periods=50).std()

    vpin_signed = vpin_z * np.sign(ofi_z.fillna(0))
    raw = ofi_weight * ofi_z.fillna(0) + vpin_weight * vpin_signed.fillna(0)
    alpha = np.tanh(raw / 3.0)  # squash to (-1, 1), 3.0 chosen so +-3 sigma -> near +-1

    out = pd.DataFrame({
        "ofi": ofi, "vpin": vpin,
        "ofi_z": ofi_z, "vpin_z": vpin_z,
        "alpha": alpha,
    })
    return out
