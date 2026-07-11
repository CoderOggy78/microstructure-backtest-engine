"""
risk.py
=======
Risk management layer: position sizing, stop-losses, and circuit breakers.
This module is deliberately kept separate from signal generation — the
signal says *direction/conviction*, this module says *how much and when
to cut it off*. That separation is what lets you swap signals without
ever touching risk controls, and vice versa.

Components
----------
1. VolTargetSizer      — size positions so each trade contributes a constant
                          ex-ante risk budget (inverse-volatility sizing),
                          the standard approach used by CTAs/vol-targeting funds.
2. ATRTrailingStop      — per-position stop that trails price by a multiple
                          of Average True Range; recalculated every bar,
                          only ever tightens in the trade's favor (ratchet).
3. CircuitBreaker       — account-level kill switches:
                          (a) daily loss limit -> flatten & halt for the session
                          (b) rolling max-drawdown limit -> de-lever to zero
                          (c) volatility spike breaker -> pause new entries
                              when realized vol jumps beyond a threshold
                              (protects against trading through flash-crash-like
                              regimes where microstructure signals become unreliable).
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass


@dataclass
class SizingParams:
    target_annual_vol: float = 0.10   # fraction of capital risked at target, annualized
    capital: float = 1_000_000.0
    max_leverage: float = 5.0
    min_contracts: int = 0
    max_contracts: int = 200


class VolTargetSizer:
    """
    Inverse-volatility position sizing:
        contracts = (target_daily_$vol) / (sigma_bar * price * point_value)
    scaled so that, ex-ante, each open position contributes roughly the same
    dollar risk regardless of the instrument's current volatility regime.
    This is the standard risk-parity-style approach — it prevents a strategy
    from unknowingly taking 5x the risk in a calm regime just because ATR
    happened to be small that week is not what we want either; sizing is
    proportional to *inverse* vol, so risk is roughly constant, not the size.
    """
    def __init__(self, spec, params: SizingParams | None = None):
        self.spec = spec
        self.p = params or SizingParams()

    def size(self, signal: pd.Series, sigma_bar: pd.Series, price: pd.Series,
             bars_per_day: float) -> pd.Series:
        target_bar_dollar_vol = self.p.capital * self.p.target_annual_vol / np.sqrt(252 * bars_per_day)
        dollar_vol_per_contract = (sigma_bar * price * self.spec.point_value).clip(lower=1e-8)
        raw_contracts = target_bar_dollar_vol / dollar_vol_per_contract
        capped = raw_contracts.clip(lower=self.p.min_contracts, upper=self.p.max_contracts)
        notional_cap = (self.p.capital * self.p.max_leverage) / (price * self.spec.point_value)
        capped = np.minimum(capped, notional_cap)
        return np.sign(signal) * np.floor(capped.abs())


@dataclass
class StopParams:
    atr_lookback: int = 20
    atr_multiple: float = 2.5


class ATRTrailingStop:
    """
    Ratcheting ATR-based trailing stop. Computed vectorized over the whole
    series for backtesting speed, but the ratchet logic (stop only ever
    moves in the position's favor) is applied causally bar-by-bar via a
    small forward pass — this is the one place a Python loop is justified,
    since the ratchet is path-dependent and can't be vectorized safely
    without risking subtle look-ahead bugs.
    """
    def __init__(self, params: StopParams | None = None):
        self.p = params or StopParams()

    @staticmethod
    def atr(high: pd.Series, low: pd.Series, close: pd.Series, lookback: int) -> pd.Series:
        prev_close = close.shift(1)
        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ], axis=1).max(axis=1)
        return tr.rolling(lookback, min_periods=lookback).mean()

    def compute_stops(self, high: pd.Series, low: pd.Series, close: pd.Series,
                       position: pd.Series) -> pd.Series:
        """
        position: the *intended* signed position at each bar (already lagged
        by the caller). Returns the trailing stop LEVEL at each bar for
        whatever position is open; the backtest engine checks whether
        low/high breached this level intrabar to force an exit.
        """
        atr = self.atr(high, low, close, self.p.atr_lookback).bfill()
        stop = pd.Series(index=close.index, dtype=float)
        cur_stop = np.nan
        cur_side = 0
        for i in range(len(close)):
            side = np.sign(position.iloc[i])
            if side == 0:
                cur_stop = np.nan
                cur_side = 0
            elif side != cur_side:
                # new position opened this bar -> initialize stop
                cur_stop = close.iloc[i] - side * self.p.atr_multiple * atr.iloc[i]
                cur_side = side
            else:
                candidate = close.iloc[i] - side * self.p.atr_multiple * atr.iloc[i]
                # ratchet: only tighten (move stop in the position's favor)
                cur_stop = max(cur_stop, candidate) if side > 0 else min(cur_stop, candidate)
            stop.iloc[i] = cur_stop
        return stop


@dataclass
class CircuitBreakerParams:
    daily_loss_limit_pct: float = 0.02     # halt trading for the session
    max_drawdown_limit_pct: float = 0.10   # de-lever to zero
    vol_spike_zscore: float = 3.0          # pause new entries
    vol_spike_lookback: int = 500


class CircuitBreaker:
    """
    Account-level kill switches applied on top of whatever the signal wants.
    These are the automated stop-losses / halts referenced in institutional
    risk frameworks: even a perfect signal must be overridable by capital
    preservation rules.
    """
    def __init__(self, params: CircuitBreakerParams | None = None):
        self.p = params or CircuitBreakerParams()

    def apply(self, equity: pd.Series, sigma_bar: pd.Series,
              trading_day: pd.Series) -> pd.DataFrame:
        """
        Returns boolean masks (all computed causally, no look-ahead):
          - halted_daily: session-loss breaker tripped (no new risk rest of day)
          - delever_dd:   rolling drawdown breaker tripped (position scaled to 0)
          - vol_pause:    realized-vol spike breaker (no *new* entries, existing
                          positions may still be closed/stopped out)
        """
        day_start_equity = equity.groupby(trading_day).transform("first")
        intraday_ret = equity / day_start_equity - 1.0
        halted_daily = intraday_ret <= -self.p.daily_loss_limit_pct

        running_max = equity.cummax()
        dd = equity / running_max - 1.0
        delever_dd = dd <= -self.p.max_drawdown_limit_pct

        vol_z = (sigma_bar - sigma_bar.rolling(self.p.vol_spike_lookback, min_periods=50).mean()) / \
                sigma_bar.rolling(self.p.vol_spike_lookback, min_periods=50).std()
        vol_pause = vol_z >= self.p.vol_spike_zscore

        return pd.DataFrame({
            "halted_daily": halted_daily.fillna(False),
            "delever_dd": delever_dd.fillna(False),
            "vol_pause": vol_pause.fillna(False),
        })
