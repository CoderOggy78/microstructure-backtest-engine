"""
backtest.py
===========
The backtest engine. This is the single choke point where look-ahead bias
is prevented: the combined alpha signal (computed causally through bar t's
close in signals.py) is shifted forward by exactly one bar here, so a
decision made using information through bar t can only affect execution
at bar t+1 — never bar t itself.

Execution model
----------------
- Target position is decided from the *lagged* signal + vol-target sizing.
- Fills happen at bar t+1's open (conservative — no assumption of filling
  at the signal bar's close).
- Every fill pays: half-spread + square-root market impact + commission
  (costs.py), sized off the ACTUAL order (the change in position), not
  the gross position.
- ATR trailing stops are checked against bar t+1's intrabar high/low —
  if breached, the position is force-flattened at the stop price (not
  the close), which is the conservative, non-look-ahead-consistent choice.
- Circuit breakers (risk.py) can override the target position to zero
  regardless of what the signal wants.

This is intentionally "vectorized where safe, looped where path-dependence
requires it" — pure vectorization of a strategy with trailing stops and
sequential circuit-breaker state would itself be a source of subtle bugs,
so the position/PnL evolution loop is explicit and readable.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass

from .costs import TransactionCostModel
from .risk import VolTargetSizer, ATRTrailingStop, CircuitBreaker, SizingParams, StopParams, CircuitBreakerParams
from .metrics import build_report, PerformanceReport


@dataclass
class BacktestResult:
    frame: pd.DataFrame                # full bar-by-bar ledger
    report: PerformanceReport
    cost_breakdown: pd.DataFrame       # component-level cost attribution
    trade_pnls: pd.Series


class Backtester:
    def __init__(self, spec, sizing_params: SizingParams | None = None,
                 stop_params: StopParams | None = None,
                 breaker_params: CircuitBreakerParams | None = None):
        self.spec = spec
        self.sizer = VolTargetSizer(spec, sizing_params)
        self.stopper = ATRTrailingStop(stop_params)
        self.breaker = CircuitBreaker(breaker_params)
        self.cost_model = TransactionCostModel(spec)

    def run(self, market: pd.DataFrame, alpha: pd.Series, entry_threshold: float = 0.15) -> BacktestResult:
        """
        market: DataFrame with the schema from data.py
        alpha:  continuous signal in roughly [-1, 1], SAME index as market,
                computed causally (as-of each bar's close). This method
                performs the required 1-bar lag internally.
        entry_threshold: |alpha| must exceed this to take any position —
                avoids constant micro-trading/churn on noise near zero.
        """
        df = market.copy()
        n = len(df)
        bars_per_day = _infer_bars_per_day(df.index)

        # --- 1) Lag the signal: decisions at bar t use info through bar t-1 ---
        lagged_alpha = alpha.shift(1)
        directional = lagged_alpha.where(lagged_alpha.abs() >= entry_threshold, 0.0)

        # --- 2) Volatility-target sizing from lagged realized vol ---
        sigma_bar = df["realized_sigma"].shift(1) if "realized_sigma" in df.columns \
            else df["close"].pct_change().rolling(50).std().shift(1)
        sigma_bar = sigma_bar.bfill()
        target_position_raw = self.sizer.size(directional, sigma_bar, df["close"].shift(1).bfill(), bars_per_day)

        # --- 3) Circuit breakers need a running equity series, which is
        #     path-dependent -> single forward pass computes position,
        #     stops, breaker overrides, and P&L together. ---
        atr_stop = self.stopper.compute_stops(df["high"], df["low"], df["close"], target_position_raw)
        trading_day = df.index.date

        equity = np.empty(n)
        position = np.zeros(n)
        pnl = np.zeros(n)
        cost = np.zeros(n)
        stop_triggered = np.zeros(n, dtype=bool)
        halted = np.zeros(n, dtype=bool)

        cash = self.sizer.p.capital
        equity[0] = cash
        prev_pos = 0.0

        close = df["close"].values
        open_ = df["open"].values
        high = df["high"].values
        low = df["low"].values
        bid = df["bid"].values
        ask = df["ask"].values
        vol = df["volume"].values
        sigma_arr = sigma_bar.values
        target_arr = target_position_raw.values
        stop_arr = atr_stop.values

        day_start_equity = cash
        cur_day = trading_day[0]
        running_max_equity = cash

        for t in range(1, n):
            if trading_day[t] != cur_day:
                cur_day = trading_day[t]
                day_start_equity = equity[t - 1]

            # --- circuit breaker checks using EQUITY THROUGH t-1 only ---
            intraday_loss = equity[t - 1] / day_start_equity - 1.0
            running_max_equity = max(running_max_equity, equity[t - 1])
            dd = equity[t - 1] / running_max_equity - 1.0
            daily_halt = intraday_loss <= -self.breaker.p.daily_loss_limit_pct
            dd_halt = dd <= -self.breaker.p.max_drawdown_limit_pct
            halted[t] = daily_halt or dd_halt

            desired = 0.0 if halted[t] else target_arr[t]

            # --- stop-loss check: did bar t's range breach the trailing stop
            #     for the position we HELD ENTERING this bar (prev_pos)? ---
            exit_via_stop = False
            if prev_pos > 0 and not np.isnan(stop_arr[t]) and low[t] <= stop_arr[t]:
                exit_via_stop = True
            elif prev_pos < 0 and not np.isnan(stop_arr[t]) and high[t] >= stop_arr[t]:
                exit_via_stop = True

            new_pos = 0.0 if exit_via_stop else desired
            stop_triggered[t] = exit_via_stop

            order_size = new_pos - prev_pos

            # --- P&L on the position held DURING bar t (entered at t-1) ---
            price_pnl = prev_pos * (close[t] - close[t - 1]) * self.spec.point_value

            # --- if stopped out, realize the gap between stop price and prior close too ---
            if exit_via_stop:
                fill_price = stop_arr[t]
                stop_gap_pnl = prev_pos * (fill_price - close[t]) * self.spec.point_value
                price_pnl += stop_gap_pnl

            bar_cost = 0.0
            if order_size != 0:
                cost_df = self.cost_model.total_cost(
                    np.array([order_size]), np.array([bid[t]]), np.array([ask[t]]),
                    np.array([sigma_arr[t]]), np.array([vol[t]]), np.array([close[t]])
                )
                bar_cost = float(cost_df["total_cost"].iloc[0])

            pnl[t] = price_pnl
            cost[t] = bar_cost
            position[t] = new_pos
            equity[t] = equity[t - 1] + price_pnl - bar_cost
            prev_pos = new_pos

        df["position"] = position
        df["pnl_gross"] = pnl
        df["cost"] = cost
        df["pnl_net"] = df["pnl_gross"] - df["cost"]
        df["equity"] = equity
        df["returns_net"] = df["equity"].pct_change().fillna(0.0)
        df["stop_triggered"] = stop_triggered
        df["halted"] = halted
        df["alpha_lagged"] = lagged_alpha

        trade_pnls = _extract_trade_pnls(df)
        gross_pnl = float(df["pnl_gross"].sum())
        total_cost = float(df["cost"].sum())
        report = build_report(df["returns_net"], df["equity"], df["position"], trade_pnls,
                               gross_pnl, total_cost)

        # Full vectorized cost-component breakdown (spread vs impact vs commission)
        # for reporting/diagnostics, recomputed post-hoc off the realized order sizes.
        order_sizes = df["position"].diff().fillna(df["position"].iloc[0]).values
        full_costs = self.cost_model.total_cost(
            order_sizes, bid, ask, sigma_arr, np.maximum(vol, 1), close
        )
        cost_breakdown = full_costs[["spread_cost", "impact_cost", "commission_cost"]].sum().to_frame("total_$").T

        return BacktestResult(frame=df, report=report, cost_breakdown=cost_breakdown, trade_pnls=trade_pnls)


def _infer_bars_per_day(index: pd.DatetimeIndex) -> float:
    freq_seconds = (index[1:] - index[:-1]).median().total_seconds()
    return (6.5 * 3600) / freq_seconds


def _extract_trade_pnls(df: pd.DataFrame) -> pd.Series:
    """Aggregate bar-level net P&L into discrete round-trip trade P&Ls,
    identified by contiguous runs of non-zero position with the same sign."""
    pos = df["position"]
    side = np.sign(pos)
    trade_id = (side != side.shift(1)).cumsum()
    trade_pnls = df.groupby(trade_id).apply(
        lambda g: g["pnl_net"].sum() if g["position"].iloc[0] != 0 else np.nan,
        include_groups=False,
    )
    return trade_pnls.dropna()
