"""
metrics.py
==========
Institutional-standard performance metrics, all computed on NET-of-cost
returns unless explicitly labeled 'gross'. Annualization uses the actual
bars-per-year implied by the data's bar frequency, not a hardcoded 252 —
critical for sub-minute bars where naive daily-style annualization wildly
overstates Sharpe.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass


def _bars_per_year(index: pd.DatetimeIndex) -> float:
    freq_seconds = (index[1:] - index[:-1]).median().total_seconds()
    seconds_per_year = 252 * 6.5 * 3600  # RTH-equivalent trading year
    return seconds_per_year / freq_seconds


def sharpe_ratio(returns: pd.Series, bars_per_year: float, risk_free: float = 0.0) -> float:
    excess = returns - risk_free / bars_per_year
    if excess.std(ddof=1) == 0 or len(excess) < 2:
        return 0.0
    return float(np.sqrt(bars_per_year) * excess.mean() / excess.std(ddof=1))


def sortino_ratio(returns: pd.Series, bars_per_year: float, risk_free: float = 0.0) -> float:
    excess = returns - risk_free / bars_per_year
    downside = excess[excess < 0]
    dd_std = downside.std(ddof=1)
    if dd_std == 0 or np.isnan(dd_std) or len(excess) < 2:
        return 0.0
    return float(np.sqrt(bars_per_year) * excess.mean() / dd_std)


def max_drawdown(equity: pd.Series) -> tuple[float, pd.Timestamp, pd.Timestamp]:
    running_max = equity.cummax()
    dd = equity / running_max - 1.0
    trough = dd.idxmin()
    peak = equity.loc[:trough].idxmax()
    return float(dd.min()), peak, trough


def calmar_ratio(returns: pd.Series, equity: pd.Series, bars_per_year: float) -> float:
    ann_return = (1 + returns.mean()) ** bars_per_year - 1
    mdd, _, _ = max_drawdown(equity)
    if mdd == 0:
        return 0.0
    return float(ann_return / abs(mdd))


def turnover(position: pd.Series) -> float:
    """Average absolute change in position per bar — proxy for trading intensity/cost drag."""
    return float(position.diff().abs().mean())


def hit_rate(trade_pnls: pd.Series) -> float:
    if len(trade_pnls) == 0:
        return float("nan")
    return float((trade_pnls > 0).mean())


@dataclass
class PerformanceReport:
    sharpe: float
    sortino: float
    calmar: float
    max_dd: float
    max_dd_peak: pd.Timestamp
    max_dd_trough: pd.Timestamp
    ann_return: float
    ann_vol: float
    turnover: float
    hit_rate: float
    total_cost: float
    gross_pnl: float
    net_pnl: float
    cost_drag_pct: float  # what fraction of gross PnL was eaten by costs

    def __str__(self) -> str:
        return (
            f"{'Metric':<22}{'Value':>14}\n" + "-" * 36 + "\n"
            f"{'Sharpe':<22}{self.sharpe:>14.3f}\n"
            f"{'Sortino':<22}{self.sortino:>14.3f}\n"
            f"{'Calmar':<22}{self.calmar:>14.3f}\n"
            f"{'Max Drawdown':<22}{self.max_dd:>14.2%}\n"
            f"{'Ann. Return':<22}{self.ann_return:>14.2%}\n"
            f"{'Ann. Vol':<22}{self.ann_vol:>14.2%}\n"
            f"{'Turnover/bar':<22}{self.turnover:>14.3f}\n"
            f"{'Hit Rate':<22}{self.hit_rate:>14.2%}\n"
            f"{'Gross PnL':<22}{self.gross_pnl:>14,.0f}\n"
            f"{'Total Costs':<22}{self.total_cost:>14,.0f}\n"
            f"{'Net PnL':<22}{self.net_pnl:>14,.0f}\n"
            f"{'Cost Drag %':<22}{self.cost_drag_pct:>14.1%}\n"
        )


def build_report(net_returns: pd.Series, equity: pd.Series, position: pd.Series,
                  trade_pnls: pd.Series, gross_pnl: float, total_cost: float) -> PerformanceReport:
    bpy = _bars_per_year(equity.index)
    mdd, peak, trough = max_drawdown(equity)
    net_pnl = gross_pnl - total_cost
    return PerformanceReport(
        sharpe=sharpe_ratio(net_returns, bpy),
        sortino=sortino_ratio(net_returns, bpy),
        calmar=calmar_ratio(net_returns, equity, bpy),
        max_dd=mdd, max_dd_peak=peak, max_dd_trough=trough,
        ann_return=float((1 + net_returns.mean()) ** bpy - 1),
        ann_vol=float(net_returns.std(ddof=1) * np.sqrt(bpy)),
        turnover=turnover(position),
        hit_rate=hit_rate(trade_pnls),
        total_cost=total_cost,
        gross_pnl=gross_pnl,
        net_pnl=net_pnl,
        cost_drag_pct=float(total_cost / gross_pnl) if gross_pnl != 0 else float("nan"),
    )
