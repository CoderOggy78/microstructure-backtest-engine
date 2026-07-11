"""
costs.py
========
Transaction cost model for futures/FX execution.

Every backtest metric downstream (Sharpe, Sortino, drawdown) is computed
NET of the costs modeled here. Three additive components, all standard in
institutional TCA (transaction cost analysis):

1. Spread cost      — pay half the quoted bid/ask spread crossing to trade.
2. Market impact     — square-root model (Almgren-Chriss / Kyle-style):
                       impact ∝ sigma * sqrt(participation_rate)
                       This is the standard institutional functional form:
                       impact grows sub-linearly with size but accelerates
                       with volatility and with your share of bar volume.
3. Commission/fees   — fixed $ per contract/lot (exchange + broker fees).

Slippage here is *not* a separate flat fudge factor — it emerges naturally
as spread + impact, which is more defensible than an arbitrary bps haircut.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass


@dataclass
class CostModelParams:
    commission_per_contract: float = 2.25   # $ round-turn/2, i.e. per side
    impact_coefficient: float = 0.10        # k in impact = k * sigma * sqrt(participation)
    max_participation_rate: float = 0.20    # risk control: cap our share of bar volume
    min_spread_ticks: float = 0.5           # floor to avoid unrealistically free fills


class TransactionCostModel:
    def __init__(self, spec, params: CostModelParams | None = None):
        self.spec = spec
        self.p = params or CostModelParams()

    def participation_rate(self, order_size: np.ndarray, bar_volume: np.ndarray) -> np.ndarray:
        """Fraction of bar volume our order represents. Capped as a risk control —
        orders sized beyond max_participation_rate should be split upstream by the
        position-sizing module, not silently absorbed here."""
        bar_volume = np.maximum(bar_volume, 1)
        rate = np.abs(order_size) / bar_volume
        return np.minimum(rate, self.p.max_participation_rate)

    def spread_cost(self, order_size: np.ndarray, bid: np.ndarray, ask: np.ndarray) -> np.ndarray:
        """$ cost from crossing the spread, per trade (half-spread * |size| * point_value)."""
        spread = np.maximum(ask - bid, self.p.min_spread_ticks * self.spec.tick_size)
        return 0.5 * spread * np.abs(order_size) * self.spec.point_value

    def impact_cost(self, order_size: np.ndarray, sigma_bar: np.ndarray, bar_volume: np.ndarray,
                     price: np.ndarray) -> np.ndarray:
        """
        Square-root market impact cost in $:
            impact_price_frac = k * sigma_bar * sqrt(participation_rate)
            impact_$ = impact_price_frac * price * |size| * point_value / price
                     = impact_price_frac * price_units_moved... simplified below.
        We express sigma_bar as a *return* volatility (fractional), so the
        price impact per unit is k * sigma_bar * sqrt(participation) * price.
        """
        participation = self.participation_rate(order_size, bar_volume)
        impact_frac = self.p.impact_coefficient * sigma_bar * np.sqrt(participation)
        impact_price_move = impact_frac * price          # price units moved against us
        return impact_price_move * np.abs(order_size) * self.spec.point_value

    def commission_cost(self, order_size: np.ndarray) -> np.ndarray:
        return self.p.commission_per_contract * np.abs(order_size)

    def total_cost(self, order_size: np.ndarray, bid: np.ndarray, ask: np.ndarray,
                   sigma_bar: np.ndarray, bar_volume: np.ndarray, price: np.ndarray) -> pd.DataFrame:
        """
        Returns a DataFrame breaking down $ cost by component plus the total,
        so the backtest report can show a clean cost attribution (useful for
        deciding whether a strategy dies from spread, impact, or fees).
        """
        sc = self.spread_cost(order_size, bid, ask)
        ic = self.impact_cost(order_size, sigma_bar, bar_volume, price)
        cc = self.commission_cost(order_size)
        return pd.DataFrame({
            "spread_cost": sc,
            "impact_cost": ic,
            "commission_cost": cc,
            "total_cost": sc + ic + cc,
        })
