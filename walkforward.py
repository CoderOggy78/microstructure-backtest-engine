"""
walkforward.py
==============
Walk-forward optimization: the standard defense against data snooping /
overfitting in strategy research.

Procedure
---------
For each rolling window:
  1. TRAIN on an in-sample slice: grid-search signal/sizing parameters,
     selecting the combination that maximizes NET-of-cost Sharpe.
  2. TEST on the immediately following out-of-sample slice using the
     parameters chosen in step 1 — NEVER re-optimized on test data.
  3. Slide the window forward and repeat.

The final reported performance is the CONCATENATION of only the
out-of-sample segments. This number is what should be trusted; the
in-sample numbers are shown separately purely for diagnosing overfitting
(a large in-sample vs out-of-sample gap means the parameter surface is
overfit to noise).
"""

from __future__ import annotations
import itertools
import numpy as np
import pandas as pd
from dataclasses import dataclass, field

from .signals import combined_alpha
from .backtest import Backtester


@dataclass
class ParamGrid:
    ofi_lookback: list = field(default_factory=lambda: [10, 20, 40])
    vpin_lookback: list = field(default_factory=lambda: [30, 50, 100])
    entry_threshold: list = field(default_factory=lambda: [0.10, 0.15, 0.25])

    def combinations(self):
        keys = ["ofi_lookback", "vpin_lookback", "entry_threshold"]
        for combo in itertools.product(self.ofi_lookback, self.vpin_lookback, self.entry_threshold):
            yield dict(zip(keys, combo))


def walk_forward(market: pd.DataFrame, spec, grid: ParamGrid,
                  train_bars: int, test_bars: int, step_bars: int | None = None) -> dict:
    """
    Returns a dict with:
      - 'oos_frame': concatenated out-of-sample bar ledger (what to trust)
      - 'is_sharpes': in-sample Sharpe achieved by the chosen params, per window
      - 'oos_sharpes': out-of-sample Sharpe achieved, per window (compare to above)
      - 'chosen_params': list of the winning param dict per window
    """
    step_bars = step_bars or test_bars
    n = len(market)
    bt = Backtester(spec)

    oos_frames = []
    is_sharpes, oos_sharpes, chosen_params = [], [], []

    start = 0
    while start + train_bars + test_bars <= n:
        train_slice = market.iloc[start: start + train_bars]
        test_slice = market.iloc[start + train_bars: start + train_bars + test_bars]

        best_sharpe, best_params = -np.inf, None
        for params in grid.combinations():
            alpha_df = combined_alpha(train_slice, ofi_lookback=params["ofi_lookback"],
                                       vpin_lookback=params["vpin_lookback"])
            result = bt.run(train_slice, alpha_df["alpha"], entry_threshold=params["entry_threshold"])
            sharpe = result.report.sharpe
            if sharpe > best_sharpe:
                best_sharpe, best_params = sharpe, params

        # --- Evaluate the WINNING params strictly out-of-sample ---
        # Alpha for the test slice is computed over train+test combined so
        # rolling windows have real lookback history at the test boundary,
        # then we slice out only the test portion. Parameter SELECTION
        # (the loop above) never sees test_slice; only this final scoring
        # step touches it, and only once, with parameters already fixed.
        combined_slice = market.iloc[start: start + train_bars + test_bars]
        alpha_full = combined_alpha(combined_slice, ofi_lookback=best_params["ofi_lookback"],
                                     vpin_lookback=best_params["vpin_lookback"])
        test_alpha = alpha_full["alpha"].iloc[train_bars:]
        test_result = bt.run(test_slice, test_alpha, entry_threshold=best_params["entry_threshold"])

        oos_frames.append(test_result.frame)
        is_sharpes.append(best_sharpe)
        oos_sharpes.append(test_result.report.sharpe)
        chosen_params.append(best_params)

        start += step_bars

    oos_frame = pd.concat(oos_frames) if oos_frames else pd.DataFrame()
    return {
        "oos_frame": oos_frame,
        "is_sharpes": is_sharpes,
        "oos_sharpes": oos_sharpes,
        "chosen_params": chosen_params,
    }
