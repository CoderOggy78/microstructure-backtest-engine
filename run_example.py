"""
run_example.py
===============
End-to-end demonstration of the framework:

  1. Generate synthetic ES futures microstructure data (swap for
     load_ohlcv_csv(...) / your own feed when you have one).
  2. Build the microstructure alpha (OFI + VPIN blend).
  3. BEFORE: run a naive backtest — no vol-target sizing (fixed size),
     no stop-loss, no circuit breakers, no market impact costs (spread only).
  4. AFTER: run the full risk-managed backtest — vol-target sizing, ATR
     trailing stops, circuit breakers, full cost model.
  5. Run walk-forward optimization over the signal parameter grid and
     report in-sample vs out-of-sample Sharpe (the honest number).

Run with:  python3 run_example.py
"""
import numpy as np
import pandas as pd

from quant_framework import (
    generate_synthetic_futures_data, INSTRUMENTS,
    TransactionCostModel, CostModelParams,
    Backtester, SizingParams, StopParams, CircuitBreakerParams,
    combined_alpha, walk_forward, ParamGrid,
)

pd.set_option("display.width", 120)


def naive_backtest(market: pd.DataFrame, alpha: pd.Series, spec, fixed_contracts: int = 5,
                    entry_threshold: float = 0.15):
    """
    A deliberately naive baseline for comparison:
      - Fixed position size regardless of volatility regime.
      - No stop-loss (ride the position until the signal flips).
      - No circuit breakers.
      - Cost = spread only (no market impact, no commission) — this is the
        kind of backtest that looks great and is NOT institutional-grade,
        which is exactly why we show the delta.
    """
    lagged = alpha.shift(1)
    direction = np.sign(lagged.where(lagged.abs() >= entry_threshold, 0.0))
    position = direction * fixed_contracts

    close = market["close"].values
    bid = market["bid"].values
    ask = market["ask"].values
    pos = position.values
    n = len(market)

    pnl = np.zeros(n)
    cost = np.zeros(n)
    for t in range(1, n):
        pnl[t] = pos[t - 1] * (close[t] - close[t - 1]) * spec.point_value
        order = pos[t] - pos[t - 1]
        if order != 0:
            spread = max(ask[t] - bid[t], 0.01)
            cost[t] = 0.5 * spread * abs(order) * spec.point_value  # spread only, no impact/commission

    equity = 1_000_000.0 + np.cumsum(pnl - cost)
    df = market.copy()
    df["position"] = pos
    df["pnl_gross"] = pnl
    df["cost"] = cost
    df["pnl_net"] = pnl - cost
    df["equity"] = equity
    df["returns_net"] = pd.Series(equity, index=market.index).pct_change().fillna(0.0)

    from quant_framework.metrics import build_report
    from quant_framework.backtest import _extract_trade_pnls
    trade_pnls = _extract_trade_pnls(df)
    report = build_report(df["returns_net"], df["equity"], df["position"], trade_pnls,
                           float(pnl.sum()), float(cost.sum()))
    return df, report


def main():
    spec = INSTRUMENTS["ES"]

    print("=" * 70)
    print("1) GENERATING SYNTHETIC ES FUTURES DATA (30s bars, ~1 week of RTH)")
    print("=" * 70)
    market = generate_synthetic_futures_data(n_bars=15_000, bar_seconds=30, instrument="ES", seed=7)
    print(market.head(3))
    print(f"... {len(market)} bars total\n")

    print("=" * 70)
    print("2) BUILDING MICROSTRUCTURE ALPHA (OFI + VPIN blend)")
    print("=" * 70)
    alpha_df = combined_alpha(market, ofi_lookback=20, vpin_lookback=50)
    print(alpha_df[["ofi", "vpin", "alpha"]].dropna().head(3))
    print()

    print("=" * 70)
    print("3) BEFORE — naive backtest (fixed size, no stops, spread-only costs)")
    print("=" * 70)
    naive_df, naive_report = naive_backtest(market, alpha_df["alpha"], spec)
    print(naive_report)

    print("=" * 70)
    print("4) AFTER — risk-managed backtest (vol-target sizing, ATR trailing")
    print("   stop, circuit breakers, full spread+impact+commission costs)")
    print("=" * 70)
    bt = Backtester(
        spec,
        sizing_params=SizingParams(target_annual_vol=0.10, capital=1_000_000.0, max_contracts=200),
        stop_params=StopParams(atr_lookback=20, atr_multiple=2.5),
        breaker_params=CircuitBreakerParams(daily_loss_limit_pct=0.02, max_drawdown_limit_pct=0.10),
    )
    result = bt.run(market, alpha_df["alpha"], entry_threshold=0.15)
    print(result.report)
    print("Cost breakdown ($):")
    print(result.cost_breakdown.round(0))
    print()

    print("=" * 70)
    print("BEFORE vs AFTER COMPARISON")
    print("=" * 70)
    comp = pd.DataFrame({
        "Naive (before)": {
            "Sharpe": naive_report.sharpe, "Sortino": naive_report.sortino,
            "Max DD": naive_report.max_dd, "Net PnL": naive_report.net_pnl,
            "Cost Drag %": naive_report.cost_drag_pct,
        },
        "Risk-managed (after)": {
            "Sharpe": result.report.sharpe, "Sortino": result.report.sortino,
            "Max DD": result.report.max_dd, "Net PnL": result.report.net_pnl,
            "Cost Drag %": result.report.cost_drag_pct,
        },
    })
    print(comp.round(4))
    print()

    print("=" * 70)
    print("5) WALK-FORWARD OPTIMIZATION (data-snooping check)")
    print("=" * 70)
    grid = ParamGrid(ofi_lookback=[10, 20], vpin_lookback=[30, 50], entry_threshold=[0.10, 0.15])
    wf = walk_forward(market, spec, grid, train_bars=4000, test_bars=1500, step_bars=1500)
    for i, (is_s, oos_s, params) in enumerate(zip(wf["is_sharpes"], wf["oos_sharpes"], wf["chosen_params"])):
        print(f"Window {i}: params={params} | in-sample Sharpe={is_s:.2f} | out-of-sample Sharpe={oos_s:.2f}")

    if len(wf["oos_frame"]):
        oos_returns = wf["oos_frame"]["returns_net"]
        oos_equity = wf["oos_frame"]["equity"]
        from quant_framework.metrics import sharpe_ratio, _bars_per_year
        bpy = _bars_per_year(oos_equity.index)
        print(f"\nConcatenated OUT-OF-SAMPLE Sharpe (the number to trust): "
              f"{sharpe_ratio(oos_returns, bpy):.3f}")

    print("\n" + "=" * 70)
    print("IMPORTANT CAVEAT ON THESE NUMBERS")
    print("=" * 70)
    print(
        "The Sharpe ratios above (double/triple digits) are NOT realistic and\n"
        "should not be read as 'this strategy works'. The synthetic generator\n"
        "in data.py deliberately embeds a detectable informed-flow impulse\n"
        "directly into both returns AND signed volume, so OFI/VPIN trivially\n"
        "'detect' a signal that was hand-planted — real markets are far more\n"
        "adversarial and this edge will not survive contact with live flow.\n"
        "This script exists to prove the ENGINE is wired correctly (no look-\n"
        "ahead, costs are deducted, stops/breakers fire, walk-forward isolates\n"
        "OOS data) — not to claim an edge. Swap in real tick/quote data via\n"
        "load_ohlcv_csv() before drawing any conclusions about alpha."
    )


if __name__ == "__main__":
    main()
