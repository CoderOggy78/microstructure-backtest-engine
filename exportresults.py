import json
import numpy as np
import pandas as pd
from quant_framework import (
    generate_synthetic_futures_data, INSTRUMENTS,
    Backtester, SizingParams, StopParams, CircuitBreakerParams,
    combined_alpha, walk_forward, ParamGrid,
)
from quant_framework.metrics import sharpe_ratio, _bars_per_year
from run_example import naive_backtest

spec = INSTRUMENTS["ES"]
market = generate_synthetic_futures_data(n_bars=15_000, bar_seconds=30, instrument="ES", seed=7)
alpha_df = combined_alpha(market, ofi_lookback=20, vpin_lookback=50)

naive_df, naive_report = naive_backtest(market, alpha_df["alpha"], spec)

bt = Backtester(
    spec,
    sizing_params=SizingParams(target_annual_vol=0.10, capital=1_000_000.0, max_contracts=200),
    stop_params=StopParams(atr_lookback=20, atr_multiple=2.5),
    breaker_params=CircuitBreakerParams(daily_loss_limit_pct=0.02, max_drawdown_limit_pct=0.10),
)
result = bt.run(market, alpha_df["alpha"], entry_threshold=0.15)

grid = ParamGrid(ofi_lookback=[10, 20], vpin_lookback=[30, 50], entry_threshold=[0.10, 0.15])
wf = walk_forward(market, spec, grid, train_bars=4000, test_bars=1500, step_bars=1500)

def downsample(series, n=400):
    if len(series) <= n:
        return series
    step = max(1, len(series) // n)
    return series.iloc[::step]

def eq_series(df):
    s = downsample(df["equity"])
    return [{"t": t.isoformat(), "v": round(float(v), 2)} for t, v in s.items()]

def dd_series(df):
    eq = df["equity"]
    dd = (eq / eq.cummax() - 1.0)
    s = downsample(dd)
    return [{"t": t.isoformat(), "v": round(float(v)*100, 4)} for t, v in s.items()]

out = {
    "meta": {
        "instrument": "ES (E-mini S&P 500 futures)",
        "bar_seconds": 30,
        "n_bars": int(len(market)),
        "capital": 1_000_000.0,
        "generated": "synthetic (data.py generate_synthetic_futures_data, seed=7)",
    },
    "naive": {
        "equity": eq_series(naive_df),
        "drawdown": dd_series(naive_df),
        "metrics": {
            "sharpe": naive_report.sharpe, "sortino": naive_report.sortino,
            "calmar": naive_report.calmar, "max_dd": naive_report.max_dd,
            "ann_return": naive_report.ann_return, "ann_vol": naive_report.ann_vol,
            "turnover": naive_report.turnover, "hit_rate": naive_report.hit_rate,
            "gross_pnl": naive_report.gross_pnl, "total_cost": naive_report.total_cost,
            "net_pnl": naive_report.net_pnl, "cost_drag_pct": naive_report.cost_drag_pct,
        },
    },
    "managed": {
        "equity": eq_series(result.frame),
        "drawdown": dd_series(result.frame),
        "metrics": {
            "sharpe": result.report.sharpe, "sortino": result.report.sortino,
            "calmar": result.report.calmar, "max_dd": result.report.max_dd,
            "ann_return": result.report.ann_return, "ann_vol": result.report.ann_vol,
            "turnover": result.report.turnover, "hit_rate": result.report.hit_rate,
            "gross_pnl": result.report.gross_pnl, "total_cost": result.report.total_cost,
            "net_pnl": result.report.net_pnl, "cost_drag_pct": result.report.cost_drag_pct,
        },
        "cost_breakdown": {
            "spread": round(float(result.cost_breakdown["spread_cost"].iloc[0]), 2),
            "impact": round(float(result.cost_breakdown["impact_cost"].iloc[0]), 2),
            "commission": round(float(result.cost_breakdown["commission_cost"].iloc[0]), 2),
        },
        "stops_triggered": int(result.frame["stop_triggered"].sum()),
        "breaker_halts": int(result.frame["halted"].sum()),
    },
    "walk_forward": {
        "windows": [
            {"window": i, "params": p, "is_sharpe": round(is_s, 3), "oos_sharpe": round(oos_s, 3)}
            for i, (is_s, oos_s, p) in enumerate(zip(wf["is_sharpes"], wf["oos_sharpes"], wf["chosen_params"]))
        ],
        "oos_sharpe_concat": round(sharpe_ratio(wf["oos_frame"]["returns_net"], _bars_per_year(wf["oos_frame"].index)), 3),
    },
}

with open("/home/claude/quant_framework/results.json", "w") as f:
    json.dump(out, f)

print("exported", len(json.dumps(out)), "bytes")
