from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import pandas as pd

from .charts import generate_dashboards
from .data_io import collect_from_api, read_unified_csv, standardize_dataset
from .modeling import (
    ForecastConfig,
    build_network_to_iops_frame,
    build_timeseries_frame,
    lag_correlation,
    recursive_forecast_network_to_iops,
    recursive_forecast_timeseries,
    train_and_score,
)


def _prepare_overall(df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        df.groupby("timestamp", as_index=False)
        .agg(
            {
                "read_iops": "mean",
                "write_iops": "mean",
                "total_iops": "mean",
                "in_bandwidth": "mean",
                "out_bandwidth": "mean",
                "total_network": "mean",
            }
        )
        .sort_values("timestamp")
    )
    grouped["ip"] = "OVERALL"
    return grouped


def _run_timeseries_for_scope(scope_df: pd.DataFrame, scope: str, cfg: ForecastConfig, freq_minutes: int) -> tuple[list[dict], list[dict], list[dict]]:
    metrics_rows: List[dict] = []
    backtest_rows: List[dict] = []
    forecast_rows: List[dict] = []

    for metric in ["total_iops", "in_bandwidth", "out_bandwidth", "total_network"]:
        frame = build_timeseries_frame(scope_df, target_col=metric, max_lag=cfg.max_lag_steps)
        if len(frame) < 50:
            continue

        model, metrics, scored = train_and_score(frame, test_fraction=cfg.test_fraction)
        metrics_rows.append(
            {
                "model_family": "time_series",
                "scope": scope,
                "metric": metric,
                **metrics,
            }
        )

        bt = scored.rename(columns={"y": "actual"}).copy()
        bt["scope"] = scope
        bt["metric"] = metric
        bt["model_family"] = "time_series"
        backtest_rows.extend(bt.to_dict("records"))

        short_fc = recursive_forecast_timeseries(scope_df, metric, model, cfg.short_horizon_steps, cfg.max_lag_steps, freq_minutes)
        short_fc["horizon_type"] = "short_term"

        long_fc = recursive_forecast_timeseries(scope_df, metric, model, cfg.long_horizon_steps, cfg.max_lag_steps, freq_minutes)
        long_fc["horizon_type"] = "long_term"

        fc = pd.concat([short_fc, long_fc], ignore_index=True)
        fc["scope"] = scope
        fc["metric"] = metric
        fc["model_family"] = "time_series"
        fc["actual"] = pd.NA
        fc["lower_bound"] = fc["predicted"] * 0.9
        fc["upper_bound"] = fc["predicted"] * 1.1
        forecast_rows.extend(fc.to_dict("records"))

    return metrics_rows, backtest_rows, forecast_rows


def _run_network_to_iops_for_scope(scope_df: pd.DataFrame, scope: str, cfg: ForecastConfig, freq_minutes: int) -> tuple[list[dict], list[dict], list[dict], pd.DataFrame]:
    frame = build_network_to_iops_frame(scope_df, max_lag=cfg.max_lag_steps)
    if len(frame) < 50:
        return [], [], [], pd.DataFrame()

    model, metrics, scored = train_and_score(frame, test_fraction=cfg.test_fraction)

    metrics_rows = [
        {
            "model_family": "network_to_iops",
            "scope": scope,
            "metric": "total_iops",
            **metrics,
        }
    ]

    bt = scored.rename(columns={"y": "actual"}).copy()
    bt["scope"] = scope
    bt["metric"] = "total_iops"
    bt["model_family"] = "network_to_iops"

    short_fc = recursive_forecast_network_to_iops(scope_df, model, cfg.short_horizon_steps, cfg.max_lag_steps, freq_minutes)
    short_fc["horizon_type"] = "short_term"

    long_fc = recursive_forecast_network_to_iops(scope_df, model, cfg.long_horizon_steps, cfg.max_lag_steps, freq_minutes)
    long_fc["horizon_type"] = "long_term"

    fc = pd.concat([short_fc, long_fc], ignore_index=True)
    fc["scope"] = scope
    fc["metric"] = "total_iops"
    fc["model_family"] = "network_to_iops"
    fc["actual"] = pd.NA
    fc["lower_bound"] = fc["predicted"] * 0.9
    fc["upper_bound"] = fc["predicted"] * 1.1

    corr_df = lag_correlation(scope_df, x_col="total_network", y_col="total_iops", max_lag=min(24, cfg.max_lag_steps))
    if not corr_df.empty:
        corr_df["scope"] = scope

    return metrics_rows, bt.to_dict("records"), fc.to_dict("records"), corr_df


def _write_outputs(
    base_dir: Path,
    dataset: pd.DataFrame,
    metrics_rows: list[dict],
    backtest_rows: list[dict],
    forecast_rows: list[dict],
    corr_rows: list[dict],
    methodology_text: str,
) -> list[str]:
    base_dir.mkdir(parents=True, exist_ok=True)
    dataset.to_json(base_dir / "dataset_unified.json", orient="records", date_format="iso", indent=2)

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_json(base_dir / "metrics.json", orient="records", indent=2)

    backtest_df = pd.DataFrame(backtest_rows)
    backtest_df.to_json(base_dir / "backtest_predictions.json", orient="records", date_format="iso", indent=2)

    forecast_df = pd.DataFrame(forecast_rows)
    forecast_df.to_json(base_dir / "forecast_output.json", orient="records", date_format="iso", indent=2)

    corr_df = pd.DataFrame(corr_rows)
    corr_df.to_json(base_dir / "correlation_insights.json", orient="records", indent=2)

    chart_files = generate_dashboards(base_dir, forecast_df, backtest_df)

    with (base_dir / "METHODOLOGY.md").open("w", encoding="utf-8") as f:
        f.write(methodology_text)

    return chart_files


def run_pipeline(raw_cfg: dict) -> Dict[str, str]:
    input_cfg = raw_cfg.get("input", {})

    if input_cfg.get("unified_history_csv", "").strip():
        df = read_unified_csv(input_cfg["unified_history_csv"])
    elif raw_cfg.get("api", {}).get("enabled", False):
        df = collect_from_api(raw_cfg)
    else:
        raise ValueError("Provide input.unified_history_csv or enable api")

    df = standardize_dataset(df)

    overall_df = _prepare_overall(df)
    combined = pd.concat([df, overall_df], ignore_index=True)

    fc_cfg = raw_cfg.get("forecast", {})
    cfg = ForecastConfig(
        max_lag_steps=int(fc_cfg.get("max_lag_steps", 24)),
        short_horizon_steps=int(fc_cfg.get("short_horizon_steps", 24)),
        long_horizon_steps=int(fc_cfg.get("long_horizon_steps", 168)),
        test_fraction=float(fc_cfg.get("test_fraction", 0.2)),
    )
    freq_minutes = int(raw_cfg.get("project", {}).get("frequency_minutes", 1))

    metrics_rows: List[dict] = []
    backtest_rows: List[dict] = []
    forecast_rows: List[dict] = []
    corr_rows: List[dict] = []

    for scope, scope_df in combined.groupby("ip", sort=True):
        scope_df = scope_df.sort_values("timestamp").reset_index(drop=True)
        ts_metrics, ts_backtest, ts_forecast = _run_timeseries_for_scope(scope_df, scope, cfg, freq_minutes)
        metrics_rows.extend(ts_metrics)
        backtest_rows.extend(ts_backtest)
        forecast_rows.extend(ts_forecast)

        n2i_metrics, n2i_backtest, n2i_forecast, corr_df = _run_network_to_iops_for_scope(scope_df, scope, cfg, freq_minutes)
        metrics_rows.extend(n2i_metrics)
        backtest_rows.extend(n2i_backtest)
        forecast_rows.extend(n2i_forecast)
        if not corr_df.empty:
            corr_rows.extend(corr_df.to_dict("records"))

    methodology = (
        "# Forecasting Methodology\n\n"
        "- Data source: unified time-indexed server performance data with IOPS and network fields.\n"
        "- Preprocessing: timestamp normalization, numeric coercion, missing-row drop, and per-scope sorting.\n"
        "- Time-series models: RandomForestRegressor with time/lag/rolling features per metric.\n"
        "- Network-to-IOPS model: RandomForestRegressor using lagged network signals to predict total_iops.\n"
        "- Horizons: short_term and long_term as configured in config.yaml.\n"
        "- Scopes: individual per ip and OVERALL aggregate (mean across servers by timestamp).\n"
        "- Evaluation: MAE, RMSE, MAPE, SMAPE on holdout split.\n"
        "- Correlation insights: Pearson and Spearman lag correlation between total_network and total_iops.\n"
        "- Forecast export: visualization-ready JSON with timestamp, scope, metric, model family, predicted value, and bounds.\n"
    )

    out_dir = Path(raw_cfg.get("output", {}).get("base_dir", "outputs"))
    chart_files = _write_outputs(out_dir, combined, metrics_rows, backtest_rows, forecast_rows, corr_rows, methodology)

    return {
        "output_dir": str(out_dir.resolve()),
        "metrics_file": str((out_dir / "metrics.json").resolve()),
        "forecast_file": str((out_dir / "forecast_output.json").resolve()),
        "correlation_file": str((out_dir / "correlation_insights.json").resolve()),
        "charts_dir": str((out_dir / "charts").resolve()),
        "charts_count": str(len(chart_files)),
    }
