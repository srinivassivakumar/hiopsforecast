from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd


def _pick_individual_scope(forecast_df: pd.DataFrame) -> str | None:
    scopes = [s for s in forecast_df["scope"].unique() if s != "OVERALL"]
    return scopes[0] if scopes else None


def _pick_forecast_15min(forecast_df: pd.DataFrame, metric: str, scope: str | None) -> pd.DataFrame:
    if scope is None:
        return pd.DataFrame()

    sub = forecast_df[
        (forecast_df["metric"] == metric)
        & (forecast_df["scope"] == scope)
        & (forecast_df["model_family"] == "time_series")
    ].copy()
    if sub.empty:
        sub = forecast_df[(forecast_df["metric"] == metric) & (forecast_df["scope"] == scope)].copy()
    if sub.empty:
        return pd.DataFrame()

    sub = sub.sort_values("timestamp")
    # Use the nearest horizon and keep only first 60 points (next 60 minutes at 1-minute cadence).
    for horizon in ("short_term", "long_term"):
        part = sub[sub["horizon_type"] == horizon]
        if not part.empty:
            return part.head(15).reset_index(drop=True)

    return sub.head(15).reset_index(drop=True)


def _pick_actual_15min(backtest_df: pd.DataFrame, metric: str, scope: str | None) -> pd.DataFrame:
    if scope is None or backtest_df.empty:
        return pd.DataFrame()

    sub = backtest_df[
        (backtest_df["metric"] == metric)
        & (backtest_df["scope"] == scope)
        & (backtest_df["model_family"] == "time_series")
    ].copy()
    if sub.empty:
        sub = backtest_df[(backtest_df["metric"] == metric) & (backtest_df["scope"] == scope)].copy()
    if sub.empty:
        return pd.DataFrame()

    return sub.sort_values("timestamp").tail(15).reset_index(drop=True)


def _plot_actual_then_forecast(
    ax: plt.Axes,
    actual_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
    title: str,
    ylabel: str,
) -> None:
    # Plot only two lines: green actual 60 min then blue forecast next 60 min.
    if not actual_df.empty:
        ax.plot(
            actual_df["timestamp"],
            actual_df["actual"],
            color="green",
            linewidth=2.3,
            label="Actual (15 min)",
        )

    if not forecast_df.empty:
        fc = forecast_df.copy()

        # Start blue line from where green ended so both segments are continuous.
        if not actual_df.empty:
            bridge_row = pd.DataFrame(
                [{"timestamp": actual_df.iloc[-1]["timestamp"], "predicted": float(actual_df.iloc[-1]["actual"])}]
            )
            fc = pd.concat([bridge_row, fc], ignore_index=True)

        ax.plot(
            fc["timestamp"],
            fc["predicted"],
            color="blue",
            linewidth=2.3,
            label="Forecast (next 15 min)",
        )

    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Time (IST)")
    ax.set_ylabel(ylabel)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d-%b %I:%M %p"))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.tick_params(axis="x", rotation=20)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=10)


# ── public API ────────────────────────────────────────────────────────────────

def generate_dashboards(base_dir: Path, forecast_df: pd.DataFrame, backtest_df: pd.DataFrame) -> list[str]:
    """
        Produce exactly 2 PNG charts in simple format:
            - green line: last 60 minutes actual
            - blue line: next 60 minutes forecast
        Timestamps are shown in IST.
    """
    charts_dir = base_dir / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)

    generated: list[str] = []

    if forecast_df.empty:
        return generated

    forecast_df = forecast_df.copy()
    forecast_df["timestamp"] = pd.to_datetime(forecast_df["timestamp"], utc=True)
    forecast_df["timestamp"] = forecast_df["timestamp"].dt.tz_convert("Asia/Kolkata")
    backtest_df = backtest_df.copy()
    if not backtest_df.empty:
        backtest_df["timestamp"] = pd.to_datetime(backtest_df["timestamp"], utc=True)
        backtest_df["timestamp"] = backtest_df["timestamp"].dt.tz_convert("Asia/Kolkata")

    individual_scope = _pick_individual_scope(forecast_df)

    # ── Graph 1: IOPS Forecast ────────────────────────────────────────────────
    fig1, ax1 = plt.subplots(figsize=(14, 5))

    iops_actual = _pick_actual_15min(backtest_df, "total_iops", individual_scope)
    iops_forecast = _pick_forecast_15min(forecast_df, "total_iops", individual_scope)

    _plot_actual_then_forecast(
        ax=ax1,
        actual_df=iops_actual,
        forecast_df=iops_forecast,
        title=f"Disk IOPS: 15 min Actual + Next 15 min Forecast ({individual_scope or 'N/A'})",
        ylabel="IOPS",
    )

    fig1.tight_layout()
    iops_path = charts_dir / "graph_iops_forecast.png"
    fig1.savefig(iops_path, dpi=130)
    plt.close(fig1)
    generated.append(str(iops_path))

    # ── Graph 2: Network Forecast ─────────────────────────────────────────────
    fig2, ax2 = plt.subplots(figsize=(14, 5))

    net_actual = _pick_actual_15min(backtest_df, "total_network", individual_scope)
    net_forecast = _pick_forecast_15min(forecast_df, "total_network", individual_scope)

    _plot_actual_then_forecast(
        ax=ax2,
        actual_df=net_actual,
        forecast_df=net_forecast,
        title=f"Network: 15 min Actual + Next 15 min Forecast ({individual_scope or 'N/A'})",
        ylabel="Bandwidth (Mbps)",
    )

    fig2.tight_layout()
    net_path = charts_dir / "graph_network_forecast.png"
    fig2.savefig(net_path, dpi=130)
    plt.close(fig2)
    generated.append(str(net_path))

    return generated
