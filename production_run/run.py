"""
Simple VM Performance Forecasting Script
-----------------------------------------
Asks you for:
  - IP address to query
  - Training data date range (start date → end date) which it converts to epoch-ms for the URL
  - How many minutes of actual and forecast to show on the chart

Then: fetches data → trains LightGBM → plots 2 graphs (IOPS + Network)
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ──────────────────────────────────────────────
# CONSTANTS  (edit these if they ever change)
# ──────────────────────────────────────────────
API_BASE = "https://dev-admin-console.zybisys.com/api/admin-api/vm-performance"
SECRET_KEY = "A0tziuB02IrdIS"
OUTPUT_DIR = Path("outputs/charts")
IST = timezone(timedelta(hours=5, minutes=30))

# ──────────────────────────────────────────────
# 1. ASK USER FOR INPUT
# ──────────────────────────────────────────────

def ask(prompt: str, default: str) -> str:
    val = input(f"{prompt} [{default}]: ").strip()
    return val if val else default


print("\n=== VM Performance Forecasting ===\n")

ip = ask("Enter VM IP address", "10.192.1.71")

print("\nEnter the date range for training data.")
print("  Format: YYYY-MM-DD  (time defaults to 00:00:00 IST)")
start_str = ask("Start date", (datetime.now(IST) - timedelta(days=2)).strftime("%Y-%m-%d"))
end_str   = ask("End date  ", datetime.now(IST).strftime("%Y-%m-%d"))

show_actual_min   = int(ask("\nMinutes of actual data to show on chart", "15"))
show_forecast_min = int(ask("Minutes of forecast to generate & show  ", "15"))

iops_smoothing_on = ask("Enable IOPS smoothing on chart (y/n)", "n").strip().lower() in {"y", "yes"}
iops_smooth_window = 1
if iops_smoothing_on:
    iops_smooth_window = max(1, int(ask("IOPS smoothing window (minutes)", "3")))

# ──────────────────────────────────────────────
# 2. CONVERT DATES → EPOCH MILLISECONDS FOR URL
# ──────────────────────────────────────────────

def date_to_ms(date_str: str, end_of_day: bool = False) -> int:
    """Parse YYYY-MM-DD in IST and return epoch-milliseconds (UTC)."""
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    if end_of_day:
        dt = dt.replace(hour=23, minute=59, second=59)
    dt_ist = dt.replace(tzinfo=IST)          # treat as IST
    dt_utc = dt_ist.astimezone(timezone.utc)  # convert to UTC for epoch
    return int(dt_utc.timestamp() * 1000)


from_ms = date_to_ms(start_str, end_of_day=False)
to_ms   = date_to_ms(end_str,   end_of_day=True)

url = f"{API_BASE}/{ip}?from={from_ms}&to={to_ms}"
print(f"\nFetching: {url}")

# ──────────────────────────────────────────────
# 3. FETCH DATA FROM API
# ──────────────────────────────────────────────

def to_float(v, default=0.0):
    try:
        return float(v)
    except Exception:
        return default


def dig(obj, path: str, default=None):
    for part in path.split("."):
        if not isinstance(obj, dict) or part not in obj:
            return default
        obj = obj[part]
    return obj


headers = {"X-SECRET-KEY": SECRET_KEY}

def fetch_vm_data(ip: str, from_ms: int, to_ms: int, timeout: int = 60) -> pd.DataFrame:
    req_url = f"{API_BASE}/{ip}?from={from_ms}&to={to_ms}"
    try:
        resp = requests.get(req_url, headers=headers, timeout=timeout)
        resp.raise_for_status()
        payload = resp.json()
    except Exception as e:
        raise RuntimeError(f"fetch failed: {e}") from e

    metrics = payload.get("performance_metrics", [])
    rows = []
    for item in metrics:
        ts_raw = dig(item, "disk_io_summary.io_operations_data.time_str", None)
        if ts_raw is None:
            continue
        ts_int = int(to_float(ts_raw))
        ts = datetime.fromtimestamp(ts_int / 1000.0 if ts_int > 10_000_000_000 else ts_int, tz=timezone.utc)

        read_iops = to_float(dig(item, "disk_io_summary.io_operations_data.read_data", 0.0))
        write_iops = to_float(dig(item, "disk_io_summary.io_operations_data.write_data", 0.0))

        in_bw = out_bw = 0.0
        for itf in (item.get("interface") or []):
            # API returns bytes/s → convert to Mbps (÷ 125,000)
            in_bw += to_float(itf.get("in_bandwidth", 0.0)) / 125_000
            out_bw += to_float(itf.get("out_bandwidth", 0.0)) / 125_000

        rows.append({
            "timestamp": ts,
            "read_iops": read_iops,
            "write_iops": write_iops,
            "total_iops": read_iops + write_iops,
            "in_bandwidth": in_bw,
            "out_bandwidth": out_bw,
            "total_network": in_bw + out_bw,
            "cpu_percent": to_float(dig(item, "cpu.percent_used", np.nan), np.nan),
            "ram_percent": to_float(dig(item, "ram.percent_used", np.nan), np.nan),
            "cpu_load_total": to_float(dig(item, "cpu_load.total", np.nan), np.nan),
            "cpu_load_1": to_float(dig(item, "cpu_load.total1", np.nan), np.nan),
            "cpu_load_5": to_float(dig(item, "cpu_load.total5", np.nan), np.nan),
        })

    if not rows:
        return pd.DataFrame(columns=[
            "timestamp", "read_iops", "write_iops", "total_iops",
            "in_bandwidth", "out_bandwidth", "total_network",
            "cpu_percent", "ram_percent", "cpu_load_total", "cpu_load_1", "cpu_load_5",
        ])

    return pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)


try:
    df = fetch_vm_data(ip, from_ms, to_ms)
except Exception as e:
    print(f"\nERROR fetching data: {e}")
    sys.exit(1)

if df.empty:
    print("\nERROR: No data returned from API. Check IP and date range.")
    sys.exit(1)

print(f"Loaded {len(df)} rows  |  {df['timestamp'].min()} → {df['timestamp'].max()}")

# Fill sparse exogenous values for stable feature generation.
for c in ["cpu_percent", "ram_percent", "cpu_load_total", "cpu_load_1", "cpu_load_5"]:
    df[c] = pd.to_numeric(df[c], errors="coerce").ffill().bfill().fillna(0.0)

if len(df) < 30:
    print("\nERROR: Not enough data to train. Widen the date range.")
    sys.exit(1)

if len(df) < (24 + show_forecast_min + 5):
    print("\nERROR: Not enough data for stable actual-vs-forecast comparison.")
    print(f"Need at least {24 + show_forecast_min + 5} rows, got {len(df)}.")
    sys.exit(1)

# ──────────────────────────────────────────────
# 4. FEATURE ENGINEERING
# ──────────────────────────────────────────────

# All candidate lags; actual lags used are filtered by available data at training time.
LAGS_BASE = [1, 2, 3, 6, 12, 24, 60, 120, 240, 480, 1440]


def time_features(ts: pd.Series) -> pd.DataFrame:
    minute_of_day = ts.dt.hour * 60 + ts.dt.minute
    return pd.DataFrame({
        "hour_sin":          np.sin(2 * np.pi * ts.dt.hour / 24.0),
        "hour_cos":          np.cos(2 * np.pi * ts.dt.hour / 24.0),
        "dow_sin":           np.sin(2 * np.pi * ts.dt.dayofweek / 7.0),
        "dow_cos":           np.cos(2 * np.pi * ts.dt.dayofweek / 7.0),
        "minute_of_day_sin": np.sin(2 * np.pi * minute_of_day / 1440.0),
        "minute_of_day_cos": np.cos(2 * np.pi * minute_of_day / 1440.0),
        "is_weekend":        (ts.dt.dayofweek >= 5).astype(float),
        "is_business_hour":  ((ts.dt.hour >= 9) & (ts.dt.hour < 18) & (ts.dt.dayofweek < 5)).astype(float),
    }, index=ts.index)


def lag_features(series: pd.Series, name: str, max_lag_allowed: int = 1440) -> pd.DataFrame:
    out = {}
    lags = [l for l in LAGS_BASE if l <= max_lag_allowed]
    for lag in lags:
        out[f"{name}_lag_{lag}"] = series.shift(lag)
    # Use only past values for rolling stats to avoid target leakage.
    past = series.shift(1)
    ewma15 = past.ewm(span=15, adjust=False).mean()
    ewma60 = past.ewm(span=60, adjust=False).mean()
    out[f"{name}_roll3"]     = past.rolling(3,   min_periods=1).mean()
    out[f"{name}_roll12"]    = past.rolling(12,  min_periods=1).mean()
    out[f"{name}_roll24"]    = past.rolling(24,  min_periods=1).mean()
    out[f"{name}_roll60"]    = past.rolling(60,  min_periods=1).mean()
    out[f"{name}_rollstd3"]  = past.rolling(3,   min_periods=2).std().fillna(0.0)
    out[f"{name}_rollstd12"] = past.rolling(12,  min_periods=2).std().fillna(0.0)
    out[f"{name}_ewma5"]     = past.ewm(span=5,  adjust=False).mean()
    out[f"{name}_ewma15"]    = ewma15
    out[f"{name}_ewma60"]    = ewma60
    # Mean-reversion features: signed deviation from recent baseline
    # Positive = currently above baseline (spike), negative = below (dip)
    out[f"{name}_dev_ewma15"]   = past - ewma15
    out[f"{name}_dev_ewma60"]   = past - ewma60
    # Relative spike strength: deviation as fraction of baseline (clamped to avoid div-by-zero)
    baseline = ewma15.clip(lower=1e-6)
    out[f"{name}_rel_dev_ewma15"] = (past - ewma15) / baseline
    return pd.DataFrame(out, index=series.index)


def build_design_matrix(df: pd.DataFrame, target: str, exogenous_cols: list[str], max_lag: int = 1440) -> pd.DataFrame:
    tf = time_features(df["timestamp"])
    feats = [tf, lag_features(df[target], target, max_lag_allowed=max_lag)]
    for col in exogenous_cols:
        feats.append(lag_features(df[col], col, max_lag_allowed=max_lag))
    x = pd.concat(feats, axis=1)
    x.insert(0, "timestamp", df["timestamp"])
    return x

# ──────────────────────────────────────────────
# 5. TRAIN DIRECT-HORIZON MODELS
# ──────────────────────────────────────────────

USE_LGBM = False
try:
    from lightgbm import LGBMRegressor
    USE_LGBM = True
    def make_model():
        return LGBMRegressor(
            n_estimators=1200,
            learning_rate=0.03,
            num_leaves=63,
            min_child_samples=25,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_alpha=0.05,
            reg_lambda=0.05,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )
    print("\nUsing LightGBM")
except Exception:
    from sklearn.ensemble import RandomForestRegressor
    def make_model():
        return RandomForestRegressor(n_estimators=500, max_depth=14, random_state=42, n_jobs=-1)
    print("\nUsing RandomForest (LightGBM not installed)")


def train_direct_models(
    df_train: pd.DataFrame,
    target: str,
    exogenous_cols: list[str],
    horizons: list[int],
) -> tuple[dict[int, object], list[str], pd.Series, pd.DataFrame]:
    # Limit lags to at most 1/3 of training length so lag columns stay dense
    max_lag = min(1440, max(24, len(df_train) // 3))
    x_frame = build_design_matrix(df_train, target, exogenous_cols, max_lag=max_lag)
    feature_cols = [c for c in x_frame.columns if c != "timestamp"]
    x_all = x_frame[feature_cols]

    # Compute per-feature 2nd/98th percentile from training for prediction-time clipping
    feat_lo = x_all.quantile(0.02)
    feat_hi = x_all.quantile(0.98)

    # Clip the latest-x row to training feature range to prevent spike blowout
    x_latest_raw = x_all.iloc[[-1]].copy()
    x_latest = x_latest_raw.clip(lower=feat_lo, upper=feat_hi, axis=1)

    models: dict[int, object] = {}
    for h in horizons:
        y_log = np.log1p(df_train[target].shift(-h))
        train_mask = (~x_all.isna().any(axis=1)) & y_log.notna()
        x_h = x_all.loc[train_mask]
        y_h = y_log.loc[train_mask]
        if len(x_h) < 80:
            continue
        mdl = make_model()
        mdl.fit(x_h, y_h)
        models[h] = mdl
    return models, feature_cols, x_latest.iloc[0], x_all


def forecast_direct(
    train_df: pd.DataFrame,
    target: str,
    exogenous_cols: list[str],
    steps: int,
) -> pd.DataFrame:
    horizons = list(range(1, steps + 1))
    models, _, latest_x, _ = train_direct_models(train_df, target, exogenous_cols, horizons)

    last_ts = train_df["timestamp"].iloc[-1]
    out = []
    for h in horizons:
        ts = last_ts + pd.Timedelta(minutes=h)
        mdl = models.get(h)
        if mdl is None:
            pred = np.nan
        else:
            y_log = float(mdl.predict(latest_x.to_frame().T)[0])
            pred = max(0.0, float(np.expm1(y_log)))
        out.append({"timestamp": ts, "predicted": pred})
    return pd.DataFrame(out)


comparison_steps = show_forecast_min
split_idx = len(df) - comparison_steps
train_df = df.iloc[:split_idx].copy().reset_index(drop=True)
holdout_df = df.iloc[split_idx:].copy().reset_index(drop=True)

print("Training direct-horizon IOPS models ...")
iops_exog = [
    "read_iops", "write_iops",
    "in_bandwidth", "out_bandwidth", "total_network",
    "cpu_percent", "ram_percent", "cpu_load_total", "cpu_load_1", "cpu_load_5",
]
iops_fcast = forecast_direct(
    train_df,
    target="total_iops",
    exogenous_cols=iops_exog,
    steps=comparison_steps,
)

print("Training direct-horizon Network models ...")
net_exog = [
    "in_bandwidth", "out_bandwidth",
    "cpu_percent", "ram_percent", "cpu_load_total", "cpu_load_1", "cpu_load_5",
]
net_fcast = forecast_direct(
    train_df,
    target="total_network",
    exogenous_cols=net_exog,
    steps=comparison_steps,
)


def align_actual_vs_forecast(
    fcast_df: pd.DataFrame,
    actual_df: pd.DataFrame,
    actual_col: str,
) -> pd.DataFrame:
    fc = fcast_df[["timestamp", "predicted"]].copy()
    fc["ts_key"] = fc["timestamp"].dt.floor("min")
    fc = fc.groupby("ts_key", as_index=False)["predicted"].mean()

    if actual_df.empty or actual_col not in actual_df.columns:
        fc["actual"] = np.nan
        return fc

    ac = actual_df[["timestamp", actual_col]].copy()
    ac["ts_key"] = ac["timestamp"].dt.floor("min")
    ac = ac.groupby("ts_key", as_index=False)[actual_col].mean().rename(columns={actual_col: "actual"})

    return fc.merge(ac, on="ts_key", how="left").sort_values("ts_key").reset_index(drop=True)


iops_compare = align_actual_vs_forecast(iops_fcast, holdout_df, "total_iops")
net_compare = align_actual_vs_forecast(net_fcast, holdout_df, "total_network")

# ──────────────────────────────────────────────
# 6. PLOT CHARTS
# ──────────────────────────────────────────────

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def to_ist(ts_series: pd.Series) -> pd.Series:
    """Convert UTC timestamps to IST for display."""
    if ts_series.dt.tz is None:
        ts_series = ts_series.dt.tz_localize("UTC")
    return ts_series.dt.tz_convert(IST)


def mae_mape(compare_df: pd.DataFrame) -> tuple[Optional[float], Optional[float], int]:
    valid = compare_df.dropna(subset=["actual", "predicted"]).copy()
    if valid.empty:
        return None, None, 0
    mae = float(np.mean(np.abs(valid["actual"] - valid["predicted"])))
    non_zero = valid[np.abs(valid["actual"]) > 1e-9]
    if non_zero.empty:
        mape = None
    else:
        mape = float(np.mean(np.abs((non_zero["actual"] - non_zero["predicted"]) / non_zero["actual"])) * 100.0)
    return mae, mape, len(valid)


def plot_chart(history_df: pd.DataFrame, compare_df: pd.DataFrame,
               history_col: str, title: str, ylabel: str, out_path: Path,
               smooth_window: int = 1):

    # History = last N minutes from observed training data
    history = history_df[["timestamp", history_col]].sort_values("timestamp").tail(show_actual_min).copy()
    history["ts_ist"] = to_ist(history["timestamp"])

    cmp = compare_df.sort_values("ts_key").head(show_forecast_min).copy()
    cmp["timestamp"] = pd.to_datetime(cmp["ts_key"], utc=True)
    cmp["ts_ist"] = to_ist(cmp["timestamp"])

    if smooth_window > 1:
        history[history_col] = history[history_col].rolling(smooth_window, min_periods=1).mean()
        cmp["predicted"] = cmp["predicted"].rolling(smooth_window, min_periods=1).mean()
        if "actual" in cmp.columns:
            cmp["actual"] = cmp["actual"].rolling(smooth_window, min_periods=1).mean()

    # Bridge from history into forecast for visual continuity
    bridge = pd.DataFrame({
        "ts_ist": [history["ts_ist"].iloc[-1]],
        "predicted": [history[history_col].iloc[-1]],
    })
    pred_plot = pd.concat([bridge, cmp[["ts_ist", "predicted"]]], ignore_index=True)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(history["ts_ist"], history[history_col], color="green", linewidth=1.8,
            label=f"Actual ({show_actual_min} min)")
    ax.plot(pred_plot["ts_ist"], pred_plot["predicted"], color="blue", linewidth=1.8,
            label=f"Forecast (next {show_forecast_min} min)")

    future_actual = cmp.dropna(subset=["actual"])
    if not future_actual.empty:
        ax.plot(future_actual["ts_ist"], future_actual["actual"], color="red", linewidth=1.8,
                linestyle="--", marker="o", markersize=3, label="Actual (forecast window)")

    mae, mape, matched = mae_mape(cmp)
    if matched > 0:
        stats_txt = f"Matched points: {matched}/{len(cmp)}\\nMAE: {mae:.3f}"
        if mape is not None:
            stats_txt += f"\\nMAPE: {mape:.2f}%"
        ax.text(0.01, 0.98, stats_txt, transform=ax.transAxes, va="top", ha="left",
                fontsize=9, bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"})
    else:
        ax.text(0.01, 0.98, "No actual points yet for forecast window", transform=ax.transAxes,
                va="top", ha="left", fontsize=9,
                bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"})

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=IST))
    fig.autofmt_xdate()
    ax.set_title(title, fontsize=13)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Time (IST)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved → {out_path}")


plot_chart(
    train_df, iops_compare,
    history_col="total_iops",
    title=f"Disk IOPS — Holdout Actual vs Forecast ({show_forecast_min} min)  ({ip})",
    ylabel="IOPS",
    out_path=OUTPUT_DIR / "graph_iops_forecast.png",
    smooth_window=iops_smooth_window,
)

plot_chart(
    train_df, net_compare,
    history_col="total_network",
    title=f"Network — Holdout Actual vs Forecast ({show_forecast_min} min)  ({ip})",
    ylabel="Bandwidth (Mbps)",
    out_path=OUTPUT_DIR / "graph_network_forecast.png",
)

print("\nDone! Charts saved to:", OUTPUT_DIR.resolve())
