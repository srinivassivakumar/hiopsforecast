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

try:
    resp = requests.get(url, headers=headers, timeout=60)
    resp.raise_for_status()
    payload = resp.json()
except Exception as e:
    print(f"\nERROR fetching data: {e}")
    sys.exit(1)

metrics = payload.get("performance_metrics", [])
if not metrics:
    print("\nERROR: No data returned from API. Check IP and date range.")
    sys.exit(1)

rows = []
for item in metrics:
    ts_raw = dig(item, "disk_io_summary.io_operations_data.time_str", None)
    if ts_raw is None:
        continue
    ts_int = int(to_float(ts_raw))
    ts = datetime.fromtimestamp(ts_int / 1000.0 if ts_int > 10_000_000_000 else ts_int, tz=timezone.utc)

    read_iops  = to_float(dig(item, "disk_io_summary.io_operations_data.read_data",  0.0))
    write_iops = to_float(dig(item, "disk_io_summary.io_operations_data.write_data", 0.0))

    in_bw = out_bw = 0.0
    for itf in (item.get("interface") or []):
        in_bw  += to_float(itf.get("in_bandwidth",  0.0))
        out_bw += to_float(itf.get("out_bandwidth", 0.0))

    rows.append({
        "timestamp":     ts,
        "read_iops":     read_iops,
        "write_iops":    write_iops,
        "total_iops":    read_iops + write_iops,
        "in_bandwidth":  in_bw,
        "out_bandwidth": out_bw,
        "total_network": in_bw + out_bw,
    })

df = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)
print(f"Loaded {len(df)} rows  |  {df['timestamp'].min()} → {df['timestamp'].max()}")

if len(df) < 30:
    print("\nERROR: Not enough data to train. Widen the date range.")
    sys.exit(1)

# ──────────────────────────────────────────────
# 4. FEATURE ENGINEERING
# ──────────────────────────────────────────────

MAX_LAG = 24
LAGS    = [l for l in (1, 2, 3, 6, 12, 24) if l <= MAX_LAG]


def time_features(ts: pd.Series) -> pd.DataFrame:
    return pd.DataFrame({
        "hour_sin":  np.sin(2 * np.pi * ts.dt.hour / 24.0),
        "hour_cos":  np.cos(2 * np.pi * ts.dt.hour / 24.0),
        "dow_sin":   np.sin(2 * np.pi * ts.dt.dayofweek / 7.0),
        "dow_cos":   np.cos(2 * np.pi * ts.dt.dayofweek / 7.0),
        "minute":    ts.dt.minute,
    }, index=ts.index)


def lag_features(series: pd.Series, name: str) -> pd.DataFrame:
    out = {}
    for lag in LAGS:
        out[f"{name}_lag_{lag}"] = series.shift(lag)
    out[f"{name}_roll3"]  = series.rolling(3,  min_periods=1).mean()
    out[f"{name}_roll12"] = series.rolling(12, min_periods=1).mean()
    return pd.DataFrame(out, index=series.index)


def build_frame(df: pd.DataFrame, target: str) -> pd.DataFrame:
    tf = time_features(df["timestamp"])
    lf = lag_features(df[target], target)
    frame = pd.concat([df[["timestamp", target]].rename(columns={target: "y"}), tf, lf], axis=1)
    return frame.dropna().reset_index(drop=True)

# ──────────────────────────────────────────────
# 5. TRAIN MODEL
# ──────────────────────────────────────────────

try:
    from lightgbm import LGBMRegressor
    def make_model():
        return LGBMRegressor(n_estimators=600, learning_rate=0.05, num_leaves=31,
                             subsample=0.9, colsample_bytree=0.9, random_state=42, n_jobs=-1)
    print("\nUsing LightGBM")
except Exception:
    from sklearn.ensemble import RandomForestRegressor
    def make_model():
        return RandomForestRegressor(n_estimators=300, max_depth=12, random_state=42, n_jobs=-1)
    print("\nUsing RandomForest (LightGBM not installed)")


def train(df: pd.DataFrame, target: str, test_fraction: float = 0.2):
    frame = build_frame(df, target)
    feat_cols = [c for c in frame.columns if c not in {"timestamp", "y"}]
    split = max(1, int(len(frame) * (1 - test_fraction)))

    model = make_model()
    model.fit(frame.iloc[:split][feat_cols], frame.iloc[:split]["y"])

    test     = frame.iloc[split:]
    pred     = model.predict(test[feat_cols])
    backtest = test[["timestamp"]].copy()
    backtest["actual"]    = test["y"].values
    backtest["predicted"] = pred
    return model, backtest


def forecast(df: pd.DataFrame, target: str, model, steps: int, freq_min: int = 1):
    work = df[["timestamp", target]].copy().reset_index(drop=True)
    preds = []
    for _ in range(steps):
        next_ts = work["timestamp"].iloc[-1] + pd.Timedelta(minutes=freq_min)
        temp = pd.concat([work, pd.DataFrame({"timestamp": [next_ts], target: [np.nan]})], ignore_index=True)
        frame = build_frame(temp, target)
        feat_cols = [c for c in frame.columns if c not in {"timestamp", "y"}]
        y_hat = max(0.0, float(model.predict(frame[feat_cols].iloc[[-1]])[0]))
        work.loc[len(work)] = {"timestamp": next_ts, target: y_hat}
        preds.append({"timestamp": next_ts, "predicted": y_hat})
    return pd.DataFrame(preds)


print("Training IOPS model  ...")
iops_model,    iops_backtest    = train(df, "total_iops")
print("Training Network model ...")
net_model,     net_backtest     = train(df, "total_network")

print(f"Generating {show_forecast_min}-step forecast ...")
iops_fcast = forecast(df, "total_iops",    iops_model, steps=show_forecast_min)
net_fcast  = forecast(df, "total_network", net_model,  steps=show_forecast_min)

# ──────────────────────────────────────────────
# 6. PLOT CHARTS
# ──────────────────────────────────────────────

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def to_ist(ts_series: pd.Series) -> pd.Series:
    """Convert UTC timestamps to IST for display."""
    if ts_series.dt.tz is None:
        ts_series = ts_series.dt.tz_localize("UTC")
    return ts_series.dt.tz_convert(IST)


def plot_chart(backtest: pd.DataFrame, fcast: pd.DataFrame,
               actual_col: str, title: str, ylabel: str, out_path: Path):

    # Actual = last N minutes from backtest
    actual = backtest.sort_values("timestamp").tail(show_actual_min).copy()
    actual["ts_ist"] = to_ist(actual["timestamp"])

    # Forecast = next N minutes
    fc = fcast.head(show_forecast_min).copy()
    fc["ts_ist"] = to_ist(fc["timestamp"])

    # Bridge: add last actual point as first forecast point so lines join
    bridge = pd.DataFrame({
        "ts_ist":    [actual["ts_ist"].iloc[-1]],
        "predicted": [actual[actual_col].iloc[-1]],
    })
    fc_plot = pd.concat([bridge, fc[["ts_ist", "predicted"]]], ignore_index=True)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(actual["ts_ist"], actual[actual_col],  color="green", linewidth=1.8,
            label=f"Actual ({show_actual_min} min)")
    ax.plot(fc_plot["ts_ist"], fc_plot["predicted"], color="blue",  linewidth=1.8,
            label=f"Forecast (next {show_forecast_min} min)")

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
    iops_backtest, iops_fcast,
    actual_col="actual",
    title=f"Disk IOPS — {show_actual_min} min Actual + {show_forecast_min} min Forecast  ({ip})",
    ylabel="IOPS",
    out_path=OUTPUT_DIR / "graph_iops_forecast.png",
)

plot_chart(
    net_backtest, net_fcast,
    actual_col="actual",
    title=f"Network — {show_actual_min} min Actual + {show_forecast_min} min Forecast  ({ip})",
    ylabel="Bandwidth (Mbps)",
    out_path=OUTPUT_DIR / "graph_network_forecast.png",
)

print("\nDone! Charts saved to:", OUTPUT_DIR.resolve())
