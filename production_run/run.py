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
import time
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
API_BASE = "https://staging-admin-console.zybisys.com/api/admin-api/vm-performance"
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
print("Select training mode:")
print("  1  Day-based  — train on a date range, forecast a full day")
print("  2  Hour-based — train on a time range, forecast N hours ahead")
MODE = ask("Mode (1 or 2)", "1").strip()

ip = ask("Enter VM IP address", "10.192.1.71")

iops_smoothing_on = False
iops_smooth_window = 1

if MODE == "1":
    # ── Day-based ──────────────────────────────────────────────────────────
    print("\nFormat: YYYY-MM-DD")
    train_start_str   = ask("Train from (date)", (datetime.now(IST) - timedelta(days=8)).strftime("%Y-%m-%d"))
    train_end_str     = ask("Train to   (date)", (datetime.now(IST) - timedelta(days=1)).strftime("%Y-%m-%d"))
    forecast_date_str = ask("Forecast for (date)", datetime.now(IST).strftime("%Y-%m-%d"))
    show_actual_min   = 60
    iops_smoothing_on = False
    iops_smooth_window = 1
    show_forecast_min = 1440  # recalculated after fetch from actual holdout row count
else:
    # ── Hour-based ─────────────────────────────────────────────────────────
    print("\nFormat: YYYY-MM-DD HH:MM")
    train_start_str = ask("Train from", (datetime.now(IST) - timedelta(hours=4)).strftime("%Y-%m-%d %H:%M"))
    train_end_str   = ask("Train to  ", (datetime.now(IST) - timedelta(hours=1)).strftime("%Y-%m-%d %H:%M"))
    forecast_hours  = float(ask("Forecast duration (hours)", "1"))
    show_forecast_min = max(1, int(forecast_hours * 60))
    show_actual_min   = int(ask("Minutes of actual history to show on chart", "60"))
    iops_smoothing_on = False
    iops_smooth_window = 1

# ──────────────────────────────────────────────
# 2. CONVERT DATES → EPOCH MILLISECONDS FOR URL
# ──────────────────────────────────────────────

def date_to_ms(date_str: str, end_of_day: bool = False) -> int:
    """Parse YYYY-MM-DD in IST and return epoch-milliseconds (UTC)."""
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    if end_of_day:
        dt = dt.replace(hour=23, minute=59, second=59)
    dt_ist = dt.replace(tzinfo=IST)
    return int(dt_ist.astimezone(timezone.utc).timestamp() * 1000)


def datetime_to_ms(dt_str: str) -> int:
    """Parse 'YYYY-MM-DD HH:MM' in IST and return epoch-milliseconds (UTC)."""
    dt = datetime.strptime(dt_str.strip(), "%Y-%m-%d %H:%M")
    return int(dt.replace(tzinfo=IST).astimezone(timezone.utc).timestamp() * 1000)


try:
    if MODE == "1":
        from_ms = date_to_ms(train_start_str, end_of_day=False)
        to_ms   = date_to_ms(forecast_date_str, end_of_day=True)
    else:
        from_ms = datetime_to_ms(train_start_str)
        to_ms   = datetime_to_ms(train_end_str) + show_forecast_min * 60 * 1000
except ValueError:
    if MODE == "1":
        print("\nERROR: Invalid date format. Use YYYY-MM-DD (example: 2026-04-15).")
    else:
        print("\nERROR: Invalid date/time format. Use YYYY-MM-DD HH:MM (example: 2026-04-15 14:30).")
    sys.exit(1)

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

RETRYABLE_STATUSES = {502, 503, 504}
MAX_FETCH_RETRIES = 4

def _fetch_chunk(ip: str, from_ms: int, to_ms: int, timeout: int = 90) -> list:
    req_url = f"{API_BASE}/{ip}?from={from_ms}&to={to_ms}"
    last_err: Exception | None = None
    for attempt in range(1, MAX_FETCH_RETRIES + 1):
        try:
            resp = requests.get(req_url, headers=headers, timeout=timeout)
            if resp.status_code == 404:
                # API sometimes has sparse history windows; treat them as empty chunks.
                return []
            if resp.status_code in RETRYABLE_STATUSES and attempt < MAX_FETCH_RETRIES:
                wait_s = 2 ** (attempt - 1)
                print(
                    f"    transient HTTP {resp.status_code}; retry {attempt}/{MAX_FETCH_RETRIES - 1} in {wait_s}s",
                    flush=True,
                )
                time.sleep(wait_s)
                continue
            resp.raise_for_status()
            payload = resp.json()
            return payload.get("performance_metrics", [])
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
            last_err = e
            if attempt >= MAX_FETCH_RETRIES:
                break
            wait_s = 2 ** (attempt - 1)
            print(
                f"    transient network error ({type(e).__name__}); retry {attempt}/{MAX_FETCH_RETRIES - 1} in {wait_s}s",
                flush=True,
            )
            time.sleep(wait_s)
        except Exception as e:
            last_err = e
            break
    raise RuntimeError(f"fetch failed: {last_err}") from last_err


CHUNK_DAYS = 3  # smaller chunks reduce timeout risk on staging


def fetch_vm_data(ip: str, from_ms: int, to_ms: int, timeout: int = 180) -> pd.DataFrame:
    chunk_ms = CHUNK_DAYS * 24 * 3600 * 1000
    all_metrics = []
    cursor = from_ms
    while cursor < to_ms:
        end = min(cursor + chunk_ms, to_ms)
        print(f"  Fetching chunk {pd.Timestamp(cursor, unit='ms', tz='UTC').date()} → {pd.Timestamp(end, unit='ms', tz='UTC').date()} …", flush=True)
        all_metrics.extend(_fetch_chunk(ip, cursor, end, timeout=90))
        cursor = end + 1
    metrics = all_metrics
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

def infer_freq_minutes(ts: pd.Series) -> int:
    diffs = ts.sort_values().diff().dropna()
    if diffs.empty:
        return 1
    median_diff = diffs.median()
    freq = int(median_diff.total_seconds() / 60)
    return max(1, freq)


def make_seasonal_lags(n_rows: int, freq_minutes: int) -> list[int]:
    lags = set()

    # Recent memory (dense)
    for x in [1, 2, 3, 5, 10, 15]:
        if x < n_rows:
            lags.add(x)

    # Local trend (medium)
    for x in [30, 60, 120, 240]:
        if x < n_rows:
            lags.add(x)

    # Daily seasonality
    day_lag = int((24 * 60) / freq_minutes)
    if day_lag < n_rows:
        lags.add(day_lag)

    # Weekly seasonality
    week_lag = int((7 * 24 * 60) / freq_minutes)
    if week_lag < n_rows:
        lags.add(week_lag)

    # Monthly seasonality
    month_lag = int((30 * 24 * 60) / freq_minutes)
    if month_lag < n_rows:
        lags.add(month_lag)

    # For short training windows, cap very long lags to preserve enough clean rows.
    if n_rows < 500:
        max_allowed = max(12, n_rows // 4)
        lags = {x for x in lags if x <= max_allowed}

    return sorted(lags)


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


def lag_features(series: pd.Series, name: str, lags: list[int]) -> pd.DataFrame:
    out = {}
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


def build_design_matrix(df: pd.DataFrame, target: str, exogenous_cols: list[str]) -> pd.DataFrame:
    freq_minutes = infer_freq_minutes(df["timestamp"])
    lags = make_seasonal_lags(len(df), freq_minutes)
    tf = time_features(df["timestamp"])
    feats = [tf, lag_features(df[target], target, lags)]
    for col in exogenous_cols:
        feats.append(lag_features(df[col], col, lags))
    x = pd.concat(feats, axis=1)
    x.insert(0, "timestamp", df["timestamp"])
    return x

# ──────────────────────────────────────────────
# 5. TRAIN ONE-STEP MODEL + RECURSIVE FORECAST
# ──────────────────────────────────────────────

try:
    from lightgbm import LGBMRegressor
except ImportError as exc:
    raise SystemExit(
        "LightGBM is required but not installed. Run: pip install lightgbm"
    ) from exc

def make_model():
    return LGBMRegressor(
        n_estimators=400,
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


def train_single_model(
    df_train: pd.DataFrame,
    target: str,
    exogenous_cols: list[str],
) -> tuple[Optional[object], list[str], pd.Series, pd.Series, float]:
    """Train a one-step model and return training feature bounds for recursive inference."""
    x_frame = build_design_matrix(df_train, target, exogenous_cols)
    feature_cols = [c for c in x_frame.columns if c != "timestamp"]
    x_all = x_frame[feature_cols]

    feat_lo = x_all.quantile(0.02)
    feat_hi = x_all.quantile(0.98)

    y_log = np.log1p(df_train[target].shift(-1))
    train_mask = (~x_all.isna().any(axis=1)) & y_log.notna()
    x_h = x_all.loc[train_mask]
    y_h = y_log.loc[train_mask]
    fallback_pred = float(df_train[target].tail(min(30, len(df_train))).median())
    if len(x_h) < 20:
        print(
            f"Warning: limited clean rows for {target} ({len(x_h)}). "
            "Using recent-median fallback forecast."
        )
        return None, feature_cols, feat_lo, feat_hi, max(0.0, fallback_pred)
    mdl = make_model()
    mdl.fit(x_h, y_h)
    return mdl, feature_cols, feat_lo, feat_hi, max(0.0, fallback_pred)


def forecast_recursive(
    train_df: pd.DataFrame,
    target: str,
    exogenous_cols: list[str],
    steps: int,
    future_exog: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Forecast `steps` minutes ahead by rolling one-step predictions forward."""
    mdl, feature_cols, feat_lo, feat_hi, fallback_pred = train_single_model(
        train_df, target, exogenous_cols
    )

    history = train_df.copy().reset_index(drop=True)
    future_exog_lookup = None
    if future_exog is not None and not future_exog.empty:
        future_exog_lookup = future_exog.copy()
        future_exog_lookup = future_exog_lookup.set_index("timestamp")

    out = []
    for h in range(1, steps + 1):
        next_ts = history["timestamp"].iloc[-1] + pd.Timedelta(minutes=1)

        next_row = history.iloc[[-1]].copy()
        next_row.loc[:, "timestamp"] = next_ts
        next_row.loc[:, target] = np.nan
        if future_exog_lookup is not None and next_ts in future_exog_lookup.index:
            future_vals = future_exog_lookup.loc[next_ts]
            for col in exogenous_cols:
                if col in future_vals.index:
                    next_row.loc[:, col] = future_vals[col]

        history_for_features = pd.concat([history, next_row], ignore_index=True)
        x_frame = build_design_matrix(history_for_features, target, exogenous_cols)
        latest_x = x_frame[feature_cols].iloc[[-1]].copy().clip(
            lower=feat_lo,
            upper=feat_hi,
            axis=1,
        )

        if mdl is None or latest_x.isna().any().any():
            pred_value = fallback_pred
        else:
            y_log = float(mdl.predict(latest_x)[0])
            pred_value = max(0.0, float(np.expm1(y_log)))

        history_for_features.loc[history_for_features.index[-1], target] = pred_value
        history = history_for_features
        out.append({"timestamp": next_ts, "predicted": pred_value})
    return pd.DataFrame(out)


def forecast_future_exogenous(train_df: pd.DataFrame, steps: int) -> pd.DataFrame:
    # Instead of forecasting, repeat the last known values for exogenous stability
    last_row = train_df.iloc[-1][[
        "total_network", "cpu_percent", "ram_percent", "cpu_load_total", "cpu_load_1", "cpu_load_5"
    ]].to_dict()
    future_timestamps = pd.date_range(
        start=train_df["timestamp"].iloc[-1] + pd.Timedelta(minutes=1),
        periods=steps,
        freq="min"
    )
    future = pd.DataFrame({
        "timestamp": future_timestamps,
        **{k: [last_row[k]] * steps for k in last_row}
    })
    return future


if MODE == "1":
    # Day-based: train on [train_from, train_to], compare on forecast day
    train_start_utc = pd.Timestamp(
        datetime.strptime(train_start_str, "%Y-%m-%d").replace(tzinfo=IST)
    ).tz_convert("UTC")
    train_end_utc = pd.Timestamp(
        datetime.strptime(train_end_str, "%Y-%m-%d").replace(tzinfo=IST)
    ).tz_convert("UTC") + pd.Timedelta(days=1)
    forecast_date_utc = pd.Timestamp(
        datetime.strptime(forecast_date_str, "%Y-%m-%d").replace(tzinfo=IST)
    ).tz_convert("UTC")
    forecast_date_end = forecast_date_utc + pd.Timedelta(days=1)
    train_df   = df[
        (df["timestamp"] >= train_start_utc) & (df["timestamp"] < train_end_utc)
    ].copy().reset_index(drop=True)
    holdout_df = df[
        (df["timestamp"] >= forecast_date_utc) & (df["timestamp"] < forecast_date_end)
    ].copy().reset_index(drop=True)
    show_forecast_min = max(len(holdout_df), 1)
    chart_label = f"Forecast for {forecast_date_str}"
else:
    # Hour-based: split by train_end timestamp
    train_cutoff_ts = pd.Timestamp(datetime_to_ms(train_end_str), unit="ms", tz="UTC")
    train_df   = df[df["timestamp"] <= train_cutoff_ts].copy().reset_index(drop=True)
    holdout_df = df[df["timestamp"] > train_cutoff_ts].head(show_forecast_min).copy().reset_index(drop=True)
    chart_label = f"Forecast — next {show_forecast_min} min"

print("Training IOPS model ...")
future_exog = forecast_future_exogenous(train_df, show_forecast_min)
iops_exog = [
    # "total_network",
    # "cpu_percent", "ram_percent", "cpu_load_total", "cpu_load_1", "cpu_load_5",
]
iops_fcast = forecast_recursive(
    train_df,
    target="total_iops",
    exogenous_cols=iops_exog,
    steps=show_forecast_min,
    future_exog=future_exog,
)

print("Training Network model ...")
net_fcast = forecast_recursive(
    train_df,
    target="total_network",
    exogenous_cols=[],
    steps=show_forecast_min,
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
    title=f"Disk IOPS — {chart_label}  ({ip})",
    ylabel="IOPS",
    out_path=OUTPUT_DIR / "graph_iops_forecast.png",
    smooth_window=iops_smooth_window,
)

plot_chart(
    train_df, net_compare,
    history_col="total_network",
    title=f"Network — {chart_label}  ({ip})",
    ylabel="Bandwidth (Mbps)",
    out_path=OUTPUT_DIR / "graph_network_forecast.png",
)

print("\nDone! Charts saved to:", OUTPUT_DIR.resolve())
