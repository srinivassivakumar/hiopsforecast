"""
Network -> IOPS Forecasting (Direct-Horizon)
--------------------------------------------
Per scope (each IP + OVERALL aggregate):
  - Train direct-horizon models for IOPS using rich lag/time/exogenous features
  - Hold out the last N minutes and compare forecast vs actual on the same chart
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import matplotlib
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests

matplotlib.use("Agg")

# ------------------------------------------------------------------
# CONSTANTS
# ------------------------------------------------------------------
API_BASE = "https://dev-admin-console.zybisys.com/api/admin-api/vm-performance"
SECRET_KEY = "A0tziuB02IrdIS"
OUTPUT_DIR = Path("outputs/network_to_iops")
IST = timezone(timedelta(hours=5, minutes=30))
LAGS_BASE = [1, 2, 3, 6, 12, 24, 60, 120, 240, 480, 1440]


# ------------------------------------------------------------------
# INPUT
# ------------------------------------------------------------------
def ask(prompt: str, default: str) -> str:
    val = input(f"{prompt} [{default}]: ").strip()
    return val if val else default


print("\n=== Network -> IOPS Forecasting ===")
print("Direct-horizon per scope (INDIVIDUAL + OVERALL).\n")
print("Select training mode:")
print("  1  Day-based  — train on a date range, forecast a full day")
print("  2  Hour-based — train on a time range, forecast N hours ahead")
MODE = ask("Mode (1 or 2)", "1").strip()

raw_ips = ask("Enter VM IP address(es), comma-separated", "10.192.1.71")
ip_list = [ip.strip() for ip in raw_ips.split(",") if ip.strip()]

iops_smoothing_on = False
iops_smooth_window = 1

if MODE == "1":
    print("\nFormat: YYYY-MM-DD")
    train_start_str = ask("Train from (date)", (datetime.now(IST) - timedelta(days=8)).strftime("%Y-%m-%d"))
    train_end_str = ask("Train to   (date)", (datetime.now(IST) - timedelta(days=1)).strftime("%Y-%m-%d"))
    forecast_date_str = ask("Forecast for (date)", datetime.now(IST).strftime("%Y-%m-%d"))
    show_actual_min = 60
    show_forecast_min = 1440
else:
    print("\nFormat: YYYY-MM-DD HH:MM")
    train_start_str = ask("Train from", (datetime.now(IST) - timedelta(hours=4)).strftime("%Y-%m-%d %H:%M"))
    train_end_str = ask("Train to  ", (datetime.now(IST) - timedelta(hours=1)).strftime("%Y-%m-%d %H:%M"))
    forecast_hours = float(ask("Forecast duration (hours)", "1"))
    show_forecast_min = max(1, int(forecast_hours * 60))
    show_actual_min = int(ask("Minutes of actual history to show on chart", "60"))


# ------------------------------------------------------------------
# DATA FETCH
# ------------------------------------------------------------------
def date_to_ms(date_str: str, end_of_day: bool = False) -> int:
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    if end_of_day:
        dt = dt.replace(hour=23, minute=59, second=59)
    dt_ist = dt.replace(tzinfo=IST)
    return int(dt_ist.astimezone(timezone.utc).timestamp() * 1000)


def datetime_to_ms(dt_str: str) -> int:
    dt = datetime.strptime(dt_str.strip(), "%Y-%m-%d %H:%M")
    return int(dt.replace(tzinfo=IST).astimezone(timezone.utc).timestamp() * 1000)


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


def _fetch_chunk(ip: str, from_ms: int, to_ms: int, timeout: int = 90) -> list:
    req_url = f"{API_BASE}/{ip}?from={from_ms}&to={to_ms}"
    resp = requests.get(req_url, headers={"X-SECRET-KEY": SECRET_KEY}, timeout=timeout)
    if resp.status_code == 404:
        return []
    resp.raise_for_status()
    payload = resp.json()
    return payload.get("performance_metrics", [])


def fetch_one_ip(ip: str, from_ms: int, to_ms: int, timeout: int = 180) -> pd.DataFrame:
    chunk_ms = 7 * 24 * 3600 * 1000
    metrics: list = []
    cursor = from_ms
    while cursor < to_ms:
        end = min(cursor + chunk_ms, to_ms)
        metrics.extend(_fetch_chunk(ip, cursor, end, timeout=90))
        cursor = end + 1
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
            in_bw += to_float(itf.get("in_bandwidth", 0.0)) / 125_000
            out_bw += to_float(itf.get("out_bandwidth", 0.0)) / 125_000

        rows.append({
            "ip": ip,
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
            "ip", "timestamp", "read_iops", "write_iops", "total_iops",
            "in_bandwidth", "out_bandwidth", "total_network",
            "cpu_percent", "ram_percent", "cpu_load_total", "cpu_load_1", "cpu_load_5",
        ])

    return pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)


if MODE == "1":
    from_ms = date_to_ms(train_start_str, end_of_day=False)
    to_ms = date_to_ms(forecast_date_str, end_of_day=True)
else:
    from_ms = datetime_to_ms(train_start_str)
    to_ms = datetime_to_ms(train_end_str) + show_forecast_min * 60 * 1000

print(f"\nFetching data for {len(ip_list)} IP(s)...")
frames = []
for ip in ip_list:
    try:
        df_ip = fetch_one_ip(ip, from_ms, to_ms)
        if df_ip.empty:
            print(f"  WARNING: no data returned for {ip}, skipping.")
        else:
            print(f"  {ip}: {len(df_ip)} rows")
            frames.append(df_ip)
    except Exception as e:
        print(f"  ERROR for {ip}: {e}")

if not frames:
    print("\nNo data fetched. Check IPs and date range.")
    sys.exit(1)

def minute_align_scope(df_scope: pd.DataFrame) -> pd.DataFrame:
    """Align to 1-minute grid and aggregate duplicates to stabilize lag features."""
    work = df_scope.copy()
    work["minute_ts"] = work["timestamp"].dt.floor("min")
    num_cols = [
        "read_iops", "write_iops", "total_iops",
        "in_bandwidth", "out_bandwidth", "total_network",
        "cpu_percent", "ram_percent", "cpu_load_total", "cpu_load_1", "cpu_load_5",
    ]
    out = work.groupby("minute_ts", as_index=False)[num_cols].mean()
    out = out.rename(columns={"minute_ts": "timestamp"}).sort_values("timestamp").reset_index(drop=True)

    # Fill per scope only to avoid cross-IP leakage.
    fill_cols = ["cpu_percent", "ram_percent", "cpu_load_total", "cpu_load_1", "cpu_load_5"]
    for c in fill_cols:
        out[c] = pd.to_numeric(out[c], errors="coerce").ffill().bfill().fillna(0.0)
    return out


aligned_frames = []
for df_ip in frames:
    aligned = minute_align_scope(df_ip)
    aligned["ip"] = df_ip["ip"].iloc[0]
    aligned_frames.append(aligned)

all_data = pd.concat(aligned_frames, ignore_index=True)

# OVERALL scope built from minute-aligned per-IP data.
overall = (
    all_data.groupby("timestamp", as_index=False)
    .agg({
        "read_iops": "mean",
        "write_iops": "mean",
        "total_iops": "mean",
        "in_bandwidth": "mean",
        "out_bandwidth": "mean",
        "total_network": "mean",
        "cpu_percent": "mean",
        "ram_percent": "mean",
        "cpu_load_total": "mean",
        "cpu_load_1": "mean",
        "cpu_load_5": "mean",
    })
)
overall["ip"] = "OVERALL"
all_data = pd.concat([all_data, overall], ignore_index=True)

# Safety fill by scope, keeps each scope independent.
for c in ["cpu_percent", "ram_percent", "cpu_load_total", "cpu_load_1", "cpu_load_5"]:
    all_data[c] = (
        all_data.groupby("ip", group_keys=False)[c]
        .apply(lambda s: pd.to_numeric(s, errors="coerce").ffill().bfill().fillna(0.0))
    )


# ------------------------------------------------------------------
# FEATURE ENGINEERING
# ------------------------------------------------------------------
def infer_freq_minutes(ts: pd.Series) -> int:
    diffs = ts.sort_values().diff().dropna()
    if diffs.empty:
        return 1
    median_diff = diffs.median()
    freq = int(median_diff.total_seconds() / 60)
    return max(1, freq)


def make_seasonal_lags(n_rows: int, freq_minutes: int) -> list[int]:
    lags = set()
    for x in [1, 2, 3, 5, 10, 15]:
        if x < n_rows:
            lags.add(x)
    for x in [30, 60, 120, 240]:
        if x < n_rows:
            lags.add(x)
    day_lag = int((24 * 60) / freq_minutes)
    if day_lag < n_rows:
        lags.add(day_lag)
    week_lag = int((7 * 24 * 60) / freq_minutes)
    if week_lag < n_rows:
        lags.add(week_lag)
    month_lag = int((30 * 24 * 60) / freq_minutes)
    if month_lag < n_rows:
        lags.add(month_lag)

    if n_rows < 500:
        max_allowed = max(12, n_rows // 4)
        lags = {x for x in lags if x <= max_allowed}

    return sorted(lags)


def time_features(ts: pd.Series) -> pd.DataFrame:
    minute_of_day = ts.dt.hour * 60 + ts.dt.minute
    return pd.DataFrame({
        "hour_sin": np.sin(2 * np.pi * ts.dt.hour / 24.0),
        "hour_cos": np.cos(2 * np.pi * ts.dt.hour / 24.0),
        "dow_sin": np.sin(2 * np.pi * ts.dt.dayofweek / 7.0),
        "dow_cos": np.cos(2 * np.pi * ts.dt.dayofweek / 7.0),
        "minute_of_day_sin": np.sin(2 * np.pi * minute_of_day / 1440.0),
        "minute_of_day_cos": np.cos(2 * np.pi * minute_of_day / 1440.0),
        "is_weekend": (ts.dt.dayofweek >= 5).astype(float),
        "is_business_hour": ((ts.dt.hour >= 9) & (ts.dt.hour < 18) & (ts.dt.dayofweek < 5)).astype(float),
    }, index=ts.index)


def lag_features(series: pd.Series, name: str, lags: list[int]) -> pd.DataFrame:
    out = {}
    for lag in lags:
        out[f"{name}_lag_{lag}"] = series.shift(lag)
    past = series.shift(1)
    ewma15 = past.ewm(span=15, adjust=False).mean()
    ewma60 = past.ewm(span=60, adjust=False).mean()
    out[f"{name}_roll3"] = past.rolling(3, min_periods=1).mean()
    out[f"{name}_roll12"] = past.rolling(12, min_periods=1).mean()
    out[f"{name}_roll24"] = past.rolling(24, min_periods=1).mean()
    out[f"{name}_roll60"] = past.rolling(60, min_periods=1).mean()
    out[f"{name}_rollstd3"] = past.rolling(3, min_periods=2).std().fillna(0.0)
    out[f"{name}_rollstd12"] = past.rolling(12, min_periods=2).std().fillna(0.0)
    out[f"{name}_ewma5"] = past.ewm(span=5, adjust=False).mean()
    out[f"{name}_ewma15"] = ewma15
    out[f"{name}_ewma60"] = ewma60
    out[f"{name}_dev_ewma15"] = past - ewma15
    out[f"{name}_dev_ewma60"] = past - ewma60
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


# ------------------------------------------------------------------
# MODEL
# ------------------------------------------------------------------
try:
    from lightgbm import LGBMRegressor
except ImportError as exc:
    raise SystemExit("LightGBM is required but not installed. Run: pip install lightgbm") from exc


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
) -> tuple[Optional[object], list[str], pd.Series, float]:
    exog = [c for c in exogenous_cols if c != target]
    x_frame = build_design_matrix(df_train, target, exog)
    feature_cols = [c for c in x_frame.columns if c != "timestamp"]
    x_all = x_frame[feature_cols]

    # Defensive safeguard: keep first occurrence if a feature name repeats.
    if x_all.columns.duplicated().any():
        x_all = x_all.loc[:, ~x_all.columns.duplicated()].copy()

    feat_lo = x_all.quantile(0.02)
    feat_hi = x_all.quantile(0.98)
    x_latest = x_all.iloc[[-1]].copy().clip(lower=feat_lo, upper=feat_hi, axis=1)

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
        return None, feature_cols, x_latest.iloc[0], max(0.0, fallback_pred)

    mdl = make_model()
    mdl.fit(x_h, y_h)
    return mdl, feature_cols, x_latest.iloc[0], max(0.0, fallback_pred)


def forecast_direct(
    train_df: pd.DataFrame,
    target: str,
    exogenous_cols: list[str],
    steps: int,
) -> pd.DataFrame:
    mdl, _, latest_x, fallback_pred = train_single_model(train_df, target, exogenous_cols)

    last_ts = train_df["timestamp"].iloc[-1]
    if mdl is None:
        pred_base = fallback_pred
    else:
        y_log = float(mdl.predict(latest_x.to_frame().T)[0])
        pred_base = max(0.0, float(np.expm1(y_log)))

    out = []
    for h in range(1, steps + 1):
        ts = last_ts + pd.Timedelta(minutes=h)
        out.append({"timestamp": ts, "predicted": pred_base})
    return pd.DataFrame(out)


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


# ------------------------------------------------------------------
# PLOTTING
# ------------------------------------------------------------------
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def to_ist(ts_series: pd.Series) -> pd.Series:
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


def plot_scope_chart(
    scope: str,
    history_df: pd.DataFrame,
    compare_df: pd.DataFrame,
    smooth_window: int = 1,
):
    history = history_df[["timestamp", "total_iops"]].sort_values("timestamp").tail(show_actual_min).copy()
    history["ts_ist"] = to_ist(history["timestamp"])

    cmp = compare_df.sort_values("ts_key").head(show_forecast_min).copy()
    cmp["timestamp"] = pd.to_datetime(cmp["ts_key"], utc=True)
    cmp["ts_ist"] = to_ist(cmp["timestamp"])

    if smooth_window > 1:
        history["total_iops"] = history["total_iops"].rolling(smooth_window, min_periods=1).mean()
        cmp["predicted"] = cmp["predicted"].rolling(smooth_window, min_periods=1).mean()
        if "actual" in cmp.columns:
            cmp["actual"] = cmp["actual"].rolling(smooth_window, min_periods=1).mean()

    bridge = pd.DataFrame({
        "ts_ist": [history["ts_ist"].iloc[-1]],
        "predicted": [history["total_iops"].iloc[-1]],
    })
    pred_plot = pd.concat([bridge, cmp[["ts_ist", "predicted"]]], ignore_index=True)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(history["ts_ist"], history["total_iops"], color="green", linewidth=1.8,
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

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=IST))
    fig.autofmt_xdate()
    ax.set_title(f"Network -> IOPS Holdout Actual vs Forecast ({show_forecast_min} min) ({scope})", fontsize=13)
    ax.set_ylabel("IOPS")
    ax.set_xlabel("Time (IST)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    safe_scope = (
        scope.replace(".", "_")
        .replace("/", "_")
        .replace("|", "_")
        .replace(":", "_")
        .replace(" ", "_")
    )
    out_path = OUTPUT_DIR / f"net_to_iops_{safe_scope}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved -> {out_path}")


# ------------------------------------------------------------------
# RUN PER SCOPE
# ------------------------------------------------------------------
scopes = ip_list + (["OVERALL"] if len(ip_list) > 1 else [])
print(f"\nRunning Network -> IOPS for scopes: {scopes}\n")

for scope in scopes:
    scope_df = all_data[all_data["ip"] == scope].sort_values("timestamp").reset_index(drop=True)

    if MODE == "1":
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

        train_df = scope_df[
            (scope_df["timestamp"] >= train_start_utc) & (scope_df["timestamp"] < train_end_utc)
        ].copy().reset_index(drop=True)
        holdout_df = scope_df[
            (scope_df["timestamp"] >= forecast_date_utc) & (scope_df["timestamp"] < forecast_date_end)
        ].copy().reset_index(drop=True)
        comparison_steps = max(len(holdout_df), 1)
        chart_label = f"Forecast for {forecast_date_str}"
    else:
        train_cutoff_ts = pd.Timestamp(datetime_to_ms(train_end_str), unit="ms", tz="UTC")
        train_df = scope_df[scope_df["timestamp"] <= train_cutoff_ts].copy().reset_index(drop=True)
        holdout_df = scope_df[scope_df["timestamp"] > train_cutoff_ts].head(show_forecast_min).copy().reset_index(drop=True)
        comparison_steps = show_forecast_min
        chart_label = f"Forecast - next {comparison_steps} min"

    if len(train_df) < 30:
        print(f"[{scope}] Not enough training rows ({len(train_df)}), skipping.")
        continue

    iops_exog = [
        "read_iops", "write_iops",
        "in_bandwidth", "out_bandwidth", "total_network",
        "cpu_percent", "ram_percent", "cpu_load_total", "cpu_load_1", "cpu_load_5",
    ]

    print(f"[{scope}] Training READ IOPS model ...")
    read_fcast = forecast_direct(
        train_df,
        target="read_iops",
        exogenous_cols=iops_exog,
        steps=comparison_steps,
    )

    print(f"[{scope}] Training WRITE IOPS model ...")
    write_fcast = forecast_direct(
        train_df,
        target="write_iops",
        exogenous_cols=iops_exog,
        steps=comparison_steps,
    )

    iops_fcast = read_fcast.rename(columns={"predicted": "pred_read"}).merge(
        write_fcast.rename(columns={"predicted": "pred_write"}), on="timestamp", how="inner"
    )
    iops_fcast["predicted"] = iops_fcast["pred_read"].fillna(0.0) + iops_fcast["pred_write"].fillna(0.0)
    iops_fcast = iops_fcast[["timestamp", "predicted"]]

    compare_df = align_actual_vs_forecast(iops_fcast, holdout_df, "total_iops")
    mae, mape, matched = mae_mape(compare_df.head(comparison_steps))
    if matched > 0:
        mape_txt = f"{mape:.2f}%" if mape is not None else "NA"
        print(f"[{scope}] Holdout metrics: MAE={mae:.3f}  MAPE={mape_txt}  matched={matched}/{comparison_steps}")
    else:
        print(f"[{scope}] Holdout metrics: no matched actual points")

    # Update label used by chart legend horizon text.
    show_forecast_min = comparison_steps
    plot_scope_chart(f"{scope} | {chart_label}", train_df, compare_df, smooth_window=iops_smooth_window)

print(f"\nDone! All charts saved to: {OUTPUT_DIR.resolve()}")
