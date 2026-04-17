"""
Network → IOPS Forecasting
---------------------------
Concept:
  Uses NETWORK BANDWIDTH (in_bw + out_bw) as input features to PREDICT DISK IOPS.
  The model learns the relationship: "when network is high, IOPS tends to be X".

  Two projection views are produced per run:
    - INDIVIDUAL : one model trained per IP address
    - OVERALL    : all IPs aggregated by timestamp (mean), one model trained on the combined view

Steps:
  1. Ask user for IP(s), date range, forecast minutes
  2. Fetch data from API and convert ctime → UTC → IST for display
  3. Build feature matrix  X = [lagged in_bw, out_bw, total_network] + time features
                           y = total_iops
  4. Train LightGBM (or RandomForest fallback) per scope
  5. Forecast future IOPS:
       Step A – forecast network forward N steps using a separate time-series model
       Step B – feed those predicted network values into the IOPS model each step
  6. Plot Individual charts (one per IP) + one Overall chart
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

# ─────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────
API_BASE   = "https://dev-admin-console.zybisys.com/api/admin-api/vm-performance"
SECRET_KEY = "A0tziuB02IrdIS"
OUTPUT_DIR = Path("outputs/network_to_iops")
IST        = timezone(timedelta(hours=5, minutes=30))
MAX_LAG    = 24
LAGS       = [l for l in (1, 2, 3, 6, 12, 24) if l <= MAX_LAG]

# ─────────────────────────────────────────────────────────────────
# 1. USER INPUT
# ─────────────────────────────────────────────────────────────────

def ask(prompt: str, default: str) -> str:
    val = input(f"{prompt} [{default}]: ").strip()
    return val if val else default


print("\n=== Network → IOPS Forecasting ===")
print("Model uses network bandwidth as features to predict disk IOPS.\n")

raw_ips       = ask("Enter VM IP address(es), comma-separated", "10.192.1.71")
ip_list       = [ip.strip() for ip in raw_ips.split(",") if ip.strip()]

print("\nDate range for training data (YYYY-MM-DD, treated as IST midnight):")
start_str     = ask("Start date", (datetime.now(IST) - timedelta(days=2)).strftime("%Y-%m-%d"))
end_str       = ask("End date  ", datetime.now(IST).strftime("%Y-%m-%d"))
forecast_min  = int(ask("\nForecast horizon (minutes ahead to predict)", "15"))
show_actual   = int(ask("Minutes of actual data to show on chart    ", "15"))

# ─────────────────────────────────────────────────────────────────
# 2. FETCH DATA
# ─────────────────────────────────────────────────────────────────

def date_to_ms(date_str: str, end_of_day: bool = False) -> int:
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    if end_of_day:
        dt = dt.replace(hour=23, minute=59, second=59)
    return int(dt.replace(tzinfo=IST).astimezone(timezone.utc).timestamp() * 1000)


def to_float(v, default: float = 0.0) -> float:
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


def fetch_one_ip(ip: str, from_ms: int, to_ms: int) -> pd.DataFrame:
    url = f"{API_BASE}/{ip}?from={from_ms}&to={to_ms}"
    print(f"  Fetching {ip} → {url}")
    resp = requests.get(url, headers={"X-SECRET-KEY": SECRET_KEY}, timeout=60)
    resp.raise_for_status()
    payload = resp.json()

    rows = []
    for item in payload.get("performance_metrics", []):
        ts_raw = dig(item, "disk_io_summary.io_operations_data.time_str")
        if ts_raw is None:
            continue
        ts_int = int(to_float(ts_raw))
        ts = datetime.fromtimestamp(
            ts_int / 1000.0 if ts_int > 10_000_000_000 else ts_int,
            tz=timezone.utc,
        )
        read_iops  = to_float(dig(item, "disk_io_summary.io_operations_data.read_data",  0.0))
        write_iops = to_float(dig(item, "disk_io_summary.io_operations_data.write_data", 0.0))
        in_bw = out_bw = 0.0
        for itf in (item.get("interface") or []):
            in_bw  += to_float(itf.get("in_bandwidth",  0.0))
            out_bw += to_float(itf.get("out_bandwidth", 0.0))
        rows.append({
            "ip":            ip,
            "timestamp":     ts,
            "in_bandwidth":  in_bw,
            "out_bandwidth": out_bw,
            "total_network": in_bw + out_bw,
            "read_iops":     read_iops,
            "write_iops":    write_iops,
            "total_iops":    read_iops + write_iops,
        })
    return pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)


from_ms = date_to_ms(start_str, end_of_day=False)
to_ms   = date_to_ms(end_str,   end_of_day=True)

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

all_data = pd.concat(frames, ignore_index=True)

# Build OVERALL scope: mean across all IPs at each timestamp
overall = (
    all_data.groupby("timestamp", as_index=False)
    .agg({"in_bandwidth": "mean", "out_bandwidth": "mean",
          "total_network": "mean", "total_iops": "mean",
          "read_iops": "mean", "write_iops": "mean"})
)
overall["ip"] = "OVERALL"
all_data = pd.concat([all_data, overall], ignore_index=True)

# ─────────────────────────────────────────────────────────────────
# 3. FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────

def time_features(ts: pd.Series) -> pd.DataFrame:
    return pd.DataFrame({
        "hour_sin": np.sin(2 * np.pi * ts.dt.hour / 24.0),
        "hour_cos": np.cos(2 * np.pi * ts.dt.hour / 24.0),
        "dow_sin":  np.sin(2 * np.pi * ts.dt.dayofweek / 7.0),
        "dow_cos":  np.cos(2 * np.pi * ts.dt.dayofweek / 7.0),
        "minute":   ts.dt.minute,
    }, index=ts.index)


def lag_features(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Build lag + rolling features for each column in cols."""
    out = {}
    for col in cols:
        for lag in LAGS:
            out[f"{col}_lag{lag}"] = df[col].shift(lag)
        out[f"{col}_roll3"]  = df[col].rolling(3,  min_periods=1).mean()
        out[f"{col}_roll12"] = df[col].rolling(12, min_periods=1).mean()
    return pd.DataFrame(out, index=df.index)


def build_net_to_iops_frame(scope_df: pd.DataFrame) -> pd.DataFrame:
    """
    X = lagged network columns + time features
    y = total_iops
    """
    tf = time_features(scope_df["timestamp"])
    lf = lag_features(scope_df, ["in_bandwidth", "out_bandwidth", "total_network"])
    frame = pd.concat(
        [scope_df[["timestamp", "total_iops"]].rename(columns={"total_iops": "y"}), tf, lf],
        axis=1,
    )
    return frame.dropna().reset_index(drop=True)


def build_ts_frame(scope_df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Simple time-series frame for a single column (used to forecast network)."""
    tf = time_features(scope_df["timestamp"])
    lf = lag_features(scope_df, [col])
    frame = pd.concat(
        [scope_df[["timestamp", col]].rename(columns={col: "y"}), tf, lf],
        axis=1,
    )
    return frame.dropna().reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────
# 4. MODEL
# ─────────────────────────────────────────────────────────────────

try:
    from lightgbm import LGBMRegressor
    def make_model():
        return LGBMRegressor(
            n_estimators=600, learning_rate=0.05, num_leaves=31,
            subsample=0.9, colsample_bytree=0.9, random_state=42, n_jobs=-1,
        )
    print("\nUsing LightGBM")
except Exception:
    from sklearn.ensemble import RandomForestRegressor
    def make_model():
        return RandomForestRegressor(n_estimators=300, max_depth=12, random_state=42, n_jobs=-1)
    print("\nUsing RandomForest (LightGBM not installed)")


def train_model(frame: pd.DataFrame, test_fraction: float = 0.2):
    feat_cols = [c for c in frame.columns if c not in {"timestamp", "y"}]
    split = max(1, int(len(frame) * (1 - test_fraction)))
    model = make_model()
    model.fit(frame.iloc[:split][feat_cols], frame.iloc[:split]["y"])
    test  = frame.iloc[split:]
    pred  = model.predict(test[feat_cols])
    bt = test[["timestamp"]].copy()
    bt["actual"]    = test["y"].values
    bt["predicted"] = pred
    mae  = float(np.mean(np.abs(test["y"].values - pred)))
    mape = float(np.mean(np.abs((test["y"].values - pred) / np.clip(np.abs(test["y"].values), 1e-9, None))) * 100)
    return model, bt, mae, mape


# ─────────────────────────────────────────────────────────────────
# 5. FORECAST
#    Step A: forecast network N steps ahead using time-series model
#    Step B: use those predicted network values to forecast IOPS
# ─────────────────────────────────────────────────────────────────

def forecast_network_col(scope_df: pd.DataFrame, col: str, model, steps: int) -> list[float]:
    """Recursively forecast a single network column N steps ahead.

    To prevent recursive drift the predicted value is anchored so it
    cannot stray more than 50% from the last *observed* value.  This
    keeps the network signal realistic over a short horizon.
    """
    work = scope_df[["timestamp", col]].copy().reset_index(drop=True)
    last_observed = float(work[col].iloc[-1])       # anchor = last real value
    lo = last_observed * 0.5
    hi = last_observed * 1.5 + 1e-6                # +epsilon avoids lo==hi==0
    preds = []
    for _ in range(steps):
        next_ts = work["timestamp"].iloc[-1] + pd.Timedelta(minutes=1)
        temp = pd.concat(
            [work, pd.DataFrame({"timestamp": [next_ts], col: [np.nan]})],
            ignore_index=True,
        )
        frame = build_ts_frame(temp, col)
        feat_cols = [c for c in frame.columns if c not in {"timestamp", "y"}]
        y_hat = float(model.predict(frame[feat_cols].iloc[[-1]])[0])
        y_hat = float(np.clip(y_hat, lo, hi))       # anchor to last-observed range
        work.loc[len(work)] = {"timestamp": next_ts, col: y_hat}
        preds.append(y_hat)
    return preds


def forecast_iops_from_network(
    scope_df: pd.DataFrame,
    iops_model,
    net_in_model,
    net_out_model,
    steps: int,
) -> pd.DataFrame:
    """
    Step A: forecast in_bandwidth and out_bandwidth N steps ahead (anchored).
    Step B: for each future step, build features using *predicted network* but
            freeze the IOPS lag slots at the last real observed IOPS value.
            This breaks the self-compounding IOPS→IOPS feedback loop that
            causes predictions to collapse toward zero.
    Returns DataFrame with columns: timestamp, predicted_iops,
            predicted_in_bw, predicted_out_bw.
    """
    # Step A – forecast network columns (drift-anchored)
    in_preds  = forecast_network_col(scope_df, "in_bandwidth",  net_in_model,  steps)
    out_preds = forecast_network_col(scope_df, "out_bandwidth", net_out_model, steps)

    # Step B – build a working copy where we append predicted network rows
    # but keep IOPS frozen at the last real value so lags stay stable.
    last_real_iops = float(scope_df["total_iops"].iloc[-1])
    work = scope_df[["timestamp", "in_bandwidth", "out_bandwidth", "total_network", "total_iops"]].copy().reset_index(drop=True)
    results = []

    for i in range(steps):
        next_ts = work["timestamp"].iloc[-1] + pd.Timedelta(minutes=1)
        p_in    = in_preds[i]
        p_out   = out_preds[i]
        p_net   = p_in + p_out

        # Append the next network step; fill IOPS with last real value
        # so the lag features see a stable baseline, not a decaying value.
        next_row = pd.DataFrame([{
            "timestamp":     next_ts,
            "in_bandwidth":  p_in,
            "out_bandwidth": p_out,
            "total_network": p_net,
            "total_iops":    last_real_iops,  # frozen baseline — no self-feedback
        }])
        temp = pd.concat([work, next_row], ignore_index=True)

        frame     = build_net_to_iops_frame(temp)
        feat_cols = [c for c in frame.columns if c not in {"timestamp", "y"}]
        iops_hat  = max(0.0, float(iops_model.predict(frame[feat_cols].iloc[[-1]])[0]))

        # Do NOT write iops_hat back — keep frozen baseline to avoid drift
        work = pd.concat([work, next_row], ignore_index=True)

        results.append({
            "timestamp":       next_ts,
            "predicted_iops":  iops_hat,
            "predicted_in_bw": p_in,
            "predicted_out_bw":p_out,
        })

    return pd.DataFrame(results)


# ─────────────────────────────────────────────────────────────────
# 6. PLOT
# ─────────────────────────────────────────────────────────────────

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def to_ist(ts: pd.Series) -> pd.Series:
    if ts.dt.tz is None:
        ts = ts.dt.tz_localize("UTC")
    return ts.dt.tz_convert(IST)


def plot_scope(scope: str, scope_df: pd.DataFrame, fcast: pd.DataFrame):
    """
    Single figure, dual Y-axis:
      Left  axis (green) — actual network usage over the last N minutes.
      Right axis (blue)  — HIOPS forecast for the next N minutes,
                           continuing from where the network segment ends.
    Each metric keeps its own natural scale so neither line collapses.
    """
    recent = scope_df.sort_values("timestamp").tail(show_actual).copy()
    recent["ts_ist"] = to_ist(recent["timestamp"])

    fc = fcast.copy()
    fc["ts_ist"] = to_ist(fc["timestamp"])

    fig, ax_net = plt.subplots(figsize=(13, 5.5))
    ax_iops = ax_net.twinx()   # second Y-axis shares the same X-axis

    # ── Green: network usage (left axis) ─────────────────────────
    ax_net.plot(
        recent["ts_ist"],
        recent["total_network"],
        color="green",
        lw=2.3,
        label=f"Network usage ({show_actual} min)",
    )
    ax_net.set_ylabel("Network bandwidth (Mbps)", color="green")
    ax_net.tick_params(axis="y", labelcolor="green")

    # ── Blue: HIOPS forecast (right axis) ────────────────────────
    # Bridge point: first blue point lines up at last green timestamp
    # using the last real IOPS value so the join is smooth.
    last_real_iops = float(scope_df.sort_values("timestamp")["total_iops"].iloc[-1])
    bridge = pd.DataFrame({
        "ts_ist":        [recent["ts_ist"].iloc[-1]],
        "predicted_iops":[last_real_iops],
    })
    fc_plot = pd.concat([bridge, fc[["ts_ist", "predicted_iops"]]], ignore_index=True)

    ax_iops.plot(
        fc_plot["ts_ist"],
        fc_plot["predicted_iops"],
        color="blue",
        lw=2.3,
        label=f"HIOPS forecast (next {forecast_min} min)",
    )
    ax_iops.set_ylabel("Disk IOPS", color="blue")
    ax_iops.tick_params(axis="y", labelcolor="blue")

    # ── Shared formatting ─────────────────────────────────────────
    ax_net.set_title(
        f"Network usage  →  HIOPS forecast  ({scope})",
        fontsize=12, fontweight="bold",
    )
    ax_net.set_xlabel("Time (IST)")
    ax_net.grid(alpha=0.25)
    ax_net.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=IST))

    # Combined legend from both axes
    lines1, labels1 = ax_net.get_legend_handles_labels()
    lines2, labels2 = ax_iops.get_legend_handles_labels()
    ax_net.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc="upper left")

    fig.autofmt_xdate()
    fig.tight_layout()
    safe_scope = scope.replace(".", "_").replace("/", "_")
    out_path   = OUTPUT_DIR / f"net_to_iops_{safe_scope}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved → {out_path}")


# ─────────────────────────────────────────────────────────────────
# 7. RUN PER SCOPE
# ─────────────────────────────────────────────────────────────────

scopes = ip_list + ["OVERALL"]
print(f"\nRunning network → IOPS for scopes: {scopes}\n")

for scope in scopes:
    scope_df = all_data[all_data["ip"] == scope].sort_values("timestamp").reset_index(drop=True)

    if len(scope_df) < 50:
        print(f"[{scope}] Not enough rows ({len(scope_df)}), skipping.")
        continue

    print(f"[{scope}] {len(scope_df)} rows  |  training network→IOPS model ...")

    # Train IOPS model (features = network lags)
    iops_frame = build_net_to_iops_frame(scope_df)
    if len(iops_frame) < 30:
        print(f"[{scope}] Not enough rows after feature engineering, skipping.")
        continue

    iops_model, backtest, mae, mape = train_model(iops_frame)
    print(f"[{scope}] IOPS model  MAE={mae:.2f}  MAPE={mape:.1f}%")

    # Train network forecasting models (needed to project future inputs)
    print(f"[{scope}] Training in_bandwidth  time-series model ...")
    in_frame = build_ts_frame(scope_df, "in_bandwidth")
    in_model, _, _, _ = train_model(in_frame)

    print(f"[{scope}] Training out_bandwidth time-series model ...")
    out_frame = build_ts_frame(scope_df, "out_bandwidth")
    out_model, _, _, _ = train_model(out_frame)

    # Forecast IOPS using predicted network as input
    print(f"[{scope}] Generating {forecast_min}-step IOPS forecast ...")
    fcast = forecast_iops_from_network(scope_df, iops_model, in_model, out_model, steps=forecast_min)

    # Plot
    plot_scope(scope, scope_df, fcast)

print(f"\nDone! All charts saved to: {OUTPUT_DIR.resolve()}")
