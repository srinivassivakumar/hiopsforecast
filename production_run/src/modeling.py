from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

try:
    from lightgbm import LGBMRegressor
    HAS_LIGHTGBM = True
except Exception:
    LGBMRegressor = None  # type: ignore[assignment]
    HAS_LIGHTGBM = False


@dataclass
class ForecastConfig:
    max_lag_steps: int
    short_horizon_steps: int
    long_horizon_steps: int
    test_fraction: float


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    denom = np.where(denom == 0, 1.0, denom)
    return float(np.mean(np.abs(y_true - y_pred) / denom) * 100)


def _time_features(ts: pd.Series) -> pd.DataFrame:
    hour = ts.dt.hour
    dow = ts.dt.dayofweek
    minute = ts.dt.minute

    return pd.DataFrame(
        {
            "hour_sin": np.sin(2 * np.pi * hour / 24.0),
            "hour_cos": np.cos(2 * np.pi * hour / 24.0),
            "dow_sin": np.sin(2 * np.pi * dow / 7.0),
            "dow_cos": np.cos(2 * np.pi * dow / 7.0),
            "minute": minute,
        },
        index=ts.index,
    )


def _lag_features(df: pd.DataFrame, columns: List[str], max_lag: int) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for col in columns:
        for lag in (1, 2, 3, 6, 12, 24):
            if lag <= max_lag:
                out[f"{col}_lag_{lag}"] = df[col].shift(lag)
        out[f"{col}_roll_3"] = df[col].rolling(3, min_periods=1).mean()
        out[f"{col}_roll_12"] = df[col].rolling(12, min_periods=1).mean()
    return out


def build_timeseries_frame(df: pd.DataFrame, target_col: str, max_lag: int) -> pd.DataFrame:
    base = pd.DataFrame(index=df.index)
    base["timestamp"] = df["timestamp"]
    base["y"] = df[target_col]

    tf = _time_features(df["timestamp"])
    lf = _lag_features(df, [target_col], max_lag)
    frame = pd.concat([base, tf, lf], axis=1)
    return frame.dropna().reset_index(drop=True)


def build_network_to_iops_frame(df: pd.DataFrame, max_lag: int) -> pd.DataFrame:
    base = pd.DataFrame(index=df.index)
    base["timestamp"] = df["timestamp"]
    base["y"] = df["total_iops"]

    tf = _time_features(df["timestamp"])
    lf = _lag_features(df, ["in_bandwidth", "out_bandwidth", "total_network"], max_lag)
    frame = pd.concat([base, tf, lf], axis=1)
    return frame.dropna().reset_index(drop=True)


def _make_regressor() -> Any:
    """Build model. Prefer LightGBM; fallback to RandomForest if unavailable."""
    if HAS_LIGHTGBM:
        return LGBMRegressor(
            objective="regression",
            n_estimators=600,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
            n_jobs=-1,
        )

    return RandomForestRegressor(
        n_estimators=300,
        max_depth=12,
        random_state=42,
        n_jobs=-1,
    )


def train_and_score(frame: pd.DataFrame, test_fraction: float) -> Tuple[Any, Dict[str, float], pd.DataFrame]:
    feature_cols = [c for c in frame.columns if c not in {"timestamp", "y"}]
    split = max(1, int(len(frame) * (1 - test_fraction)))
    split = min(split, len(frame) - 1)

    train = frame.iloc[:split]
    test = frame.iloc[split:]

    model = _make_regressor()
    model.fit(train[feature_cols], train["y"])

    pred = model.predict(test[feature_cols])
    metrics = {
        "mae": float(mean_absolute_error(test["y"], pred)),
        "rmse": float(np.sqrt(mean_squared_error(test["y"], pred))),
        "mape": float(np.mean(np.abs((test["y"] - pred) / np.clip(np.abs(test["y"]), 1e-9, None))) * 100),
        "smape": smape(test["y"].to_numpy(), pred),
        "n_train": int(len(train)),
        "n_test": int(len(test)),
    }

    scored = test[["timestamp", "y"]].copy()
    scored["predicted"] = pred
    return model, metrics, scored


def recursive_forecast_timeseries(df: pd.DataFrame, target_col: str, model: Any, steps: int, max_lag: int, freq_minutes: int) -> pd.DataFrame:
    work = df[["timestamp", target_col]].copy().reset_index(drop=True)
    forecasts = []

    for _ in range(steps):
        next_ts = work["timestamp"].iloc[-1] + pd.Timedelta(minutes=freq_minutes)
        temp = pd.DataFrame({"timestamp": pd.concat([work["timestamp"], pd.Series([next_ts])], ignore_index=True), target_col: pd.concat([work[target_col], pd.Series([np.nan])], ignore_index=True)})

        frame = build_timeseries_frame(temp, target_col=target_col, max_lag=max_lag)
        x_next = frame.drop(columns=["timestamp", "y"]).iloc[-1:]
        y_hat = float(model.predict(x_next)[0])
        y_hat = max(0.0, y_hat)

        work.loc[len(work)] = {"timestamp": next_ts, target_col: y_hat}
        forecasts.append({"timestamp": next_ts, "predicted": y_hat})

    return pd.DataFrame(forecasts)


def recursive_forecast_network_to_iops(df: pd.DataFrame, model: Any, steps: int, max_lag: int, freq_minutes: int) -> pd.DataFrame:
    work = df[["timestamp", "in_bandwidth", "out_bandwidth", "total_network", "total_iops"]].copy().reset_index(drop=True)
    last_in = float(work["in_bandwidth"].iloc[-1])
    last_out = float(work["out_bandwidth"].iloc[-1])

    forecasts = []
    for _ in range(steps):
        next_ts = work["timestamp"].iloc[-1] + pd.Timedelta(minutes=freq_minutes)
        next_row = {
            "timestamp": next_ts,
            "in_bandwidth": last_in,
            "out_bandwidth": last_out,
            "total_network": last_in + last_out,
            "total_iops": np.nan,
        }
        temp = pd.concat([work, pd.DataFrame([next_row])], ignore_index=True)
        frame = build_network_to_iops_frame(temp, max_lag=max_lag)
        x_next = frame.drop(columns=["timestamp", "y"]).iloc[-1:]
        y_hat = float(model.predict(x_next)[0])
        y_hat = max(0.0, y_hat)

        next_row["total_iops"] = y_hat
        work = pd.concat([work, pd.DataFrame([next_row])], ignore_index=True)
        forecasts.append({"timestamp": next_ts, "predicted": y_hat})

    return pd.DataFrame(forecasts)


def lag_correlation(df: pd.DataFrame, x_col: str, y_col: str, max_lag: int = 24) -> pd.DataFrame:
    rows = []
    base = df[[x_col, y_col]].dropna().copy()
    for lag in range(-max_lag, max_lag + 1):
        shifted = base.copy()
        shifted["x"] = shifted[x_col].shift(lag)
        shifted = shifted.dropna()
        if len(shifted) < 3:
            continue
        pearson = shifted["x"].corr(shifted[y_col], method="pearson")
        spearman = shifted["x"].corr(shifted[y_col], method="spearman")
        rows.append({"lag_steps": lag, "pearson": float(pearson), "spearman": float(spearman), "n": int(len(shifted))})

    return pd.DataFrame(rows)
