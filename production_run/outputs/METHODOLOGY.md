# Forecasting Methodology

- Data source: unified time-indexed server performance data with IOPS and network fields.
- Preprocessing: timestamp normalization, numeric coercion, missing-row drop, and per-scope sorting.
- Time-series models: RandomForestRegressor with time/lag/rolling features per metric.
- Network-to-IOPS model: RandomForestRegressor using lagged network signals to predict total_iops.
- Horizons: short_term and long_term as configured in config.yaml.
- Scopes: individual per ip and OVERALL aggregate (mean across servers by timestamp).
- Evaluation: MAE, RMSE, MAPE, SMAPE on holdout split.
- Correlation insights: Pearson and Spearman lag correlation between total_network and total_iops.
- Forecast export: visualization-ready JSON with timestamp, scope, metric, model family, predicted value, and bounds.
