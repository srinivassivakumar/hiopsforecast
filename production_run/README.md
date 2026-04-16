# Production Run

This is a production-only minimal forecasting package.

## Included
- API ingestion for disk IOPS and VM performance
- Forecasting models:
  - Time-series per metric
  - Network-to-IOPS
- Individual and OVERALL scope forecasts
- JSON outputs only
- PNG dashboards

## Run

1. Set real values in `config.yaml` (`secret_key`, `ips`, and API mappings if needed).
2. Execute:

```powershell
.\run_production.ps1
```

## Output Files

Generated under `outputs/`:
- `dataset_unified.json`
- `metrics.json`
- `backtest_predictions.json`
- `forecast_output.json`
- `correlation_insights.json`
- `METHODOLOGY.md`
- `charts/*.png`

## Removed For Production

This folder intentionally excludes demo/sample generators and non-runtime files.
