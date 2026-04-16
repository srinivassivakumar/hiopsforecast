from __future__ import annotations

import argparse

from src.config import load_config
from src.pipeline import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="File-based forecasting pipeline")
    parser.add_argument("--config", required=True, help="Path to config yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    result = run_pipeline(cfg.raw)

    print("Pipeline completed.")
    print(f"Output directory: {result['output_dir']}")
    print(f"Metrics file: {result['metrics_file']}")
    print(f"Forecast file: {result['forecast_file']}")
    print(f"Correlation file: {result['correlation_file']}")
    print(f"Charts directory: {result['charts_dir']}")
    print(f"Charts generated: {result['charts_count']}")


if __name__ == "__main__":
    main()
