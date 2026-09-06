"""
Executable script for running the Phase 3 baseline numerical fitting pipeline.

Usage:
    python scripts/fit_curve.py [--data xy_data.csv] [--output results/baseline_fit.json]
"""

import argparse
import sys
from pathlib import Path

# Ensure repo root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.fitting import (
    load_dataset,
    fit_baseline_multistart,
    save_baseline_results,
    summarize_fit,
)


def main():
    parser = argparse.ArgumentParser(description="Fit baseline parametric curve parameters.")
    parser.add_argument(
        "--data",
        type=str,
        default="xy_data.csv",
        help="Path to observed points CSV file (default: xy_data.csv)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/baseline_fit.json",
        help="Path to output JSON file (default: results/baseline_fit.json)",
    )
    parser.add_argument(
        "--grid-size",
        type=int,
        default=500,
        help="Number of discrete t-points for projection grid (default: 500)",
    )
    args = parser.parse_args()

    print(f"Loading observation dataset from: {args.data}")
    x_obs, y_obs = load_dataset(args.data)
    print(f"Loaded {len(x_obs)} coordinate points.")

    print("\nRunning multi-start baseline numerical optimization across parameter bounds...")
    results = fit_baseline_multistart(
        x_obs=x_obs,
        y_obs=y_obs,
        n_grid=args.grid_size,
        degrees=True,
        method="Nelder-Mead",
    )

    summary_text = summarize_fit(results)
    print("\n" + summary_text)

    out_path = Path(args.output)
    save_baseline_results(results, out_path)
    print(f"\nResults successfully saved to: {out_path.resolve()}")


if __name__ == "__main__":
    main()
