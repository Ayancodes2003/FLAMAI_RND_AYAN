"""
Master evaluation script for Flam AI R&D Curve Parameter Estimation.

Runs end-to-end evaluation:
1. Executes Phase 4 geometric solver on xy_data.csv.
2. Evaluates sampling convergence across resolutions N in [500, 1000, 1500, 5000, 10000].
3. Compares Phase 3 (baseline numerical) vs Phase 4 (geometric/analytical) using identical L1 metrics.
4. Saves final_parameters.json and evaluation_convergence.json.
5. Generates publication-quality diagnostic figures in results/figures/.

Usage:
    python scripts/evaluate_solution.py [--data xy_data.csv] [--output-dir results]
"""

import argparse
import json
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.curve import curve, T_MIN, T_MAX
from src.fitting import load_dataset
from src.geometry import geometry_based_estimate, rotate_to_curve_frame, oscillation_nodes
from src.evaluation import (
    sample_uniform_curve,
    evaluate_curve_l1_discrete,
    evaluate_sampling_convergence,
    save_final_parameters,
    save_evaluation_convergence,
)


def generate_diagnostic_plots(
    x_obs: np.ndarray,
    y_obs: np.ndarray,
    theta_deg: float,
    M: float,
    X: float,
    convergence_records: list,
    figures_dir: Path,
) -> list:
    """Generate and save publication-quality diagnostic plots."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    generated_files = []
    
    # 1. Curve Overlay Plot
    fig_path1 = figures_dir / "reconstructed_curve_overlay.png"
    plt.figure(figsize=(10, 6), dpi=300)
    
    # Dense predicted curve
    _, x_dense, y_dense = sample_uniform_curve(theta_deg, M, X, n_samples=2000, degrees=True)
    
    # Centerline line
    t_line = np.linspace(T_MIN, T_MAX, 100)
    x_centerline = X + t_line * np.cos(np.radians(theta_deg))
    y_centerline = 42.0 + t_line * np.sin(np.radians(theta_deg))
    
    # Theoretical nodes
    nodes_t = oscillation_nodes()
    x_nodes = X + nodes_t * np.cos(np.radians(theta_deg))
    y_nodes = 42.0 + nodes_t * np.sin(np.radians(theta_deg))
    
    plt.scatter(x_obs, y_obs, s=8, color="#1f77b4", alpha=0.45, label="Observed Points (N=1500)")
    plt.plot(x_dense, y_dense, color="#d62728", linewidth=2.0, label=f"Reconstructed Curve (θ={theta_deg:.2f}°, M={M:.3f}, X={X:.2f})")
    plt.plot(x_centerline, y_centerline, "--", color="#2ca02c", linewidth=1.5, alpha=0.8, label="Centerline Trend")
    plt.scatter(x_nodes, y_nodes, color="#ff7f0e", s=60, zorder=5, edgecolors="black", label="Oscillation Nodes (k=1..5)")
    
    plt.title("Flam AI R&D Curve Reconstruction vs. Observed Point Cloud", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("x coordinate", fontsize=11)
    plt.ylabel("y coordinate", fontsize=11)
    plt.legend(loc="upper left", framealpha=0.9)
    plt.grid(True, linestyle=":", alpha=0.5)
    plt.tight_layout()
    plt.savefig(fig_path1)
    plt.close()
    generated_files.append(fig_path1)
    
    # 2. L1 Distance Error Distribution Histogram
    fig_path2 = figures_dir / "l1_error_distribution.png"
    plt.figure(figsize=(9, 5.5), dpi=300)
    
    _, x_pred1500, y_pred1500 = sample_uniform_curve(theta_deg, M, X, n_samples=1500, degrees=True)
    eval_res = evaluate_curve_l1_discrete(x_obs, y_obs, x_pred1500, y_pred1500)
    dists = eval_res["dists_obs_to_pred"]
    
    plt.hist(dists, bins=50, color="#3498db", edgecolor="#2980b9", alpha=0.8, density=True)
    plt.axvline(eval_res["mean_obs_to_pred_l1"], color="#e74c3c", linestyle="--", linewidth=2.0,
                label=f"Mean L1 Error = {eval_res['mean_obs_to_pred_l1']:.5f}")
    plt.axvline(eval_res["median_obs_to_pred_l1"], color="#2ecc71", linestyle=":", linewidth=2.0,
                label=f"Median L1 Error = {eval_res['median_obs_to_pred_l1']:.5f}")
    
    plt.title("Distribution of Point-to-Curve Manhattan (L1) Residuals (N=1500)", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("L1 Distance ($|\\Delta x| + |\\Delta y|$)", fontsize=11)
    plt.ylabel("Probability Density", fontsize=11)
    plt.legend(loc="upper right", framealpha=0.9)
    plt.grid(True, linestyle=":", alpha=0.5)
    plt.tight_layout()
    plt.savefig(fig_path2)
    plt.close()
    generated_files.append(fig_path2)
    
    # 3. Residual vs Recovered Parameter t
    fig_path3 = figures_dir / "residual_vs_t.png"
    plt.figure(figsize=(10, 5.5), dpi=300)
    
    u_rec, v_rec = rotate_to_curve_frame(x_obs, y_obs, theta_deg, X, degrees=True)
    v_expected = np.exp(M * u_rec) * np.sin(0.3 * u_rec)
    transverse_res = v_rec - v_expected
    
    plt.scatter(u_rec, transverse_res, s=10, color="#8e44ad", alpha=0.5, label="Observed Transverse Deviation ($v - A(t)$)")
    plt.axhline(0.0, color="black", linestyle="--", linewidth=1.0, alpha=0.7)
    for tk in nodes_t:
        plt.axvline(tk, color="#e67e22", linestyle=":", alpha=0.6, label=f"Node $t={tk:.2f}$" if tk == nodes_t[0] else "")
        
    plt.title("Transverse Residual vs. Recovered Parameter $t$ Across Domain [6, 60]", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Recovered Parameter $t$ (Parallel Projection $u$)", fontsize=11)
    plt.ylabel("Transverse Residual $v(t) - e^{Mt}\\sin(0.3t)$", fontsize=11)
    plt.legend(loc="upper right", framealpha=0.9)
    plt.grid(True, linestyle=":", alpha=0.5)
    plt.tight_layout()
    plt.savefig(fig_path3)
    plt.close()
    generated_files.append(fig_path3)
    
    # 4. Sampling Convergence Plot
    fig_path4 = figures_dir / "sampling_convergence.png"
    plt.figure(figsize=(9, 5.5), dpi=300)
    
    n_vals = [r["N"] for r in convergence_records]
    mean_l1_vals = [r["mean_obs_to_pred_l1"] for r in convergence_records]
    max_l1_vals = [r["max_obs_to_pred_l1"] for r in convergence_records]
    sym_l1_vals = [r["symmetric_mean_l1"] for r in convergence_records]
    
    plt.plot(n_vals, mean_l1_vals, "o-", color="#2980b9", linewidth=2.0, label="Mean Obs $\\to$ Pred L1")
    plt.plot(n_vals, sym_l1_vals, "s--", color="#27ae60", linewidth=2.0, label="Symmetric Mean L1")
    plt.plot(n_vals, max_l1_vals, "^:", color="#c0392b", linewidth=1.5, label="Max Obs $\\to$ Pred L1")
    
    plt.xscale("log")
    plt.yscale("log")
    plt.title("L1 Curve Distance vs. Predicted Curve Sampling Resolution $N$", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Sampling Resolution $N$ (Uniform Samples in $t \\in [6, 60]$)", fontsize=11)
    plt.ylabel("L1 Distance (Manhattan Norm)", fontsize=11)
    plt.xticks(n_vals, [str(n) for n in n_vals])
    plt.legend(loc="upper right", framealpha=0.9)
    plt.grid(True, which="both", linestyle=":", alpha=0.5)
    plt.tight_layout()
    plt.savefig(fig_path4)
    plt.close()
    generated_files.append(fig_path4)
    
    return generated_files


def main():
    parser = argparse.ArgumentParser(description="Run full evaluation and sampling convergence.")
    parser.add_argument("--data", type=str, default="xy_data.csv", help="Path to xy_data.csv")
    parser.add_argument("--output-dir", type=str, default="results", help="Directory for output files")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = out_dir / "figures"

    print("=" * 70)
    print("FLAM AI R&D ASSIGNMENT: FULL SOLUTION EVALUATION PIPELINE")
    print("=" * 70)

    # 1. Load data
    print(f"\n1. Loading observations from: {args.data}")
    x_obs, y_obs = load_dataset(args.data)
    print(f"   Loaded {len(x_obs)} unordered coordinate pairs.")

    # 2. Run Phase 4 Geometry Solver
    print("\n2. Executing Phase 4 Analytical / Geometry-Based Solver...")
    geom_res = geometry_based_estimate(x_obs, y_obs, degrees=True)
    theta_deg = float(geom_res["theta_degrees"])
    theta_rad = float(geom_res["theta_radians"])
    M = float(geom_res["M"])
    X = float(geom_res["X"])

    print(f"   Estimated theta: {theta_deg:.6f} deg ({theta_rad:.6f} rad)")
    print(f"   Estimated M:     {M:.6f}")
    print(f"   Estimated X:     {X:.6f}")
    print(f"   Solver Runtime:  {geom_res['duration_sec']:.4f} s")

    # Load baseline fit if available for comparison
    baseline_path = out_dir / "baseline_fit.json"
    baseline_params = None
    if baseline_path.exists():
        with open(baseline_path, "r", encoding="utf-8") as f:
            baseline_params = json.load(f)

    # 3. Save Final Parameters JSON
    final_params_payload = {
        "theta_deg": theta_deg,
        "theta_rad": theta_rad,
        "M": M,
        "X": X,
        "t_min": float(T_MIN),
        "t_max": float(T_MAX),
        "solver_method": "Analytical Orthonormal Geometric Decomposition",
        "transverse_residual_l1": float(geom_res["transverse_residual_l1"]),
        "valid_points_for_M": int(geom_res["valid_points_for_M"]),
        "total_points": int(geom_res["total_points"]),
        "duration_sec": float(geom_res["duration_sec"]),
    }
    if baseline_params is not None:
        final_params_payload["baseline_comparison"] = {
            "theta_deg": float(baseline_params["theta_degrees"]),
            "M": float(baseline_params["M"]),
            "X": float(baseline_params["X"]),
            "optimization_method": baseline_params.get("optimization_method", "Nelder-Mead (Multi-Start)"),
            "total_duration_sec": float(baseline_params.get("total_duration_sec", 0.0)),
        }

    final_param_path = out_dir / "final_parameters.json"
    save_final_parameters(final_params_payload, final_param_path)
    print(f"   Final parameters saved to: {final_param_path}")

    # 4. Sampling Convergence Analysis
    print("\n3. Evaluating Sampling Convergence across N in [500, 1000, 1500, 5000, 10000]...")
    resolutions = [500, 1000, 1500, 5000, 10000]
    convergence_records = evaluate_sampling_convergence(
        x_obs, y_obs, theta_deg, M, X, resolutions=resolutions, degrees=True
    )
    
    conv_path = out_dir / "evaluation_convergence.json"
    save_evaluation_convergence(convergence_records, conv_path)
    print(f"   Convergence table saved to: {conv_path}")

    print("\n   SAMPLING CONVERGENCE TABLE:")
    print("   " + "-" * 78)
    print(f"   {'N':>6} | {'Mean Obs->Pred L1':>18} | {'Max Obs->Pred L1':>17} | {'Symmetric Mean L1':>18} | {'Time (ms)':>9}")
    print("   " + "-" * 78)
    for r in convergence_records:
        print(f"   {r['N']:6d} | {r['mean_obs_to_pred_l1']:18.6f} | {r['max_obs_to_pred_l1']:17.6f} | {r['symmetric_mean_l1']:18.6f} | {r['runtime_ms']:9.2f}")
    print("   " + "-" * 78)

    # 5. Same-Metric Method Comparison
    print("\n4. Method Comparison (Evaluated on Identical L1 Metric at N=1500 samples):")
    print("   " + "-" * 88)
    print(f"   {'Method':<32} | {'theta (deg)':>11} | {'M':>9} | {'X':>9} | {'Curve Mean L1':>14} | {'Runtime':>8}")
    print("   " + "-" * 88)
    
    # Phase 4
    _, x_p4, y_p4 = sample_uniform_curve(theta_deg, M, X, n_samples=1500, degrees=True)
    eval_p4 = evaluate_curve_l1_discrete(x_obs, y_obs, x_p4, y_p4)
    print(f"   {'Phase 4: Geometric Solver':<32} | {theta_deg:11.6f} | {M:9.6f} | {X:9.6f} | {eval_p4['mean_obs_to_pred_l1']:14.6f} | {geom_res['duration_sec']:7.3f}s")
    
    # Phase 3
    if baseline_params is not None:
        th_p3 = float(baseline_params["theta_degrees"])
        m_p3 = float(baseline_params["M"])
        x_p3 = float(baseline_params["X"])
        _, x_p3_curve, y_p3_curve = sample_uniform_curve(th_p3, m_p3, x_p3, n_samples=1500, degrees=True)
        eval_p3 = evaluate_curve_l1_discrete(x_obs, y_obs, x_p3_curve, y_p3_curve)
        t_p3 = baseline_params.get("total_duration_sec", 0.0)
        print(f"   {'Phase 3: Numerical Baseline':<32} | {th_p3:11.6f} | {m_p3:9.6f} | {x_p3:9.6f} | {eval_p3['mean_obs_to_pred_l1']:14.6f} | {t_p3:7.3f}s")
    print("   " + "-" * 88)

    # 6. Generate Diagnostic Figures
    print("\n5. Generating diagnostic figures...")
    fig_files = generate_diagnostic_plots(
        x_obs, y_obs, theta_deg, M, X, convergence_records, figures_dir
    )
    for ff in fig_files:
        print(f"   Created: {ff}")

    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE: All artifacts successfully generated.")
    print("=" * 70)


if __name__ == "__main__":
    main()
