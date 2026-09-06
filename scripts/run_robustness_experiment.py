"""
Robustness, Noise Sensitivity & Limitation Analysis Experiment.

Evaluates:
1. Noise Sensitivity: Gaussian perturbations sigma in [0, 0.001, 0.005, 0.01, 0.05, 0.10]
   with 10 independent trials per noise level.
2. Parameter Sensitivity: One-at-a-time perturbations around ground truth (theta, M, X).
3. Observation Density: Recovery accuracy across N in [100, 250, 500, 1000, 1500].
4. Limitation Regimes: Extreme noise (sigma up to 2.0), severe sparsity (N < 50), and boundary M.
5. Saves results to results/robustness_experiment.json and diagnostic figures to results/figures/.

Usage:
    python scripts/run_robustness_experiment.py [--output-dir results]
"""

import argparse
import json
import sys
import time
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Ensure repo root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.curve import curve, T_MIN, T_MAX
from src.geometry import geometry_based_estimate
from src.evaluation import sample_uniform_curve, evaluate_curve_l1_discrete

# Ground truth reference parameters
THETA_TRUE = 30.0
M_TRUE = 0.03
X_TRUE = 55.0


def run_noise_experiments(
    sigmas=(0.0, 0.001, 0.005, 0.01, 0.05, 0.10),
    n_trials_per_sigma=10,
    base_seed=100,
    n_obs=1500,
) -> dict:
    """Run multi-trial noise sensitivity experiments across sigma levels."""
    t_clean = np.linspace(T_MIN, T_MAX, n_obs)
    x_clean, y_clean = curve(t_clean, THETA_TRUE, M_TRUE, X_TRUE, degrees=True)
    _, x_gt, y_gt = sample_uniform_curve(THETA_TRUE, M_TRUE, X_TRUE, n_samples=1500, degrees=True)

    summary_by_sigma = []
    all_trial_records = []

    for sigma in sigmas:
        n_trials = 1 if sigma == 0.0 else n_trials_per_sigma
        th_errors, m_errors, x_errors, l1_errors = [], [], [], []

        for trial_idx in range(n_trials):
            seed = base_seed + trial_idx
            rng = np.random.RandomState(seed)

            if sigma > 0.0:
                x_noisy = x_clean + rng.normal(0.0, sigma, size=len(x_clean))
                y_noisy = y_clean + rng.normal(0.0, sigma, size=len(y_clean))
            else:
                x_noisy = x_clean.copy()
                y_noisy = y_clean.copy()

            perm = rng.permutation(len(x_noisy))
            x_shuf, y_shuf = x_noisy[perm], y_noisy[perm]

            fit = geometry_based_estimate(x_shuf, y_shuf, degrees=True)
            th_est = float(fit["theta_degrees"])
            m_est = float(fit["M"])
            x_est = float(fit["X"])

            err_th = abs(th_est - THETA_TRUE)
            err_m = abs(m_est - M_TRUE)
            err_x = abs(x_est - X_TRUE)

            _, x_pred, y_pred = sample_uniform_curve(th_est, m_est, x_est, n_samples=1500, degrees=True)
            eval_res = evaluate_curve_l1_discrete(x_gt, y_gt, x_pred, y_pred)
            curve_l1 = float(eval_res["mean_obs_to_pred_l1"])

            th_errors.append(err_th)
            m_errors.append(err_m)
            x_errors.append(err_x)
            l1_errors.append(curve_l1)

            all_trial_records.append({
                "sigma": float(sigma),
                "trial": int(trial_idx + 1),
                "seed": int(seed),
                "theta_est": th_est,
                "M_est": m_est,
                "X_est": x_est,
                "err_theta": err_th,
                "err_M": err_m,
                "err_X": err_x,
                "curve_l1_error": curve_l1,
            })

        summary_by_sigma.append({
            "sigma": float(sigma),
            "num_trials": int(n_trials),
            "theta_error_mean": float(np.mean(th_errors)),
            "theta_error_median": float(np.median(th_errors)),
            "theta_error_std": float(np.std(th_errors)),
            "theta_error_max": float(np.max(th_errors)),
            "M_error_mean": float(np.mean(m_errors)),
            "M_error_median": float(np.median(m_errors)),
            "M_error_std": float(np.std(m_errors)),
            "M_error_max": float(np.max(m_errors)),
            "X_error_mean": float(np.mean(x_errors)),
            "X_error_median": float(np.median(x_errors)),
            "X_error_std": float(np.std(x_errors)),
            "X_error_max": float(np.max(x_errors)),
            "curve_l1_mean": float(np.mean(l1_errors)),
            "curve_l1_median": float(np.median(l1_errors)),
            "curve_l1_std": float(np.std(l1_errors)),
            "curve_l1_max": float(np.max(l1_errors)),
        })

    return {"summary_by_sigma": summary_by_sigma, "all_trials": all_trial_records}


def run_parameter_sensitivity_experiments() -> dict:
    """Evaluate one-at-a-time parameter perturbation sensitivity against ground truth."""
    _, x_gt, y_gt = sample_uniform_curve(THETA_TRUE, M_TRUE, X_TRUE, n_samples=1500, degrees=True)

    theta_vals = [29.0, 29.5, 29.9, 30.0, 30.1, 30.5, 31.0]
    m_vals = [0.025, 0.028, 0.029, 0.030, 0.031, 0.032, 0.035]
    x_vals = [54.0, 54.5, 54.9, 55.0, 55.1, 55.5, 56.0]

    res_theta = []
    for th in theta_vals:
        _, xp, yp = sample_uniform_curve(th, M_TRUE, X_TRUE, n_samples=1500, degrees=True)
        eval_r = evaluate_curve_l1_discrete(x_gt, y_gt, xp, yp)
        res_theta.append({
            "theta": float(th),
            "delta_theta": float(th - THETA_TRUE),
            "curve_l1_error": float(eval_r["mean_obs_to_pred_l1"]),
        })

    res_m = []
    for m in m_vals:
        _, xp, yp = sample_uniform_curve(THETA_TRUE, m, X_TRUE, n_samples=1500, degrees=True)
        eval_r = evaluate_curve_l1_discrete(x_gt, y_gt, xp, yp)
        res_m.append({
            "M": float(m),
            "delta_M": float(m - M_TRUE),
            "curve_l1_error": float(eval_r["mean_obs_to_pred_l1"]),
        })

    res_x = []
    for x in x_vals:
        _, xp, yp = sample_uniform_curve(THETA_TRUE, M_TRUE, x, n_samples=1500, degrees=True)
        eval_r = evaluate_curve_l1_discrete(x_gt, y_gt, xp, yp)
        res_x.append({
            "X": float(x),
            "delta_X": float(x - X_TRUE),
            "curve_l1_error": float(eval_r["mean_obs_to_pred_l1"]),
        })

    return {"theta_sensitivity": res_theta, "M_sensitivity": res_m, "X_sensitivity": res_x}


def run_observation_density_experiments(
    density_levels=(100, 250, 500, 1000, 1500),
    seed=42,
) -> list:
    """Evaluate noiseless parameter recovery across observation counts N."""
    records = []
    rng = np.random.RandomState(seed)

    for n_obs in density_levels:
        t_obs = np.linspace(T_MIN, T_MAX, n_obs)
        x_c, y_c = curve(t_obs, THETA_TRUE, M_TRUE, X_TRUE, degrees=True)
        perm = rng.permutation(len(x_c))

        fit = geometry_based_estimate(x_c[perm], y_c[perm], degrees=True)
        th_est = float(fit["theta_degrees"])
        m_est = float(fit["M"])
        x_est = float(fit["X"])

        records.append({
            "N": int(n_obs),
            "theta_est": th_est,
            "M_est": m_est,
            "X_est": x_est,
            "err_theta": float(abs(th_est - THETA_TRUE)),
            "err_M": float(abs(m_est - M_TRUE)),
            "err_X": float(abs(x_est - X_TRUE)),
            "transverse_residual": float(fit["transverse_residual_l1"]),
        })
    return records


def run_limitation_analysis() -> dict:
    """Analyze behavior under extreme noise, severe sparsity, and boundary parameters."""
    rng = np.random.RandomState(100)

    # 1. Extreme noise
    extreme_noise = []
    t_clean = np.linspace(T_MIN, T_MAX, 1500)
    x_clean, y_clean = curve(t_clean, THETA_TRUE, M_TRUE, X_TRUE, degrees=True)
    for sigma in [0.25, 0.50, 1.0, 2.0]:
        xn = x_clean + rng.normal(0, sigma, len(x_clean))
        yn = y_clean + rng.normal(0, sigma, len(y_clean))
        p = rng.permutation(len(xn))
        fit = geometry_based_estimate(xn[p], yn[p], degrees=True)
        extreme_noise.append({
            "sigma": float(sigma),
            "err_theta": float(abs(fit["theta_degrees"] - THETA_TRUE)),
            "err_M": float(abs(fit["M"] - M_TRUE)),
            "err_X": float(abs(fit["X"] - X_TRUE)),
        })

    # 2. Extreme sparsity (N < 100)
    extreme_sparsity = []
    for n_pts in [10, 20, 30, 50, 75]:
        t_sp = np.linspace(T_MIN, T_MAX, n_pts)
        xc, yc = curve(t_sp, THETA_TRUE, M_TRUE, X_TRUE, degrees=True)
        p = rng.permutation(len(xc))
        fit = geometry_based_estimate(xc[p], yc[p], degrees=True)
        extreme_sparsity.append({
            "N": int(n_pts),
            "err_theta": float(abs(fit["theta_degrees"] - THETA_TRUE)),
            "err_M": float(abs(fit["M"] - M_TRUE)),
            "err_X": float(abs(fit["X"] - X_TRUE)),
        })

    # 3. Boundary M values
    boundary_m = []
    for m_val in [-0.048, -0.040, -0.010, 0.000, 0.010, 0.040, 0.048]:
        xc, yc = curve(t_clean, THETA_TRUE, m_val, X_TRUE, degrees=True)
        p = rng.permutation(len(xc))
        fit = geometry_based_estimate(xc[p], yc[p], degrees=True)
        boundary_m.append({
            "M_true": float(m_val),
            "err_M": float(abs(fit["M"] - m_val)),
            "err_theta": float(abs(fit["theta_degrees"] - THETA_TRUE)),
        })

    return {
        "extreme_noise": extreme_noise,
        "extreme_sparsity": extreme_sparsity,
        "boundary_M": boundary_m,
    }


def generate_robustness_figures(
    noise_summary: list,
    sensitivity_res: dict,
    density_records: list,
    figures_dir: Path,
) -> list:
    """Generate and save publication-quality diagnostic plots for robustness experiments."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    generated = []

    # 1. Parameter Errors vs Noise Sigma
    fig1 = figures_dir / "noise_parameter_error.png"
    plt.figure(figsize=(9, 5.5), dpi=300)
    sigmas = [r["sigma"] for r in noise_summary]
    err_th = [r["theta_error_mean"] for r in noise_summary]
    err_m = [r["M_error_mean"] * 100 for r in noise_summary]  # scale for visibility
    err_x = [r["X_error_mean"] for r in noise_summary]

    plt.plot(sigmas, err_th, "o-", color="#e74c3c", linewidth=2.0, label="|Δθ| (degrees)")
    plt.plot(sigmas, err_x, "s--", color="#2980b9", linewidth=2.0, label="|ΔX| (units)")
    plt.plot(sigmas, err_m, "^:", color="#27ae60", linewidth=2.0, label="|ΔM| × 100")

    plt.title("Parameter Estimation Error vs. Gaussian Noise Level σ (10 Trials/Level)", fontsize=12, fontweight="bold", pad=12)
    plt.xlabel("Gaussian Noise Standard Deviation σ", fontsize=11)
    plt.ylabel("Mean Absolute Parameter Error", fontsize=11)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="upper left", framealpha=0.9)
    plt.tight_layout()
    plt.savefig(fig1)
    plt.close()
    generated.append(fig1)

    # 2. Reconstruction L1 Error vs Noise Sigma
    fig2 = figures_dir / "noise_reconstruction_error.png"
    plt.figure(figsize=(9, 5.5), dpi=300)
    l1_mean = [r["curve_l1_mean"] for r in noise_summary]
    l1_max = [r["curve_l1_max"] for r in noise_summary]
    l1_std = [r["curve_l1_std"] for r in noise_summary]

    plt.errorbar(sigmas, l1_mean, yerr=l1_std, fmt="o-", color="#8e44ad", linewidth=2.0, capsize=5, label="Mean Curve L1 Error ± 1σ")
    plt.plot(sigmas, l1_max, "^--", color="#c0392b", linewidth=1.5, alpha=0.7, label="Max Trial Curve L1 Error")

    plt.title("Reconstructed Curve L1 Distance vs. Observation Noise Level σ", fontsize=12, fontweight="bold", pad=12)
    plt.xlabel("Noise Level σ", fontsize=11)
    plt.ylabel("Reconstruction L1 Error (units)", fontsize=11)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="upper left", framealpha=0.9)
    plt.tight_layout()
    plt.savefig(fig2)
    plt.close()
    generated.append(fig2)

    # 3. Observation Density
    fig3 = figures_dir / "observation_density.png"
    plt.figure(figsize=(9, 5.5), dpi=300)
    n_pts = [r["N"] for r in density_records]
    eth_dense = [max(r["err_theta"], 1e-12) for r in density_records]
    ex_dense = [max(r["err_X"], 1e-12) for r in density_records]
    em_dense = [max(r["err_M"], 1e-12) for r in density_records]

    plt.plot(n_pts, eth_dense, "o-", color="#e74c3c", linewidth=2.0, label="|Δθ|")
    plt.plot(n_pts, ex_dense, "s--", color="#2980b9", linewidth=2.0, label="|ΔX|")
    plt.plot(n_pts, em_dense, "^:", color="#27ae60", linewidth=2.0, label="|ΔM|")

    plt.title("Parameter Recovery Error vs. Number of Observations N (Noiseless)", fontsize=12, fontweight="bold", pad=12)
    plt.xlabel("Number of Observation Points N", fontsize=11)
    plt.ylabel("Absolute Parameter Error", fontsize=11)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="upper right", framealpha=0.9)
    plt.tight_layout()
    plt.savefig(fig3)
    plt.close()
    generated.append(fig3)

    # 4. Parameter Sensitivity - Theta
    fig4 = figures_dir / "parameter_sensitivity_theta.png"
    plt.figure(figsize=(8, 5), dpi=300)
    d_th = [r["delta_theta"] for r in sensitivity_res["theta_sensitivity"]]
    l1_th = [r["curve_l1_error"] for r in sensitivity_res["theta_sensitivity"]]
    plt.plot(d_th, l1_th, "o-", color="#e74c3c", linewidth=2.0)
    plt.title("Sensitivity: Curve L1 Error vs. Perturbation in θ around 30°", fontsize=12, fontweight="bold", pad=12)
    plt.xlabel("Δθ (degrees)", fontsize=11)
    plt.ylabel("Reconstruction L1 Error", fontsize=11)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(fig4)
    plt.close()
    generated.append(fig4)

    # 5. Parameter Sensitivity - M
    fig5 = figures_dir / "parameter_sensitivity_M.png"
    plt.figure(figsize=(8, 5), dpi=300)
    d_m = [r["delta_M"] for r in sensitivity_res["M_sensitivity"]]
    l1_m = [r["curve_l1_error"] for r in sensitivity_res["M_sensitivity"]]
    plt.plot(d_m, l1_m, "s-", color="#27ae60", linewidth=2.0)
    plt.title("Sensitivity: Curve L1 Error vs. Perturbation in M around 0.03", fontsize=12, fontweight="bold", pad=12)
    plt.xlabel("ΔM", fontsize=11)
    plt.ylabel("Reconstruction L1 Error", fontsize=11)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(fig5)
    plt.close()
    generated.append(fig5)

    # 6. Parameter Sensitivity - X
    fig6 = figures_dir / "parameter_sensitivity_X.png"
    plt.figure(figsize=(8, 5), dpi=300)
    d_x = [r["delta_X"] for r in sensitivity_res["X_sensitivity"]]
    l1_x = [r["curve_l1_error"] for r in sensitivity_res["X_sensitivity"]]
    plt.plot(d_x, l1_x, "^-", color="#2980b9", linewidth=2.0)
    plt.title("Sensitivity: Curve L1 Error vs. Perturbation in X around 55.0", fontsize=12, fontweight="bold", pad=12)
    plt.xlabel("ΔX", fontsize=11)
    plt.ylabel("Reconstruction L1 Error", fontsize=11)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(fig6)
    plt.close()
    generated.append(fig6)

    return generated


def main():
    parser = argparse.ArgumentParser(description="Run complete robustness and sensitivity analysis.")
    parser.add_argument("--output-dir", type=str, default="results", help="Target results directory")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = out_dir / "figures"

    print("=" * 70)
    print("PHASE 6: ROBUSTNESS, NOISE SENSITIVITY & LIMITATION ANALYSIS")
    print("=" * 70)

    t_start = time.time()

    # Part B & C: Noise experiments
    print("\n1. Running Controlled Noise Experiments (sigma in [0, 0.10], 10 trials/level)...")
    noise_results = run_noise_experiments()
    print("   Completed noise trials.")

    # Part D: Parameter sensitivity
    print("\n2. Running Parameter Sensitivity Analysis...")
    sensitivity_results = run_parameter_sensitivity_experiments()
    print("   Completed sensitivity analysis.")

    # Part E: Observation density
    print("\n3. Running Observation Density Analysis (N in [100, 1500])...")
    density_results = run_observation_density_experiments()
    print("   Completed density analysis.")

    # Part F: Limitation analysis
    print("\n4. Running Boundary and Failure Regime Analysis...")
    limitation_results = run_limitation_analysis()
    print("   Completed limitation analysis.")

    # Compile structured payload
    payload = {
        "ground_truth_parameters": {"theta_deg": THETA_TRUE, "M": M_TRUE, "X": X_TRUE},
        "noise_experiment": noise_results,
        "parameter_sensitivity": sensitivity_results,
        "observation_density": density_results,
        "limitation_analysis": limitation_results,
        "total_runtime_sec": float(time.time() - t_start),
    }

    # Save JSON
    json_path = out_dir / "robustness_experiment.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=4)
    print(f"\n   Saved structured data to: {json_path.resolve()}")

    # Generate figures
    print("\n5. Generating diagnostic figures in results/figures/...")
    fig_files = generate_robustness_figures(
        noise_results["summary_by_sigma"],
        sensitivity_results,
        density_results,
        figures_dir,
    )
    for ff in fig_files:
        print(f"   Created: {ff.resolve()}")

    # Print summary tables
    print("\n" + "=" * 78)
    print("NOISE ROBUSTNESS SUMMARY TABLE (10 Trials per Non-Zero sigma):")
    print("=" * 78)
    print(f"{'sigma':>6} | {'Mean |d_theta| (deg)':>20} | {'Mean |d_M|':>12} | {'Mean |d_X|':>12} | {'Mean Curve L1':>15}")
    print("-" * 78)
    for r in noise_results["summary_by_sigma"]:
        print(f"{r['sigma']:6.3f} | {r['theta_error_mean']:20.6f} | {r['M_error_mean']:12.6f} | {r['X_error_mean']:12.6f} | {r['curve_l1_mean']:15.6f}")
    print("=" * 78)

    print("\n" + "=" * 70)
    print("PHASE 6 ROBUSTNESS EXPERIMENT COMPLETE.")
    print("=" * 70)


if __name__ == "__main__":
    main()
