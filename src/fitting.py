"""
Baseline numerical fitting module for unordered parametric curve observations.

Problem Statement:
    Given unordered observation points (x_i, y_i), i = 1, ..., N, estimate:
        theta (0 < theta < 50 deg)
        M     (-0.05 < M < 0.05)
        X     (0 < X < 100)
    for curve(t) with 6 < t < 60.

Fitting Objective:
    For an unordered point cloud without known t labels, the baseline objective
    is the mean point-to-curve L1 (Manhattan) distance:
        min_{theta, M, X} (1 / N) * sum_{i=1}^N min_{t in [6, 60]} [ |x_i - x(t)| + |y_i - y(t)| ]

Optimization Strategy:
    - Efficient two-stage point-to-curve mapping:
      1. Vectorized broadcast evaluation across a discrete grid of t in [6, 60].
      2. Optional scalar continuous refinement per point.
    - Robust multi-start local optimization across deterministic grid initializations
      covering the entire bounded domain [0, 50] x [-0.05, 0.05] x [0, 100].
"""

from typing import Dict, List, Optional, Tuple, Union
import json
import os
import time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar

from src.curve import (
    curve,
    to_radians,
    to_degrees,
    validate_parameters,
    THETA_MIN_DEG,
    THETA_MAX_DEG,
    M_MIN,
    M_MAX,
    X_MIN,
    X_MAX,
    T_MIN,
    T_MAX,
)


def load_dataset(filepath: Union[str, Path] = "xy_data.csv") -> Tuple[np.ndarray, np.ndarray]:
    """
    Load observed (x, y) coordinates from a CSV file.
    
    Parameters
    ----------
    filepath : str or Path
        Path to the CSV file.
        
    Returns
    -------
    tuple of (np.ndarray, np.ndarray)
        Arrays of observed x and y coordinates.
        
    Raises
    ------
    FileNotFoundError
        If the dataset file does not exist.
    ValueError
        If expected columns 'x' and 'y' are missing.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found at: {path.resolve()}")
        
    df = pd.read_csv(path)
    if "x" not in df.columns or "y" not in df.columns:
        raise ValueError(f"Dataset must contain 'x' and 'y' columns, got: {list(df.columns)}")
        
    x_obs = df["x"].to_numpy(dtype=np.float64)
    y_obs = df["y"].to_numpy(dtype=np.float64)
    return x_obs, y_obs


def estimate_closest_t(
    x_obs: np.ndarray,
    y_obs: np.ndarray,
    theta: float,
    M: float,
    X: float,
    n_grid: int = 500,
    degrees: bool = True,
    refine: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    For each observed point (x_i, y_i), estimate the parameter t in [6, 60]
    that minimizes the L1 distance to the curve.
    
    Parameters
    ----------
    x_obs : np.ndarray
        Observed x coordinates (N,).
    y_obs : np.ndarray
        Observed y coordinates (N,).
    theta : float
        Rotation angle parameter.
    M : float
        Exponential growth parameter.
    X : float
        X-offset parameter.
    n_grid : int, default=500
        Number of discrete evaluation points in t in [6, 60].
    degrees : bool, default=True
        Whether theta is provided in degrees.
    refine : bool, default=False
        If True, performs local continuous 1D scalar refinement around grid minimum.
        
    Returns
    -------
    tuple of (np.ndarray, np.ndarray)
        - t_est: Estimated t parameter per point (N,).
        - min_distances: Minimum L1 distance per point (N,).
    """
    th_rad = to_radians(theta, degrees=degrees)
    t_grid = np.linspace(T_MIN, T_MAX, n_grid)
    
    # Evaluate curve across t_grid: shapes (n_grid,)
    x_c, y_c = curve(t_grid, th_rad, M, X, degrees=False)
    
    # Broadcast pairwise L1 distance: shape (N, n_grid)
    dx = np.abs(x_obs[:, np.newaxis] - x_c[np.newaxis, :])
    dy = np.abs(y_obs[:, np.newaxis] - y_c[np.newaxis, :])
    d_l1 = dx + dy
    
    best_indices = np.argmin(d_l1, axis=1)
    min_distances = d_l1[np.arange(len(x_obs)), best_indices]
    t_est = t_grid[best_indices]
    
    if not refine:
        return t_est, min_distances
        
    # Local continuous refinement per point
    refined_t = np.empty_like(t_est)
    refined_dist = np.empty_like(min_distances)
    
    for i in range(len(x_obs)):
        idx = best_indices[i]
        t_lo = max(T_MIN, t_grid[max(0, idx - 1)])
        t_hi = min(T_MAX, t_grid[min(n_grid - 1, idx + 1)])
        
        target_x = x_obs[i]
        target_y = y_obs[i]
        
        def local_obj(t_val: float) -> float:
            xc, yc = curve(t_val, th_rad, M, X, degrees=False)
            return abs(target_x - xc) + abs(target_y - yc)
            
        res = minimize_scalar(
            local_obj,
            bracket=(t_lo, t_grid[idx], t_hi),
            bounds=(t_lo, t_hi),
            method="bounded",
            options={"xatol": 1e-5},
        )
        refined_t[i] = res.x
        refined_dist[i] = res.fun
        
    return refined_t, refined_dist


def baseline_l1_objective(
    params: Union[List[float], np.ndarray],
    x_obs: np.ndarray,
    y_obs: np.ndarray,
    n_grid: int = 500,
    degrees: bool = True,
    reduction: str = "mean",
) -> float:
    """
    Compute the baseline L1 point-to-curve fitting objective for unordered points.
    
    Parameters
    ----------
    params : array-like
        [theta, M, X] parameter values.
    x_obs : np.ndarray
        Observed x coordinates (N,).
    y_obs : np.ndarray
        Observed y coordinates (N,).
    n_grid : int, default=500
        Number of points in discrete t grid for projection.
    degrees : bool, default=True
        Whether theta is in degrees.
    reduction : str, default='mean'
        'mean' or 'sum' reduction across all points.
        
    Returns
    -------
    float
        Computed objective loss value.
    """
    theta, M, X = params
    
    # Check bounds; penalize invalid regions heavily for unconstrained optimizer calls
    if degrees:
        theta_deg = theta
    else:
        theta_deg = to_degrees(theta)
        
    if not (THETA_MIN_DEG < theta_deg < THETA_MAX_DEG and M_MIN < M < M_MAX and X_MIN < X < X_MAX):
        # Smooth barrier penalty outside feasible domain
        penalty = 1e4 + 1e3 * (
            max(0.0, THETA_MIN_DEG - theta_deg) ** 2
            + max(0.0, theta_deg - THETA_MAX_DEG) ** 2
            + max(0.0, 100 * (M_MIN - M)) ** 2
            + max(0.0, 100 * (M - M_MAX)) ** 2
            + max(0.0, X_MIN - X) ** 2
            + max(0.0, X - X_MAX) ** 2
        )
        return float(penalty)
        
    _, min_dists = estimate_closest_t(
        x_obs, y_obs, theta=theta, M=M, X=X, n_grid=n_grid, degrees=degrees, refine=False
    )
    
    if reduction == "mean":
        return float(np.mean(min_dists))
    elif reduction == "sum":
        return float(np.sum(min_dists))
    elif reduction == "median":
        return float(np.median(min_dists))
    else:
        raise ValueError(f"Unknown reduction: {reduction}. Choose 'mean', 'sum', or 'median'.")


def get_default_multi_starts() -> List[List[float]]:
    """
    Generate deterministic, diverse initial parameter seeds [theta_deg, M, X]
    spanning the constrained hypercube.
    
    Returns
    -------
    list of list of float
        Deterministic initial parameter vectors in degrees.
    """
    return [
        [10.0, -0.02, 20.0],
        [25.0, 0.00, 50.0],
        [40.0, 0.02, 80.0],
        [15.0, 0.03, 60.0],
        [35.0, -0.03, 40.0],
        [45.0, 0.04, 90.0],
    ]


def fit_baseline_multistart(
    x_obs: np.ndarray,
    y_obs: np.ndarray,
    initial_guesses: Optional[List[List[float]]] = None,
    n_grid: int = 500,
    degrees: bool = True,
    method: str = "Nelder-Mead",
) -> Dict[str, Union[float, int, bool, str, list]]:
    """
    Fit baseline curve parameters using multi-start local optimization.
    
    Parameters
    ----------
    x_obs : np.ndarray
        Observed x coordinates.
    y_obs : np.ndarray
        Observed y coordinates.
    initial_guesses : list of list of float, optional
        Custom list of [theta, M, X] initial parameter sets.
    n_grid : int, default=500
        Number of grid points for internal t estimation.
    degrees : bool, default=True
        Whether parameters are treated in degrees.
    method : str, default='Nelder-Mead'
        SciPy optimization method.
        
    Returns
    -------
    dict
        Dictionary containing best parameters, multi-start run details, and runtime.
    """
    if initial_guesses is None:
        initial_guesses = get_default_multi_starts()
        
    bounds = [
        (THETA_MIN_DEG + 0.1, THETA_MAX_DEG - 0.1),
        (M_MIN + 0.001, M_MAX - 0.001),
        (X_MIN + 0.5, X_MAX - 0.5),
    ] if degrees else [
        (np.radians(THETA_MIN_DEG + 0.1), np.radians(THETA_MAX_DEG - 0.1)),
        (M_MIN + 0.001, M_MAX - 0.001),
        (X_MIN + 0.5, X_MAX - 0.5),
    ]
    
    start_time = time.time()
    run_records = []
    best_fun = float("inf")
    best_params = None
    best_run_info = None
    
    for i, x0 in enumerate(initial_guesses):
        t_start = time.time()
        res = minimize(
            baseline_l1_objective,
            x0=x0,
            args=(x_obs, y_obs, n_grid, degrees, "mean"),
            method=method,
            bounds=bounds,
            options={"maxiter": 600, "xatol": 1e-4, "fatol": 1e-5},
        )
        t_run = time.time() - t_start
        
        theta_val = float(res.x[0])
        m_val = float(res.x[1])
        x_val = float(res.x[2])
        
        record = {
            "run_index": i + 1,
            "initial_params": [float(p) for p in x0],
            "fitted_params": [theta_val, m_val, x_val],
            "objective_value": float(res.fun),
            "nfev": int(res.nfev),
            "nit": int(getattr(res, "nit", res.nfev)),
            "success": bool(res.success),
            "status_message": str(res.message),
            "duration_sec": float(t_run),
        }
        run_records.append(record)
        
        if res.fun < best_fun:
            best_fun = float(res.fun)
            best_params = res.x
            best_run_info = record
            
    total_time = time.time() - start_time
    
    if best_params is None:
        raise RuntimeError("All multi-start optimization runs failed to converge.")
        
    best_theta_deg = float(best_params[0]) if degrees else to_degrees(float(best_params[0]))
    best_theta_rad = to_radians(float(best_params[0]), degrees=True) if degrees else float(best_params[0])
    best_m = float(best_params[1])
    best_x = float(best_params[2])
    
    # Compute refined continuous metric at the fitted solution
    _, refined_dists = estimate_closest_t(
        x_obs, y_obs, theta=best_theta_deg, M=best_m, X=best_x, n_grid=1000, degrees=True, refine=True
    )
    refined_mean_l1 = float(np.mean(refined_dists))
    
    return {
        "theta_degrees": best_theta_deg,
        "theta_radians": best_theta_rad,
        "M": best_m,
        "X": best_x,
        "grid_objective_value": best_fun,
        "continuous_l1_mean": refined_mean_l1,
        "optimization_method": f"{method} (Multi-Start)",
        "num_runs": len(initial_guesses),
        "total_duration_sec": float(total_time),
        "all_runs": run_records,
        "best_run": best_run_info,
    }


def save_baseline_results(
    results: dict,
    output_path: Union[str, Path] = "results/baseline_fit.json",
) -> None:
    """
    Save baseline optimization results to a JSON file.
    
    Parameters
    ----------
    results : dict
        Results dictionary from fit_baseline_multistart.
    output_path : str or Path
        Target filepath.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)


def summarize_fit(results: dict) -> str:
    """
    Generate a formatted human-readable summary of baseline fitting results.
    
    Parameters
    ----------
    results : dict
        Results dictionary.
        
    Returns
    -------
    str
        Formatted summary string.
    """
    lines = [
        "=" * 50,
        "BASELINE NUMERICAL FIT SUMMARY",
        "=" * 50,
        f"theta: {results['theta_degrees']:.6f} degrees ({results['theta_radians']:.6f} rad)",
        f"M:     {results['M']:.6f}",
        f"X:     {results['X']:.6f}",
        "-" * 50,
        f"Grid Mean L1 Objective:       {results['grid_objective_value']:.6f}",
        f"Continuous Mean L1 Objective: {results['continuous_l1_mean']:.6f}",
        f"Optimization Method:          {results['optimization_method']}",
        f"Runs Completed:               {results['num_runs']}",
        f"Total Runtime:                {results['total_duration_sec']:.2f} s",
        "=" * 50,
    ]
    return "\n".join(lines)
