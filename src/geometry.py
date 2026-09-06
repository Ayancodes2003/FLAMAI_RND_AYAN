"""
Geometry-based analytical parameter estimation module.

Mathematical Formulation:
    The parametric curve is decomposed in an orthonormal basis {a, b}:
        r(t) = [X, 42]^T + t * a + A(t) * b
    where:
        a = [cos(theta), sin(theta)]^T   (parallel / longitudinal drift axis)
        b = [-sin(theta), cos(theta)]^T  (perpendicular / transverse oscillation axis)
        A(t) = exp(M * t) * sin(0.3 * t)

Key Geometric Properties:
    1. Parallel Projection (t recovery):
       u(t) = (r(t) - [X, 42]^T) . a = (x - X)*cos(theta) + (y - 42)*sin(theta) = t
    2. Perpendicular Projection (transverse envelope recovery):
       v(t) = (r(t) - [X, 42]^T) . b = -(x - X)*sin(theta) + (y - 42)*cos(theta) = exp(M * t) * sin(0.3 * t)
    3. Oscillation Nodes:
       A(t_k) = 0 <=> sin(0.3 * t_k) = 0 <=> t_k = k * pi / 0.3
       For t in (6, 60), nodes occur at t_k in {10.472, 20.944, 31.416, 41.888, 52.360}.
    4. Closed-form M Estimation:
       For points where |sin(0.3 * u)| > threshold:
           |v / sin(0.3 * u)| = exp(M * u) => ln|v / sin(0.3 * u)| = M * u
       M is obtained via linear regression through the origin.
"""

from typing import Dict, List, Optional, Tuple, Union
import json
import time
from pathlib import Path
import numpy as np
from scipy.optimize import minimize

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
    OMEGA,
    Y_OFFSET,
)


def basis_vectors(theta: float, degrees: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute orthonormal basis vectors a (parallel) and b (perpendicular).
    
    Parameters
    ----------
    theta : float
        Rotation angle.
    degrees : bool, default=False
        Whether theta is in degrees.
        
    Returns
    -------
    tuple of (np.ndarray, np.ndarray)
        a = [cos(theta), sin(theta)], b = [-sin(theta), cos(theta)]
    """
    th_rad = to_radians(theta, degrees=degrees)
    a = np.array([np.cos(th_rad), np.sin(th_rad)], dtype=np.float64)
    b = np.array([-np.sin(th_rad), np.cos(th_rad)], dtype=np.float64)
    return a, b


def rotate_to_curve_frame(
    x: np.ndarray,
    y: np.ndarray,
    theta: float,
    X: float,
    degrees: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Project 2D observation coordinates into the intrinsic curve frame (u, v).
    
    Parameters
    ----------
    x : np.ndarray
        Observed x coordinates.
    y : np.ndarray
        Observed y coordinates.
    theta : float
        Rotation angle.
    X : float
        X-offset parameter.
    degrees : bool, default=True
        Whether theta is in degrees.
        
    Returns
    -------
    tuple of (np.ndarray, np.ndarray)
        - u: Parallel projection coordinate (recovers parameter t).
        - v: Perpendicular projection coordinate (recovers transverse amplitude).
    """
    th_rad = to_radians(theta, degrees=degrees)
    cos_th = np.cos(th_rad)
    sin_th = np.sin(th_rad)
    
    dx = x - X
    dy = y - Y_OFFSET
    
    u = dx * cos_th + dy * sin_th
    v = -dx * sin_th + dy * cos_th
    return u, v


def project_parallel(
    x: np.ndarray,
    y: np.ndarray,
    theta: float,
    X: float,
    degrees: bool = True,
) -> np.ndarray:
    """Compute parallel projection coordinate u = (x - X)*cos(theta) + (y - 42)*sin(theta)."""
    u, _ = rotate_to_curve_frame(x, y, theta, X, degrees=degrees)
    return u


def project_perpendicular(
    x: np.ndarray,
    y: np.ndarray,
    theta: float,
    X: float,
    degrees: bool = True,
) -> np.ndarray:
    """Compute perpendicular projection coordinate v = -(x - X)*sin(theta) + (y - 42)*cos(theta)."""
    _, v = rotate_to_curve_frame(x, y, theta, X, degrees=degrees)
    return v


def oscillation_nodes(t_min: float = T_MIN, t_max: float = T_MAX) -> np.ndarray:
    """
    Calculate theoretical oscillation node parameter values t_k where sin(0.3 * t) = 0.
    
    Parameters
    ----------
    t_min : float, default=6.0
        Lower domain bound.
    t_max : float, default=60.0
        Upper domain bound.
        
    Returns
    -------
    np.ndarray
        Array of node t_k values within (t_min, t_max).
    """
    k_min = int(np.ceil(t_min * OMEGA / np.pi))
    k_max = int(np.floor(t_max * OMEGA / np.pi))
    k_vals = np.arange(k_min, k_max + 1)
    return k_vals * np.pi / OMEGA


def estimate_M_from_amplitude(
    u: np.ndarray,
    v: np.ndarray,
    sin_threshold: float = 0.15,
) -> Tuple[float, int, Dict[str, float]]:
    """
    Estimate the exponential growth parameter M from intrinsic coordinates (u, v).
    
    Uses closed-form log-envelope regression through the origin:
        ln|v / sin(0.3 * u)| = M * u
    Points near sinusoidal zero-crossings (|sin(0.3 * u)| <= sin_threshold) are
    excluded to guarantee numerical stability.
    
    Parameters
    ----------
    u : np.ndarray
        Parallel projection coordinates (t).
    v : np.ndarray
        Perpendicular projection coordinates (A(t)).
    sin_threshold : float, default=0.15
        Threshold below which near-node points are excluded.
        
    Returns
    -------
    tuple of (float, int, dict)
        - M_est: Estimated parameter M.
        - num_valid: Number of points used in regression.
        - diagnostics: Dictionary containing R^2 and residual variance.
    """
    sin_wt = np.sin(OMEGA * u)
    valid_mask = np.abs(sin_wt) > sin_threshold
    num_valid = int(np.sum(valid_mask))
    
    if num_valid < 10:
        return 0.0, num_valid, {"r_squared": 0.0, "mean_abs_error": float("inf")}
        
    u_valid = u[valid_mask]
    v_valid = v[valid_mask]
    sin_valid = sin_wt[valid_mask]
    
    ratio = np.abs(v_valid / sin_valid)
    log_ratio = np.log(np.maximum(ratio, 1e-12))
    
    # Linear regression through origin: y = M * x
    denom = np.sum(u_valid ** 2)
    if denom == 0.0:
        return 0.0, num_valid, {"r_squared": 0.0, "mean_abs_error": float("inf")}
        
    M_est = float(np.sum(u_valid * log_ratio) / denom)
    
    # Regression diagnostics
    y_pred = M_est * u_valid
    ss_tot = float(np.sum((log_ratio - np.mean(log_ratio)) ** 2))
    ss_res = float(np.sum((log_ratio - y_pred) ** 2))
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 1.0
    mae = float(np.mean(np.abs(np.abs(v_valid) - np.exp(M_est * u_valid) * np.abs(sin_valid))))
    
    diagnostics = {
        "r_squared": float(r2),
        "mean_abs_error": mae,
        "valid_points_fraction": float(num_valid / len(u)),
    }
    return M_est, num_valid, diagnostics


def estimate_centerline_pca(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """
    Compute initial geometric estimate of (theta, X) via Principal Component Analysis
    constrained to pass through the pivot (X, 42).
    
    Parameters
    ----------
    x : np.ndarray
        Observed x coordinates.
    y : np.ndarray
        Observed y coordinates.
        
    Returns
    -------
    tuple of (float, float)
        (theta_deg_init, X_init)
    """
    coords = np.column_stack([x, y])
    center = np.mean(coords, axis=0)
    cov = np.cov(coords, rowvar=False)
    
    eigvals, eigvecs = np.linalg.eigh(cov)
    major_axis = eigvecs[:, 1]
    if major_axis[0] < 0:
        major_axis = -major_axis
        
    theta_init_deg = float(np.degrees(np.arctan2(major_axis[1], major_axis[0])))
    theta_init_deg = float(np.clip(theta_init_deg, THETA_MIN_DEG + 1.0, THETA_MAX_DEG - 1.0))
    
    # Centerline passes through centroid and (X, 42): center_y - 42 = tan(theta) * (center_x - X)
    slope = np.tan(np.radians(theta_init_deg))
    if abs(slope) > 1e-4:
        X_init = float(center[0] - (center[1] - Y_OFFSET) / slope)
    else:
        X_init = float(center[0])
    X_init = float(np.clip(X_init, X_MIN + 1.0, X_MAX - 1.0))
    
    return theta_init_deg, X_init


def geometry_based_estimate(
    x_obs: np.ndarray,
    y_obs: np.ndarray,
    sin_threshold: float = 0.15,
    degrees: bool = True,
) -> Dict[str, Union[float, int, dict]]:
    """
    Perform analytical/geometric parameter estimation for (theta, M, X).
    
    Procedure:
        1. Obtain initial centerline orientation and pivot from PCA.
        2. Minimize transverse harmonic distortion over (theta, X), evaluating M
           in closed-form via log-envelope regression at each evaluation.
        3. Extract final high-precision (theta, M, X) and regression diagnostics.
        
    Parameters
    ----------
    x_obs : np.ndarray
        Observed x coordinates.
    y_obs : np.ndarray
        Observed y coordinates.
    sin_threshold : float, default=0.15
        Threshold for masking near-node singularities during M estimation.
    degrees : bool, default=True
        Whether to report angles in degrees.
        
    Returns
    -------
    dict
        Dictionary containing estimated parameters, node information, and diagnostics.
    """
    start_time = time.time()
    
    # 1. PCA initial centerline estimate
    theta_init_deg, X_init = estimate_centerline_pca(x_obs, y_obs)
    
    # 2. Geometric alignment loss function over (theta, X)
    def alignment_loss(params: List[float]) -> float:
        th_deg, X_cand = params
        if not (THETA_MIN_DEG < th_deg < THETA_MAX_DEG and X_MIN < X_cand < X_MAX):
            return 1e6
            
        th_rad_c = np.radians(th_deg)
        u_c = (x_obs - X_cand) * np.cos(th_rad_c) + (y_obs - Y_OFFSET) * np.sin(th_rad_c)
        v_c = -(x_obs - X_cand) * np.sin(th_rad_c) + (y_obs - Y_OFFSET) * np.cos(th_rad_c)
        
        M_c, n_val, _ = estimate_M_from_amplitude(u_c, v_c, sin_threshold=sin_threshold)
        if n_val < 50:
            return 1e6
            
        M_c = float(np.clip(M_c, M_MIN + 0.001, M_MAX - 0.001))
        
        sin_wt = np.sin(OMEGA * u_c)
        valid = np.abs(sin_wt) > sin_threshold
        v_model = np.exp(M_c * u_c[valid]) * sin_wt[valid]
        
        return float(np.mean(np.abs(v_c[valid] - v_model)))
        
    bounds = [(THETA_MIN_DEG + 0.1, THETA_MAX_DEG - 0.1), (X_MIN + 0.5, X_MAX - 0.5)]
    res = minimize(
        alignment_loss,
        x0=[theta_init_deg, X_init],
        bounds=bounds,
        method="Nelder-Mead",
        options={"xatol": 1e-7, "fatol": 1e-7, "maxiter": 400},
    )
    
    best_theta_deg, best_X = float(res.x[0]), float(res.x[1])
    best_theta_rad = float(np.radians(best_theta_deg))
    
    # 3. Final projection and M recovery
    u_final, v_final = rotate_to_curve_frame(x_obs, y_obs, best_theta_deg, best_X, degrees=True)
    best_M, num_valid, diag = estimate_M_from_amplitude(u_final, v_final, sin_threshold=sin_threshold)
    
    elapsed = time.time() - start_time
    nodes = oscillation_nodes()
    
    # Overall transverse reconstruction error across all valid points
    sin_wt_all = np.sin(OMEGA * u_final)
    valid_mask = np.abs(sin_wt_all) > sin_threshold
    v_expected = np.exp(best_M * u_final[valid_mask]) * sin_wt_all[valid_mask]
    transverse_l1 = float(np.mean(np.abs(v_final[valid_mask] - v_expected)))
    
    return {
        "theta_degrees": best_theta_deg,
        "theta_radians": best_theta_rad,
        "M": best_M,
        "X": best_X,
        "transverse_residual_l1": transverse_l1,
        "num_candidate_nodes": len(nodes),
        "node_t_values": [float(t_n) for t_n in nodes],
        "valid_points_for_M": num_valid,
        "total_points": len(x_obs),
        "m_regression_diagnostics": diag,
        "optimizer_iterations": int(res.nit) if hasattr(res, "nit") else int(res.nfev),
        "duration_sec": float(elapsed),
    }


def save_geometry_results(
    results: dict,
    output_path: Union[str, Path] = "results/geometry_fit.json",
) -> None:
    """Save geometric estimation results to a JSON file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)


def summarize_geometry_fit(results: dict) -> str:
    """Generate formatted summary string of geometric fit."""
    lines = [
        "=" * 50,
        "GEOMETRY-BASED ESTIMATION SUMMARY",
        "=" * 50,
        f"theta: {results['theta_degrees']:.6f} degrees ({results['theta_radians']:.6f} rad)",
        f"M:     {results['M']:.6f}",
        f"X:     {results['X']:.6f}",
        "-" * 50,
        f"Transverse L1 Residual:       {results['transverse_residual_l1']:.8f}",
        f"Points Used for M Estimation: {results['valid_points_for_M']} / {results['total_points']}",
        f"Nodes in Interval (6, 60):    {results['num_candidate_nodes']} {results['node_t_values']}",
        f"Regression R^2:               {results['m_regression_diagnostics']['r_squared']:.6f}",
        f"Solver Runtime:               {results['duration_sec']:.3f} s",
        "=" * 50,
    ]
    return "\n".join(lines)
