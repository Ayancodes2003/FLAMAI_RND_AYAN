"""
Rigorous curve distance evaluation and sampling convergence module.

Assignment Requirement:
    "The L1 distance between uniformly sampled points between the expected curve and the predicted curve"

Methodology for Unordered Observations:
    Because observation points in xy_data.csv are unordered and unindexed w.r.t parameter t:
    1. Direct row-by-row subtraction (sum |x_i - x_pred_i| + |y_i - y_pred_i|) is mathematically
       invalid as it imposes an artificial, false 1-to-1 index correspondence.
    2. We compute a rigorous point-to-manifold L1 (Manhattan) distance:
       - Observed -> Predicted: For each observed point p_i, find nearest point on predicted curve:
         d(p_i, C_pred) = min_{q in C_pred} (|p_i.x - q.x| + |p_i.y - q.y|)
       - Predicted -> Observed: For each predicted sample q_j, find nearest observed point:
         d(q_j, P_obs) = min_{p in P_obs} (|q_j.x - p.x| + |q_j.y - p.y|)
       - Symmetric Mean L1: Average of bidirectional nearest-neighbor Manhattan distances:
         L1_sym = 0.5 * (mean(d(P_obs, C_pred)) + mean(d(C_pred, P_obs)))
       - Symmetric Max L1 (Hausdorff L1): max(max(d(P_obs, C_pred)), max(d(C_pred, P_obs)))
"""

from typing import Dict, List, Optional, Sequence, Tuple, Union
import json
import time
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree

from src.curve import (
    curve,
    to_radians,
    to_degrees,
    T_MIN,
    T_MAX,
)


def sample_uniform_curve(
    theta: float,
    M: float,
    X: float,
    n_samples: int = 1500,
    t_min: float = T_MIN,
    t_max: float = T_MAX,
    endpoint: bool = True,
    degrees: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate uniformly spaced sample points along parameter domain t in [t_min, t_max]
    and evaluate corresponding predicted curve coordinates (x(t), y(t)).
    
    Parameters
    ----------
    theta : float
        Rotation parameter.
    M : float
        Exponential growth parameter.
    X : float
        X-offset parameter.
    n_samples : int, default=1500
        Number of uniform evaluation points in t.
    t_min : float, default=6.0
        Minimum parameter value.
    t_max : float, default=60.0
        Maximum parameter value.
    endpoint : bool, default=True
        Whether to include t_max as the final sample point.
    degrees : bool, default=True
        Whether theta is provided in degrees.
        
    Returns
    -------
    tuple of (np.ndarray, np.ndarray, np.ndarray)
        - t_samples: Uniformly spaced parameter array (n_samples,).
        - x_pred: Evaluated x coordinates (n_samples,).
        - y_pred: Evaluated y coordinates (n_samples,).
    """
    if n_samples < 2:
        raise ValueError(f"n_samples must be at least 2, got {n_samples}")
    if t_max <= t_min:
        raise ValueError(f"t_max ({t_max}) must be strictly greater than t_min ({t_min})")
        
    th_rad = to_radians(theta, degrees=degrees)
    t_samples = np.linspace(t_min, t_max, n_samples, endpoint=endpoint, dtype=np.float64)
    x_pred, y_pred = curve(t_samples, th_rad, M, X, degrees=False)
    return t_samples, x_pred, y_pred


def compute_manhattan_distance(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    """
    Compute pairwise or elementwise Manhattan (L1) distance |x1 - x2| + |y1 - y2|.
    
    Parameters
    ----------
    p1 : np.ndarray
        Points array of shape (..., 2).
    p2 : np.ndarray
        Points array of shape (..., 2).
        
    Returns
    -------
    np.ndarray
        Computed L1 distance array.
    """
    p1_arr = np.asarray(p1, dtype=np.float64)
    p2_arr = np.asarray(p2, dtype=np.float64)
    return np.sum(np.abs(p1_arr - p2_arr), axis=-1)


def evaluate_curve_l1_discrete(
    x_obs: np.ndarray,
    y_obs: np.ndarray,
    x_pred: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, Union[float, np.ndarray]]:
    """
    Compute bidirectional discrete L1 (Manhattan) point-to-point distances between
    observed points and predicted uniform curve samples.
    
    Uses exact cKDTree queries with p=1 (Manhattan norm) for sub-millisecond efficiency.
    
    Parameters
    ----------
    x_obs : np.ndarray
        Observed x coordinates (N_obs,).
    y_obs : np.ndarray
        Observed y coordinates (N_obs,).
    x_pred : np.ndarray
        Predicted curve x samples (N_pred,).
    y_pred : np.ndarray
        Predicted curve y samples (N_pred,).
        
    Returns
    -------
    dict
        Dictionary containing bidirectional L1 distance metrics and distributions.
    """
    pts_obs = np.column_stack([x_obs, y_obs]).astype(np.float64)
    pts_pred = np.column_stack([x_pred, y_pred]).astype(np.float64)
    
    # 1. Observed -> Predicted: for each observed point, find nearest point on predicted curve
    tree_pred = cKDTree(pts_pred)
    dists_obs_to_pred, idxs_obs_to_pred = tree_pred.query(pts_obs, p=1)
    
    # 2. Predicted -> Observed: for each predicted curve sample, find nearest observed point
    tree_obs = cKDTree(pts_obs)
    dists_pred_to_obs, idxs_pred_to_obs = tree_obs.query(pts_pred, p=1)
    
    mean_obs_to_pred = float(np.mean(dists_obs_to_pred))
    max_obs_to_pred = float(np.max(dists_obs_to_pred))
    median_obs_to_pred = float(np.median(dists_obs_to_pred))
    std_obs_to_pred = float(np.std(dists_obs_to_pred))
    
    mean_pred_to_obs = float(np.mean(dists_pred_to_obs))
    max_pred_to_obs = float(np.max(dists_pred_to_obs))
    median_pred_to_obs = float(np.median(dists_pred_to_obs))
    
    symmetric_mean = 0.5 * (mean_obs_to_pred + mean_pred_to_obs)
    symmetric_max = max(max_obs_to_pred, max_pred_to_obs)
    
    return {
        "mean_obs_to_pred_l1": mean_obs_to_pred,
        "max_obs_to_pred_l1": max_obs_to_pred,
        "median_obs_to_pred_l1": median_obs_to_pred,
        "std_obs_to_pred_l1": std_obs_to_pred,
        "mean_pred_to_obs_l1": mean_pred_to_obs,
        "max_pred_to_obs_l1": max_pred_to_obs,
        "median_pred_to_obs_l1": median_pred_to_obs,
        "symmetric_mean_l1": symmetric_mean,
        "symmetric_max_l1": symmetric_max,
        "num_obs_points": len(x_obs),
        "num_pred_samples": len(x_pred),
        "dists_obs_to_pred": dists_obs_to_pred,
        "dists_pred_to_obs": dists_pred_to_obs,
        "nearest_pred_indices": idxs_obs_to_pred,
    }


def evaluate_sampling_convergence(
    x_obs: np.ndarray,
    y_obs: np.ndarray,
    theta: float,
    M: float,
    X: float,
    resolutions: Sequence[int] = (500, 1000, 1500, 5000, 10000),
    degrees: bool = True,
) -> List[Dict[str, Union[int, float]]]:
    """
    Evaluate curve distance across multiple uniform sampling resolutions to assess
    discretization convergence.
    
    Parameters
    ----------
    x_obs : np.ndarray
        Observed x coordinates.
    y_obs : np.ndarray
        Observed y coordinates.
    theta : float
        Rotation parameter.
    M : float
        Exponential growth parameter.
    X : float
        X-offset parameter.
    resolutions : sequence of int, default=(500, 1000, 1500, 5000, 10000)
        Sampling resolutions to test.
    degrees : bool, default=True
        Whether theta is in degrees.
        
    Returns
    -------
    list of dict
        Convergence records for each sampling resolution.
    """
    records = []
    
    for n_pts in resolutions:
        t0 = time.time()
        _, x_pred, y_pred = sample_uniform_curve(
            theta=theta,
            M=M,
            X=X,
            n_samples=n_pts,
            t_min=T_MIN,
            t_max=T_MAX,
            endpoint=True,
            degrees=degrees,
        )
        eval_metrics = evaluate_curve_l1_discrete(x_obs, y_obs, x_pred, y_pred)
        elapsed_sec = time.time() - t0
        
        record = {
            "N": int(n_pts),
            "mean_obs_to_pred_l1": float(eval_metrics["mean_obs_to_pred_l1"]),
            "max_obs_to_pred_l1": float(eval_metrics["max_obs_to_pred_l1"]),
            "mean_pred_to_obs_l1": float(eval_metrics["mean_pred_to_obs_l1"]),
            "max_pred_to_obs_l1": float(eval_metrics["max_pred_to_obs_l1"]),
            "symmetric_mean_l1": float(eval_metrics["symmetric_mean_l1"]),
            "symmetric_max_l1": float(eval_metrics["symmetric_max_l1"]),
            "runtime_ms": float(elapsed_sec * 1000.0),
        }
        records.append(record)
        
    return records


def save_final_parameters(
    params: dict,
    output_path: Union[str, Path] = "results/final_parameters.json",
) -> None:
    """Save final parameter estimates to JSON file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(params, f, indent=4)


def save_evaluation_convergence(
    convergence_records: List[dict],
    output_path: Union[str, Path] = "results/evaluation_convergence.json",
) -> None:
    """Save sampling convergence table to JSON file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"convergence_records": convergence_records}, f, indent=4)
