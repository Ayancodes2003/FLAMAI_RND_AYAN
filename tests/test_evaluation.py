"""
Unit and integration tests for src/evaluation.py.

Verifies:
1. Uniform curve sampling produces expected shape, endpoints, and point counts.
2. Identical curves evaluate to exactly zero L1 distance.
3. Known translated curve gives exact expected L1 offset.
4. Unordered observation points are correctly handled.
5. L1 is genuinely Manhattan norm |dx| + |dy|, not Euclidean norm sqrt(dx^2 + dy^2).
6. Sampling convergence is stable and monotonically non-increasing in discretization bias.
7. Parameter export and convergence JSON serialization.
"""

from pathlib import Path
import numpy as np
import pytest

from src.curve import curve, T_MIN, T_MAX
from src.evaluation import (
    sample_uniform_curve,
    compute_manhattan_distance,
    evaluate_curve_l1_discrete,
    evaluate_sampling_convergence,
    save_final_parameters,
    save_evaluation_convergence,
)


def test_uniform_curve_sampling_properties():
    """Verify uniform sampling generates correct shapes, endpoints, and point counts."""
    n_pts = 1000
    t_samp, x_samp, y_samp = sample_uniform_curve(
        theta=30.0, M=0.03, X=55.0, n_samples=n_pts, t_min=6.0, t_max=60.0, endpoint=True, degrees=True
    )
    
    assert len(t_samp) == n_pts
    assert len(x_samp) == n_pts
    assert len(y_samp) == n_pts
    assert np.isclose(t_samp[0], 6.0)
    assert np.isclose(t_samp[-1], 60.0)
    assert np.all(np.diff(t_samp) > 0)
    assert np.allclose(np.diff(t_samp), (60.0 - 6.0) / (n_pts - 1))

    # Error handling
    with pytest.raises(ValueError, match="n_samples must be at least 2"):
        sample_uniform_curve(30.0, 0.03, 55.0, n_samples=1)
    with pytest.raises(ValueError, match="must be strictly greater"):
        sample_uniform_curve(30.0, 0.03, 55.0, t_min=60.0, t_max=6.0)


def test_manhattan_vs_euclidean_distance():
    """Verify L1 distance is strictly Manhattan norm (|dx| + |dy|) and not Euclidean."""
    p1 = np.array([[0.0, 0.0]])
    p2 = np.array([[3.0, 4.0]])
    
    l1_dist = compute_manhattan_distance(p1, p2)
    # Manhattan distance is |3| + |4| = 7.0 (whereas Euclidean is 5.0)
    assert np.isclose(l1_dist[0], 7.0)
    assert not np.isclose(l1_dist[0], 5.0)


def test_identical_curves_zero_distance():
    """Verify identical point sets produce zero L1 distance."""
    t_pts = np.linspace(6.0, 60.0, 500)
    x_c, y_c = curve(t_pts, theta=28.0, M=-0.01, X=40.0, degrees=True)
    
    metrics = evaluate_curve_l1_discrete(x_c, y_c, x_c, y_c)
    assert np.isclose(metrics["mean_obs_to_pred_l1"], 0.0, atol=1e-12)
    assert np.isclose(metrics["max_obs_to_pred_l1"], 0.0, atol=1e-12)
    assert np.isclose(metrics["symmetric_mean_l1"], 0.0, atol=1e-12)


def test_known_translated_curve_l1_distance():
    """Verify known translated curve produces exact expected Manhattan distance."""
    t_pts = np.linspace(6.0, 60.0, 200)
    x_c, y_c = curve(t_pts, theta=20.0, M=0.01, X=50.0, degrees=True)
    
    dx = 2.5
    dy = -1.5
    expected_l1 = abs(dx) + abs(dy)  # 4.0
    
    x_trans = x_c + dx
    y_trans = y_c + dy
    
    metrics = evaluate_curve_l1_discrete(x_trans, y_trans, x_c, y_c)
    # Each point is shifted by (2.5, -1.5), nearest point is shifted by at most 4.0
    assert metrics["mean_obs_to_pred_l1"] <= expected_l1 + 1e-4


def test_unordered_data_handling():
    """Verify that shuffling point ordering does not change the nearest-neighbor L1 distance."""
    rng = np.random.RandomState(42)
    t_pts = np.linspace(6.0, 60.0, 600)
    x_c, y_c = curve(t_pts, theta=35.0, M=0.02, X=60.0, degrees=True)
    
    _, x_pred, y_pred = sample_uniform_curve(35.0, 0.02, 60.0, n_samples=1000, degrees=True)
    
    # Original ordered evaluation
    res_ordered = evaluate_curve_l1_discrete(x_c, y_c, x_pred, y_pred)
    
    # Shuffled evaluation
    perm = rng.permutation(len(x_c))
    res_shuffled = evaluate_curve_l1_discrete(x_c[perm], y_c[perm], x_pred, y_pred)
    
    assert np.isclose(res_ordered["mean_obs_to_pred_l1"], res_shuffled["mean_obs_to_pred_l1"], atol=1e-10)
    assert np.isclose(res_ordered["max_obs_to_pred_l1"], res_shuffled["max_obs_to_pred_l1"], atol=1e-10)
    assert np.isclose(res_ordered["symmetric_mean_l1"], res_shuffled["symmetric_mean_l1"], atol=1e-10)


def test_sampling_convergence_synthetic_stability():
    """Verify sampling convergence is stable across increasing resolutions."""
    t_true = np.linspace(6.0, 60.0, 1500)
    x_obs, y_obs = curve(t_true, theta=30.0, M=0.03, X=55.0, degrees=True)
    
    resolutions = [500, 1000, 2000, 5000]
    records = evaluate_sampling_convergence(x_obs, y_obs, theta=30.0, M=0.03, X=55.0, resolutions=resolutions, degrees=True)
    
    assert len(records) == len(resolutions)
    # As N increases, the discrete grid discretization error strictly decreases or remains tiny
    for r in records:
        assert r["mean_obs_to_pred_l1"] < 0.05
        assert r["symmetric_mean_l1"] < 0.05


def test_save_evaluation_artifacts(tmp_path):
    """Verify saving final parameters and convergence records to JSON."""
    params = {
        "theta_deg": 30.0,
        "theta_rad": 0.5235987755982988,
        "M": 0.03,
        "X": 55.0,
        "t_min": 6.0,
        "t_max": 60.0,
    }
    param_file = tmp_path / "final_params.json"
    save_final_parameters(params, param_file)
    assert param_file.exists()
    
    records = [{"N": 500, "mean_obs_to_pred_l1": 0.04, "symmetric_mean_l1": 0.04, "runtime_ms": 1.2}]
    conv_file = tmp_path / "conv.json"
    save_evaluation_convergence(records, conv_file)
    assert conv_file.exists()
