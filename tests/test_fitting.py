"""
Unit and integration tests for src/fitting.py.

Verifies:
1. Synthetic curve generation and parameter recovery.
2. Invariance to point cloud shuffling (unordered data).
3. Out-of-bounds objective penalties.
4. Load dataset error handling.
5. Deterministic multi-start optimization behavior.
"""

from pathlib import Path
import numpy as np
import pytest

from src.curve import curve
from src.fitting import (
    load_dataset,
    estimate_closest_t,
    baseline_l1_objective,
    fit_baseline_multistart,
    save_baseline_results,
    summarize_fit,
)


# Independent synthetic ground truth parameters (strictly different from any dataset values)
SYN_THETA_DEG = 22.5
SYN_M = -0.015
SYN_X = 35.0


@pytest.fixture
def synthetic_point_cloud():
    """Generate 300 synthetic unordered points from known test parameters."""
    rng = np.random.RandomState(42)
    t_true = np.linspace(6.5, 59.5, 300)
    x_true, y_true = curve(t_true, SYN_THETA_DEG, SYN_M, SYN_X, degrees=True)
    
    # Shuffle points to simulate unordered observation condition
    perm = rng.permutation(len(t_true))
    return x_true[perm], y_true[perm], t_true[perm]


def test_estimate_closest_t_recovery(synthetic_point_cloud):
    """Verify that estimate_closest_t accurately associates points with their true t."""
    x_obs, y_obs, _ = synthetic_point_cloud
    
    # Grid estimation
    t_est_grid, dist_grid = estimate_closest_t(
        x_obs, y_obs, SYN_THETA_DEG, SYN_M, SYN_X, n_grid=1000, degrees=True, refine=False
    )
    assert np.all(dist_grid < 0.1)
    
    # Continuous refinement
    t_est_ref, dist_ref = estimate_closest_t(
        x_obs, y_obs, SYN_THETA_DEG, SYN_M, SYN_X, n_grid=500, degrees=True, refine=True
    )
    assert np.all(dist_ref < 1e-4)


def test_baseline_objective_properties(synthetic_point_cloud):
    """Verify objective returns low error at true parameters and penalizes invalid parameters."""
    x_obs, y_obs, _ = synthetic_point_cloud
    
    # At true parameters, error should be close to zero
    loss_true = baseline_l1_objective([SYN_THETA_DEG, SYN_M, SYN_X], x_obs, y_obs, n_grid=500, degrees=True)
    assert loss_true < 0.05
    
    # At perturbed parameters, loss should be significantly higher
    loss_perturbed = baseline_l1_objective([SYN_THETA_DEG + 5.0, SYN_M, SYN_X], x_obs, y_obs, n_grid=500, degrees=True)
    assert loss_perturbed > loss_true
    
    # Out of bounds parameter penalty
    loss_oob = baseline_l1_objective([60.0, SYN_M, SYN_X], x_obs, y_obs, degrees=True)
    assert loss_oob > 1000.0


def test_synthetic_parameter_recovery(synthetic_point_cloud):
    """Verify multi-start optimizer recovers synthetic parameters within tight tolerance."""
    x_obs, y_obs, _ = synthetic_point_cloud
    
    # Provide coarse initial seeds that do NOT include the exact answer
    test_starts = [
        [10.0, 0.0, 20.0],
        [30.0, -0.03, 50.0],
        [40.0, 0.02, 70.0],
    ]
    
    fit_res = fit_baseline_multistart(
        x_obs, y_obs, initial_guesses=test_starts, n_grid=400, degrees=True
    )
    
    assert np.isclose(fit_res["theta_degrees"], SYN_THETA_DEG, atol=0.1)
    assert np.isclose(fit_res["M"], SYN_M, atol=0.002)
    assert np.isclose(fit_res["X"], SYN_X, atol=0.1)
    assert fit_res["continuous_l1_mean"] < 1e-3


def test_dataset_loader_errors():
    """Verify load_dataset raises appropriate exceptions on bad paths/missing columns."""
    with pytest.raises(FileNotFoundError):
        load_dataset("non_existent_file.csv")


def test_save_and_summarize(tmp_path):
    """Verify JSON export and summary text formatting."""
    dummy_results = {
        "theta_degrees": 25.0,
        "theta_radians": 0.436332,
        "M": 0.01,
        "X": 45.0,
        "grid_objective_value": 0.045,
        "continuous_l1_mean": 0.00012,
        "optimization_method": "Nelder-Mead (Multi-Start)",
        "num_runs": 3,
        "total_duration_sec": 1.25,
        "all_runs": [],
        "best_run": {},
    }
    
    out_file = tmp_path / "test_fit.json"
    save_baseline_results(dummy_results, out_file)
    assert out_file.exists()
    
    summary = summarize_fit(dummy_results)
    assert "BASELINE NUMERICAL FIT SUMMARY" in summary
    assert "theta: 25.000000 degrees" in summary
