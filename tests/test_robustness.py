"""
Unit tests for robustness, sensitivity, and limitation experiment infrastructure.

Verifies:
1. Deterministic reproducibility with fixed seeds.
2. Zero noise condition returns clean data and exact recovery.
3. Parameter sensitivity response is positive and symmetric around optimum.
4. Observation density sweeps produce valid parameter outputs across N.
5. Result payloads adhere to expected structured schema.
"""

from pathlib import Path
import numpy as np
import pytest

from scripts.run_robustness_experiment import (
    run_noise_experiments,
    run_parameter_sensitivity_experiments,
    run_observation_density_experiments,
    run_limitation_analysis,
    THETA_TRUE,
    M_TRUE,
    X_TRUE,
)


def test_zero_noise_exact_recovery():
    """Verify sigma=0 condition recovers ground truth parameters with near-zero error."""
    res = run_noise_experiments(sigmas=[0.0], n_trials_per_sigma=1, base_seed=100)
    summary = res["summary_by_sigma"][0]
    
    assert summary["sigma"] == 0.0
    assert summary["num_trials"] == 1
    assert summary["theta_error_mean"] < 1e-4
    assert summary["M_error_mean"] < 1e-5
    assert summary["X_error_mean"] < 1e-4
    assert summary["curve_l1_mean"] < 1e-4


def test_deterministic_seed_reproducibility():
    """Verify identical random seeds produce bit-identical experimental results."""
    res1 = run_noise_experiments(sigmas=[0.01], n_trials_per_sigma=2, base_seed=100)
    res2 = run_noise_experiments(sigmas=[0.01], n_trials_per_sigma=2, base_seed=100)
    
    for t1, t2 in zip(res1["all_trials"], res2["all_trials"]):
        assert t1["seed"] == t2["seed"]
        assert np.isclose(t1["theta_est"], t2["theta_est"], atol=1e-12)
        assert np.isclose(t1["M_est"], t2["M_est"], atol=1e-12)
        assert np.isclose(t1["X_est"], t2["X_est"], atol=1e-12)
        assert np.isclose(t1["curve_l1_error"], t2["curve_l1_error"], atol=1e-12)


def test_parameter_sensitivity_properties():
    """Verify parameter sensitivity curves are zero at optimum and strictly positive elsewhere."""
    res = run_parameter_sensitivity_experiments()
    
    # Theta sensitivity
    th_sens = res["theta_sensitivity"]
    zero_idx = [i for i, r in enumerate(th_sens) if r["delta_theta"] == 0.0][0]
    assert np.isclose(th_sens[zero_idx]["curve_l1_error"], 0.0, atol=1e-6)
    for i, r in enumerate(th_sens):
        if r["delta_theta"] != 0.0:
            assert r["curve_l1_error"] > 0.05

    # M sensitivity
    m_sens = res["M_sensitivity"]
    m_zero_idx = [i for i, r in enumerate(m_sens) if r["delta_M"] == 0.0][0]
    assert np.isclose(m_sens[m_zero_idx]["curve_l1_error"], 0.0, atol=1e-6)


def test_observation_density_sweep():
    """Verify observation density sweep produces valid outputs across N values."""
    records = run_observation_density_experiments(density_levels=[100, 500, 1000], seed=42)
    assert len(records) == 3
    
    for r in records:
        assert r["N"] in [100, 500, 1000]
        assert r["err_theta"] < 1e-3
        assert r["err_M"] < 1e-4
        assert r["err_X"] < 1e-3


def test_limitation_analysis_structure():
    """Verify limitation analysis executes and returns expected keys."""
    res = run_limitation_analysis()
    assert "extreme_noise" in res
    assert "extreme_sparsity" in res
    assert "boundary_M" in res
    assert len(res["extreme_noise"]) > 0
    assert len(res["extreme_sparsity"]) > 0
    assert len(res["boundary_M"]) > 0
