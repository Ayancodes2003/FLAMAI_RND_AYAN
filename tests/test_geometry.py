"""
Unit and integration tests for src/geometry.py.

Verifies:
1. Orthonormality of basis vectors {a, b}.
2. Isometry / distance preservation of coordinate rotation.
3. Exact t-recovery via parallel projection u.
4. Exact amplitude recovery via perpendicular projection v.
5. Theoretical node location verification and zero-amplitude check.
6. Robust handling of near-node singularities during M estimation.
7. Multi-case synthetic parameter recovery across diverse valid parameters.
8. Invariance to point cloud shuffling (unordered data).
9. Negative and positive M recovery.
10. JSON serialization and summary reporting.
"""

from pathlib import Path
import numpy as np
import pytest

from src.curve import curve, OMEGA, Y_OFFSET
from src.geometry import (
    basis_vectors,
    rotate_to_curve_frame,
    project_parallel,
    project_perpendicular,
    oscillation_nodes,
    estimate_M_from_amplitude,
    estimate_centerline_pca,
    geometry_based_estimate,
    save_geometry_results,
    summarize_geometry_fit,
)


def test_basis_vectors_orthonormality():
    """Verify {a, b} forms a right-handed orthonormal basis."""
    for th in [0.0, 15.0, 30.0, 45.0, 80.0]:
        a, b = basis_vectors(th, degrees=True)
        assert np.isclose(np.linalg.norm(a), 1.0)
        assert np.isclose(np.linalg.norm(b), 1.0)
        assert np.isclose(np.dot(a, b), 0.0)
        # Det([a, b]) = a_x * b_y - a_y * b_x = cos*cos - sin*(-sin) = 1
        rot_mat = np.column_stack([a, b])
        assert np.isclose(np.linalg.det(rot_mat), 1.0)


def test_rotation_preserves_pairwise_distances():
    """Verify coordinate transformation is isometric (preserves Euclidean distance)."""
    rng = np.random.RandomState(42)
    x = rng.uniform(50, 100, 20)
    y = rng.uniform(40, 70, 20)
    
    u, v = rotate_to_curve_frame(x, y, theta=25.0, X=40.0, degrees=True)
    
    d_orig = np.sqrt((x[0] - x[1]) ** 2 + (y[0] - y[1]) ** 2)
    d_rot = np.sqrt((u[0] - u[1]) ** 2 + (v[0] - v[1]) ** 2)
    assert np.isclose(d_orig, d_rot)


def test_parallel_projection_exact_t_recovery():
    """Verify that u = (x - X)*cos(theta) + (y - 42)*sin(theta) recovers t identically."""
    t_true = np.linspace(8.0, 55.0, 50)
    th_deg = 32.0
    M = 0.02
    X = 48.0
    
    x_exact, y_exact = curve(t_true, th_deg, M, X, degrees=True)
    u_recovered = project_parallel(x_exact, y_exact, th_deg, X, degrees=True)
    
    assert np.allclose(u_recovered, t_true, atol=1e-12)


def test_perpendicular_projection_exact_amplitude_recovery():
    """Verify that v = -(x - X)*sin(theta) + (y - 42)*cos(theta) recovers A(t) identically."""
    t_true = np.linspace(8.0, 55.0, 50)
    th_deg = 32.0
    M = 0.02
    X = 48.0
    
    x_exact, y_exact = curve(t_true, th_deg, M, X, degrees=True)
    v_recovered = project_perpendicular(x_exact, y_exact, th_deg, X, degrees=True)
    
    expected_v = np.exp(M * t_true) * np.sin(OMEGA * t_true)
    assert np.allclose(v_recovered, expected_v, atol=1e-12)


def test_oscillation_nodes_theory():
    """Verify theoretical oscillation nodes in (6, 60) and confirm A(t_k) = 0."""
    nodes = oscillation_nodes(6.0, 60.0)
    assert len(nodes) == 5  # Exactly 5 nodes: k=1,2,3,4,5
    
    for tk in nodes:
        assert 6.0 < tk < 60.0
        # sin(0.3 * tk) should be 0
        assert np.isclose(np.sin(OMEGA * tk), 0.0, atol=1e-12)


def test_near_node_singularity_masking():
    """Verify estimate_M_from_amplitude safely ignores points with small |sin(0.3t)|."""
    t = np.linspace(7.0, 58.0, 200)
    M_true = -0.025
    u = t.copy()
    v = np.exp(M_true * t) * np.sin(OMEGA * t)
    
    # Intentionally corrupt near-node points to check masking robustness
    near_node_mask = np.abs(np.sin(OMEGA * t)) <= 0.15
    v[near_node_mask] = 999.0  # Would break log without masking
    
    M_est, num_valid, diag = estimate_M_from_amplitude(u, v, sin_threshold=0.15)
    assert np.isclose(M_est, M_true, atol=1e-6)
    assert diag["r_squared"] > 0.999


@pytest.mark.parametrize(
    "th_true, M_true, X_true",
    [
        (12.0, -0.020, 20.0),
        (27.0, 0.015, 45.0),
        (43.0, 0.035, 75.0),
        (38.0, -0.045, 10.0),
        (5.0, 0.000, 90.0),
    ],
)
def test_synthetic_parameter_recovery_multi_case(th_true, M_true, X_true):
    """Verify geometry solver accurately recovers multiple synthetic parameter sets."""
    rng = np.random.RandomState(42)
    t = np.linspace(6.2, 59.8, 1000)
    x_c, y_c = curve(t, th_true, M_true, X_true, degrees=True)
    
    # Shuffle points to simulate unordered observation condition
    perm = rng.permutation(len(x_c))
    x_obs = x_c[perm]
    y_obs = y_c[perm]
    
    fit = geometry_based_estimate(x_obs, y_obs, degrees=True)
    
    assert np.isclose(fit["theta_degrees"], th_true, atol=1e-3)
    assert np.isclose(fit["M"], M_true, atol=1e-4)
    assert np.isclose(fit["X"], X_true, atol=1e-3)
    assert fit["transverse_residual_l1"] < 1e-4


def test_shuffled_point_invariance():
    """Verify results are invariant under different random point permutations."""
    t = np.linspace(6.5, 59.5, 800)
    x_c, y_c = curve(t, 33.0, 0.022, 50.0, degrees=True)
    
    rng1 = np.random.RandomState(1)
    perm1 = rng1.permutation(len(x_c))
    fit1 = geometry_based_estimate(x_c[perm1], y_c[perm1])
    
    rng2 = np.random.RandomState(2)
    perm2 = rng2.permutation(len(x_c))
    fit2 = geometry_based_estimate(x_c[perm2], y_c[perm2])
    
    assert np.isclose(fit1["theta_degrees"], fit2["theta_degrees"], atol=1e-5)
    assert np.isclose(fit1["M"], fit2["M"], atol=1e-5)
    assert np.isclose(fit1["X"], fit2["X"], atol=1e-5)


def test_save_and_summarize_geometry(tmp_path):
    """Verify JSON export and summary output for geometry results."""
    dummy_results = {
        "theta_degrees": 30.0,
        "theta_radians": 0.523599,
        "M": 0.03,
        "X": 55.0,
        "transverse_residual_l1": 0.000002,
        "num_candidate_nodes": 5,
        "node_t_values": [10.47, 20.94, 31.42, 41.89, 52.36],
        "valid_points_for_M": 1350,
        "total_points": 1500,
        "m_regression_diagnostics": {"r_squared": 0.9999, "mean_abs_error": 0.0001, "valid_points_fraction": 0.9},
        "optimizer_iterations": 45,
        "duration_sec": 0.05,
    }
    
    out_file = tmp_path / "test_geom_fit.json"
    save_geometry_results(dummy_results, out_file)
    assert out_file.exists()
    
    summary = summarize_geometry_fit(dummy_results)
    assert "GEOMETRY-BASED ESTIMATION SUMMARY" in summary
    assert "theta: 30.000000 degrees" in summary
