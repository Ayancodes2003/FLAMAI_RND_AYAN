"""
Comprehensive unit tests for src/curve.py.

Verifies:
1. Scalar and vectorized evaluations.
2. Output shape consistency.
3. Known mathematical sanity checks.
4. Degree vs radian conversions.
5. Parameter bound and t-domain validations.
6. Finite-difference verification of analytic parameter Jacobians.
7. Finite-difference verification of analytic time derivatives.
8. Domain and negative-t handling.
"""

import numpy as np
import pytest

from src.curve import (
    to_radians,
    to_degrees,
    validate_parameters,
    validate_t,
    eval_amplitude,
    eval_x,
    eval_y,
    curve,
    curve_jacobian_params,
    curve_derivative_t,
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


# Arbitrary valid test parameters (strictly independent of any dataset values)
TEST_THETA_DEG = 35.0
TEST_THETA_RAD = float(np.radians(TEST_THETA_DEG))
TEST_M = 0.02
TEST_X = 50.0
TEST_T_SCALAR = 25.0
TEST_T_ARRAY = np.linspace(10.0, 50.0, 100)


def test_degree_radian_conversions():
    """Verify angular conversion helpers."""
    assert np.isclose(to_radians(180.0, degrees=True), np.pi)
    assert np.isclose(to_radians(np.pi, degrees=False), np.pi)
    assert np.isclose(to_degrees(np.pi), 180.0)
    assert np.isclose(to_degrees(to_radians(45.0, degrees=True)), 45.0)


def test_parameter_bound_validation():
    """Verify parameter constraint validation for valid and invalid inputs."""
    # Valid parameters
    assert validate_parameters(TEST_THETA_RAD, TEST_M, TEST_X, degrees=False)
    assert validate_parameters(TEST_THETA_DEG, TEST_M, TEST_X, degrees=True)

    # Invalid theta
    assert not validate_parameters(-1.0, TEST_M, TEST_X, degrees=True, strict=False)
    assert not validate_parameters(55.0, TEST_M, TEST_X, degrees=True, strict=False)
    with pytest.raises(ValueError, match="Parameter bound violation"):
        validate_parameters(55.0, TEST_M, TEST_X, degrees=True, strict=True)

    # Invalid M
    assert not validate_parameters(TEST_THETA_DEG, -0.06, TEST_X, degrees=True, strict=False)
    assert not validate_parameters(TEST_THETA_DEG, 0.06, TEST_X, degrees=True, strict=False)
    with pytest.raises(ValueError, match="Parameter bound violation"):
        validate_parameters(TEST_THETA_DEG, 0.08, TEST_X, degrees=True, strict=True)

    # Invalid X
    assert not validate_parameters(TEST_THETA_DEG, TEST_M, -5.0, degrees=True, strict=False)
    assert not validate_parameters(TEST_THETA_DEG, TEST_M, 105.0, degrees=True, strict=False)
    with pytest.raises(ValueError, match="Parameter bound violation"):
        validate_parameters(TEST_THETA_DEG, TEST_M, 150.0, degrees=True, strict=True)


def test_t_domain_validation():
    """Verify t domain validation."""
    assert validate_t(TEST_T_ARRAY, strict=False)
    assert validate_t(np.array([T_MIN, 30.0, T_MAX]), strict=False)
    
    assert not validate_t(np.array([4.0, 30.0]), strict=False)
    assert not validate_t(np.array([30.0, 65.0]), strict=False)
    
    with pytest.raises(ValueError, match="t-domain violation"):
        validate_t(np.array([2.0, 20.0]), strict=True)


def test_scalar_and_vector_evaluation_consistency():
    """Verify scalar outputs match array elements exactly."""
    x_scalar = eval_x(TEST_T_SCALAR, TEST_THETA_RAD, TEST_M, TEST_X)
    y_scalar = eval_y(TEST_T_SCALAR, TEST_THETA_RAD, TEST_M)
    
    t_arr = np.array([TEST_T_SCALAR])
    x_arr = eval_x(t_arr, TEST_THETA_RAD, TEST_M, TEST_X)
    y_arr = eval_y(t_arr, TEST_THETA_RAD, TEST_M)
    
    assert isinstance(x_scalar, float)
    assert isinstance(y_scalar, float)
    assert np.isclose(x_scalar, x_arr[0])
    assert np.isclose(y_scalar, y_arr[0])


def test_curve_shape_consistency():
    """Verify curve() output shapes for scalar and array inputs."""
    x_s, y_s = curve(TEST_T_SCALAR, TEST_THETA_RAD, TEST_M, TEST_X)
    assert isinstance(x_s, float)
    assert isinstance(y_s, float)
    
    x_v, y_v = curve(TEST_T_ARRAY, TEST_THETA_RAD, TEST_M, TEST_X)
    assert isinstance(x_v, np.ndarray)
    assert isinstance(y_v, np.ndarray)
    assert x_v.shape == (100,)
    assert y_v.shape == (100,)


def test_known_mathematical_cases():
    """Verify specific mathematical edge cases."""
    # Case 1: At theta = 0, x(t) = t + X, y(t) = 42 + exp(M*t)*sin(0.3*t)
    t_val = 20.0
    x_0, y_0 = curve(t_val, theta=0.0, M=TEST_M, X=TEST_X, degrees=False)
    assert np.isclose(x_0, t_val + TEST_X)
    expected_y_0 = Y_OFFSET + np.exp(TEST_M * t_val) * np.sin(OMEGA * t_val)
    assert np.isclose(y_0, expected_y_0)

    # Case 2: At nodes where 0.3*t = k*pi => sin(0.3*t) = 0
    # Then x(t) = t*cos(theta) + X, y(t) = 42 + t*sin(theta)
    k = 2
    t_node = k * np.pi / OMEGA
    x_node, y_node = curve(t_node, TEST_THETA_RAD, TEST_M, TEST_X)
    assert np.isclose(x_node, t_node * np.cos(TEST_THETA_RAD) + TEST_X)
    assert np.isclose(y_node, Y_OFFSET + t_node * np.sin(TEST_THETA_RAD))

    # Case 3: When M = 0, amplitude is purely sin(0.3*t) without exponential growth
    amp_m0 = eval_amplitude(t_val, M=0.0)
    assert np.isclose(amp_m0, np.sin(OMEGA * t_val))


def test_degree_vs_radian_flag():
    """Verify that setting degrees=True evaluates properly and differs from radians."""
    # 30 radians is completely different from 30 degrees (pi/6 rad)
    x_deg, y_deg = curve(TEST_T_SCALAR, theta=30.0, M=TEST_M, X=TEST_X, degrees=True)
    x_rad, y_rad = curve(TEST_T_SCALAR, theta=np.radians(30.0), M=TEST_M, X=TEST_X, degrees=False)
    
    assert np.isclose(x_deg, x_rad)
    assert np.isclose(y_deg, y_rad)

    # Ensure passing 30 without degrees=True produces a different result
    x_raw, y_raw = curve(TEST_T_SCALAR, theta=30.0, M=TEST_M, X=TEST_X, degrees=False)
    assert not np.isclose(x_deg, x_raw)


def test_abs_t_handling():
    """Verify that negative t values are computed with |t| as specified in formula."""
    t_pos = 15.0
    t_neg = -15.0
    
    amp_pos = eval_amplitude(t_pos, TEST_M)
    amp_neg = eval_amplitude(t_neg, TEST_M)
    
    # amp(t) = exp(M*|t|)*sin(0.3*t)
    # amp(-t) = exp(M*|t|)*sin(-0.3*t) = -amp(t) (odd function)
    assert np.isclose(amp_neg, -amp_pos)


def test_finite_difference_parameter_jacobian():
    """Verify analytic Jacobian w.r.t (theta, M, X) matches finite differences."""
    h = 1e-7
    t_eval = np.array([12.0, 24.5, 38.0, 52.1])
    theta = TEST_THETA_RAD
    M = TEST_M
    X = TEST_X

    J_x_analytic, J_y_analytic = curve_jacobian_params(t_eval, theta, M, X, degrees=False)

    # Finite difference w.r.t theta
    x_th_p, y_th_p = curve(t_eval, theta + h, M, X, degrees=False)
    x_th_m, y_th_m = curve(t_eval, theta - h, M, X, degrees=False)
    dx_dtheta_num = (x_th_p - x_th_m) / (2 * h)
    dy_dtheta_num = (y_th_p - y_th_m) / (2 * h)

    # Finite difference w.r.t M
    x_m_p, y_m_p = curve(t_eval, theta, M + h, X, degrees=False)
    x_m_m, y_m_m = curve(t_eval, theta, M - h, X, degrees=False)
    dx_dM_num = (x_m_p - x_m_m) / (2 * h)
    dy_dM_num = (y_m_p - y_m_m) / (2 * h)

    # Finite difference w.r.t X
    x_x_p, y_x_p = curve(t_eval, theta, M, X + h, degrees=False)
    x_x_m, y_x_m = curve(t_eval, theta, M, X - h, degrees=False)
    dx_dX_num = (x_x_p - x_x_m) / (2 * h)
    dy_dX_num = (y_x_p - y_x_m) / (2 * h)

    # Compare J_x
    assert np.allclose(J_x_analytic[:, 0], dx_dtheta_num, rtol=1e-5, atol=1e-6)
    assert np.allclose(J_x_analytic[:, 1], dx_dM_num, rtol=1e-5, atol=1e-6)
    assert np.allclose(J_x_analytic[:, 2], dx_dX_num, rtol=1e-5, atol=1e-6)

    # Compare J_y
    assert np.allclose(J_y_analytic[:, 0], dy_dtheta_num, rtol=1e-5, atol=1e-6)
    assert np.allclose(J_y_analytic[:, 1], dy_dM_num, rtol=1e-5, atol=1e-6)
    assert np.allclose(J_y_analytic[:, 2], dy_dX_num, rtol=1e-5, atol=1e-6)


def test_finite_difference_parameter_jacobian_degrees():
    """Verify analytic Jacobian w.r.t theta_deg when degrees=True matches finite differences."""
    h = 1e-6
    t_eval = np.array([15.0, 35.0, 48.0])
    theta_deg = TEST_THETA_DEG
    M = TEST_M
    X = TEST_X

    J_x_analytic, J_y_analytic = curve_jacobian_params(t_eval, theta_deg, M, X, degrees=True)

    x_th_p, y_th_p = curve(t_eval, theta_deg + h, M, X, degrees=True)
    x_th_m, y_th_m = curve(t_eval, theta_deg - h, M, X, degrees=True)
    dx_dtheta_num = (x_th_p - x_th_m) / (2 * h)
    dy_dtheta_num = (y_th_p - y_th_m) / (2 * h)

    assert np.allclose(J_x_analytic[:, 0], dx_dtheta_num, rtol=1e-5, atol=1e-6)
    assert np.allclose(J_y_analytic[:, 0], dy_dtheta_num, rtol=1e-5, atol=1e-6)


def test_finite_difference_time_derivatives():
    """Verify analytic time derivatives dx/dt and dy/dt against finite differences."""
    h = 1e-7
    t_eval = np.linspace(8.0, 58.0, 20)
    
    dx_dt_ana, dy_dt_ana = curve_derivative_t(t_eval, TEST_THETA_RAD, TEST_M, degrees=False)
    
    x_p, y_p = curve(t_eval + h, TEST_THETA_RAD, TEST_M, TEST_X, degrees=False)
    x_m, y_m = curve(t_eval - h, TEST_THETA_RAD, TEST_M, TEST_X, degrees=False)
    
    dx_dt_num = (x_p - x_m) / (2 * h)
    dy_dt_num = (y_p - y_m) / (2 * h)
    
    assert np.allclose(dx_dt_ana, dx_dt_num, rtol=1e-5, atol=1e-6)
    assert np.allclose(dy_dt_ana, dy_dt_num, rtol=1e-5, atol=1e-6)
