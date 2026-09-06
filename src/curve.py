"""
Parametric curve forward model and analytic derivatives.

Mathematical model:
    x(t) = t * cos(theta) - exp(M * |t|) * sin(0.3 * t) * sin(theta) + X
    y(t) = 42 + t * sin(theta) + exp(M * |t|) * sin(0.3 * t) * cos(theta)

Parameter Constraints:
    - 0 < theta < 50 degrees  (0 < theta_rad < 50 * pi / 180 rad)
    - -0.05 < M < 0.05
    - 0 < X < 100
    - 6 < t < 60

Note on domain:
    Because t in (6, 60), |t| = t throughout the valid domain.
    The implementation retains |t| in the formulas to adhere faithfully to
    the mathematical definition while optimizing for vectorization.
"""

from typing import Tuple, Union
import numpy as np

# Parameter bound constants
THETA_MIN_DEG: float = 0.0
THETA_MAX_DEG: float = 50.0
THETA_MIN_RAD: float = 0.0
THETA_MAX_RAD: float = float(np.radians(50.0))

M_MIN: float = -0.05
M_MAX: float = 0.05

X_MIN: float = 0.0
X_MAX: float = 100.0

T_MIN: float = 6.0
T_MAX: float = 60.0

OMEGA: float = 0.3  # Angular frequency in sin(0.3 * t)
Y_OFFSET: float = 42.0  # Base y-intercept constant


def to_radians(theta: float, degrees: bool = False) -> float:
    """
    Convert theta to radians if degrees is True, otherwise return as is.
    
    Parameters
    ----------
    theta : float
        Angle value in radians or degrees.
    degrees : bool, default=False
        If True, theta is interpreted as degrees and converted to radians.
        
    Returns
    -------
    float
        Angle in radians.
    """
    return float(np.radians(theta)) if degrees else float(theta)


def to_degrees(theta_rad: float) -> float:
    """
    Convert theta from radians to degrees.
    
    Parameters
    ----------
    theta_rad : float
        Angle in radians.
        
    Returns
    -------
    float
        Angle in degrees.
    """
    return float(np.degrees(theta_rad))


def validate_parameters(
    theta: float,
    M: float,
    X: float,
    degrees: bool = False,
    strict: bool = True,
) -> bool:
    """
    Validate that curve parameters theta, M, X satisfy the problem constraints.
    
    Constraints:
        - 0 < theta < 50 degrees
        - -0.05 < M < 0.05
        - 0 < X < 100
        
    Parameters
    ----------
    theta : float
        Rotation parameter (degrees if degrees=True, radians otherwise).
    M : float
        Exponential amplitude growth rate.
    X : float
        X-offset parameter.
    degrees : bool, default=False
        Whether theta is provided in degrees.
    strict : bool, default=True
        If True, raises ValueError on invalid parameters. If False, returns bool.
        
    Returns
    -------
    bool
        True if all parameters are within valid bounds.
        
    Raises
    ------
    ValueError
        If strict=True and any parameter violates the constraint bounds.
    """
    theta_deg = to_degrees(theta) if not degrees else theta
    
    valid_theta = THETA_MIN_DEG < theta_deg < THETA_MAX_DEG
    valid_M = M_MIN < M < M_MAX
    valid_X = X_MIN < X < X_MAX
    
    is_valid = valid_theta and valid_M and valid_X
    
    if not is_valid and strict:
        errors = []
        if not valid_theta:
            errors.append(f"theta={theta_deg:.4f} deg (must be in ({THETA_MIN_DEG}, {THETA_MAX_DEG}))")
        if not valid_M:
            errors.append(f"M={M:.6f} (must be in ({M_MIN}, {M_MAX}))")
        if not valid_X:
            errors.append(f"X={X:.4f} (must be in ({X_MIN}, {X_MAX}))")
        raise ValueError(f"Parameter bound violation: {'; '.join(errors)}")
        
    return is_valid


def validate_t(
    t: Union[float, np.ndarray],
    strict: bool = False,
) -> bool:
    """
    Validate that parameter t values fall within the designated domain (6, 60).
    
    Parameters
    ----------
    t : float or np.ndarray
        Evaluation parameter t.
    strict : bool, default=False
        If True, raises ValueError if any value is out of bounds.
        
    Returns
    -------
    bool
        True if all t values are within (6, 60).
        
    Raises
    ------
    ValueError
        If strict=True and any t is out of bounds.
    """
    t_arr = np.asarray(t)
    is_valid = bool(np.all((t_arr >= T_MIN) & (t_arr <= T_MAX)))
    
    if not is_valid and strict:
        min_val = float(np.min(t_arr))
        max_val = float(np.max(t_arr))
        raise ValueError(
            f"t-domain violation: values range [{min_val:.4f}, {max_val:.4f}], "
            f"expected within [{T_MIN}, {T_MAX}]"
        )
    return is_valid


def eval_amplitude(
    t: Union[float, np.ndarray],
    M: float,
) -> Union[float, np.ndarray]:
    """
    Evaluate the transverse oscillation envelope term:
        A(t, M) = exp(M * |t|) * sin(0.3 * t)
        
    Parameters
    ----------
    t : float or np.ndarray
        Curve parameter t.
    M : float
        Exponential growth parameter.
        
    Returns
    -------
    float or np.ndarray
        Computed oscillation term A(t, M).
    """
    t_arr = np.asarray(t)
    # Using np.abs(t) as per exact problem formulation
    amp = np.exp(M * np.abs(t_arr)) * np.sin(OMEGA * t_arr)
    return amp.item() if np.isscalar(t) else amp


def eval_x(
    t: Union[float, np.ndarray],
    theta: float,
    M: float,
    X: float,
    degrees: bool = False,
) -> Union[float, np.ndarray]:
    """
    Evaluate x(t) coordinate on the parametric curve:
        x(t) = t * cos(theta) - exp(M * |t|) * sin(0.3 * t) * sin(theta) + X
        
    Parameters
    ----------
    t : float or np.ndarray
        Parameter t.
    theta : float
        Rotation parameter (radians by default, degrees if degrees=True).
    M : float
        Exponential growth parameter.
    X : float
        X-offset parameter.
    degrees : bool, default=False
        If True, theta is treated as degrees.
        
    Returns
    -------
    float or np.ndarray
        Evaluated x coordinate(s).
    """
    th_rad = to_radians(theta, degrees=degrees)
    t_arr = np.asarray(t)
    amp = eval_amplitude(t_arr, M)
    x_val = t_arr * np.cos(th_rad) - amp * np.sin(th_rad) + X
    return x_val.item() if np.isscalar(t) else x_val


def eval_y(
    t: Union[float, np.ndarray],
    theta: float,
    M: float,
    degrees: bool = False,
) -> Union[float, np.ndarray]:
    """
    Evaluate y(t) coordinate on the parametric curve:
        y(t) = 42 + t * sin(theta) + exp(M * |t|) * sin(0.3 * t) * cos(theta)
        
    Parameters
    ----------
    t : float or np.ndarray
        Parameter t.
    theta : float
        Rotation parameter (radians by default, degrees if degrees=True).
    M : float
        Exponential growth parameter.
    degrees : bool, default=False
        If True, theta is treated as degrees.
        
    Returns
    -------
    float or np.ndarray
        Evaluated y coordinate(s).
    """
    th_rad = to_radians(theta, degrees=degrees)
    t_arr = np.asarray(t)
    amp = eval_amplitude(t_arr, M)
    y_val = Y_OFFSET + t_arr * np.sin(th_rad) + amp * np.cos(th_rad)
    return y_val.item() if np.isscalar(t) else y_val


def curve(
    t: Union[float, np.ndarray],
    theta: float,
    M: float,
    X: float,
    degrees: bool = False,
) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray]]:
    """
    Evaluate the complete (x, y) coordinates for parameter values t:
        x(t) = t * cos(theta) - exp(M * |t|) * sin(0.3 * t) * sin(theta) + X
        y(t) = 42 + t * sin(theta) + exp(M * |t|) * sin(0.3 * t) * cos(theta)
        
    Parameters
    ----------
    t : float or np.ndarray
        Parameter t values.
    theta : float
        Rotation parameter (radians by default, degrees if degrees=True).
    M : float
        Exponential growth parameter.
    X : float
        X-offset parameter.
    degrees : bool, default=False
        If True, theta is treated as degrees.
        
    Returns
    -------
    tuple of (float or np.ndarray, float or np.ndarray)
        (x(t), y(t)) coordinate arrays or floats.
    """
    x = eval_x(t, theta, M, X, degrees=degrees)
    y = eval_y(t, theta, M, degrees=degrees)
    return x, y


def curve_jacobian_params(
    t: Union[float, np.ndarray],
    theta: float,
    M: float,
    X: float,
    degrees: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the analytic Jacobian of [x(t), y(t)] with respect to parameters (theta, M, X).
    
    Derivation for theta (in radians):
        dx/dtheta = -t * sin(theta) - A(t, M) * cos(theta)
        dy/dtheta =  t * cos(theta) - A(t, M) * sin(theta)
        
    Derivation for M (for any t, d|t|/dM = 0, d(exp(M|t|))/dM = |t|*exp(M|t|)):
        dA/dM = |t| * exp(M * |t|) * sin(0.3 * t)
        dx/dM = -dA/dM * sin(theta)
        dy/dM =  dA/dM * cos(theta)
        
    Derivation for X:
        dx/dX = 1.0
        dy/dX = 0.0
        
    Note on degrees:
        If degrees=True, the derivative w.r.t theta_deg is scaled by (pi / 180)
        via the chain rule: d/d(theta_deg) = (pi / 180) * d/d(theta_rad).
        
    Parameters
    ----------
    t : float or np.ndarray
        Parameter t values.
    theta : float
        Rotation parameter.
    M : float
        Exponential growth parameter.
    X : float
        X-offset parameter (not explicitly used in value, but present in signature).
    degrees : bool, default=False
        Whether theta is in degrees.
        
    Returns
    -------
    tuple of (np.ndarray, np.ndarray)
        - J_x: shape (N, 3) or (3,) containing [dx/dtheta, dx/dM, dx/dX]
        - J_y: shape (N, 3) or (3,) containing [dy/dtheta, dy/dM, dy/dX]
    """
    th_rad = to_radians(theta, degrees=degrees)
    t_arr = np.asarray(t, dtype=np.float64)
    is_scalar = np.isscalar(t)
    if is_scalar:
        t_arr = np.array([t_arr])
        
    abs_t = np.abs(t_arr)
    sin_wt = np.sin(OMEGA * t_arr)
    exp_mt = np.exp(M * abs_t)
    amp = exp_mt * sin_wt
    
    sin_th = np.sin(th_rad)
    cos_th = np.cos(th_rad)
    
    # Derivatives w.r.t theta (rad)
    dx_dtheta = -t_arr * sin_th - amp * cos_th
    dy_dtheta = t_arr * cos_th - amp * sin_th
    
    # If theta is in degrees, scale by pi / 180
    if degrees:
        rad_scale = np.pi / 180.0
        dx_dtheta = dx_dtheta * rad_scale
        dy_dtheta = dy_dtheta * rad_scale
        
    # Derivatives w.r.t M
    dA_dM = abs_t * exp_mt * sin_wt
    dx_dM = -dA_dM * sin_th
    dy_dM = dA_dM * cos_th
    
    # Derivatives w.r.t X
    dx_dX = np.ones_like(t_arr)
    dy_dX = np.zeros_like(t_arr)
    
    # Stack into (N, 3)
    J_x = np.column_stack([dx_dtheta, dx_dM, dx_dX])
    J_y = np.column_stack([dy_dtheta, dy_dM, dy_dX])
    
    if is_scalar:
        return J_x[0], J_y[0]
    return J_x, J_y


def curve_derivative_t(
    t: Union[float, np.ndarray],
    theta: float,
    M: float,
    degrees: bool = False,
) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray]]:
    """
    Compute analytic tangent / velocity derivatives [dx/dt, dy/dt].
    
    Derivation for t != 0:
        d/dt (|t|) = sign(t)
        dA/dt = exp(M * |t|) * [ M * sign(t) * sin(0.3 * t) + 0.3 * cos(0.3 * t) ]
        dx/dt = cos(theta) - (dA/dt) * sin(theta)
        dy/dt = sin(theta) + (dA/dt) * cos(theta)
        
    For valid domain t > 6, sign(t) = 1 identically.
    
    Parameters
    ----------
    t : float or np.ndarray
        Parameter t values.
    theta : float
        Rotation parameter.
    M : float
        Exponential growth parameter.
    degrees : bool, default=False
        Whether theta is in degrees.
        
    Returns
    -------
    tuple of (float or np.ndarray, float or np.ndarray)
        (dx/dt, dy/dt) tangent vector components.
    """
    th_rad = to_radians(theta, degrees=degrees)
    t_arr = np.asarray(t, dtype=np.float64)
    
    abs_t = np.abs(t_arr)
    sgn_t = np.sign(t_arr)
    exp_mt = np.exp(M * abs_t)
    sin_wt = np.sin(OMEGA * t_arr)
    cos_wt = np.cos(OMEGA * t_arr)
    
    # dA/dt by product rule and chain rule
    dA_dt = exp_mt * (M * sgn_t * sin_wt + OMEGA * cos_wt)
    
    cos_th = np.cos(th_rad)
    sin_th = np.sin(th_rad)
    
    dx_dt = cos_th - dA_dt * sin_th
    dy_dt = sin_th + dA_dt * cos_th
    
    if np.isscalar(t):
        return float(dx_dt.item()), float(dy_dt.item())
    return dx_dt, dy_dt
