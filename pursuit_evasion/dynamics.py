"""Closed-loop pursuit-evasion dynamics (report Section III).

State X = (x_P, y_P, vx_P, vy_P, x_E, y_E, vx_E, vy_E) in R^8. Both agents are double integrators;
the inputs are the sigmoid-blended feedback laws of Sections III-A and III-B.
"""
import numpy as np
from scipy.integrate import solve_ivp

from .params import Params


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def unit(v, eps=1e-12):
    n = np.linalg.norm(v)
    return v / n if n > eps else np.zeros_like(v)


def saturate(v, vmax):
    """Velocity saturation v <- v * min(1, vmax/||v||) (Section III-C)."""
    n = np.linalg.norm(v)
    return v if n <= vmax else v * (vmax / n)


def pursuer_direction(xP, xE, p: Params):
    """n_P = alpha(rho) r/rho + (1 - alpha(rho)) (x_Z - x_P)/||x_Z - x_P||   (eq. 4-5)."""
    r = xE - xP
    rho = np.linalg.norm(r)
    alpha = sigmoid(p.k_alpha * (rho - p.rho_star))
    return alpha * unit(r) + (1.0 - alpha) * unit(p.x_Z - xP)


def evader_direction(xP, xE, p: Params):
    """n_E = d_hat + w(rho) f_perp: approach the zone, dodge sideways (eq. 6-8)."""
    r = xE - xP
    rho = np.linalg.norm(r)
    d_hat = unit(p.x_Z - xE)
    flee = unit(r)
    f_perp = unit(flee - np.dot(flee, d_hat) * d_hat)
    w = sigmoid(-p.k_beta * (rho - p.d_star))
    return d_hat + w * f_perp


def u_pursuer(X, p: Params):
    return p.a_P * unit(pursuer_direction(X[0:2], X[4:6], p))


def u_evader(X, p: Params):
    return p.a_E * unit(evader_direction(X[0:2], X[4:6], p))


def f(X, p: Params):
    """Xdot = f(X); positions move with the saturated velocity."""
    vP = saturate(X[2:4], p.v_P)
    vE = saturate(X[6:8], p.v_E)
    return np.concatenate([vP, u_pursuer(X, p), vE, u_evader(X, p)])


def outcome(X, p: Params):
    """'capture', 'breach', or None while the game is running."""
    if np.linalg.norm(X[4:6] - X[0:2]) <= p.r_c:
        return "capture"
    if np.linalg.norm(X[4:6] - p.x_Z) <= p.R_Z:
        return "breach"
    return None


def simulate(X0, p: Params, T=40.0, dt=0.01, stop_at_end=True):
    """Fixed-step RK4 with velocity projection after each step.
    Returns (t, X, result) where result is 'capture', 'breach' or 'timeout'."""
    n = int(round(T / dt))
    X = np.empty((n + 1, 8)); X[0] = X0
    t = np.arange(n + 1) * dt
    result = "timeout"
    for k in range(n):
        x = X[k]
        k1 = f(x, p); k2 = f(x + 0.5 * dt * k1, p); k3 = f(x + 0.5 * dt * k2, p); k4 = f(x + dt * k3, p)
        x = x + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        x[2:4] = saturate(x[2:4], p.v_P); x[6:8] = saturate(x[6:8], p.v_E)
        X[k + 1] = x
        res = outcome(x, p)
        if res and stop_at_end:
            return t[:k + 2], X[:k + 2], res
    return t, X, result


def simulate_ivp(X0, p: Params, T, t_eval, rtol=1e-9, atol=1e-11):
    """RK45 reference solution (used for Picard iteration and continuous dependence)."""
    sol = solve_ivp(lambda t, x: f(x, p), (0.0, T), X0, t_eval=t_eval, method="RK45", rtol=rtol, atol=atol)
    return sol.y.T


def jacobian(fun, x, h=1e-6):
    """Central finite-difference Jacobian of fun at x."""
    n = len(x)
    J = np.empty((len(fun(x)), n))
    for i in range(n):
        e = np.zeros(n); e[i] = h
        J[:, i] = (fun(x + e) - fun(x - e)) / (2 * h)
    return J
