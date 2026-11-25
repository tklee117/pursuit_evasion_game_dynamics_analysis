"""Nonlinear analysis tools applied to the game (report Sections IV-VIII)."""
import numpy as np
from scipy.integrate import cumulative_trapezoid, solve_ivp
from scipy.linalg import solve_continuous_lyapunov

from .dynamics import f, jacobian, pursuer_direction, simulate, simulate_ivp, unit
from .params import X_E_STAR, X_P_STAR, Params


# ---------- Sections IV-V: Lipschitz constant and Picard iteration ----------

def lipschitz_along(X_traj, p: Params, stride=5):
    """L_num = max_t sigma_1(Df(X(t))) along a trajectory (eq. 19)."""
    return max(np.linalg.norm(jacobian(lambda x: f(x, p), x), 2) for x in X_traj[::stride])


def picard(X0, p: Params, T, n_iter=10, n_grid=400):
    """Picard iterates phi_{k+1}(t) = X0 + int_0^t f(phi_k(s)) ds on a uniform grid (trapezoid rule)."""
    t = np.linspace(0.0, T, n_grid)
    phi = np.tile(X0, (n_grid, 1))
    iterates = [phi.copy()]
    for _ in range(n_iter):
        F = np.array([f(x, p) for x in phi])
        phi = X0 + cumulative_trapezoid(F, t, axis=0, initial=0.0)
        iterates.append(phi.copy())
    return t, iterates


# ---------- Section V-B: continuous dependence ----------

def continuous_dependence(X0, p: Params, T=0.5, eps=0.5, dA=0.3, n=200):
    """Two experiments for ||x(t) - z(t)|| <= ||x0-z0|| e^{Lt} + mu t e^{Lt} (eq. 18).
    (A) initial-condition gap eps on the pursuer position; (B) evader acceleration raised by dA.
    Returns t, gaps and bounds for both experiments and the L used."""
    t = np.linspace(0.0, T, n)
    x = simulate_ivp(X0, p, T, t)
    L = lipschitz_along(x, p, stride=10)
    z0 = X0.copy(); z0[0] += eps
    gap_ic = np.linalg.norm(simulate_ivp(z0, p, T, t) - x, axis=1)
    bound_ic = eps * np.exp(L * t)
    gap_par = np.linalg.norm(simulate_ivp(X0, p.with_(a_E=p.a_E + dA), T, t) - x, axis=1)
    bound_par = dA * t * np.exp(L * t)
    return dict(t=t, L=L, gap_ic=gap_ic, bound_ic=bound_ic, gap_par=gap_par, bound_par=bound_par)


# ---------- Section VI: modified pursuer law, reduced system, Lyapunov ----------

def g_reduced(z, p: Params, xE=X_E_STAR):
    """Reduced pursuer subsystem z = (x_P, v_P), evader frozen at x_E* (eq. 20-22):
    u_P = a_P tanh(||n_P||) n_P/||n_P|| - k_v v_P."""
    xP, vP = z[:2], z[2:]
    n = pursuer_direction(xP, xE, p)
    u = p.a_P * np.tanh(np.linalg.norm(n)) * unit(n) - p.k_v * vP
    return np.concatenate([vP, u])


def linearize(p: Params):
    z_star = np.concatenate([X_P_STAR, np.zeros(2)])
    A = jacobian(lambda z: g_reduced(z, p), z_star)
    return z_star, A


def lyapunov(A):
    """P from A^T P + P A = -I."""
    return solve_continuous_lyapunov(A.T, -np.eye(A.shape[0]))


def roa_level(p: Params, P, z_star, n_dir=500, c_max=1e4, seed=0, iters=40):
    """Largest c with Vdot < 0 on {V = c}, by directional binary search (Section VI-D)."""
    rng = np.random.default_rng(seed)
    dirs = rng.normal(size=(n_dir, 4)); dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)

    def vdot_ok(c, d):
        dz = d * np.sqrt(c / (d @ P @ d))  # point on the level set V = c
        return 2 * dz @ P @ g_reduced(z_star + dz, p) < 0

    # The level set must stay inside the game domain: it cannot reach the frozen evader,
    # where the pursuer law is singular (rho = 0). This caps c at V(x_E*, 0).
    dz_E = np.concatenate([X_E_STAR, np.zeros(2)]) - z_star
    best = min(c_max, float(dz_E @ P @ dz_E))
    for d in dirs:
        lo, hi = 0.0, best
        if vdot_ok(hi, d):
            continue
        for _ in range(iters):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if vdot_ok(mid, d) else (lo, mid)
        best = min(best, lo)
    return best


# ---------- Section VII: input-to-state stability ----------

def iss_gain(A, P):
    """gamma_ISS = 2 ||P B||_2 with B = [0; I] (disturbance on the pursuer acceleration)."""
    B = np.vstack([np.zeros((2, 2)), np.eye(2)])
    return 2 * np.linalg.norm(P @ B, 2)


def iss_steady_state(p: Params, mu, T=60.0):
    """Steady-state deviation of the reduced system under a constant disturbance of size mu."""
    z_star = np.concatenate([X_P_STAR, np.zeros(2)])
    d = np.array([0.0, mu])
    sol = solve_ivp(lambda t, z: g_reduced(z, p) + np.concatenate([np.zeros(2), d]), (0, T), z_star, rtol=1e-9, atol=1e-11)
    return np.linalg.norm(sol.y[:, -1] - z_star)


# ---------- Section VIII: barrier function ----------

def barrier_min(X0, p: Params, T=40.0):
    """B = rho - r_c along a game; returns (B_min, outcome)."""
    t, X, res = simulate(X0, p, T=T)
    B = np.linalg.norm(X[:, 4:6] - X[:, 0:2], axis=1) - p.r_c
    return B.min(), res


def speed_ratio_sweep(X0, p: Params, ratios):
    """B_min as the pursuer's acceleration grows relative to the evader's (a_E fixed)."""
    return np.array([barrier_min(X0, p.with_(a_P=r * p.a_E, v_P=max(p.v_P, r * p.v_E)))[0] for r in ratios])
