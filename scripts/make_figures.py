"""Regenerate the analysis figures into outputs/.

    python scripts/make_figures.py
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
from pursuit_evasion import analysis as A  # noqa: E402
from pursuit_evasion.dynamics import simulate, simulate_ivp  # noqa: E402
from pursuit_evasion.params import T_PICARD, X0, Params  # noqa: E402
from pursuit_evasion.scenarios import SCENARIOS  # noqa: E402

OUT = os.path.join(ROOT, "outputs")
BLUE, RED, GREEN = "#1f5fa8", "#c0392b", "#2e8b57"


def zone(ax, p):
    ax.add_patch(plt.Circle(p.x_Z, p.R_Z, color="#e8a33d", alpha=0.25, lw=1.5, ls="--"))


def fig_trajectories(p):
    fig, axes = plt.subplots(1, len(SCENARIOS), figsize=(5 * len(SCENARIOS), 4.6))
    for ax, (name, sc) in zip(axes, SCENARIOS.items()):
        q = p.with_(**sc["params"])
        t, X, res = simulate(sc["X0"], q, T=sc["T"])
        zone(ax, q)
        ax.plot(X[:, 0], X[:, 1], color=BLUE, label="pursuer P")
        ax.plot(X[:, 4], X[:, 5], color=RED, label="evader E")
        ax.plot(*X[0, :2], "^", color=BLUE); ax.plot(*X[0, 4:6], "^", color=RED)
        ax.plot(*X[-1, :2], "o", color=BLUE); ax.plot(*X[-1, 4:6], "s", color=RED)
        ax.set_title(f"{name}: {res} at t = {t[-1]:.1f} s"); ax.set_aspect("equal"); ax.grid(alpha=0.3)
    axes[0].legend(loc="best")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "trajectories.png"), dpi=150); plt.close(fig)


def fig_picard(p):
    t = np.linspace(0.0, T_PICARD, 400)
    X = simulate_ivp(X0, p, T_PICARD, t)
    _, its = A.picard(X0, p, T_PICARD)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.4))
    zone(a1, p)
    for k, phi in enumerate(its):
        g = 0.85 - 0.8 * k / len(its)
        a1.plot(phi[:, 0], phi[:, 1], color=str(g), lw=0.8); a1.plot(phi[:, 4], phi[:, 5], color=str(g), lw=0.8)
    a1.plot(X[:, 0], X[:, 1], color=BLUE, lw=2, label="P (RK45)"); a1.plot(X[:, 4], X[:, 5], color=RED, lw=2, label="E (RK45)")
    a1.set_aspect("equal"); a1.legend(); a1.set_title("Picard iterates (grey to black) vs. RK45")
    a2.semilogy(range(len(its)), [np.linalg.norm(phi - X, axis=1).max() for phi in its], "o-", color=BLUE)
    a2.set_xlabel("iterate k"); a2.set_ylabel(r"$\max_t\,\|\varphi_k - X_{RK45}\|$"); a2.grid(alpha=0.3, which="both")
    a2.set_title("Picard convergence")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "picard.png"), dpi=150); plt.close(fig)


def fig_stability(p):
    z_star, Am = A.linearize(p)
    P = A.lyapunov(Am)
    c = A.roa_level(p, P, z_star)
    ev = np.linalg.eigvals(Am)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.4))
    a1.axvspan(-2, 0, color=GREEN, alpha=0.1); a1.axvline(0, color="k", lw=0.8); a1.axhline(0, color="k", lw=0.8)
    a1.plot(ev.real, ev.imag, "o", ms=9, color=BLUE)
    a1.set_xlim(-1.5, 0.5); a1.set_xlabel(r"Re $\lambda$"); a1.set_ylabel(r"Im $\lambda$"); a1.set_title(r"Eigenvalues of $A = Dg(z^*)$")
    xs, ys = np.meshgrid(np.linspace(-6, 15, 160), np.linspace(-9, 9, 140))
    V = np.zeros_like(xs)
    for i in range(xs.shape[0]):
        for j in range(xs.shape[1]):
            dz = np.array([xs[i, j], ys[i, j], 0, 0]) - z_star
            V[i, j] = dz @ P @ dz
    zone(a2, p)
    a2.contourf(xs, ys, V, levels=[0, 2, 8, 32, c], cmap="Blues_r", alpha=0.8)
    a2.contour(xs, ys, V, levels=[c], colors=RED)
    a2.plot(*z_star[:2], "k*", ms=12, label=r"$x_P^*$"); a2.plot(12.5, 0, "s", color=RED, label=r"$x_E^*$ (fixed)")
    a2.set_aspect("equal"); a2.legend(loc="lower left"); a2.set_title(rf"Lyapunov level sets, ROA $V \leq {c:.0f}$ ($v_P = 0$)")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "stability.png"), dpi=150); plt.close(fig)


def fig_iss(p):
    z_star, Am = A.linearize(p)
    gam = A.iss_gain(Am, A.lyapunov(Am))
    mus = np.linspace(0.05, 1.0, 12)
    ss = [A.iss_steady_state(p, m) for m in mus]
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    ax.plot(mus, ss, "o", color=BLUE, label="simulated")
    ax.plot(mus, gam * mus, "--", color=RED, label=rf"$2\|PB\|\,\mu$ = {gam:.2f} $\mu$")
    ax.plot(mus, 4 * mus, ":", color="purple", label=r"$4\mu$")
    ax.set_xlabel(r"disturbance $\mu$ [m/s$^2$]"); ax.set_ylabel(r"$\|X(\infty) - X^*\|$ [m]"); ax.grid(alpha=0.3); ax.legend()
    ax.set_title("ISS: steady-state deviation")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "iss.png"), dpi=150); plt.close(fig)


def fig_continuous_dependence(p):
    cd = A.continuous_dependence(X0, p)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    for ax, gap, bound, title in [(a1, cd["gap_ic"], cd["bound_ic"], "(A) initial-condition gap 0.5 m"),
                                  (a2, cd["gap_par"], cd["bound_par"], r"(B) evader acceleration $+0.3$ m/s$^2$")]:
        ax.plot(cd["t"], gap, color=BLUE, label="actual difference"); ax.plot(cd["t"], bound, "--", color=RED, label="bound (eq. 18)")
        ax.fill_between(cd["t"], gap, bound, color=RED, alpha=0.08); ax.set_xlabel("t [s]"); ax.set_title(title); ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "continuous_dependence.png"), dpi=150); plt.close(fig)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    p = Params()
    for fn in (fig_trajectories, fig_picard, fig_continuous_dependence, fig_stability, fig_iss):
        fn(p); print("wrote", fn.__name__)
