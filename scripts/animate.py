"""Animate the example games to outputs/pursuit_evasion.gif.

    python scripts/animate.py
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402
import numpy as np  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
from pursuit_evasion.dynamics import simulate  # noqa: E402
from pursuit_evasion.params import Params  # noqa: E402
from pursuit_evasion.scenarios import SCENARIOS  # noqa: E402


def main(fps=20, step=5):
    p = Params()
    runs = []
    for name, sc in SCENARIOS.items():
        q = p.with_(**sc["params"])
        t, X, res = simulate(sc["X0"], q, T=sc["T"])
        runs.append((name, q, t, X, res))
    n = max(len(r[2]) for r in runs)
    fig, axes = plt.subplots(1, len(runs), figsize=(5 * len(runs), 4.6))
    arts = []
    for ax, (name, q, t, X, res) in zip(axes, runs):
        ax.add_patch(plt.Circle(q.x_Z, q.R_Z, color="#e8a33d", alpha=0.25, ls="--"))
        allxy = np.vstack([X[:, :2], X[:, 4:6]])
        lo, hi = allxy.min(0) - 2, allxy.max(0) + 2
        ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_aspect("equal"); ax.grid(alpha=0.3)
        ax.set_title(name)
        lp, = ax.plot([], [], color="#1f5fa8"); le, = ax.plot([], [], color="#c0392b")
        dp, = ax.plot([], [], "o", color="#1f5fa8", ms=8); de, = ax.plot([], [], "s", color="#c0392b", ms=8)
        txt = ax.text(0.03, 0.95, "", transform=ax.transAxes, va="top")
        arts.append((t, X, res, lp, le, dp, de, txt))

    def update(frame):
        k = frame * step
        out = []
        for t, X, res, lp, le, dp, de, txt in arts:
            i = min(k, len(t) - 1)
            lp.set_data(X[:i + 1, 0], X[:i + 1, 1]); le.set_data(X[:i + 1, 4], X[:i + 1, 5])
            dp.set_data([X[i, 0]], [X[i, 1]]); de.set_data([X[i, 4]], [X[i, 5]])
            txt.set_text(f"t = {t[i]:.1f} s" + (f"  {res.upper()}" if i == len(t) - 1 else ""))
            out += [lp, le, dp, de, txt]
        return out

    anim = FuncAnimation(fig, update, frames=n // step + fps, blit=True)
    os.makedirs(os.path.join(ROOT, "outputs"), exist_ok=True)
    path = os.path.join(ROOT, "outputs", "pursuit_evasion.gif")
    anim.save(path, writer=PillowWriter(fps=fps)); print("wrote", path)


if __name__ == "__main__":
    main()
