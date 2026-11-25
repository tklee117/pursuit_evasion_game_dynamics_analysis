"""Print the main numbers of the analysis next to the values in the report.

    python scripts/run_analysis.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pursuit_evasion import analysis as A  # noqa: E402
from pursuit_evasion.dynamics import simulate_ivp  # noqa: E402
from pursuit_evasion.params import T_PICARD, X0, Params  # noqa: E402


def main():
    p = Params()
    t = np.linspace(0.0, T_PICARD, 400)
    X = simulate_ivp(X0, p, T_PICARD, t)

    print("Section IV   L_num along the Picard trajectory     ", f"{A.lipschitz_along(X, p):.4f}", "(report: 1.0012)")
    _, its = A.picard(X0, p, T_PICARD)
    errs = [np.linalg.norm(phi - X, axis=1).max() for phi in its]
    print("Section V    Picard max error, k = 0, 2, ..., 10    ", " ".join(f"{e:.1e}" for e in errs[::2]))
    cd = A.continuous_dependence(X0, p)
    print("Section V-B  gap below bound (IC / parameter)       ",
          bool((cd["gap_ic"] <= cd["bound_ic"]).all()), "/", bool((cd["gap_par"] <= cd["bound_par"]).all()))

    z_star, Am = A.linearize(p)
    print("Section VI   eigenvalues of A = Dg(z*)              ", np.round(np.sort_complex(np.linalg.eigvals(Am)), 4),
          "(report: -0.5 ± 0.354j, -0.5 ± 0.102j)")
    P = A.lyapunov(Am)
    print("Section VI   eigenvalues of P                       ", np.round(np.linalg.eigvalsh(P), 3), "(report: 0.564 ... 4.406)")
    print("Section VI   ROA level c*                           ", f"{A.roa_level(p, P, z_star):.1f}", "(report: 129.0)")
    print("Section VII  2 ||P B||_2                            ", f"{A.iss_gain(Am, P):.2f} m", "(report states 8.74 m; see README)")
    for mu in (0.1, 0.25, 0.5):
        print(f"Section VII  steady-state deviation, mu = {mu:<4}      ", f"{A.iss_steady_state(p, mu):.3f} m", f"(~4 mu = {4 * mu:.2f})")


if __name__ == "__main__":
    main()
