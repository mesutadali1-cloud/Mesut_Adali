"""
DEP 521E - Homework 3, Part C: Newton-Raphson iteration with the Newmark-beta solver
for a bilinear SDOF system (coded with AI).

Run:  python nr_newmark.py              (original Newton-Raphson, Section 1)
      python nr_newmark.py --modified   (modified Newton-Raphson, Section 2)

Outputs (written next to this script, in figures/):
  displacement_history.png, spring_force_history.png, hysteresis.png,
  relative_acceleration_history.png, absolute_acceleration_history.png
and, on the command line, the Newton-Raphson iteration counts for the yielding
steps and the analysis time.
"""

import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ACC_FILE = HERE.parents[2] / "hw01" / "python" / "ground_acc_HW1.txt"
G = 9.806          # m/s^2
DT = 0.005         # s, time step of the HW1 ground motion

INTERIOR = "interior"
TOP = "top"
BOTTOM = "bottom"

LINE_COLOR = "#1f5a99"
BORDER_COLOR = "#8a8a8a"


# --------------------------------------------------------------------------- #
# Bilinear spring (same model as HW2 Part C)
# --------------------------------------------------------------------------- #
def bilinear(dy, fy, k2, x_ini, f_ini, dx, param_a=INTERIOR):
    """Force increment of the bilinear spring for one displacement increment.

    The step always starts from the last converged state (x_ini, f_ini, param_a),
    so calling it repeatedly inside the Newton-Raphson loop is path independent.
    Returns (df, state at the end of the increment).
    """
    kp = fy / dy
    if dx == 0:
        return 0.0, param_a

    qy = fy - k2 * dy
    x_end = x_ini + dx
    f_top = qy + k2 * x_end
    f_bottom = -qy + k2 * x_end

    f_trial = f_ini + kp * dx
    if f_trial >= f_top:
        return f_top - f_ini, TOP
    if f_trial <= f_bottom:
        return f_bottom - f_ini, BOTTOM
    return f_trial - f_ini, INTERIOR


def tangent(state, kp, k2):
    """Tangent stiffness of the spring: k2 on a yield border, kp inside."""
    return k2 if state in (TOP, BOTTOM) else kp


# --------------------------------------------------------------------------- #
# Newmark-beta with Newton-Raphson iteration
# --------------------------------------------------------------------------- #
def newmark_nr(m, c, dy, fy, k2, p, dt, beta=0.25, gamma=0.5,
               modified=False, tol=0.01, max_iter=50):
    """Nonlinear Newmark-beta (average acceleration by default) with Newton-Raphson.

    original NR : the tangent stiffness is updated at every iteration.
    modified NR : the tangent stiffness of the start of the step is kept
                  through all iterations of that step.
    Convergence : |f_ub^(j+1)| / |f_ub^(j)| < tol, where f_ub^(j) is the unbalanced
                  spring force left after iteration j (a zero unbalance, as in an
                  elastic step, also counts as converged).
    """
    kp = fy / dy
    abs_tol = 1e-9 * fy        # round-off floor so an exact zero unbalance is recognised
    n = len(p)
    x = np.zeros(n)
    v = np.zeros(n)
    a = np.zeros(n)
    fs = np.zeros(n)
    states = [INTERIOR] * n
    iters = np.zeros(n, dtype=int)

    a[0] = (p[0] - c * v[0] - fs[0]) / m

    a1 = m / (beta * dt**2) + gamma / (beta * dt) * c
    a2 = m / (beta * dt) + (gamma / beta - 1) * c
    a3 = (1 / (2 * beta) - 1) * m + dt * (gamma / (2 * beta) - 1) * c

    for i in range(n - 1):
        p_hat = p[i + 1] + a1 * x[i] + a2 * v[i] + a3 * a[i]

        x_trial = x[i]
        state = states[i]
        f_trial = fs[i]
        k_fixed = tangent(states[i], kp, k2) + a1

        r = p_hat - f_trial - a1 * x_trial              # load before the first iteration
        r_prev = None
        j = 0
        while True:
            k_hat = k_fixed if modified else tangent(state, kp, k2) + a1
            x_trial += r / k_hat
            df, state = bilinear(dy, fy, k2, x[i], fs[i], x_trial - x[i], states[i])
            f_trial = fs[i] + df
            j += 1
            r = p_hat - f_trial - a1 * x_trial          # unbalanced spring force f_ub^(j)
            if abs(r) <= abs_tol:                       # elastic step or exact solution
                break
            if r_prev is not None and abs(r) / abs(r_prev) < tol:
                break
            if j >= max_iter:
                raise RuntimeError(f"Newton-Raphson did not converge at step {i + 1}")
            r_prev = r

        x[i + 1] = x_trial
        fs[i + 1] = f_trial
        states[i + 1] = state
        iters[i + 1] = j

        v[i + 1] = (gamma / (beta * dt) * (x[i + 1] - x[i])
                    + (1 - gamma / beta) * v[i]
                    + dt * (1 - gamma / (2 * beta)) * a[i])
        a[i + 1] = ((x[i + 1] - x[i]) / (beta * dt**2)
                    - v[i] / (beta * dt)
                    - (1 / (2 * beta) - 1) * a[i])

    return x, v, a, fs, states, iters


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
def save_history_plot(t, y, ylabel, title, path):
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(t, y, color=LINE_COLOR, linewidth=1)
    ax.axhline(0, color=BORDER_COLOR, linewidth=0.8)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def save_hysteresis_plot(x, f, dy, fy, k2, title, path):
    qy = fy - k2 * dy
    span = max(abs(x.min()), abs(x.max())) * 1.1
    xr = np.array([-span, span])

    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.plot(xr * 100, (qy + k2 * xr) / 1000, color=BORDER_COLOR, linestyle="--", linewidth=1)
    ax.plot(xr * 100, (-qy + k2 * xr) / 1000, color=BORDER_COLOR, linestyle="--", linewidth=1)
    ax.plot(x * 100, f / 1000, color=LINE_COLOR, linewidth=1)
    ax.axhline(0, color=BORDER_COLOR, linewidth=0.8)
    ax.axvline(0, color=BORDER_COLOR, linewidth=0.8)
    ax.set_xlabel("Displacement [cm]")
    ax.set_ylabel("Spring force [kN]")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--modified", action="store_true",
                        help="use the modified Newton-Raphson algorithm")
    args = parser.parse_args()

    # SDOF parameters (Table 1)
    m = 1000.0                 # kg
    xi = 0.02
    T = 1.0                    # s
    wn = 2 * np.pi / T
    kp = m * wn**2             # N/m
    c = 2 * xi * m * wn        # N s/m
    dy = 0.075                 # m
    fy = kp * dy               # N
    k2 = 0.1 * kp              # N/m

    ag = np.loadtxt(ACC_FILE) * G        # m/s^2
    t = np.arange(len(ag)) * DT
    p = -m * ag

    name = "Modified" if args.modified else "Original"
    t0 = time.perf_counter()
    x, v, a, fs, states, iters = newmark_nr(m, c, dy, fy, k2, p, DT, modified=args.modified)
    elapsed = time.perf_counter() - t0

    a_abs = a + ag
    yield_steps = [i for i in range(1, len(t)) if states[i] in (TOP, BOTTOM)]

    print(f"{name} Newton-Raphson + Newmark-beta (beta = 1/4, gamma = 1/2), dt = {DT} s")
    print(f"kp = {kp:.1f} N/m, fy = {fy:.1f} N, ks = {k2:.1f} N/m, c = {c:.2f} N s/m")
    print(f"Steps: {len(t) - 1}, steps on a yield border: {len(yield_steps)}")
    if yield_steps:
        it_y = iters[yield_steps]
        counts = {k: int((it_y == k).sum()) for k in sorted(set(it_y))}
        print("Iterations in yielding steps:",
              ", ".join(f"{k} iter -> {n} steps" for k, n in counts.items()))
        print(f"  max = {it_y.max()}, mean = {it_y.mean():.2f}")
        print("First yielding steps:")
        for i in yield_steps[:5]:
            print(f"  t = {t[i]:7.3f} s  x = {x[i] * 100:8.3f} cm  "
                  f"fs = {fs[i]:9.1f} N  state = {states[i]:6s}  iterations = {iters[i]}")
    print(f"Total iterations: {iters.sum()}")
    print(f"Max |x| = {np.abs(x).max() * 100:.3f} cm, max |fs| = {np.abs(fs).max():.1f} N, "
          f"max |v| = {np.abs(v).max():.4f} m/s, max |a_abs| = {np.abs(a_abs).max():.4f} m/s^2")
    print(f"Analysis time: {elapsed:.4f} s")

    out = HERE / "figures"
    out.mkdir(exist_ok=True)
    tag = "modified_" if args.modified else ""
    save_hysteresis_plot(x, fs, dy, fy, k2, f"{name} NR - displacement vs spring force",
                         out / f"{tag}hysteresis.png")
    if not args.modified:
        save_history_plot(t, x * 100, "Displacement [cm]", "Displacement time-history",
                          out / "displacement_history.png")
        save_history_plot(t, fs / 1000, "Spring force [kN]", "Spring force time-history",
                          out / "spring_force_history.png")
        save_history_plot(t, a, "Relative acceleration [m/s$^2$]",
                          "Relative acceleration time-history",
                          out / "relative_acceleration_history.png")
        save_history_plot(t, a_abs, "Absolute acceleration [m/s$^2$]",
                          "Absolute acceleration time-history",
                          out / "absolute_acceleration_history.png")


if __name__ == "__main__":
    main()
