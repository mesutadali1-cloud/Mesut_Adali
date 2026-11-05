"""
DEP 521E - Homework 2, Part C: Bilinear element model and testing (coded with AI).

Run:  python bilinear_element.py
Output: displacement_history.png, force_history.png, hysteresis.png and a results table.

Time is expressed as the step index because the assignment gives no time step.
"""

from pathlib import Path

import matplotlib.pyplot as plt

INTERIOR = "interior"
TOP = "top"
BOTTOM = "bottom"

LINE_COLOR = "#1f5a99"
BORDER_COLOR = "#8a8a8a"


def bilinear(dy, fy, k2, x_ini, f_ini, dx, param_a=INTERIOR):
    """Return the force increment of a bilinear element for one displacement step.

    dy, fy  : yield displacement and yield force
    k2      : secondary stiffness (must be smaller than fy / dy)
    x_ini   : displacement at the start of the step
    f_ini   : force at the start of the step
    dx      : displacement increment
    param_a : state at the start of the step (INTERIOR, TOP or BOTTOM)

    Returns (df, param_a), with param_a updated to the state at the end of the step.
    """
    kp = fy / dy
    if k2 >= kp:
        raise ValueError("k2 must be smaller than the primary stiffness fy / dy")

    if dx == 0:
        return 0.0, param_a

    qy = fy - k2 * dy
    x_end = x_ini + dx
    f_top = qy + k2 * x_end
    f_bottom = -qy + k2 * x_end

    # Elastic trial with kp; a trial beyond a border means the step ends on that border.
    f_trial = f_ini + kp * dx
    if f_trial >= f_top:
        return f_top - f_ini, TOP
    if f_trial <= f_bottom:
        return f_bottom - f_ini, BOTTOM
    return f_trial - f_ini, INTERIOR


def run_history(x, dy, fy, k2):
    """Drive the element through displacement history x, starting from zero force."""
    forces = [0.0]
    states = [INTERIOR]
    for x_ini, x_end in zip(x[:-1], x[1:]):
        df, state = bilinear(dy, fy, k2, x_ini, forces[-1], x_end - x_ini, states[-1])
        forces.append(forces[-1] + df)
        states.append(state)
    return forces, states


def save_history_plot(y, ylabel, title, path):
    steps = range(len(y))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(steps, y, color=LINE_COLOR, linewidth=2, marker="o", markersize=5)
    ax.axhline(0, color=BORDER_COLOR, linewidth=0.8)
    ax.set_xlabel("Step")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def save_hysteresis_plot(x, f, dy, fy, k2, path):
    qy = fy - k2 * dy
    x_range = [min(x) - 0.25, max(x) + 0.25]

    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.plot(x_range, [qy + k2 * v for v in x_range], color=BORDER_COLOR, linestyle="--", linewidth=1)
    ax.plot(x_range, [-qy + k2 * v for v in x_range], color=BORDER_COLOR, linestyle="--", linewidth=1)
    ax.plot(x, f, color=LINE_COLOR, linewidth=2, marker="o", markersize=5)
    ax.annotate("Top border", (x_range[0], qy + k2 * x_range[0]), xytext=(4, 6),
                textcoords="offset points", color="#555555")
    ax.annotate("Bottom border", (x_range[1], -qy + k2 * x_range[1]), xytext=(-4, -14),
                textcoords="offset points", ha="right", color="#555555")
    ax.axhline(0, color=BORDER_COLOR, linewidth=0.8)
    ax.axvline(0, color=BORDER_COLOR, linewidth=0.8)
    ax.set_xlabel("Displacement")
    ax.set_ylabel("Force")
    ax.set_title("Hysteresis: force vs displacement")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    dy = 1.0
    fy = 5.0
    k2 = 0.25
    x = [0, 0.5, 0.75, 1.25, 1.5, 1.0, 1.75, 0.75, 0.25, 0.5, 0.0, -0.5,
         -0.75, -1.5, -1.0, -1.75, -2.0, -2.5, -2.25, -1.75, -1.0, -0.25, 0, 1.0]

    f, states = run_history(x, dy, fy, k2)

    print(f"kp = {fy / dy:g}, ks = {k2:g}, qy = {fy - k2 * dy:g}")
    print(f"{'Step':>4} {'x':>7} {'F':>10}  State")
    for i, (xi, fi, state) in enumerate(zip(x, f, states)):
        print(f"{i:>4} {xi:>7.2f} {fi:>10.4f}  {state}")

    out = Path(__file__).parent
    save_history_plot(x, "Displacement", "Displacement time-history", out / "displacement_history.png")
    save_history_plot(f, "Force", "Force time-history", out / "force_history.png")
    save_hysteresis_plot(x, f, dy, fy, k2, out / "hysteresis.png")


if __name__ == "__main__":
    main()
