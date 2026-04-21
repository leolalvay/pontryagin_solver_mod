"""
Example 1: Linear Quadratic Regulator (LQR).

This script sets up a 2D double-integrator-like system with quadratic running
and terminal costs.  It solves the optimal control problem using the adaptive
Pontryagin solver and prints basic diagnostics.  The setup follows
the problem description in the DeepResearch plan.
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from functools import partial

# The experiments package is a sibling of the core package.  When this script
# is executed with PYTHONPATH pointing at the `python/src` directory, we can
# import from the `core` package directly.  Using absolute imports here avoids
# issues with relative imports beyond the top‑level package.
from core.problem import OCPProblem
from core.pa_bundle import PABundle
from core.adaptivity import solve_optimal_control
from core.hamiltonian import compute_H

def run_ex1_lqr_solver(
    n_init=20,
    tol_time=1e-3,
    tol_PA=1e-2,
    tol_delta=1e-2,
    max_iters=15,
    delta0=0.15,
    u_min=-11.0,
    u_max=5.0,
):
    """
    Build and solve Example 1 LQR with the adaptive Pontryagin solver.

    Returns
    -------
    result : dict
        Raw output from solve_optimal_control(...).
    prob : OCPProblem
        Problem instance (useful for post-processing).
    """
    A = np.array([[0.0, 1.0], [0.0, 0.0]])
    B = np.array([[0.0], [1.0]])
    Q = np.eye(2)
    R = 1e-2 * np.eye(1)
    Qf = Q

    x0 = np.array([1.0, 0.0])
    T = 1.0

    def dynamics(x, u, t):
        return A @ x + B @ u

    def stage_cost(x, u, t):
        return float(x @ Q @ x + u.T @ R @ u)

    def terminal_cost(x):
        return float(x @ Qf @ x)

    prob = OCPProblem(
        dynamics=dynamics,
        stage_cost=stage_cost,
        terminal_cost=terminal_cost,
        x0=x0,
        T=T,
        control_bounds=(np.array([u_min]), np.array([u_max])),
        state_bounds=None,
    )

    t_nodes = np.linspace(0.0, T, n_init + 1)

    result = solve_optimal_control(
        prob,
        t_nodes,
        tol_time=tol_time,
        tol_PA=tol_PA,
        tol_delta=tol_delta,
        max_iters=max_iters,
        delta0=delta0,
    )

    return result, prob
def summarize_ex1_results(result, prob, print_last_log_only=True):
    """
    Print compact diagnostics for Example 1 solver output.
    """
    log = result["log"]
    last = log[-1]

    t_nodes = np.asarray(result["t_nodes"])
    X = np.asarray(result["X"])
    P = np.asarray(result["P"])
    bundle = result["bundle"]

    # Reconstruct node controls from Hamiltonian minimization
    controls = []
    for i in range(len(t_nodes)):
        _, u_star = compute_H(prob, P[i], X[i], t_nodes[i], bundle.controls, restricted=True)
        controls.append(np.asarray(u_star, dtype=float))
    controls = np.asarray(controls)

    # Simple objective approximation on mesh
    obj = prob.g(X[-1])
    for i in range(len(t_nodes) - 1):
        dt = t_nodes[i + 1] - t_nodes[i]
        obj += prob.l(X[i], controls[i], t_nodes[i]) * dt

    print("=== Example 1 (LQR) ===")
    print(f"outer iterations logged: {len(log)}")
    print(f"last outer iteration:    {last.get('iteration')}")
    print(f"mesh points:             {len(t_nodes)}")
    print(f"state shape:             {X.shape}")
    print(f"costate shape:           {P.shape}")
    print(f"planes:                  {bundle.num_planes()}")
    print(f"objective (mesh approx): {obj:.12e}")

    # Key adaptivity metrics (if present)
    for k in ["eta_time", "eta_PA", "eta_delta", "delta", "newton_iter", "newton_residual", "tol_time_star", "mark_thr", "note"]:
        if k in last:
            print(f"{k:24s}: {last[k]}")

    # Optional: only print last log entry, not full history
    if not print_last_log_only:
        print("\nIndicator history (compact):")
        for e in log:
            msg = (
                f"it={e.get('iteration')} "
                f"N={e.get('N')} M={e.get('M')} "
                f"eta_time={e.get('eta_time', float('nan')):.2e} "
                f"eta_PA={e.get('eta_PA', float('nan')):.2e} "
                f"eta_delta={e.get('eta_delta', float('nan')):.2e} "
                f"newton_it={e.get('newton_iter')} "
                f"res={e.get('newton_residual', float('nan')):.2e}"
            )
            if "note" in e:
                msg += f" note={e['note']}"
            print(msg)


def save_plot(fig, stem, fig_dir, ext="pdf"):
    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_dir / f"{stem}.{ext}", bbox_inches="tight")
    plt.close(fig)

def keep_plot(fig, stem=None):
    pass

def plot_ex1_results(
    result,
    prob,
    out_prefix="example1_solver",
    save_plots=False,
    plot_ext="pdf",
    fig_dir=None,
):
    t = np.asarray(result["t_nodes"])
    X = np.asarray(result["X"])
    P = np.asarray(result["P"])
    bundle = result["bundle"]
    log = result.get("log", [])
    last = log[-1] if log else {}

    # for density/rbar plots: use the last entry that actually has arrays
    last_with_indicators = {}
    for e in reversed(log):
        if ("rho" in e) and ("rho_bar" in e) and ("r_bar" in e):
            last_with_indicators = e
            break

    if fig_dir is None:
        fig_dir = Path(__file__).resolve().parent / "figures"

    plot_action = partial(save_plot, fig_dir=fig_dir, ext=plot_ext) if save_plots else keep_plot
    render_plots = (lambda: None) if save_plots else plt.show

    # --- reconstruct control at nodes ---
    controls = []
    for i in range(len(t)):
        _, u_star = compute_H(prob, P[i], X[i], t[i], bundle.controls, restricted=True)
        controls.append(np.asarray(u_star, dtype=float).reshape(-1))
    U = np.vstack(controls)  # shape (len(t), m_u)

    # 1) state
    fig = plt.figure(figsize=(8, 5))
    plt.plot(t, X[:, 0], label="x1(t)")
    plt.plot(t, X[:, 1], label="x2(t)")
    plt.xlabel("t")
    plt.ylabel("state")
    plt.title("State trajectory")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plot_action(fig, f"{out_prefix}_state_x")

    # 2) costate
    fig = plt.figure(figsize=(8, 5))
    plt.plot(t, P[:, 0], label="p1(t)")
    plt.plot(t, P[:, 1], label="p2(t)")
    plt.xlabel("t")
    plt.ylabel("costate")
    plt.title("Costate trajectory")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plot_action(fig, f"{out_prefix}_costate_p")

    # 3) control
    fig = plt.figure(figsize=(8, 5))
    plt.plot(t, U[:, 0], label="u(t)")
    plt.xlabel("t")
    plt.ylabel("control")
    plt.title("Control trajectory")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plot_action(fig, f"{out_prefix}_control_u")

    # 4) dt vs t
    if len(t) > 1:
        dt = np.diff(t)
        fig = plt.figure(figsize=(8, 5))
        plt.step(t[:-1], dt, where="post", label=r"$\Delta t_n$")
        plt.yscale("log")
        plt.xlabel("t")
        plt.ylabel(r"$\Delta t$")
        plt.title("Time mesh step sizes")
        plt.grid(True, which="both")
        plt.legend()
        plt.tight_layout()
        plot_action(fig, f"{out_prefix}_t_vs_dt")

    # 5) rho density (from adaptivity log)
    if "rho" in last_with_indicators and "rho_bar" in last_with_indicators:
        rho = np.asarray(last_with_indicators["rho"], dtype=float)
        rho_bar = np.asarray(last_with_indicators["rho_bar"], dtype=float)
        t_int = t[:-1][:len(rho)]  # safe alignment
        fig = plt.figure(figsize=(8, 5))
        plt.step(t_int, rho, where="post", label=r"$\rho_n$")
        plt.step(t_int, rho_bar, where="post", label=r"$\bar{\rho}_n$")
        plt.xlabel("t")
        plt.ylabel(r"$\rho$")
        plt.title(r"Error density: $\rho_n,\ \bar{\rho}_n$")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plot_action(fig, f"{out_prefix}_rho_density")

    # 6) r indicator (from adaptivity log)
    if "r_bar" in last_with_indicators:
        r_bar = np.asarray(last_with_indicators["r_bar"], dtype=float)
        t_int = t[:-1][:len(r_bar)]  # safe alignment

        fig = plt.figure(figsize=(8, 5))
        plt.step(t_int, r_bar, where="post", label=r"$\bar r_n = |\bar\rho_n|\Delta t_n^2$")
        plt.yscale("log")
        plt.xlabel("t")
        plt.ylabel(r"$\bar r$")
        plt.title(r"Time indicator: $\bar r_n$")
        plt.grid(True, which="both")
        plt.legend()
        plt.tight_layout()
        plot_action(fig, f"{out_prefix}_r_indicator")

    render_plots()

def run_example():
    result, prob = run_ex1_lqr_solver()
    summarize_ex1_results(result, prob, print_last_log_only=True)
    plot_ex1_results(
        result,
        prob,
        out_prefix="example1_solver",
        save_plots=False,   # True -> save files
        plot_ext="pdf",
    )
    return result

if __name__ == '__main__':
    run_example()