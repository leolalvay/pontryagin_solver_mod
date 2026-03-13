"""
Example 6 (Paper Example 3.2): 1D, nonsmooth Hamiltonian.

Minimize   J = ∫_0^1 x(t)^10 dt
subject to x'(t) = a(t),  a(t) ∈ [-1, 1],  x(0)=0.5,  g(x(1))=0.

Exact solution (paper):
  x*(t) = 0.5 - t   for t ∈ [0, 0.5],  and  0  for t ∈ [0.5, 1]
  a*(t) = -1        for t ∈ [0, 0.5],  and  0  for t ∈ [0.5, 1]
  J*    = 0.5^11 / 11
"""
import json
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from core.problem import OCPProblem
from core.adaptivity import solve_optimal_control
from core.smoothing import eval_H_smooth

def run_example():
    # ============================================================
    # 0) Problem definition (Paper Example 3.2)
    # ============================================================
    x0 = np.array([0.5])   # scalar state (shape (1,))
    T = 1.0

    def dynamics(x, u, t):
        # x' = a
        a = u[0]
        return np.array([a], dtype=float)

    def stage_cost(x, u, t):
        # l(x,u,t) = x^10   (control does not appear)
        y = x[0]
        return float(y**10)

    def terminal_cost(x):
        # g(x(T)) = 0  => terminal BC: p(T)=0 in solver convention
        return 0.0

    u_min = np.array([-1.0])
    u_max = np.array([1.0])

    # Paper regularization parameter (Example 3.2)
    #delta_paper = 1e-6 

    def ham_grad_paper(x: np.ndarray, p: np.ndarray, t: float):
        """
        Paper regularization:
            H^δ(x,p) = x^10 - sqrt(p^2 + δ^2)

        Returns (∂_p H^δ, ∂_x H^δ) as arrays of shape (1,).
        """
        x0 = float(x[0])
        p0 = float(p[0])
        denom = np.sqrt(p0 * p0 + delta_paper * delta_paper)

        grad_p = np.array([-p0 / denom], dtype=float)      # in [-1,1]
        grad_x = np.array([10.0 * (x0 ** 9)], dtype=float) # derivative of x^10
        return grad_p, grad_x

    prob = OCPProblem(
        dynamics=dynamics,
        stage_cost=stage_cost,
        terminal_cost=terminal_cost,
        x0=x0,
        T=T,
        control_bounds=(u_min, u_max),
        state_bounds=None,
        # IMPORTANT: do NOT pass explicit Hamiltonian gradients here
        # (true H = x^10 - |p| is nonsmooth at p=0).
        hamiltonian_true=None,
        u_star_fn=None,
        hamiltonian_grad_fn=None,
    )

    # Flags (match your intended workflow)
    use_oracle_bootstrap = False
    use_oracle_PA = False
    use_explicit_hamiltonian_gradients = False

    # ============================================================
    # 1) Solve with the repo's adaptive outer loop
    # ============================================================
    # Start with a uniform mesh (like ex5 style)
    t_nodes = np.linspace(0.0, T, 20)  # dt=0.025

    t0 = time.perf_counter()
    result = solve_optimal_control(
        prob,
        t_nodes,
        tol_time=1e-5,
        tol_PA=1e-4,
        tol_delta=1e-10,
        max_iters=40,
        delta0=0.02,
        use_oracle_bootstrap=use_oracle_bootstrap,
        use_oracle_PA=use_oracle_PA,
        use_explicit_hamiltonian_gradients=use_explicit_hamiltonian_gradients,
    )
    t1 = time.perf_counter()
    wall_time = t1 - t0
    print(f"[benchmark] wall_time_sec = {wall_time:.3f}")

    # ============================================================
    # 2) Summary prints
    # ============================================================
    print("\nExample 6 (Paper Example 3.2: x' = a, J=∫x^10)")
    print("len(log) =", len(result["log"]))
    print("last outer iter =", result["log"][-1]["iteration"])
    print("len(t_nodes) =", len(result["t_nodes"]))
    print("X.shape =", result["X"].shape)
    print("P.shape =", result["P"].shape)
    print("final delta =", result["delta"])
    print("last log entry =", result["log"][-1])

    # ============================================================
    # 3) Save benchmark JSON (same pattern as ex5)
    # ============================================================
    Path("benchmarks").mkdir(parents=True, exist_ok=True)

    last = result["log"][-1]
    bench = {
        "example": "ex6_paper_example32",
        "wall_time_sec": float(wall_time),
        "max_iters": int(last["iteration"]),
        "N_final": int(last["N"]),
        "M_final": int(last["M"]),
        "delta_final": float(result["delta"]),
        "eta_time": float(last["eta_time"]),
        "eta_PA": float(last["eta_PA"]),
        "eta_delta": float(last["eta_delta"]),
        "len_t_nodes": int(len(result["t_nodes"])),
        "use_oracle_bootstrap": bool(use_oracle_bootstrap),
        "use_oracle_PA": bool(use_oracle_PA),
        "use_explicit_hamiltonian_gradients": bool(use_explicit_hamiltonian_gradients),
    }

    tag_parts = []
    if use_oracle_bootstrap:
        tag_parts.append("oracleBoot")
    if use_oracle_PA:
        tag_parts.append("oraclePA")
    if use_explicit_hamiltonian_gradients:
        tag_parts.append("explicitGrads")
    tag = "_".join(tag_parts) if tag_parts else "baseline"

    out_path = f"benchmarks/ex6_{tag}.json"
    with open(out_path, "w") as f:
        json.dump(bench, f, indent=2)
    print(f"[benchmark] wrote {out_path}")

    # ============================================================
    # 4) Postprocess: exact solution checks
    # ============================================================
    t = np.asarray(result["t_nodes"])
    dt = np.diff(t)

    X = np.asarray(result["X"])[:, 0]
    P = np.asarray(result["P"])[:, 0]
    bundle = result["bundle"]

    # Exact state trajectory
    x0_val = float(prob.x0[0])
    x_exact = np.maximum(x0_val - t, 0.0)
    # Exact costate trajectory
    p_exact = np.where(t <= x0_val, (x0_val - t)**10, 0.0)
    # "Exact" control is defined on intervals: a=-1 for t<0.5 else 0
    a_exact = np.where(t[:-1] < 0.5, -1.0, 0.0)

    # Reconstruct surrogate control (active plane) on each interval
    a_bar = np.zeros_like(t[:-1])
    for i in range(len(t) - 1):
        # match integrator evaluation point: (p_{i+1}, x_i, t_i)
        Hbar_i, idx = bundle.evaluate(prob, np.array([P[i + 1]]), np.array([X[i]]), float(t[i]))
        u_i = bundle.controls[int(idx)]
        a_bar[i] = float(u_i[0])
#=============== Diagnosis ============================
    N = len(t) - 1
    u_delta = np.zeros(N)

    for i in range(N):
        # IMPORTANT: same point used in integrators.py
        _, grad_p, _ = eval_H_smooth(
            prob, bundle,
            np.array([P[i + 1]]),
            np.array([X[i]]),
            float(t[i]),
            float(result["delta"]),
        )
        u_delta[i] = float(grad_p[0])   # since x' = u in 1D
#==============================================================
    #u_paper = -P[1:] / np.sqrt(P[1:]**2 + delta_paper**2)

    # Cost J (left Riemann, consistent with symplectic Euler's x_i usage)
    J_hat = float(np.sum(dt * (X[:-1] ** 10)))
    J_star = float((0.5 ** 11) / 11.0)
    rel_err_J = abs(J_hat - J_star) / max(abs(J_star), 1e-16)
    rho_bar = np.asarray(result["rhobar"], dtype=float)   # length N
    eta_apost_local = (dt ** 2) * rho_bar
    eta_apost = float(np.sum(eta_apost_local))

    print(f"[check] J_hat = {J_hat:.16e}")
    print(f"[check] J_star = {J_star:.16e}")
    print(f"[check] |J_hat - J_star| = {abs(J_hat - J_star):.3e}")
    print(f"[check] rel_err_J = {rel_err_J:.3e}")
    print(f"[apost] estimator sum(dt^2 * rhobar) = {eta_apost:.16e}")


    err_p_inf = float(np.max(np.abs(P - p_exact)))
    print(f"[check] ||P - P*||_inf = {err_p_inf:.3e}")
    err_x_inf = float(np.max(np.abs(X - x_exact)))
    print(f"[check] ||X - X*||_inf = {err_x_inf:.3e}")

    tail = t[:-1] >= 0.5
    print("[diag] min P on [0.5,1]:", float(np.min(P[t >= 0.5])))
    print("[diag] min u_delta on [0.5,1):", float(np.min(u_delta[tail])))
    print("[diag] max u_delta on [0.5,1):", float(np.max(u_delta[tail])))
    # ============================================================
    # 5) Plots (same style as ex5)
    # ============================================================
    # (a) time mesh dt(t)
    fig_dt = plt.figure()
    plt.step(t[:-1], dt, where="post")
    plt.yscale("log")
    plt.xlabel("t")
    plt.ylabel("Δt")
    plt.title("Example 3.2: Time mesh Δt(t)")
    plt.grid(True, which="both")
    plt.savefig("example32_tvsdt.pdf", format="pdf", bbox_inches="tight")

    # (b) state
    fig_x = plt.figure()
    plt.plot(t, X, label="X (solver)")
    plt.plot(t, x_exact, "--", label="X* (exact)")
    plt.xlabel("t")
    plt.ylabel("X")
    plt.title("Example 3.2: State X(t)")
    plt.legend()
    plt.grid(True)
    plt.savefig("example32_state_X.pdf", format="pdf", bbox_inches="tight")

    # (c) costate
    fig_p = plt.figure()
    plt.plot(t, P, label="P (solver)")
    plt.plot(t, p_exact, "--", label="P* (exact)")
    plt.xlabel("t")
    plt.ylabel("P")
    plt.title("Example 3.2: Costate P(t)")
    plt.legend()
    plt.grid(True)
    plt.savefig("example32_costate_P.pdf", format="pdf", bbox_inches="tight")

    # (d) control (active plane)
    fig_u = plt.figure()
    plt.step(t[:-1], a_bar, where="post", label=r"$\bar{a}$ (active plane of $\bar H$)")
    plt.step(t[:-1], u_delta, where="post", label=r"$u_\delta=\partial_p H_\delta$ (used in dynamics)")
    # Paper-regularized control actually used in explicit-gradient mode:
    #plt.step(t[:-1], u_paper, where="post", label=r"$u_{\mathrm{paper}}=\partial_p H^\delta$")
    plt.step(t[:-1], a_exact, where="post", linestyle="--", label=r"$a^*$ (exact)")
    plt.xlabel("t")
    plt.ylabel("a")
    plt.title("Example 3.2: Control comparison")
    plt.legend()
    plt.grid(True)
    plt.savefig("example32_control_compare.pdf", format="pdf", bbox_inches="tight")
    plt.close(fig_u)
    # (e) rho_bar and r_bar (from adaptivity)
    #rho_bar = np.asarray(result["rhobar"])  # length N
    r_bar = np.asarray(result["rbar"])      # length N

    fig_rho = plt.figure()
    plt.step(t[:-1], np.abs(rho_bar), where="post", label=r"$|\bar{\rho}_n|$")
    plt.yscale("log")
    plt.xlabel("t")
    plt.ylabel(r"$|\bar{\rho}|$")
    plt.title(r"Example 3.2: density-like term $\bar{\rho}_n$")
    plt.grid(True, which="both")
    plt.legend()
    plt.savefig("example32_rho_bar.pdf", format="pdf", bbox_inches="tight")

    fig_r = plt.figure()
    plt.step(t[:-1], r_bar, where="post", label=r"$\bar{r}_n$")
    plt.yscale("log")
    plt.xlabel("t")
    plt.ylabel(r"$\bar{r}$")
    plt.title(r"Example 3.2: time error indicator $\bar{r}_n = |\bar{\rho}_n|\,\Delta t_n^2$")
    plt.grid(True, which="both")
    plt.legend()
    plt.savefig("example32_r_bar.pdf", format="pdf", bbox_inches="tight")

    # (d) control (paper regularization, explicit mode)
    '''
    print("[diag] len(t[:-1]) =", len(t[:-1]), "len(u_paper) =", len(u_paper))
    print("[diag] u_paper min/max =", float(np.min(u_paper)), float(np.max(u_paper)))
    print("[diag] max |u_paper - a_exact| =", float(np.max(np.abs(u_paper - a_exact))))

    fig_u = plt.figure()
    plt.step(t[:-1], u_paper, where="post", linewidth=2.5,
            label=r"$u_{\mathrm{paper}}=-p/\sqrt{p^2+\delta^2}$")
    plt.step(t[:-1], a_exact, where="post", linestyle="--", linewidth=2.0,
            label=r"$a^*$ (exact)")

    plt.xlabel("t")
    plt.ylabel("a")
    plt.title("Example 3.2: Control (paper regularization)")
    plt.legend()
    plt.grid(True)

    out_u = Path("example32_control_paper_compare.pdf").resolve()
    print(f"[plot] wrote {out_u}")
    plt.savefig(out_u, format="pdf", bbox_inches="tight")
    plt.close(fig_u)

    # Close figs
    plt.close(fig_dt)
    plt.close(fig_x)
    plt.close(fig_p)
    plt.close(fig_u)
    plt.close(fig_rho)
    plt.close(fig_r)
    '''
    return result


if __name__ == "__main__":
    run_example()