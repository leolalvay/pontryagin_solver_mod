"""
Example 5: Hypersensitive optimal control.

Minimize   ∫_0^25 (x(t)^2+alpha(t)^2)dt + gamma(x(25)-1)^2
subject to x'(t) = -x(t)^3 + alpha(t),  x(0)=1,  g(x(25))=gamma*(x(25)-1)^2.
"""
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import splu
from scipy.sparse import issparse
from core.problem import OCPProblem
from core.adaptivity import solve_optimal_control
from core.integrators import pack_unknowns
from core.shooting import shooting_residual, shooting_jacobian


def run_example():
    # ============================================================
    # 0) Problem definition (matches PDF Example 7)
    # ============================================================
    x0 = np.array([1.0])   # scalar state, stored as shape (1,)
    T = 25.0
    gamma = 1e6

    def dynamics(x, u, t):
        y = x[0]; a = u[0]
        return np.array([-(y**3) + a])

    def stage_cost(x, u, t):
        y = x[0]; a = u[0]
        return float(y**2 + a**2)

    def terminal_cost(x):
        yT = x[0]
        return float(gamma * ((yT - 1.0)**2))

    u_min = np.array([-1.0])
    u_max = np.array([3.0])




    prob = OCPProblem(
        dynamics=dynamics,
        stage_cost=stage_cost,
        terminal_cost=terminal_cost,
        x0=x0,
        T=T,
        control_bounds=(u_min, u_max),
        state_bounds=None,
    )

    # ============================================================
    # 1) Solve with the repo's adaptive outer loop
    # ============================================================
    t_nodes = np.linspace(0.0, T, 30)  # dt = 0.02 (fast initial mesh)

    result = solve_optimal_control(
        prob,
        t_nodes,
        tol_time=1e-2,   # relaxed -> fewer refinements
        tol_PA=1e-2,
        tol_delta=1e-2,
        max_iters=15,
        delta0=0.02,
    )

    print("\nExample 5 (Hypersensitive optimal control)")
    print("len(log) =", len(result["log"]))
    print("last outer iter =", result["log"][-1]["iteration"])
    print("len(t_nodes) =", len(result["t_nodes"]))
    print("X.shape =", result["X"].shape)
    print("P.shape =", result["P"].shape)
    print("final delta =", result["delta"])
    print("last log entry =", result["log"][-1])

    # ============================================================
    #    
    #     
    #      
    # ============================================================
    t = np.asarray(result["t_nodes"])
    dt = np.diff(t)

    fig1 = plt.figure()
    #plt.figure()
    plt.step(t[:-1], dt, where="post")   # Δt constant in [t_n, t_{n+1})
    plt.yscale("log")                   #log scale in y
    plt.xlabel("t")
    plt.ylabel("Δt")
    plt.title("Time mesh: Δt(t) (step plot)")
    plt.grid(True, which="both")
    plt.savefig("example5_tvsdt.pdf", format="pdf", bbox_inches="tight")
    #plt.show()

    X = np.asarray(result["X"])[:, 0]
    P = np.asarray(result["P"])[:, 0]
    x0_scalar = float(x0[0])


    # Plot X and P (two windows at once)
    fig1 = plt.figure()
    plt.plot(t, X, label="X (solver)")
    plt.xlabel("t")
    plt.ylabel("X")
    plt.title("Example 5: State X(t)")
    plt.legend()
    plt.grid(True)
    plt.savefig("example5_state_X.pdf", format="pdf", bbox_inches="tight")


    fig2 = plt.figure()
    plt.plot(t, P, label="P (solver)")
    plt.xlabel("t")
    plt.ylabel("P")
    plt.title("Example 5: Costate P(t)")
    plt.legend()
    plt.grid(True)
    plt.savefig("example5_costate_P.pdf", format="pdf", bbox_inches="tight")

    # --- arrays ---
    rho_bar = np.asarray(result["rhobar"])   # length N
    r_bar   = np.asarray(result["rbar"])     # length N
    t_mid   = 0.5*(t[:-1] + t[1:])           # length N

    # Plot rho_bar
    fig3 = plt.figure()
    plt.step(t[:-1], np.abs(rho_bar), where="post", label=r"$|\bar{\rho}_n|$")
    plt.yscale("log")
    plt.xlabel("t")
    plt.ylabel(r"$|\bar{\rho}|$")
    plt.title(r"Example 5: density-like term $\bar{\rho}_n$")   # <-- TEXTO PRIMERO
    plt.grid(True)
    plt.legend()
    plt.savefig("example5_rho_bar.pdf", format="pdf", bbox_inches="tight")


    # Plot r_bar
    fig4 = plt.figure()
    plt.step(t[:-1], r_bar, where="post", label=r"$\bar{r}_n$")
    plt.yscale("log")
    plt.xlabel("t")
    plt.ylabel(r"$\bar{r}$")
    plt.title(r"Example 5: time error indicator $\bar{r}_n = |\bar{\rho}_n|\,\Delta t_n^2$")  # <-- TEXTO PRIMERO
    plt.grid(True, which="both")
    plt.legend()
    plt.savefig("example5_r_bar.pdf", format="pdf", bbox_inches="tight")
   

    # Show all figures at once
    #plt.show()

    # Close figs (optional)
    plt.close(fig1)
    plt.close(fig2)
    plt.close(fig3)
    plt.close(fig4)
    

    return result


if __name__ == "__main__":
    run_example()
