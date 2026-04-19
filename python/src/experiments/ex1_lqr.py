"""
Example 1: Linear Quadratic Regulator (LQR).

This script sets up a 2D double-integrator-like system with quadratic running
and terminal costs.  It solves the optimal control problem using the adaptive
Pontryagin solver and prints basic diagnostics.  The setup follows
the problem description in the DeepResearch plan.
"""
import numpy as np
import matplotlib.pyplot as plt


# The experiments package is a sibling of the core package.  When this script
# is executed with PYTHONPATH pointing at the `python/src` directory, we can
# import from the `core` package directly.  Using absolute imports here avoids
# issues with relative imports beyond the top‑level package.
from core.problem import OCPProblem
from core.pa_bundle import PABundle
from core.adaptivity import solve_optimal_control
from core.hamiltonian import compute_H


def run_example():
    # dynamics matrices
    A = np.array([[0.0, 1.0], [0.0, 0.0]])
    B = np.array([[0.0], [1.0]])
    # cost matrices
    Q = np.eye(2)
    R = 1e-2 * np.eye(1)
    Qf = Q
    # initial state and horizon
    x0 = np.array([1.0, 0.0])
    T = 1.0
    # define dynamics and costs
    def dynamics(x, u, t):
        return A.dot(x) + B.dot(u)
    def stage_cost(x, u, t):
        return float(x.dot(Q.dot(x)) + u.T.dot(R).dot(u))
    def terminal_cost(x):
        return float(x.dot(Qf.dot(x)))
    # control bounds (approximate unconstrained by large bounds)
    u_min = np.array([-11.0])
    u_max = np.array([5.0]) 
    # create problem
    prob = OCPProblem(dynamics, stage_cost, terminal_cost, x0, T,
                      control_bounds=(u_min, u_max), state_bounds=None)
    # initial mesh: uniform with 20 segments
    t_nodes = np.linspace(0.0, T, 21)
    # solve adaptively
    result = solve_optimal_control(prob, t_nodes, tol_time=1e-2, tol_PA=1e-2, tol_delta=1e-2, max_iters=10, delta0=0.1)
    # extract solution
    print("len(log) =", len(result["log"]))
    print("last outer iter =", result["log"][-1]["iteration"])
    print("len(t_nodes) =", len(result["t_nodes"]))
    print("X.shape =", result["X"].shape)
    print("P.shape =", result["P"].shape)

    X = result['X']
    P = result['P']
    mesh = result['t_nodes']
    bundle = result['bundle']
    # approximate control at nodes by computing H argmin
    controls = []
   
    for i in range(len(mesh)):
        _, u_star = compute_H(prob, P[i], X[i], mesh[i], bundle.controls, restricted=True)
        controls.append(u_star)
    controls = np.asarray(controls)
    # approximate objective
    obj = prob.g(X[-1])
    for i in range(len(mesh) - 1):
        dt = mesh[i + 1] - mesh[i]
        u_i = controls[i]
        obj += prob.l(X[i], u_i, mesh[i]) * dt
    # print diagnostics
    print("LQR Example")
    print(f"Mesh points: {len(mesh)}")
    print(f"Planes: {bundle.num_planes()}")
    print(f"Objective value: {obj}")
    print("Indicator history:")
    for entry in result['log']:
        print(entry)

    t = np.array(result["t_nodes"])
    X = np.asarray(result["X"])
    P = np.asarray(result["P"])

    # X(t)
    plt.figure()
    plt.plot(t, X[:, 0], label="x1")
    plt.plot(t, X[:, 1], label="x2")
    plt.xlabel("t")
    plt.title("State trajectory X(t)")
    plt.legend()
    plt.tight_layout()

    #plt.savefig("ex1_X.png", dpi=200)

    # P(t)
    plt.figure()
    plt.plot(t, P[:, 0], label="p1")
    plt.plot(t, P[:, 1], label="p2")
    plt.xlabel("t")
    plt.title("Costate trajectory P(t)")
    plt.legend()
    plt.tight_layout()
    #plt.savefig("ex1_P.png", dpi=200)


    plt.show()
    
    return result


if __name__ == '__main__':
    run_example()