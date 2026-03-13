import numpy as np
from scipy.optimize import root
import matplotlib.pyplot as plt

def init_ex6(
    T=1.0,
    x0=0.5,
    delta=1e-6,
    N0=20,
    tol_time=1e-5,
    max_refine=20,
):
    """
    Initialize Example 3.2 test data in a simple, self-contained way.

    Returns
    -------
    params : dict
        Scalar settings and initial time mesh.
    model : dict
        Problem functions: dynamics, costs, H_delta and derivatives.
    ref : dict
        Exact/reference functions and J_star for diagnostics.
    """
    if T <= 0:
        raise ValueError("T must be > 0")
    if delta <= 0:
        raise ValueError("delta must be > 0")
    if N0 < 1:
        raise ValueError("N0 must be >= 1")

    t = np.linspace(0.0, T, N0 + 1)

    # Model (paper-smoothed Hamiltonian)
    def f(x, a, t_):
        return a  # x' = a

    def L(x, a, t_):
        return x**10

    def g(xT):
        return 0.0

    def H_delta(x, lam):
        return x**10 - np.sqrt(lam**2 + delta**2)

    def dH_dlam(x, lam):
        return -lam / np.sqrt(lam**2 + delta**2)

    def dH_dx(x, lam):
        return 10.0 * x**9

    # Exact/reference
    def x_star(tt):
        return np.maximum(x0 - tt, 0.0)

    def a_star(tt):
        return np.where(tt < x0, -1.0, 0.0)

    def p_star(tt):
        return np.where(tt <= x0, (x0 - tt)**10, 0.0)

    params = {
        "T": T,
        "x0": x0,
        "delta": delta,
        "N0": N0,
        "tol_time": tol_time,
        "max_refine": max_refine,
        "t": t,
    }

    model = {
        "f": f,
        "L": L,
        "g": g,
        "H_delta": H_delta,
        "dH_dlam": dH_dlam,
        "dH_dx": dH_dx,
    }

    ref = {
        "x_star": x_star,
        "a_star": a_star,
        "p_star": p_star,
        "J_star": x0**11 / 11.0,
    }

    return params, model, ref

def pack_z(x, lam):
    """
    x, lam: arrays of length N+1
    Returns z = [x_1,...,x_N, lam_0,...,lam_N]
    """
    return np.concatenate([x[1:], lam])


def unpack_z(z, x0):
    """
    Inverse of pack_z for 1D case.
    x0 is fixed boundary value x(0).
    """
    total = z.size               # = 2N + 1
    N = (total - 1) // 2

    x = np.empty(N + 1)
    lam = np.empty(N + 1)

    x[0] = x0
    x[1:] = z[:N]
    lam[:] = z[N:]

    return x, lam

import numpy as np

def residual_symplectic_euler(z, params, model):
    """
    Residual F(z) for 1D symplectic-Euler discretization of the smoothed PMP system.

    Unknown ordering:
        z = [x_1, ..., x_N, lam_0, ..., lam_N]   (length 2N+1)

    Residual ordering:
        F = [r_x^0, r_lam^0, r_x^1, r_lam^1, ..., r_x^{N-1}, r_lam^{N-1}, r_bc]
          where:
            r_x^i   = x_{i+1} - x_i - dt_i * dH_dlam(x_i, lam_{i+1})
            r_lam^i = lam_i - lam_{i+1} - dt_i * dH_dx(x_i, lam_{i+1})
            r_bc    = lam_N + g_x(x_N)

    For ex6, g(x_T)=0 => g_x(x_N)=0, so r_bc = lam_N.
    """
    t = params["t"]
    x0 = params["x0"]

    # unpack_z should be your simple helper already added
    x, lam = unpack_z(z, x0)

    N = len(t) - 1
    dt = np.diff(t)

    r = np.zeros(2 * N + 1, dtype=float)

    dH_dlam = model["dH_dlam"]   # analytic: -lam/sqrt(lam^2+delta^2)
    dH_dx = model["dH_dx"]       # analytic: 10*x^9

    for i in range(N):
        gp = dH_dlam(x[i], lam[i + 1])  # ∂H/∂λ at (x_i, λ_{i+1})
        gx = dH_dx(x[i], lam[i + 1])    # ∂H/∂x at (x_i, λ_{i+1})

        r[2 * i] = x[i + 1] - x[i] - dt[i] * gp
        r[2 * i + 1] = lam[i] - lam[i + 1] - dt[i] * gx

    # terminal condition λ_N + g_x(x_N)=0 (here g_x=0)
    r[-1] = lam[-1]

    return r

def initial_guess(params):
    """
    Build a simple initial guess for z = [x1..xN, lam0..lamN].
    """
    t = params["t"]
    x0 = params["x0"]
    N = len(t) - 1

    # Linear state guess from x0 to 0
    x_guess = np.linspace(x0, 0.0, N + 1)

    # Zero costate guess
    lam_guess = np.zeros(N + 1)

    return pack_z(x_guess, lam_guess)


def solve_on_mesh(params, model, z0=None, method="hybr", tol=1e-10, maxfev=20000):
    """
    Solve F(z)=0 for one fixed mesh using scipy.optimize.root.
    """
    if z0 is None:
        z0 = initial_guess(params)

    sol = root(
        fun=residual_symplectic_euler,
        x0=z0,
        args=(params, model),
        method=method,           # "hybr" is a good default
        tol=tol,
        options={"maxfev": maxfev},
    )

    # Unpack solution
    x, lam = unpack_z(sol.x, params["x0"])
    t = params["t"]
    N = len(t) - 1

    # Control from symplectic point (x_i, lam_{i+1})
    a = np.zeros(N)
    for i in range(N):
        a[i] = model["dH_dlam"](x[i], lam[i + 1])

    # Cost on mesh (left-point rule, consistent with your setup)
    dt = np.diff(t)
    J = np.sum(dt * (x[:-1] ** 10))

    out = {
        "success": bool(sol.success),
        "message": sol.message,
        "nfev": sol.nfev,
        "res_norm_inf": np.linalg.norm(sol.fun, ord=np.inf),
        "z": sol.x,
        "x": x,
        "lam": lam,
        "a": a,
        "J": float(J),
        "solver_obj": sol,
    }
    return out

def jacobian_symplectic_euler(z, params, model):

    t = params["t"]
    x0 = params["x0"]
    delta = params["delta"]

    x, lam = unpack_z(z, x0)
    N = len(t) - 1
    dt = np.diff(t)

    m = 2 * N + 1
    J = np.zeros((m, m), dtype=float)

    # helper indices in z
    def ix(k):          # x_k, k=1..N
        return k - 1

    def il(j):          # lam_j, j=0..N
        return N + j

    for i in range(N):
        rx = 2 * i
        rl = 2 * i + 1

        xi = x[i]
        lip1 = lam[i + 1]
        den = (lip1 * lip1 + delta * delta)

        # second derivatives
        d2H_dlam2 = -(delta * delta) / (den ** 1.5)   # < 0
        d2H_dx2 = 90.0 * (xi ** 8)

        # r_x^i
        if i >= 1:
            J[rx, ix(i)] += -1.0
        J[rx, ix(i + 1)] += +1.0
        J[rx, il(i + 1)] += -dt[i] * d2H_dlam2

        # r_lam^i
        if i >= 1:
            J[rl, ix(i)] += -dt[i] * d2H_dx2
        # i=0 depends on x0 fixed => no column
        J[rl, il(i)] += +1.0
        J[rl, il(i + 1)] += -1.0

    # boundary r_bc = lam_N
    J[-1, il(N)] = 1.0
    return J

def solve_on_mesh(params, model, z0=None, tol=1e-10, maxfev=20000):
    # Initial guess
    if z0 is None:
        t = params["t"]
        x0 = params["x0"]
        N = len(t) - 1

        x_guess = np.linspace(x0, 0.0, N + 1)
        lam_guess = np.zeros(N + 1)
        z0 = pack_z(x_guess, lam_guess)

    sol = root(
        fun=residual_symplectic_euler,
        x0=z0,
        args=(params, model),
        jac=jacobian_symplectic_euler,   # exact Jacobian
        method="hybr",
        tol=tol,
        options={"maxfev": maxfev},
    )

    # unpack + postprocess
    x, lam = unpack_z(sol.x, params["x0"])
    t = params["t"]
    N = len(t) - 1
    a = np.array([model["dH_dlam"](x[i], lam[i + 1]) for i in range(N)])
    J = float(np.sum(np.diff(t) * (x[:-1] ** 10)))

    return {
        "success": bool(sol.success),
        "message": sol.message,
        "nfev": int(sol.nfev),
        "njev": int(getattr(sol, "njev", -1)),
        "res_inf": float(np.linalg.norm(sol.fun, ord=np.inf)),
        "x": x,
        "lam": lam,
        "a": a,
        "J": J,
        "z": sol.x,
        "solver_obj": sol,
    }

def compute_time_indicator(params, model, x, lam, K_time=1.0):
    """
    Article-aligned indicator:
      rho_i      = -0.5 * H_lambda * H_x
      rho_bar_i  = sign(rho_i) * max(|rho_i|, K_time*sqrt(dt_max))
      r_bar_i    = |rho_bar_i| * dt_i^2
    """
    t = np.asarray(params["t"], dtype=float)
    x = np.asarray(x, dtype=float)
    lam = np.asarray(lam, dtype=float)

    N = t.size - 1
    if N <= 0:
        return {
            "dt": np.array([]),
            "rho": np.array([]),
            "rho_bar": np.array([]),
            "r_bar": np.array([]),
            "eta_time_max": 0.0,
            "eta_time_sum": 0.0,
            "tol_star": float(params["tol_time"]),
            "mark_thr": 0.0,
            "floor": 0.0,
        }

    dt = np.diff(t)
    dt_max = float(np.max(dt))
    floor = float(K_time * np.sqrt(dt_max))

    rho = np.zeros(N, dtype=float)
    rho_bar = np.zeros(N, dtype=float)
    r_bar = np.zeros(N, dtype=float)

    dH_dlam = model["dH_dlam"]
    dH_dx = model["dH_dx"]

    for i in range(N):
        Hp = float(dH_dlam(x[i], lam[i + 1]))  # symplectic point
        Hx = float(dH_dx(x[i], lam[i + 1]))
        rho_i = -0.5 * Hp * Hx
        rho[i] = rho_i

        # article form: signed rho_bar
        mag = max(abs(rho_i), floor)
        rho_bar_i = np.sign(rho_i) * mag
        rho_bar[i] = rho_bar_i

        r_bar[i] = abs(rho_bar_i) * (dt[i] ** 2)

    tol_star = float(params["tol_time"] / N)          # TOL / N
    mark_thr = float(params["s_mark"] * params["tol_time"] / N)  # s*TOL/N

    return {
        "dt": dt,
        "rho": rho,
        "rho_bar": rho_bar,
        "r_bar": r_bar,
        "eta_time_max": float(np.max(r_bar)),
        "eta_time_sum": float(np.sum(r_bar)),
        "tol_star": tol_star,
        "mark_thr": mark_thr,
        "floor": floor,
    }

def refine_mesh_article(t_old, r_bar, tol_time, s_mark=0.8, M_sub=2):
    """
    Subdivide interval (t_i,t_{i+1}) into M_sub parts if r_bar[i] > s_mark*TOL/N.
    """
    t_old = np.asarray(t_old, dtype=float)
    N = len(t_old) - 1
    if N <= 0:
        return t_old.copy(), np.array([], dtype=bool)

    thr = float(s_mark * tol_time / N)
    marked = (np.asarray(r_bar) > thr)

    new_nodes = [t_old[0]]
    for i in range(N):
        a, b = t_old[i], t_old[i + 1]
        if marked[i]:
            # insert M_sub equal subintervals
            mids = np.linspace(a, b, M_sub + 1)[1:]  # exclude left endpoint
            new_nodes.extend(mids.tolist())
        else:
            new_nodes.append(b)

    t_new = np.asarray(new_nodes, dtype=float)
    return t_new, marked

def prolongate_guess(t_old, x_old, lam_old, t_new):
    """
    Linear interpolation warm-start on refined mesh.
    """
    x_new = np.interp(t_new, t_old, x_old)
    lam_new = np.interp(t_new, t_old, lam_old)
    return x_new, lam_new

import numpy as np

def run_adaptivity_test(params, model, ref=None, verbose=True, maxit=None):
    """
    Article-style adaptivity loop (time refinement only) with robust while-flow.

    Key behavior:
    - `iters_used` counts solved+evaluated configurations.
    - If an update is applied at budget boundary, one extra solve is allowed
      to resolve that updated configuration (`pending_update` logic).
    """

    # ---- config ----
    tol_time = float(params["tol_time"])
    s_mark = float(params["s_mark"])
    M_sub = int(params["M_sub"])
    K_time = float(params["K_time"])

    if maxit is None:
        maxit = int(params.get("max_refine", 20))
    else:
        maxit = int(maxit)

    # ---- state ----
    t = np.asarray(params["t"], dtype=float).copy()
    z0 = None
    log = []

    iters_used = 0
    pending_update = False
    converged = False
    stop_reason = "unknown"

    while True:
        # Stop only if budget exhausted and there is no unresolved update
        if (iters_used >= maxit) and (not pending_update):
            stop_reason = "maxit_reached"
            break

        # Build per-iteration params with current mesh
        p = dict(params)
        p["t"] = t

        # 1) Solve current configuration
        sol = solve_on_mesh(p, model, z0=z0)
        if not sol["success"]:
            stop_reason = "nonlinear_solve_failed"
            log.append({
                "iter": iters_used,
                "N": len(t) - 1,
                "success": False,
                "message": sol["message"],
                "res_inf": sol["res_inf"],
            })
            break

        x = sol["x"]
        lam = sol["lam"]

        # 2) Indicators on solved configuration
        ind = compute_time_indicator(p, model, x, lam, K_time=K_time)

        N = len(t) - 1
        eta_time = float(ind["eta_time_max"])
        tol_star = float(ind["tol_star"])
        mark_thr = float(ind["mark_thr"])
        n_marked = int(np.sum(ind["r_bar"] > mark_thr)) if N > 0 else 0

        entry = {
            "iter": iters_used,
            "N": N,
            "success": True,
            "res_inf": float(sol["res_inf"]),
            "nfev": int(sol["nfev"]),
            "njev": int(sol["njev"]),
            "J": float(sol["J"]),
            "eta_time_max": eta_time,
            "eta_time_sum": float(ind["eta_time_sum"]),
            "tol_star": tol_star,
            "mark_thr": mark_thr,
            "n_marked": n_marked,
            "dt_min": float(np.min(ind["dt"])) if N > 0 else 0.0,
            "dt_max": float(np.max(ind["dt"])) if N > 0 else 0.0,
            "floor": float(ind["floor"]),
        }

        if ref is not None:
            x_star = ref["x_star"](t)
            p_star = ref["p_star"](t)
            entry["err_x_inf"] = float(np.max(np.abs(x - x_star)))
            entry["err_p_inf"] = float(np.max(np.abs(lam - p_star)))
            entry["err_J_abs"] = float(abs(sol["J"] - ref["J_star"]))

        log.append(entry)

        if verbose:
            print(
                f"[adapt {iters_used:02d}] N={N:4d} "
                f"res={sol['res_inf']:.2e} "
                f"eta={eta_time:.2e}/{tol_star:.2e} "
                f"marked={n_marked}"
            )

        # This configuration is solved and evaluated
        iters_used += 1
        pending_update = False

        # 3) Stop criterion (article)
        if eta_time < tol_star:
            converged = True
            stop_reason = "time_tolerance_reached"
            z0 = sol["z"]  # keep consistent final pack
            break

        # 4) Decide and apply update (time refinement)
        t_new, marked = refine_mesh_article(
            t_old=t,
            r_bar=ind["r_bar"],
            tol_time=tol_time,
            s_mark=s_mark,
            M_sub=M_sub,
        )

        if not np.any(marked):
            stop_reason = "no_intervals_marked"
            z0 = sol["z"]
            break

        # 5) Prolongate warm start to refined mesh
        xg, lg = prolongate_guess(t, x, lam, t_new)
        z0 = pack_z(xg, lg)

        # Commit update and mark pending resolve
        t = t_new
        pending_update = True

        # Loop continues; if budget already reached, one extra resolve is still allowed
        # because pending_update=True (handled at loop top).

    # ---- final result payload ----
    result = {
        "converged": converged,
        "stop_reason": stop_reason,
        "iterations": iters_used,
        "maxit": maxit,
        "t": t,
        "log": log,
    }

    # Ensure final fields correspond to current returned mesh
    # (if last action was update, pending_update would have forced one more solve)
    p_final = dict(params)
    p_final["t"] = t
    sol_final = solve_on_mesh(p_final, model, z0=z0)

    result.update({
        "success": bool(sol_final["success"]),
        "message": sol_final["message"],
        "res_inf": float(sol_final["res_inf"]),
        "x": sol_final["x"],
        "lam": sol_final["lam"],
        "a": sol_final["a"],
        "J": float(sol_final["J"]),
        "z": sol_final["z"],
    })

    return result

def plot_ex6_results(result, ref, out_prefix="ex6_test"):
    """
    Plot:
      1) dt(t)
      2) x vs x*
      3) p vs p*
      4) a vs a*
      5) rho, rho_bar
      6) r_bar
    Assumes:
      result has keys: t, x, lam, a, log
      ref has keys: x_star, p_star, a_star
      last log entry has: rho, rho_bar, r_bar
    """

    t = np.asarray(result["t"], dtype=float)
    x = np.asarray(result["x"], dtype=float)
    p = np.asarray(result["lam"], dtype=float)
    a = np.asarray(result["a"], dtype=float)

    dt = np.diff(t)
    t_int = t[:-1]

    x_star = ref["x_star"](t)
    p_star = ref["p_star"](t)
    a_star = ref["a_star"](t_int)

    # indicators from last adapt iteration
    last = result["log"][-1]
    rho = np.asarray(last["rho"], dtype=float)
    rho_bar = np.asarray(last["rho_bar"], dtype=float)
    r_bar = np.asarray(last["r_bar"], dtype=float)

    # ------------------------------------------------------------
    # (1) dt(t)
    # ------------------------------------------------------------
    fig = plt.figure(figsize=(6, 4))
    plt.step(t_int, dt, where="post", label=r"$\Delta t_n$")
    plt.yscale("log")
    plt.xlabel("t")
    plt.ylabel(r"$\Delta t$")
    plt.title(r"Time mesh: $\Delta t(t)$")
    plt.grid(True, which="both")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_t_vs_dt.pdf", bbox_inches="tight")
    plt.close(fig)

    # ------------------------------------------------------------
    # (2) state trajectory
    # ------------------------------------------------------------
    fig = plt.figure(figsize=(6, 4))
    plt.plot(t, x, label="x (computed)", linewidth=2.0)
    plt.plot(t, x_star, "--", label="x* (exact)", linewidth=2.0)
    plt.xlabel("t")
    plt.ylabel("x")
    plt.title("State trajectory")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_state_x.pdf", bbox_inches="tight")
    plt.close(fig)

    # ------------------------------------------------------------
    # (3) costate trajectory
    # ------------------------------------------------------------
    fig = plt.figure(figsize=(6, 4))
    plt.plot(t, p, label="p (computed)", linewidth=2.0)
    plt.plot(t, p_star, "--", label="p* (exact)", linewidth=2.0)
    plt.xlabel("t")
    plt.ylabel("p")
    plt.title("Costate trajectory")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_costate_p.pdf", bbox_inches="tight")
    plt.close(fig)

    # ------------------------------------------------------------
    # (4) control trajectory
    # ------------------------------------------------------------
    fig = plt.figure(figsize=(6, 4))
    plt.step(t_int, a, where="post", label="a (computed)", linewidth=2.0)
    plt.step(t_int, a_star, where="post", linestyle="--", label="a* (exact)", linewidth=2.0)
    plt.xlabel("t")
    plt.ylabel("a")
    plt.title("Control trajectory")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_control_a.pdf", bbox_inches="tight")
    plt.close(fig)

    # ------------------------------------------------------------
    # (5) rho_n and rho_bar_n
    # ------------------------------------------------------------
    fig = plt.figure(figsize=(6, 4))
    plt.step(t_int, rho, where="post", label=r"$\rho_n$")
    plt.step(t_int, rho_bar, where="post", label=r"$\bar{\rho}_n$")
    plt.xlabel("t")
    plt.ylabel(r"$\rho$")
    plt.title(r"Error density: $\rho_n,\ \bar{\rho}_n$")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_rho_density.pdf", bbox_inches="tight")
    plt.close(fig)

    # ------------------------------------------------------------
    # (6) r_bar_n indicator
    # ------------------------------------------------------------
    fig = plt.figure(figsize=(6, 4))
    plt.step(t_int, r_bar, where="post", label=r"$\bar r_n = |\bar\rho_n|\Delta t_n^2$")
    plt.yscale("log")
    plt.xlabel("t")
    plt.ylabel(r"$\bar r_n$")
    plt.title(r"Time error indicator $\bar r_n$")
    plt.grid(True, which="both")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_r_indicator.pdf", bbox_inches="tight")
    plt.close(fig)

if __name__ == "__main__":
    params, model, ref = init_ex6(...)  # your initializer
    result = run_adaptivity_test(params, model, ref=ref, verbose=True, maxit=20)
    plot_ex6_results(result, ref, out_prefix="example32_test")

    t = result["t"]
    x = result["x"]
    p = result["lam"]
    a = result["a"]

    x_star = ref["x_star"](t)
    p_star = ref["p_star"](t)
    a_star = ref["a_star"](t[:-1])

    print("\n=== FINAL SUMMARY ===")
    print("converged   :", result["converged"], "| reason:", result["stop_reason"])
    print("iters_used  :", result["iterations"], "/", result["maxit"])
    print("N_final     :", len(t) - 1)
    print("res_inf     :", result["res_inf"])
    print("J_hat       :", result["J"])
    print("J_star      :", ref["J_star"])
    print("||x-x*||inf :", np.max(np.abs(x - x_star)))
    print("||p-p*||inf :", np.max(np.abs(p - p_star)))
    print("||a-a*||inf :", np.max(np.abs(a - a_star)))

