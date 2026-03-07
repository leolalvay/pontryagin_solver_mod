import numpy as np
import atexit
from pathlib import Path
from .pa_bundle import PABundle
from .smoothing import eval_H_smooth
from .hamiltonian import compute_H
from .newton import solve_tpbvp

def _grads_for_indicators(problem, bundle, p, x, t, delta, use_explicit_hamiltonian_gradients=False):
    if use_explicit_hamiltonian_gradients and problem.hamiltonian_grad_fn is not None:
        Hp, Hx = problem.hamiltonian_gradients(x, p, t)
        return None, Hp, Hx
    return eval_H_smooth(problem, bundle, p, x, t, delta)


def bootstrap_bundle_from_trajectory(
    problem,
    t_nodes: np.ndarray,
    X: np.ndarray,
    P: np.ndarray,
    bundle,
    restricted: bool = True,
    num_support_nodes: int = 20,
    grid_size: int = 3,
    use_oracle: bool = False,
) -> int:
    """
    Bootstrap for PA bundle:
    - If an explicit oracle u_star is available, use it to add candidate controls.
    - Otherwise (or if oracle is not feasible under `restricted`), fall back to a cheap 1D grid search
      (only for scalar control with bounds).
    Returns the number of *new* controls added.
    """

    # detect whether an oracle exists (your OCPProblem.u_star returns (u, ok) or (None, False))
    has_oracle = hasattr(problem, "u_star")

    bounds = problem.control_bounds_tuple()
    u_grid = None

    # grid search is only possible if bounds exist and control is scalar
    if bounds is not None:
        u_min, u_max = bounds
        m = int(u_min.size)
        if m == 1:
            u_grid = np.linspace(float(u_min[0]), float(u_max[0]), int(grid_size))

    N = len(t_nodes) - 1
    if N <= 0:
        return 0

    # pick representative node indices (including endpoints)
    k = min(num_support_nodes, N + 1)
    idx = np.unique(np.round(np.linspace(0, N, k)).astype(int))

    added = 0

    for i in idx:
        x_i = X[i]
        p_i = P[i]
        t_i = float(t_nodes[i])

        # ------------------------------------------------------------
        # 1) Try oracle u_star first (does projection to bounds inside u_star)
        # ------------------------------------------------------------
        if use_oracle and has_oracle:
            u_oracle, ok = problem.u_star(x_i, p_i, t_i, restricted=restricted)
            if (u_oracle is not None) and (not restricted or ok):
                before = bundle.num_planes()
                bundle.add_control(u_oracle)
                if bundle.num_planes() > before:
                    added += 1
                # oracle succeeded -> no need for grid search at this node
                continue

        # ------------------------------------------------------------
        # 2) Fallback: grid search (only if available)
        # ------------------------------------------------------------
        if u_grid is None:
            continue

        best_val = np.inf
        best_u = None

        for a in u_grid:
            u = np.array([a], dtype=float)

            if not problem.admissible_control(u):
                continue
            if restricted:
                if hasattr(problem, "tangent_ok") and (not problem.tangent_ok(x_i, u, t_i)):
                    continue

            val = float(np.dot(p_i, problem.f(x_i, u, t_i)) + problem.l(x_i, u, t_i))
            if val < best_val:
                best_val = val
                best_u = u

        if best_u is not None:
            before = bundle.num_planes()
            bundle.add_control(best_u)
            if bundle.num_planes() > before:
                added += 1

    return added



def solve_optimal_control(
    problem,
    initial_mesh: np.ndarray,
    tol_time: float = 1e-3,
    tol_PA: float = 1e-3,
    tol_delta: float = 1e-3,
    max_iters: int = 10,
    delta0: float = 0.1,
    verbose: bool =  True,
    print_every: int=1,
    log_path: str = "logs/last_run.txt",
    use_oracle_bootstrap: bool = False,
    use_oracle_PA: bool = False,
    use_explicit_hamiltonian_gradients: bool = False,
):
    """
    Solve an optimal control problem adaptively by refining time mesh,
    adding new control planes, and reducing the smoothing parameter.

    Parameters
    ----------
    problem : OCPProblem
        Problem definition.
    initial_mesh : np.ndarray
        Initial time grid (including 0 and T).  Should be sorted.
    tol_time : float
        Tolerance for the time discretisation error indicator.
    tol_PA : float
        Tolerance for the PA surrogate error indicator.
    tol_delta : float
        Tolerance for the smoothing error indicator.
    max_iters : int
        Maximum number of outer adaptivity iterations.
    delta0 : float
        Initial smoothing parameter.

    Returns
    -------
    dict
        Dictionary with solution, mesh, bundle, delta, and log information.
    """
# -------------------------
    # Persistent run log file (overwritten each run)
    # -------------------------
    log_f = None
    if log_path is not None:
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        log_f = open(log_path, "w", buffering=1)  # overwrite, line-buffered
        atexit.register(log_f.close)

    def _log(msg: str):
        if verbose:
            print(msg, flush=True)
        if log_f is not None:
            print(msg, file=log_f, flush=True)

    # copy mesh
    t_nodes = np.asarray(initial_mesh, dtype=float).copy()
    # initialize PA bundle with zero control if dimension known, otherwise empty
    bundle = PABundle()
    # try to add zero control (or mean of bounds) if possible
    bounds = problem.control_bounds_tuple()
    m = problem.m
    if m is None and bounds is not None:
        m = bounds[0].size
    if m is not None:
        if bounds is not None:
            u_min, u_max = bounds
            u_mid = 0.5 * (u_min + u_max)
            bundle.add_control(u_mid)
            bundle.add_control(u_min)
            bundle.add_control(u_max)
        else:
            u0 = np.zeros(m)
            bundle.add_control(u0)
    delta = delta0
    log = []
    # initial guesses for X and P: None (will be set in Newton)
    X_guess = None
    P_guess = None
    s_time = 0.25                   # paper parameter s
    K_time = 1e-6                  # paper parameter K

#==================================== Outer Loop =====================================================
    for k in range(max_iters):
        # solve TPBVP on current mesh with current bundle and delta
        X, P, info = solve_tpbvp(problem, t_nodes, bundle, delta, X_guess, P_guess,use_explicit_hamiltonian_gradients=use_explicit_hamiltonian_gradients)
        # --- bootstrap PA bundle after first coarse solve (minimal change) ---
        if k == 0 and (not use_explicit_hamiltonian_gradients):
            M_before = bundle.num_planes()
            added = bootstrap_bundle_from_trajectory(
                problem,
                t_nodes=t_nodes,
                X=X,
                P=P,
                bundle=bundle,
                restricted=True,
                num_support_nodes=12,
                grid_size=51,
                use_oracle=use_oracle_bootstrap,
            )
            print(f"[bootstrap] M_before={M_before}, added={added}, M_after={bundle.num_planes()}")
            if added > 0:
                # re-solve once with improved bundle (same mesh, same delta)
                X_guess, P_guess = X, P
                X, P, info = solve_tpbvp(problem, t_nodes, bundle, delta, X_guess, P_guess,use_explicit_hamiltonian_gradients)
        delta_solved = delta
        # compute error indicators
        #=======================================================================
        # TIME DISCRETIZATION ERROR
        #=======================================================================
        N = len(t_nodes) - 1
        eta_time_local = np.zeros(N)   # will store r_bar_n
        
        if N > 0:
            dt = np.diff(t_nodes)
            dt_max = float(np.max(dt))
            floor = K_time * np.sqrt(dt_max)

            rho_arr = np.zeros(N)
            rho_bar_arr = np.zeros(N)
            for i in range(N):
                # evaluate at symplectic-Euler point (p_{i+1}, x_i, t_i)
                _, Hp, Hx = _grads_for_indicators(problem, bundle, P[i + 1], X[i], t_nodes[i], delta,use_explicit_hamiltonian_gradients=use_explicit_hamiltonian_gradients )

                rho_arr[i] = -0.5 * float(np.dot(Hp, Hx))
                rho_bar_arr[i] = max(abs(rho_arr[i]), floor)
                #rho_bar_arr[i] = np.sign(rho_arr[i]) * max(abs(rho_arr[i]), floor)

                eta_time_local[i] = abs(rho_bar_arr[i]) * (dt[i] ** 2)  # r_bar_i

            eta_time = float(np.max(eta_time_local))          # r*
            tol_time_star = float(tol_time / N)               # TOL/N
            mark_thr = float(s_time * tol_time / N)           # s*TOL/N
        else:
            eta_time = 0.0
            tol_time_star = tol_time
            mark_thr = 0.0
        
    
        # PA error: integrate (Hbar - H)
        eta_PA = 0.0
        for i in range(N):
            # at node i and i+1, compute gap
            Hbar_i, _ = bundle.evaluate(problem, P[i], X[i], t_nodes[i])
            Hbar_ip1, _ = bundle.evaluate(problem, P[i + 1], X[i + 1], t_nodes[i + 1])
            # compute true H (restricted) at i and i+1
            H_i, _ = compute_H(problem, P[i], X[i], t_nodes[i], bundle.controls, restricted=True,use_oracle=use_oracle_PA)
            H_ip1, _ = compute_H(problem, P[i + 1], X[i + 1], t_nodes[i + 1], bundle.controls, restricted=True, use_oracle=use_oracle_PA)
            gap_i = Hbar_i - H_i
            gap_ip1 = Hbar_ip1 - H_ip1
            dt = t_nodes[i + 1] - t_nodes[i]
            eta_PA += 0.5 * (gap_i + gap_ip1) * dt
        # smoothing error: integrate (H_delta - Hbar)
        eta_delta = 0.0
        for i in range(N):
            Hdelta_i, _, _ = eval_H_smooth(problem, bundle, P[i], X[i], t_nodes[i], delta)
            Hdelta_ip1, _, _ = eval_H_smooth(problem, bundle, P[i + 1], X[i + 1], t_nodes[i + 1], delta)
            Hbar_i, _ = bundle.evaluate(problem, P[i], X[i], t_nodes[i])
            Hbar_ip1, _ = bundle.evaluate(problem, P[i + 1], X[i + 1], t_nodes[i + 1])
            diff_i = Hbar_i - Hdelta_i 
            diff_ip1 = Hbar_ip1 - Hdelta_ip1
            dt = t_nodes[i + 1] - t_nodes[i]
            eta_delta += 0.5 * (diff_i + diff_ip1) * dt
        log.append({
            'iteration': k,
            'N': N,
            'M': bundle.num_planes(),
            'delta': delta,
            'eta_time': eta_time,
            'eta_PA': eta_PA,
            'eta_delta': eta_delta,
            'rho': rho_arr.copy(),
            'rho_bar': rho_bar_arr.copy(),
            'r_bar': eta_time_local.copy(),
            'tol_time_star': tol_time_star,
            'mark_thr': mark_thr,
            't_nodes_iter': t_nodes.copy(),
            'newton_iter': info['iterations'],
            'newton_residual': info['residual_norm'],
        })
# ------------------------------------------------------------
        # Per-iteration concise print (1 line): progress + next action
        # ------------------------------------------------------------
        if (k % max(int(print_every), 1)) == 0:
            dt_all = np.diff(t_nodes)
            dt_min = float(np.min(dt_all)) if dt_all.size else 0.0
            dt_max_ = float(np.max(dt_all)) if dt_all.size else 0.0
            n_mark = int(np.sum(eta_time_local > mark_thr)) if N > 0 else 0

            converged = (eta_time <= tol_time_star) and (eta_PA <= tol_PA) and (eta_delta <= tol_delta)

            if converged:
                action = "STOP"
            elif eta_time > tol_time_star:
                action = f"refine_time(marked={n_mark})"
            elif eta_PA > tol_PA:
                action = "add_plane"
            elif eta_delta > tol_delta:
                action = "delta*=0.5"
            else:
                action = "continue"

            _log(
                f"[adapt {k:02d}] "
                f"N={N:4d} M={bundle.num_planes():3d} dt=[{dt_min:.2e},{dt_max_:.2e}] delta={delta:.2e} | "
                f"Newton it={info['iterations']:2d} res={info['residual_norm']:.2e} | "
                f"eta_time={eta_time:.2e}/{tol_time_star:.2e} "
                f"eta_PA={eta_PA:.2e}/{tol_PA:.2e} "
                f"eta_delta={eta_delta:.2e}/{tol_delta:.2e} -> {action}"
            )


        # check convergence
        #if (eta_time <= tol_time) and (eta_PA <= tol_PA) and (eta_delta <= tol_delta):
        if (eta_time <= tol_time_star) and (eta_PA <= tol_PA) and (eta_delta <= tol_delta):
            break
        # priority: refine time first, then PA planes, then reduce delta
        
        #if eta_time > tol_time:
        if eta_time > tol_time_star:
            # refine time mesh: subdivide intervals with high local error
            new_nodes = [t_nodes[0]]
            X_new = [X[0]]
            P_new = [P[0]]
            for i in range(N):
                dt = t_nodes[i + 1] - t_nodes[i]
                # compute midpoint and error indicator
                err = eta_time_local[i]
                if err > mark_thr:
                #if err > tol_time:
                    # insert midpoint
                    t_mid = 0.5 * (t_nodes[i] + t_nodes[i + 1])
                    # linear interpolate X and P
                    alpha = (t_mid - t_nodes[i]) / dt
                    x_mid = (1 - alpha) * X[i] + alpha * X[i + 1]
                    p_mid = (1 - alpha) * P[i] + alpha * P[i + 1]
                    new_nodes.extend([t_mid])
                    X_new.extend([x_mid])
                    P_new.extend([p_mid])
                new_nodes.append(t_nodes[i + 1])
                X_new.append(X[i + 1])
                P_new.append(P[i + 1])
            t_nodes = np.array(new_nodes, dtype=float)
            X_guess = np.array(X_new)
            P_guess = np.array(P_new)
            continue
        
        if eta_PA > tol_PA:
            # add new plane: find worst gap index
            max_gap = -np.inf
            max_idx = 0
            for i in range(N + 1):
                Hbar_i, _ = bundle.evaluate(problem, P[i], X[i], t_nodes[i])
                H_i, u_star = compute_H(problem, P[i], X[i], t_nodes[i], bundle.controls, restricted=True, use_oracle=use_oracle_PA)
                gap = Hbar_i - H_i
                if gap > max_gap:
                    max_gap = gap
                    max_idx = i
                    best_u = u_star
            # add best_u to bundle
            if best_u is not None:
                bundle.add_control(best_u)
            X_guess = X
            P_guess = P
            continue
        # else reduce delta
        if eta_delta > tol_delta:
            delta = delta * 0.5
            # do not change mesh or bundle
            X_guess = X
            P_guess = P
            continue
    # return final solution and log
    if (X is None) or (len(t_nodes) != X.shape[0]) or (len(t_nodes) != P.shape[0]) or (delta_solved != delta):
        X, P, info = solve_tpbvp(problem, t_nodes, bundle, delta, X_guess, P_guess,use_explicit_hamiltonian_gradients)
        #(final_resolve): At this point we re-solve TPBVP so that (X,P) match the returned `delta`.
        # However, the error indicators (eta_time, eta_PA, eta_delta) below are NOT recomputed at this final delta;
        # they may correspond to the previous outer-iteration values. Recompute them here later if needed.
        N = len(t_nodes) - 1
        eta_time_local = np.zeros(N)

        dt = np.diff(t_nodes) if N > 0 else np.zeros(0)
        dt_max = float(np.max(dt)) if N > 0 else 0.0
        floor = K_time * np.sqrt(dt_max) if N > 0 else 0.0

        rho_arr = np.zeros(N)
        rho_bar_arr = np.zeros(N)

        for i in range(N):
            _, Hp, Hx = _grads_for_indicators(problem, bundle, P[i + 1], X[i], t_nodes[i], delta, use_explicit_hamiltonian_gradients=use_explicit_hamiltonian_gradients)
            rho_arr[i] = -0.5 * float(np.dot(Hp, Hx))
            rho_bar_arr[i] = np.sign(rho_arr[i]) * max(abs(rho_arr[i]), floor)
            eta_time_local[i] = abs(rho_bar_arr[i]) * (dt[i] ** 2)

        eta_time = float(np.max(eta_time_local)) if N > 0 else 0.0
        tol_time_star = float(tol_time / N) if N > 0 else tol_time
        mark_thr = float(s_time * tol_time / N) if N > 0 else 0.0
        # --- recompute eta_PA at final (consistent with returned X,P,t_nodes,bundle,delta) ---
        eta_PA = 0.0
        for i in range(N):
            Hbar_i, _ = bundle.evaluate(problem, P[i], X[i], t_nodes[i])
            Hbar_ip1, _ = bundle.evaluate(problem, P[i + 1], X[i + 1], t_nodes[i + 1])

            H_i, _ = compute_H(problem, P[i], X[i], t_nodes[i], bundle.controls, restricted=True, use_oracle=use_oracle_PA)
            H_ip1, _ = compute_H(problem, P[i + 1], X[i + 1], t_nodes[i + 1], bundle.controls, restricted=True, use_oracle=use_oracle_PA)

            gap_i = Hbar_i - H_i
            gap_ip1 = Hbar_ip1 - H_ip1
            dt_i = t_nodes[i + 1] - t_nodes[i]
            eta_PA += 0.5 * (gap_i + gap_ip1) * dt_i

        # --- recompute eta_delta at final ---
        eta_delta = 0.0
        for i in range(N):
            Hdelta_i, _, _ = eval_H_smooth(problem, bundle, P[i], X[i], t_nodes[i], delta)
            Hdelta_ip1, _, _ = eval_H_smooth(problem, bundle, P[i + 1], X[i + 1], t_nodes[i + 1], delta)

            Hbar_i, _ = bundle.evaluate(problem, P[i], X[i], t_nodes[i])
            Hbar_ip1, _ = bundle.evaluate(problem, P[i + 1], X[i + 1], t_nodes[i + 1])

            diff_i = Hbar_i - Hdelta_i
            diff_ip1 = Hbar_ip1 - Hdelta_ip1
            dt_i = t_nodes[i + 1] - t_nodes[i]
            eta_delta += 0.5 * (diff_i + diff_ip1) * dt_i


        if len(log) > 0:
            log.append({
                'iteration': log[-1]['iteration'] + 1,
                'N': len(t_nodes) - 1,
                'M': bundle.num_planes(),
                'delta': delta,
                'eta_time': eta_time,
                'eta_PA': eta_PA,
                'eta_delta': eta_delta,
                #'eta_time': log[-1]['eta_time'],   # (opcional) si quieres exactitud, luego lo recalculamos
                #'eta_PA': log[-1]['eta_PA'],
                #'eta_delta': log[-1]['eta_delta'],
                'newton_iter': info['iterations'],
                'newton_residual': info['residual_norm'],
                'note': 'final_resolve',
                #'rho': rho_arr.copy(),
                #'rho_bar': rho_bar_arr.copy(),
                #'r_bar': eta_time_local.copy(),
                'tol_time_star': tol_time_star,
                'mark_thr': mark_thr,
                #'t_nodes_iter': t_nodes.copy(),
            })
    return {
        't_nodes': t_nodes,
        'X': X,
        'P': P,
        'bundle': bundle,
        'rhobar'  : rho_bar_arr,
        'rbar'  : eta_time_local,
        'delta': delta,
        'log': log
    }

  

