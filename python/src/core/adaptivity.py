import numpy as np

from .pa_bundle import PABundle
from .smoothing import eval_H_smooth
from .hamiltonian import compute_H
from .newton import solve_tpbvp

def solve_optimal_control(
    problem,
    initial_mesh: np.ndarray,
    tol_time: float = 1e-3,
    tol_PA: float = 1e-3,
    tol_delta: float = 1e-3,
    max_iters: int = 10,
    delta0: float = 0.1,
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
    s_time = 0.5                   # paper parameter s
    K_time = 1e-6                  # paper parameter K
    for k in range(max_iters):
        # solve TPBVP on current mesh with current bundle and delta
        X, P, info = solve_tpbvp(problem, t_nodes, bundle, delta, X_guess, P_guess)
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
                _, Hp, Hx = eval_H_smooth(problem, bundle, P[i + 1], X[i], t_nodes[i], delta)

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
        '''
        N = len(t_nodes) - 1
        eta_time_local = np.zeros(N)
        grad_p_list = []
        grad_x_list = []
        # compute gradients at nodes for time error and smoothing error
        Hdelta_list =[]
        for i in range(N + 1):
            # grad_p, grad_x at each node
            # use smoothing evaluation
            Hdelta_i, grad_p_i, grad_x_i = eval_H_smooth(problem, bundle, P[i], X[i], t_nodes[i], delta)
            Hdelta_list.append(Hdelta_i)
            grad_p_list.append(grad_p_i)
            grad_x_list.append(grad_x_i)
        # compute local error indicator as difference of gradient across interval
        for i in range(N):
            dt = t_nodes[i + 1] - t_nodes[i]
            gp0 = grad_p_list[i]
            gp1 = grad_p_list[i + 1]
            gx0 = grad_x_list[i]
            gx1 = grad_x_list[i + 1]
            # measure change
            diff_gp = gp1 - gp0
            diff_gx = gx1 - gx0
            eta_time_local[i] = dt * (np.linalg.norm(diff_gp) + np.linalg.norm(diff_gx))
        eta_time = np.max(eta_time_local) if N > 0 else 0.0
        '''

        # ======== DIAG: CURRENT REPO TIME INDICATOR (eta_time_local) ========
        '''
        if N > 0:
            dg_idx = int(np.argmax(eta_time_local))
            dg_dt = t_nodes[dg_idx + 1] - t_nodes[dg_idx]

            dg_gp0 = grad_p_list[dg_idx]
            dg_gp1 = grad_p_list[dg_idx + 1]
            dg_gx0 = grad_x_list[dg_idx]
            dg_gx1 = grad_x_list[dg_idx + 1]

            dg_dgp = dg_gp1 - dg_gp0
            dg_dgx = dg_gx1 - dg_gx0
            dg_norm_dgp = float(np.linalg.norm(dg_dgp))
            dg_norm_dgx = float(np.linalg.norm(dg_dgx))

            dg_err = float(eta_time_local[dg_idx])
            dg_marked = np.where(eta_time_local > tol_time)[0]

            print(f"\n[iter {k}] TIME-ADAPT DIAG (repo-current)  N={N}  M(bundle)={bundle.num_planes()}  delta={delta:.3e}")
            print(f"[iter {k}] GLOBAL: eta_time = max_i eta_time_local[i] = {eta_time:.6e}   vs   tol_time = {tol_time:.6e}")
            print(f"[iter {k}] REFINE rule (repo): refine interval i if eta_time_local[i] > tol_time  -> marked = {len(dg_marked)}")
            if len(dg_marked) > 0:
                print(f"[iter {k}] first marked indices (up to 10): {dg_marked[:10].tolist()}")

            print(f"[iter {k}] WORST interval (argmax): i*={dg_idx}, t_i={t_nodes[dg_idx]:.6e}, dt={dg_dt:.6e}")
            print(f"[iter {k}] eta_time_local(i*) = dt*(||ΔHp|| + ||ΔHx||) = {dg_err:.6e}")
            print(f"[iter {k}] components: dt*||ΔHp|| = {dg_dt*dg_norm_dgp:.6e},  dt*||ΔHx|| = {dg_dt*dg_norm_dgx:.6e}")
            print(f"[iter {k}] norms: ||ΔHp|| = {dg_norm_dgp:.6e},  ||ΔHx|| = {dg_norm_dgx:.6e}")
        else:
            print(f"\n[iter {k}] TIME-ADAPT DIAG (repo-current)  N=0  (nothing to compute)")
            '''
# ======== END DIAG ========

        #============================================================================
        #========================= DIAGNOSIS ========================================
        # --- DIAGNOSTIC (paper-style): rho_n and r_bar_n at current iteration ---
        # ---- parameters for paper marking ----
        '''
        dg_s = 0.5   # <-- set s from the paper/your experiment
        dg_K = 1e-6  # <-- K from the paper/your experiment

        dg_N = len(t_nodes) - 1
        dg_dt = np.diff(t_nodes)

        if dg_N > 0:
            dg_dt_max = float(np.max(dg_dt))

            # 1) compute rho_n (at symplectic Euler evaluation point: (p_{n+1}, x_n, t_n))
            dg_rho = np.zeros(dg_N)
            for dg_n in range(dg_N):
                _, dg_Hp, dg_Hx = eval_H_smooth(problem, bundle, P[dg_n + 1], X[dg_n], t_nodes[dg_n], delta)
                dg_rho[dg_n] = -0.5 * float(np.dot(dg_Hp, dg_Hx))

            # 2) paper regularization: rho_bar_n = sgn(rho_n) * max(|rho_n|, K*sqrt(dt_max))
            dg_floor = dg_K * np.sqrt(dg_dt_max)
            dg_rho_bar = np.sign(dg_rho) * np.maximum(np.abs(dg_rho), dg_floor)

            # 3) indicators: r_bar_n = |rho_bar_n| * dt_n^2
            dg_r_bar = np.abs(dg_rho_bar) * (dg_dt ** 2)

            # 4) paper thresholds
            dg_stop_thr = tol_time / dg_N              # TOL/N
            dg_mark_thr = dg_s * tol_time / dg_N       # s*TOL/N

            # key quantities
            dg_r_star = float(np.max(dg_r_bar))         # max_n r_bar_n
            dg_n_star = int(np.argmax(dg_r_bar))        # argmax
            dg_marked = np.where(dg_r_bar > dg_mark_thr)[0]

            # prints (minimal + paper-aligned)
            print(f"\n[iter {k}] TIME-ADAPT DIAG (paper)  N={dg_N}  M(bundle)={bundle.num_planes()}  delta={delta:.3e}")
            print(f"[iter {k}] floor term: K*sqrt(dt_max) = {dg_floor:.6e}   (dt_max={dg_dt_max:.6e})")

            print(f"[iter {k}] STOP check: r* = max_n r_bar(n) = {dg_r_star:.6e}   vs   TOL/N = {dg_stop_thr:.6e}")
            print(f"[iter {k}] MARK check: marked if r_bar(n) > s*TOL/N = {dg_mark_thr:.6e} (s={dg_s:.3g})  -> marked = {len(dg_marked)}")
            if len(dg_marked) > 0:
                print(f"[iter {k}] first marked indices (up to 10): {dg_marked[:10].tolist()}")

            print(f"[iter {k}] worst interval: n*={dg_n_star}, t_n={t_nodes[dg_n_star]:.6e}, dt={dg_dt[dg_n_star]:.6e}")
            print(f"[iter {k}] rho_bar(n*) = {dg_rho_bar[dg_n_star]: .6e}   r_bar(n*) = {dg_r_bar[dg_n_star]: .6e}")

            # optional (only to understand degeneracy like your Hp=0 case)
            _, dg_Hp_star, dg_Hx_star = eval_H_smooth(problem, bundle, P[dg_n_star + 1], X[dg_n_star], t_nodes[dg_n_star], delta)
            print(f"[iter {k}] Hp(n*) = {dg_Hp_star}   Hx(n*) = {dg_Hx_star}")

        else:
            print(f"\n[iter {k}] TIME-ADAPT DIAG (paper)  N=0  (nothing to compute)")
            '''
#============================================================================
        #============================================================================
    
        # PA error: integrate (Hbar - H)
        eta_PA = 0.0
        for i in range(N):
            # at node i and i+1, compute gap
            Hbar_i, _ = bundle.evaluate(problem, P[i], X[i], t_nodes[i])
            Hbar_ip1, _ = bundle.evaluate(problem, P[i + 1], X[i + 1], t_nodes[i + 1])
            # compute true H (restricted) at i and i+1
            H_i, _ = compute_H(problem, P[i], X[i], t_nodes[i], bundle.controls, restricted=True)
            H_ip1, _ = compute_H(problem, P[i + 1], X[i + 1], t_nodes[i + 1], bundle.controls, restricted=True)
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
                H_i, u_star = compute_H(problem, P[i], X[i], t_nodes[i], bundle.controls, restricted=True)
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
        X, P, info = solve_tpbvp(problem, t_nodes, bundle, delta, X_guess, P_guess)
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
            _, Hp, Hx = eval_H_smooth(problem, bundle, P[i + 1], X[i], t_nodes[i], delta)
            rho_arr[i] = -0.5 * float(np.dot(Hp, Hx))
            rho_bar_arr[i] = np.sign(rho_arr[i]) * max(abs(rho_arr[i]), floor)
            eta_time_local[i] = abs(rho_bar_arr[i]) * (dt[i] ** 2)

        eta_time = float(np.max(eta_time_local)) if N > 0 else 0.0
        tol_time_star = float(tol_time / N) if N > 0 else tol_time
        mark_thr = float(s_time * tol_time / N) if N > 0 else 0.0
        if len(log) > 0:
            log.append({
                'iteration': log[-1]['iteration'] + 1,
                'N': len(t_nodes) - 1,
                'M': bundle.num_planes(),
                'delta': delta,
                'eta_time': log[-1]['eta_time'],   # (opcional) si quieres exactitud, luego lo recalculamos
                'eta_PA': log[-1]['eta_PA'],
                'eta_delta': log[-1]['eta_delta'],
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