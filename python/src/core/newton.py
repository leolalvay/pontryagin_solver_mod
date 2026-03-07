import numpy as np
import time
from .integrators import unpack_unknowns, pack_unknowns
from .shooting import shooting_residual, shooting_jacobian
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import splu
from scipy.sparse import issparse


def solve_tpbvp(problem, t_nodes: np.ndarray, bundle, delta: float,
                 X_init: np.ndarray = None, P_init: np.ndarray = None,
                 tol: float = 1e-8, max_iter: int = 20,  use_explicit_hamiltonian_gradients: bool = False) -> tuple:
    """
    Solve the two-point boundary value problem by damped Newton method.

    Parameters
    ----------
    problem : OCPProblem
        The optimal control problem to solve.
    t_nodes : np.ndarray
        Discretised time mesh (length N+1).
    bundle : PABundle
        Bundle of control candidates for Hamiltonian smoothing.
    delta : float
        Smoothing parameter for the Hamiltonian.
    X_init : np.ndarray, optional
        Initial guess for state trajectory (shape (N+1, n)).  If None,
        a linear interpolation between x0 and zeros is used.
    P_init : np.ndarray, optional
        Initial guess for costate trajectory (shape (N+1, n)).  If None,
        the costate is initialised to zeros.
    tol : float
        Tolerance for residual norm to declare convergence.
    max_iter : int
        Maximum number of Newton iterations.

    Returns
    -------
    (X, P, info)
        Solved state and costate trajectories and an info dict containing
        convergence diagnostics.
    """
    N_plus_1 = t_nodes.size
    n = problem.x0.size
    # initial guess
    if X_init is None:
        # linearly interpolate from x0 to zeros (rough guess)
        X_init = np.zeros((N_plus_1, n))
        X_init[0] = problem.x0
        for i in range(1, N_plus_1):
            alpha = i / (N_plus_1 - 1)
            X_init[i] = (1 - alpha) * problem.x0
    if P_init is None:
        P_init = np.zeros((N_plus_1, n))
    # pack unknowns z: x1..xN, p0..pN
    z = pack_unknowns(X_init, P_init)
    # Newton iteration
    for it in range(max_iter):
        F = shooting_residual(problem, t_nodes, z, bundle, delta,use_explicit_gradients=use_explicit_hamiltonian_gradients)
        normF = np.linalg.norm(F, ord=np.inf)
        if normF < tol:
            # converged
            break
        J = shooting_jacobian(problem, t_nodes, z, bundle, delta, use_explicit_gradients=use_explicit_hamiltonian_gradients)
        # solve J * dz = -F (sparse LU on a CSC view of J)
        try:
            lu = splu(csc_matrix(J), permc_spec="COLAMD")
            dz = lu.solve(-F)
        except Exception:
            J_dense = J.toarray() if issparse(J) else J
            dz = np.linalg.solve(J_dense, -F)
            
        #--------------------------------------------------
        # solve J * dz = -F
        #try:
            #dz = np.linalg.solve(J, -F)
        #except np.linalg.LinAlgError:
            # fallback to least squares if singular
            #dz, *_ = np.linalg.lstsq(J, -F, rcond=None)
        #------------------------------------------------------
        # line search
        lam = 1.0
        z_new = z + lam * dz
        F_new = shooting_residual(problem, t_nodes, z_new, bundle, delta, use_explicit_gradients=use_explicit_hamiltonian_gradients)
        normF_new = np.linalg.norm(F_new, ord=np.inf)
        # backtracking Armijo
        while normF_new > (1 - 1e-4 * lam) * normF and lam > 1e-4:
            lam *= 0.5
            z_new = z + lam * dz
            F_new = shooting_residual(problem, t_nodes, z_new, bundle, delta, use_explicit_gradients=use_explicit_hamiltonian_gradients)
            normF_new = np.linalg.norm(F_new, ord=np.inf)
        z = z_new
    # reconstruct solution
    X_sol, P_sol = unpack_unknowns(z, problem.x0)
    info = {
        'iterations': it + 1,
        'residual_norm': np.linalg.norm(shooting_residual(problem, t_nodes, z, bundle, delta, use_explicit_gradients=use_explicit_hamiltonian_gradients), ord=np.inf),
    }
    return X_sol, P_sol, info