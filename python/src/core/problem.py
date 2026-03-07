"""
problem.py
-------------

This module defines the ``OCPProblem`` class, which encapsulates all
problem‑specific data for an optimal control problem (OCP).  It stores
the system dynamics, running cost, terminal cost, initial state, final
horizon, and optional control and state constraints.  The solver code
delegates calls to these methods and properties so that the core
algorithms remain problem‑agnostic.  The interface mirrors the
``OCPProblem`` definition used in the MATLAB implementation.

Functions are vectorized wherever possible.  Control and state bounds
are specified as NumPy arrays of shape ``(m,)`` and ``(n,)``
respectively, where ``m`` is the number of control inputs and ``n`` is
the number of states.
"""

from __future__ import annotations

from typing import Callable, Optional, Tuple, Sequence
import numpy as np


class OCPProblem:
    """Encapsulates an optimal control problem.

    The problem is specified by the continuous dynamics

        ẋ(t) = f(x(t), u(t), t),

    a running cost ℓ(x,u,t), and a terminal cost g(x(T)).  The initial
    state x(0) = x0 and horizon T are fixed.  Optionally, box
    constraints on the control (u) and state (x) can be supplied.

    Parameters
    ----------
    dynamics : Callable[[np.ndarray, np.ndarray, float], np.ndarray]
        The system dynamics function f(x,u,t) → ẋ of dimension n.
    stage_cost : Callable[[np.ndarray, np.ndarray, float], float]
        The running cost ℓ(x,u,t).
    terminal_cost : Callable[[np.ndarray], float]
        The terminal cost g(x).  It is a function of the terminal
        state only.
    x0 : np.ndarray
        Initial state vector of shape (n,).
    T : float
        Final horizon.  Must be positive.
    control_bounds : Optional[Tuple[np.ndarray, np.ndarray]], optional
        Tuple (u_min, u_max) specifying componentwise lower and upper
        bounds for the control.  Each is a 1D array of shape (m,).
        If None, the control is unconstrained.  Default is None.
    state_bounds : Optional[Tuple[np.ndarray, np.ndarray]], optional
        Tuple (x_min, x_max) specifying componentwise bounds for the
        state.  Each is a 1D array of shape (n,).  If None, the
        state is unconstrained.  Default is None.
    """

    def __init__(
        self,
        dynamics: Callable[[np.ndarray, np.ndarray, float], np.ndarray],
        stage_cost: Callable[[np.ndarray, np.ndarray, float], float],
        terminal_cost: Callable[[np.ndarray], float],
        x0: np.ndarray,
        T: float,
        control_bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        state_bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        hamiltonian_true: Optional[Callable[[np.ndarray, np.ndarray, float], float]] = None,
        u_star_fn: Optional[Callable[[np.ndarray, np.ndarray, float], np.ndarray]] = None,
        hamiltonian_grad_fn: Optional[Callable[[np.ndarray, np.ndarray, float], Tuple[np.ndarray, np.ndarray]]] = None,
    ) -> None:
        self.f_fn = dynamics
        self.l_fn = stage_cost
        self.g_fn = terminal_cost
        self.x0 = np.asarray(x0, dtype=float)
        self.T = float(T)
        self.hamiltonian_true_fn = hamiltonian_true
        self.u_star_fn = u_star_fn
        self.hamiltonian_grad_fn = hamiltonian_grad_fn
        # copy bounds if provided
        if control_bounds is not None:
            u_min, u_max = control_bounds
            self.u_min = np.asarray(u_min, dtype=float).copy()
            self.u_max = np.asarray(u_max, dtype=float).copy()
        else:
            self.u_min = None
            self.u_max = None
        if state_bounds is not None:
            x_min, x_max = state_bounds
            self.x_min = np.asarray(x_min, dtype=float).copy()
            self.x_max = np.asarray(x_max, dtype=float).copy()
        else:
            self.x_min = None
            self.x_max = None

    # ------------------------------------------------------------------
    # Dynamics and cost wrappers
    def f(self, x: np.ndarray, u: np.ndarray, t: float) -> np.ndarray:
        """Compute state derivative f(x,u,t).

        Parameters
        ----------
        x : np.ndarray
            State vector of shape (n,).
        u : np.ndarray
            Control vector of shape (m,).
        t : float
            Time.

        Returns
        -------
        np.ndarray
            The state derivative of shape (n,).
        """
        return self.f_fn(x, u, t)

    def l(self, x: np.ndarray, u: np.ndarray, t: float) -> float:
        """Compute running cost ℓ(x,u,t).

        Parameters
        ----------
        x : np.ndarray
        u : np.ndarray
        t : float

        Returns
        -------
        float
        """
        return self.l_fn(x, u, t)

    def g(self, x: np.ndarray) -> float:
        """Compute terminal cost g(x).

        Parameters
        ----------
        x : np.ndarray
            Terminal state vector of shape (n,).

        Returns
        -------
        float
        """
        return self.g_fn(x)


    def u_star(
        self,
        x: np.ndarray,
        p: np.ndarray,
        t: float,
        restricted: bool = False,
        tol: float = 1e-8,
    ):
        """
        Returns (u, feasible). If u_star_fn is not provided -> (None, False).
        Always projects to control bounds (if any). If restricted=True, checks tangent_ok.
        """
        if self.u_star_fn is None:
            return None, False

        u_raw = self.u_star_fn(x, p, t)
        u = np.atleast_1d(np.asarray(u_raw, dtype=float))
        u = self.project_control(u)

        if restricted and not self.tangent_ok(x, u, t, tol=tol):
            return u, False

        return u, True

    def hamiltonian_true(
        self,
        x: np.ndarray,
        p: np.ndarray,
        t: float,
        restricted: bool = False,
        tol: float = 1e-8,
    ):
        """
        Returns (H, u, feasible).

        - Uses u_star_fn if available (through self.u_star), so bounds are enforced.
        - If restricted=True and u_star is not feasible, returns (inf, u, False).
        """
        u, ok = self.u_star(x, p, t, restricted=restricted, tol=tol)

        if u is None:
            raise ValueError("u_star_fn is not provided, cannot compute true Hamiltonian from u_star.")

        if restricted and not ok:
            return float("inf"), u, False

        H = float(p @ self.f_fn(x, u, t) + self.l_fn(x, u, t))
        return H, u, True

    def hamiltonian_gradients(self, x, p, t):
        if self.hamiltonian_grad_fn is None:
            raise ValueError("hamiltonian_grad_fn is not provided.")
        grad_p, grad_x = self.hamiltonian_grad_fn(x, p, t)
        grad_p = np.atleast_1d(np.asarray(grad_p, dtype=float))
        grad_x = np.atleast_1d(np.asarray(grad_x, dtype=float))
        return grad_p, grad_x


    # ------------------------------------------------------------------
    # Control bounds
    def get_control_bounds(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Return control bounds (u_min, u_max) if specified.

        Returns
        -------
        Optional[Tuple[np.ndarray, np.ndarray]]
            Tuple of lower and upper control bounds, or None if
            unconstrained.
        """
        if self.u_min is None or self.u_max is None:
            return None
        return (self.u_min.copy(), self.u_max.copy())

    def admissible_control(self, u: np.ndarray, x: Optional[np.ndarray] = None, t: Optional[float] = None) -> bool:
        """Check if control u satisfies the control bounds.

        Parameters
        ----------
        u : np.ndarray
            Control vector.
        x, t : optional
            State and time (unused here but kept for uniform interface).

        Returns
        -------
        bool
            True if u is within bounds or no bounds defined.
        """
        if self.u_min is None or self.u_max is None:
            return True
        return np.all(u >= self.u_min) and np.all(u <= self.u_max)

    def project_control(self, u: np.ndarray) -> np.ndarray:
        """Project control u onto the control bounds by clipping.

        If no bounds are defined, returns u unchanged.

        Parameters
        ----------
        u : np.ndarray
            Control vector.

        Returns
        -------
        np.ndarray
            The clipped control.
        """
        if self.u_min is None or self.u_max is None:
            return u
        return np.minimum(np.maximum(u, self.u_min), self.u_max)

    # ------------------------------------------------------------------
    # State bounds
    def get_state_bounds(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Return state bounds (x_min, x_max) if specified.

        Returns
        -------
        Optional[Tuple[np.ndarray, np.ndarray]]
            Tuple of lower and upper state bounds, or None if
            unconstrained.
        """
        if self.x_min is None or self.x_max is None:
            return None
        return (self.x_min.copy(), self.x_max.copy())

    def admissible_state(self, x: np.ndarray) -> bool:
        """Check if state x satisfies the state bounds.

        Parameters
        ----------
        x : np.ndarray
            State vector.

        Returns
        -------
        bool
            True if x is within bounds or no bounds defined.
        """
        if self.x_min is None or self.x_max is None:
            return True
        return np.all(x >= self.x_min) and np.all(x <= self.x_max)

    # ------------------------------------------------------------------
    # Convenience accessors used by adaptivity module
    def control_bounds_tuple(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Return the control bounds as a tuple (u_min, u_max) or None.

        The adaptivity module calls this method to obtain the control
        range when initializing the PA bundle.  If no bounds are set,
        returns None.

        Returns
        -------
        Optional[Tuple[np.ndarray, np.ndarray]]
        """
        return self.get_control_bounds()

    def state_bounds_tuple(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Return the state bounds as a tuple (x_min, x_max) or None.
        Similar to ``control_bounds_tuple`` for states.
        """
        return self.get_state_bounds()

    @property
    def n(self) -> int:
        """Alias for state dimension (number of states).  Provided for
        compatibility with the MATLAB version and the adaptivity
        implementation.  Equivalent to ``state_dim``.
        """
        return self.state_dim

    @property
    def m(self) -> Optional[int]:
        """Alias for control dimension.  Provided for compatibility.

        Returns
        -------
        Optional[int]
            The dimension of the control if control bounds are known,
            otherwise None.
        """
        if self.u_min is not None:
            return self.u_min.size
        if self.u_max is not None:
            return self.u_max.size
        # unknown until a control vector is passed to f or l; return None
        return None

    # ------------------------------------------------------------------
    # Viability / tangent cone filter
    def tangent_cone_filter(self, x: np.ndarray, f_candidates: Sequence[np.ndarray], tol: float = 1e-8) -> np.ndarray:
        """Return a boolean mask of candidates that lie in the tangent cone of the state constraints.

        For each candidate derivative f(x,a,t) in ``f_candidates``, this
        checks whether moving in that direction at the current state x
        remains within the state bounds K, using a box constraint model.

        If ``self.x_min`` or ``self.x_max`` are None (i.e. no state
        constraints), all candidates are marked as feasible.

        Parameters
        ----------
        x : np.ndarray of shape (n,)
            Current state.
        f_candidates : Sequence[np.ndarray]
            List or array of candidate state derivatives of shape (k, n) or
            list of shape (n,).
        tol : float
            Tolerance for boundary checks.  If ``|x_i - x_min_i| < tol``
            or ``|x_i - x_max_i| < tol`` then x is treated as on the
            boundary in dimension i.

        Returns
        -------
        np.ndarray
            Boolean array of length equal to len(f_candidates), where
            True indicates the candidate derivative is viable (does not
            point outside the feasible set).
        """
        # If no state bounds, all are viable
        if self.x_min is None or self.x_max is None:
            return np.ones(len(f_candidates), dtype=bool)
        x = np.asarray(x, dtype=float)
        feasible = np.ones(len(f_candidates), dtype=bool)
        for idx, f_vec in enumerate(f_candidates):
            f_vec = np.asarray(f_vec, dtype=float)
            ok = True
            for i in range(x.size):
                if (self.x_min is not None and abs(x[i] - self.x_min[i]) < tol and f_vec[i] < 0.0):
                    # on lower boundary and pointing outwards
                    ok = False
                    break
                if (self.x_max is not None and abs(x[i] - self.x_max[i]) < tol and f_vec[i] > 0.0):
                    ok = False
                    break
            feasible[idx] = ok
        return feasible

    def tangent_ok(self, x: np.ndarray, u: np.ndarray, t: float, tol: float = 1e-8) -> bool:
        """Check if the velocity f(x,u,t) lies in the tangent cone of state constraints.

        This convenience method evaluates the state derivative at (x,u,t) and
        then calls :meth:`tangent_cone_filter` on that single vector.  It
        returns True if the motion does not violate the box constraints
        (i.e. it points inward or tangent on active boundaries).

        Parameters
        ----------
        x : np.ndarray
            State vector.
        u : np.ndarray
            Control vector.
        t : float
            Time.
        tol : float
            Boundary tolerance for viability.

        Returns
        -------
        bool
            True if f(x,u,t) ∈ T_K(x) or if no state constraints.
        """
        # if no state bounds defined, everything is viable
        if self.x_min is None or self.x_max is None:
            return True
        f_vec = self.f(x, u, t)
        mask = self.tangent_cone_filter(x, [f_vec], tol)
        return bool(mask[0])

    # ------------------------------------------------------------------
    # Helper for dimension inference
    @property
    def state_dim(self) -> int:
        """Return the dimension of the state vector."""
        return self.x0.size

    @property
    def control_dim(self) -> int:
        """Return the dimension of the control vector.  If control bounds
        are not specified, this is inferred the first time a control
        vector is passed to ``f`` or ``l``.  Here we attempt to infer
        from bounds if available.  If bounds are None, returns 0
        (caller should handle appropriately).
        """
        if self.u_min is not None:
            return self.u_min.size
        elif self.u_max is not None:
            return self.u_max.size
        else:
            return 0

    # ------------------------------------------------------------------
    # User convenience wrappers
    def __repr__(self) -> str:
        return (f"OCPProblem(state_dim={self.state_dim}, control_dim={self.control_dim}, "
                f"T={self.T}, bounds={'yes' if self.u_min is not None else 'no'})")