# `core/integrators.py` — Symplectic Euler residual + finite-difference Jacobian

This module turns the **δ-smoothed Pontryagin system** into a **nonlinear algebraic system**
$$
F(z)=0
$$
that can be solved by Newton (via the shooting layer). The key is that we do **not** “integrate forward” as a standalone simulator; instead, we **assemble residual blocks** that enforce the symplectic Euler updates and the terminal boundary condition.

---

## 1) Continuous system being discretized (what we want to enforce)

Using the smoothed Hamiltonian $H_\delta(p,x,t)$, the canonical PMP dynamics are
$$
\dot x(t)=\nabla_p H_\delta(p(t),x(t),t),
\qquad
-\dot p(t)=\nabla_x H_\delta(p(t),x(t),t),
$$
with boundary conditions
$$
x(0)=x_0,
\qquad
p(T)+\nabla g(x(T))=0.
$$

In the implementation, $\nabla_p H_\delta$ and $\nabla_x H_\delta$ are obtained from
`eval_H_smooth(...)`, and $\nabla g$ is approximated by finite differences.

---

## 2) Unknown vector $z$ (how the solver stores $(X,P)$)

On a mesh $0=t_0<\dots<t_N=T$, we store samples
$$
X_i\approx x(t_i),\qquad P_i\approx p(t_i).
$$

The initial state $X_0=x_0$ is **fixed**, so the unknown vector concatenates
$$
z = (X_1,\dots,X_N,\;P_0,\dots,P_N).
$$

This is exactly what `pack_unknowns` does:

```python
def pack_unknowns(X, P):
    N_plus_1, n = X.shape
    N = N_plus_1 - 1
    z = np.zeros((N * n + (N + 1) * n,)) #z=(X_1,..,P_N)
    z[0:N * n] = X[1:, :].reshape(N * n)      # x_1,...,x_N
    z[N * n:]  = P.reshape((N + 1) * n)       # p_0,...,p_N
    return z
```

And `unpack_unknowns` reconstructs $(X,P)$ from $z$ while reinserting $X_0=x_0$:

```python
def unpack_unknowns(z, x0):
    n = x0.size
    total = z.size // n
    N = (total - 1) // 2
    X = np.zeros((N + 1, n))
    P = np.zeros((N + 1, n))
    X[0, :]   = x0
    X[1:, :]  = z[0:N * n].reshape((N, n))
    P[:, :]   = z[N * n:].reshape((N + 1, n))
    return X, P
```

---

## 3) Symplectic Euler discretization (residual blocks)

For each step $i=0,\dots,N-1$ with $\Delta t_i=t_{i+1}-t_i$, symplectic Euler is enforced as:

**State update (gradient at the start):**
$$
X_{i+1} = X_i + \Delta t_i\,\nabla_p H_\delta(P_i,X_i,t_i).
$$

**Costate update (gradient at the end):**
$$
P_i = P_{i+1} + \Delta t_i\,\nabla_x H_\delta(P_{i+1},X_{i+1},t_{i+1}).
$$

The code enforces these by assembling residuals
$$
r_x^{(i)} = X_i + \Delta t_i\,\nabla_p H_\delta(P_i,X_i,t_i) - X_{i+1},
$$
$$
r_p^{(i)} = P_{i+1} + \Delta t_i\,\nabla_x H_\delta(P_{i+1},X_{i+1},t_{i+1}) - P_i.
$$

Here is the exact implementation pattern inside `assemble_residual`:
```python
def assemble_residual(problem, t_nodes: np.ndarray, X: np.ndarray, P: np.ndarray, bundle, delta: float) -> np.ndarray:
N_plus_1 = t_nodes.size        # number of time nodes = N+1
N = N_plus_1 - 1               # number of steps
n = X.shape[1]                 # dimension of the state (and the costate)
residual = np.zeros((2 * N * n + n,))
offset =0
```

There are $N$ steps.

At each step, there are **2 vector equations** of dimension $n$:

- One for the **state**  $r_{x_i} \in \mathbb{R}^n$
- One for the **costate** $r_{p_i} \in \mathbb{R}^n$

That gives  $2 N n$ components.

At the end, there is a **terminal boundary condition** $r_{bc} \in \mathbb{R}^n $.

**Total:**  
$$
2 N n + n
$$

`offset` is a cursor that indicates the position in the residual vector where we are writting in.
```python
for i in range(N):
        dt = t_nodes[i + 1] - t_nodes[i]
        x_i = X[i]
        x_ip1 = X[i + 1]
        p_i = P[i]
        p_ip1 = P[i + 1]
        # gradient at start
        _, grad_p_i, _ = eval_H_smooth(problem, bundle, p_i, x_i, t_nodes[i], delta)
        # gradient at end (for costate update)
        _, _, grad_x_ip1 = eval_H_smooth(problem, bundle, p_ip1, x_ip1, t_nodes[i + 1], delta)
        # state residual r_x = x_i + dt * grad_p - x_{i+1}
        r_x = x_i + dt * grad_p_i - x_ip1
        residual[offset:offset + n] = r_x
        offset += n
        # costate residual r_p = p_{i+1} + dt * grad_x_ip1 - p_i
        r_p = p_ip1 + dt * grad_x_ip1 - p_i
        residual[offset:offset + n] = r_p
        offset += n
```

For each time step iteration `i`, the residual $r_x^{i}$ and $r_p^{i}$ are stored in this order. This means that the assembly of the residual is 
$$
F=(r_x^{0},r_p^{0},r_x^{1},r_p^1,\cdots,r_x^{N-1},r_p^{N-1},.)
$$
```python
# terminal boundary condition: p_N + ∇g(x_N) = 0
    x_N = X[-1]
    p_N = P[-1]
    # gradient of g by finite difference
    g_grad = np.zeros_like(p_N)
    eps = 1e-6
    for j in range(n):
        x_plus = x_N.copy()
        x_minus = x_N.copy()
        x_plus[j] += eps
        x_minus[j] -= eps
        g_plus = problem.g(x_plus)
        g_minus = problem.g(x_minus)
        g_grad[j] = (g_plus - g_minus) / (2 * eps)
    r_bc = p_N + g_grad
    residual[offset:] = r_bc
```
The terminal boundary condition is added to obtain
$$
F=(r_x^{0},r_p^{0},r_x^{1},r_p^1,\cdots,r_x^{N-1},r_p^{N-1},r_{bc})
$$
---

## 4) Terminal boundary condition $p(T)+\nabla g(x(T))=0$

At the final node $(X_N,P_N)$, the code appends the boundary residual
$$
r_{\mathrm{bc}} = P_N + \nabla g(X_N).
$$

Since the problem object exposes $g(x)$, the gradient is computed by central finite differences:

```python
x_N = X[-1]
p_N = P[-1]
g_grad = np.zeros_like(p_N)
eps = 1e-6
for j in range(n):
    x_plus  = x_N.copy(); x_plus[j]  += eps
    x_minus = x_N.copy(); x_minus[j] -= eps
    g_plus  = problem.g(x_plus)
    g_minus = problem.g(x_minus)
    g_grad[j] = (g_plus - g_minus) / (2 * eps)

r_bc = p_N + g_grad
```

---
# 5) `assemble_jacobian` (core / `integrators.py`)

This note documents the **mathematics** and the **implementation** of:

```python
assemble_jacobian(problem, t_nodes, X, P, bundle, delta) -> np.ndarray
```

The goal is to make future debugging and maintenance straightforward.

---

## 1) What this function computes

`assemble_jacobian(...)` builds the Jacobian matrix

$$
J \;=\;\frac{\partial F}{\partial z},
$$

where:

- $z$ is the flattened vector of unknowns (state + costate),
- $F(z)$ is the residual produced by `assemble_residual(...)` for a **symplectic Euler** discretization of the Hamiltonian system + terminal boundary condition.

The implementation **exploits locality** of the discretization: instead of computing all columns by global finite differences on the full residual, it fills only the **necessary block entries** (a block stencil), using local finite differences only for the nonlinear pieces.

---

## 2) Shapes and indexing conventions (repo order)

Let:

- $N = \texttt{t\_nodes.size} - 1$  (number of time steps),
- $n = \texttt{X.shape[1]}$         (dimension of state/costate).

Inputs:

- `t_nodes`: shape $(N+1,)$, nodes $t_0<\dots<t_N$
- `X`: shape $(N+1,n)$, state trajectory $x_0,\dots,x_N$
- `P`: shape $(N+1,n)$, costate trajectory $p_0,\dots,p_N$

### Unknown vector $z$

`pack_unknowns(X,P)` defines the unknown ordering used throughout the repo:

$$
z \;=\; (x_1,\dots,x_N,\; p_0,\dots,p_N)\in\mathbb{R}^{(2N+1)n}.
$$

Important:
- $x_0$ is **fixed** (given by the problem) and is **not** included in $z$.

So the Jacobian is square:

$$
m = (2N+1)n,\qquad J\in\mathbb{R}^{m\times m}.
$$

### Residual vector $F$ ordering

`assemble_residual(...)` returns:

$$
F = (r_x^0, r_p^0, r_x^1, r_p^1, \dots, r_x^{N-1}, r_p^{N-1}, r_{bc}),
$$

so:

- each $r_x^i\in\mathbb{R}^n$
- each $r_p^i\in\mathbb{R}^n$
- $r_{bc}\in\mathbb{R}^n$
- total length: $2Nn+n=(2N+1)n$.

---

## 3) Discretization behind the residual

For each step $i=0,\dots,N-1$, define:

- $\Delta t_i = t_{i+1}-t_i$
- gradients from the smoothed Hamiltonian (as returned by `eval_H_smooth`):

$$
(\_, \nabla_p H_\delta, \nabla_x H_\delta)
= \texttt{eval\_H\_smooth}(\texttt{problem},\texttt{bundle},p_{i+1},x_i,t_i,\delta).
$$

The residual blocks are:

### State residual
$$
r_x^i \;=\; x_{i+1} - x_i - \Delta t_i\,\nabla_p H_\delta(p_{i+1},x_i,t_i).
$$

### Costate residual
$$
r_p^i \;=\; p_i - p_{i+1} - \Delta t_i\,\nabla_x H_\delta(p_{i+1},x_i,t_i).
$$

### Terminal boundary condition
The terminal condition is

$$
r_{bc} \;=\; p_N + \nabla g(x_N),
$$

where `assemble_residual` computes $$\nabla g(x_N)$$ by **central finite differences**.

---

## 4) Sparsity / locality structure (block stencil)

From the formulas:

- $r_x^i$ depends on $x_{i+1}, x_i, p_{i+1}$
- $r_p^i$ depends on $p_i, p_{i+1}, x_i$
- $r_{bc}$ depends on $x_N, p_N$

So the Jacobian is **block-banded** (block size $n\times n$), with only a few nonzero blocks per residual block row.

A convenient “stencil view”:

- Row block for $r_x^i$ touches columns for $x_{i+1}$, $x_i$ (if $i\ge 1$), and $p_{i+1}$.
- Row block for $r_p^i$ touches columns for $p_i$, $p_{i+1}$, and $x_i$ (if $i\ge 1$).
- Row block for $r_{bc}$ touches columns for $x_N$ and $p_N$.

Why the condition $i\ge 1$?
- For $i=0$, $x_0$ is not part of $z$, so there is **no column** corresponding to $x_0$.

---

## 5) Exact linear blocks vs. nonlinear blocks

Split each residual block into “linear” parts plus “nonlinear” parts.

### Define nonlinear helpers (as in code)

The code defines:

$$
\phi(i) = -\Delta t_i \,\nabla_p H_\delta(p_{i+1},x_i,t_i),
$$

so

$$
r_x^i = (x_{i+1}-x_i) + \phi(i).
$$

And:

$$
\psi(i) = -\Delta t_i \,\nabla_x H_\delta(p_{i+1},x_i,t_i),
$$

so

$$
r_p^i = (p_i-p_{i+1}) + \psi(i).
$$

### Linear Jacobian blocks (filled analytically)

From the explicit linear parts:

- $$\frac{\partial r_x^i}{\partial x_{i+1}} = I$$
- $$\frac{\partial r_x^i}{\partial x_i} = -I, \quad i\ge1$$

- $$\frac{\partial r_p^i}{\partial p_i} = I$$
- $$\frac{\partial r_p^i}{\partial p_{i+1}} = -I$$

For the boundary condition:

- $$\frac{\partial r_{bc}}{\partial p_N} = I$$

These blocks are inserted directly into `J[...] = I` or `-I`.

### Nonlinear Jacobian blocks (approximated by finite differences)

The remaining dependencies come from derivatives of $\phi(i)$ and $\psi(i)$:

- $\frac{\partial \phi(i)}{\partial x_i}$, $\frac{\partial \psi(i)}{\partial x_i}$  (only if $i\ge1$)
- $\frac{\partial \phi(i)}{\partial p_{i+1}}$, $\frac{\partial \psi(i)}{\partial p_{i+1}}$

These correspond (conceptually) to second derivatives of the smoothed Hamiltonian:

- $\partial_{x}\nabla_p H_\delta$, $\partial_{p}\nabla_p H_\delta$
- $\partial_{x}\nabla_x H_\delta$, $\partial_{p}\nabla_x H_\delta$

The code approximates them using **central differences** with step:

- `eps = 1e-7` (local FD step for Jacobian blocks)

---

## 6) Row/column maps used in the implementation

The code builds index helpers consistent with the repo ordering:

### Column starts in $z$

- `col_x(k)` for $x_k$ (valid only for $k=1,\dots,N$):
  $$
  \texttt{col\_x}(k) = (k-1)n
  $$
- `col_p(j)` for $p_j$ (valid for $j=0,\dots,N$):
  $$
  \texttt{col\_p}(j) = Nn + jn
  $$

### Row starts in $F$

- `row_rx(i)` for $r_x^i$:
  $$
  \texttt{row\_rx}(i) = (2i)n
  $$
- `row_rp(i)` for $r_p^i$:
  $$
  \texttt{row\_rp}(i) = (2i+1)n
  $$
- boundary block:
  $$
  \texttt{row\_bc} = (2N)n
  $$

These are used to place each $n\times n$ block into the correct slice of `J`.

---

## 7) Implementation walkthrough (what the code does)

### Step A — Initialize
- allocates:
  - `J = np.zeros((m,m))`
  - `I = np.eye(n)`
- chooses FD step:
  - `eps = 1e-7`

> Note: `J` is dense (`np.zeros`). The *pattern* is sparse/block-banded, but stored densely here.

### Step B — Define local nonlinear functions
- `phi(i)` calls `eval_H_smooth(...)` and returns $$-\Delta t_i\nabla_p H_\delta(...)$$
- `psi(i)` calls `eval_H_smooth(...)` and returns $$-\Delta t_i\nabla_x H_\delta(...)$$

### Step C — Loop over time steps $$i=0,\dots,N-1$$ and fill blocks
For each step:

1) Fill exact linear blocks:

- in `r_x^i` rows:
  - `d r_x^i / d x_{i+1} = I`
  - `d r_x^i / d x_i = -I` only if `i >= 1`

- in `r_p^i` rows:
  - `d r_p^i / d p_i = I`
  - `d r_p^i / d p_{i+1} = -I`

2) Add nonlinear corrections via local FD:

- If `i >= 1`, for each coordinate `ell = 0..n-1`, perturb `X[i, ell]`:

  - set `X[i, ell] = old + eps`, compute `phi_p, psi_p`
  - set `X[i, ell] = old - eps`, compute `phi_m, psi_m`
  - restore `X[i, ell] = old`

  Then:

  $$
  d\phi \approx \frac{\phi_p-\phi_m}{2\,\texttt{eps}},\qquad
  d\psi \approx \frac{\psi_p-\psi_m}{2\,\texttt{eps}}.
  $$

  And add these column vectors into `J`:

  - `r_x^i` rows get `dphi`
  - `r_p^i` rows get `dpsi`

- For all `i`, for each coordinate `ell = 0..n-1`, perturb `P[i+1, ell]` similarly to approximate:

  - $$\partial\phi/\partial p_{i+1}$$
  - $$\partial\psi/\partial p_{i+1}$$

  Then add:

  - `r_x^i` rows get `dphi` in columns of `p_{i+1}`
  - `r_p^i` rows get `dpsi` in columns of `p_{i+1}` (this is the “extra” beyond the linear `-I`)

### Step D — Boundary condition blocks
- sets `d r_bc / d p_N = I`

- computes `d r_bc / d x_N` by central differences on `bc_block()` with step `eps = 1e-7`:

  - `bc_block()` returns $$p_N + \nabla g(x_N)$$
  - and it computes $$\nabla g(x_N)$$ internally by central differences with `epsg = 1e-6`

**Important numerical note (nested FD):**  
`d r_bc / d x_N` is computed by finite differences of a quantity that already uses finite differences for $$\nabla g$$. In practice this acts like a finite-difference approximation of the Hessian of $$g$$ (up to the chosen steps).

---

## 8) Numerical / implementation caveats (for future debugging)

1) **In-place perturbations**
The function perturbs entries of `X` and `P` **in place** and restores them.  
This is fine in a single-threaded Newton solve, but be careful if:
- `X` or `P` is shared elsewhere,
- you ever attempt parallelization of Jacobian assembly.

2) **Finite difference steps**
- Local Jacobian FD step: `eps = 1e-7`
- Gradient of terminal cost: `epsg = 1e-6` inside `bc_block()`

If Newton becomes unstable or noisy, one possible knob is tuning these steps.

3) **Dense storage**
Even though the Jacobian is block-banded, it is stored in a dense `np.ndarray`.
If scalability becomes a concern, consider switching to `scipy.sparse` and a sparse linear solver.

4) **Consistency with `assemble_residual`**
This Jacobian must remain consistent with the residual definition:
- gradients evaluated at $$(p_{i+1}, x_i, t_i)$$
- residual ordering $$F = (r_x^0,r_p^0,\dots,r_{bc})$$
- unknown ordering $$z=(x_1,\dots,x_N,p_0,\dots,p_N)$$

If any of those conventions change, update the index maps and block dependencies accordingly.

---

## 9) Complexity (rough)

Let each call to `eval_H_smooth` cost $C_H$.

For each step $i$:
- `x_i`-derivatives (only if `i>=1`): $$2n$$ calls to `eval_H_smooth` (plus/minus for each coordinate) for `phi` and `psi`.
- `p_{i+1}`-derivatives: $$2n$$ calls similarly.

So per step: about $$4n$$ calls to `eval_H_smooth` (and about $4n-?$ for `i=0` since `x_0` part is skipped).

Total: $$O(Nn\,C_H)$$ calls, instead of **global FD** which would be $$O(m\,C_F)$$ with $m=(2N+1)n$ and $C_F\sim O(NC_H)$, i.e. much worse scaling.

---

## 10) Minimal correctness checklist (debug-friendly)

When debugging suspected Jacobian issues, check:

1) Dimensions:
- `J.shape == ((2*N+1)*n, (2*N+1)*n)`

2) Known exact blocks:
- `d r_x^i / d x_{i+1} == I`
- `d r_p^i / d p_i == I`
- boundary: `d r_bc / d p_N == I`

3) Special case `i=0`:
- No columns exist for `x_0`, so the code must *not* write `d r_*^0 / d x_0`.

4) Sign conventions:
- `r_x^i = x_{i+1} - x_i - dt * grad_p`
- `r_p^i = p_i - p_{i+1} - dt * grad_x`

If you flip signs in the residual, you must flip the corresponding Jacobian blocks.

---

## 11) Relationship to the old implementation

The file still contains the previous “global finite difference” version (commented out).  
Conceptually, this new routine aims to produce the **same Jacobian** as that global FD approach, but:
- it avoids building columns that should be structurally zero (locality),
- it inserts linear blocks exactly,
- and it uses FD only for the nonlinear sensitivity blocks.

This makes the Jacobian assembly significantly cheaper while keeping the implementation relatively simple.

---

