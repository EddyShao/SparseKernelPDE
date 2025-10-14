import numpy as np
import jax.numpy as jnp
import jax
# jax.config.update("jax_enable_x64", True)
from itertools import product
import jax.numpy as jnp
import jax.random 

def computeProx(v, mu):
    """
    Compute the proximal operator using soft shrinkage.

    Args:
        v (ndarray): Input array.
        mu (float): Shrinkage parameter.

    Returns:
        vprox (ndarray): Output array after applying the shrinkage operator.
    """
    # Compute vector norms
    normsv = jnp.abs(v)

    # Safeguard against division by zero
    normsv_safe = jnp.maximum(normsv, (mu + jnp.finfo(float).eps) * jnp.finfo(float).eps)

    # Apply soft shrinkage operator
    shrink_factor = jnp.maximum(0, 1 - mu / normsv_safe)
    vprox = shrink_factor * v  # Element-wise multiplication

    return vprox


class Objective:
    """
    Class for defining the Objective: objective function and its derivatives.
    """

    def __init__(self, Nx_int, Nx_bnd, scale=200.0):
        self.scale = scale
        self.Nx_int, self.Nx_bnd = Nx_int, Nx_bnd
        self.Nx = self.Nx_int + self.Nx_bnd
        self.p_vec = jnp.ones(self.Nx) / self.Nx_int # Weight vector
        # self.p_vec[-self.Nx_bnd:] = scale / self.Nx_bnd # Apply penalty for boundary conditions
        self.p_vec = self.p_vec.at[-self.Nx_bnd:].set(scale / self.Nx_bnd)
        self.p_vec = self.p_vec.reshape(-1, 1) # reshape to column vector (Nx, 1)

    def F(self, y):
        y = y.reshape(-1, 1)
        """Computes the objective function F(y)."""
        return 0.5 * jnp.sum(self.p_vec * y ** 2)

    def dF(self, y):
        y = y.reshape(-1, 1)
        """Computes the gradient of F(y)."""
        return self.p_vec * y

    def ddF(self, y):
        """Computes the Hessian (second derivative) of F(y)."""
        return jnp.diag(self.p_vec.flatten())
    
    def ddF_quad(self, y, Q):
        """Computes the Hessian-vector product q^{T} ddF(y) * q."""
        # return matrix Q^T @ ddF @ Q
        return Q.T @ (self.p_vec * Q)
    
# def sample_cube_obs(D, Nobs_int, Nobs_bnd, method='grid', rng=None):
#     """
#     Sample interior and boundary points from a d-D box.
#     - Grid: builds a Cartesian grid with n points/axis so that (n-2)^d ≈ Nobs_int.
#             Returns all interior/boundary points from that grid (counts may differ from requested).
#     - Uniform: samples exactly Nobs_int interior and Nobs_bnd boundary points i.i.d.

#     Args:
#         D: (d, 2) array; row i is [low_i, high_i]
#         Nobs_int: desired total interior points (used to pick grid density for 'grid')
#         Nobs_bnd: desired total boundary points (only used in 'uniform')
#         method: 'grid' or 'uniform'
#         rng: optional jax.random.PRNGKey (used in 'uniform'); if None, jax.random.key(0)
#     Returns:
#         obs_int: (Ni, d)
#         obs_bnd: (Nb, d)
#     """
#     d = D.shape[0]
#     lows, highs = D[:, 0], D[:, 1]

#     if method == 'grid':
#         # Choose n points/axis so that (n-2)^d ≈ Nobs_int (and at least 1 interior if possible)
#         n_axis = int(jnp.maximum(3, 2 + jnp.ceil(Nobs_int ** (1.0 / d))))
#         axes = [jnp.linspace(lows[i], highs[i], n_axis) for i in range(d)]
#         mesh = jnp.meshgrid(*axes, indexing='ij')
#         pts = jnp.vstack([m.ravel() for m in mesh]).T  # (n_axis**d, d)

#         # Boundary mask: any coordinate equals low or high (use isclose for numeric safety)
#         is_low  = jnp.isclose(pts, lows[None, :])
#         is_high = jnp.isclose(pts, highs[None, :])
#         mask_bnd = jnp.any(is_low | is_high, axis=1)

#         obs_bnd = pts[ mask_bnd]
#         obs_int = pts[~mask_bnd]
#         return obs_int, obs_bnd

#     elif method == 'uniform':
#         if rng is None:
#             rng = jax.random.key(0)
#         key_int, key_bnd = jax.random.split(rng, 2)

#         # Interior: i.i.d. uniform in the open box
#         eps = jnp.finfo(jnp.float32).eps
#         lo_i = lows + eps
#         hi_i = highs - eps
#         obs_int = jax.random.uniform(key_int, (Nobs_int, d), minval=lo_i, maxval=hi_i)

#         # Boundary: sample faces uniformly among 2d faces
#         axes_idx = jax.random.randint(key_bnd, (Nobs_bnd,), 0, d)      # which axis is clamped
#         key_side, key_rest = jax.random.split(key_bnd)
#         sides = jax.random.bernoulli(key_side, p=0.5, shape=(Nobs_bnd,))  # 0=low, 1=high

#         pts = jax.random.uniform(key_rest, (Nobs_bnd, d), minval=lo_i, maxval=hi_i)
#         clamp_vals = jnp.where(sides[:, None] == 1, highs[None, :], lows[None, :])
#         rows = jnp.arange(Nobs_bnd)
#         pts = pts.at[rows, axes_idx].set(clamp_vals[rows, axes_idx])

#         obs_bnd = pts
#         return obs_int, obs_bnd

#     else:
#         raise ValueError("method must be 'grid' or 'uniform'")


# ---------- Low-discrepancy helpers ----------
def _van_der_corput(n, base):
    v = 0.0
    denom = 1.0
    while n > 0:
        n, rem = divmod(n, base)
        denom *= base
        v += rem / denom
    return v

def _first_primes(k):
    # tiny prime list is enough for d<=10
    primes = []
    x = 2
    while len(primes) < k:
        for p in primes:
            if x % p == 0:
                break
        else:
            primes.append(x)
        x += 1
    return primes

def halton_sequence(n, d):
    bases = _first_primes(d)
    idx = jnp.arange(1, n + 1)
    # vectorized van-der-corput via vmap over python loop is awkward; do python loop—small n is fine
    cols = []
    for b in bases:
        vals = jnp.array([_van_der_corput(int(i), b) for i in range(1, n + 1)])
        cols.append(vals)
    return jnp.stack(cols, axis=1)

def hammersley_sequence(n, d):
    # dim 0: i/n, others: van der Corput in first_primes(d-1)
    bases = _first_primes(max(d - 1, 0))
    i = jnp.arange(n)
    cols = [i / n]
    for b in bases:
        vals = jnp.array([_van_der_corput(int(k + 1), b) for k in range(n)])
        cols.append(vals)
    return jnp.stack(cols[:d], axis=1)

def sobol_2d(n):
    # Simple 2D Sobol using direction numbers (good for demos)
    # Source: standard 2D construction; no scrambling
    def _sobol_point(i):
        # i starts at 1
        # x uses direction numbers v_j = 1/2, 1/4, 1/8, ...
        # y uses primitive polynomial x^3 + x^2 + 1 -> direction ints [1,3,5]
        x = 0
        y = 0
        # x
        v = 0.5
        ii = i
        while ii:
            if ii & 1:
                x ^= v
            ii >>= 1
            v *= 0.5
        # y
        dirs = [0.5, 0.25, 0.75 * 0.25]  # quick-and-dirty set for illustration
        ii = i
        j = 0
        while ii:
            if ii & 1:
                y ^= dirs[j] if j < len(dirs) else (0.5 ** (j + 1))
            ii >>= 1
            j += 1
        return jnp.array([x, y])
    return jnp.stack([_sobol_point(i) for i in range(1, n + 1)], axis=0)

# ---------- Latin hypercube ----------
def latin_hypercube(key, n, d):
    # Build n points: for each dim, permute n strata
    u = jax.random.uniform(key, (n, d))
    # centers in each stratum
    strata = (jnp.arange(n)[..., None] + u) / n  # (n, d)
    # permute independently per dim
    permuted = []
    subkeys = jax.random.split(key, d)
    for j in range(d):
        perm = jax.random.permutation(subkeys[j], n)
        permuted.append(strata[perm, j])
    return jnp.stack(permuted, axis=1)

# ---------- Utility ----------
def _scale_to_box(unit_pts, D):
    lows, highs = D[:, 0], D[:, 1]
    return lows + unit_pts * (highs - lows)

def _sample_boundary_uniform(key, D, N):
    d = D.shape[0]
    lows, highs = D[:, 0], D[:, 1]
    eps = jnp.finfo(jnp.float32).eps
    lo_i, hi_i = lows + eps, highs - eps
    key_axis, key_side, key_rest = jax.random.split(key, 3)
    axes_idx = jax.random.randint(key_axis, (N,), 0, d)
    sides = jax.random.bernoulli(key_side, 0.5, (N,))
    pts = jax.random.uniform(key_rest, (N, d), minval=lo_i, maxval=hi_i)
    clamp_vals = jnp.where(sides[:, None] == 1, highs[None, :], lows[None, :])
    rows = jnp.arange(N)
    pts = pts.at[rows, axes_idx].set(clamp_vals[rows, axes_idx])
    return pts

# ---------- Main API ----------
def sample_cube_obs(
    D, Nobs_int, Nobs_bnd, method="grid",
    rng=None, residual_fn=None, alpha=2.0, topk_ratio=0.1
):
    """
    Methods:
      'grid'       : Cartesian grid (approx counts).
      'uniform'    : i.i.d. uniform interior + uniform faces.
      'lhs'        : Latin hypercube (interior) + uniform faces.
      'halton'     : Halton seq (interior) + uniform faces.
      'hammersley' : Hammersley seq (interior) + uniform faces.
      'sobol2'     : 2D Sobol (interior) + uniform faces (only d=2).
      'rad'        : Residual-based Adaptive Distribution (needs residual_fn).
      'rar_d'      : Residual-based Adaptive Refinement with Distribution (needs residual_fn).

    Args:
      D           : (d,2) box.
      Nobs_int    : total interior points requested (exact for quasi/uniform; approx for grid).
      Nobs_bnd    : total boundary points requested.
      rng         : jax.random.PRNGKey, optional.
      residual_fn : callable(x: (N,d)) -> residual magnitude (N,), used by 'rad' and 'rar_d'.
      alpha       : RAD exponent (importance ∝ residual^alpha).
      topk_ratio  : RAR-D: fraction of points to refine near top residuals.
    """
    if rng is None:
        rng = jax.random.key(0)
    d = D.shape[0]
    lows, highs = D[:, 0], D[:, 1]

    # ---- Boundary sampling (shared) ----
    key_int, key_bnd, key_aux = jax.random.split(rng, 3)
    obs_bnd = _sample_boundary_uniform(key_bnd, D, Nobs_bnd)

    # ---- Interior by method ----
    if method == "grid":
        n_axis = int(jnp.maximum(3, 2 + jnp.ceil(Nobs_int ** (1.0 / d))))
        axes = [jnp.linspace(lows[i], highs[i], n_axis) for i in range(d)]
        mesh = jnp.meshgrid(*axes, indexing="ij")
        pts = jnp.vstack([m.ravel() for m in mesh]).T
        is_low = jnp.isclose(pts, lows[None, :])
        is_high = jnp.isclose(pts, highs[None, :])
        obs_int = pts[~jnp.any(is_low | is_high, axis=1)]

    elif method == "uniform":
        eps = jnp.finfo(jnp.float32).eps
        lo, hi = lows + eps, highs - eps
        obs_int = jax.random.uniform(key_int, (Nobs_int, d), minval=lo, maxval=hi)

    elif method == "lhs":
        unit = latin_hypercube(key_int, Nobs_int, d)
        obs_int = _scale_to_box(unit, D)

    elif method == "halton":
        unit = halton_sequence(Nobs_int, d)
        obs_int = _scale_to_box(unit, D)
        
    elif method == "hammersley":
        unit = hammersley_sequence(Nobs_int, d)
        obs_int = _scale_to_box(unit, D)

    elif method == "sobol2":
        assert d == 2, "sobol2 only implemented for 2D demo."
        unit = sobol_2d(Nobs_int)
        obs_int = _scale_to_box(unit, D)

    elif method in ("rad", "rar_d"):
        if residual_fn is None:
            raise ValueError(f"{method} requires residual_fn(x)->residual magnitude.")

        # base pool (quasi-uniform) to evaluate residuals
        base_N = max(Nobs_int, 5_000)  # pool size; tune as you like
        unit_pool = halton_sequence(base_N, d)
        pool = _scale_to_box(unit_pool, D)

        # residuals & RAD weights
        r = residual_fn(pool)  # shape (base_N,)
        # stabilize: add epsilon
        w = (r + 1e-12) ** alpha
        w = w / jnp.sum(w)

        if method == "rad":
            idx = jax.random.choice(key_int, base_N, shape=(Nobs_int,), p=w, replace=True)
            obs_int = pool[idx]

        else:  # 'rar_d'
            K = max(1, int(topk_ratio * base_N))
            # top-K residual points
            top_idx = jnp.argsort(-r)[:K]
            top_pts = pool[top_idx]
            # local refinement: jitter around top points
            # draw M points near each top point
            per = max(1, Nobs_int // K)
            subkeys = jax.random.split(key_aux, K)
            neigh = []
            scale = 0.02 * (highs - lows)  # 2% box size; tune
            for k in range(K):
                eps = jax.random.normal(subkeys[k], (per, d)) * scale
                cand = top_pts[k][None, :] + eps
                # clamp back to box
                cand = jnp.clip(cand, lows, highs)
                neigh.append(cand)
            neigh = jnp.vstack(neigh)
            # mix with RAD samples to keep distributional coverage
            mix_rad = max(0, Nobs_int - neigh.shape[0])
            if mix_rad > 0:
                idx = jax.random.choice(key_int, base_N, shape=(mix_rad,), p=w, replace=True)
                obs_int = jnp.vstack([neigh, pool[idx]])[:Nobs_int]
            else:
                obs_int = neigh[:Nobs_int]

    else:
        raise ValueError("Unknown method")

    return obs_int, obs_bnd


class Phi:
    """
    Class for defining a penalty function Phi and its derivatives.
    """

    def __init__(self, gamma):
        """
        Initializes the Phi object with given gamma.
        Args:
            gamma (float): Regularization parameter.
        """
        self.gamma = gamma
        self.th = 1 / 2  # Threshold parameter
        self.gam = gamma / (1 - self.th) if gamma != 0 else 0  # Adjusted gamma

    def phi(self, t):
        """Evaluate phi(t)."""
        if self.gamma == 0:
            return t
        return self.th * t + (1 - self.th) *  jnp.log(1 + self.gam * t) / self.gam

    def dphi(self, t):
        """Evaluate derivative dphi(t)."""
        if self.gamma == 0:
            return jnp.ones_like(t)
        return self.th + (1 - self.th) / (1 + self.gam * t)

    def ddphi(self, t):
        """Evaluate second derivative ddphi(t)."""
        if self.gamma == 0:
            return jnp.zeros_like(t)
        return -(1 - self.th) * self.gam / (1 + self.gam * t) ** 2

    def inv(self, y):
        """Evaluate inverse or upper bound."""
        if self.gamma == 0:
            return y
        return y / self.th  # Upper bound for the inverse

    def prox(self, sigma, g):
        """Evaluate proximity operator."""
        if self.gamma == 0:
            return jnp.maximum(g - sigma, 0)
        return 0.5 * jnp.maximum(
            (g - sigma * self.th - 1 / self.gam) + jnp.sqrt((g - sigma * self.th - 1 / self.gam) ** 2 + 4 * (g - sigma) / self.gam),
            0
        )


def compute_rhs(p, x, s, c, xhat_int=None, xhat_bnd=None):
    if xhat_int is None or xhat_bnd is None:
        xhat_int = p.xhat_int
        xhat_bnd = p.xhat_bnd
    linear_results_int = p.kernel.linear_E_results_X_c_Xhat(x, s, c, xhat_int)
    linear_results_bnd = p.kernel.linear_B_results_X_c_Xhat(x, s, c, xhat_bnd)
    rhs_int = p.kernel.E_gauss_X_c_Xhat(*linear_results_int)
    rhs_bnd = p.kernel.B_gauss_X_c_Xhat(*linear_results_bnd)
    return jnp.hstack([rhs_int, rhs_bnd]), linear_results_int, linear_results_bnd


def compute_y(p, x, s, c, xhat_int=None, xhat_bnd=None, func=None):
    if xhat_int is None or xhat_bnd is None:
        xhat_int = p.xhat_int
        xhat_bnd = p.xhat_bnd
    if func is None:
        func = p.kernel.gauss_X_c_Xhat
    y_int = func(x, s, c, xhat_int)
    y_bnd = func(x, s, c, xhat_bnd)
    return y_int, y_bnd

def compute_errors(p, x, s, c, test_int=None, test_bnd=None, y_true_int=None, y_true_bnd=None):
    if test_int is None or test_bnd is None:
        test_int = p.test_int
        test_bnd = p.test_bnd
    if y_true_int is None or y_true_bnd is None:
        y_true_int = p.ex_sol(test_int)
        y_true_bnd = p.ex_sol(test_bnd)
    len_test = test_bnd.shape[0] + test_int.shape[0]
    y_pred_int, y_pred_bnd = compute_y(p, x, s, c, test_int, test_bnd)
    l2_error = jnp.sqrt((jnp.sum((y_pred_int - y_true_int)**2) + jnp.sum((y_pred_bnd - y_true_bnd)**2)) * p.vol_D / len_test)
    l_inf_error_int = jnp.max(jnp.abs(y_pred_int - y_true_int))
    l_inf_error_bnd = jnp.max(jnp.abs(y_pred_bnd - y_true_bnd))
    return {
        'L_2': l2_error,
        'L_inf_int': l_inf_error_int,
        'L_inf_bnd': l_inf_error_bnd,
        'L_inf': max(l_inf_error_int, l_inf_error_bnd)
    }





if __name__ == '__main__':
    D = np.array([
        [0., 1.],
        [0., 1.],
        [0., 1.],
    ])

    Nobs = 20

    obs_int_grid, obs_bnd_grid = sample_cube_obs(D, Nobs, method='grid')
    obs_int_uniform, obs_bnd_uniform = sample_cube_obs(D, Nobs, method='uniform')

    print(obs_int_grid.shape, obs_bnd_grid.shape)
    print(obs_int_uniform.shape, obs_bnd_uniform.shape)


    # import matplotlib.pyplot as plt

    # plt.figure(figsize=(12, 6))
    # fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    # axs[0].scatter(obs_int_grid[:, 0], obs_int_grid[:, 1], label='Grid Sampling')
    # axs[0].scatter(obs_bnd_grid[:, 0], obs_bnd_grid[:, 1], label='Grid Sampling')

    # axs[1].scatter(obs_int_uniform[:, 0], obs_int_uniform[:, 1], label='Uniform Sampling')
    # axs[1].scatter(obs_bnd_uniform[:, 0], obs_bnd_uniform[:, 1], label='Uniform Sampling')
    # plt.suptitle('Sampling Observations')
    # axs[0].set_title('Interior Observations')
    # axs[1].set_title('Boundary Observations')
    
    # plt.title('Grid Sampling')
    # plt.legend()
    # plt.show()

