import numpy as np
import jax.numpy as jnp
import jax
# jax.config.update("jax_enable_x64", True)
from itertools import product

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
    
def sample_cube_obs(D, Nobs, method='grid'):
    d = D.shape[0]
    if method == 'grid':
        obs = []
        for i in range(d):
            obs.append(jnp.linspace(D[i, 0], D[i, 1], Nobs))  # Exclude boundaries
        obs = jnp.meshgrid(*obs, indexing='ij')
        obs = jnp.vstack([obs[i].flatten() for i in range(d)]).T

        mask = jnp.any(jnp.isclose(obs, D[:, 0]) | jnp.isclose(obs, D[:, 1]), axis=1)
        obs_int = obs[~mask]
        obs_bnd = obs[mask]

    elif method == 'uniform':
        obs_int  = D[:, 0] + (D[:, 1] - D[:, 0]) * jnp.random.rand((Nobs-2)**d, D.shape[0])
        obs = []
        d = D.shape[0]
        N_per_side = (Nobs ** d - (Nobs-2) ** d) // (2 * d) + 1

        for i in range(D.shape[0]):
            face1 = jnp.full((N_per_side, D.shape[0]), D[i, 0])
            face2 = jnp.full((N_per_side, D.shape[0]), D[i, 1])
            mask = jnp.arange(D.shape[0]) != i
            # D[mask, 0] and D[mask, 1] have shape (d-1,)
            # We add a new axis so that they broadcast to shape (N_per_side, d-1)
            low = D[mask, 0][jnp.newaxis, :]
            high = D[mask, 1][jnp.newaxis, :]
            face1[:, mask] = jnp.random.uniform(low=low, high=high, size=(N_per_side, d-1))
            face2[:, mask] = jnp.random.uniform(low=low, high=high, size=(N_per_side, d-1))
            obs.append(face1)
            obs.append(face2)
        obs_bnd = jnp.vstack(obs)

    else:
        raise ValueError("Unsupported sampling method")
    
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

def merge_kernel_cluster(suppc, xk, sk, ck, p, tau=1.0):
    # do merge of close points every 500 iterations
    active = jnp.asarray(jnp.where(suppc)[0])
    centers_ind = [[active[0]]]
    coefs = [ck[active[0]]] 
    centers = [xk[active[0], :]]
    ss = [sk[active[0], :] if sk.ndim > 1 else sk[active[0]]]
    for ind in active[1:]:
        x = xk[ind, :]
        s = sk[ind, :] if sk.ndim > 1 else sk[ind]
        sig1 = p.kernel.sigma(s)
        flag = False

        for i, (center, s0) in enumerate(zip(centers, ss)):
            sig0 = p.kernel.sigma(s0)
            score = jnp.exp(-jnp.sum((x - center)**2) / (sig0**2 + sig1**2))
            score *= jnp.exp( - jnp.sum((jnp.log(sig0) - jnp.log(sig1))**2) / (2 * tau**2))


            if score > 0.8:
                new_center = (len(centers_ind[i]) / (len(centers_ind[i])+1)) *  center + x / (len(centers_ind[i])+1)
                new_s = (len(centers_ind[i]) / (len(centers_ind[i])+1)) *  s0 + s / (len(centers_ind[i])+1)
                centers_ind[i].append(ind)
                centers[i] = new_center
                ss[i] = new_s
                coefs[i] += ck[ind]
                flag = True
                break
        if not flag:
            centers.append(x)
            ss.append(s)
            centers_ind.append([ind])
            coefs.append(ck[ind])
    print("After merge, number of points reduced from {} to {}".format(jnp.sum(suppc), len(centers)))
    
    # reconstruct xk, sk, ck
    ck = jnp.array(coefs)
    # qk = jnp.sign(ck) + ck 
    xk = jnp.array(centers)
    sk = jnp.array(ss)

    return xk, sk, ck