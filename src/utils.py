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
    normsv = np.abs(v)

    # Safeguard against division by zero
    normsv_safe = np.maximum(normsv, (mu + np.finfo(float).eps) * np.finfo(float).eps)

    # Apply soft shrinkage operator
    shrink_factor = np.maximum(0, 1 - mu / normsv_safe)
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
        self.p_vec = np.ones(self.Nx) / self.Nx_int # Weight vector
        self.p_vec[-self.Nx_bnd:] = scale / self.Nx_bnd # Apply penalty for boundary conditions
        self.p_vec = self.p_vec.reshape(-1, 1) # reshape to column vector (Nx, 1)

    def F(self, y):
        y = y.reshape(-1, 1)
        """Computes the objective function F(y)."""
        return 0.5 * np.sum(self.p_vec * y ** 2)

    def dF(self, y):
        y = y.reshape(-1, 1)
        """Computes the gradient of F(y)."""
        return self.p_vec * y

    def ddF(self, y):
        """Computes the Hessian (second derivative) of F(y)."""
        return np.diag(self.p_vec.flatten())
    
def sample_cube_obs(D, Nobs, method='grid'):
    d = D.shape[0]
    if method == 'grid':
        obs = []
        for i in range(d):
            obs.append(np.linspace(D[i, 0], D[i, 1], Nobs))  # Exclude boundaries
        obs = np.meshgrid(*obs, indexing='ij')
        obs = np.vstack([obs[i].flatten() for i in range(d)]).T

        mask = np.any(np.isclose(obs, D[:, 0]) | np.isclose(obs, D[:, 1]), axis=1)
        obs_int = obs[~mask]
        obs_bnd = obs[mask]

    elif method == 'uniform':
        obs_int  = D[:, 0] + (D[:, 1] - D[:, 0]) * np.random.rand((Nobs-2)**d, D.shape[0])
        obs = []
        d = D.shape[0]
        N_per_side = (Nobs ** d - (Nobs-2) ** d) // (2 * d) + 1

        for i in range(D.shape[0]):
            face1 = np.full((N_per_side, D.shape[0]), D[i, 0])
            face2 = np.full((N_per_side, D.shape[0]), D[i, 1])
            mask = np.arange(D.shape[0]) != i
            # D[mask, 0] and D[mask, 1] have shape (d-1,)
            # We add a new axis so that they broadcast to shape (N_per_side, d-1)
            low = D[mask, 0][np.newaxis, :]
            high = D[mask, 1][np.newaxis, :]
            face1[:, mask] = np.random.uniform(low=low, high=high, size=(N_per_side, d-1))
            face2[:, mask] = np.random.uniform(low=low, high=high, size=(N_per_side, d-1))
            obs.append(face1)
            obs.append(face2)
        obs_bnd = np.vstack(obs)

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
        return self.th * t + (1 - self.th) *  np.log(1 + self.gam * t) / self.gam

    def dphi(self, t):
        """Evaluate derivative dphi(t)."""
        if self.gamma == 0:
            return np.ones_like(t)
        return self.th + (1 - self.th) / (1 + self.gam * t)

    def ddphi(self, t):
        """Evaluate second derivative ddphi(t)."""
        if self.gamma == 0:
            return np.zeros_like(t)
        return -(1 - self.th) * self.gam / (1 + self.gam * t) ** 2

    def inv(self, y):
        """Evaluate inverse or upper bound."""
        if self.gamma == 0:
            return y
        return y / self.th  # Upper bound for the inverse

    def prox(self, sigma, g):
        """Evaluate proximity operator."""
        if self.gamma == 0:
            return np.maximum(g - sigma, 0)
        return 0.5 * np.maximum(
            (g - sigma * self.th - 1 / self.gam) + np.sqrt((g - sigma * self.th - 1 / self.gam) ** 2 + 4 * (g - sigma) / self.gam),
            0
        )

    

def shapeParser(func, pad=False):
    if not pad:
        def wrapper(self, X, *args):
            return func(self, X.shape, X, *args)
    else:
        def wrapper(self, X, s, c, *args):
            pad_size = self.pad_size
            N, d = X.shape  # Extract shape
            _, s_dim = s.shape
            X_padded = jnp.zeros((pad_size, d)).at[:N, :].set(X)  # Pad along first dim
            s_padded = jnp.zeros((pad_size, s_dim)).at[:N, :].set(s)
            c_padded = jnp.zeros(pad_size).at[:N].set(c)  # Pad along first dim

            # Call the function and get the output
            output = func(self, (pad_size, d), X_padded, s_padded, c_padded, *args)

            def slice_arr(arr):
                arr_shape = arr.shape
                # if no dimension is padded, return the original array
                if arr_shape[0] == pad_size:
                    slice_sizes = (N,) + arr_shape[1:]
                elif arr_shape[1] == pad_size:
                    slice_sizes = (arr_shape[0], N) + arr_shape[2:]  
                else:
                    return arr 
                arr_sliced = jax.lax.dynamic_slice(arr, (0,) * arr.ndim, slice_sizes)

                return arr_sliced
            
            if type(output) == dict:
                for key in output.keys():
                    output[key] = slice_arr(output[key])
                output_sliced = output
            elif output.ndim == 1:
                return output
            else:
                # output_sliced = output
                output_sliced = slice_arr(output)

            return output_sliced
    return wrapper




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

