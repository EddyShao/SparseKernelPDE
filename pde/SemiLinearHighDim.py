# import jax.numpy as np
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from src.GaussianKernel import GaussianKernel
# from kernel.GaussianKernel_backup import GaussianKernel
# from src.GaussianKernel_backup import GaussianKernel
from src.utils import Objective, sample_cube_obs
import jax
import jax.numpy as jnp
from functools import partial
jax.config.update("jax_enable_x64", True)

# set the random seed


    
class Kernel(GaussianKernel):
    def __init__(self, d, power, sigma_max, sigma_min, anisotropic=False, mask=False, D=None):
        super().__init__(d=d, power=power, sigma_max=sigma_max, sigma_min=sigma_min, anisotropic=anisotropic)
        self.mask = mask
        self.D = D

        # linear results for computing E and B
        self.linear_E = (self.gauss_X_c_Xhat, self.Lap_gauss_X_c_Xhat)
    
        self.linear_B = (self.gauss_X_c_Xhat,)

        self.DE = (0,) 
        self.DB = ()   

    @partial(jax.jit, static_argnums=(0,))
    def gauss(self, x, s, xhat):
        output = super().gauss(x, s, xhat)
        if self.mask:
            mask = jnp.prod(xhat - self.D[:, 0]) * jnp.prod(self.D[:, 1] - xhat)
            output = output * mask
        return output
    
    @partial(jax.jit, static_argnums=(0,))
    def Lap_gauss_X_c(self, X, S, c, xhat):
        # return jnp.trace(jax.hessian(self.gauss_X_c, argnums=3)(X, S, c, xhat))
        diff = X - xhat
        squared_diff = jnp.sum(diff ** 2, axis=1)
        sigma = self.sigma(S).squeeze()
        temp =  (squared_diff - self.d * sigma**2) / sigma ** 4
        lap_phis =  self.gauss_X(X, S, xhat) * temp

        return jnp.dot(c, lap_phis)
        
    @partial(jax.jit, static_argnums=(0,))
    def Lap_gauss_X_c_Xhat(self, X, S, c, Xhat): 
        return jax.vmap(self.Lap_gauss_X_c, in_axes=(None, None, None, 0))(X, S, c, Xhat)

    @partial(jax.jit, static_argnums=(0,))
    def E_gauss_X_c(self, X, S, c, xhat):
        return - self.Lap_gauss_X_c(X, S, c, xhat) + self.gauss_X_c(X, S, c, xhat) ** 3

    @partial(jax.jit, static_argnums=(0,))
    def B_gauss_X_c(self, X, S, c, xhat):
        return self.gauss_X_c(X, S, c, xhat)
    

    def E_gauss_X_c_Xhat(self, *linear_results):
        return - linear_results[1] + linear_results[0] ** 3

    def B_gauss_X_c_Xhat(self, *linear_results):
        return linear_results[0]
    
    @partial(jax.jit, static_argnums=(0,))
    def DE_gauss(self, x, s, xhat, *args):
        # return jnp.trace(jax.hessian(self.gauss_X_c, argnums=3)(X, S, c, xhat))
        diff = x - xhat
        squared_diff = jnp.sum(diff ** 2)
        sigma = self.sigma(s).squeeze()
        temp =  (squared_diff - self.d * sigma**2) / sigma ** 4
        return  - self.gauss(x, s, xhat) * temp + 3 * args[0] ** 2 * self.gauss(x, s, xhat)
        
    @partial(jax.jit, static_argnums=(0,))
    def DB_gauss(self, x, s, xhat, *args):
        return self.gauss(x, s, xhat)

    
class PDE:
    def __init__(self, alg_opt):
        """
        Initializes the problem setup for a neural network-based Laplacian solver.
        """
        # Problem parameters
        self.name = 'GaussianHighDim'
        self.sigma_min = alg_opt.get('sigma_min', 1e-3)
        self.sigma_max = alg_opt.get('sigma_max', 1.0)

        self.d = alg_opt.get('d', 4)  # spatial dimension
        
        self.scale = alg_opt.get('scale', 1.0) # Domain size
        self.seed = alg_opt.get('seed', 200)
        self.key = jax.random.PRNGKey(self.seed)
        self.pad_size = 16


        self.D = jnp.zeros((self.d, 2))
        # self.D[:, 0] = -1.0
        # self.D[:, 1] = 1.0
        # use jnp syntax to set value
        self.D = self.D.at[:, 0].set(-1.0)
        self.D = self.D.at[:, 1].set(1.0)

        self.vol_D = jnp.prod(self.D[:, 1] - self.D[:, 0])

        self.anisotropic = alg_opt.get('anisotropic', False)
        self.kernel = Kernel(d=self.d, power=self.d+2.01, 
                             mask=(self.scale<1e-8), D=self.D, 
                             anisotropic=self.anisotropic,
                             sigma_max=self.sigma_max, sigma_min=self.sigma_min)
        
        if self.anisotropic:
            self.dim = 2 * self.d # weight dimension
        else:
            self.dim = self.d + 1



        # self.Omega = jnp.array([
        #     [-2.0, 2.0],
        #     [-2.0, 2.0],
        #     [-2.0, 2.0],
        #     [-2.0, 2.0],
        #     [-10.0, 0.0]
        # ])

        self.Omega = jnp.zeros((self.dim, 2))
        # self.Omega[:self.d, 0] = -2.0
        # self.Omega[:self.d, 1] = 2.0
        # self.Omega[self.d:, 0] = -10.0
        # self.Omega[self.d:, 1] = 0.0
        
        self.Omega = self.Omega.at[:self.d, 0].set(-2.0)
        self.Omega = self.Omega.at[:self.d, 1].set(2.0)
        self.Omega = self.Omega.at[self.d:, 0].set(-10.0)
        self.Omega = self.Omega.at[self.d:, 1].set(0.0)

        if self.anisotropic:
            self.Omega = jnp.vstack([self.Omega[:self.d, :], jnp.tile(self.Omega[self.d, :], (self.d, 1))])

        assert self.dim == self.Omega.shape[0] and self.d == self.D.shape[0]


        self.u_zero = {"x": jnp.zeros((self.pad_size, self.d)), "s": jnp.zeros((self.pad_size, self.dim-self.d)),  "u": jnp.zeros((self.pad_size))} # initial solution for anisotropic


        # Observation set
        self.Nobs = alg_opt.get('Nobs', 50)

        self.xhat_int, self.xhat_bnd = self.sample_obs(self.Nobs, method=alg_opt.get('sampling', 'grid'))
        self.xhat = jnp.vstack([self.xhat_int, self.xhat_bnd])
        self.Nx_int = self.xhat_int.shape[0]
        self.Nx_bnd = self.xhat_bnd.shape[0]
        self.Nx = self.Nx_int + self.Nx_bnd
        # Optimization-related attributes
        self.obj = Objective(self.Nx_int, self.Nx_bnd, scale=self.scale)

        self.Ntest = self.Nobs
        self.test_int, self.test_bnd = self.xhat_int, self.xhat_bnd

    
    def f(self, x):
        pass

    def ex_sol(self, x):
        pass

    def sample_obs(self, Nobs, method='grid'):
        """
        Samples observations from D
        method: 'uniform' or 'grid'
        """

        return sample_cube_obs(self.D, Nobs, method=method)

    def sample_param(self, Ntarget):
        """
        Generates Ntarget random parameters in the desired parameter set.
        """
        # randomx = self.Omega[0, 0] + (self.Omega[0, 1] - self.Omega[0, 0]) * np.random.rand(1, Ntarget)
        
        self.key, subkey1, subkey2 = jax.random.split(self.key, 3)  # make sure self.key is updated if reused

        randomx = self.Omega[0, 0] + (self.Omega[:self.d, 1] - self.Omega[:self.d, 0]) * jax.random.uniform(subkey1, (Ntarget, self.d))

        rand_vals = jax.random.uniform(subkey2, (Ntarget, 1))  # shape (Ntarget, 1)
        randoms = self.Omega[-1, 0] + (self.Omega[self.d:, 1] - self.Omega[self.d:, 0]) * jnp.tile(rand_vals, (1, self.dim - self.d))

        return randomx, randoms

    def plot_forward(self, x, s, c, suppc):
        """
        Plots the forward solution.
        """
        pass
        
        
        




