# import jax.numpy as np
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from src.Kernels import GaussianKernel
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
        self.linear_E = (self.kappa_X_c_Xhat, self.Lap_kappa_X_c_Xhat)
        self.linear_B = (self.kappa_X_c_Xhat,)
        self.DE = (0,) 
        self.DB = ()  



    @partial(jax.jit, static_argnums=(0,))
    def kappa(self, x, s, xhat):
        output = super().kappa(x, s, xhat)
        if self.mask:
            mask = jnp.prod(xhat - self.D[:, 0]) * jnp.prod(self.D[:, 1] - xhat)
            output = output * mask
        return output
    
    @partial(jax.jit, static_argnums=(0,))
    def Lap_kappa_X_c(self, X, S, c, xhat):
        return jnp.trace(jax.hessian(self.kappa_X_c, argnums=3)(X, S, c, xhat))
    
    @partial(jax.jit, static_argnums=(0,))
    def Lap_kappa_X_c_Xhat(self, X, S, c, Xhat): 
        return jax.vmap(self.Lap_kappa_X_c, in_axes=(None, None, None, 0))(X, S, c, Xhat)

    @partial(jax.jit, static_argnums=(0,))
    def E_kappa_X_c(self, X, S, c, xhat):
        return - self.Lap_kappa_X_c(X, S, c, xhat) + self.kappa_X_c(X, S, c, xhat) ** 3

    @partial(jax.jit, static_argnums=(0,))
    def B_kappa_X_c(self, X, S, c, xhat):
        return self.kappa_X_c(X, S, c, xhat)
    

    def E_kappa_X_c_Xhat(self, *linear_results):
        return - linear_results[1] + linear_results[0] ** 3

    def B_kappa_X_c_Xhat(self, *linear_results):
        return linear_results[0]

    @partial(jax.jit, static_argnums=(0,))
    def DE_kappa(self, x, s, xhat, *args):
            return -jnp.trace(jax.hessian(self.kappa, argnums=2)(x, s, xhat)) + 3 * args[0] ** 2 * self.kappa(x, s, xhat)

    @partial(jax.jit, static_argnums=(0,))
    def DB_kappa(self, x, s, xhat, *args):
        return self.kappa(x, s, xhat)

    
class PDE:
    def __init__(self, alg_opt):
        """
        Initializes the problem setup for a neural network-based Laplacian solver.
        """
        # Problem parameters
        self.name = 'SemiLinear1D'
        self.sigma_min = alg_opt.get('sigma_min', 1e-3)
        self.sigma_max = alg_opt.get('sigma_max', 1.0)

        self.d = 1  # spatial dimension
        
        self.scale = alg_opt.get('scale', 1.0) # Domain size
        self.seed = alg_opt.get('seed', 200)
        self.key = jax.random.PRNGKey(self.seed)


        # domain for the input weights
        self.D = jnp.array([
                [-1., 1.],
        ])

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


        self.Omega = jnp.array([
            [-2.0, 2.0],
            [-10.0, 0.0],
        ])
        
        if self.anisotropic:
            self.Omega = jnp.vstack([self.Omega[:self.d, :], jnp.tile(self.Omega[self.d, :], (self.d, 1))])


        self.pad_size = 16
        self.u_zero = {"x": jnp.zeros((self.pad_size, self.d)),
                       "s": jnp.zeros((self.pad_size, self.dim-self.d)),  
                       "u": jnp.zeros((self.pad_size))} # initial solution for anisotropic


        # Observation set
        self.Nobs = alg_opt.get('Nobs', 50)

        self.xhat_int, self.xhat_bnd = self.sample_obs(self.Nobs, method=alg_opt.get('sampling', 'grid'))
        self.xhat = jnp.vstack([self.xhat_int, self.xhat_bnd])
        self.Nx_int = self.xhat_int.shape[0]
        self.Nx_bnd = self.xhat_bnd.shape[0]
        self.Nx = self.Nx_int + self.Nx_bnd
        # Optimization-related attributes
        self.obj = Objective(self.Nx_int, self.Nx_bnd, scale=self.scale)
        self.Ntest = 200

        self.test_int, self.test_bnd = self.sample_obs(self.Ntest)
    
    def f(self, x):
        pass

    def ex_sol(self, x):
        pass

    
    def sample_obs(self, Nobs, method='grid'):
        """
        Samples observations from D
        method: 'uniform' or 'grid'
        """
        if method == 'grid':
            obs_int = jnp.linspace(self.D[0, 0], self.D[0, 1], Nobs)[1:-1]
            obs_int = obs_int.reshape(-1, 1)
        elif method == 'uniform':
            self.key, subkey = jax.random.split(self.key)
            obs_int = self.D[0, 0] + (self.D[0, 1] - self.D[0, 0]) * jax.random.uniform(
                subkey, shape=(Nobs - 2, 1)
            )
        else:
            raise ValueError("Invalid method")
        obs_bnd = jnp.array([
            [1.],
            [-1.],
        ])
        return obs_int, obs_bnd

    def sample_param(self, Ntarget):
        """
        Generates Ntarget random parameters in the desired parameter set.
        """

        self.key, subkey1, subkey2 = jax.random.split(self.key, 3)
        randomx = self.Omega[0, 0] + (self.Omega[:self.d, 1] - self.Omega[:self.d, 0]) * jax.random.uniform(
            subkey1, shape=(Ntarget, self.d)
        )
        randoms = self.Omega[-1, 0] + (self.Omega[self.d:, 1] - self.Omega[self.d:, 0]) * jnp.tile(
            jax.random.uniform(subkey2, shape=(Ntarget, 1)), (1, self.dim - self.d)
        )

        return randomx, randoms

    def plot_forward(self, x, s, c, suppc):
        """
        Plots the forward solution.
        """
        # assert self.dim == 3 

        # # Extract the domain range
        # pO = self.Omega[:-1, :]
        plt.close('all')  # Close previous figure to prevent multiple windows

        # Create a new figure
        fig = plt.figure(figsize=(5, 5))
        ax1 = fig.add_subplot(111)
        t_x = np.linspace(self.D[0, 0], self.D[0, 1], 100)
        # extend this to d-dimensions, by adding d - 1 zeros
        t = np.zeros((100, self.d))
        t[:, 0] = t_x

        f1 = self.ex_sol(t)
        # Plot exact solution

        ax1.plot(t_x, f1, label="Exact Solution")
    
        # Compute predicted solution
        Gu = self.kernel.kappa_X_c_Xhat(x, s, c, t)
        # sigma is sigmoid of S
        ax1.plot(t_x, Gu, label="Predicted Solution")
        sigma = self.kernel.sigma(s).flatten()
        # plot all collocation point X
        # together with error countour plot


        plt.show(block=False)
        plt.pause(1.0)  




