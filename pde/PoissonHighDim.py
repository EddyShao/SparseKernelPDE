# import jax.numpy as np
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from src.GaussianKernel import GaussianKernel
# from kernel.GaussianKernel_backup import GaussianKernel
# from src.GaussianKernel_backup import GaussianKernel
from src.utils import Objective, shapeParser, sample_cube_obs
import jax
import jax.numpy as jnp
from functools import partial
jax.config.update("jax_enable_x64", False)

# set the random seed


    
class Kernel(GaussianKernel):
    def __init__(self, d, power, sigma_max, sigma_min, anisotropic=False, mask=False, D=None):
        super().__init__(d=d, power=power, sigma_max=sigma_max, sigma_min=sigma_min, anisotropic=anisotropic)
        self.mask = mask
        self.D = D

        # linear results for computing E and B
        self.linear_E = {
            'Lap': self.Lap_gauss_X_c_Xhat,
        }
        self.linear_B = {'Id': self.gauss_X_c_Xhat}

        # linear results required for computing linearized E and B
        self.DE = [] 
        self.DB = []

    @partial(jax.jit, static_argnums=(0,))
    def gauss(self, x, s, xhat):
        output = super().gauss(x, s, xhat)
        if self.mask:
            mask = jnp.prod(xhat - self.D[:, 0]) * jnp.prod(self.D[:, 1] - xhat)
            output = output * mask
        return output
    
    @shapeParser
    @partial(jax.jit, static_argnums=(0, 1))
    def Lap_gauss_X_c(self, X_shape, X, S, c, xhat):
        # return jnp.trace(jax.hessian(self.gauss_X_c, argnums=3)(X, S, c, xhat))
        diff = X - xhat
        squared_diff = jnp.sum(diff ** 2, axis=1)
        sigma = self.sigma(S).squeeze()
        temp =  (squared_diff - self.d * sigma**2) / sigma ** 4
        lap_phis =  self.gauss_X(X, S, xhat) * temp

        
        return jnp.dot(c, lap_phis)
        

    
    @partial(shapeParser, pad=True)
    @partial(jax.jit, static_argnums=(0, 1))
    def Lap_gauss_X_c_Xhat(self, X_shape, X, S, c, Xhat): 
        return jax.vmap(self.Lap_gauss_X_c, in_axes=(None, None, None, 0))(X, S, c, Xhat)

    @shapeParser
    @partial(jax.jit, static_argnums=(0, 1))
    def E_gauss_X_c(self, X_shape, X, S, c, xhat):
        return - self.Lap_gauss_X_c(X, S, c, xhat) 

    @shapeParser
    @partial(jax.jit, static_argnums=(0, 1))
    def B_gauss_X_c(self, X_shape, X, S, c, xhat):
        return self.gauss_X_c(X, S, c, xhat)
    

    @partial(jax.jit, static_argnums=(0,))
    def E_gauss_X_c_Xhat(self, **linear_results):
        return - linear_results['Lap'] 

    @partial(jax.jit, static_argnums=(0,))
    def B_gauss_X_c_Xhat(self, **linear_results):
        return linear_results['Id']
    
    @partial(jax.jit, static_argnums=(0,))
    def DE_gauss(self, x, s, xhat, *args):
        # return jnp.trace(jax.hessian(self.gauss_X_c, argnums=3)(X, S, c, xhat))
        diff = x - xhat
        squared_diff = jnp.sum(diff ** 2)
        sigma = self.sigma(s).squeeze()
        temp =  (squared_diff - self.d * sigma**2) / sigma ** 4
        return  - self.gauss(x, s, xhat) * temp

        

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
        np.random.seed(self.seed)


        # domain for the input weights
        # self.D = np.array([
        #         [-1., 1.],
        #         [-1., 1.],
        #         [-1., 1.],
        #         [-1., 1.]
        # ])
        # use dimensionality of the problem to set the domain
        self.D = np.zeros((self.d, 2))
        self.D[:, 0] = -1.0
        self.D[:, 1] = 1.0

        self.vol_D = np.prod(self.D[:, 1] - self.D[:, 0])

        self.anisotropic = alg_opt.get('anisotropic', False)
        self.kernel = Kernel(d=self.d, power=self.d+2.01, 
                             mask=(self.scale<1e-8), D=self.D, 
                             anisotropic=self.anisotropic,
                             sigma_max=self.sigma_max, sigma_min=self.sigma_min)
        
        if self.anisotropic:
            self.dim = 2 * self.d # weight dimension
        else:
            self.dim = self.d + 1



        # self.Omega = np.array([
        #     [-2.0, 2.0],
        #     [-2.0, 2.0],
        #     [-2.0, 2.0],
        #     [-2.0, 2.0],
        #     [-10.0, 0.0]
        # ])

        self.Omega = np.zeros((self.dim, 2))
        self.Omega[:self.d, 0] = -2.0
        self.Omega[:self.d, 1] = 2.0
        self.Omega[self.d:, 0] = -10.0
        self.Omega[self.d:, 1] = 0.0
        
        if self.anisotropic:
            self.Omega = np.vstack([self.Omega[:self.d, :], np.tile(self.Omega[self.d, :], (self.d, 1))])

        assert self.dim == self.Omega.shape[0] and self.d == self.D.shape[0]


        self.u_zero = {"x": np.zeros((0, self.d)), "s": np.zeros((0, self.dim-self.d)),  "u": np.zeros((0))} # initial solution for anisotropic


        # Observation set
        self.Nobs = alg_opt.get('Nobs', 50)

        self.xhat_int, self.xhat_bnd = self.sample_obs(self.Nobs, method=alg_opt.get('sampling', 'grid'))
        self.xhat = np.vstack([self.xhat_int, self.xhat_bnd])
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
        
        randomx = self.Omega[0, 0] + (self.Omega[:self.d, 1] - self.Omega[:self.d, 0]) * np.random.rand(Ntarget, self.d)
        randoms = self.Omega[-1, 0] + (self.Omega[self.d:, 1] - self.Omega[self.d:, 0]) * np.tile(np.random.rand(Ntarget)[:, None], (1, self.dim-self.d))

        return randomx, randoms

    def plot_forward(self, x, s, c):
        """
        Plots the forward solution.
        """
        pass
        # # # assert self.dim == 3 

        # # # # Extract the domain range
        # # # pO = self.Omega[:-1, :]
        # plt.close('all')  # Close previous figure to prevent multiple windows

        # # Create a new figure
        # fig = plt.figure(figsize=(5, 5))
        # ax1 = fig.add_subplot(111)
        # t_x = np.linspace(self.D[0, 0], self.D[0, 1], 100)
        # # extend this to d-dimensions, by adding d - 1 zeros
        # t = np.zeros((100, self.d))
        # t[:, 0] = t_x

        # f1 = self.ex_sol(t)
        # # Plot exact solution

        # ax1.plot(t_x, f1, label="Exact Solution")
        

        # # Compute predicted solution
        # Gu = self.kernel.gauss_X_c_Xhat(x, s, c, t)
        # # sigma is sigmoid of S
        # ax1.plot(t_x, Gu, label="Predicted Solution")


        # sigma = 1 / (1 + np.exp(-s))
        # # plot all collocation point X
        # # together with error countour plot

        # ax1.legend()
        # plt.show(block=False)
        # plt.pause(0.1)
        
        
        




