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
jax.config.update("jax_enable_x64", True)



    
class Kernel(GaussianKernel):
    def __init__(self, d, power, sigma_max, sigma_min, anisotropic=False, mask=False, D=None):
        super().__init__(d=d, power=power, sigma_max=sigma_max, sigma_min=sigma_min, anisotropic=anisotropic)
        self.mask = mask
        self.D = D
        self.nu = 0.02
        self.dt = 0.001
        self.pad_size = 100

        # linear results for computing E and B
        self.linear_E = {
            'Id': self.gauss_X_c_Xhat,
            'D_x': self.D_x_gauss_X_c_Xhat,
            'D_xx': self.D_xx_gauss_X_c_Xhat,
        }
        self.linear_B = {
            'Id': self.gauss_X_c_Xhat,
        } 

        # linear results required for computing linearized E and B
        self.DE = ['Id', 'D_x'] 
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
    def D_x_gauss_X_c(self, X_shape, X, S, c, xhat):
        return jax.grad(self.gauss_X_c, argnums=3)(X, S, c, xhat).squeeze()
    
    @partial(shapeParser, pad=True)
    @partial(jax.jit, static_argnums=(0, 1))
    def D_x_gauss_X_c_Xhat(self, X_shape, X, S, c, Xhat):
        return jax.vmap(self.D_x_gauss_X_c, in_axes=(None, None, None, 0))(X, S, c, Xhat)
    
    @shapeParser
    @partial(jax.jit, static_argnums=(0, 1))
    def D_xx_gauss_X_c(self, X_shape, X, S, c, xhat):
        return jax.hessian(self.gauss_X_c, argnums=3)(X, S, c, xhat)[0, 0]
    
    @partial(shapeParser, pad=True)
    @partial(jax.jit, static_argnums=(0, 1))
    def D_xx_gauss_X_c_Xhat(self, X_shape, X, S, c, Xhat):
        return jax.vmap(self.D_xx_gauss_X_c, in_axes=(None, None, None, 0))(X, S, c, Xhat)

    @shapeParser
    @partial(jax.jit, static_argnums=(0, 1))
    def E_gauss_X_c(self, X_shape, X, S, c, xhat):
        u = self.gauss_X_c(X, S, c, xhat)
        u_x = self.D_x_gauss_X_c(X, S, c, xhat)
        u_xx = self.D_xx_gauss_X_c(X, S, c, xhat)
        return u - self.dt * (self.nu * u_xx - u_x * u)

    @shapeParser
    @partial(jax.jit, static_argnums=(0, 1))
    def B_gauss_X_c(self, X_shape, X, S, c, xhat):
        return self.gauss_X_c(X, S, c, xhat)


    @partial(jax.jit, static_argnums=(0,))
    def E_gauss_X_c_Xhat(self, **linear_results):
        u = linear_results['Id']
        u_x = linear_results['D_x']
        u_xx = linear_results['D_xx']
        return u - self.dt * (self.nu * u_xx - u_x * u)

    @partial(jax.jit, static_argnums=(0,))
    def B_gauss_X_c_Xhat(self, **linear_results):
        return linear_results['Id']
    
    @partial(jax.jit, static_argnums=(0,))
    def DE_gauss(self, x, s, xhat, *args):
        v = self.gauss(x, s, xhat)
        v_x = jax.grad(self.gauss, argnums=2)(x, s, xhat).squeeze()
        v_xx = jax.hessian(self.gauss, argnums=2)(x, s, xhat)[0, 0]
        u = args[0]
        u_x = args[1]

        temp = v_x * u + v * u_x
        return v - self.dt * self.nu * v_xx + self.dt * temp


    @partial(jax.jit, static_argnums=(0,))
    def DB_gauss(self, x, s, xhat, *args):
        return self.gauss(x, s, xhat)
        
    
class PDE:
    def __init__(self, alg_opt):
        """
        Initializes the problem setup for a neural network-based Laplacian solver.
        """
        # Problem parameters
        self.name = 'Burgers1D'
        self.sigma_min = alg_opt.get('sigma_min', 1e-3)
        self.sigma_max = alg_opt.get('sigma_max', 1.0)

        self.d = 1  # spatial dimension
        
        self.scale = alg_opt.get('scale', 1.0) # Domain size
        
        self.seed = alg_opt.get('seed', 200)
        np.random.seed(self.seed)


        # domain for the input weights
        self.D = np.array([
                [-1., 1.],
        ])

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


        self.Omega = np.array([
            [-1.0, 1.0],
            [-10.0, 0.0],
        ])
        

        assert self.dim == self.Omega.shape[0] 


        # self.u_zero = {"x": np.zeros((0, self.d)), "s": np.zeros((0, self.dim-self.d)),  "u": np.zeros((0))} # initial solution for anisotropic
        # initial_x, initial_s = self.sample_param(1000)

        # self.u_zero = {"x": initial_x, "s": initial_s,  "u": (5 * np.ones(1000) + np.random.random(1000)) * np.random.choice([-1, 1], size=1000)} # initial solution for anisotropic
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
        
        self.Ntest = 100
        self.test_int, self.test_bnd = self.sample_obs(self.Ntest, method='grid')
    
    def f(self, x):
        pass

    def ex_sol(self, x):
        pass
    
    def sample_obs(self, Nobs, method='grid'):
        """
        Samples observations from D
        method: 'uniform' or 'grid'
        """

        obs_bnd = np.array([[-1.0], [1.0]])
        if method == 'grid':
            obs_int = np.linspace(-1, 1, Nobs)[1:-1].reshape(-1, 1)
        elif method == 'uniform':
            # use chebyshev nodes
            # obs_int = -1 + 2 * np.cos((np.pi * np.arange(1, Nobs-1) / (Nobs-1)))[:, None]
            obs_int = np.random.uniform(-1, 1, (Nobs-2, 1))

        return obs_int, obs_bnd

    def sample_param(self, Ntarget):
        """
        Generates Ntarget random parameters in the desired parameter set.
        """
        # randomx = self.Omega[0, 0] + (self.Omega[0, 1] - self.Omega[0, 0]) * np.random.rand(1, Ntarget)
        
        randomx = self.Omega[:self.d, 0] + (self.Omega[:self.d, 1] - self.Omega[:self.d, 0]) * np.random.rand(Ntarget, self.d)
        # randoms = self.Omega[-1, 0] + (self.Omega[self.d:, 1] - self.Omega[self.d:, 0]) * np.random.rand(Ntarget, self.dim-self.d)
        randoms = self.Omega[-1, 0] + (self.Omega[self.d:, 1] - self.Omega[self.d:, 0]) * np.tile(np.random.rand(Ntarget)[:, None], (1, self.dim-self.d))

        return randomx, randoms

    def plot_forward(self, x, s, c):
        plt.figure(figsize=(10, 10))
        t = np.linspace(-1, 1, 100)
        y_true = self.ex_sol(t).flatten()
        y_pred = self.kernel.gauss_X_c_Xhat(x, s, c, t.reshape(-1, 1)).flatten()
        sigma = self.kernel.sigma(s).flatten()
        plt.scatter(x, np.zeros_like(x), c='r', label='Support Points')
        plt.plot(t, y_true, label='True')
        plt.plot(t, y_pred, label='Predicted')
        plt.legend()
        plt.draw()
        plt.pause(0.1)
        plt.close()


# if __name__ == '__main__':
#     # Define parameters
#     sigma_min = 0.0001
#     p = ProblemNNLapVar(sigma_min)
#     print(p.dim)
#     print(p.xhat.shape)
#     print(p.xhat_int.shape)
#     print(p.xhat_bnd.shape)
#     print(p.u_zero['u'].shape)
#     # sample_obs = p.sample_obs(10)
#     # import matplotlib.pyplot as plt
#     # fig, ax = plt.subplots()
#     # ax.scatter(sample_obs[0][:, 0], sample_obs[0][:, 1])
#     # ax.scatter(sample_obs[1][:, 0], sample_obs[1][:, 1])
#     # plt.show()
    


#     # x = np.array([
#     #     [0, .5, -.5],
#     #     [0, .5, -.5],
#     #     [1, 3., 10],
#     # ])
#     # t_1 = np.linspace(-1, 1, 100)
#     # t_2 = np.linspace(-1, 1, 100)
#     # # build the meshgrid
#     # t1, t2 = np.meshgrid(t_1, t_2)
#     # t = np.vstack((t1.flatten(), t2.flatten()))
#     # k, dk, gauss = p.k(t, x)       
#     # print(k.shape)
#     # print(dk.shape)
#     # print(gauss.shape)




    

