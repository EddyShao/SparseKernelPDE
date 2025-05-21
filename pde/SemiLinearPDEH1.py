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
from jax import lax
from functools import partial
jax.config.update("jax_enable_x64", True)

# set the random seed


    
class Kernel(GaussianKernel):
    def __init__(self, d, power, sigma_max, sigma_min, anisotropic=False, mask=False, D=None):
        super().__init__(d=d, power=power, sigma_max=sigma_max, sigma_min=sigma_min, anisotropic=anisotropic)
        self.mask = mask
        self.D = D

        # linear results for computing E and B
        self.linear_E = {
            'Id': self.gauss_X_c_Xhat,
            'Lap': self.Lap_gauss_X_c_Xhat,
        }

        self.linear_B = {
            'Id': self.gauss_X_c_Xhat,
            'nabla_n':self.nabla_n_gauss_X_c_Xhat,
        }

        # linear results required for computing linearized E and B
        self.DE = ['Id'] 
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
        return jnp.trace(jax.hessian(self.gauss_X_c, argnums=3)(X, S, c, xhat))
    
    @partial(shapeParser, pad=True)
    @partial(jax.jit, static_argnums=(0, 1))
    def Lap_gauss_X_c_Xhat(self, X_shape, X, S, c, Xhat): 
        return jax.vmap(self.Lap_gauss_X_c, in_axes=(None, None, None, 0))(X, S, c, Xhat)
    
    @shapeParser
    @partial(jax.jit, static_argnums=(0, 1))
    def nabla_n_gauss_X_c(self, X_shape, X, S, c, xhat):
        # Compute the gradient of B_gauss_X_c with respect to xhat
        nabla = jax.grad(self.gauss_X_c, argnums=3)(X, S, c, xhat)

        # Determine if xhat is on the boundary in the first coordinate (axis 0)
        # and in the second coordinate (axis 1)
        # (Assuming you know the boundary is at some predefined value, say +/- L)
        # You can adjust the tolerance as needed
        tol = 1e-6
        is_boundary_first = jnp.any(jnp.isclose(xhat[0], jnp.array(self.D[0, :]), atol=tol))

        # Define direction based on which axis is on the boundary
        # Default direction is (0, 1) if on first coordinate boundary
        # Otherwise (1, 0) if on second coordinate boundary
        direction = lax.cond(
            is_boundary_first,
            lambda _: jnp.array([0.0, 1.0]),
            lambda _: jnp.array([1.0, 0.0]),
            operand=None
        )

        return jnp.dot(nabla, direction)
    
    @partial(shapeParser, pad=True)
    @partial(jax.jit, static_argnums=(0, 1))
    def nabla_n_gauss_X_c_Xhat(self, X_shape, X, S, c, Xhat):
        return jax.vmap(self.nabla_n_gauss_X_c, in_axes=(None, None, None, 0))(X, S, c, Xhat)
    

    @shapeParser
    @partial(jax.jit, static_argnums=(0, 1))
    def E_gauss_X_c(self, X_shape, X, S, c, xhat):
        return - self.Lap_gauss_X_c(X, S, c, xhat) + self.gauss_X_c(X, S, c, xhat) ** 3

    @shapeParser
    @partial(jax.jit, static_argnums=(0, 1))
    def B_gauss_X_c(self, X_shape, X, S, c, xhat):
        return self.gauss_X_c(X, S, c, xhat)
    
    @shapeParser
    @partial(jax.jit, static_argnums=(0, 1))
    def B_aux_gauss_X_c(self, X_shape, X, S, c, xhat):
        return self.nabla_n_gauss_X_c(X, S, c, xhat)

    @partial(jax.jit, static_argnums=(0,))
    def E_gauss_X_c_Xhat(self, **linear_results):
        return - linear_results['Lap'] + linear_results['Id'] ** 3

    @partial(jax.jit, static_argnums=(0,))
    def B_gauss_X_c_Xhat(self, **linear_results):
        return linear_results['Id']
    
    @partial(jax.jit, static_argnums=(0,))
    def B_aux_gauss_X_c_Xhat(self, **linear_results):
        return linear_results['nabla_n']
    
    @shapeParser
    @partial(jax.jit, static_argnums=(0, 1))
    def Grad_B_aux_gauss_X_c(self, X_shape, X, S, c, xhat):
        grads = jax.grad(self.B_aux_gauss_X_c, argnums=(0, 1, 2))(X, S, c, xhat)
        return {'grad_X': grads[0], 'grad_S': grads[1], 'grad_c': grads[2]}
    
    @partial(shapeParser, pad=True)
    @partial(jax.jit, static_argnums=(0, 1))
    def Grad_B_aux_gauss_X_c_Xhat(self, X_shape, X, S, c, Xhat):
        return jax.tree_util.tree_map(lambda g: jax.vmap(lambda xh: g(X, S, c, xh))(Xhat), self.Grad_B_aux_gauss_X_c)
    
    @partial(jax.jit, static_argnums=(0,))
    def DE_gauss(self, x, s, xhat, *args):
        return -jnp.trace(jax.hessian(self.gauss, argnums=2)(x, s, xhat)) + 3 * args[0] ** 2 * self.gauss(x, s, xhat)

    @partial(jax.jit, static_argnums=(0,))
    def DB_gauss(self, x, s, xhat, *args):
        return self.gauss(x, s, xhat)
    
    @partial(jax.jit, static_argnums=(0,))
    def DB_aux_gauss(self, x, s, xhat, *args):
        nabla = jax.grad(self.gauss, argnums=2)(x, s, xhat)
        tol = 1e-6
        is_boundary_first = jnp.any(jnp.isclose(xhat[0], jnp.array(self.D[0, :]), atol=tol))

        # Define direction based on which axis is on the boundary
        # Default direction is (0, 1) if on first coordinate boundary
        # Otherwise (1, 0) if on second coordinate boundary
        direction = lax.cond(
            is_boundary_first,
            lambda _: jnp.array([0.0, 1.0]),
            lambda _: jnp.array([1.0, 0.0]),
            operand=None
        )

        return jnp.dot(nabla, direction)
    
    @partial(jax.jit, static_argnums=(0,))
    def DB_aux_gauss_X(self, X, S, xhat, *args):
        return jax.vmap(self.DB_aux_gauss, in_axes=(0, 0, None) + (None,)*len(args))(X, S, xhat, *args)

    @partial(jax.jit, static_argnums=(0,))
    def DB_aux_gauss_X_Xhat(self, X, S, Xhat, **linear_results):
        """
        Compute the linearized PDE operator of the Gaussian kernel at u.
        """
        args = []
        for key in self.DB:
            args.append(linear_results[key])
            
        return jax.vmap(self.DB_aux_gauss_X, in_axes=(None, None, 0)+(0,)*len(args))(X, S, Xhat, *args)

    
class PDE:
    def __init__(self, alg_opt):
        """
        Initializes the problem setup for a neural network-based Laplacian solver.
        """
        # Problem parameters
        self.name = 'SemiLinearPDEH1'
        self.sigma_min = alg_opt.get('sigma_min', 1e-3)
        self.sigma_max = alg_opt.get('sigma_max', 1.0)

        self.d = 2  # spatial dimension
        
        
        self.scale = alg_opt.get('scale', 1.0) # Domain size
        self.seed = alg_opt.get('seed', 200)
        np.random.seed(self.seed)


        # domain for the input weights
        self.D = np.array([
                [-1., 1.],
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
            [-2.0, 2.0],
            [-2.0, 2.0],
            [-10.0, 0.0],
        ])
        
        if self.anisotropic:
            self.Omega = np.vstack([self.Omega[:self.d, :], np.tile(self.Omega[self.d, :], (self.d, 1))])

        assert self.dim == self.Omega.shape[0] and self.d == self.Omega.shape[1]


        self.u_zero = {"x": np.zeros((0, self.d)), "s": np.zeros((0, self.dim-self.d)),  "u": np.zeros((0))} # initial solution for anisotropic


        # Observation set
        self.Nobs = alg_opt.get('Nobs', 50)

        self.xhat_int, self.xhat_bnd = self.sample_obs(self.Nobs, method=alg_opt.get('sampling', 'grid'))
        self.xhat = np.vstack([self.xhat_int, self.xhat_bnd])
        self.Nx_int = self.xhat_int.shape[0]
        self.Nx_bnd = self.xhat_bnd.shape[0]
        self.Nx = self.Nx_int + self.Nx_bnd
        # Optimization-related attributes
        self.obj = Objective(self.Nx_int, 2*self.Nx_bnd, scale= 2*self.scale)
        
        self.Ntest = 100
        self.test_int, self.test_bnd = self.sample_obs(self.Ntest, method='grid')
        self.obj_test = Objective(self.test_int.shape[0], 2*self.test_bnd.shape[0], scale=2*self.scale)
    
    def f(self, x):
        x = np.atleast_2d(x)  # Ensures x has shape (N, 2)
        result = (2 * np.pi**2 * np.sin(np.pi * x[:, 0]) * np.sin(np.pi * x[:, 1]) + 
                 16 * np.pi**2 * np.sin(2 * np.pi * x[:, 0]) * np.sin(2 * np.pi * x[:, 1]))
        result += self.ex_sol(x) ** 3

        return result if x.shape[0] > 1 else result[0]  # Return scalar if input was (2,)

    def ex_sol(self, x):
        x = np.atleast_2d(x)  # Ensures x has shape (N, 2)
        result = (np.sin(np.pi * x[:, 0]) * np.sin(np.pi * x[:, 1]) +
                2*np.sin(2*np.pi * x[:, 0]) * np.sin(2 * np.pi * x[:, 1]))
        return result if x.shape[0] > 1 else result[0]  # Return scalar if input was (2,)

    
    def sample_obs(self, Nobs, method='grid'):
        """
        Samples observations from D
        method: 'uniform' or 'grid'
        """

        obs_int, obs_bnd = sample_cube_obs(self.D, Nobs, method=method)
        return obs_int, obs_bnd

    def sample_param(self, Ntarget):
        """
        Generates Ntarget random parameters in the desired parameter set.
        """
        # randomx = self.Omega[0, 0] + (self.Omega[0, 1] - self.Omega[0, 0]) * np.random.rand(1, Ntarget)
        
        randomx = self.Omega[:self.d, 0] + (self.Omega[:self.d, 1] - self.Omega[:self.d, 0]) * np.random.rand(Ntarget, self.d)
        randoms = self.Omega[-1, 0] + (self.Omega[self.d:, 1] - self.Omega[self.d:, 0]) * np.tile(np.random.rand(Ntarget)[:, None], (1, self.dim-self.d))

        return randomx, randoms

    def plot_forward(self, x, s, c):
        """
        Plots the forward solution.
        """
        # assert self.dim == 3 

        # # Extract the domain range
        # pO = self.Omega[:-1, :]
        plt.close('all')  # Close previous figure to prevent multiple windows

        # Create a new figure
        fig = plt.figure(figsize=(15, 5))
        ax1 = fig.add_subplot(131, projection='3d')
        ax2 = fig.add_subplot(132, projection='3d')
        ax3 = fig.add_subplot(133)
        # Manually set figure position on screen
        # try:
        #     fig_manager = plt.get_current_fig_manager()
        #     fig_manager.window.wm_geometry("+100+100")  # Move window to (100, 100)
        # except:
        #     pass  # Some backends (e.g., inline Jupyter) may not support this

        # Generate grid
        t_x = np.linspace(self.D[0, 0], self.D[0, 1], 100)
        t_y = np.linspace(self.D[1, 0], self.D[1, 1], 100)
        X, Y = np.meshgrid(t_x, t_y)
        t = np.vstack((X.flatten(), Y.flatten())).T

        if self.ex_sol is not None:
            f1 = self.ex_sol(t).reshape(X.shape)
        # Plot exact solution
        surf1 = ax1.plot_surface(X, Y, f1, cmap='viridis', edgecolor='none')
        ax1.set_title("Exact Solution")
        ax1.set_xlabel("X-axis")
        ax1.set_ylabel("Y-axis")
        fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=5)

        # Compute predicted solution
        Gu = self.kernel.gauss_X_c_Xhat(x, s, c, t)
        # sigma is sigmoid of S
        sigma = self.kernel.sigma(s)

        # Plot predicted solution
        surf2 = ax2.plot_surface(X, Y, Gu.reshape(X.shape), cmap='viridis', edgecolor='none')
        ax2.set_title("Predicted Solution") 
        ax2.set_xlabel("X-axis")
        ax2.set_ylabel("Y-axis")
        ax2.set_zlabel("$f_2(x, y)$")
        fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=5)


        # plot all collocation point X
        # together with error countour plot
        contour = ax3.contourf(X, Y, np.abs(Gu.reshape(100, 100) - f1), cmap='viridis')        
        ax3.scatter(x[:, 0].flatten(), x[:, 1].flatten(), color='r', marker='x')
        if self.anisotropic:
            for xi, yi, ai, bi in zip(x[:, 0].flatten(), x[:, 1].flatten(), sigma[:, 0].flatten(), sigma[:, 1].flatten()):
                ellipse = patches.Ellipse((xi, yi), width=2*ai, height=2*bi,
                              edgecolor='r', facecolor='none',
                              linestyle='dashed', label="Reference ellipse")
                ax3.add_patch(ellipse)
        else:
            for xi, yi, r in zip(x[:, 0].flatten(), x[:, 1].flatten(), sigma.flatten()):
                circle = plt.Circle((xi, yi), r, color='r', fill=False, linestyle='dashed', label="Reference circle")
                ax3.add_patch(circle)

        ax3.set_aspect('equal')  # Ensures circles are properly shaped
        # # set colorbars
        ax3.set_xlim(self.Omega[0, 0], self.Omega[0, 1])
        ax3.set_ylim(self.Omega[1, 0], self.Omega[1, 1])
        ax3.set_title("Collocation Points, Error Contour") 
        fig.colorbar(contour, ax=ax3, shrink=0.5, aspect=5)   

        plt.show(block=False)
        plt.pause(1.0)  



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




    

