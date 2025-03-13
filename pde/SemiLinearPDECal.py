# import jax.numpy as np
import numpy as np
import matplotlib.pyplot as plt
from src.GaussianKernel import GaussianKernel
# from kernel.GaussianKernel_backup import GaussianKernel
# from src.GaussianKernel_backup import GaussianKernel
from src.utils import Objective, shapeParser, sample_int_obs, sample_bnd_obs
import jax
import jax.numpy as jnp
from functools import partial
jax.config.update("jax_enable_x64", False)

# set the random seed



class Kernel(GaussianKernel):
    def __init__(self, d, power, mask=False, D=None):
        super().__init__(d=d, power=power)
        self.mask = mask
        self.D = D
    # OVERRIDE 1. P_gauss_X_c 2. DP_gauss

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

    @shapeParser
    @partial(jax.jit, static_argnums=(0, 1))
    def P_gauss_X_c(self, X_shape, X, S, c, xhat):
        """
        Compute the PDE operator of the Gaussian kernel.
        """
        return - self.Lap_gauss_X_c(X, S, c, xhat) + self.gauss_X_c(X, S, c, xhat) ** 3
    
    @partial(jax.jit, static_argnums=(0,))
    def DP_gauss(self, x, s, xhat, u):
        return -jnp.trace(jax.hessian(self.gauss, argnums=2)(x, s, xhat)) + 3 * u ** 2 * self.gauss(x, s, xhat)


class ProblemNNLapVar:
    def __init__(self, alg_opt):
        """
        Initializes the problem setup for a neural network-based Laplacian solver.
        """
        # Problem parameters
        self.name = 'SemiLinearPDE'
        self.sigma_min = alg_opt.get('sigma_min', 1e-3)

        self.d = 2  # spatial dimension
        self.dim = self.d + 1 # weight dimension
        
        self.scale = alg_opt.get('scale', 1.0) # Domain size
        self.seed = alg_opt.get('seed', 200)
        np.random.seed(self.seed)


        # Relevant domain for the input weights
        # repeat [-2 * L, 2 * L] self.dim - 1 times, and add the log of sigma_min to the Omega
        # self.Omega = np.array([
        #     [-2.0, 2.0],
        #     [-2.0, 2.0],
        #     [np.log(self.sigma_min), 0]
        # ])
        # self.D = np.array([
        #         [-1., 1.],
        #         [-1., 1.],
        # ])

        self.Omega = np.array([
            [-0.5, 1.5],
            [-0.5, 1.5],
            [np.log(self.sigma_min), 0]
        ])
        self.D = np.array([
                [0., 1.],
                [0., 1.],
        ])
        self.vol_D = np.prod(self.D[:, 1] - self.D[:, 0])

        self.kernel = Kernel(d=self.d, power=self.d+2.01, mask=(self.scale<1e-5), D=self.D)
        self.kernel.sigma_max = alg_opt.get('sigma_max', 1.0)

        self.u_zero = {"x": np.zeros((0, self.d)), "s": np.zeros((0)),  "u": np.zeros((0))} # initial solution

        # Observation set
        self.Nobs = alg_opt.get('Nobs', 50)

        self.xhat_int, self.xhat_bnd = self.sample_obs(self.Nobs, method=alg_opt.get('sampling', 'grid'))
        self.xhat = np.vstack([self.xhat_int, self.xhat_bnd])
        self.Nx_int = self.xhat_int.shape[0]
        self.Nx_bnd = self.xhat_bnd.shape[0]
        self.Nx = self.Nx_int + self.Nx_bnd
        # Optimization-related attributes
        self.obj = Objective(self.Nx_int, self.Nx_bnd, scale=self.scale)

        self.test_int, self.test_bnd = self.sample_obs(100)


    
    def f(self, x):
        x = np.atleast_2d(x)  # Ensures x has shape (N, 2)
        result = (2 * np.pi**2 * np.sin(np.pi * x[:, 0]) * np.sin(np.pi * x[:, 1]) + 
                128 * np.pi**2 * np.sin(4 * np.pi * x[:, 0]) * np.sin(4 * np.pi * x[:, 1]))
        result += self.ex_sol(x) ** 3

        return result if x.shape[0] > 1 else result[0]  # Return scalar if input was (2,)

    def ex_sol(self, x):
        x = np.atleast_2d(x)  # Ensures x has shape (N, 2)
        result = (np.sin(np.pi * x[:, 0]) * np.sin(np.pi * x[:, 1]) +
                4*np.sin(4*np.pi * x[:, 0]) * np.sin(4 * np.pi * x[:, 1]))
        return result if x.shape[0] > 1 else result[0]  # Return scalar if input was (2,)

    
    def sample_obs(self, Nobs, method='grid'):
        """
        Samples observations from D
        method: 'uniform' or 'grid'
        """

        obs_int, obs_bnd = sample_int_obs(self.D, Nobs, method), sample_bnd_obs(self.D, Nobs, method)
        return obs_int, obs_bnd

    # def at_boundary(self, xhat, D=None):
    #     """
    #     return a boolean array indicating whether the points are at the boundary
    #     """
    #     if D is None:
    #         D = self.D
    #     Gamma_D =  np.any(
    #         np.isclose(xhat, D[:, 0]) |  # Compare with lower bound
    #         np.isclose(xhat, D[:, 1]),   # Compare with upper bound
    #         axis=1  # Ensure it checks each row (N,)
    #     )

    #     return Gamma_D

    # def sample_obs(self, Nobs, D=None, method='grid'):
    #     """
    #     Samples observations from D
    #     method: 'uniform' or 'grid'
    #     """

    #     if D is None:
    #         D = self.D
        
    #     if method == 'uniform':
    #         raise NotImplementedError("Uniform sampling not implemented yet.")
    #     elif method == 'grid':
    #         obs = []
    #         for i in range(self.d):
    #             obs.append(np.linspace(D[i, 0], D[i, 1], Nobs))
    #         obs = np.meshgrid(*obs, indexing='ij')
    #         obs = np.vstack([obs[i].flatten() for i in range(self.dim - 1)]).T
        
    #     Gamma_D = self.at_boundary(obs)
    #     obs_int = obs[~Gamma_D, :]
    #     obs_bnd = obs[Gamma_D, :]
    #     return obs_int, obs_bnd

    def sample_param(self, Ntarget):
        """
        Generates Ntarget random parameters in the desired parameter set.
        """
        # randomx = self.Omega[0, 0] + (self.Omega[0, 1] - self.Omega[0, 0]) * np.random.rand(1, Ntarget)
        
        randomx = self.Omega[0, 0] + (self.Omega[:-1, 1] - self.Omega[:-1, 0]) * np.random.rand(Ntarget, self.d)
        randoms = self.Omega[-1, 0] + (self.Omega[-1, 1] - self.Omega[-1, 0]) * np.random.rand(Ntarget)

        return randomx, randoms

    def plot_forward(self, x, s, c):
        """
        Plots the forward solution.
        """
        assert self.dim == 3 

        # # Extract the domain range
        # pO = self.Omega[:-1, :]
        plt.close('all')  # Close previous figure to prevent multiple windows

        # Create a new figure
        fig = plt.figure(figsize=(10, 8))
        ax1 = fig.add_subplot(221, projection='3d')
        ax2 = fig.add_subplot(222, projection='3d')
        ax3 = fig.add_subplot(223)
        ax4 = fig.add_subplot(224)
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
        sigma = 1 / (1 + np.exp(-s))

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
        for xi, yi, r in zip(x[:, 0].flatten(), x[:, 1].flatten(), sigma):
            circle = plt.Circle((xi, yi), r, color='r', fill=False, linestyle='dashed', label="Reference circle")
            ax3.add_patch(circle)
        ax3.set_aspect('equal')  # Ensures circles are properly shaped
        # # set colorbar
        ax3.set_xlim(self.Omega[0, 0], self.Omega[0, 1])
        ax3.set_ylim(self.Omega[1, 0], self.Omega[1, 1])
        ax3.set_title("Collocation Points, Error Contour") 
        fig.colorbar(contour, ax=ax3, shrink=0.5, aspect=5)   
        # # l2 norm of the difference integrated on the domain
        # l2_error = np.sqrt(np.sum((Gu - f1.flatten())**2) * (self.L**2) / (100 * 100))
        # # L_inf norm of the difference
        # L_inf_error = np.max(np.abs(Gu - f1.flatten()))
        # self.epochs.append(epoch)
        # self.L_inf_error.append(L_inf_error)
        # self.L2_error.append(l2_error)
        # ax4.plot(self.epochs, self.L_inf_error, label='L_inf error')
        # ax4.plot(self.epochs, self.L2_error, label='L2 error')
        # ax4.set_xlabel("Epochs")
        # ax4.legend()
        # plt.suptitle(f"L_inf error: {L_inf_error:.4f}, L2 error: {l2_error:.4f}, # points: {u['x'].shape[1]} \n alpha: {alpha}, BNDscale: {self.obj.scale}")
        # plt.tight_layout()
        plt.draw()
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




    

