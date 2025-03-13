# import jax.numpy as np
import numpy as np
import matplotlib.pyplot as plt
from kernel.GaussianKernelAllenCahn import GaussianKernel
from .utils import Objective, sample_int_obs


class ProblemNNLapVar:
    def __init__(self, sigma_min):
        """
        Initializes the problem setup for a neural network-based Laplacian solver.
        """
        # Problem parameters
        self.sigma_min = sigma_min
        self.d = 2  # spatial dimension
        self.dim = self.d + 1 # weight dimension
        self.kernel = GaussianKernel(d=self.d, pow=self.d+1.5)
        self.L = 1.0 # Domain size
        self.T = 0.3

        # Relevant domain for the input weights
        # repeat [-2 * L, 2 * L] self.dim - 1 times, and add the log of sigma_min to the Omega
        self.Omega = np.array([
            [-2*self.L, 2*self.L],
            [-0.5, self.T+0.5],
            [np.log(sigma_min), 0]
        ])

        self.D = np.array([
                [-self.L, self.L],
                [0, self.T],
        ])

        self.vol_D = np.prod(self.D[:, 1] - self.D[:, 0])

        self.u_zero = {"x": np.zeros((0, self.d)), "s": np.zeros((0)),  "u": np.zeros((0))} # empty net to be initial solution
        # Observation set
        self.Nobs = 50
        self.xhat_int, self.xhat_bnd = self.sample_int_obs(self.Nobs), self.sample_bnd_obs(self.Nobs)
        self.xhat = np.vstack([self.xhat_int, self.xhat_bnd])
        self.Nx_int = self.xhat_int.shape[0]
        self.Nx_bnd = self.xhat_bnd.shape[0]
        
        self.Nx = self.Nx_int + self.Nx_bnd
        # Optimization-related attributes
        self.obj = Objective(self.Nx_int, self.Nx_bnd, scale=50)

        self.ex_sol = None

        self.test_int, self.test_bnd = self.sample_int_obs(100), self.sample_bnd_obs(100)

    def sample_int_obs(self, Nobs, D=None, method='grid'):
        """
        Samples observations from D
        method: 'uniform' or 'grid'
        """

        return sample_int_obs(self.D, Nobs, method, eps)

    def sample_bnd_obs(self, Nobs, D=None, method='grid'):
        """
        Samples observations from boundary of D
        method: 'uniform' or 'grid'
        """
        if D is None:
            D = self.D
        
        if method == 'uniform':
            raise NotImplementedError("Uniform sampling not implemented yet.")
        
        elif method == 'grid':
            obs = []

            # 2 sides of the boundary
            left_bound = np.full((Nobs, self.d), D[0, 0])
            left_bound[:, 1] = np.linspace(D[1, 0], D[1, 1], Nobs)
            right_bound = np.full((Nobs, self.d), D[0, 1])
            right_bound[:, 1] = np.linspace(D[1, 0], D[1, 1], Nobs)

            # initial time
            t0 = np.full((Nobs, self.d), D[1, 0])
            t0[:, 0] = np.linspace(D[0, 0], D[0, 1], Nobs)
            t0 = t0[1:-1, :]
        
        return np.vstack([left_bound, right_bound, t0])
    

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

        plt.draw()
        plt.pause(1.0)  # Pause for 0.5 seconds



if __name__ == '__main__':
    # Define parameters
    sigma_min = 0.01
    p = ProblemNNLapVar(sigma_min)
    print(p.dim)
    print(p.xhat.shape)
    print(p.xhat_int.shape)
    print(p.xhat_bnd.shape)
    print(p.u_zero['u'].shape)
    # sample_obs = p.sample_obs(10)
    # import matplotlib.pyplot as plt
    # fig, ax = plt.subplots()
    # ax.scatter(sample_obs[0][:, 0], sample_obs[0][:, 1])
    # ax.scatter(sample_obs[1][:, 0], sample_obs[1][:, 1])
    # plt.show()
    


    # x = np.array([
    #     [0, .5, -.5],
    #     [0, .5, -.5],
    #     [1, 3., 10],
    # ])
    # t_1 = np.linspace(-1, 1, 100)
    # t_2 = np.linspace(-1, 1, 100)
    # # build the meshgrid
    # t1, t2 = np.meshgrid(t_1, t_2)
    # t = np.vstack((t1.flatten(), t2.flatten()))
    # k, dk, gauss = p.k(t, x)       
    # print(k.shape)
    # print(dk.shape)
    # print(gauss.shape)


