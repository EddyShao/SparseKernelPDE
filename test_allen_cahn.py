# from setup_problems_Lap_var import ProblemNNLapVar, Phi
from AllenCahn import ProblemNNLapVar, Phi
from solve_TV_CGNAP import solve_TV_CGNAP

import matplotlib.pyplot as plt

# import jax.numpy as np
import numpy as np

p = ProblemNNLapVar(sigma_min=1e-5)
EPSILON = p.kernel.epsilon



def ex(x):
    x = np.atleast_2d(x)  # Ensures x has shape (N, 2)
    result = np.exp(-((np.pi**2 / (4 * p.L**2)) + EPSILON**-2) * x[:, 1]) * np.sin(np.pi * (x[:, 0] + p.L) / (2 * p.L))
    return result if x.shape[0] > 1 else result[0]  # Return scalar if input was (2,)


p.ex_sol = ex



# Data vector for the loss function
rhs = np.zeros(p.Nx)
# raise ValueError("Stop here")
rhs[-p.Nx_bnd:] = p.ex_sol(p.xhat[-p.Nx_bnd:])



alpha = .01
alg_opts = {}
gamma = 1
phi = Phi(gamma)
alg_opts = {
    "TOL": 1e-4,
    "max_step": 30000,
    "print_every": 100,
    "plot_every": 100,
    "blocksize": 100
}



solve_TV_CGNAP(p, rhs, alpha, phi, alg_opts)
