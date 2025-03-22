from pde.SemiLinearPDE import PDE
from src.solver import solve


import numpy as np
import matplotlib.pyplot as plt

import os
import datetime
import argparse

# write argparse here

parser = argparse.ArgumentParser(description='Run the algorithm to solve PDE problem.')


parser.add_argument('--anisotropic', action='store_true', help='Enable anisotropic mode (default: False)')
parser.add_argument('--sigma_max', type=float, default=1.0, help='Maximum value of the kernel width.')
parser.add_argument('--sigma_min', type=float, default=1e-3, help='Minimum value of the kernel width.')
parser.add_argument('--Nobs', type=int, default=50, help='Base number of observations')
parser.add_argument('--sampling', type=str, default='grid', help='Sampling method for the observations.')
parser.add_argument('--scale', type=int, default=0, help='penalty for the boundary condition')
parser.add_argument('--TOL', type=float, default=1e-5, help='Tolerance for stopping.')
parser.add_argument('--max_step', type=int, default=10000, help='Maximum number of steps.')
parser.add_argument('--print_every', type=int, default=100, help='Print every n steps.')
parser.add_argument('--plot_every', type=int, default=100, help='Plot every n steps.')
parser.add_argument('--insertion_coef', type=float, default=0.1, help='coefficient for thereshold of insertion.') # with metroplis-hasting heuristic insertion coef is not used.
parser.add_argument('--gamma', type=float, default=0, help='gamma.')
parser.add_argument('--alpha', type=float, default=0.001, help='Alpha parameter.')
parser.add_argument('--Ntrial', type=int, default=10000, help='Number of candidate parameters sampled each iter.')
parser.add_argument('--plot_final', type=bool, default=True, help='Plot the final result or not.')
parser.add_argument('--seed', type=int, default=200, help='Random seed for reproductivity.')
parser.add_argument('--index', type=str, default=None, help='index of the configuration to load.')
parser.add_argument('--add_noise', type=bool, default=False, help='Add noise to the rhs.')
parser.add_argument('--save_dir', type=str, default=None, help='Directory to save the output.')
parser.add_argument('--save_idx', type=int, default=None, help='Index to save the output.')
parser.add_argument('--T', type=float, default=300.0, help='Temperature for MCMC.')


args = parser.parse_args()
alg_opts = vars(args)

print(alg_opts)



############################################
### Option 1: Sharp-transition function ####
############################################

R_0 = 0.5
k = 10

# comment out if you want to use the smooth transition function
def ex_sol(x):
    x = np.atleast_2d(x)  # Ensures x has shape (N, 2)
    R = np.sqrt(x[:, 0]**2 + x[:, 1]**2)
    return np.tanh(k * (R_0 - R)) + 1

def f(x):
        x = np.atleast_2d(x)  # Ensures x has shape (N, 2)

        R = np.sqrt(x[:, 0]**2 + x[:, 1]**2) + 1e-8  # Avoid division by zero
        tanh_term = np.tanh(k * (R_0 - R))
        tanh_sq = tanh_term**2
        
        term_x = (-2 * k * x[:, 0]**2 * tanh_term / R**2) + (x[:, 0]**2 / R**3) - (1 / R)
        term_y = (-2 * k * x[:, 1]**2 * tanh_term / R**2) + (x[:, 1]**2 / R**3) - (1 / R)
        result = k * (tanh_sq - 1) * (term_x + term_y)
        result += ex_sol(x) ** 3

        return result if x.shape[0] > 1 else result[0]  # Return scalar if input was (2,)

p = PDE(alg_opts)

p.f = f
p.ex_sol = ex_sol

rhs = p.f(p.xhat)

# optional: add noise to the rhs
if args.add_noise:
    rhs_mag = np.max(np.abs(rhs[:-p.Nx_bnd]))
    noise = np.random.randn(p.Nx) * 0.01 * rhs_mag
    rhs += noise
rhs[-p.Nx_bnd:] = p.ex_sol(p.xhat_bnd)


alg_out = solve(p, rhs, alg_opts)

# post-processing alg_out

assert len(alg_out['xk']) == len(alg_out['sk']) == len(alg_out['ck'])
num_iter = len(alg_out['sk'])
max_supp = max([xk.shape[0] for xk in alg_out['xk']])
xk_padded = np.zeros((num_iter, max_supp, p.d))
sk_padded = np.zeros((num_iter, max_supp, p.dim-p.d))
ck_padded = np.zeros((num_iter, max_supp))

for i in range(num_iter):
    xk_padded[i, :alg_out['xk'][i].shape[0]] = alg_out['xk'][i]
    sk_padded[i, :alg_out['sk'][i].shape[0]] = alg_out['sk'][i]
    ck_padded[i, :alg_out['ck'][i].shape[0]] = alg_out['ck'][i]

alg_out['xk'] = xk_padded
alg_out['sk'] = sk_padded
alg_out['ck'] = ck_padded

# combine alg_out with alg_opts
alg_out.update(alg_opts)

date = datetime.datetime.now().strftime("%m%d_%H%M")


if args.save_dir is None:
    if not os.path.exists(f"output/{p.name}"):
        os.makedirs(f"output/{p.name}")
    np.savez(f"output/{p.name}/out_{date}.npz", **alg_out)
else:
    if not os.path.exists(f"output/{args.save_dir}"):
        os.makedirs(f"output/{args.save_dir}")
    if args.save_idx is not None:
        np.savez(f"output/{args.save_dir}/out_{args.save_idx}.npz", **alg_out)
    else:
        np.savez(f"output/{args.save_dir}/out_{date}.npz", **alg_out)