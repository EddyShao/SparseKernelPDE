from pde.SemiLinearPDE import PDE
# from src.solver import solve
from src.solver_active import solve


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
parser.add_argument('--blocksize', type=int, default=100, help='Block size for the anisotropic mode.')
parser.add_argument('--Nobs', type=int, default=50, help='Base number of observations')
parser.add_argument('--sampling', type=str, default='grid', help='Sampling method for the observations.')
parser.add_argument('--scale', type=float, default=1000, help='penalty for the boundary condition')
parser.add_argument('--TOL', type=float, default=1e-5, help='Tolerance for stopping.')
parser.add_argument('--max_step', type=int, default=5000, help='Maximum number of steps.')
parser.add_argument('--print_every', type=int, default=100, help='Print every n steps.')
parser.add_argument('--plot_every', type=int, default=100, help='Plot every n steps.')
parser.add_argument('--insertion_coef', type=float, default=0.01, help='coefficient for thereshold of insertion.') # with metroplis-hasting heuristic insertion coef is not used.
parser.add_argument('--gamma', type=float, default=0, help='gamma.')
parser.add_argument('--alpha', type=float, default=0.001, help='Alpha parameter.')
parser.add_argument('--Ntrial', type=int, default=10000, help='Number of candidate parameters sampled each iter.')
parser.add_argument('--plot_final', action='store_true', help='Plot the final result.')
parser.add_argument('--seed', type=int, default=200, help='Random seed for reproductivity.')
parser.add_argument('--index', type=str, default=None, help='index of the configuration to load.')
parser.add_argument('--add_noise', type=bool, default=False, help='Add noise to the rhs.')
parser.add_argument('--save_dir', type=str, default=None, help='Directory to save the output.')
parser.add_argument('--save_idx', type=int, default=None, help='Index to save the output.')
parser.add_argument('--T', type=float, default=300.0, help='Temperature for MCMC.')


args = parser.parse_args()
alg_opts = vars(args)

print(alg_opts)


p = PDE(alg_opts)
p.name = 'SemiLinearSines'
rhs = p.f(p.xhat)

# optional: add noise to the rhs
if args.add_noise:
    rhs_mag = np.max(np.abs(rhs[:-p.Nx_bnd]))
    noise = np.random.randn(p.Nx) * 0.01 * rhs_mag
    rhs += noise
rhs[-p.Nx_bnd:] = p.ex_sol(p.xhat_bnd)


alg_out = solve(p, rhs, alg_opts)

#### test L_inf and L_2 error again ####

u_pred = lambda xhat_vec: p.kernel.gauss_X_c_Xhat(alg_out['xk'][-1], alg_out['sk'][-1], alg_out['ck'][-1], xhat_vec)

y_pred_bnd = u_pred(p.test_bnd)
y_true_bnd = p.ex_sol(p.test_bnd)
y_pred_int = u_pred(p.test_int)
y_true_int = p.ex_sol(p.test_int)
L_inf_bnd = np.max(np.abs(y_pred_bnd - y_true_bnd))
L_inf_int = np.max(np.abs(y_pred_int - y_true_int))

L_2 = np.sqrt((np.sum((y_pred_int - y_true_int)**2) + np.sum((y_pred_bnd - y_true_bnd)**2)) * p.vol_D / (p.Ntest **p.d))

print('#' * 20)
print(f'L_inf error (boundary): {L_inf_bnd:.2e}')
print(f'L_inf error (interior): {L_inf_int:.2e}')
print(f'L_inf error (total): {max(L_inf_bnd, L_inf_int):.2e}')
print(f'L_2 error: {L_2:.2e}')
print('#' * 20)

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

alg_out['error_all'] = np.array([L_inf_int, L_inf_bnd, L_2]) 

date = datetime.datetime.now().strftime("%m%d_%H%M")


if args.save_dir and args.save_idx:
        if not os.path.exists(f"output/{p.name}/{args.save_dir}"):
            os.makedirs(f"output/{p.name}/{args.save_dir}")
        np.savez(f"output/{p.name}/{args.save_dir}/out_{args.save_idx}.npz", **alg_out)