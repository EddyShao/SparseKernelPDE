from pde.SemiLinear1D import PDE
# from src.solver import solve
from src.solver_active import solve
# from src.solver_first_order import solve


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

c1, c2 = -0.0, 0.3
k1, k2 = 1, 6
R1, R2 = 0.3, 0.15


# def ex_sol_help(x, c, k, R):
#     z = k**2 * (R**2 - (x - c)**2)
#     return np.tanh(z).flatten()

# def f_help(x, c, k, R):
#     z = k**2 * (R**2 - (x - c)**2)
#     sech2 = 1 / np.cosh(z)**2
#     term1 = -2 * k**2 * sech2
#     term2 = 4 * k**4 * (x - c)**2 * np.tanh(z) * sech2
#     results = term1 + term2
#     return results.flatten()

# def ex_sol(x):
#     return ex_sol_help(x, c1, k1, R1) 

# def f(x):
#     return - (f_help(x, c1, k1, R1) ) + ex_sol(x)**3

# def two_valley_function(x, c1=-0.5, c2=0.5, k1=5, k2=20, a1=1.0, a2=1.0):
#     valley1 = a1 * np.exp(-k1 * (x - c1)**2)
#     valley2 = a2 * np.exp(-k2 * (x - c2)**2)
#     return valley1 + valley2

# def ex_sol(x):
#     return two_valley_function(x, c1=-0.5, c2=0.5, k1=5, k2=20, a1=1.0, a2=1.0).flatten()

# def laplacian_two_valley(x, c1=-0.5, c2=0.5, k1=5, k2=20, a1=1.0, a2=1.0):
#     term1 = -2 * a1 * k1 * np.exp(-k1 * (x - c1)**2) * (1 - 2 * k1 * (x - c1)**2)
#     term2 = -2 * a2 * k2 * np.exp(-k2 * (x - c2)**2) * (1 - 2 * k2 * (x - c2)**2)
#     rerults = term1 + term2
#     return rerults.flatten()

# def f(x):
#     return - laplacian_two_valley(x, c1=-0.5, c2=0.5, k1=5, k2=20, a1=1.0, a2=1.0) + ex_sol(x)**3

def tanh_valley(x, c, d, k, a):
    return -a * (np.tanh(k * (x - c)) - np.tanh(k * (x - d)))

def laplacian_tanh_valley(x, c, d, k, a):
    term_c = np.tanh(k * (x - c)) * (1 / np.cosh(k * (x - c)))**2
    term_d = np.tanh(k * (x - d)) * (1 / np.cosh(k * (x - d)))**2
    return 2 * a * k**2 * (term_c - term_d)

def tanh_two_valleys(x):
    v1 = tanh_valley(x, c=-1.0, d=-0.2, k=10, a=1.0)
    v2 = tanh_valley(x, c=0.3, d=0.4, k=30, a=0.5)
    return v1 + v2

def laplacian_tanh_two_valleys(x):
    l1 = laplacian_tanh_valley(x, c=-1.0, d=-0.2, k=10, a=1.0)
    l2 = laplacian_tanh_valley(x, c=0.3, d=0.4, k=30, a=0.5)
    return l1 + l2

def ex_sol(x):
    return tanh_two_valleys(x).flatten()

def f(x):
    return - (laplacian_tanh_two_valleys(x)).flatten() + ex_sol(x)**3


p = PDE(alg_opts)

p.f = f
p.ex_sol = ex_sol


rhs = p.f(p.xhat).flatten()
print(rhs.shape)

# optional: add noise to the rhs
if args.add_noise:
    rhs_mag = np.max(np.abs(rhs[:-p.Nx_bnd]))
    noise = np.random.randn(p.Nx) * 0.01 * rhs_mag
    rhs += noise
rhs[-p.Nx_bnd:] = p.ex_sol(p.xhat_bnd).flatten()




def evaluate_and_save_solution(p, rhs, alg_opts, args):
    """
    Solves the system, evaluates L_inf and L_2 error, pads solution history,
    and saves the result to file if specified.

    Parameters:
        p: problem definition (should include kernel, test points, exact solution, etc.)
        rhs: right-hand side for the solver
        alg_opts: algorithm options (e.g., tolerances, initialization)
        args: arguments including save_dir and save_idx
    """
    print()
    print('#' * 20)
    print('alpha:', alg_opts['alpha'])
    print('#' * 20)
    print()
    alg_out = solve(p, rhs, alg_opts)

    # Define prediction function
    u_pred = lambda xhat_vec: p.kernel.gauss_X_c_Xhat(
        alg_out['xk'][-1],
        alg_out['sk'][-1],
        alg_out['ck'][-1],
        xhat_vec
    )

    # Compute predictions and errors
    y_pred_bnd = u_pred(p.test_bnd)
    y_true_bnd = p.ex_sol(p.test_bnd)
    y_pred_int = u_pred(p.test_int)
    y_true_int = p.ex_sol(p.test_int)

    L_inf_bnd = np.max(np.abs(y_pred_bnd - y_true_bnd))
    L_inf_int = np.max(np.abs(y_pred_int - y_true_int))
    L_2 = np.sqrt(
        (np.sum((y_pred_int - y_true_int)**2) + np.sum((y_pred_bnd - y_true_bnd)**2))
        * p.vol_D / (p.Ntest ** p.d)
    )
    print()
    print('#' * 20)
    print(f'alpha: {alg_opts["alpha"]:.1e}')
    print(f'L_inf error (boundary): {L_inf_bnd:.2e}')
    print(f'L_inf error (interior): {L_inf_int:.2e}')
    print(f'L_inf error (total): {max(L_inf_bnd, L_inf_int):.2e}')
    print(f'L_2 error: {L_2:.2e}')
    print('#' * 20)
    print()

    # Post-process alg_out
    num_iter = len(alg_out['sk'])
    max_supp = max([xk.shape[0] for xk in alg_out['xk']])

    xk_padded = np.zeros((num_iter, max_supp, p.d))
    sk_padded = np.zeros((num_iter, max_supp, p.dim - p.d))
    ck_padded = np.zeros((num_iter, max_supp))

    for i in range(num_iter):
        xk_padded[i, :alg_out['xk'][i].shape[0]] = alg_out['xk'][i]
        sk_padded[i, :alg_out['sk'][i].shape[0]] = alg_out['sk'][i]
        ck_padded[i, :alg_out['ck'][i].shape[0]] = alg_out['ck'][i]

    alg_out['xk'] = xk_padded
    alg_out['sk'] = sk_padded
    alg_out['ck'] = ck_padded

    # Combine with options and errors
    alg_out.update(alg_opts)
    alg_out['error_all'] = np.array([L_inf_int, L_inf_bnd, L_2])

    # Save output
    if args.save_dir and args.save_idx is not None:
        out_dir = f"output/{p.name}/{args.save_dir}/out_{args.save_idx}"
        os.makedirs(out_dir, exist_ok=True)
        np.savez(f"{out_dir}/out_{args.save_idx}_{alg_opts['alpha']:.0e}.npz", **alg_out)

    return alg_out



alg_out = evaluate_and_save_solution(p, rhs, alg_opts, args)