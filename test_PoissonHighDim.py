from pde.PoissonHighDim import PDE
# from src.solver import solve
from src.solver_active import solve


import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp
import jax

import os
import datetime
import argparse

# write argparse here

parser = argparse.ArgumentParser(description='Run the algorithm to solve PDE problem.')

parser.add_argument('--anisotropic', action='store_true', help='Enable anisotropic mode (default: False)')
parser.add_argument('--d', type=int, default=2, help='Dimension of the problem.')
parser.add_argument('--sigma_max', type=float, default=1.0, help='Maximum value of the kernel width.')
parser.add_argument('--sigma_min', type=float, default=1e-3, help='Minimum value of the kernel width.')
parser.add_argument('--blocksize', type=int, default=10000, help='Block size for the anisotropic mode.')
parser.add_argument('--Nobs', type=int, default=10, help='Base number of observations')
parser.add_argument('--sampling', type=str, default='grid', help='Sampling method for the observations.')
parser.add_argument('--scale', type=float, default=0, help='penalty for the boundary condition')
parser.add_argument('--TOL', type=float, default=1e-5, help='Tolerance for stopping.')
parser.add_argument('--max_step', type=int, default=3000, help='Maximum number of steps.')
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
p.name = 'GaussianHighDim'

e = jnp.ones(p.d)
# e = e.at[0].set(1.0)
# e = e.at[2].set(-1.0)
# e = e.at[3].set(1.0)



# def ex_sol_help(x, center):
#     x = np.atleast_2d(x)
#     # use vmap to vectorize the function
#     inner = np.dot(x, e)
#     result = np.exp(-(inner-center) ** 2)
#     return result

# def f_help(x, center):
#     x = np.atleast_2d(x)
#     # use vmap to vectorize the function
#     inner = np.dot(x, e)
#     lap = (4 * (inner - center)**2 - 2) * ex_sol_help(x, center) * np.dot(e, e)
#     return - lap

# def ex_sol(x):
#     x = np.atleast_2d(x)
#     # use vmap to vectorize the function
#     inner = np.dot(x, e)
#     result = np.sin(np.pi * inner)
#     return result

# def f(x):
#     x = np.atleast_2d(x)
#     # use vmap to vectorize the function
#     inner = np.dot(x, e)
#     lap = - (np.pi**2) * ex_sol(x) * np.dot(e, e)
#     return - lap

# center_1 = 0.30
# center_2 = -0.30

# p.ex_sol = ex_sol
# p.f = f

def ex_sol(x):
    x = np.atleast_2d(x)
    result = np.prod(np.sin(np.pi * x), axis=1) 
    return result if len(result) > 1 else result[0]

def f(x):
    return p.d * np.pi**2 * ex_sol(x)




# def ex_sol(x):
#     x = np.atleast_2d(x)
#     # use vmap to vectorize the function
#     inner = np.dot(x, e)
#     result = np.tanh(20 * (inner + 0.3)) - np.tanh(20 * (inner - 0.3))
#     return result


# def f(x):
#     x = np.atleast_2d(x)
#     inner = np.dot(x, e)

#     # First and second derivatives of tanh
#     sech_sq_1 = 1 / np.cosh(20 * (inner + 0.3)) ** 2
#     sech_sq_2 = 1 / np.cosh(20 * (inner - 0.3)) ** 2

#     tanh_1 = np.tanh(20 * (inner + 0.3))
#     tanh_2 = np.tanh(20 * (inner - 0.3))

#     lap = -800 * (tanh_1 * sech_sq_1 - tanh_2 * sech_sq_2)

#     return -lap if x.shape[0] > 1 else -lap[0]


# @jax.jit
# def ex_sol_1d(x):
#     return 2*jnp.exp(-x**2)

# @jax.jit
# def f_1d(x):
#     hess_fn = jax.hessian(ex_sol_1d)
#     return -hess_fn(x)  # hessian is scalar since input is 1D

# def ex_sol(x):
#     x = np.atleast_2d(x)
#     inner = np.dot(x, e).flatten()
#     result = jax.vmap(ex_sol_1d)(inner)
#     return np.array(result) if result.shape[0] > 1 else float(result[0])

# def f(x):
#     x = np.atleast_2d(x)
#     inner = np.dot(x, e).flatten()
#     result = jax.vmap(f_1d)(inner)
#     result *= np.dot(e, e)
#     return np.array(result) if result.shape[0] > 1 else float(result[0])

# p.f = f
# p.ex_sol = ex_sol




# def ex_sol(x):
#     x = np.atleast_2d(x)
#     # use vmap to vectorize the function
#     inner = np.dot(x, e)
#     return inner ** 2 - 2 * inner **1 + 1

# def f(x):
#     x = np.atleast_2d(x)
#     # use vmap to vectorize the function
#     inner = np.dot(x, e)
#     norm_square_e = np.dot(e, e)
#     return 2 * norm_square_e * np.ones_like(inner) 

p.f = f
p.ex_sol = ex_sol
rhs = p.f(p.xhat)

# optional: add noise to the rhs
if args.add_noise:
    rhs_mag = np.max(np.abs(rhs[:-p.Nx_bnd]))
    noise = np.random.randn(p.Nx) * 0.01 * rhs_mag
    rhs += noise

rhs[-p.Nx_bnd:] = p.ex_sol(p.xhat_bnd)



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
