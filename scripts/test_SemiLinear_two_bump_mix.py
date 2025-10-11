import sys
sys.path.append("./")
from pde.SemiLinearPDE import PDE
from src.utils import Objective, compute_errors, compute_y, compute_rhs, merge_kernel_cluster
import jax.numpy as jnp


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
parser.add_argument('--blocksize', type=int, default=300, help='Block size for the anisotropic mode.')
parser.add_argument('--Nobs', type=int, default=50, help='Base number of observations')
parser.add_argument('--sampling', type=str, default='grid', help='Sampling method for the observations.')
parser.add_argument('--scale', type=float, default=0, help='penalty for the boundary condition')
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
parser.add_argument('--solver', type=str, default='newton', help='Solver to use.')


args = parser.parse_args()
alg_opts = vars(args)

print(alg_opts)

if 'newton' in alg_opts['solver']:
    from src.solver import solve
elif 'first' in alg_opts['solver']:
    from src.solver_first_order import solve
    from src.solver import solve as solve_second_order
    alg_opts['max_step'] = 500
else:
    raise ValueError('Unknown solver type.')

# comment out if you want to use the smooth transition function
def ex_sol_help(x, center=(0.30, 0.30), k=8, R_0=0.2):
    x = jnp.atleast_2d(x)  # Ensures x has shape (N, 2)
    R = jnp.sqrt((x[:, 0] - center[0])**2 + (x[:, 1] - center[1])**2)
    return jnp.tanh(k * (R_0 - R)) + 1

def f_help(x, center=(0.2, 0.30), k=8, R_0=0.2):
    x = jnp.atleast_2d(x)  # Ensures x has shape (N, 2)
    R = jnp.sqrt((x[:, 0] - center[0])**2 + (x[:, 1] - center[1])**2)
    tanh_term = jnp.tanh(k * (R_0 - R))
    tanh_sq = tanh_term**2
    term_x = (-2 * k * (x[:, 0] - center[0])**2 * tanh_term / R**2) + ((x[:, 0] - center[0])**2 / R**3) - (1 / R)
    term_y = (-2 * k * (x[:, 1] - center[1])**2 * tanh_term / R**2) + ((x[:, 1] - center[1])**2 / R**3) - (1 / R)
    result = k * (tanh_sq - 1) * (term_x + term_y)
    return result

#########################################################
####################### two bump ########################
#########################################################

R_1 = 0.3
R_2 = 0.15
center_1 = [0.30, 0.30]
center_2 = [-0.30, -0.30]
k1 = 12
k2 = 4

def ex_sol(x):
    return ex_sol_help(x, center=center_1, k=k1, R_0=R_1) + ex_sol_help(x, center=center_2, k=k2, R_0=R_2)


def f(x):
    return f_help(x, center=center_1, k=k1, R_0=R_1) + f_help(x, center=center_2, k=k2, R_0=R_2) + ex_sol(x) ** 3


p = PDE(alg_opts)

p.f = f
p.ex_sol = ex_sol
p.name = 'SemiLinearTwoBumpAdaptive'
p.kernel.pad_size = 300


rhs = p.f(p.xhat)

# optional: add noise to the rhs
if args.add_noise:
    rhs_mag = jnp.max(jnp.abs(rhs[:-p.Nx_bnd]))
    noise = jnp.random.randn(p.Nx) * 0.01 * rhs_mag
    rhs += noise
# rhs[-p.Nx_bnd:] = p.ex_sol(p.xhat_bnd)
rhs = rhs.at[-p.Nx_bnd:].set(p.ex_sol(p.xhat_bnd))

rhs_test_int = p.f(p.test_int)
rhs_test_bnd = p.ex_sol(p.test_bnd)
rhs_test = jnp.concatenate((rhs_test_int, rhs_test_bnd))


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

    if 'first' in alg_opts['solver']:
        xk  = alg_out['xk'][-1]
        sk  = alg_out['sk'][-1]
        ck  = alg_out['ck'][-1]
        suppc = alg_out['suppc'][-1]
        xk, sk, ck = merge_kernel_cluster(suppc, xk, sk, ck, p, tau=1.0)
        # pad to pad_size
        ck = jnp.pad(ck, (0, p.kernel.pad_size - len(ck)), constant_values=0.)
        sk = jnp.pad(sk, ((0, p.kernel.pad_size - sk.shape[0]), (0, 0)), constant_values=0.)
        xk = jnp.pad(xk, ((0, p.kernel.pad_size - xk.shape[0]), (0, 0)), constant_values=0.)
        # use second order solver to refine the solution
        alg_opts_second = alg_opts.copy()
        alg_opts_second['solver'] = 'newton'
        p.u_zero = {
            'x': jnp.array(xk),
            's': jnp.array(sk),
            'u': jnp.array(ck)
        }
        alg_out = solve_second_order(p, rhs, alg_opts_second)



    rhs_test = np.concatenate((p.f(p.test_int), p.ex_sol(p.test_bnd)))
    p.obj_test = Objective(p.test_int.shape[0], p.test_bnd.shape[0], scale=alg_opts['scale'])

    # compute errors
    errors_test = compute_errors(p, alg_out['xk'][-1], alg_out['sk'][-1], alg_out['ck'][-1],
                            p.test_int, p.test_bnd)
    errors_test = {k+'_test': v for k, v in errors_test.items()}
    errors_train = compute_errors(p, alg_out['xk'][-1], alg_out['sk'][-1], alg_out['ck'][-1], 
                                  p.xhat_int, p.xhat_bnd)
    errors_train = {k+'_train': v for k, v in errors_train.items()}
    
    # compute residue for both train and test
    yk, _, _ = compute_rhs(p, alg_out['xk'][-1], alg_out['sk'][-1], alg_out['ck'][-1], p.xhat_int, p.xhat_bnd)
    misfit = yk - rhs
    residue_train = p.obj.F(misfit) 

    yk_test, _, _ = compute_rhs(p, alg_out['xk'][-1], alg_out['sk'][-1], alg_out['ck'][-1], p.test_int, p.test_bnd)
    misfit_test = yk_test - rhs_test
    residue_test = p.obj_test.F(misfit_test)

    print()
    print('#' * 20)
    print(f'alpha: {alg_opts["alpha"]:.1e}')
    print(
        "L_inf error test (boundary): {L_inf_bnd_test:.2e}\n"
        "L_inf error test (interior): {L_inf_int_test:.2e}\n"
        "L_inf error test (total): {L_inf_test:.2e}\n"
        "L_2 error test: {L_2_test:.2e}".format(
            **errors_test
        )
    )
    print(f'residue test: {residue_test:.2e}')
    print(
        "L_inf error train (boundary): {L_inf_bnd_train:.2e}\n"
        "L_inf error train (interior): {L_inf_int_train:.2e}\n"
        "L_inf error train (total): {L_inf_train:.2e}\n"
        "L_2 error train: {L_2_train:.2e}".format(
            **errors_train
        )
    )
    print(f'residue train: {residue_train:.2e}')
    print(f'final support: {alg_out["supps"][-1]}')
    
    print('#' * 20)
    print()

    final_results = {
        'residue_test': residue_test,
        'residue_train': residue_train,
        'final_supp': alg_out['supps'][-1],
    }
    final_results.update(errors_test)
    final_results.update(errors_train)

    # Post-process alg_out
    num_iter = len(alg_out['sk'])
    max_supp = max([xk.shape[0] for xk in alg_out['xk']])

    xk_padded = np.zeros((num_iter, max_supp, p.d))
    sk_padded = np.zeros((num_iter, max_supp, p.dim - p.d))
    ck_padded = np.zeros((num_iter, max_supp))
    suppc = np.zeros((num_iter, max_supp), dtype=bool)
    for i in range(num_iter):
        xk_padded[i, :alg_out['xk'][i].shape[0]] = alg_out['xk'][i]
        sk_padded[i, :alg_out['sk'][i].shape[0]] = alg_out['sk'][i]
        ck_padded[i, :alg_out['ck'][i].shape[0]] = alg_out['ck'][i]
        suppc[i, :alg_out['suppc'][i].shape[0]] = alg_out['suppc'][i]

    alg_out['xk'] = xk_padded
    alg_out['sk'] = sk_padded
    alg_out['ck'] = ck_padded
    alg_out['suppc'] = suppc

    # Combine with options and errors
    alg_out.update(alg_opts)
    alg_out.update(final_results)

    # Save output
    if args.save_dir and args.save_idx is not None:
        out_dir = f"output/{p.name}/{args.save_dir}/out_{args.save_idx}"
        os.makedirs(out_dir, exist_ok=True)
        np.savez(f"{out_dir}/out_{args.save_idx}_{alg_opts['alpha']:.0e}.npz", **alg_out)

    return alg_out


for _ in range(3):
    # Solve the PDE and evaluate the solution
    alg_out = evaluate_and_save_solution(p, rhs, alg_opts, args)
    suppc = alg_out['suppc'][-1]
    p.u_zero = {
        'x': jnp.array(alg_out['xk'][-1, :, :] * suppc[:, None]),
        's': jnp.array(alg_out['sk'][-1, :, :] * suppc[:, None]),
        'u': jnp.array(alg_out['ck'][-1, :] * suppc)
    }
    
    alg_opts['alpha'] = 0.1 * alg_opts['alpha']  
    alg_opts['T'] = 10 * alg_opts['T']
    alg_opts['max_step'] = 2000