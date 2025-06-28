import sys
sys.path.append("./")
from pde.Eikonal import PDE
# from src.solver import solve
from src.solver_active import solve
# from src.solver_first_order import solve

# Scipy
import scipy.sparse
from scipy.sparse import diags
from scipy.sparse import identity
from scipy.interpolate import RegularGridInterpolator


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
parser.add_argument('--epsilon', type=float, default=0.1, help='Eikonal parameter.')


args = parser.parse_args()
alg_opts = vars(args)

print(alg_opts)


p = PDE(alg_opts)


rhs = np.ones((p.Nx,))
rhs[-p.Nx_bnd:] = 0.

rhs_test = np.ones((p.test_int.shape[0] + p.test_bnd.shape[0],))
rhs_test[-p.test_bnd.shape[0]:] = 0.



def solve_Eikonal(N, epsilon, x_left=0., x_right=1.):
    domain_length = x_right - x_left
    hg = domain_length / (N + 1)
    x_grid = x_left + hg * np.arange(1, N + 1)
    a1 = np.ones((N,N+1))
    a2 = np.ones((N+1,N))

    # diagonal element of A
    a_diag = np.reshape(a1[:,:N]+a1[:,1:]+a2[:N,:]+a2[1:,:], (1,-1))
    
    # off-diagonals
    a_super1 = np.reshape(np.append(a1[:,1:N], np.zeros((N,1)), axis = 1), (1,-1))
    a_super2 = np.reshape(a2[1:N,:], (1,-1))
    
    A = diags([[-a_super2[np.newaxis, :]], [-a_super1[np.newaxis, :]], [a_diag], [-a_super1[np.newaxis, :]], [-a_super2[np.newaxis, :]]], [-N,-1,0,1,N], shape=(N**2, N**2), format = 'csr')
    x_grid_full = np.linspace(x_left, x_right, N+2)
    XX, YY = np.meshgrid(x_grid_full, x_grid_full)
    f = np.zeros((N,N))
    f[0,:] = f[0,:] + epsilon**2 / (hg**2)
    f[N-1,:] = f[N-1,:] + epsilon**2 / (hg**2)
    f[:, 0] = f[:, 0] + epsilon**2 / (hg**2)
    f[:, N-1] = f[:, N-1] + epsilon**2 / (hg**2)
    fv = f.flatten()
    fv = fv[:, np.newaxis]
    
    mtx = identity(N**2)+(epsilon**2)*A/(hg**2)
    sol_v = scipy.sparse.linalg.spsolve(mtx, fv)
    # sol_v, exitCode = scipy.sparse.linalg.cg(mtx, fv)
    # print(exitCode)
    sol_u = -epsilon*np.log(sol_v)
    sol_u = np.reshape(sol_u, (N,N))
    sol_u = np.pad(sol_u, pad_width=1, mode='constant', constant_values=0.)

    return x_grid_full, XX, YY, sol_u

p.kernel.epsilon = args.epsilon
print(f'epsilon = {p.kernel.epsilon}')
# x_grid_full, XX, YY, test_truth = solve_Eikonal(100, p.kernel.epsilon, -1, 1)
# u_true = RegularGridInterpolator((x_grid_full, x_grid_full), test_truth, bounds_error=False, fill_value=None)
u_analytical = lambda x: np.min(1 - np.abs(x), axis=1)
p.ex_sol = u_analytical


# build interpolator




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
    print(f'epsilon: {alg_opts["epsilon"]:.1e}')
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
    u_pred_bnd_test = u_pred(p.test_bnd)
    u_true_bnd_test = p.ex_sol(p.test_bnd)
    u_pred_int_test = u_pred(p.test_int)
    u_true_int_test = p.ex_sol(p.test_int)

    L_inf_bnd_test = np.max(np.abs(u_pred_bnd_test - u_true_bnd_test))
    L_inf_int_test = np.max(np.abs(u_pred_int_test - u_true_int_test))
    L_2_test = np.sqrt(
        (np.sum((u_pred_int_test - u_true_int_test)**2) + np.sum((u_pred_bnd_test - u_true_bnd_test)**2))
        * p.vol_D / (p.test_int.shape[0] + p.test_bnd.shape[0])
    )

    u_pred_bnd_train = u_pred(p.xhat_bnd)
    u_true_bnd_train = p.ex_sol(p.xhat_bnd)
    u_pred_int_train = u_pred(p.xhat)
    u_true_int_train = p.ex_sol(p.xhat)
    L_inf_bnd_train = np.max(np.abs(u_pred_bnd_train - u_true_bnd_train))
    L_inf_int_train = np.max(np.abs(u_pred_int_train - u_true_int_train))
    L_2_train = np.sqrt(
        (np.sum((u_pred_int_train - u_true_int_train)**2) + np.sum((u_pred_bnd_train - u_true_bnd_train)**2))
        * p.vol_D / (p.xhat.shape[0] + p.xhat_bnd.shape[0])
    ) 

    # compute residue for both train and test

    linear_results_int = p.kernel.linear_E_results_X_c_Xhat(alg_out['xk'][-1], alg_out['sk'][-1], alg_out['ck'][-1], p.xhat_int)
    linear_results_bnd = p.kernel.linear_B_results_X_c_Xhat(alg_out['xk'][-1], alg_out['sk'][-1], alg_out['ck'][-1], p.xhat_bnd)
    yk_int = p.kernel.E_gauss_X_c_Xhat(**linear_results_int)
    yk_bnd = p.kernel.B_gauss_X_c_Xhat(**linear_results_bnd)
    yk = np.hstack([yk_int, yk_bnd])
    misfit = yk - rhs
    residue_train = p.obj.F(misfit) 

    linear_results_int_test = p.kernel.linear_E_results_X_c_Xhat(alg_out['xk'][-1], alg_out['sk'][-1], alg_out['ck'][-1], p.test_int)
    linear_results_bnd_test = p.kernel.linear_B_results_X_c_Xhat(alg_out['xk'][-1], alg_out['sk'][-1], alg_out['ck'][-1], p.test_bnd)
    yk_int_test = p.kernel.E_gauss_X_c_Xhat(**linear_results_int_test)
    yk_bnd_test = p.kernel.B_gauss_X_c_Xhat(**linear_results_bnd_test)
    yk_test = np.hstack([yk_int_test, yk_bnd_test])
    misfit_test = yk_test - rhs_test
    residue_test = p.obj_test.F(misfit_test)

    print()
    print('#' * 20)
    print(f'epsilon: {alg_opts["epsilon"]:.1e}')
    print(f'L_inf error test (boundary): {L_inf_bnd_test:.2e}')
    print(f'L_inf error test (interior): {L_inf_int_test:.2e}')
    print(f'L_inf error test (total): {max(L_inf_bnd_test, L_inf_int_test):.2e}')
    print(f'L_2 error test : {L_2_test:.2e}')
    print(f'residue test: {residue_test:.2e}')
    print(f'L_inf error train (boundary): {L_inf_bnd_train:.2e}')
    print(f'L_inf error train (interior): {L_inf_int_train:.2e}')
    print(f'L_inf error train (total): {max(L_inf_bnd_train, L_inf_int_train):.2e}')
    print(f'L_2 error train: {L_2_train:.2e}')
    print(f'residue train: {residue_train:.2e}')
    print(f'final support: {alg_out["supps"][-1]}')
    
    print('#' * 20)
    print()

    final_results = {
        'L_inf_bnd_test': L_inf_bnd_test,
        'L_inf_int_test': L_inf_int_test,
        'L_inf_test': max(L_inf_bnd_test, L_inf_int_test),
        'L_2_test': L_2_test,
        'residue_test': residue_test,
        'L_inf_bnd_train': L_inf_bnd_train,
        'L_inf_int_train': L_inf_int_train,
        'L_inf_train': max(L_inf_bnd_train, L_inf_int_train),
        'L_2_train': L_2_train,
        'residue_train': residue_train,
        'final_supp': alg_out['supps'][-1],
    }

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
    alg_out.update(final_results)

    # Save output
    if args.save_dir and args.save_idx is not None:
        out_dir = f"output/{p.name}/{args.save_dir}/out_{args.save_idx}"
        os.makedirs(out_dir, exist_ok=True)
        np.savez(f"{out_dir}/out_{args.save_idx}_{alg_opts['alpha']:.0e}.npz", **alg_out)

    return alg_out


alg_out = evaluate_and_save_solution(p, rhs, alg_opts, args)
