import sys
sys.path.append("./")
from pde.Burgers1Dorder1ex import PDE as PDEorder1
# from pde.Burgers1Dorder2 import PDE as PDEorder2
# from src.solver import solve
# from src.solver_active import solve
from src.solver import solve
from scipy.interpolate import RegularGridInterpolator
import jax.numpy as jnp
import jax
import time

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
parser.add_argument('--scale', type=float, default=0, help='penalty for the boundary condition')
parser.add_argument('--TOL', type=float, default=1e-5, help='Tolerance for stopping.')
parser.add_argument('--max_step', type=int, default=5000, help='Maximum number of steps.')
parser.add_argument('--print_every', type=int, default=100, help='Print every n steps.')
parser.add_argument('--plot_every', type=int, default=100, help='Plot every n steps.')
parser.add_argument('--insertion_coef', type=float, default=0.1, help='coefficient for thereshold of insertion.') # with metroplis-hasting heuristic insertion coef is not used.
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
parser.add_argument('--dt', type=float, default=0.01, help='Time step for the simulation.')


args = parser.parse_args()
alg_opts = vars(args)

print(alg_opts)

# True solution is computed from the Cole-Hopf transformation with numerical quadrature
# Refernence: https://github.com/yifanc96/NonLinPDEs-GPsolver
nu = 0.02
[Gauss_pts, weights] = np.polynomial.hermite.hermgauss(80)
def u_truth(x1, x2):
    temp = x2-np.sqrt(4*nu*x1)*Gauss_pts
    val1 = weights * np.sin(np.pi*temp) * np.exp(-np.cos(jnp.pi*temp)/(2*np.pi*nu))
    val2 = weights * np.exp(-np.cos(np.pi*temp)/(2*np.pi*nu))
    return -np.sum(val1)/np.sum(val2)


hist_u = []


def u_0(x):
    return - np.sin(np.pi * x).flatten()


hist_u.append(u_0)

# post-processing alg_out

def save_iter(p_idx, alg_out, idx):
    assert len(alg_out['xk']) == len(alg_out['sk']) == len(alg_out['ck'])
    num_iter = len(alg_out['sk'])
    max_supp = max([xk.shape[0] for xk in alg_out['xk']])
    xk_padded = np.zeros((num_iter, max_supp, p_idx.d))
    sk_padded = np.zeros((num_iter, max_supp, p_idx.dim-p_idx.d))
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
        if not os.path.exists(f"output/{p_idx.name}"):
            os.makedirs(f"output/{p_idx.name}")
        np.savez(f"output/{p_idx.name}/out_{date}.npz", **alg_out)
    else:
        if not os.path.exists(f"output/{p_idx.name}/{args.save_dir}/"):
            os.makedirs(f"output/{p_idx.name}/{args.save_dir}/")
        if args.save_idx is not None:
            if not os.path.exists(f"output/{p_idx.name}/{args.save_dir}/{args.save_idx}_hist"):
                os.makedirs(f"output/{p_idx.name}/{args.save_dir}/{args.save_idx}_hist")
            np.savez(f"output/{p_idx.name}/{args.save_dir}/{args.save_idx}_hist/out_{args.save_idx}_{idx:04d}.npz", **alg_out)


T = 1.0
n_steps = int(T / args.dt) 
ck_hist = []
xk_hist = []
sk_hist = []
supp_hist = []
t_hist = [0.0]

residue_train_hist = []
residue_test_hist = []


start = time.time()
for idx in range(1, n_steps+1):
    print()
    print(f"Step {idx} Time {time.time() - start:.2f}")
    p_idx = PDEorder1(alg_opts)
    p_idx.kernel.dt = args.dt
    if idx > 3:
        alg_opts['max_step'] = 1000
    
    def f(xhat):
        u_prev = hist_u[-1](xhat)
        return u_prev.flatten()
    
    def ex_sol(x):
        return np.array([u_truth(idx * p_idx.kernel.dt, x_i) for x_i in x]).flatten()
    
    t_hist.append(idx * p_idx.kernel.dt)
    
    p_idx.f = f
    p_idx.ex_sol = ex_sol
    if idx > 1:
        p_idx.u_zero ={
            'x':xk_hist[-1],
            's':sk_hist[-1],
            'u':ck_hist[-1]
        }
    
    rhs = np.array(p_idx.f(p_idx.xhat).flatten())
    rhs[-2:] = 0.0

    
    alg_out = solve(p_idx, rhs, alg_opts)

    # compute train residue and test residue


    sk_hist.append(alg_out['sk'][-1])
    ck_hist.append(alg_out['ck'][-1])
    xk_hist.append(alg_out['xk'][-1])
    supp_hist.append(alg_out['xk'][-1].shape[0])

    save_iter(p_idx, alg_out, idx)
    xk = alg_out['xk'][-1]
    sk = alg_out['sk'][-1]
    ck = alg_out['ck'][-1]

    u_idx = lambda xhat_var, xk=xk, sk=sk, ck=ck: p_idx.kernel.gauss_X_c_Xhat(xk, sk, ck, xhat_var)
    hist_u.append(u_idx)


# L_inf and L_2 error, 120*120 grid
u_pred_grid = np.zeros((len(t_hist), 120))

x_space = np.linspace(-1, 1, 120)
for i, t in enumerate(t_hist):
    u_true = np.array([u_truth(t, x_i) for x_i in x_space])
    u_pred = hist_u[i](x_space)
    print(f"t = {t:.2f}: {np.abs(u_true - u_pred).max():.2e}")
    u_pred_grid[i] = u_pred

u_pred_func = RegularGridInterpolator((t_hist, x_space), u_pred_grid, bounds_error=True, fill_value=None)


# test the l_inf error and l_2 error between u_pred_func and u_truth

t_space = np.linspace(0, T, 120)
x_space = np.linspace(-1, 1, 120)
tx = np.array(np.meshgrid(t_space, x_space)).T.reshape(-1, 2)

u_true = np.array([u_truth(t, x_i) for t, x_i in tx])
u_pred = u_pred_func(tx).flatten()

l_inf_err = np.abs(u_true - u_pred).max()
l_2_err = np.sqrt(np.mean((u_true - u_pred) ** 2) * 2)

print()
print("#" * 40)
print("#" * 40)
print(f"l_inf_err: {l_inf_err:.2e}, l_2_err: {l_2_err:.2e}")
print("#" * 40)
print("#" * 40)

t_train_space = np.linspace(0, T, 101)
x_train_space = np.linspace(-1, 1, 40)
tx_train = np.array(np.meshgrid(t_train_space, x_train_space)).T.reshape(-1, 2)

u_true_train = np.array([u_truth(t, x_i) for t, x_i in tx_train])
u_pred_train = u_pred_func(tx_train).flatten()

l_inf_err_train = np.abs(u_true_train - u_pred_train).max()

print()
print("#" * 40)
print("#" * 40)
print(f"l_inf_err_train: {l_inf_err_train:.2e}")
print("#" * 40)
print("#" * 40)





# post-process x_hist, s_hist, c_hist
# save 

max_supp = max([xk.shape[0] for xk in xk_hist])

xk_padded = np.zeros((n_steps, max_supp, p_idx.d))
sk_padded = np.zeros((n_steps, max_supp, p_idx.dim-p_idx.d))
ck_padded = np.zeros((n_steps, max_supp))

for i in range(n_steps):
    xk_padded[i, :xk_hist[i].shape[0]] = xk_hist[i]
    sk_padded[i, :sk_hist[i].shape[0]] = sk_hist[i]
    ck_padded[i, :ck_hist[i].shape[0]] = ck_hist[i]

    
date = datetime.datetime.now().strftime("%m%d_%H%M")

if args.save_dir and args.save_idx:
        if not os.path.exists(f"output/{p_idx.name}/{args.save_dir}"):
            os.makedirs(f"output/{p_idx.name}/{args.save_dir}")
        np.savez(f"output/{p_idx.name}/{args.save_dir}/out_{args.save_idx}.npz", 
                 xk=xk_padded, sk=sk_padded, ck=ck_padded, supp=supp_hist, 
                 t=t_hist, x_space=x_space, u_pred_grid=u_pred_grid, 
                 l_inf_err=l_inf_err, l_2_err=l_2_err, l_inf_err_train=l_inf_err_train)