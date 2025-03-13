from pde.SemiLinearPDEskew import ProblemNNLapVar
from src.utils import Phi
from src.solve_TV_CGNAP import solve_TV_CGNAP


import numpy as np
import matplotlib.pyplot as plt

import os
import datetime
import json
import argparse

# write argparse here

parser = argparse.ArgumentParser(description='Run the algorithm to solve PDE problem.')


parser.add_argument('--sigma_max', type=float, default=1.0, help='Maximum value of the kernel width.')
parser.add_argument('--sigma_min', type=float, default=1e-3, help='Minimum value of the kernel width.')
parser.add_argument('--Nobs', type=int, default=50, help='Base number of observations')
parser.add_argument('--sampling', type=str, default='grid', help='Sampling method for the observations.')
parser.add_argument('--scale', type=int, default=0, help='penalty for the boundary condition')
parser.add_argument('--TOL', type=float, default=1e-5, help='Tolerance for stopping.')
parser.add_argument('--max_step', type=int, default=10000, help='Maximum number of steps.')
parser.add_argument('--print_every', type=int, default=100, help='Print every n steps.')
parser.add_argument('--plot_every', type=int, default=100, help='Plot every n steps.')
parser.add_argument('--insertion_coef', type=float, default=0.1, help='coefficient for thereshold of insertion.')
parser.add_argument('--gamma', type=float, default=0, help='gamma.')
parser.add_argument('--alpha', type=float, default=0.0001, help='Alpha parameter.')
parser.add_argument('--Ntrial', type=int, default=6000, help='Number of candidate parameters sampled each iter.')
parser.add_argument('--plot_final', type=bool, default=True, help='Plot the final result or not.')
parser.add_argument('--seed', type=int, default=200, help='Random seed for reproductivity.')
parser.add_argument('--config', type=str, default=None, help='config file to load the parameters.')
parser.add_argument('--index', type=str, default=None, help='index of the configuration to load.')
parser.add_argument('--add_noise', type=bool, default=False, help='Add noise to the rhs.')
parser.add_argument('--save_dir', type=str, default=None, help='Directory to save the output.')

args = parser.parse_args()

if args.config is not None:
    if args.index is None:
        raise ValueError("Please provide the index of the configuration to load.")
    with open(args.config, 'r') as f:
        configs = json.load(f)
    try:
        alg_opts = configs[args.index]
    except KeyError:
        raise KeyError("Invalid index.")
else:
    alg_opts = vars(parser.parse_args())

print(alg_opts)







# alg_opts = {
#     "TOL": 1e-4,
#     "max_step": 30000,
#     "print_every": 100,
#     "plot_every": 30000,
#     "insertion_coef": 0.0001, # initially 2, set based on the problem
#     "gamma": 0,
#     "alpha": .0001,
#     "sigma_min": 1e-4,
#     "sigma_max": 0.30,
#     "Nobs": 50,
#     "sampling": "grid",
#     "Ntrial": 6000,
#     "scale": 0, # since we use the mask to manually set the boundary to be 0
#     "plot_final": False
# }


p = ProblemNNLapVar(alg_opts)
rhs = p.f(p.xhat)
rhs[-p.Nx_bnd:] = p.ex_sol(p.xhat_bnd)


alg_out = solve_TV_CGNAP(p, rhs, alg_opts)

# post-processing alg_out

assert len(alg_out['xk']) == len(alg_out['sk']) == len(alg_out['ck'])
num_iter = len(alg_out['sk'])
max_supp = max([xk.shape[0] for xk in alg_out['xk']])
xk_padded = np.zeros((num_iter, max_supp, 2))
sk_padded = np.zeros((num_iter, max_supp))
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

if args.config is not None:
    home = "/".join(args.config.split("/")[:-1])

    if not os.path.exists(os.path.join(home, f"output")):
        os.makedirs(os.path.join(home, f"output"))

    # save npz file
    filename = args.config.split("/")[-1].split(".")[0]
    np.savez(f"{home}/output/{filename}_{args.index}.npz", **alg_out)

elif args.save_dir is None:
    if not os.path.exists(f"output/{p.name}"):
        os.makedirs(f"output/{p.name}")
    np.savez(f"output/{p.name}/out_{date}.npz", **alg_out)

else:
    if not os.path.exists(f"output/{args.save_dir}"):
        os.makedirs(f"output/{args.save_dir}")
    np.savez(f"output/{args.save_dir}/out_{p.name}_{date}.npz", **alg_out)