# Scipy
import scipy.sparse
from scipy.sparse import diags
from scipy.sparse import identity
from scipy.interpolate import RegularGridInterpolator


import numpy as np
# REFERENCE: https://github.com/yifanc96/NonLinPDEs-GPsolver/blob/main/reference_solver/Cole_Hopf_for_Eikonal.py


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

EPSILON = 0.1
x_grid_full, XX, YY, test_truth = solve_Eikonal(2000, 0.1, -1, 1)


# save the solution
np.savez('eikonal_regularized_exact.npz', x_grid_full=x_grid_full, XX=XX, YY=YY, test_truth=test_truth)