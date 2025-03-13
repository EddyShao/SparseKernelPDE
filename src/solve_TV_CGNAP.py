import numpy as np
import time
import matplotlib.pyplot as plt
from datetime import datetime
from functools import partial
from .utils import computeProx, Phi

Prox = lambda v: computeProx(v, mu=1)


def solve_TV_CGNAP(p, y_ref, alg_opts):
    """
    Solve the Total Variation - regularized CGNAP problem
    Gauss-Newton active point method for TV-regularized CGNAP

    Parameters
    ----------
    p : problem class object
    y_ref : torch.Tensor
        Reference points
    alpha : float
        Regularization parameter
    phi : function
        Function handle for the forward operator
    alg_opts : dict
        Dictionary containing the algorithm options

    """

    N =  1 # fix to be 1 here; scalar problems

    # redefine for readability
    dim = p.dim 
    obj = p.obj 

    Ndata = len(y_ref)
    

    max_step = alg_opts.get('max_step', 1000)
    TOL = alg_opts.get('TOL', 1e-5)
    insertion_coef = alg_opts.get('insertion_coef', 2)
    gamma = alg_opts.get('gamma', 1)
    alpha = alg_opts.get('alpha', 0.1)

    plot_final = alg_opts.get('plot_final', True)
    plot_every = alg_opts.get('plot_every', 0)
    print_every = alg_opts.get('print_every', 20)

    blocksize = alg_opts.get('blocksize', 50) 
    Ntrial = alg_opts.get('Ntrial', 1000)
    T = alg_opts.get('T', 300)

    # Initial guess
    u0 = alg_opts.get('u0', p.u_zero)
    uk = u0.copy()

    phi = Phi(gamma)

    # initial values
    # K_big, dK_big, _ = p.k(p.xhat, uk['x'])
    # yk = K_big @ uk['u'].T

    yk_int = p.kernel.P_gauss_X_c_Xhat(uk['x'], uk['s'], uk['u'], p.xhat_int)
    yk_bnd = p.kernel.gauss_X_c_Xhat(uk['x'], uk['s'], uk['u'], p.xhat_bnd)
    yk = np.hstack([yk_int, yk_bnd])

    # norms_c = computeNorm(uk['u'], N)
    norms_c = np.abs(uk['u'])

    # Compute the objective function value
    misfit = yk - y_ref
    j = obj.F(misfit)/alpha + np.sum(phi.phi(norms_c))

    y_pred_int = p.kernel.gauss_X_c_Xhat(uk['x'], uk['s'], uk['u'], p.test_int)
    y_pred_bnd = p.kernel.gauss_X_c_Xhat(uk['x'], uk['s'], uk['u'], p.test_bnd)
    y_true_int = p.ex_sol(p.test_int)
    y_true_bnd = p.ex_sol(p.test_bnd)
    l2_error = np.sqrt((np.sum((y_pred_int - y_true_int)**2) + np.sum((y_pred_bnd - y_true_bnd))) * p.vol_D / (100 * 100))
    l_inf_error_int = np.max(np.abs(y_pred_int - y_true_int))
    l_inf_error_bnd = np.max(np.abs(y_pred_bnd - y_true_bnd))
    l_inf_error = max(l_inf_error_int, l_inf_error_bnd)


    
    suppsize = np.count_nonzero(norms_c) # support size

    ck = uk['u'] # outer weights
    xk = uk['x'] # inner weights - collocation points
    sk = uk['s'] # inner weights - shape parameter (sigma)

    alg_out = {
        'xk': [xk],
        'sk': [sk],
        'ck': [ck],
        'js': [j],
        'supps': [suppsize],
        'tics': [0],
        'l2': [l2_error],
        'l_inf': [l_inf_error]
    }

    start_time = time.time()

   

    shape_dK = lambda dK: dK.transpose(0, 2, 1).reshape(Ndata, -1) 
    Nc = len(ck) # number of Diracs
    # blcki = np.random.permutation(Nc)[:min(blocksize, Nc)]
    # Ncblck = len(blcki) # actual block size
    

    # change to robinson variale
    qk = np.sign(ck) + ck  
    # check consistency of Robinson variable
    assert ck.size == 0 or np.linalg.norm(ck - Prox(qk), ord=np.inf) < 1e-14 

    # Update ck (actually there's no update)
    ck = Prox(qk)

    # Define function equivalents
    Dphima = lambda c: (phi.dphi(np.abs(c)) - 1) * np.sign(c) # gradient of modified phi
    DDphima = lambda c: phi.ddphi(np.abs(c)) # hessian of modified phi

    # # Compute values
    # # yk = K_big @ ck.T  # prediction
    # yk_int = p.kernel.P_gauss_X_c_Xhat(xk, sk, ck, p.xhat_int)
    # yk_bnd = p.kernel.gauss_X_c_Xhat(xk, sk, ck, p.xhat_bnd)
    # yk = np.hstack([yk_int, yk_bnd])
    # misfit = yk - y_ref 
    # norms_c = np.abs(ck) 

    # # Compute the objective function value
    # j = obj.F(misfit) / alpha + np.sum(phi.phi(norms_c))


    theta_old = 1. ###### QUESTION ###### 

    for k in range(1, max_step  + 1):
        # gradient of c, x, and s

        # Dc_P_gauss = p.kernel.Dc_P_gauss_X_c_Xhat(xk[blcki, :], sk[blcki], ck[blcki], p.xhat_int)
        # Dc_gauss = p.kernel.Dc_gauss_X_c_Xhat(xk[blcki, :], sk[blcki], ck[blcki], p.xhat_bnd)

        # Dx_P_gauss = p.kernel.Dx_P_gauss_X_c_Xhat(xk[blcki, :], sk[blcki], ck[blcki], p.xhat_int)
        # Dx_gauss = p.kernel.Dx_gauss_X_c_Xhat(xk[blcki, :], sk[blcki], ck[blcki], p.xhat_bnd)

        # Ds_P_gauss = p.kernel.Ds_P_gauss_X_c_Xhat(xk[blcki, :], sk[blcki], ck[blcki], p.xhat_int)
        # Ds_gauss = p.kernel.Ds_gauss_X_c_Xhat(xk[blcki, :], sk[blcki], ck[blcki], p.xhat_bnd)


        Dc_P_gauss = p.kernel.Dc_P_gauss_X_c_Xhat(xk, sk, ck, p.xhat_int)
        Dc_gauss = p.kernel.Dc_gauss_X_c_Xhat(xk, sk, ck, p.xhat_bnd)

        Dx_P_gauss = p.kernel.Dx_P_gauss_X_c_Xhat(xk, sk, ck, p.xhat_int)
        Dx_gauss = p.kernel.Dx_gauss_X_c_Xhat(xk, sk, ck, p.xhat_bnd)

        Ds_P_gauss = p.kernel.Ds_P_gauss_X_c_Xhat(xk, sk, ck, p.xhat_int)
        Ds_gauss = p.kernel.Ds_gauss_X_c_Xhat(xk, sk, ck, p.xhat_bnd)

        # Assemble the gradient
        Gp_c = np.vstack([np.array(Dc_P_gauss), np.array(Dc_gauss)])
        Gp_x = np.vstack([np.array(Dx_P_gauss), np.array(Dx_gauss)])
        Gp_s = np.vstack([np.array(Ds_P_gauss), np.array(Ds_gauss)])
        Gp_xs = np.dstack([Gp_x, Gp_s[:, :, None]])

        Gp = np.hstack([Gp_c, shape_dK(Gp_xs)])
        
        # R = (1 / alpha) * (Gp.T @ obj.dF(misfit)) + \
        #     np.concatenate([
        #     Dphima(ck[blcki]).reshape(-1, 1) + (qk[blcki] - ck[blcki]).reshape(-1, 1), 
        #     np.zeros((Ncblck * dim, 1))
        # ]) # gradient with respect to qk, xk, and s respectively

        R = (1 / alpha) * (Gp.T @ obj.dF(misfit)) + \
            np.concatenate([
            Dphima(ck).reshape(-1, 1) + (qk - ck).reshape(-1, 1), 
            np.zeros((len(ck) * dim, 1))
        ]) # gradient with respect to qk, xk, and s respectively

        SI = obj.ddF(misfit)
        II = Gp.T @ SI @ Gp # Approximate Hessian 

        ###### QUESTION ######

        # kpp = 0.1 * np.linalg.norm(obj.dF(misfit), 1) * np.reshape(
        #     np.sqrt(np.finfo(float).eps) + np.outer(np.ones(dim), np.abs(ck[blcki].reshape(1, -1))), -1
        # )
        kpp = 0.1 * np.linalg.norm(obj.dF(misfit), 1) * np.reshape(
            np.sqrt(np.finfo(float).eps) + np.outer(np.ones(dim), np.abs(ck.reshape(1, -1))), -1
        )

        # Icor = np.block([
        #     [np.zeros((Ncblck, Ncblck)), np.zeros((Ncblck, dim * Ncblck))],
        #     [np.zeros((dim * Ncblck, Ncblck)), np.diag(kpp)]
        # ])

        Icor = np.block([
            [np.zeros((len(ck), len(ck))), np.zeros((len(ck), dim * len(ck)))],
            [np.zeros((dim * len(ck), len(ck))), np.diag(kpp)]
        ])


        HH = (1 / alpha) * (II + Icor)

        ###### QUESTION ######

        # SSN correction
        # DP = np.diag(
        #     np.concatenate([
        #         (np.abs(qk[blcki].T) >= 1).reshape(-1, 1),
        #         (np.ones((dim, 1)) @ (np.abs(ck[blcki]) > 0).reshape(1, -1)).reshape(-1, 1)
        #     ]).flatten()
        # )

        DP = np.diag(
            np.concatenate([
                (np.abs(qk.T) >= 1).reshape(-1, 1),
                (np.ones((dim, 1)) @ (np.abs(ck) > 0).reshape(1, -1)).reshape(-1, 1)
            ]).flatten()
        )


        # DDphi = np.zeros(((1 + dim) * Ncblck, (1 + dim) * Ncblck))
        # DDphi[:Ncblck, :Ncblck] = np.diag(DDphima(ck[blcki]))

        DDphi = np.zeros(((1 + dim) * len(ck), (1 + dim) * len(ck)))
        DDphi[:len(ck), :len(ck)] = np.diag(DDphima(ck))

        # DR = HH @ DP + 0*DDphi @ DP + (np.eye((1 + dim) * Ncblck) - DP)
        

        #  This is for stability concern
        # DR = HH @ DP + DDphi @ DP + (np.eye((1 + dim) * Ncblck) - DP)

        try:
            DR = HH @ DP + DDphi @ DP + (np.eye((1 + dim) * len(ck)) - DP)
            dz = - np.linalg.solve(DR, R)
            dz = dz.flatten()
        except np.linalg.LinAlgError:
            try:
                DR = HH @ DP + (np.eye((1 + dim) * len(ck)) - DP)
                dz = - np.linalg.solve(DR, R)
                dz = dz.flatten()
                
            except np.linalg.LinAlgError:
                print("WARNING: Singular matrix encountered.")
                alg_out["success"] = False
                break
        
        

        # Line search
        jold, xold, sold, qold = j, xk.copy(), sk.copy(), qk.copy()
        pred = (R.T @ (DP @ dz.reshape(-1, 1))) # estimate of the descent

        theta = min(theta_old * 2, 1 - 1e-14) 

        has_descent = False


        while not has_descent and theta > 1e-20:
            # qk[blcki] = qold[blcki] + theta * dz[:Ncblck]
            # dxs = dz[Ncblck:].reshape(dim, -1).T
            # xk[blcki, ] = xold[blcki, ] + theta * dxs[:, :dim-1]
            # sk[blcki] = sold[blcki] + theta * dxs[:, dim-1].flatten()

            qk = qold + theta * dz[:len(ck)]
            dxs = dz[len(ck):].reshape(dim, -1).T
            xk = xold + theta * dxs[:, :dim-1]
            sk = sold + theta * dxs[:, dim-1].flatten()

            ck = Prox(qk)

            # yk = p.kernel.gauss_X_c_Xhat(xk, sk, ck, p.xhat)
            yk_int = p.kernel.P_gauss_X_c_Xhat(xk, sk, ck, p.xhat_int)
            yk_bnd = p.kernel.gauss_X_c_Xhat(xk, sk, ck, p.xhat_bnd)
            yk = np.hstack([yk_int, yk_bnd])

            misfit = yk - y_ref
            norms_c = np.abs(ck)
            j = obj.F(misfit) / alpha + np.sum(phi.phi(norms_c))
            descent = j - jold
            pred = theta * (R.T @ (DP @ dz.reshape(-1, 1))) # estimate of the descent
            
            has_descent = descent <= (pred + 1e-11) / 3

            if not has_descent:
                theta /= 1.5 # shrink theta

        theta_old = theta 

        if not has_descent:
            # raise RuntimeError("Line search failed")
            # put a warning here
            print("WARNING: Line search failed.")
            alg_out["success"] = False
            break



        # Active set
        suppc = (np.abs(qk) > 1).flatten()

        # Constraint violation: Generate search grid
        # Sample Candidate Diracs

        omegas_new_x, omegas_new_s = p.sample_param(Ntrial + blocksize)

        if k > 1:
            omegas_x = np.vstack([omegas_x[ind_max_sh_eta:ind_max_sh_eta+1, :], omegas_new_x])
            omegas_s = np.hstack([omegas_s[ind_max_sh_eta:ind_max_sh_eta+1], omegas_new_s])
        else:
            omegas_x = omegas_new_x
            omegas_s = omegas_new_s
        
        center_vals = p.kernel.gauss_X_c_Xhat(xk, sk, ck, p.xhat_int)
        K_test_int = p.kernel.DP_gauss_X_Xhat(omegas_x, omegas_s, p.xhat_int, center_vals)
        K_test_bnd = p.kernel.gauss_X_Xhat(omegas_x, omegas_s, p.xhat_bnd)

        K_test = np.vstack([K_test_int, K_test_bnd])

        eta = (1 / alpha) * K_test.T @ obj.dF(misfit) 
        sh_eta = np.abs(Prox(eta)).flatten()
        sh_eta, sorted_ind = np.sort(sh_eta)[::-1], np.argsort(-sh_eta) 
        max_sh_eta, ind_max_sh_eta = sh_eta[0], sorted_ind[0]
    

        # Compute l2 and l_inf errors
        y_pred_int = p.kernel.gauss_X_c_Xhat(xk, sk, ck, p.test_int)
        y_pred_bnd = p.kernel.gauss_X_c_Xhat(xk, sk, ck, p.test_bnd)
        l2_error = np.sqrt((np.sum((y_pred_int - y_true_int)**2) + np.sum((y_pred_bnd - y_true_bnd))) * p.vol_D / (100 * 100))
        l_inf_error_int = np.max(np.abs(y_pred_int - y_true_int))
        l_inf_error_bnd = np.max(np.abs(y_pred_bnd - y_true_bnd))
        l_inf_error = max(l_inf_error_int, l_inf_error_bnd)

        # Print iteration info
        if k % print_every == 0:
            print(f"Time: {time.time() - start_time:.2f}s CGNAP iter: {k}, j={j:.6f}, supp=({Nc}->{np.sum(suppc)}), "
                f"desc={descent:.1e}, dz={np.linalg.norm(dz, np.inf):.1e}, "
                f"viol={max_sh_eta:.1e}, theta={theta:.1e}")

            
            print(f"L_2 error: {l2_error:.3e}, L_inf error: {l_inf_error:.3e}, (int: {l_inf_error_int:.3e}, bnd: {l_inf_error_bnd:.3e})")
        

        alg_out["xk"].append(xk)
        alg_out["sk"].append(sk)
        alg_out["ck"].append(ck)
        alg_out["js"].append(j)
        alg_out["l2"].append(l2_error)
        alg_out["l_inf"].append(l_inf_error)
        alg_out["supps"].append(Nc)
        alg_out["tics"].append(time.time() - start_time) 
        alg_out["success"] = True



        # Prune zero coefficient Diracs
        if np.any(~suppc):
            Nc = np.sum(suppc)
            qk = qk[suppc]
            xk = xk[suppc, :]
            sk = sk[suppc]
            ck = Prox(qk)
            Gp_c = Gp_c[:, suppc]
            Gp_xs = Gp_xs[:, suppc, :]
            print(f"  PRUNE: supp:({len(suppc)}->{Nc}) ")


        ###### QUESTION ######

        # Try adding promising new zero coefficients

        grad_supp_c = (1 / alpha) * (Gp_c.T @ obj.dF(misfit)) + Dphima(ck).reshape(-1, 1) + (qk - ck).reshape(-1, 1)

        tresh_c = np.abs(grad_supp_c).T
        grad_supp_y = (1 / alpha) * shape_dK(Gp_xs).T @ obj.dF(misfit)
        tresh_y = np.sqrt(np.sum(grad_supp_y.reshape(dim, -1) ** 2, axis=0))

        # tresh = tresh_c + 0.01 * tresh_y # We need to change this.
        tresh = tresh_c + 0.01 * tresh_y
        # ind_th = np.argsort(-tresh)
        # blcki = ind_th.flatten() # update the active point set. 
        # Ncblck = len(blcki)
        # approved_ind = np.where(sh_eta > .5 * np.linalg.norm(tresh, ord=np.inf))[0][:p.kernel.pad_size - Nc]      

        # if max_sh_eta > insertion_coef * np.linalg.norm(tresh, ord=np.inf):
        # doing MCMC here
        annealing = - 3 * np.log10(alpha) * np.max(np.abs(misfit)) / np.max(np.abs(y_ref))
        if np.random.rand() < np.exp(-(np.linalg.norm(tresh, ord=np.inf) - max_sh_eta) / (T * annealing**2 + 1e-5)):
            Nc += 1
            qk = np.hstack([qk, -np.sign(eta[ind_max_sh_eta]).flatten()])
            ck = np.hstack([ck, np.zeros((1))])
            xk = np.vstack([xk, omegas_x[ind_max_sh_eta, :]])
            sk = np.hstack([sk, omegas_s[ind_max_sh_eta]])

            print(f"  INSERT: viol={max_sh_eta:.2e}, |g_c|+|g_y|={np.max(tresh_c, initial=0):.1e}+{np.max(tresh_y, initial=0):.1e}, "
                f"supp:({np.sum(suppc)}->{Nc})")

            # if Ncblck < blocksize:
            #     blcki = np.append(blcki, Nc-1)
            #     Ncblck = len(blcki)
            # else:
            #     blcki[Ncblck-1] = Nc-1
            
            if Nc > p.kernel.pad_size:
                p.kernel.pad_size = 2 * p.kernel.pad_size
        
        # blcki = np.sort(blcki)

        # Save diagnostics
        

        # Plot results
        if k % plot_every == 0:            
            p.plot_forward(xk, sk, ck)
            # print('should plot here')

        # Stopping criterion
        
        if np.abs(pred) < (TOL / alpha) and  max_sh_eta < (TOL / alpha):
            dz_norm = np.linalg.norm(dz, np.inf) if dz.size > 0 else 0.0  
            print(f"CGNAP iter: {k}, j={j:.6f}, supp=({Nc}->{np.sum(suppc)}), "
                f"desc={descent:.1e}, dz={dz_norm:.1e}, "
                f"viol={max_sh_eta:.1e}, theta={theta:.1e}")
            
            print(f"L_2 error: {l2_error:.3e}, L_inf error: {l_inf_error:.3e}, (int: {l_inf_error_int:.3e}, bnd: {l_inf_error_bnd:.3e})")
            print(f"Converged in {k} iterations")
            break

    
        
    if plot_final:
        p.plot_forward(xk, sk, ck)  
        plt.show()
        # timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M")
        # plt.figure(2002)
        # p.plot_adjoint(u_opt, obj.dF(yk - y_ref), alpha)
        # plt.savefig(f"figs/{p.name}_{timestamp}.png")
        # plt.show()  # Ensures the figures are displayed

    return alg_out


