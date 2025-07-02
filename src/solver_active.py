import numpy as np
import time
import matplotlib.pyplot as plt
from datetime import datetime
from functools import partial
from .utils import computeProx, Phi

Prox = lambda v: computeProx(v, mu=1)


def solve(p, y_ref, alg_opts):
    """
    Solve the Total Variation problem
    Gauss-Newton (active point) method for TV-regularized

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


    # redefine for readability
    d = p.d
    dim = p.dim 
    obj = p.obj 

    Ndata = len(y_ref)
    

    max_step = alg_opts.get('max_step', 1000)
    TOL = alg_opts.get('TOL', 1e-5)
    insertion_coef = alg_opts.get('insertion_coef', 0.01)
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


    linear_results_int = p.kernel.linear_E_results_X_c_Xhat(uk['x'], uk['s'], uk['u'], p.xhat_int)
    linear_results_bnd = p.kernel.linear_B_results_X_c_Xhat(uk['x'], uk['s'], uk['u'], p.xhat_bnd)


    yk_int = p.kernel.E_gauss_X_c_Xhat(**linear_results_int)
    yk_bnd = p.kernel.B_gauss_X_c_Xhat(**linear_results_bnd)
    yk = np.hstack([yk_int, yk_bnd])

    norms_c = np.abs(uk['u'])

    # Compute the objective function value
    misfit = yk - y_ref
    j = obj.F(misfit)/alpha + np.sum(phi.phi(norms_c))

    y_pred_int = p.kernel.gauss_X_c_Xhat(uk['x'], uk['s'], uk['u'], p.test_int)
    y_pred_bnd = p.kernel.gauss_X_c_Xhat(uk['x'], uk['s'], uk['u'], p.test_bnd)
    y_true_int = p.ex_sol(p.test_int)
    y_true_bnd = p.ex_sol(p.test_bnd)
    l2_error = np.sqrt((np.sum((y_pred_int - y_true_int)**2) + np.sum((y_pred_bnd - y_true_bnd)**2)) * p.vol_D / (p.Ntest **p.d))
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
    ind_active = np.random.permutation(Nc)[:min(blocksize, Nc)]

    # change to robinson variale
    qk = np.sign(ck) + ck  
    # check consistency of Robinson variable
    assert ck.size == 0 or np.linalg.norm(ck - Prox(qk), ord=np.inf) < 1e-14 

    # Update ck (actually there's no update)
    ck = Prox(qk)

    # Define function equivalents
    Dphima = lambda c: (phi.dphi(np.abs(c)) - 1) * np.sign(c) # gradient of modified phi
    DDphima = lambda c: phi.ddphi(np.abs(c)) # hessian of modified phi


    theta_old = 1. ###### QUESTION ###### 

    print('Start Iterations')

    for k in range(1, max_step  + 1):

        Grad_E = p.kernel.Grad_E_gauss_X_c_Xhat(xk, sk, ck, p.xhat_int)
        Grad_B = p.kernel.Grad_B_gauss_X_c_Xhat(xk, sk, ck, p.xhat_bnd)

        
        Dc_E_gauss, Dx_E_gauss, Ds_E_gauss = Grad_E['grad_c'], Grad_E['grad_X'], Grad_E['grad_S']
        Dc_B_gauss, Dx_B_gauss, Ds_B_gauss = Grad_B['grad_c'], Grad_B['grad_X'], Grad_B['grad_S']

        Gp_c = np.vstack([np.array(Dc_E_gauss), np.array(Dc_B_gauss)])
        Gp_x = np.vstack([np.array(Dx_E_gauss), np.array(Dx_B_gauss)])
        Gp_s = np.vstack([np.array(Ds_E_gauss), np.array(Ds_B_gauss)])

        if Gp_s.ndim == 2:
            Gp_s = Gp_s[:, :, None]
        Gp_xs = np.dstack([Gp_x, Gp_s])
        Gp_xs_active = Gp_xs[:, ind_active, :]
    

        # We optimize over qk, xk, and sk
        # for inner werights xk and sk, we only optimize those that are active
        Gp = np.hstack([Gp_c, shape_dK(Gp_xs_active)])

        R = (1 / alpha) * (Gp.T @ obj.dF(misfit)) + \
            np.concatenate([
            Dphima(ck).reshape(-1, 1) + (qk - ck).reshape(-1, 1), 
            np.zeros((len(ind_active) * dim, 1))
        ]) # gradient with respect to qk, xk, and s respectively

        SI = obj.ddF(misfit)
        II = Gp.T @ SI @ Gp # Approximate Hessian 


        kpp = 0.1 * np.linalg.norm(obj.dF(misfit), 1) * np.reshape(
            np.sqrt(np.finfo(float).eps) + np.outer(np.ones(dim), np.abs((ck[ind_active]).reshape(1, -1))), -1
        )


        Icor = np.block([
            [np.diag(np.sqrt(np.finfo(float).eps) * np.ones(len(ck))), np.zeros((len(ck), dim * len(ind_active)))],
            [np.zeros((dim * len(ind_active), len(ck))), np.diag(kpp)]
        ])


        HH = (1 / alpha) * (II + Icor)


        DP = np.diag(
            np.concatenate([
                (np.abs(qk.T) >= 1).reshape(-1, 1),
                # (np.ones((dim, 1)) @ (np.abs(ck[ind_active]) > 0).reshape(1, -1)).reshape(-1, 1)
                np.tile((np.abs(ck[ind_active]) > 0), dim).reshape(-1, 1)
            ]).flatten()
        )


        DDphi = np.zeros((len(ck) + dim * len(ind_active), len(ck) + dim * len(ind_active)))
        DDphi[:len(ck), :len(ck)] = np.diag(DDphima(ck))


        try:
            DR = HH @ DP + DDphi @ DP + (np.eye(len(ck) + len(ind_active)*dim) - DP)
            dz = - np.linalg.solve(DR, R)
            dz = dz.flatten()
        except np.linalg.LinAlgError:
            try:
                DR = HH @ DP + (np.eye(len(ck) + len(ind_active)*dim) - DP)
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

            qk = qold + theta * dz[:len(ck)]
            dxs = dz[len(ck):].reshape(dim, -1).T
            xk[ind_active, :] = xold[ind_active, :] + theta * dxs[:, :d]
            sk[ind_active, :] = sold[ind_active, :] + theta * dxs[:, d:]

            ck = Prox(qk)

            # yk = p.kernel.gauss_X_c_Xhat(xk, sk, ck, p.xhat) # we use a 2-step scheme to compute yk instead
            linear_results_int = p.kernel.linear_E_results_X_c_Xhat(xk, sk, ck, p.xhat_int)
            linear_results_bnd = p.kernel.linear_B_results_X_c_Xhat(xk, sk, ck, p.xhat_bnd)
            yk_int = p.kernel.E_gauss_X_c_Xhat(**linear_results_int)
            yk_bnd = p.kernel.B_gauss_X_c_Xhat(**linear_results_bnd)
            yk = np.hstack([yk_int, yk_bnd])
            misfit = yk - y_ref
            norms_c = np.abs(ck)
            j = obj.F(misfit) / alpha + np.sum(phi.phi(norms_c))
            descent = j - jold
            pred = theta * (R.T @ (DP @ dz.reshape(-1, 1))) # estimate of the descent
            
            has_descent = descent <= (pred + 1e-11) / 5

            if not has_descent:
                theta /= 1.5 # shrink theta

        theta_old = theta 

        if not has_descent:
            # raise RuntimeError("Line search failed")
            # put a warning here
            print("WARNING: Line search failed.")
            alg_out["success"] = False
            break

        suppc = (np.abs(qk) > 1).flatten()

        omegas_x, omegas_s = p.sample_param(Ntrial)

        K_test_int = p.kernel.DE_gauss_X_Xhat(omegas_x, omegas_s, p.xhat_int, **linear_results_int)
        K_test_bnd = p.kernel.DB_gauss_X_Xhat(omegas_x, omegas_s, p.xhat_bnd, **linear_results_bnd)

        K_test = np.vstack([K_test_int, K_test_bnd])

        eta = (1 / alpha) * K_test.T @ obj.dF(misfit) 
        sh_eta = np.abs(Prox(eta)).flatten()
        sh_eta, sorted_ind = np.sort(sh_eta)[::-1], np.argsort(-sh_eta) 
        max_sh_eta, ind_max_sh_eta = sh_eta[0], sorted_ind[0]
    
        # Compute l2 and l_inf errors
        y_pred_int = p.kernel.gauss_X_c_Xhat(xk, sk, ck, p.test_int)
        y_pred_bnd = p.kernel.gauss_X_c_Xhat(xk, sk, ck, p.test_bnd)
        l2_error = np.sqrt((np.sum((y_pred_int - y_true_int)**2) + np.sum((y_pred_bnd - y_true_bnd)**2)) * p.vol_D / (p.Ntest ** p.d))
        l_inf_error_int = np.max(np.abs(y_pred_int - y_true_int))
        l_inf_error_bnd = np.max(np.abs(y_pred_bnd - y_true_bnd))
        l_inf_error = max(l_inf_error_int, l_inf_error_bnd)

        # Print iteration info
        if k % print_every == 0:
            dz_norm = np.linalg.norm(dz, np.inf) if dz.size > 0 else 0.0
            print(f"Time: {time.time() - start_time:.2f}s CGNAP iter: {k}, j={j:.6f}, supp=({Nc}->{np.sum(suppc)}), "
                f"desc={descent:.1e}, dz={dz_norm:.1e}, "
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
            if sk.ndim == 1:
                sk = sk[suppc]
            else:
                sk = sk[suppc, :]
            ck = Prox(qk)
            Gp_c = Gp_c[:, suppc]
            Gp_xs = Gp_xs[:, suppc, :]
            print(f"  PRUNE: supp:({len(suppc)}->{Nc}) ")

        grad_supp_c = (1 / alpha) * (Gp_c.T @ obj.dF(misfit)) + Dphima(ck).reshape(-1, 1) + (qk - ck).reshape(-1, 1)

        tresh_c = np.abs(grad_supp_c).T
        grad_supp_y = (1 / alpha) * shape_dK(Gp_xs).T @ obj.dF(misfit)
        tresh_y = np.sqrt(np.sum(grad_supp_y.reshape(dim, -1) ** 2, axis=0))

        tresh = tresh_c +  insertion_coef * tresh_y
        

        ind_active = np.argsort(-tresh_y)[:min(blocksize, Nc)]

        # doing MCMC here
        # determine active point, based on tresh_y only
        # Otherwise, we can do more heuristically:
        # if max_sh_eta > insertion_coef * np.linalg.norm(tresh, ord=np.inf):
        annealing = - 3 * np.log10(alpha) * np.max(np.abs(misfit)) / (np.max(np.abs(y_ref))) # A Heuristic annealing coefficient
        log_prob = -(np.linalg.norm(tresh, ord=np.inf) - max_sh_eta) / (T * annealing**2 + 1e-5)
        log_prob = np.clip(log_prob, -100, 100)
        if np.random.rand() < np.exp(log_prob):
            Nc += 1
            qk = np.hstack([qk, -np.sign(eta[ind_max_sh_eta]).flatten()])
            ck = np.hstack([ck, np.zeros((1))])
            xk = np.vstack([xk, omegas_x[ind_max_sh_eta, :]])
            if sk.ndim == 1:
                sk = np.hstack([sk, omegas_s[ind_max_sh_eta]])
            else:
                sk = np.vstack([sk, omegas_s[ind_max_sh_eta, :]])

            # Update the active point set
            ind_active = np.hstack([ind_active, Nc - 1])

            print(f"  INSERT: viol={max_sh_eta:.2e}, |g_c|+|g_y|={np.max(tresh_c, initial=0):.1e}+{np.max(tresh_y, initial=0):.1e}, "
                f"supp:({np.sum(suppc)}->{Nc})")

            if Nc > p.kernel.pad_size:
                p.kernel.pad_size = p.kernel.pad_size + 20

        
        

        # Plot results
        if plot_every > 0 and k % plot_every == 0:
            p.plot_forward(xk, sk, ck)

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

    return alg_out


