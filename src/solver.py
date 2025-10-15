
# import numpy as np
import jax.numpy as jnp
import time
import matplotlib.pyplot as plt
from datetime import datetime
from functools import partial
from .utils import computeProx, Phi, compute_rhs, compute_y, compute_errors
import jax 

jax.config.update("jax_enable_x64", True)  # Enable 64-bit precision    

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

    Ntrial = alg_opts.get('Ntrial', 1000)
    T = alg_opts.get('T', 300)

    # Initial guess
    u0 = alg_opts.get('u0', p.u_zero)
    uk = u0.copy()

    phi = Phi(gamma)
    yk, linear_results_int, linear_results_bnd = compute_rhs(p, uk['x'], uk['s'], uk['u'])
    norms_c = jnp.abs(uk['u'])

    # Compute initial loss, error, and etc.
    misfit = yk - y_ref
    j = obj.F(misfit)/alpha + jnp.sum(phi.phi(norms_c))
    errors = compute_errors(p, uk['x'], uk['s'], uk['u'])
    suppsize = jnp.count_nonzero(norms_c) # support size

    ck = uk['u'] # outer weights
    xk = uk['x'] # inner weights - collocation points
    sk = uk['s'] # inner weights - shape parameter (sigma)

    alg_out = {
        'xk': [xk],
        'sk': [sk],
        'ck': [ck],
        'suppc': [jnp.abs(ck) > 0],
        'supps': [suppsize],
        'js': [j],
        'tics': [0],
        'L_2': [errors['L_2']],
        'L_inf': [errors['L_inf']]
    }

    start_time = time.time()
    shape_dK = lambda dK: dK.transpose(0, 2, 1).reshape(Ndata, -1) 

    qk = jnp.sign(ck) + ck # change to robinson variale  
    # check consistency of Robinson variable
    assert ck.size == 0 or jnp.linalg.norm(ck - Prox(qk), ord=jnp.inf) < 1e-14 

    ck = Prox(qk) # Update ck, no change in default
    suppc = (jnp.abs(ck) > 0).flatten() # support of the outer weights
    suppGp = jnp.tile(suppc, dim+1) # support of the gradiento of the objective function

    # Define function equivalents
    Dphima = lambda c: (phi.dphi(jnp.abs(c)) - 1) * jnp.sign(c) # gradient of modified phi
    DDphima = lambda c: phi.ddphi(jnp.abs(c)) # hessian of modified phi


    theta_old = 1. # initial step size for line search

    print('### Start Iterations ###')
    
    pad_size = xk.shape[0]
    pad_size_dict = {
        0: pad_size,
    }
    pad_size_id = 0
    MCMC_key = jax.random.PRNGKey(p.seed) # set random key for Metropolis-Hastings algorithm
    
    for k in range(1, max_step  + 1):
        Grad_E = p.kernel.Grad_E_kappa_X_c_Xhat(xk, sk, ck, p.xhat_int)
        Grad_B = p.kernel.Grad_B_kappa_X_c_Xhat(xk, sk, ck, p.xhat_bnd)
        
        Dc_E_kappa, Dx_E_kappa, Ds_E_kappa = Grad_E['grad_c'], Grad_E['grad_X'], Grad_E['grad_S']
        Dc_B_kappa, Dx_B_kappa, Ds_B_kappa = Grad_B['grad_c'], Grad_B['grad_X'], Grad_B['grad_S']
        
        Gp_c = jnp.vstack([jnp.array(Dc_E_kappa), jnp.array(Dc_B_kappa)])
        Gp_x = jnp.vstack([jnp.array(Dx_E_kappa), jnp.array(Dx_B_kappa)])
        Gp_s = jnp.vstack([jnp.array(Ds_E_kappa), jnp.array(Ds_B_kappa)])
        
        if Gp_s.ndim == 2:
            Gp_s = Gp_s[:, :, None]
        Gp_xs = jnp.dstack([Gp_x, Gp_s])
        
        # We optimize over qk, xk, and sk
        Gp = jnp.hstack([Gp_c, shape_dK(Gp_xs)])
        Gp = Gp * suppGp[None, :] # only keep the active points


        compact_ind = jnp.argsort(~suppGp) # index used to shift all the active points to the front
        inv_compact_ind = jnp.argsort(compact_ind) # index used to shift all the active points back to the original order
      
        Gp_compact = Gp[:, compact_ind] # compacted gradient matrix 

        R1 = (1 / alpha) * (Gp_compact.T @ obj.dF(misfit)) 
        R2 = jnp.concatenate([
            ((Dphima(ck)+qk-ck) * suppc).reshape(-1, 1),
            jnp.zeros((pad_size * dim, 1))
        ]) 
        R = R1 + R2[compact_ind, :]

        # # assert the lower trunk of R is zero
        # # This is a sanity check, it should be always true, otherwise the support definition is incorrect
        # assert jnp.all(jnp.abs(R[int((dim+1)*jnp.sum(suppc)):, :]) < 1e-14), "The lower trunk of R is not zero, check the support definition."  s

        SI = obj.ddF(misfit)
        II = Gp_compact.T @ SI @ Gp_compact # Approximate Hessian 


        kpp = 0.1 * jnp.linalg.norm(obj.dF(misfit), 1) * jnp.reshape(
            jnp.sqrt(jnp.finfo(float).eps) + jnp.tile(jnp.abs(ck), (dim, 1)), -1
        )
        

        Icor_diag = jnp.concatenate([
            jnp.sqrt(jnp.finfo(float).eps) * suppc,
            kpp
        ])
        Icor_diag_compact = Icor_diag[compact_ind]
        Icor = jnp.diag(Icor_diag_compact)

        HH = (1 / alpha) * (II + Icor)

        DP_diag = jnp.concatenate([
            (jnp.abs(qk) >= 1),
            jnp.tile((jnp.abs(ck) > 0), (dim,))
        ])
        DP_diag_compact = DP_diag[compact_ind]

        DP = jnp.diag(DP_diag_compact)
        DDphi_diag = jnp.concatenate([DDphima(ck), jnp.zeros((dim * pad_size))]) * suppGp
        DDphi_diag_compact = DDphi_diag[compact_ind]
        DDphi = jnp.diag(DDphi_diag_compact)

        try:
            DR = HH @ DP + DDphi @ DP + (jnp.eye(pad_size*(1+dim)) - DP)
            dz = - jnp.linalg.solve(DR, R)
            dz = dz.flatten()
        except:
            try:
                DR = HH @ DP + (jnp.eye(pad_size*(1+dim)) - DP)
                dz = - jnp.linalg.solve(DR, R)
                dz = dz.flatten()
            except:
                print("WARNING: Singular matrix encountered.")
                alg_out["success"] = False
                break
        
        # check if dz is finite
        assert jnp.all(jnp.isfinite(dz)), "dz contains NaN or Inf values, check the problem setup."
        # # check if dz is zero in the lower trunk
        # # This is a sanity check, it should be always true, otherwise the support definition is incorrect
        # assert jnp.all(jnp.abs(dz[int(jnp.sum(suppGp)):]) < 1e-14), "dz contains non-zero values in the lower trunk, check the problem setup."

        jold, xold, sold, qold = j, xk.copy(), sk.copy(), qk.copy()
        pred = (R.T @ (DP @ dz.reshape(-1, 1))) # estimate of the descen
        theta = min(theta_old * 2, 1 - 1e-14) 
        has_descent = False

        dz = dz[inv_compact_ind] # reorder dz to the original order
        dz = dz * suppGp # only keep the active points
        while not has_descent and theta > 1e-20:
            qk = qold + theta * dz[:pad_size]
            dxs = dz[pad_size:].reshape(dim, -1).T
            xk = xold + theta * dxs[:, :d]
            sk = sold + theta * dxs[:, d:]
            ck = Prox(qk)

            yk, linear_results_int, linear_results_bnd = compute_rhs(p, xk, sk, ck)
            misfit = yk - y_ref
            norms_c = jnp.abs(ck)
            j = obj.F(misfit) / alpha + jnp.sum(phi.phi(norms_c))
           
            descent = j - jold 
            has_descent = descent <= (theta*pred + 1e-5) / 5

            if not has_descent:
                theta /= 1.5 # shrink theta

        theta_old = theta 

        if not has_descent:
            print("WARNING: Line search failed.")
            alg_out["success"] = False
            break

        suppc_new = jnp.abs(qk) > 1

        if jnp.sum(suppc_new) < jnp.sum(suppc):
            print(f" PRUNE: supp:{jnp.sum(suppc)}->{jnp.sum(suppc_new)}")
            suppc = suppc_new
            suppGp = jnp.tile(suppc, dim+1)
            
            
    
        omegas_x, omegas_s = p.sample_param(Ntrial)

        K_test_int = p.kernel.DE_kappa_X_Xhat(omegas_x, omegas_s, p.xhat_int, *linear_results_int)
        K_test_bnd = p.kernel.DB_kappa_X_Xhat(omegas_x, omegas_s, p.xhat_bnd, *linear_results_bnd)

        K_test = jnp.vstack([K_test_int, K_test_bnd])
        eta = (1 / alpha) * K_test.T @ obj.dF(misfit) 
        sh_eta = jnp.abs(Prox(eta)).flatten()
        sh_eta, sorted_ind = jnp.sort(sh_eta)[::-1], jnp.argsort(-sh_eta) 
        max_sh_eta, ind_max_sh_eta = sh_eta[0], sorted_ind[0]
    
        errors = compute_errors(p, xk, sk, ck)

        # Print iteration info
        if k % print_every == 0:
            dz_norm = jnp.linalg.norm(dz, jnp.inf) if dz.size > 0 else 0.0
            print(f"Time: {time.time() - start_time:.2f}s CGNAP iter: {k}, j={j:.6f}, supp={jnp.sum(suppc)}), "
                f"desc={descent:.1e}, dz={dz_norm:.1e}, "
                f"viol={max_sh_eta:.1e}, theta={theta:.1e}")
            
            print("L_2 error: {L_2:.3e}, L_inf error: {L_inf:.3e}, (int: {L_inf_int:.3e}, bnd: {L_inf_bnd:.3e})".format(**errors))
        

        alg_out["xk"].append(xk)
        alg_out["sk"].append(sk)
        alg_out["ck"].append(ck)
        alg_out["js"].append(j)
        alg_out["L_2"].append(errors['L_2'])
        alg_out["L_inf"].append(errors['L_inf'])
        alg_out["supps"].append(jnp.sum(suppc))
        alg_out["suppc"].append(suppc)
        alg_out["tics"].append(time.time() - start_time) 
        alg_out["success"] = True


        grad_supp_c = (1 / alpha) * (Gp_c.T @ obj.dF(misfit)) + Dphima(ck).reshape(-1, 1) + (qk - ck).reshape(-1, 1)
        grad_supp_c = grad_supp_c * suppc[:, None] # only keep the active point
        tresh_c = jnp.abs(grad_supp_c).T

        grad_supp_y = (1 / alpha) * shape_dK(Gp_xs).T @ obj.dF(misfit)
        grad_supp_y = grad_supp_y * suppGp[pad_size:, None] # only keep the active points
        tresh_y = jnp.sqrt(jnp.sum(grad_supp_y.reshape(dim, -1) ** 2, axis=0))

        tresh = tresh_c +  insertion_coef * tresh_y

        # Metropolis-Hastings step
        annealing = - 3 * jnp.log10(alpha) * jnp.max(jnp.abs(misfit)) / (jnp.max(jnp.abs(y_ref))) # A Heuristic annealing coefficient
        log_prob = -(jnp.linalg.norm(tresh, ord=jnp.inf) - max_sh_eta) / (T * annealing**2 + 1e-5)
        log_prob = jnp.clip(log_prob, -100, 100) 
        MCMC_key, subkey = jax.random.split(MCMC_key) # random key for MCMC
        if jax.random.uniform(subkey) < jnp.exp(log_prob):
            if jnp.sum(suppc) == pad_size:
                idx = pad_size
                pad_size_id += 1
                if pad_size_id not in pad_size_dict:
                    pad_size_dict[pad_size_id] = pad_size * 2
                    recompile = True
                else:
                    recompile = False
                pad_size = pad_size_dict[pad_size_id]
                # pad ck, sk, qk to pad_size
                ck = jnp.pad(ck, (0, pad_size - len(ck)), constant_values=0)
                xk = jnp.pad(xk, ((0, pad_size - xk.shape[0]), (0, 0)), constant_values=0)
                sk = jnp.pad(sk, ((0, pad_size - sk.shape[0]), (0, 0)), constant_values=0)
                qk = jnp.pad(qk, (0, pad_size - len(qk)), constant_values=0)
                suppc = jnp.pad(suppc, (0, pad_size - len(suppc)), constant_values=False)
                print(f"\n#### PAD_SIZE increased to {pad_size}, RECOMPILING: {recompile} ####\n")
            else:
                idx = jnp.argmax(~suppc)

            suppc = suppc.at[idx].set(True)
            suppGp = jnp.tile(suppc, dim+1)
            qk = qk.at[idx].set(-jnp.sign(eta[ind_max_sh_eta])[0])  # insert a new point
            xk = xk.at[idx, :].set(omegas_x[ind_max_sh_eta, :])
            if sk.ndim == 1:
                sk = sk.at[idx].set(omegas_s[ind_max_sh_eta])
            else:
                sk = sk.at[idx, :].set(omegas_s[ind_max_sh_eta, :])


            print(f"  INSERT: viol={max_sh_eta:.2e}, |g_c|+|g_y|={jnp.max(tresh_c, initial=0):.1e}+{jnp.max(tresh_y, initial=0):.1e}, "
                f"supp:({jnp.sum(suppc)-1}->{jnp.sum(suppc)})")

        # Plot results
        if k % plot_every == 0:            
            p.plot_forward(xk, sk, ck, suppc)

        # Stopping criterion
        if jnp.abs(pred * theta) < (TOL / alpha) and  max_sh_eta < (TOL / alpha):
            dz_norm = jnp.linalg.norm(dz, jnp.inf) if dz.size > 0 else 0.0  
            print(f"CGNAP iter: {k}, j={j:.6f}, supp=({jnp.sum(suppc)}), "
                f"desc={descent:.1e}, dz={dz_norm:.1e}, "
                f"viol={max_sh_eta:.1e}, theta={theta:.1e}")
            
            print("L_2 error: {L_2:.3e}, L_inf error: {L_inf:.3e}, (int: {L_inf_int:.3e}, bnd: {L_inf_bnd:.3e})".format(**errors))   
            print(f"Converged in {k} iterations")
            break


        if jnp.sum(suppc) < pad_size // 3 and k >= 100:
            pad_size_id -= 1
            if pad_size_id not in pad_size_dict:
                pad_size_dict[pad_size_id] = max(pad_size // 2, 1)
                recompile = True
            else:
                recompile = False
            pad_size = pad_size_dict[pad_size_id]
            # prune ck, sk, qk, xk, suppc
            ck = ck[suppc]
            qk = qk[suppc]
            sk = sk[suppc, :]
            xk = xk[suppc, :]
            suppc = suppc[suppc]
            # pad the arrays to pad_size
            ck = jnp.pad(ck, (0, pad_size - len(ck)), constant_values=0.)
            qk = jnp.pad(qk, (0, pad_size - len(qk)), constant_values=0.)
            sk = jnp.pad(sk, ((0, pad_size - sk.shape[0]), (0, 0)), constant_values=0.)
            xk = jnp.pad(xk, ((0, pad_size - xk.shape[0]), (0, 0)), constant_values=0.)
            suppc = jnp.pad(suppc, (0, pad_size - len(suppc)), constant_values=False)
            suppGp = jnp.tile(suppc, dim+1)
            print(f"\n#### PAD_SIZE decreased to {pad_size}, RECOMPILING: {recompile} ####\n")

    
    if plot_final:
        p.plot_forward(xk, sk, ck, suppc)

    return alg_out


