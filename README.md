# SparseKernelPDE

## 📌 Description
Code for the paper ____


## 📦 Installation
<!-- To install this project, follow these steps:

```sh
# Clone the repository
git clone https://github.com/your-username/your-repo.git

# Navigate to the project directory
cd your-repo
pip install -r requirements.txt  # For Python projects
``` -->

## The architecture of this code
### In the `src` folder:
- `GaussianKernel.py` defines the generic Gaussian Kernel, which includes implementation of derivatives with respect to $c$, $x$ ($y$ in paper), and $\sigma$. 
- `solve_TV_CGNAP.py` provides the algorithm (almost identical to the matlab version).
- `utils.py` includes `Objective`, `Phi`, `computeProximal`; `shapeParser` is the tool to pre-allocate static shape for `JAX.jit` compilation.

### In the `PDE` folder:
We define the basic properties of PDE problem here, including $D$, $\Omega$, right-hand-side term ($f$), and exact solution; we also integrates specific $\mathcal{E}$ and $\mathcal{B}$ into the class of Gaussian Kernel (by inheriting `GaussianKernel` in `GaussianKernel.py`).


### Execution files
We run the code by running `test_*.py`, which defines all the hyperparameters.

e.g.: 
CommandLine: 
```
python test_SemiLinear.py --Nobs 40 --sampling uniform --scale 1000 --alpha 1e-3 --TOL 1e-3 --T 50 
```


## Questions:

### Semi-Smooth Newton

#### Singular matrix
According to my experiments, I found the algorithm is not robust and I occasionally encounter singular matrices. I paste the code piece I put here

```python

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
```        
        
(Here I abandon the setting of active point, which I will explain the reason why later.) I haven't investigated what is the cause of a singular matrix. Also, how is this correction term defined?

#### Failed line search
Line search fails more often (especially when I loosen the threshold for admitting new nodes too much, which kind of makes sense). 

### Admitting new points
Here's the definition of threshold
```python
        grad_supp_c = (1 / alpha) * (Gp_c.T @ obj.dF(misfit)) + Dphima(ck).reshape(-1, 1) + (qk - ck).reshape(-1, 1)

        tresh_c = np.abs(grad_supp_c).T
        grad_supp_y = (1 / alpha) * shape_dK(Gp_xs).T @ obj.dF(misfit)
        tresh_y = np.sqrt(np.sum(grad_supp_y.reshape(dim, -1) ** 2, axis=0))

        # tresh = tresh_c + 0.01 * tresh_y # We need to change this.
        tresh = tresh_c + 0.01 * tresh_y
```

In practice, I loosen the therehold for admitting new nodes for speed concern. I have tried 2 ways:

1. Assign a `insertion_coef`
    ```python
    if max_sh_eta > insertion_coef * np.linalg.norm(tresh, ord=np.inf):
    
    ```
    Sometimes I need to assign `insertion_coef` to be very small ($1e-3$) to have reasonable convergence speed.

2. Use Metroplis-Hasting algorithm: 
    ``` python
    annealing = - 3 * np.log10(alpha) * np.max(np.abs(misfit)) / np.max(np.abs(y_ref))
    if np.random.rand() < np.exp(-(np.linalg.norm(tresh, ord=np.inf) - max_sh_eta) / (T * annealing**2 + 1e-5)):
    ```
    This scheme admits more points when the approximation is not close.
    

### Active point setting

Active point setting: at each iteration, we only update those nodes that are worthy of updaiing. That is done by sorting `tresh`, which is the magnitude of gradient (`G_p`) of all the nodes. Based on my understanding, this is to save computation. 

However, in the nonlinear case, the gradient of a node with respect to the weights is changing with the solution even though it is not updated. To get the correct updated `tresh`, we need to compute the gradient for all the nodes, which is of same complexity to optimize all the nodes at the same time. 
