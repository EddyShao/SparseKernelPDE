# Sparse RBFNet for solving nonlinear PDEs

##  Description
Code for the paper [Solving Nonlinear PDEs with Sparse Radial Basis Function Networks](https://arxiv.org/abs/2505.07765).


## Installation
The code is tested with Python 3.9. After activating your virtual environment, install the required dependencies by running:
```
pip install -r requirements.txt
```

## Code Structure
### In the `src` folder:
- `GaussianKernel.py` defines the generic Gaussian Kernel.
- `solver*.py` defines solvers for various usages. 
    * `solver.py`: generic solver;
    * `solver_active.py`: Extends the generic solver with active point (block descent) functionality. Note: While this solver was primarily used in our experiments, the results reported in the paper do not utilize the active point feature.
    * `solver_active_H1.py`: A variant of the solver that enforces boundary conditions using the $H^1$ norm.

### In the `PDE` folder:
We define the fundamental components of the PDE problem here, including domains ($D$ and $\Omega$), right-hand-side term, and exact solution; we also integrates specific $\mathcal{E}$ and $\mathcal{B}$ into the definition of Gaussian Kernel.

### In the `scripts` folder:
Each Python script in this folder runs a specific experiment or test configuration. To execute a script, use the following command format:

```
python scripts/<script_name>.py [--arguments]
```
For example, to demonstrate the use in solving PDEs covered in the paper, please run the following commands:

```
python scripts/test_SemiLinear_Sines.py --Nobs 20  --scale 0 --alpha 1e-3 --TOL 1e-3 --T 300 --plot_final 
python scripts/test_SemiLinear_two_bump_adaptive.py --Nobs 30 --scale 1000 --alpha 1e-2 --T 30 --TOL 1e-5 --plot_final
python scripts/test_SemiLinearHighDim.py --Nobs 8 --scale 3000 --alpha 1e-4 --TOL 1e-4 --T 1e-4 --d 4 --plot_final
python scripts/test_Burgers1Ddt_explicit.py --Nobs 40 --scale 0 --alpha 1e-8 --T 1e4 --TOL 1e-8 --dt 0.001
python scripts/test_Eikonal_regularized.py --Nobs 20 --scale 0 --alpha 1e-4 --T 5e-5 --TOL 1e-7 --epsilon 1e-1 --plot_final
python scripts/test_Eikonal_viscosity.py --Nobs 30 --scale 0 --alpha 1e-6 --T 5e-2 --TOL 1e-6 --epsilon 1e-2 --plot_final
```

## Citation

```
@article{shao2025solving,
  title={Solving Nonlinear PDEs with Sparse Radial Basis Function Networks},
  author={Shao, Zihan and Pieper, Konstantin and Tian, Xiaochuan},
  journal={arXiv preprint arXiv:2505.07765},
  year={2025}
}
```

### Acknowledgements
We would like to thank the authors of the following repository for making their implementation publicly available, which provided inspiration and reference for parts of our codebase:
* [NonLinPDEs-GPsolver](https://github.com/yifanc96/NonLinPDEs-GPsolver), 2021