from functools import partial
import jax
import jax.numpy as jnp
from jax import config
from jax.scipy.special import gamma
import math
from ._kernel import _Kernel
# from utils import shapeParser

class GaussianKernel(_Kernel):
    def __init__(self, power=4.5, d=2, sigma_max=1.0, sigma_min=1e-3, anisotropic=False):
        """
        Initialize the Gaussian kernel with scale S, power, and dimension d.
        Args:
            power (float): Power applied to the scale parameter.
            d (int): Dimensionality of the data.
        """
        super().__init__()
        self.power = power
        self.d = d
        self.pad_size = 2
        self.anisotropic = anisotropic

        if self.anisotropic:
            sigma_max = jnp.array(sigma_max)
            if sigma_max.shape == ():
                sigma_max = sigma_max * jnp.ones(d)
            else:
                if sigma_max.shape != (d,):
                    raise ValueError("sigma_max must be a scalar or a vector of length d")
            self.sigma_max = sigma_max

            sigma_min = jnp.array(sigma_min)
            if sigma_min.shape == ():
                sigma_min = sigma_min * jnp.ones(d)
            else:
                if sigma_min.shape != (d,):
                    raise ValueError("sigma_min must be a scalar or a vector of length d")
            self.sigma_min = sigma_min
        else:
            assert type(sigma_max) == float or type(sigma_max) == int
            assert type(sigma_min) == float or type(sigma_min) == int
            self.sigma_max = sigma_max
            self.sigma_min = sigma_min

        #### The following lines are specific to the PDE case #####
#         self.linear_E = (self.gauss_X_c_Xhat, self.Lap_gauss_X_c_Xhat)
        
#         self.linear_B = (self.gauss_X_c_Xhat,)
        
#         self.DE = (0,) 
#         self.DB = ()    
    
    def sigma(self, s):
        # Questions: Do we need to put a scalar here?
        exp_s = jnp.exp(s)
        return self.sigma_min + (self.sigma_max - self.sigma_min) * exp_s / (1 + exp_s)


    # @partial(jax.jit, static_argnums=(0,))  # we will compile it later in the child class
    def kappa(self, x, s, xhat):
        """Compute kernel between single points x and xhat."""
        
        if self.anisotropic:
            squared = (x - xhat) ** 2
            sigma = self.sigma(s)
            weighted_squared_dist = 0
            for i in range(self.d):
                weighted_squared_dist += squared[i] / (2 * sigma[i]**2)
            
            det_sigma = jnp.prod(sigma)
            return ((det_sigma ** (self.power / 2)) * jnp.exp(-weighted_squared_dist)) / (
                (jnp.sqrt(2 * jnp.pi) ** self.d ) * det_sigma
            )
        else:
            squared_dist = jnp.sum((x - xhat) ** 2)
            sigma = self.sigma(s)[0]
            return ((sigma ** self.power) * jnp.exp(-squared_dist / (2 * sigma**2))) / (
                (jnp.sqrt(2 * jnp.pi) * sigma) ** self.d
            )
    

class MaternKernel(_Kernel):
    r"""
    Isotropic Matérn kernel with two scaling modes:

      - **mode="unit_integral"**:
        \[
        \phi(x) = A_{\nu,d}(\sigma)\, k_{\nu}(\|x-\hat{x}\|;\sigma),
        \quad
        A_{\nu,d}(\sigma) = 
        \frac{\Gamma(\nu)\, \nu^{d/2}}{(\sqrt{2\pi}\,\sigma)^d\, \Gamma(\nu + d/2)},
        \]
        so that
        \[
        \int_{\mathbb{R}^d} \phi(x)\,dx = 1
        \quad \text{(exact, for any }\nu>0\text{).}
        \]

      - **mode="pde_balanced"**:
        \[
        \phi(x) =
        \frac{\sigma^{\texttt{power}}}{(\sqrt{2\pi}\,\sigma)^d}\,
        k_{\nu}(\|x-\hat{x}\|;\sigma),
        \]
        matching the Gaussian PDE scaling.
        For 2nd-order PDEs, use \(\texttt{power} \approx d + 2.01\).

    Matérn shape (unit height at \(r=0\)):
    \[
    k_{\nu}(r;\sigma)
      = C_{\nu}\,
        \left(\frac{\sqrt{2\nu}\,r}{\sigma}\right)^{\nu}
        K_{\nu}\!\left(\frac{\sqrt{2\nu}\,r}{\sigma}\right),
      \quad
      C_{\nu} = \frac{2^{1-\nu}}{\Gamma(\nu)},
    \]
    with closed forms for \(\nu \in \{1/2,\, 3/2,\, 5/2\}\).
    """

    def __init__(self,
                 nu=2.5,
                 d=2,
                 sigma_max=1.0,
                 sigma_min=1e-3,
                 power=4.5):             # used only in pde_balanced; default d+2.01
        super().__init__()
        self.nu = float(nu)
        self.d = int(d)
        self.pad_size = 2
        self.power = power 

        assert isinstance(sigma_max, (float, int))
        assert isinstance(sigma_min, (float, int))
        self.sigma_max = float(sigma_max)
        self.sigma_min = float(sigma_min)

    # same logistic map as your Gaussian
    def sigma(self, s):
        exp_s = jnp.exp(s)
        return self.sigma_min + (self.sigma_max - self.sigma_min) * exp_s / (1.0 + exp_s)

    # -------- Matérn shape (unit height at r=0) --------
    def _matern_shape_iso(self, r, ell):
        nu = self.nu
        ell = ell + jnp.finfo(float).eps
        r = jnp.maximum(r, 0.0)

        # Fast stable closed forms
        if math.isclose(nu, 0.5):
            return jnp.exp(-r / ell)
        elif math.isclose(nu, 1.5):
            c = jnp.sqrt(3.0) * r / ell
            return (1.0 + c) * jnp.exp(-c)
        elif math.isclose(nu, 2.5):
            c = jnp.sqrt(5.0) * r / ell
            return (1.0 + c + (c**2)/3.0) * jnp.exp(-c)
        elif math.isclose(nu, 3.5):
            c = jnp.sqrt(7.0) * r / ell
            return (1.0 + c + (c**2)/3.0 + (c**3)/15.0) * jnp.exp(-c)
        elif math.isclose(nu, 4.5):
            c = jnp.sqrt(9.0) * r / ell  # √(2ν) = √9 = 3
            return (1.0 + c + (c**2)/3.0 + (c**3)/15.0 + (c**4)/105.0) * jnp.exp(-c)
        else:
            # z = jnp.sqrt(2.0 * nu) * r / ell
            # C = (2.0 ** (1.0 - nu)) / (gamma(nu) + jnp.finfo(float).eps)  # ensures k(r→0)=1
            # return jnp.where(z < 1e-8, 1.0, C * (z ** nu) * kv(nu, z))
            raise NotImplementedError("Only nu in {0.5,1.5,2.5,3.5,4.5} are implemented.")

    # -------- Scaling factors --------
    def _factor(self, ell):
        # A_{ν,d}(σ) = Γ(ν) ν^{d/2} / [ (√(2π) σ)^d Γ(ν + d/2) ]
        nu = self.nu
        power = self.power
        num = (ell**power) * gamma(nu) * (nu ** (self.d / 2.0))
        denom = ((jnp.sqrt(2.0 * jnp.pi) * ell) ** self.d) 
        denom *= (gamma(nu + self.d / 2.0) + jnp.finfo(float).eps)
        return num / denom

    # -------- Single pair evaluation --------
    def kappa(self, x, s, xhat):
        ell = self.sigma(s)[0]  # isotropic
        r = jnp.linalg.norm(x - xhat)
        base = self._matern_shape_iso(r, ell)
        factor = self._factor(ell) 

        return factor * base
    


class WendlandKernel(_Kernel):
    r"""
    Isotropic Wendland kernel (compact support) with PDE-oriented scaling:
        \kappa(x) = [ \sigma^{power} / ( (√(2π) \sigma)^d ) ] * \psi_{d,k}( \|x-y\| / \sigma),
    where \psi_{d,k} is the Wendland function of smoothness parameter k (C^{2k}).

    Common choices:
        k=0: C^0
        k=1: C^2
        k=2: C^4   
    """

    def __init__(self,
                 d=2,
                 k=1,                    # smoothness index: 0 (C^0), 1 (C^2), 2 (C^4)
                 sigma_max=1.0,
                 sigma_min=1e-3,
                 power=4.5):            # PDE scaling exponent; default ~ d+2.01 for 2nd-order PDEs
        super().__init__()
        self.d = int(d)
        self.k = int(k)
        self.pad_size = 2
        self.power = float(power)   

        assert isinstance(sigma_max, (float, int))
        assert isinstance(sigma_min, (float, int))
        self.sigma_max = float(sigma_max)
        self.sigma_min = float(sigma_min)

    # logistic map for lengthscale (same as your Gaussian/Matern)
    def sigma(self, s):
        exp_s = jnp.exp(s)
        return self.sigma_min + (self.sigma_max - self.sigma_min) * exp_s / (1.0 + exp_s)

    # \ell = floor(d/2) + k + 1
    def _l_param(self):
        return (self.d // 2) + self.k + 1

    # Wendland \psi_{d,k}(r) for k in {0,1,2}, r >= 0, with unit support (r<=1)
    def _wendland_psi(self, r):
        r = jnp.maximum(r, 0.0)
        l = self._l_param()
        r_compact = jnp.maximum(1.0 - r, 0.0)

        if self.k == 0:
            # C^0
            psi = r_compact ** (l)
        elif self.k == 1:
            # C^2
            psi = (r_compact ** (l + 1)) * (1.0 + (l + 1.0) * r)
        elif self.k == 2:
            # C^4
            c2 = (l * l + 4.0 * l + 3.0) / 3.0
            psi = (r_compact ** (l + 2)) * (1.0 + (l + 2.0) * r + c2 * (r * r))
        else:
            raise ValueError("This implementation currently supports k \in \{0,1,2\}.")
        
        return psi

    # PDE-oriented scaling factor: σ^{power} / ( (√(2π) σ)^d )
    def _factor(self, sigma):
        denom = (jnp.sqrt(2.0 * jnp.pi) * sigma) ** self.d
        return (sigma ** self.power) / (denom + jnp.finfo(float).eps)

    # Single pair evaluation: \kappa(x, s; \hat{x})
    def kappa(self, x, s, xhat):
        sigma = self.sigma(s)[0]  # isotropic \sigma
        r = jnp.linalg.norm(x - xhat) / (sigma + jnp.finfo(float).eps)
        base = self._wendland_psi(r)
        factor = self._factor(sigma)
        return factor * base