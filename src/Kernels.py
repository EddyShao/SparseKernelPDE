from functools import partial
import jax
import jax.numpy as jnp
from jax import config
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
    

    
    
    
