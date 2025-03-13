from functools import partial
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", False)
from .utils import shapeParser

class GaussianKernel:
    def __init__(self, power=4.5, d=2):
        """
        Initialize the Gaussian kernel with scale S, power, and dimension d.
        Args:
            power (float): Power applied to the scale parameter.
            d (int): Dimensionality of the data.
        """
        self.power = power
        self.d = d
        self.pad_size = 2
        self.sigma_max = 1.0
    
    def sigma(self, s):
        # Questions: Do we need to put a scalar here?
        exp_s = jnp.exp(s)
        return self.sigma_max * exp_s / (1 + exp_s)


    # @partial(jax.jit, static_argnums=(0,))  # we will compile it later in the child class
    def gauss(self, x, s, xhat):
        """Compute kernel between single points x and xhat."""
        boundary_1 = jnp.ones(x.shape)
        boundary_2 = -jnp.ones(x.shape)
        # mask = jnp.prod(xhat - boundary_1) * jnp.prod(boundary_2 - xhat)
        squared_dist = jnp.sum((x - xhat) ** 2)
        sigma = self.sigma(s)
        return ((sigma ** self.power) * jnp.exp(-squared_dist / (2 * sigma**2))) / (
            (jnp.sqrt(2 * jnp.pi) * sigma) ** self.d
        )

    @shapeParser
    @partial(jax.jit, static_argnums=(0, 1))
    def gauss_X(self, X_shape, X, S, xhat):
        return jax.vmap(self.gauss, in_axes=(0, 0, None))(X, S, xhat)

    @shapeParser
    @partial(jax.jit, static_argnums=(0, 1))
    def gauss_X_Xhat(self, X_shape, X, S, Xhat):
        return jax.vmap(self.gauss_X, in_axes=(None, None, 0))(X, S, Xhat)

    @shapeParser
    @partial(jax.jit, static_argnums=(0, 1))
    def gauss_X_c(self, X_shape, X, S, c, xhat):
        return jnp.dot(c, self.gauss_X(X, S, xhat))

    @partial(shapeParser, pad=True)
    @partial(jax.jit, static_argnums=(0, 1))
    def gauss_X_c_Xhat(self, X_shape, X, S, c, Xhat):
        return jax.vmap(self.gauss_X_c, in_axes=(None, None, None, 0))(X, S, c, Xhat)
        
    @shapeParser
    @partial(jax.jit, static_argnums=(0, 1))
    def P_gauss_X_c(self, X_shape, X, S, c, xhat):
        pass

    @partial(shapeParser, pad=True)
    @partial(jax.jit, static_argnums=(0, 1))
    def P_gauss_X_c_Xhat(self, X_shape, X, S, c, Xhat):
        return jax.vmap(self.P_gauss_X_c, in_axes=(None, None, None, 0))(X, S, c, Xhat)

    @partial(shapeParser, pad=True)
    @partial(jax.jit, static_argnums=(0, 1))    
    def Dx_gauss_X_c(self, X_shape, X, S, c, xhat):
        return jax.grad(self.gauss_X_c, argnums=0)(X, S, c, xhat)

    @partial(shapeParser, pad=True)
    @partial(jax.jit, static_argnums=(0, 1)) 
    def Dx_gauss_X_c_Xhat(self, X_shape, X, S, c, Xhat):
        return jax.vmap(self.Dx_gauss_X_c, in_axes=(None, None, None, 0))(X, S, c, Xhat)

    @partial(shapeParser, pad=True)
    @partial(jax.jit, static_argnums=(0, 1)) 
    def Ds_gauss_X_c(self, X_shape, X, S, c, xhat):
        return jax.grad(self.gauss_X_c, argnums=1)(X, S, c, xhat)
    
    @partial(shapeParser, pad=True)
    @partial(jax.jit, static_argnums=(0, 1)) 
    def Ds_gauss_X_c_Xhat(self, X_shape, X, S, c, Xhat):
        return jax.vmap(self.Ds_gauss_X_c, in_axes=(None, None, None, 0))(X, S, c, Xhat)
    
    @partial(shapeParser, pad=True)
    @partial(jax.jit, static_argnums=(0, 1)) 
    def Dc_gauss_X_c(self, X_shape, X, S, c, xhat):
        return jax.grad(self.gauss_X_c, argnums=2)(X, S, c, xhat)
    
    @partial(shapeParser, pad=True)
    @partial(jax.jit, static_argnums=(0, 1)) 
    def Dc_gauss_X_c_Xhat(self, X_shape, X, S, c, Xhat):
        return jax.vmap(self.Dc_gauss_X_c, in_axes=(None, None, None, 0))(X, S, c, Xhat)


    @partial(shapeParser, pad=True)
    @partial(jax.jit, static_argnums=(0, 1))
    def Dx_P_gauss_X_c(self, X_shape, X, S, c, xhat):
        return jax.grad(self.P_gauss_X_c, argnums=0)(X, S, c, xhat)
    
    @partial(shapeParser, pad=True)
    @partial(jax.jit, static_argnums=(0, 1))
    def Dx_P_gauss_X_c_Xhat(self, X_shape, X, S, c, Xhat):
        return jax.vmap(self.Dx_P_gauss_X_c, in_axes=(None, None, None, 0))(X, S, c, Xhat)
    
    @partial(shapeParser, pad=True)
    @partial(jax.jit, static_argnums=(0, 1))
    def Ds_P_gauss_X_c(self, X_shape, X, S, c, xhat):
        return jax.grad(self.P_gauss_X_c, argnums=1)(X, S, c, xhat)
    
    @partial(shapeParser, pad=True)
    @partial(jax.jit, static_argnums=(0, 1))
    def Ds_P_gauss_X_c_Xhat(self, X_shape, X, S, c, Xhat):
        return jax.vmap(self.Ds_P_gauss_X_c, in_axes=(None, None, None, 0))(X, S, c, Xhat)
    
    @partial(shapeParser, pad=True)
    @partial(jax.jit, static_argnums=(0, 1))
    def Dc_P_gauss_X_c(self, X_shape, X, S, c, xhat):
        return jax.grad(self.P_gauss_X_c, argnums=2)(X, S, c, xhat)
    
    @partial(shapeParser, pad=True)
    @partial(jax.jit, static_argnums=(0, 1))
    def Dc_P_gauss_X_c_Xhat(self, X_shape, X, S, c, Xhat):
        return jax.vmap(self.Dc_P_gauss_X_c, in_axes=(None, None, None, 0))(X, S, c, Xhat)
    

    @partial(jax.jit, static_argnums=(0,))
    def DP_gauss(self, x, s, xhat, u):
        # pde specific
        pass

    @partial(jax.jit, static_argnums=(0,))
    def DP_gauss_X(self, X, S, xhat, u):
        return jax.vmap(self.DP_gauss, in_axes=(0, 0, None, None))(X, S, xhat, u)

    @partial(jax.jit, static_argnums=(0,))
    def DP_gauss_X_Xhat(self, X, S, Xhat, U):
        """
        Compute the linearized PDE operator of the Gaussian kernel at u.
        """
        return jax.vmap(self.DP_gauss_X, in_axes=(None, None, 0, 0))(X, S, Xhat, U)
