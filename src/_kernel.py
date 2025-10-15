from functools import partial
import jax
import jax.numpy as jnp
from jax import config
# from utils import shapeParser

class _Kernel:
    def __init__(self):
        """
        Initialize the Gaussian kernel with scale S, power, and dimension d.
        Args:
            power (float): Power applied to the scale parameter.
            d (int): Dimensionality of the data.
        """
        self.linear_E = ()
        self.linear_B = ()
        self.DE = ()
        self.DB = ()
    
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

    @partial(jax.jit, static_argnums=(0,))
    def kappa_X(self, X, S, xhat):
        return jax.vmap(self.kappa, in_axes=(0, 0, None))(X, S, xhat)

    @partial(jax.jit, static_argnums=(0,))
    def kappa_X_Xhat(self, X, S, Xhat):
        return jax.vmap(self.kappa_X, in_axes=(None, None, 0))(X, S, Xhat)

    @partial(jax.jit, static_argnums=(0,))
    def kappa_X_c(self, X, S, c, xhat):
        return jnp.dot(c, self.kappa_X(X, S, xhat))

    @partial(jax.jit, static_argnums=(0,))
    def kappa_X_c_Xhat(self, X, S, c, Xhat):
        return jax.vmap(self.kappa_X_c, in_axes=(None, None, None, 0))(X, S, c, Xhat)
    
    
    ############################################################
    ##### The following lines are specific to the PDE case #####
    ############################################################
    
#     @partial(jax.jit, static_argnums=(0,))
#     def Lap_kappa_X_c(self, X, S, c, xhat):
#         return jnp.trace(jax.hessian(self.kappa_X_c, argnums=3)(X, S, c, xhat))
    
    
#     @shapeParser
#     @partial(jax.jit, static_argnums=(0,))
#     def Lap_kappa_X_c_Xhat(self, X, S, c, Xhat): 
#         return jax.vmap(self.Lap_kappa_X_c, in_axes=(None, None, None, 0))(X, S, c, Xhat)

    @partial(jax.jit, static_argnums=(0,))
    def E_kappa_X_c(self, X, S, c, xhat):
        # return - self.Lap_kappa_X_c(X, S, c, xhat) + self.kappa_X_c(X, S, c, xhat) ** 3
        pass

    @partial(jax.jit, static_argnums=(0,))
    def B_kappa_X_c(self, X, S, c, xhat):
        # return self.kappa_X_c(X, S, c, xhat)
        pass
    
    ############################################################
    ############################################################

    # We do the less automated way of computing PDE operators in the vectorized case
    # This is becasue we need to compute the linearized PDE operator at u, which is a function of the linearized PDE operator at u.
    # This is a bit tricky to do in a fully automated way.
    
    # @partial(jax.jit, static_argnums=(0,))
    def linear_E_results_X_c_Xhat(self, X, S, c, Xhat):
        linear_results = []
        for func in self.linear_E:
            linear_results.append(func(X, S, c, Xhat))
        return tuple(linear_results)
    
    def linear_B_results_X_c_Xhat(self, X, S, c, Xhat):
        linear_results = []
        for func in self.linear_B:
            linear_results.append(func(X, S, c, Xhat))
        return tuple(linear_results)

    def E_kappa_X_c_Xhat(self, *linear_results):
        # return - linear_results[1] + linear_results[0] ** 3
        pass
 
    def B_kappa_X_c_Xhat(self, *linear_results):
        # return linear_results[0]
        pass


    @partial(jax.jit, static_argnums=(0,))
    def Grad_E_kappa_X_c(self, X, S, c, xhat):
        grads = jax.grad(self.E_kappa_X_c, argnums=(0, 1, 2))(X, S, c, xhat)

        return {'grad_X': grads[0], 'grad_S': grads[1], 'grad_c': grads[2]}


    # @shapeParser
    @partial(jax.jit, static_argnums=(0,))
    def Grad_E_kappa_X_c_Xhat(self, X, S, c, Xhat):
        return jax.vmap(self.Grad_E_kappa_X_c, in_axes=(None, None, None, 0))(X, S, c, Xhat)


    
    @partial(jax.jit, static_argnums=(0,))
    def Grad_B_kappa_X_c(self, X, S, c, xhat):
        grads = jax.grad(self.B_kappa_X_c, argnums=(0, 1, 2))(X, S, c, xhat)
        return {'grad_X': grads[0], 'grad_S': grads[1], 'grad_c': grads[2]}
    

    # @shapeParser
    @partial(jax.jit, static_argnums=(0,)) 
    def Grad_B_kappa_X_c_Xhat(self, X, S, c, Xhat):
        return jax.vmap(self.Grad_B_kappa_X_c, in_axes=(None, None, None, 0))(X, S, c, Xhat)
        

    @partial(jax.jit, static_argnums=(0,))
    def DE_kappa(self, x, s, xhat, *args):
        # return -jnp.trace(jax.hessian(self.kappa, argnums=2)(x, s, xhat)) + 3 * args[0] ** 2 * self.kappa(x, s, xhat)
        pass

    @partial(jax.jit, static_argnums=(0,))
    def DE_kappa_X(self, X, S, xhat, *args):
        return jax.vmap(self.DE_kappa, in_axes=(0, 0, None,) + (None,)*len(args))(X, S, xhat, *args)

    @partial(jax.jit, static_argnums=(0,))
    def DE_kappa_X_Xhat(self, X, S, Xhat, *linear_results):
        args = []
        for key in self.DE:
            args.append(linear_results[key])
        args = tuple(args)
        return jax.vmap(self.DE_kappa_X, in_axes=(None, None, 0,) + (0,)*len(args))(X, S, Xhat, *args)
    
    @partial(jax.jit, static_argnums=(0,))
    def DB_kappa(self, x, s, xhat, *args):
        # return self.kappa(x, s, xhat)
        pass
    
    @partial(jax.jit, static_argnums=(0,))
    def DB_kappa_X(self, X, S, xhat, *args):
        return jax.vmap(self.DB_kappa, in_axes=(0, 0, None) + (None,)*len(args))(X, S, xhat, *args)

    @partial(jax.jit, static_argnums=(0,))
    def DB_kappa_X_Xhat(self, X, S, Xhat, *linear_results):
        """
        Compute the linearized PDE operator of the Gaussian kernel at u.
        """
        args = []
        for key in self.DB:
            args.append(linear_results[key])
        args = tuple(args)
        return jax.vmap(self.DB_kappa_X, in_axes=(None, None, 0)+(0,)*len(args))(X, S, Xhat, *args)
    

    
    
    
