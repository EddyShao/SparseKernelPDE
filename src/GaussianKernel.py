from functools import partial
import jax
import jax.numpy as jnp
from .utils import shapeParser

class GaussianKernel:
    def __init__(self, power=4.5, d=2, sigma_max=1.0, sigma_min=1e-3, anisotropic=False):
        """
        Initialize the Gaussian kernel with scale S, power, and dimension d.
        Args:
            power (float): Power applied to the scale parameter.
            d (int): Dimensionality of the data.
        """
        self.power = power
        self.d = d
        self.pad_size = 10
        self.anisotropic = anisotropic

        if self.anisotropic:
            sigma_max = jnp.array(sigma_max)
            if sigma_max.shape == ():
                sigma_max = sigma_max * jnp.ones(d)
            else:
                if sigma_max.shape != (d,):
                    raise ValueError("sigma_max must be a scalar or a vector of length d")
            self.sigma_max = sigma_max
            print(self.sigma_max)

            sigma_min = jnp.array(sigma_min)
            if sigma_min.shape == ():
                sigma_min = sigma_min * jnp.ones(d)
            else:
                if sigma_min.shape != (d,):
                    raise ValueError("sigma_min must be a scalar or a vector of length d")
            self.sigma_min = sigma_min
            print(self.sigma_min)
        
        else:
            assert type(sigma_max) == float or type(sigma_max) == int
            assert type(sigma_min) == float or type(sigma_min) == int
            self.sigma_max = sigma_max
            self.sigma_min = sigma_min

        ##### The following lines are specific to the PDE case #####
        # self.linear = {
        #     "Id": self.gauss_X_c_Xhat,
        #     "Lap": self.Lap_gauss_X_c_Xhat,
        # }  
        # self.DE = ['Id']
        # self.DB = []     
    
    def sigma(self, s):
        # Questions: Do we need to put a scalar here?
        exp_s = jnp.exp(s)
        return self.sigma_min + (self.sigma_max - self.sigma_min) * exp_s / (1 + exp_s)


    # @partial(jax.jit, static_argnums=(0,))  # we will compile it later in the child class
    def gauss(self, x, s, xhat):
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
    
    
    ############################################################
    ##### The following lines are specific to the PDE case #####
    ############################################################
    
    # @shapeParser
    # @partial(jax.jit, static_argnums=(0, 1))
    # def Lap_gauss_X_c(self, X_shape, X, S, c, xhat):
    #     return jnp.trace(jax.hessian(self.gauss_X_c, argnums=3)(X, S, c, xhat))
    
    # @partial(shapeParser, pad=True)
    # @partial(jax.jit, static_argnums=(0, 1))
    # def Lap_gauss_X_c_Xhat(self, X_shape, X, S, c, Xhat): 
    #     return jax.vmap(self.Lap_gauss_X_c, in_axes=(None, None, None, 0))(X, S, c, Xhat)

    @shapeParser
    @partial(jax.jit, static_argnums=(0, 1))
    def E_gauss_X_c(self, X_shape, X, S, c, xhat):
        # return - self.Lap_gauss_X_c(X, S, c, xhat) + self.gauss_X_c(X, S, c, xhat) ** 3
        pass

    @shapeParser
    @partial(jax.jit, static_argnums=(0, 1))
    def B_gauss_X_c(self, X_shape, X, S, c, xhat):
        # return self.gauss_X_c(X, S, c, xhat)
        pass
    
    ############################################################
    ############################################################

    # We do the less automated way of computing PDE operators in the vectorized case
    # This is becasue we need to compute the linearized PDE operator at u, which is a function of the linearized PDE operator at u.
    # This is a bit tricky to do in a fully automated way.
    
    @partial(jax.jit, static_argnums=(0,))
    def linear_E_results_X_c_Xhat(self, X, S, c, Xhat):
        linear_results = {}
        for key in self.linear_E.keys():
            linear_results[key] = self.linear_E[key](X, S, c, Xhat)
        return linear_results
    
    @partial(jax.jit, static_argnums=(0,))
    def linear_B_results_X_c_Xhat(self, X, S, c, Xhat):
        linear_results = {}
        for key in self.linear_B.keys():
            linear_results[key] = self.linear_B[key](X, S, c, Xhat)
        return linear_results




    @partial(jax.jit, static_argnums=(0,))
    def E_gauss_X_c_Xhat(self, **linear_results):
        # return - linear_results['Lap'] + linear_results['Id'] ** 3
        pass
 
    @partial(jax.jit, static_argnums=(0,))
    def B_gauss_X_c_Xhat(self, **linear_results):
        # return linear_results['Id']
        pass


    @shapeParser
    @partial(jax.jit, static_argnums=(0, 1))
    def Grad_E_gauss_X_c(self, X_shape, X, S, c, xhat):
        grads = jax.grad(self.E_gauss_X_c, argnums=(0, 1, 2))(X, S, c, xhat)

        return {'grad_X': grads[0], 'grad_S': grads[1], 'grad_c': grads[2]}

    @partial(shapeParser, pad=True)
    @partial(jax.jit, static_argnums=(0, 1))
    def Grad_E_gauss_X_c_Xhat(self, X_shape, X, S, c, Xhat):
        # vmap over each entry in the dictionary
        return jax.tree_util.tree_map(lambda g: jax.vmap(lambda xh: g(X, S, c, xh))(Xhat), self.Grad_E_gauss_X_c)

    
    @shapeParser
    @partial(jax.jit, static_argnums=(0, 1))
    def Grad_B_gauss_X_c(self, X_shape, X, S, c, xhat):
        grads = jax.grad(self.B_gauss_X_c, argnums=(0, 1, 2))(X, S, c, xhat)
        return {'grad_X': grads[0], 'grad_S': grads[1], 'grad_c': grads[2]}
    
    @partial(shapeParser, pad=True)
    @partial(jax.jit, static_argnums=(0, 1))
    def Grad_B_gauss_X_c_Xhat(self, X_shape, X, S, c, Xhat):
        return jax.tree_util.tree_map(lambda g: jax.vmap(lambda xh: g(X, S, c, xh))(Xhat), self.Grad_B_gauss_X_c)
    

    @partial(jax.jit, static_argnums=(0,))
    def DE_gauss(self, x, s, xhat, *args):
        # return -jnp.trace(jax.hessian(self.gauss, argnums=2)(x, s, xhat)) + 3 * args[0] ** 2 * self.gauss(x, s, xhat)
        pass

    @partial(jax.jit, static_argnums=(0,))
    def DE_gauss_X(self, X, S, xhat, *args):
        return jax.vmap(self.DE_gauss, in_axes=(0, 0, None,) + (None,)*len(args))(X, S, xhat, *args)

    @partial(jax.jit, static_argnums=(0,))
    def DE_gauss_X_Xhat(self, X, S, Xhat, **linear_results):
        args = []
        for key in self.DE:
            args.append(linear_results[key])
        return jax.vmap(self.DE_gauss_X, in_axes=(None, None, 0,) + (0,)*len(args))(X, S, Xhat, *args)
    
    @partial(jax.jit, static_argnums=(0,))
    def DB_gauss(self, x, s, xhat, *args):
        # return self.gauss(x, s, xhat)
        pass
    
    @partial(jax.jit, static_argnums=(0,))
    def DB_gauss_X(self, X, S, xhat, *args):
        return jax.vmap(self.DB_gauss, in_axes=(0, 0, None) + (None,)*len(args))(X, S, xhat, *args)

    @partial(jax.jit, static_argnums=(0,))
    def DB_gauss_X_Xhat(self, X, S, Xhat, **linear_results):
        """
        Compute the linearized PDE operator of the Gaussian kernel at u.
        """
        args = []
        for key in self.DB:
            args.append(linear_results[key])
            
        return jax.vmap(self.DB_gauss_X, in_axes=(None, None, 0)+(0,)*len(args))(X, S, Xhat, *args)
    


if __name__ == '__main__':
    x = jnp.array([1.0, 2.0])
    s = jnp.array([1.0])
    xhat = jnp.array([1.0, 2.0])
    
    X = jnp.array([[1.0, 2.0], [2.0, 3.0]])
    S = jnp.array([1.0, 2.0])
    Xhat = jnp.array([[1.0, 2.0], [4.0, 5.0], [6.0, 7.0]])
    c = jnp.array([1.0, 2.0])

    kernel = GaussianKernel(d=2, power=4.01, sigma_max=1.0, sigma_min=1e-3)
    # print(kernel.gauss(x, s, xhat))
    # print(kernel.gauss_X(X, S, xhat))
    # print(kernel.gauss_X_Xhat(X, S, Xhat))
    # print(kernel.gauss_X_c(X, S, c, xhat))
    # print(kernel.gauss_X_c_Xhat(X, S, c, Xhat))

    # print(kernel.Lap_gauss_X_c(X, S, c, xhat))
    # print(kernel.Lap_gauss_X_c_Xhat(X, S, c, Xhat))
    
    linear_results = kernel.linear_results_X_c_Xhat(X, S, c, Xhat)

    print(linear_results)

    print(kernel.E_gauss_X_c(X, S, c, xhat))
    print(kernel.B_gauss_X_c(X, S, c, xhat))
    print(kernel.E_gauss_X_c_Xhat(**linear_results))
    print(kernel.B_gauss_X_c_Xhat(**linear_results))

    print(kernel.Grad_E_gauss_X_c(X, S, c, xhat))
    print(kernel.Grad_E_gauss_X_c_Xhat(X, S, c, Xhat))
    print(kernel.Grad_B_gauss_X_c(X, S, c, xhat))
    print(kernel.Grad_B_gauss_X_c_Xhat(X, S, c, Xhat))


    print(kernel.DE_gauss_X_Xhat(X, S, Xhat, **linear_results))
    print(kernel.DB_gauss_X_Xhat(X, S, Xhat, **linear_results))