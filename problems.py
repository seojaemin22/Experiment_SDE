from solver import Solver
from config import Config
from utils import *
import jax
from jax import numpy as jnp
import wandb

class HJB_Solver(Solver):
    def __init__(self, config: Config):
        super().__init__(config)

    def sample_domain(self, key, batch_size):
        t_pde = jax.random.uniform(key.newkey(), (batch_size, 1), minval=0, maxval=1)
        x_pde = jnp.sqrt(2) * jax.random.normal(key.newkey(), (batch_size, self.config.d_in))
        return t_pde, x_pde
    
    def get_analytic_X(self, T, W):
        return self.get_X0(T.shape[0])[:, jnp.newaxis, :] + jnp.sqrt(2.0)*W
    
    def analytic_u(self, t, x):
        w = jnp.sqrt(1-t) * jax.random.normal(jax.random.key(10), (10000, self.config.d_in))
        return -jnp.log(jnp.mean(jnp.exp(-self.bc_fn(x + jnp.sqrt(2)*w)), axis=0))

    def bc_fn(self, x):
        return jnp.log(0.5 * (1 + jnp.sum(x**2, keepdims=True, axis=-1)))
    
    def sigma(self, t, x, y):
        return jnp.sqrt(2) * super().sigma(t, x, y)
    
    def h(self, t, x, y, z):
        return jnp.sum(z**2, axis=-1)
    
    def pinns_pde_loss(self, params, t, x):
        _, ux, uxx = self.calc_uxx(params, t, x)
        _, ut = self.calc_ut(params, t, x)
        loss = jnp.mean((ut[..., 0] + jnp.trace(uxx, axis1=-2, axis2=-1) - jnp.sum(ux**2, axis=-1))**2)
        return loss
    
    def get_base_config():
        return Config(case = "HJB")


class BSB_Solver(Solver):
    def __init__(self, config: Config):
        super().__init__(config)

    def sample_domain(self, key, batch_size):
        t_pde = jax.random.uniform(key.newkey(), (batch_size, 1), minval=0, maxval=1)
        x_pde = 0.75 + jax.random.normal(key.newkey(), (batch_size, self.config.d_in))
        return t_pde, x_pde
    
    def get_X0(self, batch_size):
        X0 = []
        for i in range(self.config.d_in):
            if i%2 == 1:
                X0.append(jnp.ones((batch_size, 1))/2)
            else:
                X0.append(jnp.ones((batch_size, 1)))
        return jnp.concatenate(X0, axis=-1)
    
    def get_analytic_X(self, T, W):
        return self.get_X0(T.shape[0])[:, jnp.newaxis, :] * jnp.exp(0.4*W - 0.08*T)

    def analytic_u(self, t, x):
        return jnp.exp((0.05 + 0.4**2)*(1-t)) * self.bc_fn(x)

    def bc_fn(self, x):
        return jnp.sum(x**2, keepdims=True, axis=-1)
    
    def sigma(self, t, x, y=None, z=None):
        return 0.4 * jax.vmap(jnp.diag, in_axes=0)(x)
    
    def h(self, t, x, y, z):
        return 0.05 * (y - jnp.matmul(z, x[...,jnp.newaxis])[..., 0])
    
    def c(self, t, x, u, ux, uxx):
        return 0.5 * 0.4**2 * (jnp.matmul(ux, x[...,jnp.newaxis])[...,0] + jnp.trace(jnp.matmul(jax.vmap(jnp.diag)(x**2), uxx[:, 0]), axis1=-1, axis2=-2)[..., jnp.newaxis])
    
    def b_heun(self, t, x, y=None, z=None):
        return -0.5 * 0.4**2 * x
    
    def pinns_pde_loss(self, params, t, x):
        u, ux, uxx = self.calc_uxx(params, t, x)
        _, ut = self.calc_ut(params, t, x)
        loss = jnp.mean((ut[..., 0] + 0.5*jnp.trace(self.sigma(t, x, u)**2 @ uxx[..., 0, :, :], axis1=-2, axis2=-1)[..., jnp.newaxis]
                         -0.05*(u - jnp.matmul(ux, x[..., jnp.newaxis])[...,0]))**2)
        return loss

    def get_base_config():
        return Config(case = "BSB")
    

class BZ_Solver(Solver):
    def __init__(self, config: Config):
        super().__init__(config)

    def sample_domain(self, key, batch_size):
        t_pde = jax.random.uniform(key.newkey(), (batch_size, 1), minval=0, maxval=1)
        x_pde = jnp.pi/2 + 2*jax.random.normal(key.newkey(), (batch_size, self.config.d_in))
        return t_pde, x_pde
    
    def get_X0(self, batch_size):
        X0 = jnp.full((batch_size, self.config.d_in), fill_value=jnp.pi/2)
        return X0
    
    def analytic_u(self, t, x):
        return jnp.exp(-0.1 * (1-t)) * 0.1 * jnp.sum(jnp.sin(x), axis=-1, keepdims=True)

    def get_analytic_X(self, T, W):
        X = jnp.zeros_like(W)
        X = X.at[:, 0, :].set(self.get_X0(X.shape[0]))
        tau = 1/self.config.traj_len

        def loop(i, x):
            u = self.analytic_u(T[:, i-1, :], x[:, i-1, :])
            x = x.at[:, i, :].set(x[:, i-1, :] + self.b(T[:, i-1, :], x[:, i-1, :], u)*tau + jnp.matmul(self.sigma(T[:, i-1, :], x[:, i-1, :], u)*(W[:, i, :, jnp.newaxis] - W[:, i-1, :, jnp.newaxis]))[..., 0])
            return x
        
        X = jax.lax.fori_loop(1, self.config.traj_len+1, loop, X)
        return X

    def bc_fn(self, x):
        return 0.1 * jnp.sum(jnp.sin(x), axis=-1, keepdims=True)
    
    def sigma(self, x, y):
        return 0.3 * jax.vmap(lambda a, b: a*b, in_axes=(0, 0))(y, super().sigma(x, y))
    
    def h(self, t, x, y, z):
        return -0.1*y + 0.5*jnp.exp(-0.3*(1-t)) * 0.3**2 * (0.1*jnp.sum(jnp.sin(x), axis=-1, keepdims=True))**3

    def b_heun(self, t, x, u, ux):
        return jnp.matmul(u[...,jnp.newaxis], -0.3**2 * 0.5 * ux)[..., 0, :]
    
    def c(self, t, x, u, ux, uxx):
        return jnp.matmul(ux, self.b_heun(t, x, u, ux)[...,jnp.newaxis])[..., 0] + super().c(t, x, u, ux, uxx)
    
    def pinns_pde_loss(self, params, t, x):
        u, ux, uxx = self.calc_uxx(params, t, x)
        _, ut = self.calc_ut(params, t, x)
        loss = jnp.mean((ut + 0.5 * 0.3**2 * u**2 * jnp.trace(uxx, axis1=-1, axis2=-2) + self.h(t, x, u, ux))**2)
        return loss
    
    def get_base_config():
        return Config(case = "BZ")