from solver import *
from config import *
from utils import *
import jax
from jax import numpy as jnp
import wandb
import copy

# --------------------------------------------------
# Problems of PDE
# --------------------------------------------------

class HJB_Solver(PDE_Solver):
    def __init__(self, config: Config):
        super().__init__(config)

    def get_base_config():
        return PDE_Config(case = "HJB")
    
    def sample_domain(self, key, batch_size):
        t_pde = jax.random.uniform(key.newkey(), (batch_size, 1), minval=0, maxval=1)
        x_pde = jnp.sqrt(2) * jax.random.normal(key.newkey(), (batch_size, self.config.d_in))
        return t_pde, x_pde
    
    def analytic_u(self, t, x):
        w = jnp.sqrt(1-t) * jax.random.normal(jax.random.key(10), (50000, self.config.d_in))
        return -jnp.log(jnp.mean(jnp.exp(-self.bc_fn(x + jnp.sqrt(2)*w)), axis=0))
    
    def get_analytic_X(self, T, W):
        return self.get_X0(T.shape[0])[:, jnp.newaxis, :] + jnp.sqrt(2.0)*W

    def bc_fn(self, x):
        return jnp.log(0.5 * (1 + jnp.sum(x**2, keepdims=True, axis=-1)))
    
    def b(self, t, x, y=None, z=None):
        return super().b(t, x, y, z)

    def sigma(self, t, x, y=None, z=None):
        return jnp.sqrt(2) * super().sigma(t, x, y, z)
    
    def h(self, t, x, y, z):
        return jnp.sum(z**2, axis=-1)
    
    def pinns_pde_loss(self, params, t, x):
        _, ux, uxx = self.calc_uxx(params, t, x)
        _, ut = self.calc_ut(params, t, x)
        loss = jnp.mean((ut[..., 0] + jnp.trace(uxx, axis1=-2, axis2=-1) - jnp.sum(ux**2, axis=-1))**2)
        return loss



class BSB_Solver(PDE_Solver):
    def __init__(self, config: Config):
        super().__init__(config)

    def get_base_config():
        return PDE_Config(case = "BSB")
    
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
    
    def b(self, t, x, y=None, z=None):
        return super().b(t, x, y, z)
    
    def sigma(self, t, x, y=None, z=None):
        return 0.4 * jax.vmap(jnp.diag, in_axes=0)(x)
    
    def h(self, t, x, y, z):
        return 0.05 * (y - jnp.matmul(z, x[...,jnp.newaxis])[..., 0])
    
    def b_heun(self, t, x, y=None, z=None):
        return -0.5 * 0.4**2 * x
    
    def c(self, t, x, u, ux, uxx):
        return 0.5 * 0.4**2 * (jnp.matmul(ux, x[...,jnp.newaxis])[...,0] + jnp.trace(jnp.matmul(jax.vmap(jnp.diag)(x**2), uxx[:, 0]), axis1=-1, axis2=-2)[..., jnp.newaxis])
    
    def pinns_pde_loss(self, params, t, x):
        u, ux, uxx = self.calc_uxx(params, t, x)
        _, ut = self.calc_ut(params, t, x)
        loss = jnp.mean((ut[..., 0] + 0.5*jnp.trace(self.sigma(t, x, u)**2 @ uxx[..., 0, :, :], axis1=-2, axis2=-1)[..., jnp.newaxis]
                         -0.05*(u - jnp.matmul(ux, x[..., jnp.newaxis])[...,0]))**2)
        return loss

    

class BZ_Solver(PDE_Solver):
    def __init__(self, config: Config):
        super().__init__(config)

    def get_base_config():
        return PDE_Config(case = "BZ")
    
    def sample_domain(self, key, batch_size):
        t_pde = jax.random.uniform(key.newkey(), (batch_size, 1), minval=0, maxval=1)
        x_pde = jnp.pi/2 + 2*jax.random.normal(key.newkey(), (batch_size, self.config.d_in))
        return t_pde, x_pde
    
    def get_X0(self, batch_size):
        X0 = jnp.full((batch_size, self.config.d_in), fill_value=jnp.pi/2)
        return X0
    
    def analytic_u(self, t, x):
        return jnp.exp(-0.1 * (1-t)) * 0.1 * jnp.sum(jnp.sin(x), axis=-1, keepdims=True)

    def bc_fn(self, x):
        return 0.1 * jnp.sum(jnp.sin(x), axis=-1, keepdims=True)
    
    def b(self, t, x, y=None, z=None):
        return super().b(t, x, y, z)
    
    sigma_y = True
    def sigma(self, t, x, y, z=None):
        return 0.3 * jax.vmap(lambda a, b: a*b, in_axes=(0, 0))(y, super().sigma(t, x, y, z))
    
    def h(self, t, x, y, z):
        return -0.1*y + 0.5*jnp.exp(-0.3*(1-t)) * 0.3**2 * (0.1*jnp.sum(jnp.sin(x), axis=-1, keepdims=True))**3

    b_heun_y = True
    b_heun_z = True
    def b_heun(self, t, x, y, z):
        return jnp.matmul(y[...,jnp.newaxis], -0.3**2 * 0.5 * z)[..., 0, :]
    
    def c(self, t, x, u, ux, uxx):
        return jnp.matmul(ux, self.b_heun(t, x, u, ux)[...,jnp.newaxis])[..., 0] + super().c(t, x, u, ux, uxx)
    
    def pinns_pde_loss(self, params, t, x):
        u, ux, uxx = self.calc_uxx(params, t, x)
        _, ut = self.calc_ut(params, t, x)
        loss = jnp.mean((ut + 0.5 * 0.3**2 * u**2 * jnp.trace(uxx, axis1=-1, axis2=-2) + self.h(t, x, u, ux))**2)
        return loss

    
# --------------------------------------------------
# Problems of PIDE
# --------------------------------------------------


class HD_PIDE_Solver(PIDE_Solver):

    def get_base_config():
        return PIDE_Config(case = "HD_PIDE")

    def sample_domain(self, key, batch_size):
        t_pde = jax.random.uniform(key.newkey(), (batch_size, 1), minval=0, maxval=1)
        x_pde = 1 + jax.random.normal(key.newkey(), (batch_size, self.config.d_in))
        return t_pde, x_pde
    
    def get_X0(self, batch_size):
        X0 = jnp.ones((batch_size, self.config.d_in))
        return X0
    
    def analytic_u(self, t, x):
        return jnp.sum(x**2, axis=-1, keepdims=True) / self.config.d_in

    # def analytic_ut(self, t, x):
    #     return jnp.zeros_like(t)

    # def analytic_ux(self, t, x, output_pos=(0,)):
    #     return 2*x / self.config.d_in
    
    # def analytic_uxx(self, t, x, output_pos=(0,)):
    #     return jnp.zeros((x.shape[0], x.shape[1], x.shape[1]))
    
    def bc_fn(self, x):
        return jnp.sum(x**2, axis=-1, keepdims=True) / self.config.d_in
    
    def b(self, t, x, y=None, z=None):
        return 1/2 * self.config.epsilon * x

    def sigma(self, t, x, y=None, z=None):
        return self.config.tau * super().sigma(t, x, y, z)
    
    def h(self, t, x, y, z):
        return self.config.lambda_ * (self.config.mu_phi**2 + self.config.sigma_phi**2) + self.config.tau**2 + self.config.epsilon/self.config.d_in * jnp.sum(x**2, axis=-1, keepdims=True)
    
    def forward_diffusion(self, t, x, j):  # j.shape : (num_traj, self.config.mu_n, self.config.d_in)
        return jnp.sum(j, axis=-2)
    
    def forward_drift(self, t, x):
        return - self.config.lambda_ * self.config.mu_phi
    
    def backward_diffusion(self, params, t, x, y, z, j):
        def loop(i, acc):
            return acc + self.calc_u(params, t, x+j[..., i, :])
        return jax.lax.fori_loop(0, self.config.mu_n, loop, jnp.zeros_like(y)) - self.config.mu_n*y

    def backward_drift(self, t, x, y, z):
        return - self.config.lambda_ * self.config.mu_phi * jnp.sum(z, axis=-1, keepdims=True)