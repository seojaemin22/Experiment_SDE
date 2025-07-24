import jax
import flax.serialization as fs
import jax.experimental
import optax
from jax import numpy as jnp
import numpy as np
from flax import linen as nn
import tqdm
import wandb
import pandas
import matplotlib.pyplot as plt
from model import PINNs
from config import *
from utils import *
from functools import partial
import copy
from pathlib import Path
from jax.experimental import host_callback as hcb
import types

class Solver():

    def __init__(self, config: Config):
        self.config = copy.deepcopy(config)

        loss_method = self.config.loss_method
        if loss_method == 'pinns':
            self.loss_fn = self.pinns_grad_batch if self.config.micro_batch < self.config.batch else self.pinns_grad
        elif loss_method == 'fspinns':
            self.loss_fn = self.fspinns_grad_batch if self.config.micro_batch < self.config.batch else self.fspinns_grad
        elif loss_method == 'bsde':
            self.loss_fn = self.bsde_grad_batch if self.config.micro_batch < self.config.batch else self.bsde_grad
        elif loss_method == 'bsdeskip':
            self.loss_fn = self.bsde_skip_grad_batch if self.config.micro_batch < self.config.batch else self.bsde_skip_grad
        elif loss_method == 'bsdeheun':
            self.loss_fn = self.bsde_heun_grad_batch if self.config.micro_batch < self.config.batch else self.bsde_heun_grad
        elif loss_method == 'regress':
            self.loss_fn = self.reg_loss
        elif loss_method == 'bsdeheunnew':
            self.loss_fn = self.bsde_heun_new_grad_batch if self.config.micro_batch < self.config.batch else self.bsde_heun_new_grad
        else:
            raise Exception("Loss Method '" + loss_method + "' Not Implemented")
        
        self.model = self.create_model()
        self.optimizer = self.create_opt()
        if self.config.analytic_sol:
            self.sol_T, self.sol_X, self.sol_U = self.get_analytic_sol()
        if self.config.custom_eval:
            self.eval_point = self.get_eval_point()
        if self.config.save_to_wandb:
            self.init_wandb()
        
    def init_solver(self, key: Key):
        params = self.init_model(key)
        opt_state = self.init_opt(params)
        num_params = sum(x.size for x in jax.tree_util.tree_leaves(params))
        if self.config.save_to_wandb:
            wandb.config['# Params'] =  num_params
        return params, opt_state
    
    # --------------------------------------------------
    # Init Methods
    # --------------------------------------------------

    def create_model(self):
        return PINNs(self.config)
    
    def create_opt(self):
        if self.config.schedule == 'piecewise_constant':
            schedule = optax.piecewise_constant_schedule(
                init_value=self.config.lr,
                boundaries_and_scales=self.config.boundaries_and_scales
            )
        elif self.config.schedule == 'cosine_decay':
            schedule = optax.cosine_decay_schedule(
                init_value=self.config.lr,
                decay_steps=self.config.iter
            )
        elif self.config.schedule == 'cosine_onecycle':
            schedule = optax.cosine_onecycle_schedule(
                transition_steps=self.config.iter,
                peak_value=self.config.lr
            )
        else: # No scheduler
            schedule = optax.constant_schedule(
                value=self.config.lr
            )
            
        if self.config.optim == 'adam':
            return optax.chain(
                optax.scale_by_adam(),
                optax.scale_by_schedule(schedule),
                optax.scale(-1.0)
            )
        else: # SGD
            return optax.chain(
                optax.scale_by_schedule(schedule),
                optax.scale(-1.0)
            )
    
    def init_model(self, key: Key):
        t_pde, x_pde = self.sample_domain(key, self.config.batch)
        return self.model.init(key.newkey(), t_pde, x_pde)
            
    def init_opt(self,params):
        return self.optimizer.init(params)
    

    def init_wandb(self):
        print("Initializing wandb")
        wandb.init(project=self.config.project_name,
                   name=self.config.run_name,
                   config=vars(self.config))
    
    def close(self):
        wandb.finish()


    def get_base_config():
        return Config()
    
    def tab_model(self, key:Key):
        t_pde, x_pde = self.sample_domain(key,self.config.batch_pde)
        tab_fn = nn.tabulate(self.model,key.newkey())
        print(tab_fn(t_pde, x_pde))
    
    # --------------------------------------------------
    # Equation Methods
    # --------------------------------------------------

    def calc_u(self, params, t, x):
        return self.model.apply(params, t, x)

    def calc_ut(self,params, t, x):
        def t_func(t, x):
            model_fn = lambda t: self.model.apply(params, t, x)
            u, du_dt = jax.vjp(model_fn, t)
            u_t = jax.vmap(du_dt, in_axes=0)(jnp.eye(len(u)))[0]
            return u, u_t
        return jax.vmap(t_func,in_axes=(0,0))(t, x)
    
    def calc_ux(self, params, t, x, output_pos=(0,)):
        def jacrev(t, x):
            def func(x):
                u = self.model.apply(params, t, x)
                u = u[..., output_pos]
                return u
            u, vjp_fun = jax.vjp(func, x)
            ret = jax.vmap(vjp_fun, in_axes=0)(jnp.eye(len(u)))
            return u, ret[0]
        return jax.vmap(jacrev, in_axes=0)(t, x)
    
    def calc_uxx(self,params, t, x, output_pos=(0,)):
        def jacrev2(t, x):
            def func(x):
                u = self.model.apply(params, t, x)
                u = u[..., output_pos]
                return u
            def jacrev(x):
                u, vjp_fun = jax.vjp(func, x)
                ret = jax.vmap(vjp_fun,in_axes=0)(jnp.eye(len(u)))
                return ret[0], u
            func2 = lambda s: jax.jvp(jacrev, (x,), (s,), has_aux=True)
            u_x, u_xx, u = jax.vmap(func2, in_axes=1, out_axes=(None,1,None))(jnp.eye(len(x)))
            return u, u_x, u_xx
        return jax.vmap(jacrev2,in_axes=(0,0))(t, x)


    def bc_fn(self, x):
        pass

    def calc_bcx(self, x): 
        jax_x = jax.jacrev(lambda x: self.bc_fn(x), argnums=0)
        return jax.vmap(jax_x, in_axes=0)(x)
    

    def analytic_u(self, t, x):
        pass
    
    # t : (batch, 1)
    # x : (batch, d_in)
    # u (= y) : (batch, d_out)  [ ux : (batch, d_out, d_in), uxx : (batch, d_out, d_in, d_in) ]
    # z : (batch, d_out, d_in)

    b_y = False
    b_z = False
    def b(self, t, x, y=None, z=None):  # -> (batch, d_in)
        return jnp.zeros_like(x)
    
    b_heun_y = False
    b_heun_z = False
    def b_heun(self, t, x, y=None, z=None):  # b - 1/2 * sigma * sigma_x
        return jnp.zeros_like(x)

    sigma_y = False 
    sigma_z = False
    def sigma(self, t, x, y=None, z=None):  # -> (batch, d_in, d_in)
        return jnp.repeat(jnp.expand_dims(jnp.eye(x.shape[-1]), axis=0), x.shape[0], axis=0)
    
    def h(self, t, x, y, z):  # -> (batch, d_out)
        return jnp.zeros_like(y)
    
    def c(self, t, x, u, ux, uxx):
        return 0.5 * jnp.trace(jnp.matmul(jnp.matmul(self.sigma(t, x, u), self.sigma(t, x, u)), uxx[:,0]), axis1=-1, axis2=-2)[..., jnp.newaxis]

    # --------------------------------------------------
    # Util Methods
    # --------------------------------------------------

    def sample_domain(self, key: Key, batch_size):
        t_pde = jax.random.uniform(key.newkey(), (batch_size, 1), minval=0, maxval=1)
        x_pde = 2 * jax.random.normal(key.newkey(), (batch_size, self.config.d_in))
        return t_pde, x_pde
    
    def get_X0(self, batch_size):
        X0 = jnp.zeros((batch_size, self.config.d_in))
        return X0
    
    def get_analytic_X(self, T, W):  # T : (batch, traj_len+1, 1), W : (batch, traj_len+1, )
        pass
    
    def get_analytic_sol(self):
        num_traj = 5
        traj_len = 50
        tau = 1/traj_len
        
        T = jnp.repeat(jnp.linspace(0, 1, traj_len+1)[jnp.newaxis, ..., jnp.newaxis], num_traj, axis=0)
        dW = jnp.sqrt(tau) * jnp.concatenate((jnp.zeros((num_traj, 1, self.config.d_in)),
                                              jax.random.normal(jax.random.key(1), (num_traj, traj_len, self.config.d_in))), axis=1)
        W = jnp.cumsum(dW, axis=1)
        
        X = self.get_analytic_X(T, W)
        U = jax.vmap(jax.vmap(self.analytic_u, in_axes=(0, 0)), in_axes=(0, 0))(T, X)
        return T, X, U


    def plot_pred(self, params, i):
        time = self.sol_T[..., 0].T
        pred = self.calc_u(params, self.sol_T, self.sol_X)[..., 0].T
        true = self.sol_U[..., 0].T
        L1 = jnp.abs(pred - true)

        fig_pred = plt.figure(figsize=(4, 3))
        plt.plot(time, pred, "r", linewidth=1)
        plt.plot(time, true, ":b", linewidth=1)
        plt.title('Prediction') 
        plt.close(fig_pred)
        wandb.log({'Prediction': wandb.Image(fig_pred)}, step=i)

        fig_L1 = plt.figure(figsize=(4, 3))
        plt.plot(time, L1, "k", linewidth=1)
        plt.title('L1 Error')
        plt.yscale('log')
        plt.close(fig_L1)
        wandb.log({'L1 Error': wandb.Image(fig_L1)}, step=i)

    def get_eval_point(self):
        pass

    def plot_eval(self, params):
        pass

    def calc_RL(self, params):
        pred = self.calc_u(params, self.sol_T, self.sol_X)
        true = self.sol_U

        RL2 = jnp.mean(jnp.sqrt(jnp.sum((pred - true)**2, axis=1) / jnp.sum(true**2, axis=1)))
        RL_T0 = jnp.mean(jnp.abs(pred[:, 0] - true[:, 0]) / jnp.abs(true[:, 0]))
        return RL2, RL_T0
    
    def calc_eval(self, params):
        pass

    @partial(jax.jit, static_argnums=0)
    def jit_calc_RL(self,params):
        return self.calc_RL(params)

    @partial(jax.jit, static_argnums=0)
    def jit_calc_eval(self,params):
        return self.calc_eval(params)

    # --------------------------------------------------
    # Loss Methods
    # --------------------------------------------------
    
    def pinns_pde_loss(self, params, t, x):
        pass

    def pinns_pde(self, key, params):
        num_traj = self.config.batch
        traj_len = self.config.traj_len
        
        t_pde, x_pde = self.sample_domain(key, num_traj * traj_len)
        return self.pinns_pde_loss(params, t_pde, x_pde)
    
    def pinns_bc(self, key, params):
        t_bc, x_bc = self.sample_domain(key, self.config.batch)
        t_bc = jnp.ones_like(t_bc)

        u, ux = self.calc_ux(params, t_bc, x_bc)
        bc_loss = jnp.mean((u - self.bc_fn(x_bc))**2) + jnp.mean((ux - self.calc_bcx(x_bc))**2)
        return bc_loss
    
    def pinns_loss(self, key, params):
        pde_loss = self.pinns_pde(key, params) * self.config.pde_scale
        bc_loss = self.pinns_bc(key, params) * self.config.bc_scale
        return pde_loss + bc_loss, (pde_loss, bc_loss)
    
    def pinns_grad(self, key, params):
        (total, (pde_loss, bc_loss)), grad = jax.value_and_grad(self.pinns_loss, argnums=1, has_aux=True)(key, params)
        return (pde_loss, bc_loss), grad
    
    @partial(jax.jit, static_argnums=0)
    def jit_pinns_loss(self, key, params):
        return self.pinns_loss(key, params)[0]

    
    def pinns_pde_grad_batch(self, key, params):
        num_traj = self.config.batch
        micro_batch = self.config.micro_batch
        n_chunks = (num_traj + micro_batch - 1) // micro_batch
        traj_len = self.config.traj_len

        grad_make_loss = jax.jit(jax.value_and_grad(lambda pp, tt, xx: jnp.sum(self.pinns_pde_loss(pp, tt, xx))/n_chunks * self.config.pde_scale, argnums=0))

        def chunk_loop(carry, _):
            key, loss_acc, grad_acc = carry
            t_pde, x_pde = self.sample_domain(key, micro_batch*traj_len)
            loss, grad = grad_make_loss(params, t_pde, x_pde)
            loss_acc = loss_acc + loss
            grad_acc = jax.tree_util.tree_map(lambda a, b: a+b, grad_acc, grad)
            return (key, loss_acc, grad_acc), None
        
        (key, pde_loss, pde_grad), _ = jax.lax.scan(chunk_loop, (key, 0.0, jax.tree_util.tree_map(jnp.zeros_like, params)), None, length=n_chunks)
        return pde_loss, pde_grad
    
    def pinns_grad_batch(self, key, params):
        pde_loss, pde_grad = self.pinns_pde_grad_batch(key, params) 
        bc_loss, bc_grad = jax.value_and_grad(self.pinns_bc, argnums=1)(key, params) 
        grad = jax.tree_util.tree_map(lambda a, b: a+b, pde_grad, bc_grad)
        return (pde_loss, bc_loss), grad
    
    # ------------------------------

    def fspinns_loss(self, key, params):
        num_traj = self.config.batch
        traj_len = self.config.traj_len
        tau = 1/traj_len

        T = jnp.repeat(jnp.linspace(0, 1, traj_len+1)[jnp.newaxis, ..., jnp.newaxis], num_traj, axis=0)
        dW = jnp.sqrt(tau) * jnp.concatenate((jnp.zeros((num_traj, 1, self.config.d_in)),
                                              jax.random.normal(key.newkey(), (num_traj, traj_len, self.config.d_in))), axis=1)
        X = jnp.zeros((num_traj, traj_len+1, self.config.d_in))
        X = X.at[:, 0, :].set(self.get_X0(num_traj))

        def loop(i, x):
            yz_dict = {}
            if self.b_z or self.sigma_z:
                yz_dict['y'], yz_dict['z'] = self.calc_ux(params, T[:, i-1, :], x[:, i-1, :])
            elif self.b_y or self.sigma_y:
                yz_dict['y'] = self.calc_u(params, T[:, i-1, :], x[:, i-1, :])
            x = x.at[:, i, :].set(x[:, i-1, :] + self.b(T[:, i-1, :], x[:, i-1, :], **yz_dict)*tau + jnp.matmul(self.sigma(T[:, i-1, :], x[:, i-1, :], **yz_dict), dW[:, i, :, jnp.newaxis])[..., 0])
            return x
        
        X = jax.lax.fori_loop(1, self.config.traj_len+1, loop, X)
        pde_loss = self.pinns_pde_loss(params, T.reshape(-1, 1), X.reshape(-1, self.config.d_in))
        bc, bcx = self.calc_ux(params, T[:, -1, :], X[:, -1, :])
        bc_loss = (jnp.mean((bc - self.bc_fn(X[:, -1, :]))**2) + jnp.mean((bcx - self.calc_bcx(X[:, -1, :]))**2)) * self.config.bc_scale
        return pde_loss + bc_loss, (pde_loss, bc_loss)
    
    def fspinns_grad(self, key, params):
        (total, (pde_loss, bc_loss)), grad = jax.value_and_grad(self.fspinns_loss, argnums=1, has_aux=True)(key, params)
        return (pde_loss, bc_loss), grad
    
    @partial(jax.jit, static_argnums=0)
    def jit_fspinns_loss(self, key, params):
        return self.fspinns_loss(key, params)[0]
    

    def fspinns_grad_batch(self, key, params):
        num_traj = self.config.batch
        micro_batch = self.config.micro_batch
        n_chunks = (num_traj + micro_batch - 1) // micro_batch
        traj_len = self.config.traj_len
        tau = 1/traj_len

        def make_loss(key, params):
            T = jnp.repeat(jnp.linspace(0, 1, traj_len+1)[jnp.newaxis, ..., jnp.newaxis], micro_batch, axis=0)
            dW = jnp.sqrt(tau) * jnp.concatenate((jnp.zeros((micro_batch, 1, self.config.d_in)),
                                                  jax.random.normal(key.newkey(), (micro_batch, traj_len, self.config.d_in))), axis=1)
            X = jnp.zeros((micro_batch, traj_len+1, self.config.d_in))
            X = X.at[:, 0, :].set(self.get_X0(micro_batch))

            def loop(i, x):
                yz_dict = {}
                if self.b_z or self.sigma_z:
                    yz_dict['y'], yz_dict['z'] = self.calc_ux(params, T[:, i-1, :], x[:, i-1, :])
                elif self.b_y or self.sigma_y:
                    yz_dict['y'] = self.calc_u(params, T[:, i-1, :], x[:, i-1, :])
                x = x.at[:, i, :].set(x[:, i-1, :] + self.b(T[:, i-1, :], x[:, i-1, :], **yz_dict)*tau + jnp.matmul(self.sigma(T[:, i-1, :], x[:, i-1, :], **yz_dict), dW[:, i, :, jnp.newaxis])[..., 0])
                return x
            
            X = jax.lax.fori_loop(1, traj_len+1, loop, X)
            pde_loss = self.pinns_pde_loss(params, T.reshape(-1, 1), X.reshape(-1, self.config.d_in)) * self.config.pde_scale
            bc, bcx = self.calc_ux(params, T[:, -1, :], X[:, -1, :])
            bc_loss = (jnp.mean((bc - self.bc_fn(X[:, -1, :]))**2) + jnp.mean((bcx - self.calc_bcx(X[:, -1, :]))**2)) * self.config.bc_scale
            return pde_loss + bc_loss, (pde_loss, bc_loss)
        
        grad_make_loss = jax.jit(jax.value_and_grad(make_loss, argnums=1, has_aux=True))

        def chunk_loop(carry, _):
            key, pde_loss_acc, bc_loss_acc, grad_acc = carry
            (total, (pde_loss, bc_loss)), grad = grad_make_loss(key, params)
            pde_loss_acc = pde_loss_acc + pde_loss
            bc_loss_acc = bc_loss_acc + bc_loss
            grad_acc = jax.tree_util.tree_map(lambda a, b: a+b, grad_acc, grad)
            return (key, pde_loss_acc, bc_loss_acc, grad_acc), None
        
        (key, pde_loss, bc_loss, grad), _ = jax.lax.scan(chunk_loop, (key, 0.0, 0.0, jax.tree_util.tree_map(jnp.zeros_like, params)), None, length=n_chunks)
        return (pde_loss, bc_loss), grad
    
    # ------------------------------

    def bsde_loss(self, key, params):
        num_traj = self.config.batch
        traj_len = self.config.traj_len
        tau = 1/traj_len
        
        t = jnp.zeros((num_traj, 1))
        x = self.get_X0(num_traj)
        u, ux = self.calc_ux(params, t, x)
        step_loss = jnp.zeros(traj_len)

        def traj_calc(i, inputs):
            key, t, x, u, ux, step_loss = inputs

            sigma = self.sigma(t, x, u, ux)
            dw = jnp.sqrt(tau) * jax.random.normal(key.newkey(), (num_traj, self.config.d_in))
            t_new = t + tau
            x_new = x + self.b(t, x, u, ux)*tau + jnp.matmul(sigma, dw[..., jnp.newaxis])[..., 0]
            u_new = u + self.h(t, x, u, ux)*tau + jnp.matmul(jnp.matmul(ux, sigma), dw[..., jnp.newaxis])[..., 0]

            u_calc, ux_calc = self.calc_ux(params, t_new, x_new)
            step_loss = step_loss.at[i].set(jnp.sum((u_new - u_calc)**2))

            u_next = u_calc if self.config.reset_u else u_new
            return key, t_new, x_new, u_next, ux_calc, step_loss

        key, t, x, u, ux, step_loss = jax.lax.fori_loop(0, traj_len, traj_calc, (key, t, x, u, ux, step_loss))
        pde_loss = jnp.sum(step_loss) * self.config.pde_scale 
        bc_loss = jnp.sum((u - self.bc_fn(x))**2) + jnp.sum((ux - self.calc_bcx(x))**2) * self.config.bc_scale
        return pde_loss + bc_loss, (pde_loss, bc_loss)
        
    def bsde_grad(self, key, params):
        (total, (pde_loss, bc_loss)), grad = jax.value_and_grad(self.bsde_loss, argnums=1, has_aux=True)(key, params)
        return (pde_loss, bc_loss), grad
    
    @partial(jax.jit, static_argnums=0)
    def jit_bsde_loss(self, key, params):
        return self.bsde_loss(key, params)[0]
    

    def bsde_grad_batch(self, key, params):
        num_traj = self.config.batch
        micro_batch = self.config.micro_batch
        n_chunks = (num_traj + micro_batch - 1) // micro_batch
        traj_len = self.config.traj_len
        tau = 1/traj_len
        
        def make_loss(key, params):
            t = jnp.zeros((micro_batch, 1))
            x = self.get_X0(micro_batch)
            u, ux = self.calc_ux(params, t, x)
            step_loss = jnp.zeros(traj_len)

            def traj_calc(i, inputs):
                key, t, x, u, ux, step_loss = inputs

                sigma = self.sigma(t, x, u, ux)
                dw = jnp.sqrt(tau) * jax.random.normal(key.newkey(), (micro_batch, self.config.d_in))
                t_new = t + tau
                x_new = x + self.b(t, x, u, ux)*tau + jnp.matmul(sigma, dw[..., jnp.newaxis])[..., 0]
                u_new = u + self.h(t, x, u, ux)*tau + jnp.matmul(jnp.matmul(ux, sigma), dw[..., jnp.newaxis])[..., 0]

                u_calc, ux_calc = self.calc_ux(params, t_new, x_new)
                step_loss = step_loss.at[i].set(jnp.sum((u_new - u_calc)**2))

                u_next = u_calc if self.config.reset_u else u_new
                return key, t_new, x_new, u_next, ux_calc, step_loss

            key, t, x, u, ux, step_loss = jax.lax.fori_loop(0, traj_len, traj_calc, (key, t, x, u, ux, step_loss))
            pde_loss = jnp.sum(step_loss) * self.config.pde_scale 
            bc_loss = jnp.sum((u - self.bc_fn(x))**2) + jnp.sum((ux - self.calc_bcx(x))**2) * self.config.bc_scale
            return pde_loss + bc_loss, (pde_loss, bc_loss)
        
        grad_make_loss = jax.jit(jax.value_and_grad(make_loss, argnums=1, has_aux=True))

        def chunk_loop(carry, _):
            key, pde_loss_acc, bc_loss_acc, grad_acc = carry
            (total, (pde_loss, bc_loss)), grad = grad_make_loss(key, params)
            pde_loss_acc = pde_loss_acc + pde_loss
            bc_loss_acc = bc_loss_acc + bc_loss
            grad_acc = jax.tree_util.tree_map(lambda a, b: a+b, grad_acc, grad)
            return (key, pde_loss_acc, bc_loss_acc, grad_acc), None
        
        (key, pde_loss, bc_loss, grad), _ = jax.lax.scan(chunk_loop, (key, 0.0, 0.0, jax.tree_util.tree_map(jnp.zeros_like, params)), None, length=n_chunks)
        return (pde_loss, bc_loss), grad
    
    # ------------------------------

    def bsde_skip_loss(self, key, params):
        num_traj = self.config.batch
        traj_len = self.config.traj_len
        tau = 1/traj_len
        skip_len = self.config.skip_len
        
        t = jnp.zeros((num_traj, 1))
        x = self.get_X0(num_traj)
        u, ux = self.calc_ux(params, t, x)
        step_loss = jnp.zeros(traj_len)

        def traj_calc(i, inputs):
            key, t, x, u, ux, step_loss = inputs

            sigma = self.sigma(t, x, u, ux)
            dw = jnp.sqrt(tau) * jax.random.normal(key.newkey(), (num_traj, self.config.d_in))
            t_new = t + tau
            x_new = x + self.b(t, x, u, ux)*tau + jnp.matmul(sigma, dw[..., jnp.newaxis])[..., 0]
            u_new = u + self.h(t, x, u, ux)*tau + jnp.matmul(jnp.matmul(ux, sigma), dw[..., jnp.newaxis])[..., 0]

            u_calc, ux_calc = self.calc_ux(params, t_new, x_new)
            step_loss = step_loss.at[i].set(jax.lax.select(i%skip_len==0, jnp.sum((u_new - u_calc)**2), 0.))

            u_next = jax.lax.select(i%skip_len==0, u_calc, u_new) if self.config.reset_u else u_new
            return key, t_new, x_new, u_next, ux_calc, step_loss

        key, t, x, u, ux, step_loss = jax.lax.fori_loop(0, traj_len, traj_calc, (key, t, x, u, ux, step_loss))
        pde_loss = jnp.sum(step_loss) * self.config.pde_scale 
        bc_loss = jnp.sum((u - self.bc_fn(x))**2) + jnp.sum((ux - self.calc_bcx(x))**2) * self.config.bc_scale
        return pde_loss + bc_loss, (pde_loss, bc_loss)
    
    def bsde_skip_grad(self, key, params):
        (total, (pde_loss, bc_loss)), grad = jax.value_and_grad(self.bsde_loss, argnums=1, has_aux=True)(key, params)
        return (pde_loss, bc_loss), grad
    
    @partial(jax.jit, static_argnums=0)
    def jit_bsde_skip_loss(self, key, params):
        return self.bsde_skip_loss(key, params)[0]
    

    def bsde_skip_grad_batch(self, key, params):
        num_traj = self.config.batch
        micro_batch = self.config.micro_batch
        n_chunks = (num_traj + micro_batch - 1) // micro_batch
        traj_len = self.config.traj_len
        tau = 1/traj_len
        skip_len = self.config.skip_len
        
        def make_loss(key, params):
            t = jnp.zeros((micro_batch, 1))
            x = self.get_X0(micro_batch)
            u, ux = self.calc_ux(params, t, x)
            step_loss = jnp.zeros(traj_len)

            def traj_calc(i, inputs):
                key, t, x, u, ux, step_loss = inputs

                sigma = self.sigma(t, x, u, ux)
                dw = jnp.sqrt(tau) * jax.random.normal(key.newkey(), (micro_batch, self.config.d_in))
                t_new = t + tau
                x_new = x + self.b(t, x, u, ux)*tau + jnp.matmul(sigma, dw[..., jnp.newaxis])[..., 0]
                u_new = u + self.h(t, x, u, ux)*tau + jnp.matmul(jnp.matmul(ux, sigma), dw[..., jnp.newaxis])[..., 0]

                u_calc, ux_calc = self.calc_ux(params, t_new, x_new)
                step_loss = step_loss.at[i].set(jax.lax.select(i%skip_len==0, jnp.sum((u_new - u_calc)**2), 0.))

                u_next = jax.lax.select(i%skip_len==0, u_calc, u_new) if self.config.reset_u else u_new
                return key, t_new, x_new, u_next, ux_calc, step_loss

            key, t, x, u, ux, step_loss = jax.lax.fori_loop(0, traj_len, traj_calc, (key, t, x, u, ux, step_loss))
            pde_loss = jnp.sum(step_loss) * self.config.pde_scale 
            bc_loss = jnp.sum((u - self.bc_fn(x))**2) + jnp.sum((ux - self.calc_bcx(x))**2) * self.config.bc_scale
            return pde_loss + bc_loss, (pde_loss, bc_loss)
        
        grad_make_loss = jax.jit(jax.value_and_grad(make_loss, argnums=1, has_aux=True))

        def chunk_loop(carry, _):
            key, pde_loss_acc, bc_loss_acc, grad_acc = carry
            (total, (pde_loss, bc_loss)), grad = grad_make_loss(key, params)
            pde_loss_acc = pde_loss_acc + pde_loss
            bc_loss_acc = bc_loss_acc + bc_loss
            grad_acc = jax.tree_util.tree_map(lambda a, b: a+b, grad_acc, grad)
            return (key, pde_loss_acc, bc_loss_acc, grad_acc), None
        
        (key, pde_loss, bc_loss, grad), _ = jax.lax.scan(chunk_loop, (key, 0.0, 0.0, jax.tree_util.tree_map(jnp.zeros_like, params)), None, length=n_chunks)
        return (pde_loss, bc_loss), grad
    
    # ------------------------------
    
    def bsde_heun_loss(self, key, params):
        num_traj = self.config.batch
        traj_len = self.config.traj_len
        tau = 1/traj_len
        
        calc_uxx = jax.checkpoint(self.calc_uxx) if self.config.checkpointing else self.calc_uxx

        t = jnp.zeros((num_traj, 1))
        x = self.get_X0(num_traj)
        u, ux, uxx = calc_uxx(params, t, x)
        step_loss = jnp.zeros(traj_len)
        
        def traj_calc(i, inputs):
            key, t, x, u, ux, uxx, step_loss = inputs

            sigma = self.sigma(t, x, u, ux)
            dw = jnp.sqrt(tau) * jax.random.normal(key.newkey(), (num_traj, self.config.d_in))
            dx_int = self.b_heun(t, x, u, ux)*tau + jnp.matmul(sigma, dw[..., jnp.newaxis])[..., 0]
            x_int = x + dx_int
            du_int = (self.h(t, x, u, ux) - self.c(t, x, u, ux, uxx))*tau + jnp.matmul(jnp.matmul(ux, self.sigma(t, x, u, ux)), dw[..., jnp.newaxis])[..., 0]

            t_new = t + tau
            u_int, ux_int, uxx_int = calc_uxx(params, t_new, x_int)
            
            x_new = x + 0.5*dx_int + 0.5*(self.b_heun(t_new, x_int, u_int, ux_int)*tau +
                                          jnp.matmul(self.sigma(t_new, x_int, u_int), dw[..., jnp.newaxis])[..., 0])
            u_new = u + 0.5*du_int + 0.5*((self.h(t_new, x_int, u_int, ux_int) - self.c(t_new, x_int, u_int, ux_int, uxx_int))*tau +
                                          jnp.matmul(jnp.matmul(ux_int, self.sigma(t_new, x_int, u_int)), dw[..., jnp.newaxis])[..., 0])
            
            u_calc, ux_calc, uxx_calc = calc_uxx(params, t_new, x_new)

            step_loss = step_loss.at[i].set(jnp.sum((u_new - u_calc)**2))
            return key, t_new, x_new, u_calc, ux_calc, uxx_calc, step_loss

        key, t, x, u, ux, uxx, step_loss = jax.lax.fori_loop(0, traj_len, traj_calc, (key, t, x, u, ux, uxx, step_loss))
        pde_loss = jnp.sum(step_loss) * self.config.pde_scale 
        bc_loss = (jnp.sum((u - self.bc_fn(x))**2) + jnp.sum((ux - self.calc_bcx(x))**2)) * self.config.bc_scale
        return pde_loss + bc_loss, (pde_loss, bc_loss)

    def bsde_heun_grad(self, key, params):
        (total, (pde_loss, bc_loss)), grad = jax.value_and_grad(self.bsde_heun_loss, argnums=1, has_aux=True)(key, params)
        return (pde_loss, bc_loss), grad
    
    @partial(jax.jit, static_argnums=0)
    def jit_bsde_heun_loss(self, key, params):
        return self.bsde_heun_loss(key, params)[0]
    

    def bsde_heun_grad_batch(self, key, params):
        num_traj = self.config.batch
        micro_batch = self.config.micro_batch
        n_chunks = (num_traj + micro_batch - 1) // micro_batch
        traj_len = self.config.traj_len
        tau = 1/traj_len

        calc_uxx = jax.checkpoint(self.calc_uxx) if self.config.checkpointing else self.calc_uxx

        def make_loss(key, params):
            t = jnp.zeros((micro_batch, 1))
            x = self.get_X0(micro_batch)
            u, ux, uxx = calc_uxx(params, t, x)
            step_loss = jnp.zeros(traj_len)

            def traj_calc(i, inputs):
                key, t, x, u, ux, uxx, step_loss = inputs

                sigma = self.sigma(t, x, u, ux)
                dw = jnp.sqrt(tau) * jax.random.normal(key.newkey(), (micro_batch, self.config.d_in))
                dx_int = self.b_heun(t, x, u, ux)*tau + jnp.matmul(sigma, dw[..., jnp.newaxis])[..., 0]
                x_int = x + dx_int
                du_int = (self.h(t, x, u, ux) - self.c(t, x, u, ux, uxx))*tau + jnp.matmul(jnp.matmul(ux, self.sigma(t, x, u, ux)), dw[..., jnp.newaxis])[..., 0]

                t_new = t + tau
                u_int, ux_int, uxx_int = calc_uxx(params, t_new, x_int)
                
                x_new = x + 0.5*dx_int + 0.5*(self.b_heun(t_new, x_int, u_int, ux_int)*tau +
                                            jnp.matmul(self.sigma(t_new, x_int, u_int), dw[..., jnp.newaxis])[..., 0])
                u_new = u + 0.5*du_int + 0.5*((self.h(t_new, x_int, u_int, ux_int) - self.c(t_new, x_int, u_int, ux_int, uxx_int))*tau +
                                            jnp.matmul(jnp.matmul(ux_int, self.sigma(t_new, x_int, u_int)), dw[..., jnp.newaxis])[..., 0])
                
                u_calc, ux_calc, uxx_calc = calc_uxx(params, t_new, x_new)

                step_loss = step_loss.at[i].set(jnp.sum((u_new - u_calc)**2))
                return key, t_new, x_new, u_calc, ux_calc, uxx_calc, step_loss
            
            key, t, x, u, ux, uxx, step_loss = jax.lax.fori_loop(0, traj_len, traj_calc, (key, t, x, u, ux, uxx, step_loss))
            pde_loss = jnp.sum(step_loss) * self.config.pde_scale 
            bc_loss = (jnp.sum((u - self.bc_fn(x))**2) + jnp.sum((ux - self.calc_bcx(x))**2)) * self.config.bc_scale
            return pde_loss + bc_loss, (pde_loss, bc_loss)

        grad_make_loss = jax.jit(jax.value_and_grad(make_loss, argnums=1, has_aux=True))

        def chunk_loop(carry, _):
            key, pde_loss_acc, bc_loss_acc, grad_acc = carry
            (total, (pde_loss, bc_loss)), grad = grad_make_loss(key, params)
            pde_loss_acc = pde_loss_acc + pde_loss
            bc_loss_acc = bc_loss_acc + bc_loss
            grad_acc = jax.tree_util.tree_map(lambda a, b: a+b, grad_acc, grad)
            return (key, pde_loss_acc, bc_loss_acc, grad_acc), None
        
        (key, pde_loss, bc_loss, grad), _ = jax.lax.scan(chunk_loop, (key, 0.0, 0.0, jax.tree_util.tree_map(jnp.zeros_like, params)), None, length=n_chunks)
        return (pde_loss, bc_loss), grad

    # ------------------------------

    def reg_loss(self, key, params):
        t_pde, x_pde = self.sample_domain(key, self.config.batch * self.config.traj_len)
        pred = self.calc_u(params, t_pde, x_pde)
        true = jax.vmap(self.analytic_u, in_axes=(0, 0))(t_pde, x_pde)
        loss = jnp.mean(jnp.sqrt(jnp.sum((pred - true)**2, axis=1) / jnp.sum(true**2, axis=1)))
        return (loss,)
    
    # ------------------------------

    def bsde_heun_new_loss(self, key, params):
        num_traj = self.config.batch
        traj_len = self.config.traj_len
        tau = 1/traj_len
        
        calc_uxx = jax.checkpoint(self.calc_uxx) if self.config.checkpointing else self.calc_uxx

        t = jnp.zeros((num_traj, 1))
        x = self.get_X0(num_traj)
        u, ux, uxx = calc_uxx(params, t, x)
        step_loss = jnp.zeros(traj_len)
        
        def traj_calc(i, inputs):
            key, t, x, u, ux, uxx, step_loss = inputs

            sigma = self.sigma(t, x, u, ux)
            dw = jnp.sqrt(tau) * jax.random.normal(key.newkey(), (num_traj, self.config.d_in))
            dx_int = self.b_heun(t, x, u, ux)*tau + jnp.matmul(sigma, dw[..., jnp.newaxis])[..., 0]
            x_int = x + dx_int
            du_int = (self.h(t, x, u, ux) - self.c(t, x, u, ux, uxx))*tau + jnp.matmul(jnp.matmul(ux, self.sigma(t, x, u, ux)), dw[..., jnp.newaxis])[..., 0]
            u_int = u + du_int

            t_new = t + tau
            _, ux_int, uxx_int = calc_uxx(params, t_new, x_int)
            
            x_new = x + 0.5*dx_int + 0.5*(self.b_heun(t_new, x_int, u_int, ux_int)*tau +
                                          jnp.matmul(self.sigma(t_new, x_int, u_int), dw[..., jnp.newaxis])[..., 0])
            u_new = u + 0.5*du_int + 0.5*((self.h(t_new, x_int, u_int, ux_int) - self.c(t_new, x_int, u_int, ux_int, uxx_int))*tau +
                                          jnp.matmul(jnp.matmul(ux_int, self.sigma(t_new, x_int, u_int)), dw[..., jnp.newaxis])[..., 0])
            
            u_calc, ux_calc, uxx_calc = calc_uxx(params, t_new, x_new)

            step_loss = step_loss.at[i].set(jnp.sum((u_new - u_calc)**2))
            return key, t_new, x_new, u_calc, ux_calc, uxx_calc, step_loss

        key, t, x, u, ux, uxx, step_loss = jax.lax.fori_loop(0, traj_len, traj_calc, (key, t, x, u, ux, uxx, step_loss))
        pde_loss = jnp.sum(step_loss) * self.config.pde_scale 
        bc_loss = (jnp.sum((u - self.bc_fn(x))**2) + jnp.sum((ux - self.calc_bcx(x))**2)) * self.config.bc_scale
        return pde_loss + bc_loss, (pde_loss, bc_loss)

    def bsde_heun_new_grad(self, key, params):
        (total, (pde_loss, bc_loss)), grad = jax.value_and_grad(self.bsde_heun_new_loss, argnums=1, has_aux=True)(key, params)
        return (pde_loss, bc_loss), grad
    
    @partial(jax.jit, static_argnums=0)
    def jit_bsde_heun_new_loss(self, key, params):
        return self.bsde_heun_new_loss(key, params)[0]
    

    def bsde_heun_new_grad_batch(self, key, params):
        num_traj = self.config.batch
        micro_batch = self.config.micro_batch
        n_chunks = (num_traj + micro_batch - 1) // micro_batch
        traj_len = self.config.traj_len
        tau = 1/traj_len

        calc_uxx = jax.checkpoint(self.calc_uxx) if self.config.checkpointing else self.calc_uxx

        def make_loss(key, params):
            t = jnp.zeros((micro_batch, 1))
            x = self.get_X0(micro_batch)
            u, ux, uxx = calc_uxx(params, t, x)
            step_loss = jnp.zeros(traj_len)

            def traj_calc(i, inputs):
                key, t, x, u, ux, uxx, step_loss = inputs

                sigma = self.sigma(t, x, u, ux)
                dw = jnp.sqrt(tau) * jax.random.normal(key.newkey(), (micro_batch, self.config.d_in))
                dx_int = self.b_heun(t, x, u, ux)*tau + jnp.matmul(sigma, dw[..., jnp.newaxis])[..., 0]
                x_int = x + dx_int
                du_int = (self.h(t, x, u, ux) - self.c(t, x, u, ux, uxx))*tau + jnp.matmul(jnp.matmul(ux, self.sigma(t, x, u, ux)), dw[..., jnp.newaxis])[..., 0]
                u_int = u + du_int

                t_new = t + tau
                _, ux_int, uxx_int = calc_uxx(params, t_new, x_int)
                
                x_new = x + 0.5*dx_int + 0.5*(self.b_heun(t_new, x_int, u_int, ux_int)*tau +
                                            jnp.matmul(self.sigma(t_new, x_int, u_int), dw[..., jnp.newaxis])[..., 0])
                u_new = u + 0.5*du_int + 0.5*((self.h(t_new, x_int, u_int, ux_int) - self.c(t_new, x_int, u_int, ux_int, uxx_int))*tau +
                                            jnp.matmul(jnp.matmul(ux_int, self.sigma(t_new, x_int, u_int)), dw[..., jnp.newaxis])[..., 0])
                
                u_calc, ux_calc, uxx_calc = calc_uxx(params, t_new, x_new)

                step_loss = step_loss.at[i].set(jnp.sum((u_new - u_calc)**2))
                return key, t_new, x_new, u_calc, ux_calc, uxx_calc, step_loss
            
            key, t, x, u, ux, uxx, step_loss = jax.lax.fori_loop(0, traj_len, traj_calc, (key, t, x, u, ux, uxx, step_loss))
            pde_loss = jnp.sum(step_loss) * self.config.pde_scale 
            bc_loss = (jnp.sum((u - self.bc_fn(x))**2) + jnp.sum((ux - self.calc_bcx(x))**2)) * self.config.bc_scale
            return pde_loss + bc_loss, (pde_loss, bc_loss)

        grad_make_loss = jax.jit(jax.value_and_grad(make_loss, argnums=1, has_aux=True))

        def chunk_loop(carry, _):
            key, pde_loss_acc, bc_loss_acc, grad_acc = carry
            (total, (pde_loss, bc_loss)), grad = grad_make_loss(key, params)
            pde_loss_acc = pde_loss_acc + pde_loss
            bc_loss_acc = bc_loss_acc + bc_loss
            grad_acc = jax.tree_util.tree_map(lambda a, b: a+b, grad_acc, grad)
            return (key, pde_loss_acc, bc_loss_acc, grad_acc), None
        
        (key, pde_loss, bc_loss, grad), _ = jax.lax.scan(chunk_loop, (key, 0.0, 0.0, jax.tree_util.tree_map(jnp.zeros_like, params)), None, length=n_chunks)
        return (pde_loss, bc_loss), grad
    
    # ------------------------------

    def bsde_heun_ode_loss(self, key, params):
        num_traj = self.config.batch
        traj_len = self.config.traj_len
        tau = 1/traj_len
        
        calc_uxx = jax.checkpoint(self.calc_uxx) if self.config.checkpointing else self.calc_uxx

        t = jnp.zeros((num_traj, 1))
        x = self.get_X0(num_traj)
        u, ux, uxx = calc_uxx(params, t, x)
        step_loss = jnp.zeros(traj_len)
        
        def traj_calc(i, inputs):
            key, t, x, u, ux, uxx, step_loss = inputs

            sigma = self.sigma(t, x, u, ux)
            dw = jnp.sqrt(tau) * jax.random.normal(key.newkey(), (num_traj, self.config.d_in))
            dx_int = self.b_heun(t, x, u, ux)*tau + jnp.matmul(sigma, dw[..., jnp.newaxis])[..., 0]
            x_int = x + dx_int
            du_int = (self.h(t, x, u, ux) - self.c(t, x, u, ux, uxx))*tau + jnp.matmul(jnp.matmul(ux, self.sigma(t, x, u, ux)), dw[..., jnp.newaxis])[..., 0]
            u_int = u + du_int

            t_new = t + tau
            _, ux_int, uxx_int = calc_uxx(params, t_new, x_int)
            
            x_new = x + 0.5*dx_int + 0.5*(self.b_heun(t_new, x_int, u_int, ux_int)*tau +
                                          jnp.matmul(self.sigma(t_new, x_int, u_int), dw[..., jnp.newaxis])[..., 0])
            u_new = u + 0.5*du_int + 0.5*((self.h(t_new, x_int, u_int, ux_int) - self.c(t_new, x_int, u_int, ux_int, uxx_int))*tau +
                                          jnp.matmul(jnp.matmul(ux_int, self.sigma(t_new, x_int, u_int)), dw[..., jnp.newaxis])[..., 0])
            
            u_calc, ux_calc, uxx_calc = calc_uxx(params, t_new, x_new)

            step_loss = step_loss.at[i].set(jnp.sum((u_new - u_calc)**2))
            return key, t_new, x_new, u_calc, ux_calc, uxx_calc, step_loss

        key, t, x, u, ux, uxx, step_loss = jax.lax.fori_loop(0, traj_len, traj_calc, (key, t, x, u, ux, uxx, step_loss))
        pde_loss = jnp.sum(step_loss) * self.config.pde_scale 
        bc_loss = (jnp.sum((u - self.bc_fn(x))**2) + jnp.sum((ux - self.calc_bcx(x))**2)) * self.config.bc_scale
        return pde_loss + bc_loss, (pde_loss, bc_loss)

    def bsde_heun_new_grad(self, key, params):
        (total, (pde_loss, bc_loss)), grad = jax.value_and_grad(self.bsde_heun_new_loss, argnums=1, has_aux=True)(key, params)
        return (pde_loss, bc_loss), grad
    
    @partial(jax.jit, static_argnums=0)
    def jit_bsde_heun_new_loss(self, key, params):
        return self.bsde_heun_new_loss(key, params)[0]
    

    def bsde_heun_new_grad_batch(self, key, params):
        num_traj = self.config.batch
        micro_batch = self.config.micro_batch
        n_chunks = (num_traj + micro_batch - 1) // micro_batch
        traj_len = self.config.traj_len
        tau = 1/traj_len

        calc_uxx = jax.checkpoint(self.calc_uxx) if self.config.checkpointing else self.calc_uxx

        def make_loss(key, params):
            t = jnp.zeros((micro_batch, 1))
            x = self.get_X0(micro_batch)
            u, ux, uxx = calc_uxx(params, t, x)
            step_loss = jnp.zeros(traj_len)

            def traj_calc(i, inputs):
                key, t, x, u, ux, uxx, step_loss = inputs

                sigma = self.sigma(t, x, u, ux)
                dw = jnp.sqrt(tau) * jax.random.normal(key.newkey(), (micro_batch, self.config.d_in))
                dx_int = self.b_heun(t, x, u, ux)*tau + jnp.matmul(sigma, dw[..., jnp.newaxis])[..., 0]
                x_int = x + dx_int
                du_int = (self.h(t, x, u, ux) - self.c(t, x, u, ux, uxx))*tau + jnp.matmul(jnp.matmul(ux, self.sigma(t, x, u, ux)), dw[..., jnp.newaxis])[..., 0]
                u_int = u + du_int

                t_new = t + tau
                _, ux_int, uxx_int = calc_uxx(params, t_new, x_int)
                
                x_new = x + 0.5*dx_int + 0.5*(self.b_heun(t_new, x_int, u_int, ux_int)*tau +
                                            jnp.matmul(self.sigma(t_new, x_int, u_int), dw[..., jnp.newaxis])[..., 0])
                u_new = u + 0.5*du_int + 0.5*((self.h(t_new, x_int, u_int, ux_int) - self.c(t_new, x_int, u_int, ux_int, uxx_int))*tau +
                                            jnp.matmul(jnp.matmul(ux_int, self.sigma(t_new, x_int, u_int)), dw[..., jnp.newaxis])[..., 0])
                
                u_calc, ux_calc, uxx_calc = calc_uxx(params, t_new, x_new)

                step_loss = step_loss.at[i].set(jnp.sum((u_new - u_calc)**2))
                return key, t_new, x_new, u_calc, ux_calc, uxx_calc, step_loss
            
            key, t, x, u, ux, uxx, step_loss = jax.lax.fori_loop(0, traj_len, traj_calc, (key, t, x, u, ux, uxx, step_loss))
            pde_loss = jnp.sum(step_loss) * self.config.pde_scale 
            bc_loss = (jnp.sum((u - self.bc_fn(x))**2) + jnp.sum((ux - self.calc_bcx(x))**2)) * self.config.bc_scale
            return pde_loss + bc_loss, (pde_loss, bc_loss)

        grad_make_loss = jax.jit(jax.value_and_grad(make_loss, argnums=1, has_aux=True))

        def chunk_loop(carry, _):
            key, pde_loss_acc, bc_loss_acc, grad_acc = carry
            (total, (pde_loss, bc_loss)), grad = grad_make_loss(key, params)
            pde_loss_acc = pde_loss_acc + pde_loss
            bc_loss_acc = bc_loss_acc + bc_loss
            grad_acc = jax.tree_util.tree_map(lambda a, b: a+b, grad_acc, grad)
            return (key, pde_loss_acc, bc_loss_acc, grad_acc), None
        
        (key, pde_loss, bc_loss, grad), _ = jax.lax.scan(chunk_loop, (key, 0.0, 0.0, jax.tree_util.tree_map(jnp.zeros_like, params)), None, length=n_chunks)
        return (pde_loss, bc_loss), grad

    # --------------------------------------------------
    # Optimization Methods
    # --------------------------------------------------
    
    @partial(jax.jit, static_argnums=0)
    def optimize(self, key, params, opt_state):
        losses, grad = self.loss_fn(key, params)
        loss = jnp.sum(jnp.asarray(losses))

        updates, opt_state = self.optimizer.update(grad, opt_state)
        params = optax.apply_updates(params, updates)

        return loss, losses, params, opt_state, key

    
class Controller():

    def __init__(self, solver: Solver, seed = 20226074):
        self.solver = solver
        self.key = Key.create_key(seed)
        self.params, self.opt_state = self.solver.init_solver(self.key)

    def step(self, i):
        self.key.newkey()
        loss, losses, self.params, self.opt_state, self.key = self.solver.optimize(self.key, self.params, self.opt_state)

        if self.solver.config.save_to_wandb:
            wandb.log({'loss': loss, **{'loss'+str(k+1): v for k, v in dict(enumerate(losses)).items()}}, step=i)
            if self.solver.config.track_pinns_loss:
                pinns_loss = self.solver.jit_pinns_loss(self.key, self.params)
                wandb.log({"pinns Loss": pinns_loss}, step=i)
            if self.solver.config.track_fspinns_loss:
                fspinns_loss = self.solver.jit_fspinns_loss(self.key, self.params)
                wandb.log({"fspinns Loss": fspinns_loss}, step=i)
            if self.solver.config.track_bsde_loss:
                bsde_loss = self.solver.jit_bsde_loss(self.key, self.params)
                wandb.log({"bsde Loss": bsde_loss}, step=i)
            if self.solver.config.track_bsde_heun_loss:
                bsde_heun_loss = self.solver.jit_bsde_heun_loss(self.key, self.params)
                wandb.log({"fspinns Loss": bsde_heun_loss}, step=i)
            if self.solver.config.analytic_sol:
                RL2, RL_T0 = self.solver.jit_calc_RL(self.params)
                wandb.log({"RL2": RL2, "RL_T0": RL_T0}, step=i)
            if self.solver.config.custom_eval:
                custom_eval = self.solver.jit_calc_eval(self.params)
                wandb.log({"eval": custom_eval}, step=i)
            if i%(self.solver.config.iter//self.solver.config.num_figures) == 0:
                if self.solver.config.analytic_sol:
                    self.solver.plot_pred(self.params, i)
                if self.solver.config.custom_eval:
                    self.solver.plot_eval(self.params, i)

    def solve(self):
        for i in tqdm.tqdm(range(self.solver.config.iter)):
            self.step(i)

        if self.solver.config.save_to_wandb:
            if self.solver.config.analytic_sol:
                self.solver.plot_pred(self.params, self.solver.config.iter)
            if self.solver.config.custom_eval:
                self.solver.plot_eval(self.params, self.solver.config.iter)
        
        self.solver.close()

    def tab_model(self):
        self.solver.tab_model(self.key)