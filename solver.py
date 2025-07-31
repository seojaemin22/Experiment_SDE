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

        self.loss_fn = self.select_loss(self.config.loss_method)
        self.model = self.create_model()
        self.optimizer = self.create_opt()
        if self.config.analytic_sol:
            self.sol_T, self.sol_X, self.sol_U = self.get_analytic_sol()
        if self.config.custom_eval:
            self.eval_point = self.get_eval_point()
        if self.config.save_to_wandb:
            self.init_wandb()

    def get_base_config():
        return Config()
    
    def select_loss(self, loss_method):
        if loss_method == 'pinns':
            return self.pinns_grad_batch if self.config.micro_batch < self.config.batch else self.pinns_grad
        elif loss_method == 'regress':
            return self.reg_grad  # Always full micro-batch
        else:
            raise Exception("Loss Method '" + loss_method + "' Not Implemented")
    

    def init_solver(self, key: Key):
        params = self.init_model(key)
        opt_state = self.init_opt(params)

        model_path = Path(self.config.model_state)
        if model_path.exists():
            model_bytes = model_path.read_bytes()
            params = fs.from_bytes(params, model_bytes)
        
        opt_path = Path(self.config.opt_state)
        if opt_path.exists():
            opt_bytes = opt_path.read_bytes()
            opt_state = fs.from_bytes(opt_state, opt_bytes)

        num_params = sum(x.size for x in jax.tree_util.tree_leaves(params))
        if self.config.save_to_wandb:
            wandb.config['# Params'] =  num_params
        return params, opt_state

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
    
    def tab_model(self, key:Key):
        t_pde, x_pde = self.sample_domain(key,self.config.batch_pde)
        tab_fn = nn.tabulate(self.model,key.newkey())
        print(tab_fn(t_pde, x_pde))


    # --------------------------------------------------
    # Calculation Methods
    # --------------------------------------------------

    def calc_u(self, params, t, x):
        return self.model.apply(params, t, x)

    def calc_ut(self, params, t, x):
        def t_func(t, x):
            model_fn = lambda tt: self.calc_u(params, tt, x)
            u, du_dt = jax.vjp(model_fn, t)
            ut = jax.vmap(du_dt, in_axes=0)(jnp.eye(len(u)))[0]
            return u, ut
        return jax.vmap(t_func, in_axes=(0,0))(t, x)
    
    def calc_ux(self, params, t, x):
        def jacrev(t, x):
            model_fn = lambda xx: self.calc_u(params, t, xx)
            u, vjp_fun = jax.vjp(model_fn, x)
            ux = jax.vmap(vjp_fun, in_axes=0)(jnp.eye(len(u)))[0]
            return u, ux
        return jax.vmap(jacrev, in_axes=0)(t, x)
    
    def calc_uxx(self, params, t, x):
        def jacrev2(t, x):
            model_fn = lambda xx: self.calc_u(params, t, xx)
            def jacrev(x):
                u, vjp_fun = jax.vjp(model_fn, x)
                ux = jax.vmap(vjp_fun, in_axes=0)(jnp.eye(len(u)))[0]
                return ux, u
            func2 = lambda s: jax.jvp(jacrev, (x,), (s,), has_aux=True)
            ux, uxx, u = jax.vmap(func2, in_axes=1, out_axes=(None,1,None))(jnp.eye(len(x)))
            return u, ux, uxx
        return jax.vmap(jacrev2, in_axes=(0,0))(t, x)

    def bc_fn(self, x):
        pass

    def calc_bcx(self, x): 
        jax_x = jax.jacrev(lambda x: self.bc_fn(x), argnums=0)
        return jax.vmap(jax_x, in_axes=0)(x)

    def analytic_u(self, t, x):
        pass

    def analytic_ut(self, t, x):
        def t_func(t, x):
            model_fn = lambda tt: self.analytic_u(tt, x)
            u, du_dt = jax.vjp(model_fn, t)
            ut = jax.vmap(du_dt, in_axes=0)(jnp.eye(len(u)))[0]
            return u, ut
        return jax.vmap(t_func, in_axes=(0,0))(t, x)

    def analytic_ux(self, t, x, output_pos=(0,)):
        def jacrev(t, x):
            model_fn = lambda xx: self.analytic_u(t, xx)
            u, vjp_fun = jax.vjp(model_fn, x)
            ux = jax.vmap(vjp_fun, in_axes=0)(jnp.eye(len(u)))[0]
            return u, ux
        return jax.vmap(jacrev, in_axes=0)(t, x)
    
    def analytic_uxx(self, t, x, output_pos=(0,)):
        def jacrev2(t, x):
            model_fn = lambda xx: self.analytic_u(t, xx)
            def jacrev(x):
                u, vjp_fun = jax.vjp(model_fn, x)
                ux = jax.vmap(vjp_fun, in_axes=0)(jnp.eye(len(u)))[0]
                return ux, u
            func2 = lambda s: jax.jvp(jacrev, (x,), (s,), has_aux=True)
            ux, uxx, u = jax.vmap(func2, in_axes=1, out_axes=(None,1,None))(jnp.eye(len(x)))
            return u, ux, uxx
        return jax.vmap(jacrev2, in_axes=(0,0))(t, x)

    b_y = False
    b_z = False
    def b(self, t, x, y=None, z=None):  # -> (batch, d_in)
        return jnp.zeros_like(x)
    
    sigma_y = False 
    sigma_z = False
    def sigma(self, t, x, y=None, z=None):  # -> (batch, d_in, d_in)
        return jnp.repeat(jnp.expand_dims(jnp.eye(x.shape[-1]), axis=0), x.shape[0], axis=0)
    
    def h(self, t, x, y, z):  # -> (batch, d_out)
        return jnp.zeros_like(y)


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
    
    def get_analytic_X(self, T, W):
        X = jnp.zeros_like(W)
        X = X.at[:, 0, :].set(self.get_X0(X.shape[0]))
        dt = self.config.dt

        def loop(i, X):
            yz_dict = {}
            if self.b_z or self.sigma_z:
                yz_dict['y'], yz_dict['z'] = self.analytic_ux(T[:, i-1, :], X[:, i-1, :])
            elif self.b_y or self.sigma_y:
                yz_dict['y'] = self.analytic_u(T[:, i-1, :], X[:, i-1, :])
            X = X.at[:, i, :].set(X[:, i-1, :] + self.b(T[:, i-1, :], X[:, i-1, :], **yz_dict)*dt + 
                                  jnp.matmul(self.sigma(T[:, i-1, :], X[:, i-1, :], **yz_dict), W[:, i, :, jnp.newaxis] - W[:, i-1, :, jnp.newaxis]))
            return X
        
        X = jax.lax.fori_loop(1, self.config.traj_len+1, loop, X)
        return X

    def get_analytic_sol(self):
        num_traj = 5
        traj_len = self.config.traj_len
        dt = self.config.dt
        
        T = jnp.repeat(jnp.linspace(0, 1, traj_len+1)[jnp.newaxis, ..., jnp.newaxis], num_traj, axis=0)
        dW = jnp.sqrt(dt) * jnp.concatenate((jnp.zeros((num_traj, 1, self.config.d_in)),
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
        plt.ylim(1e-5, 1e+1)
        plt.close(fig_L1)
        wandb.log({'L1 Error': wandb.Image(fig_L1)}, step=i)

    def get_eval_point(self):
        pass

    def plot_eval(self, params):
        pass

    def calc_RL(self, params):
        pred = self.calc_u(params, self.sol_T, self.sol_X)
        true = self.sol_U

        RL2 = jnp.mean(jnp.sqrt(jnp.sum((pred - true)**2, axis=1) / (jnp.sum(true**2, axis=1))))
        RL_T0 = jnp.mean(jnp.abs(pred[:, 0] - true[:, 0]) / (jnp.abs(true[:, 0])))
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

    def reg_loss(self, key, params):
        t_pde, x_pde = self.sample_domain(key, self.config.batch * self.config.traj_len)
        pred = self.calc_u(params, t_pde, x_pde)
        true = jax.vmap(self.analytic_u, in_axes=(0, 0))(t_pde, x_pde)
        loss = jnp.mean(jnp.sqrt(jnp.sum((pred - true)**2, axis=1) / jnp.sum(true**2, axis=1)))
        return loss
    
    def reg_grad(self, key, params):
        loss, grad = jax.value_and_grad(self.reg_loss, argnums=1, has_aux=True)(key, params)
        return (loss,), grad
    

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


# --------------------------------------------------
# Control Class
# --------------------------------------------------

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
        
        path = Path('./checkpoints/')
        path.mkdir(exist_ok=True)
        if self.solver.config.save_model:
            model_bytes = fs.to_bytes(self.params)
            (path/f'{self.solver.config.project_name}_{self.solver.config.run_name}_model.msgpack').write_bytes(model_bytes)
        if self.solver.config.save_opt:
            opt_bytes = fs.to_Bytes(self.opt_state)
            (path/f'{self.solver.config.project_name}_{self.solver.config.run_name}_opt.msgpack').write_bytes(opt_bytes)

        self.solver.close()

    def tab_model(self):
        self.solver.tab_model(self.key)




# --------------------------------------------------
# Child class for PDE
# --------------------------------------------------

class PDE_Solver(Solver):

    def get_base_config():
        return PDE_Config()

    def select_loss(self, loss_method):
        if loss_method == 'pinns':
            return self.pinns_grad_batch if self.config.micro_batch < self.config.batch else self.pinns_grad
        elif loss_method == 'fspinns':
            return self.fspinns_grad_batch if self.config.micro_batch < self.config.batch else self.fspinns_grad
        elif loss_method == 'bsde':
            return self.bsde_grad_batch if self.config.micro_batch < self.config.batch else self.bsde_grad
        elif loss_method == 'bsdeskip':
            return self.bsde_skip_grad_batch if self.config.micro_batch < self.config.batch else self.bsde_skip_grad
        elif loss_method == 'bsdeheun':
            return self.bsde_heun_grad_batch if self.config.micro_batch < self.config.batch else self.bsde_heun_grad
        elif loss_method == 'regress':
            return self.reg_grad  # Always full micro-batch
        else:
            raise Exception("Loss Method '" + loss_method + "' Not Implemented")
    
    
    # --------------------------------------------------
    # Calculation Methods
    # --------------------------------------------------

    b_heun_y = False
    b_heun_z = False
    def b_heun(self, t, x, y=None, z=None):  # b + Correction of Forward Stratonovich SDE (- 1/2 sigma sigma_x)
        return jnp.zeros_like(x)
    
    def c(self, t, x, u, ux, uxx):  # Correction of Backward Stratnovich SDE 
        return 0.5 * jnp.trace(jnp.matmul(jnp.matmul(self.sigma(t, x, u), self.sigma(t, x, u)), uxx[:,0]), axis1=-1, axis2=-2)[..., jnp.newaxis]


    # --------------------------------------------------
    # Loss Methods
    # --------------------------------------------------

    def fspinns_loss(self, key, params):
        num_traj = self.config.batch
        traj_len = self.config.traj_len
        dt = self.config.dt

        T = jnp.repeat(jnp.linspace(0, 1, traj_len+1)[jnp.newaxis, ..., jnp.newaxis], num_traj, axis=0)
        dW = jnp.sqrt(dt) * jnp.concatenate((jnp.zeros((num_traj, 1, self.config.d_in)),
                                             jax.random.normal(key.newkey(), (num_traj, traj_len, self.config.d_in))), axis=1)
        X = jnp.zeros((num_traj, traj_len+1, self.config.d_in))
        X = X.at[:, 0, :].set(self.get_X0(num_traj))

        def loop(i, X):
            yz_dict = {}
            if self.b_z or self.sigma_z:
                yz_dict['y'], yz_dict['z'] = self.calc_ux(params, T[:, i-1, :], X[:, i-1, :])
            elif self.b_y or self.sigma_y:
                yz_dict['y'] = self.calc_u(params, T[:, i-1, :], X[:, i-1, :])
            X = X.at[:, i, :].set(X[:, i-1, :] + self.b(T[:, i-1, :], X[:, i-1, :], **yz_dict)*dt + jnp.matmul(self.sigma(T[:, i-1, :], X[:, i-1, :], **yz_dict), dW[:, i, :, jnp.newaxis])[..., 0])
            return X
        
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
        dt = self.config.dt

        def make_loss(key, params):
            T = jnp.repeat(jnp.linspace(0, 1, traj_len+1)[jnp.newaxis, ..., jnp.newaxis], micro_batch, axis=0)
            dW = jnp.sqrt(dt) * jnp.concatenate((jnp.zeros((micro_batch, 1, self.config.d_in)),
                                                 jax.random.normal(key.newkey(), (micro_batch, traj_len, self.config.d_in))), axis=1)
            X = jnp.zeros((micro_batch, traj_len+1, self.config.d_in))
            X = X.at[:, 0, :].set(self.get_X0(micro_batch))

            def loop(i, X):
                yz_dict = {}
                if self.b_z or self.sigma_z:
                    yz_dict['y'], yz_dict['z'] = self.calc_ux(params, T[:, i-1, :], X[:, i-1, :])
                elif self.b_y or self.sigma_y:
                    yz_dict['y'] = self.calc_u(params, T[:, i-1, :], X[:, i-1, :])
                X = X.at[:, i, :].set(X[:, i-1, :] + self.b(T[:, i-1, :], X[:, i-1, :], **yz_dict)*dt + jnp.matmul(self.sigma(T[:, i-1, :], X[:, i-1, :], **yz_dict), dW[:, i, :, jnp.newaxis])[..., 0])
                return X
            
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
        dt = self.config.dt
        
        t = jnp.zeros((num_traj, 1))
        x = self.get_X0(num_traj)
        u, ux = self.calc_ux(params, t, x)
        step_loss = jnp.zeros(traj_len)

        def traj_calc(i, inputs):
            key, t, x, u, ux, step_loss = inputs

            sigma = self.sigma(t, x, u, ux)
            dw = jnp.sqrt(dt) * jax.random.normal(key.newkey(), (num_traj, self.config.d_in))
            t_new = t + dt
            x_new = x + self.b(t, x, u, ux)*dt + jnp.matmul(sigma, dw[..., jnp.newaxis])[..., 0]
            u_new = u + self.h(t, x, u, ux)*dt + jnp.matmul(jnp.matmul(ux, sigma), dw[..., jnp.newaxis])[..., 0]

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
        dt = self.config.dt
        
        def make_loss(key, params):
            t = jnp.zeros((micro_batch, 1))
            x = self.get_X0(micro_batch)
            u, ux = self.calc_ux(params, t, x)
            step_loss = jnp.zeros(traj_len)

            def traj_calc(i, inputs):
                key, t, x, u, ux, step_loss = inputs

                sigma = self.sigma(t, x, u, ux)
                dw = jnp.sqrt(dt) * jax.random.normal(key.newkey(), (micro_batch, self.config.d_in))
                t_new = t + dt
                x_new = x + self.b(t, x, u, ux)*dt + jnp.matmul(sigma, dw[..., jnp.newaxis])[..., 0]
                u_new = u + self.h(t, x, u, ux)*dt + jnp.matmul(jnp.matmul(ux, sigma), dw[..., jnp.newaxis])[..., 0]

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
        dt = self.config.dt
        skip_len = self.config.skip_len
        
        t = jnp.zeros((num_traj, 1))
        x = self.get_X0(num_traj)
        u, ux = self.calc_ux(params, t, x)
        step_loss = jnp.zeros(traj_len)

        def traj_calc(i, inputs):
            key, t, x, u, ux, step_loss = inputs

            sigma = self.sigma(t, x, u, ux)
            dw = jnp.sqrt(dt) * jax.random.normal(key.newkey(), (num_traj, self.config.d_in))
            t_new = t + dt
            x_new = x + self.b(t, x, u, ux)*dt + jnp.matmul(sigma, dw[..., jnp.newaxis])[..., 0]
            u_new = u + self.h(t, x, u, ux)*dt + jnp.matmul(jnp.matmul(ux, sigma), dw[..., jnp.newaxis])[..., 0]

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
        dt = self.config.dt
        skip_len = self.config.skip_len
        
        def make_loss(key, params):
            t = jnp.zeros((micro_batch, 1))
            x = self.get_X0(micro_batch)
            u, ux = self.calc_ux(params, t, x)
            step_loss = jnp.zeros(traj_len)

            def traj_calc(i, inputs):
                key, t, x, u, ux, step_loss = inputs

                sigma = self.sigma(t, x, u, ux)
                dw = jnp.sqrt(dt) * jax.random.normal(key.newkey(), (micro_batch, self.config.d_in))
                t_new = t + dt
                x_new = x + self.b(t, x, u, ux)*dt + jnp.matmul(sigma, dw[..., jnp.newaxis])[..., 0]
                u_new = u + self.h(t, x, u, ux)*dt + jnp.matmul(jnp.matmul(ux, sigma), dw[..., jnp.newaxis])[..., 0]

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
        dt = self.config.dt
        
        calc_uxx = jax.checkpoint(self.calc_uxx) if self.config.checkpointing else self.calc_uxx

        t = jnp.zeros((num_traj, 1))
        x = self.get_X0(num_traj)
        u, ux, uxx = calc_uxx(params, t, x)
        step_loss = jnp.zeros(traj_len)
        
        def traj_calc(i, inputs):
            key, t, x, u, ux, uxx, step_loss = inputs

            sigma = self.sigma(t, x, u, ux)
            dw = jnp.sqrt(dt) * jax.random.normal(key.newkey(), (num_traj, self.config.d_in))
            dx_int = self.b_heun(t, x, u, ux)*dt + jnp.matmul(sigma, dw[..., jnp.newaxis])[..., 0]
            x_int = x + dx_int
            du_int = (self.h(t, x, u, ux) - self.c(t, x, u, ux, uxx))*dt + jnp.matmul(jnp.matmul(ux, self.sigma(t, x, u, ux)), dw[..., jnp.newaxis])[..., 0]
            u_int = u + du_int

            t_new = t + dt
            _, ux_int, uxx_int = calc_uxx(params, t_new, x_int)
            
            sigma = self.sigma(t_new, x_int, u_int, ux_int)
            x_new = x + 0.5*dx_int + 0.5*(self.b_heun(t_new, x_int, u_int, ux_int)*dt + jnp.matmul(sigma, dw[..., jnp.newaxis])[..., 0])
            u_new = u + 0.5*du_int + 0.5*((self.h(t_new, x_int, u_int, ux_int) - self.c(t_new, x_int, u_int, ux_int, uxx_int))*dt + jnp.matmul(jnp.matmul(ux_int, sigma), dw[..., jnp.newaxis])[..., 0])
            
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
        dt = self.config.dt

        calc_uxx = jax.checkpoint(self.calc_uxx) if self.config.checkpointing else self.calc_uxx

        def make_loss(key, params):
            t = jnp.zeros((micro_batch, 1))
            x = self.get_X0(micro_batch)
            u, ux, uxx = calc_uxx(params, t, x)
            step_loss = jnp.zeros(traj_len)

            def traj_calc(i, inputs):
                key, t, x, u, ux, uxx, step_loss = inputs

                sigma = self.sigma(t, x, u, ux)
                dw = jnp.sqrt(dt) * jax.random.normal(key.newkey(), (micro_batch, self.config.d_in))
                dx_int = self.b_heun(t, x, u, ux)*dt + jnp.matmul(sigma, dw[..., jnp.newaxis])[..., 0]
                x_int = x + dx_int
                du_int = (self.h(t, x, u, ux) - self.c(t, x, u, ux, uxx))*dt + jnp.matmul(jnp.matmul(ux, self.sigma(t, x, u, ux)), dw[..., jnp.newaxis])[..., 0]

                t_new = t + dt
                u_int, ux_int, uxx_int = calc_uxx(params, t_new, x_int)
                
                sigma = self.sigma(t_new, x_int, u_int, ux_int)
                x_new = x + 0.5*dx_int + 0.5*(self.b_heun(t_new, x_int, u_int, ux_int)*dt + jnp.matmul(sigma, dw[..., jnp.newaxis])[..., 0])
                u_new = u + 0.5*du_int + 0.5*((self.h(t_new, x_int, u_int, ux_int) - self.c(t_new, x_int, u_int, ux_int, uxx_int))*dt + jnp.matmul(jnp.matmul(ux_int, sigma), dw[..., jnp.newaxis])[..., 0])
            
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

    
class PDE_Controller(Controller):

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
                wandb.log({"bsdehuen Loss": bsde_heun_loss}, step=i)
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





# --------------------------------------------------
# FO-PINN based PDE Solver
# --------------------------------------------------

class PDE_FO_Solver(PDE_Solver):

    def get_base_config():
        return PDE_FO_Config()
    
    def __init__(self, config: PDE_FO_Config):
        self.config = copy.deepcopy(config)
        self.config.d_out = self.config.d_out + 1 + self.config.d_in

        self.loss_fn = self.select_loss(self.config.loss_method)
        self.model = self.create_model()
        self.optimizer = self.create_opt()
        if self.config.analytic_sol:
            self.sol_T, self.sol_X, self.sol_U = self.get_analytic_sol()
        if self.config.custom_eval:
            self.eval_point = self.get_eval_point()
        if self.config.save_to_wandb:
            self.init_wandb()
    
    def select_loss(self, loss_method):
        if loss_method == 'pinns':
            return self.pinns_grad_batch if self.config.micro_batch < self.config.batch else self.pinns_grad
        elif loss_method == 'fspinns':
            return self.fspinns_grad_batch if self.config.micro_batch < self.config.batch else self.fspinns_grad
        elif loss_method == 'bsde':
            return self.bsde_grad_batch if self.config.micro_batch < self.config.batch else self.bsde_grad
        elif loss_method == 'bsdeheun':
            return self.bsde_heun_grad_batch if self.config.micro_batch < self.config.batch else self.bsde_heun_grad
        else:
            raise Exception("Loss Method '" + loss_method + "' Not Implemented")
    

    # --------------------------------------------------
    # Calculation Methods
    # --------------------------------------------------

    def calc_u(self, params, t, x):
        return self.model.apply(params, t, x)[..., :(self.config.d_out-self.config.d_in-1)]
    
    def calc_ut(self, params, t, x):
        u = self.model.apply(params, t, x)
        return u[..., :(self.config.d_out-self.config.d_in-1)], u[..., (self.config.d_out-self.config.d_in-1):(self.config.d_out-self.config.d_in)]

    def calc_ux(self, params, t, x):
        u = self.model.apply(params, t, x)
        return u[..., :(self.config.d_out-self.config.d_in-1)], u[..., (self.config.d_out-self.config.d_in):]
    
    def calc_uxx(self, params, t, x):
        def jacrev2(t, x):
            def jacrev(x):
                real_u = self.model.apply(params, t, x)
                u = real_u[..., :(self.config.d_out-self.config.d_in-1)]
                ux = real_u[..., jnp.newaxis, (self.config.d_out-self.config.d_in):]
                return ux, u
            func2 = lambda s: jax.jvp(jacrev, (x,), (s,), has_aux=True)
            ux, uxx, u = jax.vmap(func2, in_axes=1, out_axes=(None,1,None))(jnp.eye(len(x)))
            return u, ux, uxx
        return jax.vmap(jacrev2, in_axes=(0,0))(t, x)
    
    def real_calc_ut(self, params, t, x):
        def t_func(t, x):
            model_fn = lambda tt: self.calc_u(params, tt, x)
            u, du_dt = jax.vjp(model_fn, t)
            ut = jax.vmap(du_dt, in_axes=0)(jnp.eye(len(u)))[0]
            return u, ut
        return jax.vmap(t_func, in_axes=(0,0))(t, x)
    
    def real_calc_ux(self, params, t, x):
        def jacrev(t, x):
            model_fn = lambda xx: self.calc_u(params, t, xx)
            u, vjp_fun = jax.vjp(model_fn, x)
            ux = jax.vmap(vjp_fun, in_axes=0)(jnp.eye(len(u)))[0]
            return u, ux
        return jax.vmap(jacrev, in_axes=0)(t, x)
    
    def real_calc_uxx(self, params, t, x):
        def jacrev2(t, x):
            model_fn = lambda xx: self.calc_u(params, t, xx)
            def jacrev(x):
                u, vjp_fun = jax.vjp(model_fn, x)
                ux = jax.vmap(vjp_fun, in_axes=0)(jnp.eye(len(u)))[0]
                return ux, u
            func2 = lambda s: jax.jvp(jacrev, (x,), (s,), has_aux=True)
            ux, uxx, u = jax.vmap(func2, in_axes=1, out_axes=(None,1,None))(jnp.eye(len(x)))
            return u, ux, uxx
        return jax.vmap(jacrev2, in_axes=(0,0))(t, x)
    

    # --------------------------------------------------
    # Loss Methods
    # --------------------------------------------------

    def compat_loss(self, params, t, x):
        _, ux = self.calc_ux(params, t, x)
        _, real_ux = self.real_calc_ux(params, t, x)
        return jnp.mean((ux - real_ux)**2)
    
    # ------------------------------

    def pinns_pde_loss(self, params, t, x):
        pass

    def pinns_pde(self, key, params):
        num_traj = self.config.batch
        traj_len = self.config.traj_len
        
        t_pde, x_pde = self.sample_domain(key, num_traj * traj_len)
        return self.pinns_pde_loss(params, t_pde, x_pde), self.compat_loss(params, t_pde, x_pde)

    def pinns_bc(self, key, params):
        t_bc, x_bc = self.sample_domain(key, self.config.batch)
        t_bc = jnp.ones_like(t_bc)

        u, ux = self.calc_ux(params, t_bc, x_bc)
        bc_loss = jnp.mean((u - self.bc_fn(x_bc))**2) + jnp.mean((ux - self.calc_bcx(x_bc))**2)
        return bc_loss, self.compat_loss(params, t_bc, x_bc)

    def pinns_loss(self, key, params):
        pde_loss, comp_domain_loss = self.pinns_pde(key, params) 
        bc_loss, comp_bc_loss = self.pinns_bc(key, params) * self.config.bc_scale
        pde_loss = pde_loss * self.config.pde_scale
        bc_loss = bc_loss * self.config.bc_scale
        comp_loss = (comp_domain_loss + comp_bc_loss) * self.config.comp_scale
        return pde_loss + bc_loss + comp_loss, (pde_loss, bc_loss, comp_loss)

    def pinns_grad(self, key, params):
        (total, (pde_loss, bc_loss, comp_loss)), grad = jax.value_and_grad(self.pinns_loss, argnums=1, has_aux=True)(key, params)
        return (pde_loss, bc_loss, comp_loss), grad

    @partial(jax.jit, static_argnums=0)
    def jit_pinns_loss(self, key, params):
        return self.pinns_loss(key, params)[0]
    

    def pinns_pde_grad_batch(self, key, params):
        num_traj = self.config.batch
        micro_batch = self.config.micro_batch
        n_chunks = (num_traj + micro_batch - 1) // micro_batch
        traj_len = self.config.traj_len

        grad_make_pde_loss = jax.jit(jax.value_and_grad(lambda pp, tt, xx: jnp.sum(self.pinns_pde_loss(pp, tt, xx))/n_chunks * self.config.pde_scale, argnums=0))
        grad_make_comp_loss = jax.jit(jax.value_and_grad(lambda pp, tt, xx: jnp.sum(self.compat_loss(pp, tt, xx))/n_chunks * self.config.comp_scale, argnums=0))

        def chunk_loop(carry, _):
            key, pde_loss_acc, comp_loss_acc, grad_acc = carry
            t_pde, x_pde = self.sample_domain(key, micro_batch*traj_len)
            pde_loss, pde_grad = grad_make_pde_loss(params, t_pde, x_pde)
            comp_loss, comp_grad = grad_make_comp_loss(params, t_pde, x_pde)

            pde_loss_acc = pde_loss_acc + pde_loss
            comp_loss_acc = comp_loss_acc + comp_loss
            grad_acc = jax.tree_util.tree_map(lambda a, b, c: a+b+c, grad_acc, pde_grad, comp_grad)
            return (key, pde_loss_acc, comp_loss_acc, grad_acc), None

        (key, pde_loss, comp_loss, pde_grad), _ = jax.lax.scan(chunk_loop, (key, 0.0, 0.0, jax.tree_util.tree_map(jnp.zeros_like, params)), None, length=n_chunks)
        return (pde_loss, comp_loss), pde_grad
    
    def pinns_grad_batch(self, key, params):
        pde_loss, comp_domain_loss, pde_grad = self.pinns_pde_grad_batch(key, params) 
        bc_loss, comp_bc_loss, bc_grad = jax.value_and_grad(self.pinns_bc, argnums=1)(key, params) 
        grad = jax.tree_util.tree_map(lambda a, b: a+b, pde_grad, bc_grad)
        comp_loss = comp_domain_loss + comp_bc_loss
        return (pde_loss, bc_loss, comp_loss), grad
    
    # ------------------------------

    def fspinns_loss(self, key, params):
        num_traj = self.config.batch
        traj_len = self.config.traj_len
        dt = self.config.dt

        T = jnp.repeat(jnp.linspace(0, 1, traj_len+1)[jnp.newaxis, ..., jnp.newaxis], num_traj, axis=0)
        dW = jnp.sqrt(dt) * jnp.concatenate((jnp.zeros((num_traj, 1, self.config.d_in)),
                                             jax.random.normal(key.newkey(), (num_traj, traj_len, self.config.d_in))), axis=1)
        X = jnp.zeros((num_traj, traj_len+1, self.config.d_in))
        X = X.at[:, 0, :].set(self.get_X0(num_traj))

        def loop(i, X):
            yz_dict = {}
            if self.b_z or self.sigma_z:
                yz_dict['y'], yz_dict['z'] = self.calc_ux(params, T[:, i-1, :], X[:, i-1, :])
            elif self.b_y or self.sigma_y:
                yz_dict['y'] = self.calc_u(params, T[:, i-1, :], X[:, i-1, :])
            X = X.at[:, i, :].set(X[:, i-1, :] + self.b(T[:, i-1, :], X[:, i-1, :], **yz_dict)*dt + jnp.matmul(self.sigma(T[:, i-1, :], X[:, i-1, :], **yz_dict), dW[:, i, :, jnp.newaxis])[..., 0])
            return X
        
        X = jax.lax.fori_loop(1, self.config.traj_len+1, loop, X)
        pde_loss = self.pinns_pde_loss(params, T.reshape(-1, 1), X.reshape(-1, self.config.d_in))
        bc, bcx = self.calc_ux(params, T[:, -1, :], X[:, -1, :])
        bc_loss = (jnp.mean((bc - self.bc_fn(X[:, -1, :]))**2) + jnp.mean((bcx - self.calc_bcx(X[:, -1, :]))**2)) * self.config.bc_scale
        comp_loss = self.compat_loss(params, T.reshape(-1, 1), X.reshape(-1, self.config.d_in)) * self.config.comp_scale
        return pde_loss + bc_loss + comp_loss, (pde_loss, bc_loss, comp_loss)

    def fspinns_grad(self, key, params):
        (total, (pde_loss, bc_loss, comp_loss)), grad = jax.value_and_grad(self.fspinns_loss, argnums=1, has_aux=True)(key, params)
        return (pde_loss, bc_loss, comp_loss), grad

    @partial(jax.jit, static_argnums=0)
    def jit_fspinns_loss(self, key, params):
        return self.fspinns_loss(key, params)[0]
    

    def fspinns_grad_batch(self, key, params):
        num_traj = self.config.batch
        micro_batch = self.config.micro_batch
        n_chunks = (num_traj + micro_batch - 1) // micro_batch
        traj_len = self.config.traj_len
        dt = self.config.dt

        def make_loss(key, params):
            T = jnp.repeat(jnp.linspace(0, 1, traj_len+1)[jnp.newaxis, ..., jnp.newaxis], micro_batch, axis=0)
            dW = jnp.sqrt(dt) * jnp.concatenate((jnp.zeros((micro_batch, 1, self.config.d_in)),
                                                 jax.random.normal(key.newkey(), (micro_batch, traj_len, self.config.d_in))), axis=1)
            X = jnp.zeros((micro_batch, traj_len+1, self.config.d_in))
            X = X.at[:, 0, :].set(self.get_X0(micro_batch))

            def loop(i, X):
                yz_dict = {}
                if self.b_z or self.sigma_z:
                    yz_dict['y'], yz_dict['z'] = self.calc_ux(params, T[:, i-1, :], X[:, i-1, :])
                elif self.b_y or self.sigma_y:
                    yz_dict['y'] = self.calc_u(params, T[:, i-1, :], X[:, i-1, :])
                X = X.at[:, i, :].set(X[:, i-1, :] + self.b(T[:, i-1, :], X[:, i-1, :], **yz_dict)*dt + jnp.matmul(self.sigma(T[:, i-1, :], X[:, i-1, :], **yz_dict), dW[:, i, :, jnp.newaxis])[..., 0])
                return X
            
            X = jax.lax.fori_loop(1, traj_len+1, loop, X)
            pde_loss = self.pinns_pde_loss(params, T.reshape(-1, 1), X.reshape(-1, self.config.d_in)) * self.config.pde_scale
            bc, bcx = self.calc_ux(params, T[:, -1, :], X[:, -1, :])
            bc_loss = (jnp.mean((bc - self.bc_fn(X[:, -1, :]))**2) + jnp.mean((bcx - self.calc_bcx(X[:, -1, :]))**2)) * self.config.bc_scale
            comp_loss = self.compat_loss(params, T.reshape(-1, 1), X.reshape(-1, self.config.d_in)) * self.config.comp_scale
            return pde_loss + bc_loss + comp_loss, (pde_loss, bc_loss, comp_loss)

        grad_make_loss = jax.jit(jax.value_and_grad(make_loss, argnums=1, has_aux=True))

        def chunk_loop(carry, _):
            key, pde_loss_acc, bc_loss_acc, comp_loss_acc, grad_acc = carry
            (total, (pde_loss, bc_loss, comp_loss)), grad = grad_make_loss(key, params)
            pde_loss_acc = pde_loss_acc + pde_loss
            bc_loss_acc = bc_loss_acc + bc_loss
            comp_loss_acc = comp_loss_acc + comp_loss
            grad_acc = jax.tree_util.tree_map(lambda a, b: a+b, grad_acc, grad)
            return (key, pde_loss_acc, bc_loss_acc, comp_loss_acc, grad_acc), None

        (key, pde_loss, bc_loss, comp_loss, grad), _ = jax.lax.scan(chunk_loop, (key, 0.0, 0.0, 0.0, jax.tree_util.tree_map(jnp.zeros_like, params)), None, length=n_chunks)
        return (pde_loss, bc_loss, comp_loss), grad

    # --------------------------------------------------

    def bsde_loss(self, key, params):
        num_traj = self.config.batch
        traj_len = self.config.traj_len
        dt = self.config.dt
        
        t = jnp.zeros((num_traj, 1))
        x = self.get_X0(num_traj)
        u, ux = self.calc_ux(params, t, x)
        step_loss = jnp.zeros(traj_len)
        step_comp_loss = jnp.zeros(traj_len)

        def traj_calc(i, inputs):
            key, t, x, u, ux, step_loss, step_comp_loss = inputs
            _, real_ux = self.real_calc_ux(params, t, x)
            step_comp_loss = step_comp_loss.at[i].set(jnp.sum((ux - real_ux)**2))

            sigma = self.sigma(t, x, u, ux)
            dw = jnp.sqrt(dt) * jax.random.normal(key.newkey(), (num_traj, self.config.d_in))
            t_new = t + dt
            x_new = x + self.b(t, x, u, ux)*dt + jnp.matmul(sigma, dw[..., jnp.newaxis])[..., 0]
            u_new = u + self.h(t, x, u, ux)*dt + jnp.matmul(jnp.matmul(ux, sigma), dw[..., jnp.newaxis])[..., 0]

            u_calc, ux_calc = self.calc_ux(params, t_new, x_new)
            step_loss = step_loss.at[i].set(jnp.sum((u_new - u_calc)**2))

            u_next = u_calc if self.config.reset_u else u_new
            return key, t_new, x_new, u_next, ux_calc, step_loss, step_comp_loss

        key, t, x, u, ux, step_loss, step_comp_loss = jax.lax.fori_loop(0, traj_len, traj_calc, (key, t, x, u, ux, step_loss, step_comp_loss))
        pde_loss = jnp.sum(step_loss) * self.config.pde_scale 
        bc_loss = (jnp.sum((u - self.bc_fn(x))**2) + jnp.sum((ux - self.calc_bcx(x))**2)) * self.config.bc_scale
        bc_comp_loss = self.compat_loss(params, t, x)
        comp_loss = (jnp.sum(step_comp_loss) + bc_comp_loss) * self.config.comp_scale / traj_len
        return pde_loss + bc_loss + comp_loss, (pde_loss, bc_loss, comp_loss)
        
    def bsde_grad(self, key, params):
        (total, (pde_loss, bc_loss, comp_loss)), grad = jax.value_and_grad(self.bsde_loss, argnums=1, has_aux=True)(key, params)
        return (pde_loss, bc_loss, comp_loss), grad

    @partial(jax.jit, static_argnums=0)
    def jit_bsde_loss(self, key, params):
        return self.bsde_loss(key, params)[0]
    

    def bsde_grad_batch(self, key, params):
        num_traj = self.config.batch
        micro_batch = self.config.micro_batch
        n_chunks = (num_traj + micro_batch - 1) // micro_batch
        traj_len = self.config.traj_len
        dt = self.config.dt
        
        def make_loss(key, params):
            t = jnp.zeros((micro_batch, 1))
            x = self.get_X0(micro_batch)
            u, ux = self.calc_ux(params, t, x)
            step_loss = jnp.zeros(traj_len)
            step_comp_loss = jnp.zeros(traj_len)

            def traj_calc(i, inputs):
                key, t, x, u, ux, step_loss, step_comp_loss = inputs
                _, real_ux = self.real_calc_ux(params, t, x)
                step_comp_loss = step_comp_loss.at[i].set(jnp.sum((ux - real_ux)**2))

                sigma = self.sigma(t, x, u, ux)
                dw = jnp.sqrt(dt) * jax.random.normal(key.newkey(), (micro_batch, self.config.d_in))
                t_new = t + dt
                x_new = x + self.b(t, x, u, ux)*dt + jnp.matmul(sigma, dw[..., jnp.newaxis])[..., 0]
                u_new = u + self.h(t, x, u, ux)*dt + jnp.matmul(jnp.matmul(ux, sigma), dw[..., jnp.newaxis])[..., 0]

                u_calc, ux_calc = self.calc_ux(params, t_new, x_new)
                step_loss = step_loss.at[i].set(jnp.sum((u_new - u_calc)**2))

                u_next = u_calc if self.config.reset_u else u_new
                return key, t_new, x_new, u_next, ux_calc, step_loss, step_comp_loss

            key, t, x, u, ux, step_loss, step_comp_loss = jax.lax.fori_loop(0, traj_len, traj_calc, (key, t, x, u, ux, step_loss, step_comp_loss))
            pde_loss = jnp.sum(step_loss) * self.config.pde_scale
            bc_loss = (jnp.sum((u - self.bc_fn(x))**2) + jnp.sum((ux - self.calc_bcx(x))**2)) * self.config.bc_scale
            bc_comp_loss = self.compat_loss(params, t, x)
            comp_loss = (jnp.sum(step_comp_loss) + bc_comp_loss) * self.config.comp_scale / traj_len
            return pde_loss + bc_loss + comp_loss, (pde_loss, bc_loss, comp_loss)

        grad_make_loss = jax.jit(jax.value_and_grad(make_loss, argnums=1, has_aux=True))

        def chunk_loop(carry, _):
            key, pde_loss_acc, bc_loss_acc, comp_loss_acc, grad_acc = carry
            (total, (pde_loss, bc_loss, comp_loss)), grad = grad_make_loss(key, params)
            pde_loss_acc = pde_loss_acc + pde_loss
            bc_loss_acc = bc_loss_acc + bc_loss
            comp_loss_acc = comp_loss_acc + comp_loss
            grad_acc = jax.tree_util.tree_map(lambda a, b: a+b, grad_acc, grad)
            return (key, pde_loss_acc, bc_loss_acc, comp_loss_acc, grad_acc), None

        (key, pde_loss, bc_loss, comp_loss, grad), _ = jax.lax.scan(chunk_loop, (key, 0.0, 0.0, 0.0, jax.tree_util.tree_map(jnp.zeros_like, params)), None, length=n_chunks)
        return (pde_loss, bc_loss, comp_loss), grad

    # --------------------------------------------------

    def bsde_heun_loss(self, key, params):
        num_traj = self.config.batch
        traj_len = self.config.traj_len
        dt = self.config.dt
        
        calc_uxx = jax.checkpoint(self.calc_uxx) if self.config.checkpointing else self.calc_uxx

        t = jnp.zeros((num_traj, 1))
        x = self.get_X0(num_traj)
        u, ux, uxx = calc_uxx(params, t, x)
        step_loss = jnp.zeros(traj_len)
        step_comp_loss1 = jnp.zeros(traj_len)
        step_comp_loss2 = jnp.zeros(traj_len)
        
        def traj_calc(i, inputs):
            key, t, x, u, ux, uxx, step_loss, step_comp_loss1, step_comp_loss2 = inputs
            _, real_ux = self.real_calc_ux(params, t, x)
            step_comp_loss1 = step_comp_loss1.at[i].set(jnp.sum((ux - real_ux)**2))

            sigma = self.sigma(t, x, u, ux)
            dw = jnp.sqrt(dt) * jax.random.normal(key.newkey(), (num_traj, self.config.d_in))
            dx_int = self.b_heun(t, x, u, ux)*dt + jnp.matmul(sigma, dw[..., jnp.newaxis])[..., 0]
            x_int = x + dx_int
            du_int = (self.h(t, x, u, ux) - self.c(t, x, u, ux, uxx))*dt + jnp.matmul(jnp.matmul(ux, self.sigma(t, x, u, ux)), dw[..., jnp.newaxis])[..., 0]
            u_int = u + du_int

            t_new = t + dt
            _, ux_int, uxx_int = calc_uxx(params, t_new, x_int)
            _, real_ux_int = self.real_calc_ux(params, t_new, x_int)
            step_comp_loss2 = step_comp_loss2.at[i].set(jnp.sum((ux_int - real_ux_int)**2))
            
            sigma = self.sigma(t_new, x_int, u_int, ux_int)
            x_new = x + 0.5*dx_int + 0.5*(self.b_heun(t_new, x_int, u_int, ux_int)*dt + jnp.matmul(sigma, dw[..., jnp.newaxis])[..., 0])
            u_new = u + 0.5*du_int + 0.5*((self.h(t_new, x_int, u_int, ux_int) - self.c(t_new, x_int, u_int, ux_int, uxx_int))*dt + jnp.matmul(jnp.matmul(ux_int, sigma), dw[..., jnp.newaxis])[..., 0])
            
            u_calc, ux_calc, uxx_calc = calc_uxx(params, t_new, x_new)

            step_loss = step_loss.at[i].set(jnp.sum((u_new - u_calc)**2))
            return key, t_new, x_new, u_calc, ux_calc, uxx_calc, step_loss, step_comp_loss1, step_comp_loss2

        key, t, x, u, ux, uxx, step_loss, step_comp_loss1, step_comp_loss2 = jax.lax.fori_loop(0, traj_len, traj_calc, (key, t, x, u, ux, uxx, step_loss, step_comp_loss1, step_comp_loss2))
        pde_loss = jnp.sum(step_loss) * self.config.pde_scale
        bc_loss = (jnp.sum((u - self.bc_fn(x))**2) + jnp.sum((ux - self.calc_bcx(x))**2)) * self.config.bc_scale
        bc_comp_loss = self.compat_loss(params, t, x)
        comp_loss = (jnp.sum(step_comp_loss1) + jnp.sum(step_comp_loss2) + bc_comp_loss) * self.config.comp_scale / traj_len
        return pde_loss + bc_loss + comp_loss, (pde_loss, bc_loss, comp_loss)

    def bsde_heun_grad(self, key, params):
        (total, (pde_loss, bc_loss, comp_loss)), grad = jax.value_and_grad(self.bsde_heun_loss, argnums=1, has_aux=True)(key, params)
        return (pde_loss, bc_loss, comp_loss), grad

    @partial(jax.jit, static_argnums=0)
    def jit_bsde_heun_loss(self, key, params):
        return self.bsde_heun_loss(key, params)[0]
    

    def bsde_heun_grad_batch(self, key, params):
        num_traj = self.config.batch
        micro_batch = self.config.micro_batch
        n_chunks = (num_traj + micro_batch - 1) // micro_batch
        traj_len = self.config.traj_len
        dt = self.config.dt

        calc_uxx = jax.checkpoint(self.calc_uxx) if self.config.checkpointing else self.calc_uxx

        def make_loss(key, params):
            t = jnp.zeros((micro_batch, 1))
            x = self.get_X0(micro_batch)
            u, ux, uxx = calc_uxx(params, t, x)
            step_loss = jnp.zeros(traj_len)
            step_comp_loss1 = jnp.zeros(traj_len)
            step_comp_loss2 = jnp.zeros(traj_len)

            def traj_calc(i, inputs):
                key, t, x, u, ux, uxx, step_loss, step_comp_loss1, step_comp_loss2 = inputs
                _, real_ux = self.real_calc_ux(params, t, x)
                step_comp_loss1 = step_comp_loss1.at[i].set(jnp.sum((ux - real_ux)**2))

                sigma = self.sigma(t, x, u, ux)
                dw = jnp.sqrt(dt) * jax.random.normal(key.newkey(), (micro_batch, self.config.d_in))
                dx_int = self.b_heun(t, x, u, ux)*dt + jnp.matmul(sigma, dw[..., jnp.newaxis])[..., 0]
                x_int = x + dx_int
                du_int = (self.h(t, x, u, ux) - self.c(t, x, u, ux, uxx))*dt + jnp.matmul(jnp.matmul(ux, self.sigma(t, x, u, ux)), dw[..., jnp.newaxis])[..., 0]
                u_int = u + du_int

                t_new = t + dt
                _, ux_int, uxx_int = calc_uxx(params, t_new, x_int)
                _, real_ux_int = self.real_calc_ux(params, t_new, x_int)
                step_comp_loss2 = step_comp_loss2.at[i].set(jnp.sum((ux_int - real_ux_int)**2))
                
                sigma = self.sigma(t_new, x_int, u_int, ux_int)
                x_new = x + 0.5*dx_int + 0.5*(self.b_heun(t_new, x_int, u_int, ux_int)*dt + jnp.matmul(sigma, dw[..., jnp.newaxis])[..., 0])
                u_new = u + 0.5*du_int + 0.5*((self.h(t_new, x_int, u_int, ux_int) - self.c(t_new, x_int, u_int, ux_int, uxx_int))*dt + jnp.matmul(jnp.matmul(ux_int, sigma), dw[..., jnp.newaxis])[..., 0])
            
                u_calc, ux_calc, uxx_calc = calc_uxx(params, t_new, x_new)

                step_loss = step_loss.at[i].set(jnp.sum((u_new - u_calc)**2))
                return key, t_new, x_new, u_calc, ux_calc, uxx_calc, step_loss, step_comp_loss1, step_comp_loss2

            key, t, x, u, ux, uxx, step_loss, step_comp_loss1, step_comp_loss2 = jax.lax.fori_loop(0, traj_len, traj_calc, (key, t, x, u, ux, uxx, step_loss, step_comp_loss1, step_comp_loss2))
            pde_loss = jnp.sum(step_loss) * self.config.pde_scale
            bc_loss = (jnp.sum((u - self.bc_fn(x))**2) + jnp.sum((ux - self.calc_bcx(x))**2)) * self.config.bc_scale
            bc_comp_loss = self.compat_loss(params, t, x)
            comp_loss = (jnp.sum(step_comp_loss1) + jnp.sum(step_comp_loss2) + bc_comp_loss) * self.config.comp_scale / traj_len
            return pde_loss + bc_loss + comp_loss, (pde_loss, bc_loss, comp_loss)

        grad_make_loss = jax.jit(jax.value_and_grad(make_loss, argnums=1, has_aux=True))

        def chunk_loop(carry, _):
            key, pde_loss_acc, bc_loss_acc, comp_loss_acc, grad_acc = carry
            (total, (pde_loss, bc_loss, comp_loss)), grad = grad_make_loss(key, params)
            pde_loss_acc = pde_loss_acc + pde_loss
            bc_loss_acc = bc_loss_acc + bc_loss
            comp_loss_acc = comp_loss_acc + comp_loss
            grad_acc = jax.tree_util.tree_map(lambda a, b: a+b, grad_acc, grad)
            return (key, pde_loss_acc, bc_loss_acc, comp_loss_acc, grad_acc), None

        (key, pde_loss, bc_loss, comp_loss, grad), _ = jax.lax.scan(chunk_loop, (key, 0.0, 0.0, 0.0, jax.tree_util.tree_map(jnp.zeros_like, params)), None, length=n_chunks)
        return (pde_loss, bc_loss, comp_loss), grad


class PDE_FO_Controller(PDE_Controller):
    pass





# --------------------------------------------------
# Child class for PIDE
# --------------------------------------------------

class PIDE_Solver(Solver):

    def __init__(self, config: PIDE_Config):
        self.config = copy.deepcopy(config)

        self.loss_fn = self.select_loss(self.config.loss_method)
        self.model = self.create_model()
        self.optimizer = self.create_opt()
        if self.config.analytic_sol:
            self.sol_T, self.sol_X, self.sol_J, self.sol_U = self.get_analytic_sol()
        if self.config.custom_eval:
            self.eval_point = self.get_eval_point()
        if self.config.save_to_wandb:
            self.init_wandb()
    
    def get_base_config():
        return PIDE_Config()
    
    def select_loss(self, loss_method):
        if loss_method == 'bsde':
            return self.bsde_grad_batch if self.config.micro_batch < self.config.batch else self.bsde_grad
        elif loss_method == 'regress':
            return self.reg_grad  # Always full micro-batch
        else:
            raise Exception("Loss Method '" + loss_method + "' Not Implemented")


    # --------------------------------------------------
    # Calculation Methods
    # --------------------------------------------------

    def forward_diffusion(self, t, x, j):
        return jnp.sum(j, axis=-2)
    
    def forward_drift(self, t, x):
        return jnp.zeros_like(x)
    
    def backward_diffusion(self, params, t, x, y, z, j):
        def loop(i, acc):
            return acc + self.calc_u(params, t, x+j[..., i, :])
        return jax.lax.fori_loop(0, self.config.mu_n, loop, jnp.zeros_like(y)) - self.config.mu_n*y

    def backward_drift(self, t, x, y, z):
        return jnp.zeros_like(y)


    # --------------------------------------------------
    # Util Methods
    # --------------------------------------------------

    def get_analytic_X(self, T, W, J):
        X = jnp.zeros_like(W)
        X = X.at[:, 0, :].set(self.get_X0(X.shape[0]))
        dt = self.config.dt

        def loop(i, X):
            yz_dict = {}
            if self.b_z or self.sigma_z:
                yz_dict['y'], yz_dict['z'] = self.analytic_ux(T[:, i-1, :], X[:, i-1, :])
            elif self.b_y or self.sigma_y:
                yz_dict['y'] = self.analytic_u(T[:, i-1, :], X[:, i-1, :])
            
            X = X.at[:, i, :].set(X[:, i-1, :] + self.b(T[:, i-1, :], X[:, i-1, :], **yz_dict)*dt + jnp.matmul(self.sigma(T[:, i-1, :], X[:, i-1, :], **yz_dict), (W[:, i, :, jnp.newaxis] - W[:, i-1, :, jnp.newaxis]))[..., 0]
                                  + self.forward_diffusion(T[:, i-1, :], X[:, i-1, :], J[:, i-1, :]) + self.forward_drift(T[:, i-1, :], X[:, i-1, :])*dt)
            return X

        X = jax.lax.fori_loop(1, self.config.traj_len+1, loop, X)
        return X
    
    def get_analytic_sol(self):
        num_traj = 5
        traj_len = self.config.traj_len
        dt = self.config.dt
        
        T = jnp.repeat(jnp.linspace(0, 1, traj_len+1)[jnp.newaxis, ..., jnp.newaxis], num_traj, axis=0)
        dW = jnp.sqrt(dt) * jnp.concatenate((jnp.zeros((num_traj, 1, self.config.d_in)),
                                             jax.random.normal(jax.random.key(1), (num_traj, traj_len, self.config.d_in))), axis=1)
        W = jnp.cumsum(dW, axis=1)
        jump_occurence = (jax.random.uniform(jax.random.key(10), (num_traj, traj_len, self.config.mu_n, 1)) < self.config.lambda_ * dt).astype(float)
        jump_size = self.config.sigma_phi * jax.random.normal(jax.random.key(10), (num_traj, traj_len, self.config.mu_n, self.config.d_in)) + self.config.mu_phi
        J = jump_size * jump_occurence  # shape : (num_traj, traj_len, self.config.mu_n, self.config.d_in)
        
        X = self.get_analytic_X(T, W, J)
        U = jax.vmap(jax.vmap(self.analytic_u, in_axes=(0, 0)), in_axes=(0, 0))(T, X)
        return T, X, J, U
    
    def plot_pred(self, params, i):
        time = self.sol_T[..., 0].T
        pred = self.calc_u(params, self.sol_T, self.sol_X)[..., 0].T
        true = self.sol_U[..., 0].T
        jumps = self.sol_J
        jump_mask = jnp.any(jumps[:, :-1] != 0, axis=(-2, -1))
        L1 = jnp.abs(pred - true)

        fig_pred = plt.figure(figsize=(5, 4))
        plt.plot(time, pred, "r", linewidth=1)
        plt.plot(time, true, ":b", linewidth=1)
        for j in range(jump_mask.shape[0]):
            for k in range(jump_mask.shape[1]):
                if jump_mask[j, k]:
                    plt.plot(jnp.stack((time[k][j], time[k+1][j])), jnp.stack((pred[k][j], pred[k+1][j])), linewidth=2, color='purple')
        plt.plot(time[0, :], pred[0, :], 'ko', markersize=1)
        plt.plot(time[-1, :], pred[-1, :], 'ks', markersize=1)
        plt.title('Prediction') 
        plt.close(fig_pred)
        wandb.log({'Prediction': wandb.Image(fig_pred)}, step=i)

        fig_L1 = plt.figure(figsize=(5, 4))
        plt.plot(time, L1, "k", linewidth=1)
        plt.title('L1 Error')
        plt.yscale('log')
        plt.ylim(1e-5, 1e+1)
        plt.close(fig_L1)
        wandb.log({'L1 Error': wandb.Image(fig_L1)}, step=i)
    

    # --------------------------------------------------
    # Loss Methods
    # --------------------------------------------------

    def bsde_loss(self, key, params):
        num_traj = self.config.batch
        traj_len = self.config.traj_len
        dt = self.config.dt
        
        t = jnp.zeros((num_traj, 1))
        x = self.get_X0(num_traj)
        u, ux = self.calc_ux(params, t, x)
        step_loss = jnp.zeros(traj_len)

        def traj_calc(i, inputs):
            key, t, x, u, ux, step_loss = inputs

            jump_occurence = (jax.random.uniform(key.newkey(), (num_traj, self.config.mu_n, 1)) < self.config.lambda_ * dt).astype(float)
            jump_size = self.config.sigma_phi * jax.random.normal(key.newkey(), (num_traj, self.config.mu_n, self.config.d_in)) + self.config.mu_phi
            j = jump_size * jump_occurence

            sigma = self.sigma(t, x, u, ux)
            dw = jnp.sqrt(dt) * jax.random.normal(key.newkey(), (num_traj, self.config.d_in))
            t_new = t + dt
            x_new = x + self.b(t, x, u, ux)*dt + jnp.matmul(sigma, dw[..., jnp.newaxis])[..., 0] + self.forward_diffusion(t, x, j) + self.forward_drift(t, x)*dt
            u_new = u + self.h(t, x, u, ux)*dt + jnp.matmul(jnp.matmul(ux, sigma), dw[..., jnp.newaxis])[..., 0] + self.backward_diffusion(params, t, x, u, ux, j) + self.backward_drift(t, x, u, ux)*dt

            u_calc, ux_calc = self.calc_ux(params, t_new, x_new)
            step_loss = step_loss.at[i].set(jnp.sum((u_new - u_calc)**2))

            u_next = u_calc if self.config.reset_u else u_new
            return key, t_new, x_new, u_next, ux_calc, step_loss

        key, t, x, u, ux, step_loss = jax.lax.fori_loop(0, traj_len, traj_calc, (key, t, x, u, ux, step_loss))
        pde_loss = jnp.sum(step_loss) * self.config.pde_scale 
        bc_loss = (jnp.sum((u - self.bc_fn(x))**2) + jnp.sum((ux - self.calc_bcx(x))**2)) * self.config.bc_scale
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
        dt = self.config.dt
        
        def make_loss(key, params):
            t = jnp.zeros((micro_batch, 1))
            x = self.get_X0(micro_batch)
            u, ux = self.calc_ux(params, t, x)
            step_loss = jnp.zeros(traj_len)

            def traj_calc(i, inputs):
                key, t, x, u, ux, step_loss = inputs

                jump_occurence = (jax.random.uniform(key.newkey(), (micro_batch, self.config.mu_n, 1)) < self.lambda_ * dt).astype(float)
                jump_size = self.config.sigma_phi * jax.random.normal(key.newkey(), (micro_batch, self.config.mu_n, self.config.d_in)) + self.config.mu_phi
                j = jump_size * jump_occurence

                sigma = self.sigma(t, x, u, ux)
                dw = jnp.sqrt(dt) * jax.random.normal(key.newkey(), (micro_batch, self.config.d_in))
                t_new = t + dt
                x_new = x + self.b(t, x, u, ux)*dt + jnp.matmul(sigma, dw[..., jnp.newaxis])[..., 0] + self.forward_diffusion(t, x, j) + self.forward_drift(t, x)*dt
                u_new = u + self.h(t, x, u, ux)*dt + jnp.matmul(jnp.matmul(ux, sigma), dw[..., jnp.newaxis])[..., 0] + self.backward_diffusion(params, t, x, u, ux, j) + self.backward_drift(t, x, u, ux)*dt

                u_calc, ux_calc = self.calc_ux(params, t_new, x_new)
                step_loss = step_loss.at[i].set(jnp.sum((u_new - u_calc)**2))

                u_next = u_calc if self.config.reset_u else u_new
                return key, t_new, x_new, u_next, ux_calc, step_loss

            key, t, x, u, ux, step_loss = jax.lax.fori_loop(0, traj_len, traj_calc, (key, t, x, u, ux, step_loss))
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


class PIDE_Controller(Controller):

    def step(self, i):
        self.key.newkey()
        loss, losses, self.params, self.opt_state, self.key = self.solver.optimize(self.key, self.params, self.opt_state)

        if self.solver.config.save_to_wandb:
            wandb.log({'loss': loss, **{'loss'+str(k+1): v for k, v in dict(enumerate(losses)).items()}}, step=i)
            if self.solver.config.track_bsde_loss:
                bsde_loss = self.solver.jit_bsde_loss(self.key, self.params)
                wandb.log({"bsde Loss": bsde_loss}, step=i)
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