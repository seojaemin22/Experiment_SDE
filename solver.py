import jax
import flax.serialization as fs
from jax.experimental.jet import jet
import optax
from jax import numpy as jnp
import numpy as np
from flax import linen as nn
from flax.core import freeze
import tqdm
import wandb
import matplotlib.pyplot as plt
from model import *
from config import *
from utils import *
from functools import partial
import copy
from pathlib import Path

from flax.traverse_util import flatten_dict

class Solver():

    def __init__(self, model_config: Model_Config, solver_config: Solver_Config, problem_config: Problem_Config):
        self.model_config = model_config
        self.solver_config = solver_config
        self.problem_config = problem_config  
        
        self.bc_fn = get_boundary_function(self.model_config.bc_name)
        self.grad_setting()
        self.problem_setting()
        self.grad_fn = self.loss_to_grad(self.get_loss())
        self.optimizer = self.create_opt()
        
        if self.solver_config.analytic_traj_sol:
            self.sol_T, self.sol_X, self.sol_U = self.get_analytic_sol()
        if self.solver_config.custom_eval:
            self.eval_data = self.get_eval_data()
        if self.solver_config.save_to_wandb:
            self.init_wandb()

    def grad_setting(self):
        self.calc_u = jax.checkpoint(self._calc_u) if self.solver_config.checkpointing else self._calc_u
        self.calc_ut = jax.checkpoint(self._calc_ut) if self.solver_config.checkpointing else self._calc_ut
        self.calc_ux = jax.checkpoint(self._calc_ux) if self.solver_config.checkpointing else self._calc_ux
        self.calc_ut_ux = jax.checkpoint(self._calc_ut_ux) if self.solver_config.checkpointing else self._calc_ut_ux
        self.calc_uxx = jax.checkpoint(self._calc_uxx) if self.solver_config.checkpointing else self._calc_uxx

        if self.model_config.derivative == 'forward':
            self.calc_laplacian = jax.checkpoint(self._forward_laplacian) if self.solver_config.checkpointing else self._forward_laplacian
        elif self.model_config.derivative == 'backward':
            self.calc_laplacian = jax.checkpoint(self._calc_laplacian) if self.solver_config.checkpointing else self._calc_laplacian

    def problem_setting(self):
        problem_name = self.problem_config.problem_name

        self.default_domain = getattr(self, f'{problem_name}_default_domain')
        self.get_X0 = getattr(self, f'{problem_name}_get_X0')
        self.get_exact_X0 = getattr(self, f'{problem_name}_get_exact_X0')
        self.get_exact_Y0 = getattr(self, f'{problem_name}_get_exact_Y0')
        if self.solver_config.analytic_traj_sol:
            self.analytic_X = getattr(self, f'{problem_name}_analytic_X')
            self.analytic_u = getattr(self, f'{problem_name}_analytic_u')
        
        self.pinns_residual = getattr(self, f'{problem_name}_pinns_residual')
        self.b = getattr(self, f'{problem_name}_b')
        self.sigma = getattr(self, f'{problem_name}_sigma')
        self.h = getattr(self, f'{problem_name}_h')
        self.b_heun = getattr(self, f'{problem_name}_b_heun')  # b + Correction of Forward Stratonovich SDE (- 1/2 sigma sigma_x)

    def c(self, weighted_lap):  # Correction of Backward Stratonovich SDE
        return 0.5 * weighted_lap
    
    def get_loss(self):
        loss_method = self.solver_config.loss_method
        if loss_method == 'fspinns': return self.fspinns_loss
        elif loss_method == 'fbsnn': return self.fbsnn_loss
        elif loss_method == 'fbsnnheun': return self.fbsnnheun_loss
        elif loss_method == 'shotgun': return self.shotgun_loss
        elif loss_method == 'bsde': return self.bsde_loss
        elif loss_method == 'splitting': return self.splitting_loss
        else: raise Exception("Loss Method '" + loss_method + "' Not Implemented")


    def create_opt(self):
        if self.solver_config.schedule == 'piecewise_constant':
            schedule = optax.piecewise_constant_schedule(
                init_value=self.solver_config.lr,
                boundaries_and_scales=self.solver_config.boundaries_and_scales
            )
        elif self.solver_config.schedule == 'cosine_decay':
            schedule = optax.cosine_decay_schedule(
                init_value=self.solver_config.lr,
                decay_steps=self.solver_config.iter
            )
        elif self.solver_config.schedule == 'cosine_onecycle':
            schedule = optax.cosine_onecycle_schedule(
                transition_steps=self.solver_config.iter,
                peak_value=self.solver_config.lr
            )
        else: # No schedule
            schedule = optax.constant_schedule(
                value=self.solver_config.lr
            )
            
        if self.solver_config.optim == 'adam':
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
        
    def init_model(self, key: Key, model, t=None):
        if t is None:
            key, sub = key.split()
            t_pde = jax.random.uniform(sub, (self.solver_config.micro_batch, 1), minval=0, maxval=self.solver_config.T)
        elif isinstance(t, float) or isinstance(t, int):
            t_pde = jnp.full((self.solver_config.micro_batch, 1), t)
        elif isinstance(t, jnp.ndarray):
            t_pde = jnp.broadcast_to(t, (self.solver_config.micro_batch, 1))
        else:
            raise Exception("t should be None or float(int) or jnp.ndarray")

        key, x_pde = self.default_domain(key, t_pde)
        key, sub = key.split()
        if self.model_config.time_coupled:
            return key, model.init(sub, t_pde, x_pde)
        else:
            return key, model.init(sub, x_pde)

    def init_opt(self,params):
        return self.optimizer.init(params)
    
    def init_solver(self, key: Key):
        if self.model_config.time_coupled:
            self.model = get_model(self.model_config)
            key, params = self.init_model(key, self.model)
            opt_state = self.init_opt(params)
        else:  # time_decoupled
            if self.solver_config.loss_method == 'bsde':
                self.grad_model_config = copy.deepcopy(self.model_config)
                self.grad_model_config.d_out = self.grad_model_config.d_out * self.grad_model_config.d_in
                self.grad_model = get_model(self.grad_model_config)
                param_list = []
                for t in np.linspace(0, self.solver_config.T, self.solver_config.traj_len+1)[:-1]:
                    key, param = self.init_model(key, self.grad_model, t)
                    param_list.append(param)
                traj_params = self._tree_stack(param_list)
                
                self.model = get_model(self.model_config)
                key, sol_params = self.init_model(key, self.model, 0.0)
                
                params = {'traj': traj_params, 'sol': sol_params}
                opt_state = self.init_opt(params)

            if self.solver_config.loss_method == 'splitting':
                self.model = get_model(self.model_config)
                param_list = []
                for t in np.linspace(0, self.solver_config.T, self.solver_config.traj_len+1)[:-1]:
                    key, param = self.init_model(key, self.model, t)
                    param_list.append(param)
                params = self._tree_stack(param_list)  # jax.tree_util.tree_map(lambda *param: jnp.stack(param, axis=0), *param_list)

                opt_state_list = [self.init_opt(p) for p in param_list]
                opt_state = self._tree_stack(opt_state_list)

                self.index = self.solver_config.traj_len - 1

        model_path = Path(self.solver_config.model_state)
        if model_path.exists():
            model_bytes = model_path.read_bytes()
            params = fs.from_bytes(params, model_bytes)
        
        opt_path = Path(self.solver_config.opt_state)
        if opt_path.exists():
            opt_bytes = opt_path.read_bytes()
            opt_state = fs.from_bytes(opt_state, opt_bytes)

        num_params = sum(x.size for x in jax.tree_util.tree_leaves(params))
        if self.solver_config.save_to_wandb:
            wandb.config['# Params'] =  num_params
        return key, params, opt_state

    # --------------------------------------------------
    # Problem Setting
    # --------------------------------------------------

    def HJB_default_domain(self, key: Key, t_pde):
        key, sub = key.split()
        x_pde = jnp.sqrt(2) * jax.random.normal(sub, (*t_pde.shape[:-1], self.model_config.d_in))
        return key, x_pde
    
    def HJB_get_X0(self, key, batch):
        key, sub = key.split()
        X0 = jax.random.normal(sub, (batch, self.model_config.d_in)) * self.solver_config.X0_std
        return key, X0
    
    def HJB_get_exact_X0(self):
        return jnp.zeros(self.model_config.d_in)

    def HJB_get_exact_Y0(self):
        if self.model_config.bc_name == 'HJB_default':
            return self.HJB_analytic_u(jnp.zeros(1), self.HJB_get_exact_X0())
        if self.model_config.bc_name == 'HJB_splitting':
            if self.model_config.d_in == 100:
                return jnp.full((1), fill_value=3.74471)
            if self.model_config.d_in == 1000:
                return jnp.full((1), fill_value=6.68594)
            if self.model_config.d_in == 10000:
                return jnp.full((1), fill_value=11.87860)

    def HJB_analytic_X(self, T, W):
        return jnp.broadcast_to(self.HJB_get_exact_X0()[jnp.newaxis, jnp.newaxis, :], (T.shape[0], 1, self.model_config.d_in)) + jnp.sqrt(2.0)*W

    def HJB_analytic_u(self, t, x):
        w = jnp.sqrt(self.solver_config.T-t) * jax.random.normal(jax.random.key(10), (100000, self.model_config.d_in))
        return -jnp.log(jnp.mean(jnp.exp(-self.bc_fn(x + jnp.sqrt(2)*w)), axis=0))


    def HJB_b(self, t, x):
        return jnp.zeros_like(x)

    def HJB_sigma(self, t, x):
        return jnp.sqrt(2) * jnp.broadcast_to(jnp.eye(x.shape[-1]), (*x.shape[:-1], x.shape[-1], x.shape[-1]))
    
    def HJB_h(self, t, x, y, z):
        return jnp.sum(z**2, axis=-1)

    def HJB_b_heun(self, t, x):
        return jnp.zeros_like(x)
    
    def HJB_pinns_residual(self, model, params, t, x):
        _, ut, ux = self.calc_ut_ux(model, params, x, t)
        _, lap = self.calc_laplacian(model, params, x, t)
        loss = jnp.mean((ut[..., 0] + lap - jnp.sum(ux**2, axis=-1))**2)
        return loss
    
    # --------------------

    def BSB_default_domain(self, key: Key, t_pde):
        key, sub = key.split()
        x_pde = 0.75 + jax.random.normal(sub, (*t_pde.shape[:-1], self.model_config.d_in))
        return key, x_pde

    def BSB_get_X0(self, key, batch):
        key, sub = key.split()
        X0_list = []
        for i in range(self.model_config.d_in):
            X0_list.append(jnp.ones((batch, 1))/2 if i%2 == 1 else jnp.ones((batch, 1))) 
        X0 = jnp.concatenate(X0_list, axis=-1) + jax.random.normal(sub, (batch, self.model_config.d_in)) * self.solver_config.X0_std
        return key, X0


    def BSB_get_exact_X0(self):
        return jnp.concatenate([(jnp.ones(1) if i%2 == 0 else jnp.ones(1)/2) for i in range(self.model_config.d_in)])

    def BSB_get_exact_Y0(self):
        return self.BSB_analytic_u(jnp.zeros(1), self.BSB_get_exact_X0())
    
    def BSB_analytic_X(self, T, W):
        return jnp.broadcast_to(self.BSB_get_exact_X0()[jnp.newaxis, jnp.newaxis, :], (T.shape[0], 1, self.model_config.d_in)) * jnp.exp(0.4*W - 0.5*0.4**2*T)

    def BSB_analytic_u(self, t, x):
        return jnp.exp((0.05 + 0.4**2)*(self.solver_config.T - t)) * self.bc_fn(x)
    

    def BSB_b(self, t, x):
        return jnp.zeros_like(x)
    
    def BSB_sigma(self, t, x):
        return 0.4 * jax.vmap(jnp.diag, in_axes=0)(x)
    
    def BSB_h(self, t, x, y, z):
        return 0.05 * (y - jnp.matmul(z, x[..., jnp.newaxis])[..., 0])
    
    def BSB_b_heun(self, t, x):
        return -0.5 * 0.4**2 * x
    
    def BSB_pinns_residual(self, model, params, t, x):
        u, ut, ux = self.calc_ut_ux(model, params, x, t)
        _, weighted_lap = self.calc_laplacian(model, params, x, t, weight=self.BSB_sigma(t, x))
        loss = jnp.mean((ut[..., 0] + 0.5*weighted_lap - 0.05*(u - jnp.matmul(ux, x[..., jnp.newaxis])[..., 0]))**2)
        return loss

    # --------------------------------------------------
    # Calculation Methods
    # --------------------------------------------------
    
    def calc_bcx(self, x): 
        jax_x = jax.jacrev(lambda x: self.bc_fn(x), argnums=0)
        return jax.vmap(jax_x, in_axes=0)(x)

    # --------------------------------------------------

    def _calc_u(self, model, params, x, t=None):
        if t is not None:
            return model.apply(params, t, x)
        else:
            return model.apply(params, x)
    
    def _calc_ut(self, model, params, x, t):
        def u_ut(t, x):
            model_fn = lambda tt: self._calc_u(model, params, x, tt)
            u, du_dt = jax.vjp(model_fn, t)
            ut = jax.vmap(du_dt, in_axes=0)(jnp.eye(len(u)))[0]
            return u, ut
        return jax.vmap(u_ut, in_axes=(0, 0))(t, x)

    def _calc_ux(self, model, params, x, t=None):
        if t is not None:
            def u_ux(t, x):
                model_fn = lambda xx: self._calc_u(model, params, xx, t)
                u, du_dx = jax.vjp(model_fn, x)
                ux = jax.vmap(du_dx, in_axes=0)(jnp.eye(len(u)))[0]
                return u, ux
            return jax.vmap(u_ux, in_axes=(0, 0))(t, x)
        else: 
            def u_ux(x):
                model_fn = lambda xx: self._calc_u(model, params, xx[jnp.newaxis, :])[0]
                u, du_dx = jax.vjp(model_fn, x)
                ux = jax.vmap(du_dx, in_axes=0)(jnp.eye(len(u)))[0]
                return u, ux
            return jax.vmap(u_ux, in_axes=0)(x)

    def _calc_ut_ux(self, model, params, x, t):
        model_fn = lambda ttxx: self._calc_u(model, params, ttxx[..., 1:], ttxx[..., :1])
        def u_ut_ux(tx):
            u, du_dx_dt = jax.vjp(model_fn, tx)
            ux_ut = jax.vmap(du_dx_dt, in_axes=0)(jnp.eye(len(u)))[0]
            return u, ux_ut[..., :1], ux_ut[..., 1:]
        return jax.vmap(u_ut_ux, in_axes=0)(jnp.concatenate((t, x), axis=-1))

    def _calc_uxx(self, model, params, x, t=None):
        if t is not None:
            def u_ux_uxx(t, x):
                model_fn = lambda xx: self._calc_u(model, params, xx, t)
                def ux_u(x):
                    u, du_dx = jax.vjp(model_fn, x)
                    ux = jax.vmap(du_dx, in_axes=0)(jnp.eye(len(u)))[0]
                    return ux, u
                du_dxx = lambda s: jax.jvp(ux_u, (x,), (s,), has_aux=True)
                ux, uxx, u = jax.vmap(du_dxx, in_axes=1, out_axes=(None, 1, None))(jnp.eye(len(x)))
                return u, ux, uxx
            return jax.vmap(u_ux_uxx, in_axes=(0, 0))(t, x)
        else:
            def u_ux_uxx(x):
                model_fn = lambda xx: self._calc_u(model, params, xx)
                def ux_u(x):
                    u, du_dx = jax.vjp(model_fn, x)
                    ux = jax.vmap(du_dx, in_axes=0)(jnp.eye(len(u)))[0]
                    return ux, u
                du_dxx = lambda s: jax.jvp(ux_u, (x,), (s,), has_aux=True)
                ux, uxx, u = jax.vmap(du_dxx, in_axes=1, out_axes=(None, 1, None))(jnp.eye(len(x)))
                return u, ux, uxx
            return jax.vmap(u_ux_uxx, in_axes=0)(x)

    def _calc_laplacian(self, model, params, x, t=None, weight=None):
        weight = jnp.broadcast_to(weight if weight is not None else jnp.eye(self.model_config.d_in),
                                  shape=(x.shape[0], self.model_config.d_in, self.model_config.d_in))
        H = jnp.einsum('bij,bkj->bik', weight, weight)  # H = weight weight^T
        if t is not None:
            u, _, uxx = self._calc_uxx(model, params, x, t)
        else:
            u, _, uxx = self._calc_uxx(model, params, x)
        return u, jnp.einsum('bmij,bij->bm', uxx, H)  # Tr(H uxx)
    
    def _forward_laplacian(self, model, params, x, t=None, weight=None):
        if t is not None:
            return model.forward_laplacian(params, t, x, weight=weight)
        else:
            return model.forward_laplacian(params, x, weight=weight)
    
    # --------------------------------------------------

    def bn_apply_train(self, model, params, x, t=None):
        if t is not None:
            (y, mutated) = model.apply(params, t, x, use_running_average=False, mutable=['batch_stats'])
        else:
            (y, mutated) = model.apply(params, x, use_running_average=False, mutable=['batch_stats'])

        new_params = {'params': params['params'], 'batch_stats': mutated['batch_stats']}
        return y, new_params
    

    def clamp_bn_stats(self, params, min_var=1e-6, max_abs_mean=None, max_abs_var=None):
        # params: {'params': ..., 'batch_stats': ...}
        def finite(x):
            return jnp.where(jnp.isfinite(x), x, 0.0)

        bs = params['batch_stats'].copy()

        # For MLP
        m = finite(bs['input_batch_norm']['mean'])
        v = finite(bs['input_batch_norm']['var'])
        v = jnp.maximum(v, min_var)
        if max_abs_mean is not None: m = jnp.clip(m, -max_abs_mean, max_abs_mean)
        if max_abs_var  is not None: v = jnp.clip(v, 0.0, max_abs_var)
        bs['input_batch_norm']['mean'] = m
        bs['input_batch_norm']['var']  = v

        for i in range(self.model_config.MLP_num_layers):
            key = f'hidden_batch_norms_{i}'
            m = finite(bs[key]['mean']); v = finite(bs[key]['var'])
            v = jnp.maximum(v, min_var)
            if max_abs_mean is not None: m = jnp.clip(m, -max_abs_mean, max_abs_mean)
            if max_abs_var  is not None: v = jnp.clip(v, 0.0, max_abs_var)
            bs[key]['mean'] = m; bs[key]['var'] = v

        # 출력 BN
        m = finite(bs['output_batch_norm']['mean'])
        v = finite(bs['output_batch_norm']['var'])
        v = jnp.maximum(v, min_var)
        if max_abs_mean is not None: m = jnp.clip(m, -max_abs_mean, max_abs_mean)
        if max_abs_var  is not None: v = jnp.clip(v, 0.0, max_abs_var)
        bs['output_batch_norm']['mean'] = m
        bs['output_batch_norm']['var']  = v

        return {'params': params['params'], 'batch_stats': bs}

    # --------------------------------------------------

    def analytic_ut(self, t, x):
        def u_ut(t, x):
            model_fn = lambda tt: self.analytic_u(tt, x)
            u, du_dt = jax.vjp(model_fn, t)
            ut = jax.vmap(du_dt, in_axes=0)(jnp.eye(len(u)))[0]
            return u, ut
        return jax.vmap(u_ut, in_axes=(0, 0))(t, x)
    
    def analytic_ux(self, t, x):
        def u_ux(t, x):
            model_fn = lambda xx: self.analytic_u(t, xx)
            u, vjp_fun = jax.vjp(model_fn, x)
            ux = jax.vmap(vjp_fun, in_axes=0)(jnp.eye(len(u)))[0]
            return u, ux
        return jax.vmap(u_ux, in_axes=(0, 0))(t, x)
    
    def analytic_ut_ux(self, t, x):
        model_fn = lambda ttxx: self.analytic_u(ttxx[:1], ttxx[1:])
        def u_ut_ux(tx):
            u, du_dx_dt = jax.vjp(model_fn, tx)
            ux_ut = jax.vmap(du_dx_dt, in_axes=0)(jnp.eye(len(u)))[0]
            return u, ux_ut[:1], ux_ut[1:]
        return jax.vmap(u_ut_ux, in_axes=0)(jnp.concatenate((t, x), axis=-1))
    
    def analytic_uxx(self, t, x):
        def u_ux_uxx(t, x):
            model_fn = lambda xx: self.analytic_u(t, xx)
            def ux_x(x):
                u, du_dx = jax.vjp(model_fn, x)
                ux = jax.vmap(du_dx, in_axes=0)(jnp.eye(len(u)))[0]
                return ux, u
            du_dxx = lambda s: jax.jvp(ux_x, (x,), (s,), has_aux=True)
            ux, uxx, u = jax.vmap(du_dxx, in_axes=1, out_axes=(None, 1 ,None))(jnp.eye(len(x)))
            return u, ux, uxx
        return jax.vmap(u_ux_uxx, in_axes=(0, 0))(t, x)

    def get_analytic_sol(self):
        num_traj = 128
        test_dt = self.solver_config.T / self.solver_config.test_traj_len
        T = jnp.broadcast_to(jnp.linspace(0, self.solver_config.T, self.solver_config.test_traj_len+1)[None, :, None], (num_traj, self.solver_config.test_traj_len+1, 1))
        dW = jnp.sqrt(test_dt) * jnp.concatenate((jnp.zeros((num_traj, 1, self.model_config.d_in)),                                    
                                                  jax.random.normal(jax.random.key(1), (num_traj, self.solver_config.test_traj_len, self.model_config.d_in))), axis=1)
        W = jnp.cumsum(dW, axis=1)

        X = self.analytic_X(T, W)
        U = jax.lax.scan(lambda _, tx1: (None, jax.lax.scan(lambda _, tx2: (None, self.analytic_u(tx2[0], tx2[1])), None, (tx1[0], tx1[1]))[1]), None, (T, X))[1]
        return T, X, U

    # --------------------------------------------------
    # Loss Methods  [ time-coupled model ]
    # --------------------------------------------------

    def loss_to_grad(self, loss_fn):
        if self.solver_config.loss_method == 'splitting':
            def _loss_and_grad(key, params, index):
                (total, (losses, key, params)), grad = jax.value_and_grad(lambda K, P: loss_fn(K, P, index), argnums=1, has_aux=True)(key, params)
                return (total, (losses, key, params)), grad
            
            def grad_fn(key, params, index):
                n_chunks = (self.solver_config.batch + self.solver_config.micro_batch - 1) // self.solver_config.micro_batch

                def chunk_loop(carry, _):
                    key, params, losses_acc, grad_acc = carry
                    (total, (losses, key, params)), grad = _loss_and_grad(key, params, index)
                    losses_vec = jnp.asarray(losses)
                    losses_acc = losses_acc + losses_vec
                    grad_acc = jax.tree_util.tree_map(lambda a, b: a+b, grad_acc, grad)
                    return (key, params, losses_acc, grad_acc), None
                
                losses_0 = jnp.zeros((1,), dtype=jnp.result_type(0.0))
                grad_0 = jax.tree_util.tree_map(jnp.zeros_like, params)
                (key, params, losses, grad), _ = jax.lax.scan(chunk_loop, (key, params, losses_0, grad_0), None, length=n_chunks)
                return key, params, losses, grad
            return grad_fn
        else:
            def _loss_and_grad(key, params):
                (total, (losses, key, params)), grad = jax.value_and_grad(lambda K, P: loss_fn(K, P), argnums=1, has_aux=True)(key, params)
                return (total, (losses, key, params)), grad 
            
            def grad_fn(key, params):
                n_chunks = (self.solver_config.batch + self.solver_config.micro_batch - 1) // self.solver_config.micro_batch

                def chunk_loop(carry, _):
                    key, params, losses_acc, grad_acc = carry
                    (total, (losses, key, params)), grad = _loss_and_grad(key, params)
                    losses_vec = jnp.asarray(losses)
                    losses_acc = losses_acc + losses_vec
                    grad_acc = jax.tree_util.tree_map(lambda a, b: a+b, grad_acc, grad)
                    return (key, params, losses_acc, grad_acc), None
                
                losses_0 = jnp.zeros((len(loss_fn(key, params)[1]),), dtype=jnp.result_type(0.0))
                grad_0 = jax.tree_util.tree_map(jnp.zeros_like, params)
                (key, params, losses, grad), _ = jax.lax.scan(chunk_loop, (key, params, losses_0, grad_0), None, length=n_chunks)
                return key, params, losses, grad
            return grad_fn
        
    # --------------------------------------------------

    def make_time_domain(self, key: Key, batch: int):
        if self.solver_config.use_delta:
            dt = self.solver_config.T / (self.solver_config.traj_len - 1)
            key, sub = key.split()
            t1 = jax.random.uniform(sub, (batch, 1), minval=0, maxval=dt)
            diffT = jnp.concatenate([jnp.zeros((batch, 1)),
                                     t1,
                                     jnp.full((batch, self.solver_config.traj_len-2), fill_value=dt), 
                                     jnp.full_like(t1, fill_value=dt) - t1], axis=-1)
            T = jnp.cumsum(diffT, axis=-1)[:, :, None]
        else:
            dt = self.solver_config.T / self.solver_config.traj_len
            T = jnp.broadcast_to(jnp.linspace(0, self.solver_config.T, self.solver_config.traj_len+1)[None, :, None], (batch, self.solver_config.traj_len+1, 1))

        dT = T[:, 1:, :] - T[:, :-1, :]
        return key, T, dT
    
    def make_full_domain_euler(self, key: Key, batch: int):
        key, T, dT = self.make_time_domain(key, batch)
        
        key, sub = key.split()
        dW = jnp.sqrt(dT) * jax.random.normal(sub, (batch, self.solver_config.traj_len, self.model_config.d_in))

        X = jnp.zeros((batch, self.solver_config.traj_len+1, self.model_config.d_in))
        key, X0 = self.get_X0(key, batch)
        X = X.at[:, 0, :].set(X0)

        def loop(i, X):
            X = X.at[:, i, :].set(X[:, i-1, :] + self.b(T[:, i-1, :], X[:, i-1, :])*dT[:, i-1, :]
                                + jnp.matmul(self.sigma(T[:, i-1, :], X[:, i-1, :]), dW[:, i-1, :, jnp.newaxis])[..., 0])
            return X
        X = jax.lax.fori_loop(1, self.solver_config.traj_len+1, loop, X)
        return key, T, dT, dW, X
    
    def make_full_domain_heun(self, key: Key, batch: int):
        key, T, dT = self.make_time_domain(key, batch)
        
        key, sub = key.split()
        dW = jnp.sqrt(dT) * jax.random.normal(sub, (batch, self.solver_config.traj_len, self.model_config.d_in))

        X = jnp.zeros((batch, self.solver_config.traj_len+1, self.model_config.d_in))
        key, X0 = self.get_X0(key, batch)
        X = X.at[:, 0, :].set(X0)

        X_star = jnp.zeros((batch, self.solver_config.traj_len, self.model_config.d_in))

        def loop(i, inputs):
            X, X_star = inputs
            
            dX1 = self.b_heun(T[:, i-1, :], X[:, i-1, :])*dT[:, i-1, :] + jnp.matmul(self.sigma(T[:, i-1, :], X[:, i-1, :]), dW[:, i-1, :, jnp.newaxis])[..., 0]
            X_star = X_star.at[:, i-1, :].set(X[:, i-1, :] + dX1) 
            X = X.at[:, i, :].set(X[:, i-1, :] + 0.5 * (dX1 + (self.b_heun(T[:, i, :], X_star[:, i-1, :])*dT[:, i-1, :] 
                                                               + jnp.matmul(self.sigma(T[:, i, :], X_star[:, i-1, :]), dW[:, i-1, :, jnp.newaxis])[...,0])))
            return X, X_star
        X, X_star = jax.lax.fori_loop(1, self.solver_config.traj_len+1, loop, (X, X_star))
        return key, T, dT, dW, X, X_star

    # --------------------------------------------------
    
    def fspinns_loss(self, key, params):
        batch = self.solver_config.micro_batch
        
        key, T, dT, dW, X = self.make_full_domain_euler(key, batch)
        if self.model_config.use_batch_norm:
            _, params = self.bn_apply_train(self.model, params, X.reshape(-1, self.model_config.d_in), T.reshape(-1, 1))

        pde_loss = self.solver_config.pde_scale * self.pinns_residual(self.model, params, T.reshape(-1, 1), X.reshape(-1, self.model_config.d_in))
        if self.model_config.use_hard_constraint:
            return pde_loss, ((pde_loss,), key)
        else:
            u, ux = self.calc_ux(self.model, params, X[:, -1, :], T[:, -1, :])
            bc_loss = self.model_config.bc_scale * (jnp.mean((u - self.bc_fn(X[:, -1, :]))**2) + jnp.mean((ux - self.calc_bcx(X[:, -1, :]))**2))
            return pde_loss + bc_loss, ((pde_loss, bc_loss), key, params)
    
    # --------------------------------------------------

    def fbsnn_loss(self, key, params):
        batch = self.solver_config.micro_batch
        
        key, T, dT, dW, X = self.make_full_domain_euler(key, batch)
        if self.model_config.use_batch_norm:
            _, params = self.bn_apply_train(self.model, params, X.reshape(-1, self.model_config.d_in), T.reshape(-1, 1))

        u_start, ux_start = self.calc_ux(self.model, params, X[:, 0, :], T[:, 0, :])
        traj_loss = jnp.zeros(self.solver_config.traj_len)

        def traj_calc(i, inputs):
            key, u, ux, traj_loss = inputs

            t = T[:, i, :]
            dt = dT[:, i, :]
            x = X[:, i, :]
            dw = dW[:, i, :]
            sigma = self.sigma(t, x)

            t_new = T[:, i+1, :]  # t + dt
            x_new = X[:, i+1, :]  # x + self.b(t, x)*dt + jnp.matmul(sigma, dw[..., jnp.newaxis])[..., 0]
            u_new = u + self.h(t, x, u, ux)*dt + jnp.matmul(jnp.matmul(ux, sigma), dw[..., jnp.newaxis])[..., 0]

            u_calc, ux_calc = self.calc_ux(self.model, params, x_new, t_new)
            traj_loss = traj_loss.at[i].set(jnp.mean((u_new - u_calc)**2))
            # traj_loss = traj_loss.at[i].set(jnp.mean(((u_new - u_calc)/dt)**2))

            return key, u_calc, ux_calc, traj_loss

        key, u_end, ux_end, traj_loss = jax.lax.fori_loop(0, self.solver_config.traj_len, traj_calc, (key, u_start, ux_start, traj_loss))
        pde_loss = self.solver_config.pde_scale * jnp.sum(traj_loss)
        if self.model_config.use_hard_constraint:
            return pde_loss, ((pde_loss,), key, params)
        else:
            bc_loss = self.model_config.bc_scale * (jnp.mean((u_end - self.bc_fn(X[:, -1, :]))**2) + jnp.mean((ux_end - self.calc_bcx(X[:, -1, :]))**2))
            return pde_loss + bc_loss, ((pde_loss, bc_loss), key, params)

    # --------------------------------------------------   

    def fbsnnheun_loss(self, key, params):
        batch = self.solver_config.micro_batch

        key, T, dT, dW, X, X_star = self.make_full_domain_heun(key, batch)
        if self.model_config.use_batch_norm:
            _, params = self.bn_apply_train(self.model, params, 
                                            jnp.concatenate((X.reshape(-1, self.model_config.d_in), X_star.reshape(-1, self.model_config.d_in)), axis=0),
                                            jnp.concatenate((T.reshape(-1, 1), T[1:].reshape(-1, 1)), axis=0))

        u_start, ux_start = self.calc_ux(self.model, params, X[:, 0, :], T[:, 0, :])
        traj_loss = jnp.zeros(self.solver_config.traj_len)

        def traj_calc(i, inputs):
            key, u, ux, traj_loss = inputs

            t = T[:, i, :]
            dt = dT[:, i, :]
            x = X[:, i, :]
            dw = dW[:, i, :]
            sigma = self.sigma(t, x)
            weighted_lap = self.calc_laplacian(self.model, params, x, t, weight=sigma)[1]
            
            x_star = X_star[:, i, :]  # x + self.b_heun(t, x)*dt + jnp.matmul(sigma, dw[..., jnp.newaxis])[..., 0]  (x + dx_star)
            du_star = (self.h(t, x, u, ux) - self.c(weighted_lap))*dt + jnp.matmul(jnp.matmul(ux, sigma), dw[..., jnp.newaxis])[..., 0]
            u_star = u + du_star

            t_new = T[:, i+1, :]  # t + dt
            _, ux_star = self.calc_ux(self.model, params, x_star, t_new)
            sigma_star = self.sigma(t_new, x_star)
            weighted_lap_star = self.calc_laplacian(self.model, params, x_star, t_new, weight=sigma_star)[1]

            x_new = X[:, i+1, :]  # x + 0.5*dx_star + 0.5*(self.b_heun(t_new, x_star)*dt + jnp.matmul(sigma_star, dw[..., jnp.newaxis])[..., 0])
            u_new = u + 0.5*du_star + 0.5*((self.h(t_new, x_star, u_star, ux_star) - self.c(weighted_lap_star))*dt + jnp.matmul(jnp.matmul(ux_star, sigma_star), dw[..., jnp.newaxis])[..., 0])

            u_calc, ux_calc = self.calc_ux(self.model, params, x_new, t_new)
            traj_loss = traj_loss.at[i].set(jnp.mean((u_new - u_calc)**2))
            # traj_loss = traj_loss.at[i].set(jnp.mean(((u_new - u_calc)/dt)**2))

            return key, u_calc, ux_calc, traj_loss

        key, u_end, ux_end, traj_loss = jax.lax.fori_loop(0, self.solver_config.traj_len, traj_calc, (key, u_start, ux_start, traj_loss))
        pde_loss = self.solver_config.pde_scale * jnp.sum(traj_loss)
        if self.model_config.use_hard_constraint:
            return pde_loss, ((pde_loss,), key, params)
        else:
            bc_loss = self.model_config.bc_scale * (jnp.mean((u_end - self.bc_fn(X[:, -1, :]))**2) + jnp.mean((ux_end - self.calc_bcx(X[:, -1, :]))**2))
            return pde_loss + bc_loss, ((pde_loss, bc_loss), key, params)
    
    # --------------------------------------------------

    def shotgun_loss(self, key, params):
        batch = self.solver_config.micro_batch
        
        key, T, dT, dW, X = self.make_full_domain_euler(key, batch)

        u_start, ux_start = self.calc_ux(self.model, params, X[:, 0, :], T[:, 0, :])
        step_loss = jnp.zeros(self.solver_config.traj_len+1)
        traj_loss = jnp.zeros(self.solver_config.traj_len+1)
        
        traj_len = self.solver_config.traj_len
        Delta_t = self.solver_config.shotgun_Delta_t
        local_batch = self.solver_config.shotgun_local_batch
        d_in = self.model_config.d_in

        # T.shape : (batch, traj_len+1, 1)
        # X.shape : (batch, traj_len+1, d_in)
        local_T = jnp.broadcast_to((T[:, :-1, :] + Delta_t)[:, :, jnp.newaxis, :],
                                   (batch, traj_len, local_batch, 1))
        
        local_X = jnp.broadcast_to((X[:, :-1, :] + self.b(T[:, :-1, :].reshape(-1, 1), X[:, :-1, :].reshape(-1, d_in)).reshape(batch, traj_len, d_in)*Delta_t)[:, :, jnp.newaxis, :],
                                   (batch, traj_len, local_batch, d_in))

        sigma = self.sigma(T[:, :-1, :].reshape(-1, 1), X[:, :-1, :].reshape(-1, d_in)).reshape(batch, traj_len, d_in, d_in)
        key, sub = key.split()
        etas = jnp.sqrt(Delta_t) * jax.random.normal(sub, (batch, traj_len, local_batch, d_in))
        diffs = jnp.einsum('btij,btkj->btki', sigma, etas)

        local_X_plus = local_X + diffs
        local_X_minus = local_X - diffs

        if self.model_config.use_batch_norm:
            _, params = self.bn_apply_train(self.model, params, 
                                            jnp.concatenate((T.reshape(-1, 1), local_T.reshape(-1, 1), local_T.reshape(-1, 1)), axis=0),
                                            jnp.concatenate((X.reshape(-1, 1), local_X_plus.reshape(-1, 1), local_X_minus), axis=0))
        
        def traj_calc(i, inputs):
            key, u, ux, step_loss, traj_loss = inputs

            t = T[:, i, :]
            dt = dT[:, i, :]
            x = X[:, i, :]

            sigma = self.sigma(t, x)
            
            u_local = jnp.broadcast_to(u[:, jnp.newaxis, :], (batch, local_batch, 1))
            h_local = jnp.broadcast_to(self.h(t, x, u, ux)[:, jnp.newaxis, :], (batch, local_batch, 1))

            t_local = local_T[:, i, :, :]
            x_plus = local_X_plus[:, i, :, :]
            x_minus = local_X_minus[:, i, :, :]
            
            u_plus = self.calc_u(self.model, params, x_plus.reshape(-1, d_in), t_local.reshape(-1, 1))
            u_minus = self.calc_u(self.model, params, x_minus.reshape(-1, d_in), t_local.reshape(-1, 1))
            step_loss = step_loss.at[i].set(jnp.mean(((u_plus + u_minus - 2*u_local.reshape(-1, 1))/(2*self.solver_config.shotgun_Delta_t) - h_local.reshape(-1, 1))**2))
            
            dw = dW[:, i, :]

            t_new = T[:, i+1, :]  # t + dt 
            x_new = X[:, i+1, :]  # x + self.b(t, x)*dt + jnp.matmul(sigma, dw[..., jnp.newaxis])[..., 0]
            u_new = u + self.h(t, x, u, ux)*dt + jnp.matmul(jnp.matmul(ux, sigma), dw[..., jnp.newaxis])[..., 0]

            u_calc, ux_calc = self.calc_ux(self.model, params, x_new, t_new)
            if self.solver_config.shotgun_use_traj_loss:
                traj_loss = traj_loss.at[i].set(jnp.mean((u_new - u_calc)**2))
                
            return key, u_calc, ux_calc, step_loss, traj_loss  # u_calc

        key, u_end, ux_end, step_loss, traj_loss = jax.lax.fori_loop(0, self.solver_config.traj_len, traj_calc, (key, u_start, ux_start, step_loss, traj_loss))
        pde_loss = self.solver_config.pde_scale * (jnp.mean(step_loss) + jnp.mean(traj_loss))
        if self.model_config.use_hard_constraint:
            return pde_loss, ((pde_loss,), key, params)
        else:
            bc_loss = self.model_config.bc_scale * (jnp.mean((u_end - self.bc_fn(X[:, -1, :]))**2) + jnp.mean((ux_end - self.calc_bcx(X[:, -1, :]))**2))
            return pde_loss + bc_loss, ((pde_loss, bc_loss), key, params)

    # --------------------------------------------------
    # Loss Methods  [ time-decoupled model ]
    # --------------------------------------------------

    def _tree_stack(self, list_of_pytrees):
        return jax.tree_util.tree_map(lambda *leaves: jnp.stack(leaves, axis=0), *list_of_pytrees)

    def _tree_index(self, stacked_tree, i):
        return jax.tree_util.tree_map(lambda a: jax.lax.dynamic_index_in_dim(a, i, axis=0, keepdims=False), stacked_tree)

    def _tree_update(self, stacked_tree, i, slice_tree):
        def _upd(a, s):
            if hasattr(s, 'dtype'):
                s = s.astype(a.dtype)
            return jax.lax.dynamic_update_slice_in_dim(a, jnp.expand_dims(s, 0), i, axis=0)
        return jax.tree_util.tree_map(_upd, stacked_tree, slice_tree)
        # return jax.tree_util.tree_map(lambda a, s: jax.lax.dynamic_update_slice_in_dim(a, jnp.expand_dims(s, 0), i, axis=0), stacked_tree, slice_tree)
    
    # --------------------

    def bsde_loss(self, key, params):
        batch = self.solver_config.micro_batch

        key, T, dT, dW, X = self.make_full_domain_euler(key, batch)
        if self.model_config.use_batch_norm:
            def loop(i, params):
                params_i = self._tree_index(params['traj'], i)
                _, params_i = self.bn_apply_train(self.model, params_i, X[:, i, :])
                params = self._tree_update(params, i, params_i)
                return params
            params = jax.lax.fori_loop(0, self.solver_config.traj_len, loop, params)

        u_start = self.calc_u(self.model, params['sol'], X[:, 0, :])

        def traj_calc(i, inputs):
            key, u = inputs

            t = T[:, i, :]
            dt = dT[:, i, :]
            x = X[:, i, :]

            key, sub = key.split()
            dw = jnp.sqrt(dt) * jax.random.normal(sub, (batch, self.model_config.d_in))
            sigma = self.sigma(t, x)

            ux = self.calc_u(self.grad_model, self._tree_index(params['traj'], i), x).reshape(batch, self.model_config.d_out, self.model_config.d_in)
            u_new = u + self.h(t, x, u, ux)*dt + jnp.matmul(jnp.matmul(ux, sigma), dw[..., jnp.newaxis])[..., 0]

            return key, u_new

        key, u_end = jax.lax.fori_loop(0, self.solver_config.traj_len, traj_calc, (key, u_start))
        pde_loss = self.solver_config.pde_scale * jnp.mean((u_end - self.bc_fn(X[:, -1, :]))**2)
        return pde_loss, ((pde_loss,), key)
    
    # --------------------------------------------------

    def splitting_loss(self, key, params, index):
        batch = self.solver_config.micro_batch
        
        key, T, dT, dW, X = self.make_full_domain_euler(key, batch)
        if self.model_config.use_batch_norm:
            params_i = self._tree_index(params, index)
            _, params_i = self.bn_apply_train(self.model, params_i, X[:, index, :])
            params_i = self.clamp_bn_stats(params_i, min_var=1e-6)
            params = self._tree_update(params, index, params_i)

        t_curr = T[:, index, :]
        dt = dT[:, index, :]
        x_curr = X[:, index, :]
        jax.debug.print("idx={i} dt={dt} ||x_curr||={xx}", i=index, dt=jnp.max(dt), xx=jnp.max(jnp.linalg.norm(x_curr, axis=-1)))

        t_next = T[:, index+1, :]
        x_next = X[:, index+1, :]

        use_model = index < jnp.asarray(self.solver_config.traj_len-1, dtype=index.dtype)

        def _from_model(op):
            params_all, t, x = op
            params_next = self._tree_index(params_all, index+1)  # dynamic index
            params_next = jax.tree_util.tree_map(jax.lax.stop_gradient, params_next)
            u, ux = self.calc_ux(self.model, params_next, x)
            jax.debug.print("idx={i} ||ux||^2 max={v}", i=index, v=jnp.max(jnp.sum(ux**2, axis=-1)))
            return u + self.h(t, x, u, ux)*dt
        
        def _from_bc(op):
            _, t, x = op
            u = self.bc_fn(x)
            ux = self.calc_bcx(x)
            jax.debug.print("idx={i} ||ux||^2 max={v}", i=index, v=jnp.max(jnp.sum(ux**2, axis=-1)))
            return u + self.h(t, x, u, ux)*dt
        
        u_curr_pred = jax.lax.cond(use_model, _from_model, _from_bc, operand=(params, t_next, x_next))
        u_curr = self.calc_u(self.model, self._tree_index(params, index), x_curr)
        jax.debug.print("idx={i} |u_curr|={u}", i=index, u=jnp.max(jnp.abs(u_curr)))
        loss = self.solver_config.pde_scale * jnp.mean((u_curr - u_curr_pred)**2)

        return loss, ((loss,), key, params)

    # --------------------------------------------------
    # Plot Methods
    # --------------------------------------------------
    
    def init_wandb(self):
        print("Initializing wandb")
        wandb.init(project=self.solver_config.project_name,
                   name=self.solver_config.run_name,
                   config={
                       'solver': vars(self.solver_config),
                       'model': vars(self.model_config),
                       'problem': vars(self.problem_config)
                   })

    def close(self):
        wandb.finish()


    def plot_pred(self, params, i):
        time = self.sol_T[:, :, 0].T
        pred = self.calc_u(self.model, params, self.sol_X, self.sol_T)[:, :, 0].T
        true = self.sol_U[:, :, 0].T
        L1 = jnp.mean(jnp.abs(pred - true) / jnp.abs(true), axis=1)

        fig_pred = plt.figure(figsize=(5, 3))
        plt.plot(time[:, :4], pred[:, :4], "r", linewidth=1)
        plt.plot(time[:, :4], true[:, :4], ":b", linewidth=1)
        plt.title('Prediction') 
        plt.close(fig_pred)
        wandb.log({'Prediction': wandb.Image(fig_pred)}, step=i)

        fig_L1 = plt.figure(figsize=(5, 3))
        plt.plot(time[:, 0], L1, "b", linewidth=1)
        plt.title('L1 Error')
        plt.yscale('log')
        plt.close(fig_L1)
        wandb.log({'L1 Error': wandb.Image(fig_L1)}, step=i)

    def calc_RL(self, params):
        pred = self.calc_u(self.model, params, self.sol_X, self.sol_T)
        true = self.sol_U

        RL = jnp.mean(jnp.abs(pred - true) / jnp.abs(true))
        return RL
    
    def calc_RL_T0(self, params):
        if self.solver_config.loss_method == 'bsde':
            pred = self.calc_u(self.model, params['sol'], self.get_exact_X0()[jnp.newaxis, :])
        elif self.solver_config.loss_method == 'splitting':
            pred = self.calc_u(self.model, self._tree_index(params, 0), self.get_exact_X0()[jnp.newaxis, :])
        else:
            pred = self.calc_u(self.model, params, self.get_exact_X0()[jnp.newaxis, :], jnp.zeros((1, 1)))
        true = self.get_exact_Y0()

        RL_T0 = jnp.mean(jnp.abs(pred - true) / jnp.abs(true))
        return pred, RL_T0
    
    @partial(jax.jit, static_argnums=0)
    def jit_calc_RL(self,params):
        return self.calc_RL(params)

    @partial(jax.jit, static_argnums=0)
    def jit_calc_RL_T0(self,params):
        return self.calc_RL_T0(params)


    def get_eval_data(self):
        pass

    def plot_eval(self, params):
        pass

    def calc_eval(self, params):
        pass

    @partial(jax.jit, static_argnums=0)
    def jit_calc_eval(self,params):
        return self.calc_eval(params)

    # --------------------------------------------------
    # Optimization Methods
    # --------------------------------------------------
    
    @partial(jax.jit, static_argnums=0)
    def optimize(self, key, params, opt_state):
        key, params, losses, grad = self.grad_fn(key, params)
        loss = jnp.sum(jnp.asarray(losses))
        updates, opt_state = self.optimizer.update(grad, opt_state)
        params = optax.apply_updates(params, updates)

        return key, loss, losses, params, opt_state
    
    @partial(jax.jit, static_argnums=0)
    def sequential_optimize(self, key, params, opt_state, index):
        key, params, losses, grad = self.grad_fn(key, params, index)
        loss = jnp.sum(jnp.asarray(losses))

        param_i = self._tree_index(params, index)
        grad_i = self._tree_index(grad, index)
        opt_state_i = self._tree_index(opt_state, index)
        
        # def has_bad(x):
        #     return jnp.logical_or(jnp.any(jnp.isnan(x)), jnp.any(jnp.isinf(x)))
        # bad = jax.tree_util.tree_reduce(lambda a,b: a | has_bad(b), grad_i, False)

        # updates_i, opt_state_i_new = jax.lax.cond(
        #     bad,
        #     lambda _: (jax.tree_util.tree_map(jnp.zeros_like, grad_i), opt_state_i),  # 스킵
        #     lambda _: self.optimizer.update(grad_i, opt_state_i, param_i),
        #     operand=None
        # )

        updates_i, opt_state_i_new = self.optimizer.update(grad_i, opt_state_i, param_i)
        param_i_new = optax.apply_updates(param_i, updates_i)
        
        params = self._tree_update(params, index, param_i_new)
        opt_state = self._tree_update(opt_state, index, opt_state_i_new)

        return key, loss, losses, params, opt_state


# --------------------------------------------------
# Control Class
# --------------------------------------------------

class Controller():

    def __init__(self, solver: Solver, seed=20226074):
        self.solver = solver
        self.key = Key(jax.random.PRNGKey(seed))
        self.key, self.params, self.opt_state = self.solver.init_solver(self.key)

    def step(self, i):
        if self.solver.solver_config.loss_method == 'splitting':
            if i > 0 and i%(self.solver.solver_config.iter) == 0:
                self.solver.index = self.solver.index - 1 
            idx = jnp.asarray(self.solver.index, dtype=jnp.int32)
            self.key, loss, losses, self.params, self.opt_state = self.solver.sequential_optimize(self.key, self.params, self.opt_state, idx)
        else:
            self.key, loss, losses, self.params, self.opt_state = self.solver.optimize(self.key, self.params, self.opt_state)
        
        if self.solver.solver_config.save_to_wandb:
            wandb.log({'loss': loss, **{'loss'+str(k+1): v for k, v in dict(enumerate(losses)).items()}}, step=i)
            pred_T0, RL_T0 = self.solver.jit_calc_RL_T0(self.params)
            wandb.log({'pred_T0': pred_T0, "RL_T0": RL_T0}, step=i)
            # if self.solver.model_config.time_coupled and self.solver.solver_config.analytic_traj_sol:
            #     RL = self.solver.jit_calc_RL(self.params)
            #     wandb.log({"RL": RL}, step=i)
            if self.solver.solver_config.custom_eval:
                custom_eval = self.solver.jit_calc_eval(self.params)
                wandb.log({"eval": custom_eval}, step=i)
            if i%(self.solver.solver_config.iter//self.solver.solver_config.num_figures) == 0:
                if self.solver.model_config.time_coupled and self.solver.solver_config.analytic_traj_sol:
                    RL = self.solver.jit_calc_RL(self.params)
                    wandb.log({"RL": RL}, step=i)   
                    self.solver.plot_pred(self.params, i)
                if self.solver.solver_config.custom_eval:
                    self.solver.plot_eval(self.params, i)

    def solve(self):
        if self.solver.solver_config.loss_method == 'splitting':
            for i in tqdm.tqdm(range(self.solver.solver_config.traj_len * self.solver.solver_config.iter)):
                self.step(i)
        else:
            for i in tqdm.tqdm(range(self.solver.solver_config.iter)):
                self.step(i)

        if self.solver.solver_config.save_to_wandb:
            if self.solver.model_config.time_coupled and self.solver.solver_config.analytic_traj_sol:
                self.solver.plot_pred(self.params, self.solver.solver_config.iter)
            if self.solver.solver_config.custom_eval:
                self.solver.plot_eval(self.params, self.solver.solver_config.iter)

        path = Path('./checkpoints/')
        path.mkdir(exist_ok=True)
        if self.solver.solver_config.save_model:
            model_bytes = fs.to_bytes(self.params)
            (path/f'{self.solver.solver_config.project_name}_{self.solver.solver_config.run_name}_model.msgpack').write_bytes(model_bytes)
        if self.solver.solver_config.save_opt:
            opt_bytes = fs.to_bytes(self.opt_state)
            (path/f'{self.solver.solver_config.project_name}_{self.solver.solver_config.run_name}_opt.msgpack').write_bytes(opt_bytes)

        self.solver.close()