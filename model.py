import jax.numpy as jnp
import flax.linen as nn
import jax
from config import *
from jax.nn import initializers
from abc import ABC, abstractmethod
from typing import Optional, Callable

def get_activation(activation_name):
    if activation_name == 'sin':
        return Sin()
    elif activation_name == 'tanh':
        return Tanh()
    elif activation_name == 'mish':
        return Mish()
    elif activation_name == 'relu':
        return ReLU()
    elif activation_name == 'leakyrelu':
        return LeakyReLU()
    else:
        raise Exception("Activation '" + activation_name + "' Not Implemented")

def get_boundary_function(bc_name):
    if bc_name == 'HJB_default':
        return HJB_default_bc
    if bc_name == 'HJB_splitting':
        return HJB_splitting_bc
    if bc_name == 'BSB_default':
        return BSB_default_bc
    else:
        raise Exception("Boundary Condition '" + bc_name + "' Not Implemented")

def get_model(config: Model_Config):
    model_name = config.model_name

    if model_name == 'MLP':
        activation = get_activation(config.MLP_activation)
        boundary_function = get_boundary_function(config.bc_name)
        return MLP(config, activation=activation, boundary_function=boundary_function)
    else:
        raise Exception("Model '" + model_name + "' Not Implemented")

# --------------------------------------------------

class Activation(ABC):
    @abstractmethod
    def __call__(self, x):
        pass

    @abstractmethod
    def deriv1(self, x):
        pass

    def deriv2(self, x):
        pass

class Sin(Activation):

    def __call__(self, x):
        return jnp.sin(x)
    
    def deriv1(self, x):
        return jnp.cos(x)
    
    def deriv2(self, x):
        return -jnp.sin(x)
    
class Tanh(Activation):

    def __call__(self, x):
        return jnp.tanh(x)
    
    def deriv1(self, x):
        return 1.0 - jnp.tanh(x)**2
    
    def deriv2(self, x):
        return -2.0 * jnp.tanh(x) * (1.0 - jnp.tanh(x)**2)

class Mish(Activation):

    def __call__(self, x):
        return x * jnp.tanh(nn.softplus(x))
    
    def deriv1(self, x):
        sp = nn.softplus(x)
        tsp = jnp.tanh(sp)
        return tsp + x * (1.0 - tsp**2) * jax.nn.sigmoid(x)
    
    def deriv2(self, x):
        sp = nn.softplus(x)
        tsp = jnp.tanh(sp)
        dsp = jax.nn.sigmoid(x)
        dtsp = (1.0 - tsp**2) * dsp
        d2tsp = -2.0 * tsp * dtsp * dsp + (1.0 - tsp**2) * dsp * (1.0 - dsp)
        return 2.0 * dtsp + x * d2tsp

class ReLU(Activation):
    def __call__(self, x):
        return jnp.maximum(x, 0.0)
    
    def deriv1(self, x):
        return jnp.where(x > 0.0, 1.0, 0.0)
    
    def deriv2(self, x):
        return jnp.zeros_like(x)

class LeakyReLU(Activation):
    def __call__(self, x):
        return jnp.maximum(x, 0.01*x)
    
    def deriv1(self, x):
        return jnp.where(x > 0.0, 1.0, 0.01)
    
    def deriv2(self, x):
        return jnp.zeros_like(x)

# --------------------------------------------------

def HJB_default_bc(x):
    return jnp.log(0.5 * (1 + jnp.sum(x**2, keepdims=True, axis=-1)))

def HJB_splitting_bc(x):
    return jnp.sqrt(jnp.sqrt(jnp.sum(x**2, axis=-1, keepdims=True)))

def BSB_default_bc(x):
    return jnp.sum(x**2, keepdims=True, axis=-1)

# --------------------------------------------------

class MLP(nn.Module):
    config: Model_Config
    activation: Activation
    boundary_function: Optional[Callable] = None

    def setup(self):
        param_dtype = jnp.float64 if self.config.use_float64 else jnp.float32
        xavier_init = initializers.glorot_uniform()
        zero_init   = initializers.zeros
        he_init     = nn.initializers.variance_scaling(2.0, 'fan_in', 'truncated_normal')

        self.momentum = 0.99
        self.epsilon = 1e-2

        if self.config.use_batch_norm:
            self.input_batch_norm = nn.BatchNorm(momentum=self.momentum, epsilon=self.epsilon, param_dtype=param_dtype)
            self.hidden_batch_norms = [nn.BatchNorm(momentum=self.momentum, epsilon=self.epsilon, param_dtype=param_dtype) for _ in range(self.config.MLP_num_layers)]
            self.output_batch_norm = nn.BatchNorm(momentum=self.momentum, epsilon=self.epsilon, param_dtype=param_dtype)
        
        if self.config.kernel_init == 'he':
            self.layers = [nn.Dense(self.config.MLP_d_hidden, kernel_init=he_init, bias_init=zero_init, param_dtype=param_dtype) for _ in range(self.config.MLP_num_layers)]
            self.output_layer = nn.Dense(self.config.d_out, kernel_init=he_init, bias_init=zero_init, param_dtype=param_dtype)
        else:
            self.layers = [nn.Dense(self.config.MLP_d_hidden, kernel_init=xavier_init, bias_init=zero_init, param_dtype=param_dtype) for _ in range(self.config.MLP_num_layers)]
            self.output_layer = nn.Dense(self.config.d_out, kernel_init=xavier_init, bias_init=zero_init, param_dtype=param_dtype)
    
    def __call__(self, *args, use_running_average: bool = True):
        src = jnp.concatenate(args, axis=-1)
        if self.config.use_batch_norm:
            src = self.input_batch_norm(src, use_running_average=use_running_average)
        if self.config.MLP_skip_conn:
            src_skip = jnp.zeros((src.shape[0], self.config.MLP_d_hidden))
            
        for i in range(len(self.layers)):
            src = self.layers[i](src)
            if self.config.use_batch_norm:
                src = self.hidden_batch_norms[i](src, use_running_average=use_running_average)
            src = self.activation(src)
            if self.config.MLP_skip_conn:
                if i in self.config.MLP_save_layers:
                    src_skip = src
                if i in self.config.MLP_skip_layers:
                    src = src + src_skip

        if self.config.use_batch_norm:
            src = self.output_batch_norm(src, use_running_average=use_running_average)
        src = self.output_layer(src)

        if self.config.time_coupled and self.config.use_hard_constraint:
            t, x = args
            boundary_value = self.boundary_function(x)
            return boundary_value + (self.config.T - t)/self.config.T * src
        else:
            return src

    # --------------------------------------------------
    
    def forward_laplacian(self, params, *args, weight=None):
        src = jnp.concatenate(args, axis=-1)  # (B, Din_total)
        B = src.shape[0]
        Din_total = src.shape[1]
        Din = self.config.d_in

        G = jnp.zeros((B, Din_total, Din))
        G = G.at[:, 1:, :].set(jnp.eye(Din) if weight is None else weight)  # (weighted) gradient
        S = jnp.zeros((B, Din_total))  # (weighted) laplacian

        if self.config.MLP_skip_conn:
            src_skip = jnp.zeros((B, self.config.MLP_d_hidden))
            G_skip   = jnp.zeros((B, self.config.MLP_d_hidden, Din))
            S_skip   = jnp.zeros((B, self.config.MLP_d_hidden))

        for i in range(len(params['params']) - 1):
            W = params['params'][f'layers_{i}']['kernel']  # (Hin, Hout)
            b = params['params'][f'layers_{i}']['bias']    # (Hout,)

            # Linear layer : z(v) = L(x(v)) = W x(v) + b
            # dz(v) = W dx(v)
            # d2z(v) = W d2x(v)
            # lap(z(v)) = W lap(x(v))

            z = jnp.einsum('...i,ih->...h', src, W) + b  # (B, Hout)
            G = jnp.einsum('...jD,jh->...hD', G, W)      # (B, Hout, Din)
            S = jnp.einsum('...j,jh->...h', S, W)        # (B, Hout)

            # Batchnorm layer : z(v) = gamma * (z - mu)/sigma + beta
            # dz(v) = gamma/sigma * dz(v)
            # d2z(v) = gamma/sigma * d2z(v)
            # lap(z(v)) = gamma/sigma * lap(z(v))

            if self.config.use_batch_norm:
                mean = params['batch_stats'][f'hidden_batch_norms_{i}']['mean']
                var  = params['batch_stats'][f'hidden_batch_norms_{i}']['var']
                gamma = params['params'][f'hidden_batch_norms_{i}']['scale']
                beta  = params['params'][f'hidden_batch_norms_{i}']['bias']

                z = gamma * (z - mean) / jnp.sqrt(var + self.epsilon) + beta
                G = (gamma / jnp.sqrt(var + self.epsilon))[None, :, None] * G
                S = (gamma / jnp.sqrt(var + self.epsilon))[None, :] * S

            # Activation layer : z(v) = phi(x(v))
            # dz(v) = phi'(x(v)) dx(v)
            # d2z(v) = phi''(x(v)) d2x(v) + phi'(x(v)) (dx(v))^2
            # [ lap(z(v)) = phi''(x(v)) lap(z(v)) + phi'(x(v)) sum (dx(v))^2]

            phi1 = self.activation.deriv1(z)  # (B, Hout)
            phi2 = self.activation.deriv2(z)  # (B, Hout)

            S = phi2 * jnp.sum(G**2, axis=-1) + phi1 * S
            G = phi1[..., None] * G
            src = self.activation(z)

            # Skip connection
            if self.config.MLP_skip_conn:
                if i in self.config.MLP_save_layers:
                    src_skip, G_skip, S_skip = src, G, S
                if i in self.config.MLP_skip_layers:
                    src = src + src_skip
                    G = G + G_skip
                    S = S + S_skip

        # Output layer
        W = params['params']['output_layer']['kernel']
        b = params['params']['output_layer']['bias']
        src = jnp.einsum('...h,hk->...k', src, W) + b
        G = jnp.einsum('...hD,hk->...kD', G, W)
        S = jnp.einsum('...h,hk->...k',  S, W)

        u = src[..., :self.config.d_out]
        lap = S[..., :self.config.d_out]
        return u, lap