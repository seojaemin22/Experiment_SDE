import jax.numpy as jnp
import flax.linen as nn
import jax
from config import Config
from jax.nn import initializers

class WaveAct(nn.Module):

    @nn.compact
    def __call__(self,x):
        w1 = self.param('w1',nn.initializers.normal(.1), (x.shape[-1],))
        w2 = self.param('w2',nn.initializers.normal(.1), (x.shape[-1],))
        return jnp.asarray(w1) * jnp.sin(x) + jnp.asarray(w2) * jnp.cos(x)

class Sin(nn.Module):

    @nn.compact
    def __call__(self,x):
        return jnp.sin(x)
    
class FourierEmbs(nn.Module):
    config: Config

    @nn.compact
    def __call__(self, x):
        kernel = self.param("kernel", jax.nn.initializers.normal(self.config.emb_scale), (x.shape[-1], self.config.emb_dim // 2))
        y = jnp.concatenate([jnp.cos(jnp.dot(x, kernel)), jnp.sin(jnp.dot(x, kernel))], axis=-1)
        return y

# --------------------------------------------------

class PINNs(nn.Module):
    config: Config

    def setup(self):
        custom_act = False

        if self.config.activation == 'wave':
            self.activation = WaveAct
            custom_act = True
        elif self.config.activation == 'sin':
            self.activation = Sin
            custom_act = True
        elif self.config.activation == 'tanh':
            self.activation = nn.tanh
        elif self.config.activation == 'swish':
            self.activation = nn.swish
        elif self.config.activation == 'leaky_relu':
            self.activation = nn.leaky_relu
        else:
            raise Exception("Activation '" + self.config.activation + "' Not Implemented")

        if self.config.four_emb:
            self.four_layer = FourierEmbs(self.config)
        
        xavier_init = initializers.glorot_uniform()
        zero_init   = initializers.zeros

        layers = []
        for i in range(self.config.num_layers):
            layers.append(nn.Dense(self.config.d_hidden, kernel_init=xavier_init, bias_init=zero_init))
            layers.append(self.activation() if custom_act else self.activation)
        self.output_layer = nn.Dense(self.config.d_out, kernel_init=xavier_init, bias_init=zero_init)
        self.layers = layers
    
    def __call__(self, *args):
        src = jnp.concatenate(args, axis=-1)
        if self.config.periodic:
            src = jnp.concatenate([jnp.sin(src[..., 0]), jnp.cos(src[..., 0]), src[..., 1:]], axis=-1)
        if self.config.four_emb:
            src = self.four_layer(src)

        src_skip = jnp.zeros((*src.shape[:-1], self.config.d_hidden))
        for i, layer in enumerate(self.layers):
            src = layer(src)
            if self.config.skip_conn:
                src_skip = jnp.where(jnp.any(i == jnp.asarray(self.config.save_layers)*2), src, src_skip)
                src = jnp.where(jnp.any(i == jnp.asarray(self.config.skip_layers)*2), src + src_skip, src)
        src = self.output_layer(src)
        return src
