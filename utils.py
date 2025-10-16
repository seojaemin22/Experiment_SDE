from dataclasses import dataclass
import jax, jax.numpy as jnp

@jax.tree_util.register_pytree_node_class
class Key:
    def __init__(self, key):
        self.key = key
    
    def split(self):
        k1, k2 = jax.random.split(self.key)
        return Key(k1), k2
    
    def split_n(self, n: int):
        ks = jax.random.split(self.key, n+1)
        return Key(ks[0]), ks[1:]
    
    # JAX PyTree Definitions
    def tree_flatten(self):
        return ((self.key,), {})

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(children[0])