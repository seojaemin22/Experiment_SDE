from dataclasses import dataclass, field
from typing import ClassVar
from jax import Array
from jax import numpy as jnp

@dataclass
class Config():
    case: str = 'base'
    
    # Model Def
    d_in: int = 1  # not including t
    d_out: int = 1
    d_hidden: int = 64
    num_layers: int = 8
    activation: str = 'swish'
    four_emb: bool = True
    emb_dim: int = 256
    emb_scale: float = 1
    skip_conn: bool = True
    save_layers: tuple = (0,2,4,6)
    skip_layers: tuple = (2,4,6,8)

    # Training Def
    batch: int = 256
    micro_batch: int = 4
    optim: str = 'adam'
    lr: float = 1e-3
    iter: int = 100000
    loss_method: str = 'pinns'

    # Schedule Def
    schedule: str = 'None'  # piecewise_constant, cosine_decay, cosine_onecycle
    boundaries_and_scales: dict = field(default_factory=lambda: {50000: 0.1})  # for piecewise_constant
    
    # PINNS Loss Def
    pde_scale: float = 1
    bc_scale: float = 1

    # BSDE Loss Def  ( + pde iteration )
    traj_len: int = 50
    reset_u: bool = True
    skip_len: int = 5
 
    # Extras
    project_name: str = 'BSDE_workspace'
    run_name: str = 'test'
    save_to_wandb: bool = True
    num_figures: int = 10
    track_pinns_loss: bool = False
    track_fspinns_loss: bool = False
    track_bsde_loss: bool = False
    track_bsde_heun_loss: bool = False
    checkpointing: bool = False
    analytic_sol: bool = True
    custom_eval: bool = False
    periodic: bool = False