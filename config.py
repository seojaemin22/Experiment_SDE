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
    periodic: bool = False

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
    dt: float = 0.02
    reset_u: bool = True
    skip_len: int = 5
    
    # Save and Load
    save_model: bool = True
    save_opt: bool = False
    model_state: str = 'address'
    opt_state: str = 'address'

    # Track another loss
    track_pinns_loss: bool = False

    # Extras
    project_name: str = 'BSDE_workspace'
    run_name: str = 'test'
    save_to_wandb: bool = True
    num_figures: int = 10
    checkpointing: bool = False
    analytic_sol: bool = True
    custom_eval: bool = False


@dataclass
class PDE_Config(Config):
    # Track another loss
    track_fspinns_loss: bool = False
    track_bsde_loss: bool = False
    track_bsde_heun_loss: bool = False

@dataclass
class PDE_FO_Config(PDE_Config):
    # PINNS Loss Def
    comp_scale: float = 1

@dataclass
class PIDE_Config(Config):
    # Problem Def
    mu_n: int = 2
    mu_phi: float = 0.01
    sigma_phi: float = 0.1
    lambda_: float = 0.3
    epsilon: float = 0.01
    tau: float = 0.1

    # Track another loss
    track_fspinns_loss: bool = False
    track_bsde_loss: bool = False