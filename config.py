from dataclasses import dataclass, field
from math import sqrt
# from typing import ClassVar
# from jax import Array
# from jax import numpy as jnp

@dataclass
class Model_Config():
    use_float64: bool = False

    # dimensions
    d_in: int = 1  # not including t
    d_out: int = 1

    # method of derivative
    derivative: str = 'backward'  # forward

    # time_dependent, boundary condition
    time_coupled: bool = True
    use_hard_constraint: bool = False  # if time_coupled is False, this option has no effect

    T: float = 1.0
    bc_name: str = 'HJB_example'
    bc_scale: float = 1.0  # if use_hard_constraint = False

    # model : MLP
    model_name: str = 'MLP'

    # MLP model
    MLP_d_hidden: int = 128
    MLP_num_layers: int = 6
    MLP_activation: int = 'tanh'
    MLP_skip_conn: bool = True
    MLP_save_layers: tuple = (0,2,4)
    MLP_skip_layers: tuple = (2,4,6)
    kernel_init : str = 'xavier_init'
    
    # normalization
    use_batch_norm: bool = False
    


@dataclass
class Solver_Config():

    # optimizer
    optim: str = 'adam'
    lr: float = 1e-3

    # iteration
    iter: int = 100000

    # batch
    batch: int = 64
    micro_batch: int = 64

    # scheduler
    schedule: str = 'None'  # piecewise_constant, cosine_decay, cosine_onecycle
    boundaries_and_scales: dict = field(default_factory=lambda: {50000: 0.1})  # for piecewise_constant
    
    # domain, trajectory setting
    T: float = 1.0
    traj_len: int = 50
    use_delta: bool = False  # if use_delta is True, t_1 is sampled from U(0, T/traj_len)

    # X0 distribution
    X0_std: float = 0.0

    # loss method
    # time-coupled model based : fspinns, fbsnn, fbsnnheun, shotgun
    # time-decoupled model based : bsde, splitting
    loss_method: str = 'bsde'
    pde_scale: float = 1.0

    # shotgun setting
    shotgun_local_batch: int = 64
    shotgun_Delta_t: float = 4**(-5)
    shotgun_use_traj_loss: bool = False

    # checkpointing
    checkpointing: bool = False

    # evaluation
    analytic_traj_sol = True
    test_traj_len: int = 100
    custom_eval: bool = False

    # save and load
    save_model: bool = True
    save_opt: bool = False
    model_state: str = 'address'
    opt_state: str = 'address'

    # project
    project_name: str = 'FBSDE_workspace'
    run_name: str = 'test'
    save_to_wandb: bool = True
    num_figures: int = 100  # if model is time-decoupled, num_figures can be ignored


@dataclass
class Problem_Config():
    problem_name: str = 'HJB'