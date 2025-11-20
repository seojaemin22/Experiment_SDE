from solver import *
from config import *
import jax
from datetime import datetime

use_float64 = False
if use_float64:
    jax.config.update('jax_enable_x64', True)

model_config = Model_Config()
solver_config = Solver_Config()
problem_config = Problem_Config()

changed_settings = [

    # {'loss_method': 'fbsnn', 'problem_name': 'HJB', 'd_in': 100, 'bc_name': 'default',
    #  'iter': 100000, 'batch': 256, 'T': 1, 'traj_len': 100, 'use_delta': False, 'X0_std': 0.0,
    #  'lr': 2e-3, 'schedule': 'cosine_decay', 'MLP_d_hidden': 512, 'MLP_num_layers': 4, 'MLP_activation': 'mish',
    #  'bc_scale': 1e-1, 'causal_training': True, 'start_epsilon': 1e-3, 'exp_epsilon': 10, 'delta_epsilon': 0.99},
    
    # {'loss_method': 'fbsnnheun', 'problem_name': 'HJB', 'd_in': 100, 'bc_name': 'default',
    #  'iter': 100000, 'batch': 256, 'T': 1, 'traj_len': 100, 'use_delta': False, 'X0_std': 0.0,
    #  'lr': 2e-3, 'schedule': 'cosine_decay', 'MLP_d_hidden': 512, 'MLP_num_layers': 4, 'MLP_activation': 'mish',
    #  'bc_scale': 1e-1, 'causal_training': True, 'start_epsilon': 1e-3, 'exp_epsilon': 10, 'delta_epsilon': 0.99},



    # {'loss_method': 'fbsnn', 'problem_name': 'HJB', 'd_in': 500, 'bc_name': 'default',
    #  'iter': 100000, 'batch': 256, 'T': 1, 'traj_len': 100, 'use_delta': False, 'X0_std': 0.0,
    #  'lr': 2e-3, 'schedule': 'cosine_decay', 'MLP_d_hidden': 512, 'MLP_num_layers': 4, 'MLP_activation': 'mish',
    #  'bc_scale': 1e-1, 'causal_training': True, 'start_epsilon': 1e-3, 'exp_epsilon': 10, 'delta_epsilon': 0.99},

    # {'loss_method': 'fbsnn', 'problem_name': 'HJB', 'd_in': 100, 'bc_name': 'default',
    #  'iter': 100000, 'batch': 256, 'T': 5, 'traj_len': 500, 'use_delta': False, 'X0_std': 0.0,
    #  'lr': 2e-3, 'schedule': 'cosine_decay', 'MLP_d_hidden': 512, 'MLP_num_layers': 4, 'MLP_activation': 'mish',
    #  'bc_scale': 1e-1, 'causal_training': True, 'start_epsilon': 1e-3, 'exp_epsilon': 10, 'delta_epsilon': 0.99},
    
    # {'loss_method': 'fbsnnheun', 'problem_name': 'HJB', 'd_in': 500, 'bc_name': 'default',
    #  'iter': 100000, 'batch': 256, 'T': 1, 'traj_len': 100, 'use_delta': False, 'X0_std': 0.0,
    #  'lr': 2e-3, 'schedule': 'cosine_decay', 'MLP_d_hidden': 512, 'MLP_num_layers': 4, 'MLP_activation': 'mish',
    #  'bc_scale': 1e-1, 'causal_training': True, 'start_epsilon': 1e-3, 'exp_epsilon': 10, 'delta_epsilon': 0.99},
    
    # {'loss_method': 'fbsnnheun', 'problem_name': 'HJB', 'd_in': 100, 'bc_name': 'default',
    #  'iter': 100000, 'batch': 256, 'T': 5, 'traj_len': 500, 'use_delta': False, 'X0_std': 0.0,
    #  'lr': 2e-3, 'schedule': 'cosine_decay', 'MLP_d_hidden': 512, 'MLP_num_layers': 4, 'MLP_activation': 'mish',
    #  'bc_scale': 1e-1, 'causal_training': True, 'start_epsilon': 1e-3, 'exp_epsilon': 10, 'delta_epsilon': 0.99},

    # ------------------------------
    
    # {'loss_method': 'fbsnn', 'problem_name': 'BSB', 'd_in': 100, 'bc_name': 'default',
    #  'iter': 100000, 'batch': 256, 'T': 1, 'traj_len': 100, 'use_delta': False, 'X0_std': 0.0,
    #  'lr': 2e-3, 'schedule': 'cosine_decay', 'MLP_d_hidden': 512, 'MLP_num_layers': 4, 'MLP_activation': 'mish',
    #  'bc_scale': 1e-1, 'causal_training': True, 'start_epsilon': 1e-3, 'exp_epsilon': 10, 'delta_epsilon': 0.99},
    
    # {'loss_method': 'fbsnnheun', 'problem_name': 'BSB', 'd_in': 100, 'bc_name': 'default',
    #  'iter': 100000, 'batch': 256, 'T': 1, 'traj_len': 100, 'use_delta': False, 'X0_std': 0.0,
    #  'lr': 2e-3, 'schedule': 'cosine_decay', 'MLP_d_hidden': 512, 'MLP_num_layers': 4, 'MLP_activation': 'mish',
    #  'bc_scale': 1e-1, 'causal_training': True, 'start_epsilon': 1e-3, 'exp_epsilon': 10, 'delta_epsilon': 0.99},

    

    # {'loss_method': 'fbsnn', 'problem_name': 'BSB', 'd_in': 500, 'bc_name': 'default',
    #  'iter': 100000, 'batch': 256, 'T': 1, 'traj_len': 100, 'use_delta': False, 'X0_std': 0.0,
    #  'lr': 2e-3, 'schedule': 'cosine_decay', 'MLP_d_hidden': 512, 'MLP_num_layers': 4, 'MLP_activation': 'mish',
    #  'bc_scale': 1e-1, 'causal_training': True, 'start_epsilon': 1e-3, 'exp_epsilon': 10, 'delta_epsilon': 0.99},

    # {'loss_method': 'fbsnn', 'problem_name': 'BSB', 'd_in': 100, 'bc_name': 'default',
    #  'iter': 100000, 'batch': 256, 'T': 5, 'traj_len': 500, 'use_delta': False, 'X0_std': 0.0,
    #  'lr': 2e-3, 'schedule': 'cosine_decay', 'MLP_d_hidden': 512, 'MLP_num_layers': 4, 'MLP_activation': 'mish',
    #  'bc_scale': 1e-1, 'causal_training': True, 'start_epsilon': 1e-3, 'exp_epsilon': 10, 'delta_epsilon': 0.99},
    
    # {'loss_method': 'fbsnnheun', 'problem_name': 'BSB', 'd_in': 500, 'bc_name': 'default',
    #  'iter': 100000, 'batch': 256, 'T': 1, 'traj_len': 100, 'use_delta': False, 'X0_std': 0.0,
    #  'lr': 2e-3, 'schedule': 'cosine_decay', 'MLP_d_hidden': 512, 'MLP_num_layers': 4, 'MLP_activation': 'mish',
    #  'bc_scale': 1e-1, 'causal_training': True, 'start_epsilon': 1e-3, 'exp_epsilon': 10, 'delta_epsilon': 0.99},
    
    # {'loss_method': 'fbsnnheun', 'problem_name': 'BSB', 'd_in': 100, 'bc_name': 'default',
    #  'iter': 100000, 'batch': 256, 'T': 5, 'traj_len': 500, 'use_delta': False, 'X0_std': 0.0,
    #  'lr': 2e-3, 'schedule': 'cosine_decay', 'MLP_d_hidden': 512, 'MLP_num_layers': 4, 'MLP_activation': 'mish',
    #  'bc_scale': 1e-1, 'causal_training': True, 'start_epsilon': 1e-3, 'exp_epsilon': 10, 'delta_epsilon': 0.99},

    # ------------------------------

    # {'loss_method': 'fbsnn', 'problem_name': 'HJB', 'd_in': 100, 'bc_name': 'default',
    #  'iter': 100000, 'batch': 256, 'T': 1, 'traj_len': 100, 'use_delta': False, 'X0_std': 0.0,
    #  'lr': 2e-3, 'schedule': 'cosine_decay', 'MLP_d_hidden': 512, 'MLP_num_layers': 4, 'MLP_activation': 'mish',
    #  'bc_scale': 1e-1, 'causal_training': False, 'start_epsilon': 1e-3, 'exp_epsilon': 10, 'delta_epsilon': 0.99},
    
    # {'loss_method': 'fbsnnheun', 'problem_name': 'HJB', 'd_in': 100, 'bc_name': 'default',
    #  'iter': 100000, 'batch': 256, 'T': 1, 'traj_len': 100, 'use_delta': False, 'X0_std': 0.0,
    #  'lr': 2e-3, 'schedule': 'cosine_decay', 'MLP_d_hidden': 512, 'MLP_num_layers': 4, 'MLP_activation': 'mish',
    #  'bc_scale': 1e-1, 'causal_training': False, 'start_epsilon': 1e-3, 'exp_epsilon': 10, 'delta_epsilon': 0.99},



    # {'loss_method': 'fbsnn', 'problem_name': 'HJB', 'd_in': 500, 'bc_name': 'default',
    #  'iter': 100000, 'batch': 256, 'T': 1, 'traj_len': 100, 'use_delta': False, 'X0_std': 0.0,
    #  'lr': 2e-3, 'schedule': 'cosine_decay', 'MLP_d_hidden': 512, 'MLP_num_layers': 4, 'MLP_activation': 'mish',
    #  'bc_scale': 1e-1, 'causal_training': False, 'start_epsilon': 1e-3, 'exp_epsilon': 10, 'delta_epsilon': 0.99},

    # {'loss_method': 'fbsnn', 'problem_name': 'HJB', 'd_in': 100, 'bc_name': 'default',
    #  'iter': 100000, 'batch': 256, 'T': 5, 'traj_len': 500, 'use_delta': False, 'X0_std': 0.0,
    #  'lr': 2e-3, 'schedule': 'cosine_decay', 'MLP_d_hidden': 512, 'MLP_num_layers': 4, 'MLP_activation': 'mish',
    #  'bc_scale': 1e-1, 'causal_training': False, 'start_epsilon': 1e-3, 'exp_epsilon': 10, 'delta_epsilon': 0.99},
    
    # {'loss_method': 'fbsnnheun', 'problem_name': 'HJB', 'd_in': 500, 'bc_name': 'default',
    #  'iter': 100000, 'batch': 256, 'T': 1, 'traj_len': 100, 'use_delta': False, 'X0_std': 0.0,
    #  'lr': 2e-3, 'schedule': 'cosine_decay', 'MLP_d_hidden': 512, 'MLP_num_layers': 4, 'MLP_activation': 'mish',
    #  'bc_scale': 1e-1, 'causal_training': False, 'start_epsilon': 1e-3, 'exp_epsilon': 10, 'delta_epsilon': 0.99},
    
    # {'loss_method': 'fbsnnheun', 'problem_name': 'HJB', 'd_in': 100, 'bc_name': 'default',
    #  'iter': 100000, 'batch': 256, 'T': 5, 'traj_len': 500, 'use_delta': False, 'X0_std': 0.0,
    #  'lr': 2e-3, 'schedule': 'cosine_decay', 'MLP_d_hidden': 512, 'MLP_num_layers': 4, 'MLP_activation': 'mish',
    #  'bc_scale': 1e-1, 'causal_training': False, 'start_epsilon': 1e-3, 'exp_epsilon': 10, 'delta_epsilon': 0.99},

    # # ------------------------------
    
    # {'loss_method': 'fbsnn', 'problem_name': 'BSB', 'd_in': 100, 'bc_name': 'default',
    #  'iter': 100000, 'batch': 256, 'T': 1, 'traj_len': 100, 'use_delta': False, 'X0_std': 0.0,
    #  'lr': 2e-3, 'schedule': 'cosine_decay', 'MLP_d_hidden': 512, 'MLP_num_layers': 4, 'MLP_activation': 'mish',
    #  'bc_scale': 1e-1, 'causal_training': False, 'start_epsilon': 1e-3, 'exp_epsilon': 10, 'delta_epsilon': 0.99},
    
    # {'loss_method': 'fbsnnheun', 'problem_name': 'BSB', 'd_in': 100, 'bc_name': 'default',
    #  'iter': 100000, 'batch': 256, 'T': 1, 'traj_len': 100, 'use_delta': False, 'X0_std': 0.0,
    #  'lr': 2e-3, 'schedule': 'cosine_decay', 'MLP_d_hidden': 512, 'MLP_num_layers': 4, 'MLP_activation': 'mish',
    #  'bc_scale': 1e-1, 'causal_training': False, 'start_epsilon': 1e-3, 'exp_epsilon': 10, 'delta_epsilon': 0.99},



    # {'loss_method': 'fbsnn', 'problem_name': 'BSB', 'd_in': 500, 'bc_name': 'default',
    #  'iter': 100000, 'batch': 256, 'T': 1, 'traj_len': 100, 'use_delta': False, 'X0_std': 0.0,
    #  'lr': 2e-3, 'schedule': 'cosine_decay', 'MLP_d_hidden': 512, 'MLP_num_layers': 4, 'MLP_activation': 'mish',
    #  'bc_scale': 1e-1, 'causal_training': False, 'start_epsilon': 1e-3, 'exp_epsilon': 10, 'delta_epsilon': 0.99},

    # {'loss_method': 'fbsnn', 'problem_name': 'BSB', 'd_in': 100, 'bc_name': 'default',
    #  'iter': 100000, 'batch': 256, 'T': 5, 'traj_len': 500, 'use_delta': False, 'X0_std': 0.0,
    #  'lr': 2e-3, 'schedule': 'cosine_decay', 'MLP_d_hidden': 512, 'MLP_num_layers': 4, 'MLP_activation': 'mish',
    #  'bc_scale': 1e-1, 'causal_training': False, 'start_epsilon': 1e-3, 'exp_epsilon': 10, 'delta_epsilon': 0.99},
    
    # {'loss_method': 'fbsnnheun', 'problem_name': 'BSB', 'd_in': 500, 'bc_name': 'default',
    #  'iter': 100000, 'batch': 256, 'T': 1, 'traj_len': 100, 'use_delta': False, 'X0_std': 0.0,
    #  'lr': 2e-3, 'schedule': 'cosine_decay', 'MLP_d_hidden': 512, 'MLP_num_layers': 4, 'MLP_activation': 'mish',
    #  'bc_scale': 1e-1, 'causal_training': False, 'start_epsilon': 1e-3, 'exp_epsilon': 10, 'delta_epsilon': 0.99},
    
    # {'loss_method': 'fbsnnheun', 'problem_name': 'BSB', 'd_in': 100, 'bc_name': 'default',
    #  'iter': 100000, 'batch': 256, 'T': 5, 'traj_len': 500, 'use_delta': False, 'X0_std': 0.0,
    #  'lr': 2e-3, 'schedule': 'cosine_decay', 'MLP_d_hidden': 512, 'MLP_num_layers': 4, 'MLP_activation': 'mish',
    #  'bc_scale': 1e-1, 'causal_training': False, 'start_epsilon': 1e-3, 'exp_epsilon': 10, 'delta_epsilon': 0.99},

    # ------------------------------

    # {'loss_method': 'fbsnnheun', 'problem_name': 'BSB', 'd_in': 100, 'bc_name': 'default',
    #  'iter': 100000, 'batch': 256, 'T': 5, 'traj_len': 500, 'use_delta': False, 'X0_std': 0.0,
    #  'lr': 2e-3, 'schedule': 'cosine_decay', 'MLP_d_hidden': 512, 'MLP_num_layers': 4, 'MLP_activation': 'mish', 'use_hard_constraint': True,
    #  'bc_scale': 1e-1, 'causal_training': False, 'start_epsilon': 1e-3, 'exp_epsilon': 10, 'delta_epsilon': 0.98},

    # {'loss_method': 'fbsnnheun', 'problem_name': 'BSB', 'd_in': 100, 'bc_name': 'default',
    #  'iter': 100000, 'batch': 256, 'T': 5, 'traj_len': 500, 'use_delta': False, 'X0_std': 0.0,
    #  'lr': 2e-3, 'schedule': 'cosine_decay', 'MLP_d_hidden': 512, 'MLP_num_layers': 4, 'MLP_activation': 'mish', 'use_hard_constraint': True,
    #  'bc_scale': 1e-1, 'causal_training': True, 'start_epsilon': 1e-3, 'exp_epsilon': 10, 'delta_epsilon': 0.98},

    # {'loss_method': 'fbsnn', 'problem_name': 'BSB', 'd_in': 100, 'bc_name': 'default',
    #  'iter': 100000, 'batch': 256, 'T': 5, 'traj_len': 500, 'use_delta': False, 'X0_std': 0.0,
    #  'lr': 2e-3, 'schedule': 'cosine_decay', 'MLP_d_hidden': 512, 'MLP_num_layers': 4, 'MLP_activation': 'mish', 'use_hard_constraint': True,
    #  'bc_scale': 1e-1, 'causal_training': False, 'start_epsilon': 1e-3, 'exp_epsilon': 10, 'delta_epsilon': 0.98},
    
    # {'loss_method': 'fbsnn', 'problem_name': 'BSB', 'd_in': 100, 'bc_name': 'default',
    #  'iter': 100000, 'batch': 256, 'T': 5, 'traj_len': 500, 'use_delta': False, 'X0_std': 0.0,
    #  'lr': 2e-3, 'schedule': 'cosine_decay', 'MLP_d_hidden': 512, 'MLP_num_layers': 4, 'MLP_activation': 'mish', 'use_hard_constraint': True,
    #  'bc_scale': 1e-1, 'causal_training': True, 'start_epsilon': 1e-3, 'exp_epsilon': 10, 'delta_epsilon': 0.98},

    # ------------------------------
    # ------------------------------
    # ------------------------------
    

    # {'loss_method': 'nobiasfbsnn1', 'problem_name': 'HJB', 'd_in': 100, 'bc_name': 'default',
    #  'iter': 100000, 'batch': 256, 'T': 1, 'traj_len': 100, 'use_delta': False, 'X0_std': 0.0,
    #  'lr': 2e-3, 'schedule': 'cosine_decay', 'MLP_d_hidden': 512, 'MLP_num_layers': 4, 'MLP_activation': 'mish',
    #  'bc_scale': 1e-1, 'causal_training': False, 'start_epsilon': 1e-3, 'exp_epsilon': 10, 'delta_epsilon': 0.99},

    # {'loss_method': 'nobiasfbsnn1', 'problem_name': 'HJB', 'd_in': 500, 'bc_name': 'default',
    #  'iter': 100000, 'batch': 256, 'T': 1, 'traj_len': 100, 'use_delta': False, 'X0_std': 0.0,
    #  'lr': 2e-3, 'schedule': 'cosine_decay', 'MLP_d_hidden': 512, 'MLP_num_layers': 4, 'MLP_activation': 'mish',
    #  'bc_scale': 1e-1, 'causal_training': False, 'start_epsilon': 1e-3, 'exp_epsilon': 10, 'delta_epsilon': 0.99},

    # {'loss_method': 'nobiasfbsnn1', 'problem_name': 'HJB', 'd_in': 100, 'bc_name': 'default',
    #  'iter': 100000, 'batch': 256, 'T': 5, 'traj_len': 500, 'use_delta': False, 'X0_std': 0.0,
    #  'lr': 2e-3, 'schedule': 'cosine_decay', 'MLP_d_hidden': 512, 'MLP_num_layers': 4, 'MLP_activation': 'mish',
    #  'bc_scale': 1e-1, 'causal_training': False, 'start_epsilon': 1e-3, 'exp_epsilon': 10, 'delta_epsilon': 0.99},

    
    
    # {'loss_method': 'nobiasfbsnn2', 'problem_name': 'HJB', 'd_in': 100, 'bc_name': 'default',
    #  'iter': 100000, 'batch': 256, 'T': 1, 'traj_len': 100, 'use_delta': False, 'X0_std': 0.0,
    #  'lr': 2e-3, 'schedule': 'cosine_decay', 'MLP_d_hidden': 512, 'MLP_num_layers': 4, 'MLP_activation': 'mish',
    #  'bc_scale': 1e-1, 'causal_training': False, 'start_epsilon': 1e-3, 'exp_epsilon': 10, 'delta_epsilon': 0.99},

    # {'loss_method': 'nobiasfbsnn2', 'problem_name': 'HJB', 'd_in': 500, 'bc_name': 'default',
    #  'iter': 100000, 'batch': 256, 'T': 1, 'traj_len': 100, 'use_delta': False, 'X0_std': 0.0,
    #  'lr': 2e-3, 'schedule': 'cosine_decay', 'MLP_d_hidden': 512, 'MLP_num_layers': 4, 'MLP_activation': 'mish',
    #  'bc_scale': 1e-1, 'causal_training': False, 'start_epsilon': 1e-3, 'exp_epsilon': 10, 'delta_epsilon': 0.99},

    # {'loss_method': 'nobiasfbsnn2', 'problem_name': 'HJB', 'd_in': 100, 'bc_name': 'default',
    #  'iter': 100000, 'batch': 256, 'T': 5, 'traj_len': 500, 'use_delta': False, 'X0_std': 0.0,
    #  'lr': 2e-3, 'schedule': 'cosine_decay', 'MLP_d_hidden': 512, 'MLP_num_layers': 4, 'MLP_activation': 'mish',
    #  'bc_scale': 1e-1, 'causal_training': False, 'start_epsilon': 1e-3, 'exp_epsilon': 10, 'delta_epsilon': 0.99},

    

    # {'loss_method': 'nobiasfbsnn3', 'problem_name': 'HJB', 'd_in': 100, 'bc_name': 'default',
    #  'iter': 100000, 'batch': 256, 'T': 1, 'traj_len': 100, 'use_delta': False, 'X0_std': 0.0,
    #  'lr': 2e-3, 'schedule': 'cosine_decay', 'MLP_d_hidden': 512, 'MLP_num_layers': 4, 'MLP_activation': 'mish',
    #  'bc_scale': 1e-1, 'causal_training': False, 'start_epsilon': 1e-3, 'exp_epsilon': 10, 'delta_epsilon': 0.99},

    # {'loss_method': 'nobiasfbsnn3', 'problem_name': 'HJB', 'd_in': 500, 'bc_name': 'default',
    #  'iter': 100000, 'batch': 256, 'T': 1, 'traj_len': 100, 'use_delta': False, 'X0_std': 0.0,
    #  'lr': 2e-3, 'schedule': 'cosine_decay', 'MLP_d_hidden': 512, 'MLP_num_layers': 4, 'MLP_activation': 'mish',
    #  'bc_scale': 1e-1, 'causal_training': False, 'start_epsilon': 1e-3, 'exp_epsilon': 10, 'delta_epsilon': 0.99},

    # {'loss_method': 'nobiasfbsnn3', 'problem_name': 'HJB', 'd_in': 100, 'bc_name': 'default',
    #  'iter': 100000, 'batch': 256, 'T': 5, 'traj_len': 500, 'use_delta': False, 'X0_std': 0.0,
    #  'lr': 2e-3, 'schedule': 'cosine_decay', 'MLP_d_hidden': 512, 'MLP_num_layers': 4, 'MLP_activation': 'mish',
    #  'bc_scale': 1e-1, 'causal_training': False, 'start_epsilon': 1e-3, 'exp_epsilon': 10, 'delta_epsilon': 0.99},

    # # ------------------------------
    
    # {'loss_method': 'nobiasfbsnn1', 'problem_name': 'BSB', 'd_in': 100, 'bc_name': 'default',
    #  'iter': 100000, 'batch': 256, 'T': 1, 'traj_len': 100, 'use_delta': False, 'X0_std': 0.0,
    #  'lr': 2e-3, 'schedule': 'cosine_decay', 'MLP_d_hidden': 512, 'MLP_num_layers': 4, 'MLP_activation': 'mish',
    #  'bc_scale': 1e-1, 'causal_training': False, 'start_epsilon': 1e-3, 'exp_epsilon': 10, 'delta_epsilon': 0.99},

    # {'loss_method': 'nobiasfbsnn1', 'problem_name': 'BSB', 'd_in': 500, 'bc_name': 'default',
    #  'iter': 100000, 'batch': 256, 'T': 1, 'traj_len': 100, 'use_delta': False, 'X0_std': 0.0,
    #  'lr': 2e-3, 'schedule': 'cosine_decay', 'MLP_d_hidden': 512, 'MLP_num_layers': 4, 'MLP_activation': 'mish',
    #  'bc_scale': 1e-1, 'causal_training': False, 'start_epsilon': 1e-3, 'exp_epsilon': 10, 'delta_epsilon': 0.99},

    # {'loss_method': 'nobiasfbsnn1', 'problem_name': 'BSB', 'd_in': 100, 'bc_name': 'default',
    #  'iter': 100000, 'batch': 256, 'T': 5, 'traj_len': 500, 'use_delta': False, 'X0_std': 0.0,
    #  'lr': 2e-3, 'schedule': 'cosine_decay', 'MLP_d_hidden': 512, 'MLP_num_layers': 4, 'MLP_activation': 'mish',
    #  'bc_scale': 1e-1, 'causal_training': False, 'start_epsilon': 1e-3, 'exp_epsilon': 10, 'delta_epsilon': 0.99},

    

    # {'loss_method': 'nobiasfbsnn2', 'problem_name': 'BSB', 'd_in': 100, 'bc_name': 'default',
    #  'iter': 100000, 'batch': 256, 'T': 1, 'traj_len': 100, 'use_delta': False, 'X0_std': 0.0,
    #  'lr': 2e-3, 'schedule': 'cosine_decay', 'MLP_d_hidden': 512, 'MLP_num_layers': 4, 'MLP_activation': 'mish',
    #  'bc_scale': 1e-1, 'causal_training': False, 'start_epsilon': 1e-3, 'exp_epsilon': 10, 'delta_epsilon': 0.99},

    # {'loss_method': 'nobiasfbsnn2', 'problem_name': 'BSB', 'd_in': 500, 'bc_name': 'default',
    #  'iter': 100000, 'batch': 256, 'T': 1, 'traj_len': 100, 'use_delta': False, 'X0_std': 0.0,
    #  'lr': 2e-3, 'schedule': 'cosine_decay', 'MLP_d_hidden': 512, 'MLP_num_layers': 4, 'MLP_activation': 'mish',
    #  'bc_scale': 1e-1, 'causal_training': False, 'start_epsilon': 1e-3, 'exp_epsilon': 10, 'delta_epsilon': 0.99},

    # {'loss_method': 'nobiasfbsnn2', 'problem_name': 'BSB', 'd_in': 100, 'bc_name': 'default',
    #  'iter': 100000, 'batch': 256, 'T': 5, 'traj_len': 500, 'use_delta': False, 'X0_std': 0.0,
    #  'lr': 2e-3, 'schedule': 'cosine_decay', 'MLP_d_hidden': 512, 'MLP_num_layers': 4, 'MLP_activation': 'mish',
    #  'bc_scale': 1e-1, 'causal_training': False, 'start_epsilon': 1e-3, 'exp_epsilon': 10, 'delta_epsilon': 0.99},


    
    # {'loss_method': 'nobiasfbsnn3', 'problem_name': 'BSB', 'd_in': 100, 'bc_name': 'default',
    #  'iter': 100000, 'batch': 256, 'T': 1, 'traj_len': 100, 'use_delta': False, 'X0_std': 0.0,
    #  'lr': 2e-3, 'schedule': 'cosine_decay', 'MLP_d_hidden': 512, 'MLP_num_layers': 4, 'MLP_activation': 'mish',
    #  'bc_scale': 1e-1, 'causal_training': False, 'start_epsilon': 1e-3, 'exp_epsilon': 10, 'delta_epsilon': 0.99},

    # {'loss_method': 'nobiasfbsnn3', 'problem_name': 'BSB', 'd_in': 500, 'bc_name': 'default',
    #  'iter': 100000, 'batch': 256, 'T': 1, 'traj_len': 100, 'use_delta': False, 'X0_std': 0.0,
    #  'lr': 2e-3, 'schedule': 'cosine_decay', 'MLP_d_hidden': 512, 'MLP_num_layers': 4, 'MLP_activation': 'mish',
    #  'bc_scale': 1e-1, 'causal_training': False, 'start_epsilon': 1e-3, 'exp_epsilon': 10, 'delta_epsilon': 0.99},

    # {'loss_method': 'nobiasfbsnn3', 'problem_name': 'BSB', 'd_in': 100, 'bc_name': 'default',
    #  'iter': 100000, 'batch': 256, 'T': 5, 'traj_len': 500, 'use_delta': False, 'X0_std': 0.0,
    #  'lr': 2e-3, 'schedule': 'cosine_decay', 'MLP_d_hidden': 512, 'MLP_num_layers': 4, 'MLP_activation': 'mish',
    #  'bc_scale': 1e-1, 'causal_training': False, 'start_epsilon': 1e-3, 'exp_epsilon': 10, 'delta_epsilon': 0.99},

    # ------------------------------
    
    # {'loss_method': 'nobiasfbsnn4', 'problem_name': 'HJB', 'd_in': 100, 'bc_name': 'default',
    #  'iter': 100000, 'batch': 256, 'T': 1, 'traj_len': 100, 'use_delta': False, 'X0_std': 0.0,
    #  'lr': 2e-3, 'schedule': 'cosine_decay', 'MLP_d_hidden': 512, 'MLP_num_layers': 4, 'MLP_activation': 'mish',
    #  'bc_scale': 1e-1, 'causal_training': False, 'start_epsilon': 1e-3, 'exp_epsilon': 10, 'delta_epsilon': 0.99},

    # {'loss_method': 'nobiasfbsnn4', 'problem_name': 'HJB', 'd_in': 500, 'bc_name': 'default',
    #  'iter': 100000, 'batch': 256, 'T': 1, 'traj_len': 100, 'use_delta': False, 'X0_std': 0.0,
    #  'lr': 2e-3, 'schedule': 'cosine_decay', 'MLP_d_hidden': 512, 'MLP_num_layers': 4, 'MLP_activation': 'mish',
    #  'bc_scale': 1e-1, 'causal_training': False, 'start_epsilon': 1e-3, 'exp_epsilon': 10, 'delta_epsilon': 0.99},

    # {'loss_method': 'nobiasfbsnn4', 'problem_name': 'HJB', 'd_in': 100, 'bc_name': 'default',
    #  'iter': 100000, 'batch': 256, 'T': 5, 'traj_len': 500, 'use_delta': False, 'X0_std': 0.0,
    #  'lr': 2e-3, 'schedule': 'cosine_decay', 'MLP_d_hidden': 512, 'MLP_num_layers': 4, 'MLP_activation': 'mish',
    #  'bc_scale': 1e-1, 'causal_training': False, 'start_epsilon': 1e-3, 'exp_epsilon': 10, 'delta_epsilon': 0.99},

    
    
    # {'loss_method': 'nobiasfbsnn5', 'problem_name': 'HJB', 'd_in': 100, 'bc_name': 'default',
    #  'iter': 100000, 'batch': 256, 'T': 1, 'traj_len': 100, 'use_delta': False, 'X0_std': 0.0,
    #  'lr': 2e-3, 'schedule': 'cosine_decay', 'MLP_d_hidden': 512, 'MLP_num_layers': 4, 'MLP_activation': 'mish',
    #  'bc_scale': 1e-1, 'causal_training': False, 'start_epsilon': 1e-3, 'exp_epsilon': 10, 'delta_epsilon': 0.99},

    # {'loss_method': 'nobiasfbsnn5', 'problem_name': 'HJB', 'd_in': 500, 'bc_name': 'default',
    #  'iter': 100000, 'batch': 256, 'T': 1, 'traj_len': 100, 'use_delta': False, 'X0_std': 0.0,
    #  'lr': 2e-3, 'schedule': 'cosine_decay', 'MLP_d_hidden': 512, 'MLP_num_layers': 4, 'MLP_activation': 'mish',
    #  'bc_scale': 1e-1, 'causal_training': False, 'start_epsilon': 1e-3, 'exp_epsilon': 10, 'delta_epsilon': 0.99},

    # {'loss_method': 'nobiasfbsnn5', 'problem_name': 'HJB', 'd_in': 100, 'bc_name': 'default',
    #  'iter': 100000, 'batch': 256, 'T': 5, 'traj_len': 500, 'use_delta': False, 'X0_std': 0.0,
    #  'lr': 2e-3, 'schedule': 'cosine_decay', 'MLP_d_hidden': 512, 'MLP_num_layers': 4, 'MLP_activation': 'mish',
    #  'bc_scale': 1e-1, 'causal_training': False, 'start_epsilon': 1e-3, 'exp_epsilon': 10, 'delta_epsilon': 0.99},

    

    # {'loss_method': 'nobiasfbsnn6', 'problem_name': 'HJB', 'd_in': 100, 'bc_name': 'default',
    #  'iter': 100000, 'batch': 256, 'T': 1, 'traj_len': 100, 'use_delta': False, 'X0_std': 0.0,
    #  'lr': 2e-3, 'schedule': 'cosine_decay', 'MLP_d_hidden': 512, 'MLP_num_layers': 4, 'MLP_activation': 'mish',
    #  'bc_scale': 1e-1, 'causal_training': False, 'start_epsilon': 1e-3, 'exp_epsilon': 10, 'delta_epsilon': 0.99},

    # {'loss_method': 'nobiasfbsnn6', 'problem_name': 'HJB', 'd_in': 500, 'bc_name': 'default',
    #  'iter': 100000, 'batch': 256, 'T': 1, 'traj_len': 100, 'use_delta': False, 'X0_std': 0.0,
    #  'lr': 2e-3, 'schedule': 'cosine_decay', 'MLP_d_hidden': 512, 'MLP_num_layers': 4, 'MLP_activation': 'mish',
    #  'bc_scale': 1e-1, 'causal_training': False, 'start_epsilon': 1e-3, 'exp_epsilon': 10, 'delta_epsilon': 0.99},

    # {'loss_method': 'nobiasfbsnn6', 'problem_name': 'HJB', 'd_in': 100, 'bc_name': 'default',
    #  'iter': 100000, 'batch': 256, 'T': 5, 'traj_len': 500, 'use_delta': False, 'X0_std': 0.0,
    #  'lr': 2e-3, 'schedule': 'cosine_decay', 'MLP_d_hidden': 512, 'MLP_num_layers': 4, 'MLP_activation': 'mish',
    #  'bc_scale': 1e-1, 'causal_training': False, 'start_epsilon': 1e-3, 'exp_epsilon': 10, 'delta_epsilon': 0.99},

    # # ------------------------------
    
    # {'loss_method': 'nobiasfbsnn4', 'problem_name': 'BSB', 'd_in': 100, 'bc_name': 'default',
    #  'iter': 100000, 'batch': 256, 'T': 1, 'traj_len': 100, 'use_delta': False, 'X0_std': 0.0,
    #  'lr': 2e-3, 'schedule': 'cosine_decay', 'MLP_d_hidden': 512, 'MLP_num_layers': 4, 'MLP_activation': 'mish',
    #  'bc_scale': 1e-1, 'causal_training': False, 'start_epsilon': 1e-3, 'exp_epsilon': 10, 'delta_epsilon': 0.99},

    # {'loss_method': 'nobiasfbsnn4', 'problem_name': 'BSB', 'd_in': 500, 'bc_name': 'default',
    #  'iter': 100000, 'batch': 256, 'T': 1, 'traj_len': 100, 'use_delta': False, 'X0_std': 0.0,
    #  'lr': 2e-3, 'schedule': 'cosine_decay', 'MLP_d_hidden': 512, 'MLP_num_layers': 4, 'MLP_activation': 'mish',
    #  'bc_scale': 1e-1, 'causal_training': False, 'start_epsilon': 1e-3, 'exp_epsilon': 10, 'delta_epsilon': 0.99},

    # {'loss_method': 'nobiasfbsnn4', 'problem_name': 'BSB', 'd_in': 100, 'bc_name': 'default',
    #  'iter': 100000, 'batch': 256, 'T': 5, 'traj_len': 500, 'use_delta': False, 'X0_std': 0.0,
    #  'lr': 2e-3, 'schedule': 'cosine_decay', 'MLP_d_hidden': 512, 'MLP_num_layers': 4, 'MLP_activation': 'mish',
    #  'bc_scale': 1e-1, 'causal_training': False, 'start_epsilon': 1e-3, 'exp_epsilon': 10, 'delta_epsilon': 0.99},

    

    # {'loss_method': 'nobiasfbsnn5', 'problem_name': 'BSB', 'd_in': 100, 'bc_name': 'default',
    #  'iter': 100000, 'batch': 256, 'T': 1, 'traj_len': 100, 'use_delta': False, 'X0_std': 0.0,
    #  'lr': 2e-3, 'schedule': 'cosine_decay', 'MLP_d_hidden': 512, 'MLP_num_layers': 4, 'MLP_activation': 'mish',
    #  'bc_scale': 1e-1, 'causal_training': False, 'start_epsilon': 1e-3, 'exp_epsilon': 10, 'delta_epsilon': 0.99},

    # {'loss_method': 'nobiasfbsnn5', 'problem_name': 'BSB', 'd_in': 500, 'bc_name': 'default',
    #  'iter': 100000, 'batch': 256, 'T': 1, 'traj_len': 100, 'use_delta': False, 'X0_std': 0.0,
    #  'lr': 2e-3, 'schedule': 'cosine_decay', 'MLP_d_hidden': 512, 'MLP_num_layers': 4, 'MLP_activation': 'mish',
    #  'bc_scale': 1e-1, 'causal_training': False, 'start_epsilon': 1e-3, 'exp_epsilon': 10, 'delta_epsilon': 0.99},

    # {'loss_method': 'nobiasfbsnn5', 'problem_name': 'BSB', 'd_in': 100, 'bc_name': 'default',
    #  'iter': 100000, 'batch': 256, 'T': 5, 'traj_len': 500, 'use_delta': False, 'X0_std': 0.0,
    #  'lr': 2e-3, 'schedule': 'cosine_decay', 'MLP_d_hidden': 512, 'MLP_num_layers': 4, 'MLP_activation': 'mish',
    #  'bc_scale': 1e-1, 'causal_training': False, 'start_epsilon': 1e-3, 'exp_epsilon': 10, 'delta_epsilon': 0.99},


    
    # {'loss_method': 'nobiasfbsnn6', 'problem_name': 'BSB', 'd_in': 100, 'bc_name': 'default',
    #  'iter': 100000, 'batch': 256, 'T': 1, 'traj_len': 100, 'use_delta': False, 'X0_std': 0.0,
    #  'lr': 2e-3, 'schedule': 'cosine_decay', 'MLP_d_hidden': 512, 'MLP_num_layers': 4, 'MLP_activation': 'mish',
    #  'bc_scale': 1e-1, 'causal_training': False, 'start_epsilon': 1e-3, 'exp_epsilon': 10, 'delta_epsilon': 0.99},

    # {'loss_method': 'nobiasfbsnn6', 'problem_name': 'BSB', 'd_in': 500, 'bc_name': 'default',
    #  'iter': 100000, 'batch': 256, 'T': 1, 'traj_len': 100, 'use_delta': False, 'X0_std': 0.0,
    #  'lr': 2e-3, 'schedule': 'cosine_decay', 'MLP_d_hidden': 512, 'MLP_num_layers': 4, 'MLP_activation': 'mish',
    #  'bc_scale': 1e-1, 'causal_training': False, 'start_epsilon': 1e-3, 'exp_epsilon': 10, 'delta_epsilon': 0.99},

    # {'loss_method': 'nobiasfbsnn6', 'problem_name': 'BSB', 'd_in': 100, 'bc_name': 'default',
    #  'iter': 100000, 'batch': 256, 'T': 5, 'traj_len': 500, 'use_delta': False, 'X0_std': 0.0,
    #  'lr': 2e-3, 'schedule': 'cosine_decay', 'MLP_d_hidden': 512, 'MLP_num_layers': 4, 'MLP_activation': 'mish',
    #  'bc_scale': 1e-1, 'causal_training': False, 'start_epsilon': 1e-3, 'exp_epsilon': 10, 'delta_epsilon': 0.99},

    # ----------------------------
    # ----------------------------
    # ----------------------------

    # {'loss_method': 'shotgun', 'problem_name': 'HJB', 'd_in': 100, 'bc_name': 'default',
    #  'iter': 100000, 'batch': 128, 'T': 1, 'traj_len': 11, 'use_delta': True, 'X0_std': 1e-3,
    #  'lr': 1e-3, 'schedule': 'piecewise_constant', 'boundaries_and_scales': {20000: 0.1, 50000: 0.1, 80000: 0.1},
    #  'MLP_d_hidden': 512, 'MLP_num_layers': 4, 'MLP_activation': 'mish', 'bc_scale': 1, 'causal_training': False},

    # {'loss_method': 'shotgun', 'problem_name': 'HJB', 'd_in': 500, 'bc_name': 'default',
    #  'iter': 100000, 'batch': 128, 'T': 1, 'traj_len': 11, 'use_delta': True, 'X0_std': 1e-3,
    #  'lr': 1e-3, 'schedule': 'piecewise_constant', 'boundaries_and_scales': {20000: 0.1, 50000: 0.1, 80000: 0.1},
    #  'MLP_d_hidden': 512, 'MLP_num_layers': 4, 'MLP_activation': 'mish', 'bc_scale': 1, 'causal_training': False},

    # {'loss_method': 'shotgun', 'problem_name': 'HJB', 'd_in': 100, 'bc_name': 'default',
    #  'iter': 100000, 'batch': 128, 'T': 5, 'traj_len': 51, 'use_delta': True, 'X0_std': 1e-3,
    #  'lr': 1e-3, 'schedule': 'piecewise_constant', 'boundaries_and_scales': {20000: 0.1, 50000: 0.1, 80000: 0.1},
    #  'MLP_d_hidden': 512, 'MLP_num_layers': 4, 'MLP_activation': 'mish', 'bc_scale': 1, 'causal_training': False},
    


    # {'loss_method': 'nobiasshotgun1', 'problem_name': 'HJB', 'd_in': 100, 'bc_name': 'default',
    #  'iter': 100000, 'batch': 128, 'T': 1, 'traj_len': 11, 'use_delta': True, 'X0_std': 1e-3,
    #  'lr': 1e-3, 'schedule': 'piecewise_constant', 'boundaries_and_scales': {20000: 0.1, 50000: 0.1, 80000: 0.1},
    #  'MLP_d_hidden': 512, 'MLP_num_layers': 4, 'MLP_activation': 'mish', 'bc_scale': 1, 'causal_training': False},

    # {'loss_method': 'nobiasshotgun1', 'problem_name': 'HJB', 'd_in': 500, 'bc_name': 'default',
    #  'iter': 100000, 'batch': 128, 'T': 1, 'traj_len': 11, 'use_delta': True, 'X0_std': 1e-3,
    #  'lr': 1e-3, 'schedule': 'piecewise_constant', 'boundaries_and_scales': {20000: 0.1, 50000: 0.1, 80000: 0.1},
    #  'MLP_d_hidden': 512, 'MLP_num_layers': 4, 'MLP_activation': 'mish', 'bc_scale': 1, 'causal_training': False},

    # {'loss_method': 'nobiasshotgun1', 'problem_name': 'HJB', 'd_in': 100, 'bc_name': 'default',
    #  'iter': 100000, 'batch': 128, 'T': 5, 'traj_len': 51, 'use_delta': True, 'X0_std': 1e-3,
    #  'lr': 1e-3, 'schedule': 'piecewise_constant', 'boundaries_and_scales': {20000: 0.1, 50000: 0.1, 80000: 0.1},
    #  'MLP_d_hidden': 512, 'MLP_num_layers': 4, 'MLP_activation': 'mish', 'bc_scale': 1, 'causal_training': False},
    


    {'loss_method': 'nobiasshotgun2', 'problem_name': 'HJB', 'd_in': 100, 'bc_name': 'default',
     'iter': 100000, 'batch': 128, 'T': 1, 'traj_len': 11, 'use_delta': True, 'X0_std': 1e-3,
     'lr': 1e-3, 'schedule': 'piecewise_constant', 'boundaries_and_scales': {20000: 0.1, 50000: 0.1, 80000: 0.1},
     'MLP_d_hidden': 512, 'MLP_num_layers': 4, 'MLP_activation': 'mish', 'bc_scale': 1, 'causal_training': False},

    {'loss_method': 'nobiasshotgun2', 'problem_name': 'HJB', 'd_in': 500, 'bc_name': 'default',
     'iter': 100000, 'batch': 128, 'T': 1, 'traj_len': 11, 'use_delta': True, 'X0_std': 1e-3,
     'lr': 1e-3, 'schedule': 'piecewise_constant', 'boundaries_and_scales': {20000: 0.1, 50000: 0.1, 80000: 0.1},
     'MLP_d_hidden': 512, 'MLP_num_layers': 4, 'MLP_activation': 'mish', 'bc_scale': 1, 'causal_training': False},

    {'loss_method': 'nobiasshotgun2', 'problem_name': 'HJB', 'd_in': 100, 'bc_name': 'default',
     'iter': 100000, 'batch': 128, 'T': 5, 'traj_len': 51, 'use_delta': True, 'X0_std': 1e-3,
     'lr': 1e-3, 'schedule': 'piecewise_constant', 'boundaries_and_scales': {20000: 0.1, 50000: 0.1, 80000: 0.1},
     'MLP_d_hidden': 512, 'MLP_num_layers': 4, 'MLP_activation': 'mish', 'bc_scale': 1, 'causal_training': False},

    

    {'loss_method': 'nobiasshotgun3', 'problem_name': 'HJB', 'd_in': 100, 'bc_name': 'default',
     'iter': 100000, 'batch': 128, 'T': 1, 'traj_len': 11, 'use_delta': True, 'X0_std': 1e-3,
     'lr': 1e-3, 'schedule': 'piecewise_constant', 'boundaries_and_scales': {20000: 0.1, 50000: 0.1, 80000: 0.1},
     'MLP_d_hidden': 512, 'MLP_num_layers': 4, 'MLP_activation': 'mish', 'bc_scale': 1, 'causal_training': False},

    {'loss_method': 'nobiasshotgun3', 'problem_name': 'HJB', 'd_in': 500, 'bc_name': 'default',
     'iter': 100000, 'batch': 128, 'T': 1, 'traj_len': 11, 'use_delta': True, 'X0_std': 1e-3,
     'lr': 1e-3, 'schedule': 'piecewise_constant', 'boundaries_and_scales': {20000: 0.1, 50000: 0.1, 80000: 0.1},
     'MLP_d_hidden': 512, 'MLP_num_layers': 4, 'MLP_activation': 'mish', 'bc_scale': 1, 'causal_training': False},

    {'loss_method': 'nobiasshotgun3', 'problem_name': 'HJB', 'd_in': 100, 'bc_name': 'default',
     'iter': 100000, 'batch': 128, 'T': 5, 'traj_len': 51, 'use_delta': True, 'X0_std': 1e-3,
     'lr': 1e-3, 'schedule': 'piecewise_constant', 'boundaries_and_scales': {20000: 0.1, 50000: 0.1, 80000: 0.1},
     'MLP_d_hidden': 512, 'MLP_num_layers': 4, 'MLP_activation': 'mish', 'bc_scale': 1, 'causal_training': False},


    
    {'loss_method': 'nobiasshotgun4', 'problem_name': 'HJB', 'd_in': 100, 'bc_name': 'default',
     'iter': 100000, 'batch': 128, 'T': 1, 'traj_len': 11, 'use_delta': True, 'X0_std': 1e-3,
     'lr': 1e-3, 'schedule': 'piecewise_constant', 'boundaries_and_scales': {20000: 0.1, 50000: 0.1, 80000: 0.1},
     'MLP_d_hidden': 512, 'MLP_num_layers': 4, 'MLP_activation': 'mish', 'bc_scale': 1, 'causal_training': False},

    {'loss_method': 'nobiasshotgun4', 'problem_name': 'HJB', 'd_in': 500, 'bc_name': 'default',
     'iter': 100000, 'batch': 128, 'T': 1, 'traj_len': 11, 'use_delta': True, 'X0_std': 1e-3,
     'lr': 1e-3, 'schedule': 'piecewise_constant', 'boundaries_and_scales': {20000: 0.1, 50000: 0.1, 80000: 0.1},
     'MLP_d_hidden': 512, 'MLP_num_layers': 4, 'MLP_activation': 'mish', 'bc_scale': 1, 'causal_training': False},

    {'loss_method': 'nobiasshotgun4', 'problem_name': 'HJB', 'd_in': 100, 'bc_name': 'default',
     'iter': 100000, 'batch': 128, 'T': 5, 'traj_len': 51, 'use_delta': True, 'X0_std': 1e-3,
     'lr': 1e-3, 'schedule': 'piecewise_constant', 'boundaries_and_scales': {20000: 0.1, 50000: 0.1, 80000: 0.1},
     'MLP_d_hidden': 512, 'MLP_num_layers': 4, 'MLP_activation': 'mish', 'bc_scale': 1, 'causal_training': False},

    

    {'loss_method': 'nobiasshotgun5', 'problem_name': 'HJB', 'd_in': 100, 'bc_name': 'default',
     'iter': 100000, 'batch': 128, 'T': 1, 'traj_len': 11, 'use_delta': True, 'X0_std': 1e-3,
     'lr': 1e-3, 'schedule': 'piecewise_constant', 'boundaries_and_scales': {20000: 0.1, 50000: 0.1, 80000: 0.1},
     'MLP_d_hidden': 512, 'MLP_num_layers': 4, 'MLP_activation': 'mish', 'bc_scale': 1, 'causal_training': False},

    {'loss_method': 'nobiasshotgun5', 'problem_name': 'HJB', 'd_in': 500, 'bc_name': 'default',
     'iter': 100000, 'batch': 128, 'T': 1, 'traj_len': 11, 'use_delta': True, 'X0_std': 1e-3,
     'lr': 1e-3, 'schedule': 'piecewise_constant', 'boundaries_and_scales': {20000: 0.1, 50000: 0.1, 80000: 0.1},
     'MLP_d_hidden': 512, 'MLP_num_layers': 4, 'MLP_activation': 'mish', 'bc_scale': 1, 'causal_training': False},

    {'loss_method': 'nobiasshotgun5', 'problem_name': 'HJB', 'd_in': 100, 'bc_name': 'default',
     'iter': 100000, 'batch': 128, 'T': 5, 'traj_len': 51, 'use_delta': True, 'X0_std': 1e-3,
     'lr': 1e-3, 'schedule': 'piecewise_constant', 'boundaries_and_scales': {20000: 0.1, 50000: 0.1, 80000: 0.1},
     'MLP_d_hidden': 512, 'MLP_num_layers': 4, 'MLP_activation': 'mish', 'bc_scale': 1, 'causal_training': False},

    # ----------------------------

    # {'loss_method': 'shotgun', 'problem_name': 'BSB', 'd_in': 100, 'bc_name': 'default',
    #  'iter': 100000, 'batch': 128, 'T': 1, 'traj_len': 11, 'use_delta': True, 'X0_std': 0.0, 'use_hard_constraint': True,
    #  'lr': 1e-3, 'schedule': 'piecewise_constant', 'boundaries_and_scales': {20000: 0.1, 50000: 0.1, 80000: 0.1},
    #  'MLP_d_hidden': 512, 'MLP_num_layers': 4, 'MLP_activation': 'mish', 'bc_scale': 1, 'causal_training': False},

    # {'loss_method': 'shotgun', 'problem_name': 'BSB', 'd_in': 500, 'bc_name': 'default',
    #  'iter': 100000, 'batch': 128, 'T': 1, 'traj_len': 11, 'use_delta': True, 'X0_std': 0.0, 'use_hard_constraint': True,
    #  'lr': 1e-3, 'schedule': 'piecewise_constant', 'boundaries_and_scales': {20000: 0.1, 50000: 0.1, 80000: 0.1},
    #  'MLP_d_hidden': 512, 'MLP_num_layers': 4, 'MLP_activation': 'mish', 'bc_scale': 1, 'causal_training': False},

    # {'loss_method': 'shotgun', 'problem_name': 'BSB', 'd_in': 100, 'bc_name': 'default',
    #  'iter': 100000, 'batch': 128, 'T': 5, 'traj_len': 51, 'use_delta': True, 'X0_std': 0.0, 'use_hard_constraint': True,
    #  'lr': 1e-3, 'schedule': 'piecewise_constant', 'boundaries_and_scales': {20000: 0.1, 50000: 0.1, 80000: 0.1},
    #  'MLP_d_hidden': 512, 'MLP_num_layers': 4, 'MLP_activation': 'mish', 'bc_scale': 1, 'causal_training': False},
    


    # {'loss_method': 'nobiasshotgun1', 'problem_name': 'BSB', 'd_in': 100, 'bc_name': 'default',
    #  'iter': 100000, 'batch': 128, 'T': 1, 'traj_len': 11, 'use_delta': True, 'X0_std': 0.0, 'use_hard_constraint': True,
    #  'lr': 1e-3, 'schedule': 'piecewise_constant', 'boundaries_and_scales': {20000: 0.1, 50000: 0.1, 80000: 0.1},
    #  'MLP_d_hidden': 512, 'MLP_num_layers': 4, 'MLP_activation': 'mish', 'bc_scale': 1, 'causal_training': False},

    # {'loss_method': 'nobiasshotgun1', 'problem_name': 'BSB', 'd_in': 500, 'bc_name': 'default',
    #  'iter': 100000, 'batch': 128, 'T': 1, 'traj_len': 11, 'use_delta': True, 'X0_std': 0.0, 'use_hard_constraint': True,
    #  'lr': 1e-3, 'schedule': 'piecewise_constant', 'boundaries_and_scales': {20000: 0.1, 50000: 0.1, 80000: 0.1},
    #  'MLP_d_hidden': 512, 'MLP_num_layers': 4, 'MLP_activation': 'mish', 'bc_scale': 1, 'causal_training': False},

    # {'loss_method': 'nobiasshotgun1', 'problem_name': 'BSB', 'd_in': 100, 'bc_name': 'default',
    #  'iter': 100000, 'batch': 128, 'T': 5, 'traj_len': 51, 'use_delta': True, 'X0_std': 0.0, 'use_hard_constraint': True,
    #  'lr': 1e-3, 'schedule': 'piecewise_constant', 'boundaries_and_scales': {20000: 0.1, 50000: 0.1, 80000: 0.1},
    #  'MLP_d_hidden': 512, 'MLP_num_layers': 4, 'MLP_activation': 'mish', 'bc_scale': 1, 'causal_training': False},
    


    # {'loss_method': 'nobiasshotgun2', 'problem_name': 'BSB', 'd_in': 100, 'bc_name': 'default',
    #  'iter': 100000, 'batch': 128, 'T': 1, 'traj_len': 11, 'use_delta': True, 'X0_std': 0.0, 'use_hard_constraint': True,
    #  'lr': 1e-3, 'schedule': 'piecewise_constant', 'boundaries_and_scales': {20000: 0.1, 50000: 0.1, 80000: 0.1},
    #  'MLP_d_hidden': 512, 'MLP_num_layers': 4, 'MLP_activation': 'mish', 'bc_scale': 1, 'causal_training': False},

    # {'loss_method': 'nobiasshotgun2', 'problem_name': 'BSB', 'd_in': 500, 'bc_name': 'default',
    #  'iter': 100000, 'batch': 128, 'T': 1, 'traj_len': 11, 'use_delta': True, 'X0_std': 0.0, 'use_hard_constraint': True,
    #  'lr': 1e-3, 'schedule': 'piecewise_constant', 'boundaries_and_scales': {20000: 0.1, 50000: 0.1, 80000: 0.1},
    #  'MLP_d_hidden': 512, 'MLP_num_layers': 4, 'MLP_activation': 'mish', 'bc_scale': 1, 'causal_training': False},

    # {'loss_method': 'nobiasshotgun2', 'problem_name': 'BSB', 'd_in': 100, 'bc_name': 'default',
    #  'iter': 100000, 'batch': 128, 'T': 5, 'traj_len': 51, 'use_delta': True, 'X0_std': 0.0, 'use_hard_constraint': True,
    #  'lr': 1e-3, 'schedule': 'piecewise_constant', 'boundaries_and_scales': {20000: 0.1, 50000: 0.1, 80000: 0.1},
    #  'MLP_d_hidden': 512, 'MLP_num_layers': 4, 'MLP_activation': 'mish', 'bc_scale': 1, 'causal_training': False},

    

    # {'loss_method': 'nobiasshotgun3', 'problem_name': 'BSB', 'd_in': 100, 'bc_name': 'default',
    #  'iter': 100000, 'batch': 128, 'T': 1, 'traj_len': 11, 'use_delta': True, 'X0_std': 0.0, 'use_hard_constraint': True,
    #  'lr': 1e-3, 'schedule': 'piecewise_constant', 'boundaries_and_scales': {20000: 0.1, 50000: 0.1, 80000: 0.1},
    #  'MLP_d_hidden': 512, 'MLP_num_layers': 4, 'MLP_activation': 'mish', 'bc_scale': 1, 'causal_training': False},

    # {'loss_method': 'nobiasshotgun3', 'problem_name': 'BSB', 'd_in': 500, 'bc_name': 'default',
    #  'iter': 100000, 'batch': 128, 'T': 1, 'traj_len': 11, 'use_delta': True, 'X0_std': 0.0, 'use_hard_constraint': True,
    #  'lr': 1e-3, 'schedule': 'piecewise_constant', 'boundaries_and_scales': {20000: 0.1, 50000: 0.1, 80000: 0.1},
    #  'MLP_d_hidden': 512, 'MLP_num_layers': 4, 'MLP_activation': 'mish', 'bc_scale': 1, 'causal_training': False},

    # {'loss_method': 'nobiasshotgun3', 'problem_name': 'BSB', 'd_in': 100, 'bc_name': 'default',
    #  'iter': 100000, 'batch': 128, 'T': 5, 'traj_len': 51, 'use_delta': True, 'X0_std': 0.0, 'use_hard_constraint': True,
    #  'lr': 1e-3, 'schedule': 'piecewise_constant', 'boundaries_and_scales': {20000: 0.1, 50000: 0.1, 80000: 0.1},
    #  'MLP_d_hidden': 512, 'MLP_num_layers': 4, 'MLP_activation': 'mish', 'bc_scale': 1, 'causal_training': False},


    
    # {'loss_method': 'nobiasshotgun4', 'problem_name': 'BSB', 'd_in': 100, 'bc_name': 'default',
    #  'iter': 100000, 'batch': 128, 'T': 1, 'traj_len': 11, 'use_delta': True, 'X0_std': 0.0, 'use_hard_constraint': True,
    #  'lr': 1e-3, 'schedule': 'piecewise_constant', 'boundaries_and_scales': {20000: 0.1, 50000: 0.1, 80000: 0.1},
    #  'MLP_d_hidden': 512, 'MLP_num_layers': 4, 'MLP_activation': 'mish', 'bc_scale': 1, 'causal_training': False},

    # {'loss_method': 'nobiasshotgun4', 'problem_name': 'BSB', 'd_in': 500, 'bc_name': 'default',
    #  'iter': 100000, 'batch': 128, 'T': 1, 'traj_len': 11, 'use_delta': True, 'X0_std': 0.0, 'use_hard_constraint': True,
    #  'lr': 1e-3, 'schedule': 'piecewise_constant', 'boundaries_and_scales': {20000: 0.1, 50000: 0.1, 80000: 0.1},
    #  'MLP_d_hidden': 512, 'MLP_num_layers': 4, 'MLP_activation': 'mish', 'bc_scale': 1, 'causal_training': False},

    # {'loss_method': 'nobiasshotgun4', 'problem_name': 'BSB', 'd_in': 100, 'bc_name': 'default',
    #  'iter': 100000, 'batch': 128, 'T': 5, 'traj_len': 51, 'use_delta': True, 'X0_std': 0.0, 'use_hard_constraint': True,
    #  'lr': 1e-3, 'schedule': 'piecewise_constant', 'boundaries_and_scales': {20000: 0.1, 50000: 0.1, 80000: 0.1},
    #  'MLP_d_hidden': 512, 'MLP_num_layers': 4, 'MLP_activation': 'mish', 'bc_scale': 1, 'causal_training': False},

    

    # {'loss_method': 'nobiasshotgun5', 'problem_name': 'BSB', 'd_in': 100, 'bc_name': 'default',
    #  'iter': 100000, 'batch': 128, 'T': 1, 'traj_len': 11, 'use_delta': True, 'X0_std': 0.0, 'use_hard_constraint': True,
    #  'lr': 1e-3, 'schedule': 'piecewise_constant', 'boundaries_and_scales': {20000: 0.1, 50000: 0.1, 80000: 0.1},
    #  'MLP_d_hidden': 512, 'MLP_num_layers': 4, 'MLP_activation': 'mish', 'bc_scale': 1, 'causal_training': False},

    # {'loss_method': 'nobiasshotgun5', 'problem_name': 'BSB', 'd_in': 500, 'bc_name': 'default',
    #  'iter': 100000, 'batch': 128, 'T': 1, 'traj_len': 11, 'use_delta': True, 'X0_std': 0.0, 'use_hard_constraint': True,
    #  'lr': 1e-3, 'schedule': 'piecewise_constant', 'boundaries_and_scales': {20000: 0.1, 50000: 0.1, 80000: 0.1},
    #  'MLP_d_hidden': 512, 'MLP_num_layers': 4, 'MLP_activation': 'mish', 'bc_scale': 1, 'causal_training': False},

    # {'loss_method': 'nobiasshotgun5', 'problem_name': 'BSB', 'd_in': 100, 'bc_name': 'default',
    #  'iter': 100000, 'batch': 128, 'T': 5, 'traj_len': 51, 'use_delta': True, 'X0_std': 0.0, 'use_hard_constraint': True,
    #  'lr': 1e-3, 'schedule': 'piecewise_constant', 'boundaries_and_scales': {20000: 0.1, 50000: 0.1, 80000: 0.1},
    #  'MLP_d_hidden': 512, 'MLP_num_layers': 4, 'MLP_activation': 'mish', 'bc_scale': 1, 'causal_training': False},

]


for setting in changed_settings:

    problem_config.problem_name = setting['problem_name']

    model_config.use_float64 = use_float64

    model_config.d_in = setting['d_in']
    model_config.d_out = 1

    model_config.derivative = 'forward'

    model_config.use_hard_constraint = setting.get('use_hard_constraint', False)

    model_config.T = setting['T']
    model_config.bc_name = setting['bc_name']
    model_config.bc_scale = setting.get('bc_scale', 1.0)

    model_config.MLP_d_hidden = setting['MLP_d_hidden']
    model_config.MLP_num_layers = setting['MLP_num_layers']
    model_config.MLP_activation = setting['MLP_activation']
    model_config.MLP_skip_conn = setting.get('MLP_skip_conn', False)
    model_config.MLP_save_layers = (0,2)
    model_config.MLP_skip_layers = (2,4)
    model_config.MLP_kernel_init = setting.get('kernel_init', 'xavier')

    # --------------------

    solver_config.optim = 'adam'
    solver_config.lr = setting['lr']

    solver_config.iter = setting['iter']

    solver_config.batch = setting['batch']
    # solver_config.micro_batch = setting['micro_batch']

    solver_config.schedule = setting['schedule']  # piecewise_constant, cosine_decay, cosine_onecycle
    solver_config.boundaries_and_scales = setting.get('boundaries_and_scales', {20000: 0.1, 50000: 0.1, 80000: 0.1})

    solver_config.T = setting['T']
    solver_config.traj_len = setting['traj_len']
    solver_config.use_delta = setting['use_delta']

    solver_config.X0_std = setting['X0_std']

    solver_config.loss_method = setting['loss_method']
    solver_config.pde_scale = 1.0

    solver_config.causal_training = setting.get('causal_training', False)
    solver_config.start_epsilon = setting.get('start_epsilon', 1.0)
    solver_config.exp_epsilon = setting.get('exp_epsilon', 2.0)
    solver_config.delta_epsilon = setting.get('delta_epsilon', 0.99)

    solver_config.shotgun_local_batch = 64
    solver_config.shotgun_Delta_t = 4**(-5)

    solver_config.checkpointing = True

    solver_config.test_traj_len = 100
    solver_config.custom_eval = False

    solver_config.save_model = True
    solver_config.save_opt = False
    # solver_config.model_state = './checkpoints/KSIAM_experiment_HJB_lossfbsnnheun_T1_D100_causalFalse_model.msgpack'
    # solver_config.opt_state = 'address'

    # solver_config.project_name = 'final_KSIAM_experiment2'
    solver_config.project_name = 'ICML_experiment'
    # solver_config.run_name = f"{setting['problem_name']}_loss{solver_config.loss_method}_T{solver_config.T}_D{model_config.d_in}_causal{solver_config.causal_training}_hardcon{model_config.use_hard_constraint}"
    solver_config.run_name = f"{setting['problem_name']}_loss{solver_config.loss_method}_T{solver_config.T}_D{model_config.d_in}"
    solver_config.save_to_wandb = True
    solver_config.num_figures = 100
    
    # --------------------

    seed = 20226074
    svr = Solver(model_config, solver_config, problem_config)
    ctr = Controller(svr, seed=seed)
    ctr.solve()

if use_float64:
    jax.config.update("jax_enable_x64", False)