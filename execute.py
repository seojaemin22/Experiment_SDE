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
    #  'iter': 20000, 'batch': 100, 'micro_batch': 100, 'T': 1, 'traj_len': 50, 'use_delta': False, 'X0_std': 0.0,
    #  'lr': 1e-3, 'schedule': 'cosine_decay', 'MLP_d_hidden': 512, 'MLP_num_layers': 4, 'MLP_activation': 'mish',
    #  'bc_scale': 1, 'causal_training': False, 'epsilons': [1e-2, 1e-1, 1.0, 10.0, 100.0]},
    
    # {'loss_method': 'fbsnn', 'problem_name': 'HJB', 'd_in': 500, 'bc_name': 'default',
    #  'iter': 20000, 'batch': 100, 'micro_batch': 100, 'T': 1, 'traj_len': 50, 'use_delta': False, 'X0_std': 0.0,
    #  'lr': 1e-3, 'schedule': 'cosine_decay', 'MLP_d_hidden': 512, 'MLP_num_layers': 4, 'MLP_activation': 'mish',
    #  'bc_scale': 1, 'causal_training': False, 'epsilons': [1e-2, 1e-1, 1.0, 10.0, 100.0]},
    
    # {'loss_method': 'fbsnn', 'problem_name': 'HJB', 'd_in': 100, 'bc_name': 'default',
    #  'iter': 20000, 'batch': 100, 'micro_batch': 100, 'T': 5, 'traj_len': 250, 'use_delta': False, 'X0_std': 0.0,
    #  'lr': 1e-3, 'schedule': 'cosine_decay', 'MLP_d_hidden': 512, 'MLP_num_layers': 4, 'MLP_activation': 'mish',
    #  'bc_scale': 1, 'causal_training': False, 'epsilons': [1e-2, 1e-1, 1.0, 10.0, 100.0]},

    #  {'loss_method': 'fbsnn', 'problem_name': 'BSB', 'd_in': 100, 'bc_name': 'default',
    #  'iter': 20000, 'batch': 100, 'micro_batch': 100, 'T': 1, 'traj_len': 50, 'use_delta': False, 'X0_std': 0.0,
    #  'lr': 1e-3, 'schedule': 'cosine_decay', 'MLP_d_hidden': 512, 'MLP_num_layers': 4, 'MLP_activation': 'mish',
    #  'bc_scale': 1, 'causal_training': False, 'epsilons': [1e-2, 1e-1, 1.0, 10.0, 100.0]},
    
    # {'loss_method': 'fbsnn', 'problem_name': 'BSB', 'd_in': 500, 'bc_name': 'default',
    #  'iter': 20000, 'batch': 100, 'micro_batch': 100, 'T': 1, 'traj_len': 50, 'use_delta': False, 'X0_std': 0.0,
    #  'lr': 1e-3, 'schedule': 'cosine_decay', 'MLP_d_hidden': 512, 'MLP_num_layers': 4, 'MLP_activation': 'mish',
    #  'bc_scale': 1, 'causal_training': False, 'epsilons': [1e-2, 1e-1, 1.0, 10.0, 100.0]},
    
    # {'loss_method': 'fbsnn', 'problem_name': 'BSB', 'd_in': 100, 'bc_name': 'default',
    #  'iter': 20000, 'batch': 100, 'micro_batch': 100, 'T': 5, 'traj_len': 250, 'use_delta': False, 'X0_std': 0.0,
    #  'lr': 1e-3, 'schedule': 'cosine_decay', 'MLP_d_hidden': 512, 'MLP_num_layers': 4, 'MLP_activation': 'mish',
    #  'bc_scale': 1, 'causal_training': False, 'epsilons': [1e-2, 1e-1, 1.0, 10.0, 100.0]},

    

    # {'loss_method': 'fbsnn', 'problem_name': 'HJB', 'd_in': 100, 'bc_name': 'default',
    #  'iter': 20000, 'batch': 100, 'micro_batch': 100, 'T': 1, 'traj_len': 50, 'use_delta': False, 'X0_std': 0.0,
    #  'lr': 1e-3, 'schedule': 'cosine_decay', 'MLP_d_hidden': 512, 'MLP_num_layers': 4, 'MLP_activation': 'mish',
    #  'bc_scale': 1, 'causal_training': True, 'epsilons': [1e-2, 1e-1, 1.0, 10.0, 100.0]},
    
    # {'loss_method': 'fbsnn', 'problem_name': 'HJB', 'd_in': 500, 'bc_name': 'default',
    #  'iter': 20000, 'batch': 100, 'micro_batch': 100, 'T': 1, 'traj_len': 50, 'use_delta': False, 'X0_std': 0.0,
    #  'lr': 1e-3, 'schedule': 'cosine_decay', 'MLP_d_hidden': 512, 'MLP_num_layers': 4, 'MLP_activation': 'mish',
    #  'bc_scale': 1, 'causal_training': True, 'epsilons': [1e-2, 1e-1, 1.0, 10.0, 100.0]},
    
    # {'loss_method': 'fbsnn', 'problem_name': 'HJB', 'd_in': 100, 'bc_name': 'default',
    #  'iter': 20000, 'batch': 100, 'micro_batch': 100, 'T': 5, 'traj_len': 250, 'use_delta': False, 'X0_std': 0.0,
    #  'lr': 1e-3, 'schedule': 'cosine_decay', 'MLP_d_hidden': 512, 'MLP_num_layers': 4, 'MLP_activation': 'mish',
    #  'bc_scale': 1, 'causal_training': True, 'epsilons': [1e-2, 1e-1, 1.0, 10.0, 100.0]},

    #  {'loss_method': 'fbsnn', 'problem_name': 'BSB', 'd_in': 100, 'bc_name': 'default',
    #  'iter': 20000, 'batch': 100, 'micro_batch': 100, 'T': 1, 'traj_len': 50, 'use_delta': False, 'X0_std': 0.0,
    #  'lr': 1e-3, 'schedule': 'cosine_decay', 'MLP_d_hidden': 512, 'MLP_num_layers': 4, 'MLP_activation': 'mish',
    #  'bc_scale': 1, 'causal_training': True, 'epsilons': [1e-2, 1e-1, 1.0, 10.0, 100.0]},
    
    # {'loss_method': 'fbsnn', 'problem_name': 'BSB', 'd_in': 500, 'bc_name': 'default',
    #  'iter': 20000, 'batch': 100, 'micro_batch': 100, 'T': 1, 'traj_len': 50, 'use_delta': False, 'X0_std': 0.0,
    #  'lr': 1e-3, 'schedule': 'cosine_decay', 'MLP_d_hidden': 512, 'MLP_num_layers': 4, 'MLP_activation': 'mish',
    #  'bc_scale': 1, 'causal_training': True, 'epsilons': [1e-2, 1e-1, 1.0, 10.0, 100.0]},
    
    # {'loss_method': 'fbsnn', 'problem_name': 'BSB', 'd_in': 100, 'bc_name': 'default',
    #  'iter': 20000, 'batch': 100, 'micro_batch': 100, 'T': 5, 'traj_len': 250, 'use_delta': False, 'X0_std': 0.0,
    #  'lr': 1e-3, 'schedule': 'cosine_decay', 'MLP_d_hidden': 512, 'MLP_num_layers': 4, 'MLP_activation': 'mish',
    #  'bc_scale': 1, 'causal_training': True, 'epsilons': [1e-2, 1e-1, 1.0, 10.0, 100.0]},

    



    # {'loss_method': 'fbsnnheun', 'problem_name': 'HJB', 'd_in': 100, 'bc_name': 'default',
    #  'iter': 20000, 'batch': 100, 'micro_batch': 100, 'T': 1, 'traj_len': 50, 'use_delta': False, 'X0_std': 0.0,
    #  'lr': 1e-3, 'schedule': 'cosine_decay', 'MLP_d_hidden': 512, 'MLP_num_layers': 4, 'MLP_activation': 'mish',
    #  'bc_scale': 1, 'causal_training': False, 'epsilons': [1e-2, 1e-1, 1.0, 10.0, 100.0]},
    
    # {'loss_method': 'fbsnnheun', 'problem_name': 'HJB', 'd_in': 500, 'bc_name': 'default',
    #  'iter': 20000, 'batch': 100, 'micro_batch': 100, 'T': 1, 'traj_len': 50, 'use_delta': False, 'X0_std': 0.0,
    #  'lr': 1e-3, 'schedule': 'cosine_decay', 'MLP_d_hidden': 512, 'MLP_num_layers': 4, 'MLP_activation': 'mish',
    #  'bc_scale': 1, 'causal_training': False, 'epsilons': [1e-2, 1e-1, 1.0, 10.0, 100.0]},
    
    # {'loss_method': 'fbsnnheun', 'problem_name': 'HJB', 'd_in': 100, 'bc_name': 'default',
    #  'iter': 20000, 'batch': 100, 'micro_batch': 100, 'T': 5, 'traj_len': 250, 'use_delta': False, 'X0_std': 0.0,
    #  'lr': 1e-3, 'schedule': 'cosine_decay', 'MLP_d_hidden': 512, 'MLP_num_layers': 4, 'MLP_activation': 'mish',
    #  'bc_scale': 1, 'causal_training': False, 'epsilons': [1e-2, 1e-1, 1.0, 10.0, 100.0]},

    #  {'loss_method': 'fbsnnheun', 'problem_name': 'BSB', 'd_in': 100, 'bc_name': 'default',
    #  'iter': 20000, 'batch': 100, 'micro_batch': 100, 'T': 1, 'traj_len': 50, 'use_delta': False, 'X0_std': 0.0,
    #  'lr': 1e-3, 'schedule': 'cosine_decay', 'MLP_d_hidden': 512, 'MLP_num_layers': 4, 'MLP_activation': 'mish',
    #  'bc_scale': 1, 'causal_training': False, 'epsilons': [1e-2, 1e-1, 1.0, 10.0, 100.0]},
    
    # {'loss_method': 'fbsnnheun', 'problem_name': 'BSB', 'd_in': 500, 'bc_name': 'default',
    #  'iter': 20000, 'batch': 100, 'micro_batch': 100, 'T': 1, 'traj_len': 50, 'use_delta': False, 'X0_std': 0.0,
    #  'lr': 1e-3, 'schedule': 'cosine_decay', 'MLP_d_hidden': 512, 'MLP_num_layers': 4, 'MLP_activation': 'mish',
    #  'bc_scale': 1, 'causal_training': False, 'epsilons': [1e-2, 1e-1, 1.0, 10.0, 100.0]},
    
    # {'loss_method': 'fbsnnheun', 'problem_name': 'BSB', 'd_in': 100, 'bc_name': 'default',
    #  'iter': 20000, 'batch': 100, 'micro_batch': 100, 'T': 5, 'traj_len': 250, 'use_delta': False, 'X0_std': 0.0,
    #  'lr': 1e-3, 'schedule': 'cosine_decay', 'MLP_d_hidden': 512, 'MLP_num_layers': 4, 'MLP_activation': 'mish',
    #  'bc_scale': 1, 'causal_training': False, 'epsilons': [1e-2, 1e-1, 1.0, 10.0, 100.0]},

    

    # {'loss_method': 'fbsnnheun', 'problem_name': 'HJB', 'd_in': 100, 'bc_name': 'default',
    #  'iter': 20000, 'batch': 100, 'micro_batch': 100, 'T': 1, 'traj_len': 50, 'use_delta': False, 'X0_std': 0.0,
    #  'lr': 1e-3, 'schedule': 'cosine_decay', 'MLP_d_hidden': 512, 'MLP_num_layers': 4, 'MLP_activation': 'mish',
    #  'bc_scale': 1, 'causal_training': True, 'epsilons': [1e-2, 1e-1, 1.0, 10.0, 100.0]},
    
    # {'loss_method': 'fbsnnheun', 'problem_name': 'HJB', 'd_in': 500, 'bc_name': 'default',
    #  'iter': 20000, 'batch': 100, 'micro_batch': 100, 'T': 1, 'traj_len': 50, 'use_delta': False, 'X0_std': 0.0,
    #  'lr': 1e-3, 'schedule': 'cosine_decay', 'MLP_d_hidden': 512, 'MLP_num_layers': 4, 'MLP_activation': 'mish',
    #  'bc_scale': 1, 'causal_training': True, 'epsilons': [1e-2, 1e-1, 1.0, 10.0, 100.0]},
    
    # {'loss_method': 'fbsnnheun', 'problem_name': 'HJB', 'd_in': 100, 'bc_name': 'default',
    #  'iter': 20000, 'batch': 100, 'micro_batch': 100, 'T': 5, 'traj_len': 250, 'use_delta': False, 'X0_std': 0.0,
    #  'lr': 1e-3, 'schedule': 'cosine_decay', 'MLP_d_hidden': 512, 'MLP_num_layers': 4, 'MLP_activation': 'mish',
    #  'bc_scale': 1, 'causal_training': True, 'epsilons': [1e-2, 1e-1, 1.0, 10.0, 100.0]},

    #  {'loss_method': 'fbsnnheun', 'problem_name': 'BSB', 'd_in': 100, 'bc_name': 'default',
    #  'iter': 20000, 'batch': 100, 'micro_batch': 100, 'T': 1, 'traj_len': 50, 'use_delta': False, 'X0_std': 0.0,
    #  'lr': 1e-3, 'schedule': 'cosine_decay', 'MLP_d_hidden': 512, 'MLP_num_layers': 4, 'MLP_activation': 'mish',
    #  'bc_scale': 1, 'causal_training': True, 'epsilons': [1e-2, 1e-1, 1.0, 10.0, 100.0]},
    
    # {'loss_method': 'fbsnnheun', 'problem_name': 'BSB', 'd_in': 500, 'bc_name': 'default',
    #  'iter': 20000, 'batch': 100, 'micro_batch': 100, 'T': 1, 'traj_len': 50, 'use_delta': False, 'X0_std': 0.0,
    #  'lr': 1e-3, 'schedule': 'cosine_decay', 'MLP_d_hidden': 512, 'MLP_num_layers': 4, 'MLP_activation': 'mish',
    #  'bc_scale': 1, 'causal_training': True, 'epsilons': [1e-2, 1e-1, 1.0, 10.0, 100.0]},
    
    # {'loss_method': 'fbsnnheun', 'problem_name': 'BSB', 'd_in': 100, 'bc_name': 'default',
    #  'iter': 20000, 'batch': 100, 'micro_batch': 100, 'T': 5, 'traj_len': 250, 'use_delta': False, 'X0_std': 0.0,
    #  'lr': 1e-3, 'schedule': 'cosine_decay', 'MLP_d_hidden': 512, 'MLP_num_layers': 4, 'MLP_activation': 'mish',
    #  'bc_scale': 1, 'causal_training': True, 'epsilons': [1e-2, 1e-1, 1.0, 10.0, 100.0]},
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
    model_config.MLP_skip_conn = False
    # model_config.MLP_save_layers = (0,2,4)
    # model_config.MLP_skip_layers = (2,4,6)
    model_config.MLP_kernel_init = setting.get('kernel_init', 'xavier')

    # --------------------

    solver_config.optim = 'adam'
    solver_config.lr = setting['lr']

    solver_config.iter = setting['iter']

    solver_config.batch = setting['batch']
    solver_config.micro_batch = setting['micro_batch']

    solver_config.schedule = setting['schedule']  # piecewise_constant, cosine_decay, cosine_onecycle
    solver_config.boundaries_and_scales = setting.get('boundaries_and_scales', {20000: 0.1, 50000: 0.1, 80000: 0.1})

    solver_config.T = setting['T']
    solver_config.traj_len = setting['traj_len']
    solver_config.use_delta = setting['use_delta']

    solver_config.X0_std = setting['X0_std']

    solver_config.loss_method = setting['loss_method']
    solver_config.pde_scale = 1.0

    solver_config.causal_training = setting.get('causal_training', False)
    solver_config.epsilons = setting.get('epsilons', [1e-2, 1e-1, 1.0, 10.0, 100.0])
    # solver_config.delta = setting.get('delta', 0.9)

    solver_config.shotgun_local_batch = 64
    solver_config.shotgun_Delta_t = 4**(-5)

    solver_config.checkpointing = True

    solver_config.test_traj_len = 100
    solver_config.custom_eval = False

    solver_config.save_model = True
    solver_config.save_opt = False
    # solver_config.model_state = 'address'
    # solver_config.opt_state = 'address'

    solver_config.project_name = '20251106_meeting'
    solver_config.run_name = f"{setting['problem_name']}_loss{solver_config.loss_method}_T{solver_config.T}_D{model_config.d_in}_causal{solver_config.causal_training}"
    solver_config.save_to_wandb = True
    solver_config.num_figures = 100
    
    # --------------------

    seed = 20226074
    svr = Solver(model_config, solver_config, problem_config)
    ctr = Controller(svr, seed=seed)
    ctr.solve()

if use_float64:
    jax.config.update("jax_enable_x64", False)