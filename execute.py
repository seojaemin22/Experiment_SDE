from problems import *
from solver import *
import jax
from datetime import datetime

use_float64 = False
if use_float64:
    jax.config.update('jax_enable_x64', True)

case = 'HD_PIDE'
# output_type = 'FO'
output_type = 'None'

Case_Solver = None
if case == 'HJB':
    if output_type == 'FO':
        Case_Solver = HJB_FO_Solver
        Case_Controller = PDE_FO_Controller
    else:
        Case_Solver = HJB_Solver
        Case_Controller = PDE_Controller
elif case == 'BSB':
    if output_type == 'FO':
        Case_Solver = BSB_FO_Solver
        Case_Controller = PDE_FO_Controller
    else:
        Case_Solver = BSB_Solver
        Case_Controller = PDE_FO_Controller
elif case == 'BZ':
    Case_Solver = BZ_Solver
    Case_Controller = PDE_Controller
elif case == 'HD_PIDE':
    Case_Solver = HD_PIDE_Solver
    Case_Controller = PIDE_Controller
else:
    Exception("Invalid Case")

config = Case_Solver.get_base_config()

batch = 128
changed_settings = [[batch, 'bsde']]

for micro_batch, loss_method in changed_settings:
    config.d_in = 100
    config.d_hidden = 256
    config.num_layers = 5
    config.activation = 'sin'
    config.four_emb = False
    config.skip_conn = True
    config.save_layers = (0,2)
    config.skip_layers = (2,4)

    config.batch = batch
    config.micro_batch = micro_batch
    config.optim = 'adam'
    # config.lr = 5e-4
    config.lr = 5e-4
    config.iter = 30000
    config.loss_method = loss_method

    config.schedule = 'cosine_decay'  # piecewise_constant, cosine_decay, cosine_onecycle
    # config.boundaries_and_scales = {}

    config.pde_scale = 1
    config.bc_scale = 1

    config.traj_len = 50
    config.dt = 1/config.traj_len
    config.reset_u = True
    config.skip_len = 10

    config.save_model = True
    config.save_opt = False
    # config.model_state = './checkpoints/base_bsde_model.msgpack'
    # config.opt_state = 'address'

    # config.track_pinns_loss = False
    # config.track_fspinns_loss = False
    # config.track_bsde_loss = True
    # config.track_bsde_heun_loss = False

    config.project_name = 'HD_PIDE'
    config.run_name = f'5_skip_sin'
    # config.run_name = f'error_test_{loss_method}_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    num_figures: int = 20
    config.checkpointing = True

    seed = 20226074
    svr = Case_Solver(config)
    ctr = Case_Controller(svr, seed=seed)
    ctr.solve()

if use_float64:
    jax.config.update("jax_enable_x64", False)