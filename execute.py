from problems import *
from solver import Controller
import jax
from datetime import datetime

use_float64 = False
if use_float64:
    jax.config.update('jax_enable_x64', True)

case = 'BSB'
Case_Solver = None
if case == 'HJB':
    Case_Solver = HJB_Solver
elif case == 'BSB':
    Case_Solver = BSB_Solver
elif case == 'BZ':
    Case_Solver = BZ_Solver
else:
    Exception("Invalid Case")

config = Case_Solver.get_base_config()

batch = 128
changed_settings = [[4, 'pinns'], [4, 'fspinns'], [batch, 'bsde'], [batch, 'bsdeskip'], [batch, 'bsdeheun'], [batch, 'bsdeheunnew']]

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
    config.lr = 5e-4
    config.iter = 10000
    config.loss_method = loss_method

    config.schedule = 'cosine_decay'  # piecewise_constant, cosine_decay, cosine_onecycle
    # config.boundaries_and_scales = {}

    config.pde_scale = 1
    config.bc_scale = 1

    config.traj_len = 100
    config.reset_u = True
    config.skip_len = 10

    config.track_pinns_loss = False
    config.track_fspinns_loss = False
    config.track_bsde_loss = False
    config.track_bsde_heun_loss = False

    config.project_name = 'BSDE_32bit'
    config.run_name = f'test_{loss_method}'
    num_figures: int = 20
    config.checkpointing = True

    seed = 20226074
    svr = Case_Solver(config)
    ctr = Controller(svr, seed=seed)
    ctr.solve()

if use_float64:
    jax.config.update("jax_enable_x64", False)