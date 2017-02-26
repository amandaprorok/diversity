import matplotlib.pyplot as plt
import numpy as np
import sys
try:
  import tqdm
  has_tqdm = True
except ImportError:
  has_tqdm = False
  pass

import graph
import optimization
import simulation
import trait_matrix
import utils

num_nodes = 8
num_traits = 4
num_species = 4
robots_per_species = 200
max_rate = 2.
num_simulations = 10
num_simulations_rhc = 10
rhc_steps = 10

g = graph.Graph(num_nodes)

X_init = g.CreateRobotDistribution(num_species, robots_per_species, site_restrict=range(0, num_nodes / 2))
Q = trait_matrix.CreateRandomQ(num_species, num_traits)
X_final = g.CreateRobotDistribution(num_species, robots_per_species, site_restrict=range(num_nodes / 2, num_nodes))
Y_desired = X_final.dot(Q)
A = g.AdjacencyMatrix()

sys.stdout.write('Optimizing...\t')
sys.stdout.flush()
K, t, _ = optimization.Optimize(Y_desired, A, X_init, Q, max_rate)
sys.stdout.write(utils.Highlight('[DONE]\n', utils.GREEN, bold=True))

sys.stdout.write('Integrating...\t')
sys.stdout.flush()
times = np.linspace(0., t * 2., 100)
Ys = simulation.ComputeY(times, K, X_init, Q)
error_macro = np.sum(np.abs(Y_desired - Ys), axis=(1, 2)) / (np.sum(Y_desired) * 2.)
sys.stdout.write(utils.Highlight('[DONE]\n', utils.GREEN, bold=True))

error_micro = []
for i in range(num_simulations):
  sys.stdout.write('Simulating (%d/%d)...\t' % (i + 1, num_simulations))
  sys.stdout.flush()
  Ys, timesteps = simulation.SimulateY(np.max(times), K, X_init, Q, dt=0.1 / max_rate)
  error_micro.append(np.sum(np.abs(Y_desired - Ys), axis=(1, 2)) / (np.sum(Y_desired) * 2.))
  sys.stdout.write(utils.Highlight('[DONE]\n', utils.GREEN, bold=True))
error_micro = np.stack(error_micro, axis=0)
mean_error_micro = np.mean(error_micro, axis=0)
std_error_micro = np.std(error_micro, axis=0)

error_rhc = []
for i in range(num_simulations_rhc):
  desc = 'Simulating RHC (%d/%d)' % (i + 1, num_simulations_rhc)
  if not has_tqdm:
    sys.stdout.write(desc + '...\t')
    sys.stdout.flush()
  rhc_dt = np.max(times) / rhc_steps
  X = X_init.copy()
  timesteps_rhc = []
  all_Ys = []
  previous_timestep = 0
  warm_start_params = None
  steps = range(rhc_steps)
  if has_tqdm:
    sys.stdout.write(desc)
    sys.stdout.flush()
    steps = tqdm.tqdm(steps, ncols=50, desc=desc, bar_format='{desc}{percentage:3.0f}% |{bar}|')
  for j in steps:
    K, _, warm_start_params = optimization.Optimize(Y_desired, A, X, Q, max_rate, warm_start_parameters=warm_start_params)
    Xs, ts = simulation.SimulateX(rhc_dt, K, X, dt=0.1 / max_rate)
    X = Xs[-1, :, :]
    if timesteps_rhc:
      ts = ts[1:]
      Xs = Xs[1:, :, :]
    timesteps_rhc.extend(ts + previous_timestep)
    previous_timestep = timesteps_rhc[-1]
    Ys = Xs.dot(Q)
    all_Ys.extend(Ys)
  timesteps_rhc = np.array(timesteps_rhc)
  all_Ys = np.stack(all_Ys, axis=0)
  error_rhc.append(np.sum(np.abs(Y_desired - all_Ys), axis=(1, 2)) / (np.sum(Y_desired) * 2.))
  if not has_tqdm:
    sys.stdout.write(utils.Highlight('[DONE]\n', utils.GREEN, bold=True))
error_rhc = np.stack(error_rhc, axis=0)
mean_error_rhc = np.mean(error_rhc, axis=0)
std_error_rhc = np.std(error_rhc, axis=0)

fig = plt.figure(figsize=(8, 8))
ax1 = plt.subplot2grid((2, 2), (0, 0))
ax2 = plt.subplot2grid((2, 2), (0, 1))
ax3 = plt.subplot2grid((2, 2), (1, 0), colspan=2)

g.Plot(X_init.dot(Q), ax=ax1)
g.Plot(X_final.dot(Q), ax=ax2)
ax3.plot(times, error_macro * 100., lw=2, c='b', label='Macroscopic')
ax3.plot(timesteps, mean_error_micro * 100., c='g', lw=2, label='Microscopic')
ax3.fill_between(timesteps, (mean_error_micro - std_error_micro) * 100., (mean_error_micro + std_error_micro) * 100., color='g', alpha=0.5)
ax3.plot(timesteps_rhc, mean_error_rhc * 100., c='r', lw=2, label='RHC')
ax3.fill_between(timesteps_rhc, (mean_error_rhc - std_error_rhc) * 100., (mean_error_rhc + std_error_rhc) * 100., color='r', alpha=0.5)
ax3.set_xlabel('Time [s]')
ax3.set_ylabel('Misplaced traits [%]')
ax3.set_xlim(left=0., right=np.max(times))
ax3.set_ylim(bottom=0., top=100.)
ax3.grid(True)
plt.legend()

plt.show()
