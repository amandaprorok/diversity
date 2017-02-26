import itertools
import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.linalg
import tqdm
import warnings

from mpl_toolkits.mplot3d import Axes3D

import graph
import optimization
import trait_matrix


# Computes V * exp_wt * U.
# By construction the exponential of our matrices are always real-valued.
def Expm(V, exp_wt, U):
    return np.real(V.dot(np.diag(exp_wt)).dot(U))


def ReachabilityConstraint(parameters,
                           Y_desired,
                           A, X_init, Q,
                           specified_time=None,
                           mode=optimization.QUADRATIC_EXACT, margin=None):
  # Sanity checks.
  assert (mode in (optimization.QUADRATIC_EXACT, optimization.ABSOLUTE_EXACT)) == (margin is None)

  # Prepare variable depending on whether t part of the parameters.
  num_nodes = A.shape[0]
  num_species = X_init.shape[1]
  num_traits = Q.shape[1]
  if specified_time is None:
    t = parameters[-1]
    num_parameters_i = (np.size(parameters) - 1) / num_species
  else:
    t = specified_time
    num_parameters_i = np.size(parameters) / num_species

  # Reshape adjacency matrix to make sure.
  Adj = A.astype(float).reshape((num_nodes, num_nodes))
  Adj_flatten = Adj.flatten().astype(bool)  # Flatten boolean version.

  # Loop through the species to compute the cost value.
  # At the same time, prepare the different matrices.
  Ks = []                     # K_s
  eigenvalues = []            # w
  eigenvectors = []           # V.T
  eigenvectors_inverse = []   # U.T
  exponential_wt = []         # exp(eigenvalues * t).
  x_matrix = []               # Pre-computed X matrices.
  x0s = []                    # Avoids reshaping.
  qs = []                     # Avoids reshaping.
  xts = []                    # Keeps x_s(t).
  inside_norm = np.zeros((num_nodes, num_traits))  # Will hold the value prior to using the norm.
  for s in range(num_species):
    x0 = X_init[:, s].reshape((num_nodes, 1))
    q = Q[s, :].reshape((1, num_traits))
    x0s.append(x0)
    qs.append(q)
    k_ij = parameters[s * num_parameters_i:(s + 1) * num_parameters_i]
    # Create K from individual k_{ij}.
    K = np.zeros(Adj_flatten.shape)
    K[Adj_flatten] = k_ij
    K = K.reshape((num_nodes, num_nodes))
    np.fill_diagonal(K, -np.sum(K, axis=0))
    # Store K.
    Ks.append(K)
    # Perform eigen-decomposition to compute matrix exponential.
    w, V = scipy.linalg.eig(K, right=True)
    U = scipy.linalg.inv(V)
    wt = w * t
    exp_wt = np.exp(wt)
    xt = Expm(V, exp_wt, U).dot(x0)
    inside_norm += xt.dot(q)
    # Store the transpose of these matrices for later use.
    eigenvalues.append(w)
    eigenvectors.append(V.T)
    eigenvectors_inverse.append(U.T)
    exponential_wt.append(exp_wt)
    xts.append(xt)
    # Pre-build X matrix.
    with warnings.catch_warnings():
      warnings.simplefilter('ignore', RuntimeWarning)  # We don't care about 0/0 on the diagonal.
      X = np.subtract.outer(exp_wt, exp_wt) / (np.subtract.outer(wt, wt) + 1e-10)
    np.fill_diagonal(X, exp_wt)
    x_matrix.append(X)
  inside_norm -= Y_desired

  # Compute the final cost value depending on mode.
  derivative_outer_norm = None  # Holds the derivative of inside_norm (except the multiplication by (x0 * q)^T).
  if mode == optimization.ABSOLUTE_AT_LEAST:
    derivative_outer_norm = -inside_norm + margin
    value = np.sum(np.maximum(derivative_outer_norm, 0))
    derivative_outer_norm = -(derivative_outer_norm > 0).astype(float)  # Keep only 1s for when it's larger than margin.
  elif mode == optimization.ABSOLUTE_EXACT:
    abs_inside_norm = np.abs(inside_norm)
    index_zeros = abs_inside_norm < 1e-10
    value = np.sum(np.abs(inside_norm))
    with warnings.catch_warnings():
      warnings.simplefilter('ignore', RuntimeWarning)  # We don't care about 0/0.
      derivative_outer_norm = inside_norm / abs_inside_norm  # Keep only 1s for when it's larger than 0 and -1s for when it's lower.
    derivative_outer_norm[index_zeros] = 0  # Make sure we set 0/0 to 0.
  elif mode == optimization.QUADRATIC_AT_LEAST:
    derivative_outer_norm = -inside_norm + margin
    value = np.sum(np.square(np.maximum(derivative_outer_norm, 0)))
    index_negatives = derivative_outer_norm < 0
    derivative_outer_norm *= -2.0
    derivative_outer_norm[index_negatives] = 0  # Don't propagate gradient on negative values.
  elif mode == optimization.QUADRATIC_EXACT:
    value = np.sum(np.square(inside_norm))
    derivative_outer_norm = 2.0 * inside_norm
  return value


def StabilityConstraint(parameters,
                        Y_desired,
                        A, X_init, Q,
                        specified_time=None,
                        nu=1.0):
  # Prepare variable depending on whether t part of the parameters.
  num_nodes = A.shape[0]
  num_species = X_init.shape[1]
  num_traits = Q.shape[1]
  if specified_time is None:
    t = parameters[-1]
    num_parameters_i = (np.size(parameters) - 1) / num_species
  else:
    t = specified_time
    num_parameters_i = np.size(parameters) / num_species

  # Reshape adjacency matrix to make sure.
  Adj = A.astype(float).reshape((num_nodes, num_nodes))
  Adj_flatten = Adj.flatten().astype(bool)  # Flatten boolean version.

  # Loop through the species to compute the cost value.
  # At the same time, prepare the different matrices.
  Ks = []                     # K_s
  eigenvalues = []            # w
  eigenvectors = []           # V.T
  eigenvectors_inverse = []   # U.T
  exponential_wt = []         # exp(eigenvalues * t).
  x_matrix = []               # Pre-computed X matrices.
  x0s = []                    # Avoids reshaping.
  qs = []                     # Avoids reshaping.
  xts = []                    # Keeps x_s(t).
  inside_norm = np.zeros((num_nodes, num_traits))  # Will hold the value prior to using the norm.
  for s in range(num_species):
    x0 = X_init[:, s].reshape((num_nodes, 1))
    q = Q[s, :].reshape((1, num_traits))
    x0s.append(x0)
    qs.append(q)
    k_ij = parameters[s * num_parameters_i:(s + 1) * num_parameters_i]
    # Create K from individual k_{ij}.
    K = np.zeros(Adj_flatten.shape)
    K[Adj_flatten] = k_ij
    K = K.reshape((num_nodes, num_nodes))
    np.fill_diagonal(K, -np.sum(K, axis=0))
    # Store K.
    Ks.append(K)
    # Perform eigen-decomposition to compute matrix exponential.
    w, V = scipy.linalg.eig(K, right=True)
    U = scipy.linalg.inv(V)
    wt = w * t
    exp_wt = np.exp(wt)
    xt = Expm(V, exp_wt, U).dot(x0)
    # Store the transpose of these matrices for later use.
    eigenvalues.append(w)
    eigenvectors.append(V.T)
    eigenvectors_inverse.append(U.T)
    exponential_wt.append(exp_wt)
    xts.append(xt)
    # Pre-build X matrix.
    with warnings.catch_warnings():
      warnings.simplefilter('ignore', RuntimeWarning)  # We don't care about 0/0 on the diagonal.
      X = np.subtract.outer(exp_wt, exp_wt) / (np.subtract.outer(wt, wt) + 1e-10)
    np.fill_diagonal(X, exp_wt)
    x_matrix.append(X)

  # Forcing the steady state.
  # We add a cost for keeping X(t) and X(t + nu) the same. We use the quadratic norm for this sub-cost.
  # The larger beta and the larger nu, the closer to steady state.
  value = 0.
  for s in range(num_species):
    # Compute exp of the eigenvalues of K * (t + nu).
    wtdt = eigenvalues[s] * (t + nu)
    exp_wtdt = np.exp(wtdt)
    # Compute x_s(t) - x_s(t + nu) for that species.
    # Note that since we store V.T and U.T, we do (U.T * D * V.T).T == V * D * U
    inside_norm = xts[s] - Expm(eigenvectors_inverse[s], exp_wtdt, eigenvectors[s]).T.dot(x0s[s])
    # Increment value.
    value += np.sum(np.square(inside_norm))
  return value


def BuildParameters(k1, k2):
  return np.array([k1, k2])


if __name__ == '__main__':
  num_nodes = 2  # DO NOT CHANGE.
  num_traits = 1  # DO NOT CHANGE.
  num_species = 1  # DO NOT CHANGE.

  robots_per_species = 200
  max_rate = 2.
  t = 2.
  num_points = 20

  g = graph.Graph(num_nodes, fully_connected=True)
  X_init = np.zeros((2, 1))
  X_init[0, 0] = int(robots_per_species / 3. * 2.)
  X_init[1, 0] = robots_per_species - X_init[0, 0]
  Q = np.ones((1, 1))
  X_final = np.empty_like(X_init)
  X_final[0, 0] = int(robots_per_species / 3.)
  X_final[1, 0] = robots_per_species - X_final[0, 0]
  Y_desired = X_final.dot(Q)
  A = g.AdjacencyMatrix()

  K1, K2 = np.meshgrid(np.linspace(0, max_rate, num_points), np.linspace(0, max_rate, num_points))
  Z1 = np.empty_like(K1)
  Z2 = np.empty_like(K1)
  for i, j in tqdm.tqdm(itertools.product(range(K1.shape[0]), range(K1.shape[1]))):
      Z1[i, j] = ReachabilityConstraint(BuildParameters(K1[i, j], K2[i, j]), Y_desired,
                                        A, X_init, Q, specified_time=t)
      Z2[i, j] = StabilityConstraint(BuildParameters(K1[i, j], K2[i, j]), Y_desired,
                                     A, X_init, Q, specified_time=t)

  # Draw expected k1 vs. k2 line (that reaches steady state).
  # Since we have 1 species with 1 trait, Y_desired is the expected steady state.
  # So we want: y1 * k2 = y2 * k1 => k2 = y2/y1 * k2.
  k1 = np.linspace(0, max_rate, num_points)
  k2 = Y_desired[1] / Y_desired[0] * k1
  index = np.logical_and(k1 < max_rate, k2 < max_rate)
  k1 = k1[index]
  k2 = k2[index]
  z = np.ones_like(k1) * 0.1

  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')
  ax.plot_surface(K1, K2, Z1, rstride=1, cstride=1, cmap='jet')
  ax.plot(k1, k2, z, lw=2, c='r')
  ax.set_title('Reach')
  ax.set_xlim([0, max_rate])
  ax.set_ylim([0, max_rate])

  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')
  ax.plot_surface(K1, K2, Z2, rstride=1, cstride=1, cmap='jet')
  ax.set_title('Stabilize')
  ax.set_xlim([0, max_rate])
  ax.set_ylim([0, max_rate])

  plt.show()
