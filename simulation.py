import numpy as np
import scipy
import scipy.linalg


# Computes V * exp_wt * U.
# By construction the exponential of our matrices are always real-valued.
def Expm(V, exp_wt, U):
    return np.real(V.dot(np.diag(exp_wt)).dot(U))


# Computes the robot distribution after t.
def ComputeX(t_list, K, X_init):
  is_iterable = np.iterable(t_list)
  if not is_iterable:
    t_list = [t_list]
  num_nodes = X_init.shape[0]
  num_species = X_init.shape[1]
  X = np.zeros((len(t_list), num_nodes, num_species))
  # For each species, get transition matrix and calculate final distribution of robots.
  for s in range(num_species):
    Ks = K[:, :, s]
    x0 = X_init[:, s]
    # Perform eigen-decomposition to compute matrix exponential repeatedly.
    w, V = scipy.linalg.eig(Ks, right=True)
    U = scipy.linalg.inv(V)
    for i, t in enumerate(t_list):
      wt = w * t
      exp_wt = np.exp(wt)
      X[i, :, s] = Expm(V, exp_wt, U).dot(x0)
  if is_iterable:
    return X
  return X[0, :, :]


# Computes the trait distribution after t.
def ComputeY(t_list, K, X_init, Q):
  return ComputeX(t_list, K, X_init).dot(Q)


# Simulates random transitions.
def SimulateX(max_time, K, X_init, dt=None):
  num_nodes = X_init.shape[0]
  num_species = X_init.shape[1]
  if dt is None:
    dt = 0.1 / np.max(K)  # Auto-scale step.
  # Pre-compute transition probabilities.
  P = []
  for s in range(num_species):
    Ks = K[:, :, s]
    P.append(scipy.linalg.expm(dt * Ks))

  X = X_init.copy()
  Xs = [X]
  t = 0
  ts = [0]
  while t < max_time:
    new_X = np.zeros_like(X)
    for s in range(num_species):
      for m in range(num_nodes):
        transition_probabilities = P[s][:, m]
        num_robots_in_m = X[m, s]
        choices = np.random.choice(num_nodes, num_robots_in_m, p=transition_probabilities)
        for n in range(num_nodes):
          new_X[n, s] += np.sum(choices == n)
    X = new_X
    Xs.append(X)
    t += dt
    ts.append(t)
  return np.stack(Xs, axis=0), np.array(ts)


def SimulateY(max_time, K, X_init, Q, dt=None):
  X, ts = SimulateX(max_time, K, X_init, dt)
  return X.dot(Q), ts
