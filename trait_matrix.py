import numpy as np
import sys

import utils


def CreateRandomQ(S, U):
  # Fill all rows and columns with at least one 1.
  if S == U:
    Q = np.identity(S)
  else:
    Q = np.zeros((S, U))
    if S < U:
      np.fill_diagonal(np.transpose(Q), 1, wrap=True)
      padc = range(S, U, S + 1)
      padr = np.random.randint(0, S, len(padc))
      Q[padr, padc] = 1
    else:
      np.fill_diagonal(Q, 1, wrap=True)
      padr = range(U, S, U + 1)
      padc = np.random.randint(0, U, len(padr))
      Q[padr, padc] = 1
  num = np.random.randint(0, U * S + 1)
  # Fill matrix.
  for n in range(num):
    i = np.random.randint(0, S)
    j = np.random.randint(0, U)
    Q[i, j] = 1
  return Q.astype(np.int32)


def CreateRankedQ(S, U):
  # Guarantees that Q has maximum rank (== U).
  assert U <= S
  Q = CreateRandomQ(S, U)
  while np.linalg.matrix_rank(Q) != U:
    Q = CreateRandomQ(S, U)
  return Q


if __name__ == '__main__':
  num_species = 20
  sys.stdout.write('Generating random matrix with maximum ranks...\t')
  sys.stdout.flush()
  for num_traits in range(1, num_species + 1):
    Q = CreateRankedQ(num_species, num_traits)
    assert np.linalg.matrix_rank(Q) == num_traits
  sys.stdout.write(utils.Highlight('[DONE]\n', utils.GREEN, bold=True))
