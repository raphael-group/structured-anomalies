from common import save_matrix
import numpy as np

A = np.random.rand(10, 10)

for i in range(3):
    for j in range(3):
        A[i][j] += 1
        
save_matrix('../examples/submatrix_scores.hdf5', A)