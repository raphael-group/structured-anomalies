import csv
import numpy as np
from common import save_matrix


# Example line anomaly

names = list(range(100))
scores = np.random.rand(100)
scores[45:68] += 1

with open('../examples/line_scores.tsv', 'w') as tsvfile:
    writer = csv.writer(tsvfile, delimiter='\t')
    for i in range(len(list1)):
        writer.writerow([list1[i], list2[i]])

# Example submatrix anomaly

A = np.random.rand(10, 10)

for i in range(3):
    for j in range(3):
        A[i][j] += 1
        
save_matrix('../examples/submatrix_scores.hdf5', A)


