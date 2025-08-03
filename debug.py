import numpy as np

matrix = np.random.rand(2,2)
print(matrix)

row_sums = np.sum(matrix, axis=1)
sums_arr = row_sums[:, np.newaxis]
div_matrix = matrix / sums_arr

print(div_matrix)