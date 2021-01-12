import numpy as np

a = np.array([1, 2, 3, 4]).reshape((2, 2))
print(a)
b = np.array([0, 1])
print(np.arange(a.shape[0]))
print(a[[0, 1], [0, 1]])
