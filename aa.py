import numpy as np

C = np.array([1, 2, 3])
print(C.shape)
print(C)
C = C.reshape(1, 3)
print(C.shape)
print(C)


D = np.array([[1, 2, 3], [4, 5, 6]])
print(D.shape)

print(D)

D = D.reshape(3, 2)
print(D.shape)


print(D)
