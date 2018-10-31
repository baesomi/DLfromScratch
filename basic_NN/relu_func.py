import numpy as np
import matplotlib.pylab as plt

def relu(x):
    return np.maximum(0,x)


## 다차원 배열
A= np.array([1,2,3,4])
print(A)


np.ndim(A) # 1
A.shape # (4,)
A.shape[0] # 4

B = np.array([[1,2],[3,4],[5,6]])
print(B)

np.ndim(B) # 2 (차원 수)
B.shape

##  X * W = Y

X = np.array([1,2])
X.shape

W = np.array([[1,3,5],[2,4,6]])
print(W)
W.shape

Y = np.dot(X,W)
print(Y)