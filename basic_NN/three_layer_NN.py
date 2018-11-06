import numpy as np

from basic_NN.sigmoid_func import sigmoid

X = np.array([1.0, 0.5])
W1 = np.array([[0.1,0.3,0.5], [0.2,0.4,0.6]])
B1 = np.array([0.1,0.2,0.3])

print(W1.shape) # 2,3
print(X.shape) # 2,
print(B1.shape) # 3,

## 0층에서 1층으로
# A = XW + B
A1 = np.dot(X, W1) + B1
print(A1)
# h(A) = Z
Z1 = sigmoid(A1)

print("0 층에서 1층으로")
print("A1 :" + str(A1))
print("Z1 :" + str(Z1))
print("\n")

## 1층에서 2층으로
W2 = np.array([[0.1,0.4], [0.2, 0.5], [0.3,0.6]])
B2 = np.array([0.1,0.2])

print(Z1.shape)
print(W2.shape)
print(B2.shape)

# A2 = Z1W2 + B2
A2 = np.dot(Z1,W2) + B2
# Z2 = h(A2)
Z2 = sigmoid(A2)

print("1 층에서 2층으로")
print("A2 :" + str(A2))
print("Z2 :" + str(Z2))
print("\n")

## 2층에서 출력층으로 : 활성화 함수가 다름!


# 항등함수
def identity_function(x):
    return x



W3 = np.array([[0.1,0.3],[0.2,0.4]])
B3 = np.array([0.1,0.2])

A3 = np.dot(Z2, W3) + B3
Y = identity_function(A3)

print("2 층에서 출력층으로")
print("A3 :" + str(A3))
print("Y :" + str(Y))

