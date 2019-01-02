import numpy as np
from basic_NN.softmax_func import *
### 곱셈 계층
class MulLayer:
    def __init__(self):
        self.x = None
        self.y = None


    def forward(self, x, y):
        self.x = x
        self.y = y
        out = x * y

        return out

    def backward(self, dout):
        dx = dout * self.y #x와 y를 바꾼다
        dy = dout * self.x

        return dx, dy

### 덧셈 계층
class AddLayer:
    def __init__(self):
        pass

    def forward(self, x, y):
       out = x + y
       return out


    def backward(self, dout):
        dx = dout * 1
        dy = dout * 1
        return dx, dy


### Relu 계층
class Relu:
    def __init__(self):
        # True / False로 구성된 넘파이 배열로 x값이 0이하면 true 0이상은 false임
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0

        return out


    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout

        return dx


class Sigmoid:
    def __init__(self):
        self.out = None

    def forward(self, x):
        out = 1 / (1 + np.exp(-x))
        self.out = out

        return out

    def backward(self, dout):
        dx = dout * (1.0 - self.out) * self.out

        return dx


# 배치용 affine 계층
class Affine:
    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.x = None
        self.dW = None
        self.db = None


    def forward(self, x):
        self.x = x
        out = np.dot(x, self.W) + self.b

        return out

    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dw = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)

        return dx

# Softmax with loss 계층
class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None # 손실
        self.y = None # softmax의 출력
        self.t = None # 정답 레이블(원-핫 벡터)

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)

        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        dx = (self.y - self.t) / batch_size

        return dx