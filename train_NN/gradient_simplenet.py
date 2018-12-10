import sys, os
import numpy as np
from basic_NN.softmax_func import softmax,cross_entropy_error
from train_NN.gradient_2d import numerical_gradient
sys.path.append(os.pardir)


class simple_net:
    def __init__(self):
        self.W = np.random.randn(2,3) #정규분포 초기화

    def predict(self, x):
        return np.dot(x, self.W)

    def loss(self, x, t):
        z = self.predict(x)
        y = softmax(z)
        loss = cross_entropy_error(y,t)

        return loss


    x = np.array([0.6, 0.9])
    t = np.array([0, 0, 1])

    net = simple_net()

    f = lambda w: net.loss(x, t)
    dW = numerical_gradient(f, net.W)

    print(dW)