import numpy as np

class Momentum:
    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr # 학습률
        self.momentum = momentum # 운동량
        self.v = None # 속도


    def update(self, params, grads):
        if self.v is None:
            self.v = {}
            for key, val in params.items():
                self.v[key] = np.zeros_like(val)

        for key in params.keys():
            # m*v 는 물체가 아무런 힘을 받지 않을 때 서서히 하강시키는 역할
            self.v[key] = self.momentum*self.v[key] - self.lr*grads[key]
            params[key] += self.v[key]


# 학습률을 조정하면서 갱신
class AdaGrad:
    def __init__(self, lr=0.01):
        self.lr = lr
        self.h = None # 기존 기울기 값을 제곱하여 계속 더해줌

    def update(self, params, grads):
        if self.h is None:
            self.h = {}

            for key, val in params.items():
                self.h[key] = np.zeros_like(val)

        for key in params.keys():
            self.h[key] += grads[key] * grads[key]
            params[key] -= self.lr * grads[key] / (np.sqrt(self.h[key]) + 1e-7) # 1e-7은 h가 0일때를 대비.

