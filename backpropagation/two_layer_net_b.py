from backpropagation.layer_naive import *
from train_NN.gradient_2d import numerical_gradient
from collections import OrderedDict


class TwoLayerNetB:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        # params : 매개변수를 보관하는 딕셔너리 변수
        self.params = {}
        # 1층의 기울기, 편향
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        # 2층의 기울기, 편향
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

        # 계층생성
        self.layers = OrderedDict() # 만들어진 계층 순서대로 호출되도록
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        self.layers['Relu1'] = Relu()

        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])
        self.lastLayer = SoftmaxWithLoss()

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)

        return x

    def loss(self, x, t):
        y = self.predict(x)
        return self.lastLayer.forward(y, t)


    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1:
            t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) / float(x.shape[0])

        return accuracy

    # x : 입력데이터, t: 정답 레이블
    # 수치 미분 방식
    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)
        # 기울기를 보관하는 딕셔너리 변수 (numerical_gradient 반환값)
        grads = {}
        # 1번째 층의 가중치의 기울기
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        # 1번째 층의 편향의 기울기
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        # 2번째 층의 가중치의 기울기
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        # 2번째 층의 편향의 기울기
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])

        return grads

    # 기울기
    def gradient(self, x, t):
        # 순전파
        self.loss(x,t)

        # 역전파
        dout = 1
        dout = self.lastLayer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        # 결과 저장
        grads = {}
        grads['W1'] = self.layers['Affine1'].dW
        grads['b1'] = self.layers['Affine1'].db
        grads['W2'] = self.layers['Affine2'].dW
        grads['b2'] = self.layers['Affine2'].db

        return grads
