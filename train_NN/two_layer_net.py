import sys, os
import numpy as np
sys.path.append(os.pardir)
from train_NN.gradient_2d import numerical_gradient
from basic_NN.sigmoid_func import sigmoid
from basic_NN.softmax_func import softmax,cross_entropy_error


class TwoLayerNet:


    '''
     가중치 초기화
     @:param input_size : 입력층의 뉴런수
     @:param hidden_size : 은닉층의 뉴런수
     @:param output_size : 출력층의 뉴런수
    '''
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        # params : 매개변수를 보관하는 딕셔너리 변수
        self.params = {}
        # 1층의 기울기, 편향
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        # 2층의 기울기, 편향
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)


    def predict(self, x):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']

        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)

        return y

    # 예측결과와 정답 레이블을 바탕으로 교차엔트로피 오차를 구하도록
    def loss(self, x, t):
        y = self.predict(x)
        return cross_entropy_error(y, t)


    def accuracy(self, x, t):
        y = self.predice(x)
        y = np.argmax(y, axis=1)
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





'''
net = TwoLayerNet(input_size=784,  hidden_size=100, output_size=10)

net.params['W1'].shape #(784,100)
net.params['b1'].shape #(100,1)
net.params['W2'].shape #(100,10)
net.params['b2'].shape #(10,)

# 예측 처리
x = np.random.rand(100,784) # 미니배치 100장
y = net.predict(x)

#기울기 구하기

x = np.random.rand(100, 784) # 더미 입력 데이터
t = np.random.rand(100, 10)  # 더미 정답 레이블

grads = net.numerical_gradient(x, t) # 기울기 계산

grads['W1'].shape #(784,100)
grads['b1'].shape #(100,1)
grads['W2'].shape #(100,10)
grads['b2'].shape #(10,)

'''
