# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def ReLU(x):
    return np.maximum(0, x)


def tanh(x):
    return np.tanh(x)


input_data = np.random.randn(1000, 100)  # 1000개의 데이터
node_num = 100  # 각 은닉층의 노드(뉴런) 수
hidden_layer_size = 5  # 은닉층이 5개
activations = {}  # 이곳에 활성화 결과를 저장

x = input_data

for i in range(hidden_layer_size):
    if i != 0:
        x = activations[i - 1]

    # 초깃값을 다양하게 바꿔가며 실험해보자！
    # w = np.random.randn(node_num, node_num) * 1 # 기울기 소실의 문제점
    # w = np.random.randn(node_num, node_num) * 0.01 # 뉴런을 여러개둔 의미가 없음. 0.5에 집중되어있음
    # w = np.random.randn(node_num, node_num) * np.sqrt(1.0 / node_num) #Xavier 초깃값 (활성화 값들을 광범위하게)
    w = np.random.randn(node_num, node_num) * np.sqrt(2.0 / node_num) # He 초깃값

    a = np.dot(x, w)

    # 활성화 함수도 바꿔가며 실험해보자！
    # z = sigmoid(a)
    z = ReLU(a) # He 초깃값 사용해야함
    #z = tanh(a)

    activations[i] = z # 활성화 값 저장

# 히스토그램 그리기
for i, a in activations.items():
    plt.subplot(1, len(activations), i + 1)
    plt.title(str(i + 1) + "-layer")
    if i != 0: plt.yticks([], [])
    # plt.xlim(0.1, 1)
    # plt.ylim(0, 7000)
    plt.hist(a.flatten(), 30, range=(0, 1))
plt.show()