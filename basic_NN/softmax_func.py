import numpy as np

'''
def softmax(a):
    exp_a = np.exp(a)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a

    return y
# overflow 문제로 인해 개선해야함
분자,분모에 임의의 상수(입력값 중 최댓값)를 곱해주어 log로 변환
'''

def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a-c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a

    return y


def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    # 훈련 데이터가 원-핫 벡터라면 정답 레이블의 인덱스로 반환
    if t.size == y.size:
        t = t.argmax(axis=1)

    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size


a = np.array([0.3,2.9,4.0])
y = softmax(a)

print(y) # 출력값은 항상 0에서 1.0 사이
print(np.sum(y))# 출력값의 총 합은 항상 1 -> 각 출력 값을 확률로 나타내어 결과를 도출할 수 있음

