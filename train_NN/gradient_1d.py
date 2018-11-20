import numpy as np
import matplotlib.pylab as plt

# 수치 미분
# 기존의 미분이 아닌, 근사 기울기를 구하는 미분
def numerical_diff(f, x):
    h = 1e-4 # 0.0001
    return print((f(x+h) - f(x-h)) / (2*h))


# 편미분 : 변수가 여럿인 함수에 대한 미분

# 원 식
def function_1(x):
    return 0.01*x**2 + 0.1*x

def function_2(x):
    return np.sum(x**2)

def function_tmp1(x0):
    return x0*x0 + 4.0**2.0

def function_tmp2(x1):
    return 3.0**2.0 + x1*x1


x = np.arange(0.0, 20.0, 0.1) # 0부터 20까지 0.1 간격으로 생성
y = function_1(x)
plt.xlabel("x")
plt.ylabel("f(x)")
plt.plot(x,y)
plt.show()


numerical_diff(function_1, 5)
numerical_diff(function_1, 10)
numerical_diff(function_tmp1, 3.0)
numerical_diff(function_tmp2, 4.0)