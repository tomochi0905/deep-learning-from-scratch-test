import numpy as np
import matplotlib.pyplot as plt

# 勾配の計算
def numerical_gradient(f, x):
    h = 1e-4 # 0.0001
    grad = np.zeros_like(x) # x と同じ形状の配列を生成

    for idx in range(x.size):
        tmp_val = x[idx]
        # f(x+h) の計算
        x[idx] = tmp_val + h
        fxh1 = f(x)

        # f(x-h) の計算
        x[idx] = tmp_val - h
        fxh2 = f(x)

        grad[idx] = (fxh1 - fxh2) / (2*h)
        x[idx] = tmp_val # 値を元に戻す
    
    return grad

# 購買降下法 (f : 最適化したい関数, init_x : 初期値, lr : 学習率, step_num : 勾配法による繰り返し回数)
def gradient_descent(f, init_x, lr=0.01, step_num=100):
    x = init_x

    for i in range(step_num):
        grad = numerical_gradient(f, x)
        x -= lr * grad
    
    return x

# f(x_0, x_1) = x_0^2 + x_1^2
def function_2(x):
    return x[0]**2 + x[1]**2
    # return np.sum(x**2)

init_x = np.array([-3.0, 4.0])
print(gradient_descent(function_2, init_x=init_x, lr=0.1, step_num=100))
# 最終結果 : [-6.11110793e-10  8.14814391e-10]
# 真の最小値 : [0, 0] にかなり近い値になる

# 学習率が大きすぎる例 : lr=10.0
init_x = np.array([-3.0, 4.0])
print(gradient_descent(function_2, init_x=init_x, lr=10.0, step_num=100))
# 最終結果 : [-2.58983747e+13 -1.29524862e+12]

# 学習率が大きすぎる例 : lr=1e-10
init_x = np.array([-3.0, 4.0])
print(gradient_descent(function_2, init_x=init_x, lr=1e-10, step_num=100))
# 最終結果 : [-2.99999994  3.99999992]