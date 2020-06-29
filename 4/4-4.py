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

# f(x_0, x_1) = x_0^2 + x_1^2
def function_2(x):
    return x[0]**2 + x[1]**2
    # return np.sum(x**2)


print(numerical_gradient(function_2, np.array([3.0, 4.0])))

print(numerical_gradient(function_2, np.array([0.0, 2.0])))

print(numerical_gradient(function_2, np.array([3.0, 0.0])))