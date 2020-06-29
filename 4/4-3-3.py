import numpy as np
import matplotlib.pyplot as plt

# 数値微分の実装（改善版）
def numerical_diff(f, x):
    h = 1e-4 # 0.0001
    return (f(x+h) - f(x - h)) / (2 * h) # 中身差分

# f(x_0, x_1) = x_0^2 + x_1^2
def function_2(x):
    return x[0]**2 + x[1]**2
    # return np.sum(x**2)

def function_tmp1(x0):
    return x0**2.0 + 4.0**2.0

def function_tmp2(x1):
    return 3.0**2.0 + x1**2.0

print(numerical_diff(function_tmp1, 3.0))

print(numerical_diff(function_tmp2, 4.0))