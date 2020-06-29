import numpy as np

# 数値微分の悪い実装例
# def numerical_diff(f, x):
#     h = 1e-50
#     return (f(x+h) - f(x)) / h # 前方差分

# 数値微分の実装（改善版）
def numerical_diff(f, x):
    h = 1e-4 # 0.0001
    return (f(x+h) - f(x - h)) / (2 * h) # 中身差分