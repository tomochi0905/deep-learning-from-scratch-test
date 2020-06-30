import sys, os
sys.path.append(os.pardir)
import numpy as np
from common.functions import softmax, cross_entropy_error
from common.gradient import numerical_gradient

class simpleNet:
    def __init__(self):
        self.W = np.random.randn(2, 3) # ガウス分布で初期化
    
    def predict(self, x):
        return np.dot(x, self.W)
    
    def loss(self, x, t):
        z = self.predict(x)
        y = softmax(z)
        loss = cross_entropy_error(y, t)

        return loss

net = simpleNet()
print(net.W) # 重みパラメータ

x = np.array([0.6, 0.9])
p = net.predict(x)
print(p)
print(np.argmax(p)) # 最大値のインデックス

t = np.array([0, 0, 1]) # 正解ラベル
print(net.loss(x, t))   # 損失関数の値

def f(W):
    return net.loss(x, t)

# 上記関数 f を lambda 式で書いたバージョン
# f = lambda w: net.loss(x, t)

# 勾配を計算
dW = numerical_gradient(f, net.W)
print(dW)