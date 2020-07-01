import sys, os
sys.path.append(os.pardir)
import numpy as np
from common.functions import *
from common.gradient import numerical_gradient

class TwoLayerNet:
    # 初期化メソッド
    # input_size        : 入力層のニューロンの数
    # hidden_size       : 隠れ層のニューロンの数
    # output_size       : 出力層のニューロンの数
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        # 重みの初期化
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)
    
    # 認識 (推論) メソッド
    # x : 画像データ
    def predict(self, x):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']

        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)

        return y 
    
    # 損失関数の値の計算メソッド
    # x : 入力データ (画像データ)
    # t : 教師データ (正解ラベル)
    def loss(self, x, t):
        y = self.predict(x)

        return cross_entropy_error(y, t)
    
    # 認識精度の計算メソッド
    # x : 入力データ (画像データ)
    # t : 教師データ (正解ラベル)
    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy
    
    # 重みパラメータに対する勾配の計算メソッド
    # x : 入力データ (画像データ)
    # t : 教師データ (正解ラベル)
    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)

        grads = {}
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])

        return grads

net = TwoLayerNet(input_size=784, hidden_size=100, output_size=10)
print(net.params['W1'].shape)   # (784, 100)
print(net.params['b1'].shape)   # (100,)
print(net.params['W2'].shape)   # (100, 10)
print(net.params['b2'].shape)   # (10,)

x = np.random.rand(100, 784)    # ダミーの入力データ (100 枚分)
t = np.random.rand(100, 10)     # ダミーの正解ラベル (100 枚分)
y = net.predict(x)

grads = net.numerical_gradient(x, t) # 勾配を計算

print(grads['W1'].shape)    # (784, 100)
print(grads['b1'].shape)    # (100,)
print(grads['W2'].shape)    # (100, 10)
print(grads['b2'].shape)    # (10,)
