import sys, os
sys.path.append(os.pardir)
import numpy as np
from common.layers import *
from common.gradient import numerical_gradient
from collections import OrderedDict

class TwoLayerNet:
    # 初期化メソッド
    # input_size        : 入力層のニューロンの数
    # hidden_size       : 隠れ層のニューロンの数
    # output_size       : 出力層のニューロンの数
    # weight_init_std   : 重み初期化時のガウス分布のスケール
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        # 重みの初期化
        self.params = {} # ニューラルネットワークのパラメータを保持するディクショナリ変数
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

        # レイヤの生成
        self.layers = OrderedDict() # ニューラルネットワークのレイヤを保持する順番付きディクショナリ変数
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        self.layers['Relu1'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])

        self.lastLayer = SoftmaxWithLoss() # ニューラルネットワークの最後のレイヤ (この例では SoftmaxWithLoss レイヤ)
    
    # 認識 (推論) メソッド
    # x : 画像データ
    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        
        return x
    
    # 損失関数の値の計算メソッド
    # x : 入力データ (画像データ)
    # t : 教師データ (正解ラベル)
    def loss(self, x, t):
        y = self.predict(x)
        return self.lastLayer.forward(y, t)
    
    # 認識精度の計算メソッド
    # x : 入力データ (画像データ)
    # t : 教師データ (正解ラベル)
    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1 : t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy
    
    # 重みパラメータに対する勾配の計算メソッド (数値微分による算出)
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
    
    # 重みパラメータに対する勾配の計算メソッド (誤差逆伝播法による算出)
    # x : 入力データ (画像データ)
    # t : 教師データ (正解ラベル) 
    def gradient(self, x, t):
        # forward
        self.loss(x, t)

        # backward
        dout = 1
        dout = self.lastLayer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)
        
        # 設定
        grads = {}
        grads['W1'] = self.layers['Affine1'].dW
        grads['b1'] = self.layers['Affine1'].db
        grads['W2'] = self.layers['Affine2'].dW
        grads['b2'] = self.layers['Affine1'].db

        return grads