import sys, os
sys.path.append(os.pardir)
import numpy as np
import matplotlib.pyplot as plt
from deep_convnet import DeepConvNet
from dataset.mnist import load_mnist

# データ読み込み
(x_train, t_train), (x_test, t_test) = load_mnist(flatten=False)

# ネットワークの作成
network = DeepConvNet()
network.load_params("deep_convnet_params.pkl") # 学習済み重みパラメータの読み込み

# 高速化のためデータ削減
sampled = 10000
x_test = x_test[:sampled]
t_test = t_test[:sampled]

# float64
print("caluculate accuracy (float64) ... ")
print(network.accuracy(x_test, t_test))

# float16 に変換
x_test = x_test.astype(np.float16)
for param in network.params.values():
    param[...] = param.astype(np.float16)

print("caluculate accuracy (float16) ... ")
print(network.accuracy(x_test, t_test))