import sys, os
sys.path.append(os.pardir)
import numpy as np
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist
from deep_convnet import DeepConvNet
from common.trainer import Trainer

# データの読み込み
(x_train, t_train), (x_test, t_test) = load_mnist(flatten=False)

# 学習
max_epochs = 20
network = DeepConvNet()
trainer = Trainer(network, x_train, t_train, x_test, t_test, epochs=max_epochs, mini_batch_size=100, optimizer='Adam', optimizer_param={'lr' : 0.001}, evaluate_sample_num_per_epoch=1000)
trainer.train()

# パラメータの保存
network.save_params("deep_convnet_params.pkl")
print("Saved Network Parameters!")

# グラフの描画
markers = {'train' : 'o', 'test' : 's'}
x = np.arange(max_epochs)
plt.plot(x, trainer.train_acc_list, marker=markers['train'], label='train', markevery=2)
plt.plot(x, trainer.test_acc_list, marker=markers['test'], label='test', markevery=2)
plt.xlabel("epoches")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
plt.xlim(0,)
plt.legend(loc='lower right')
plt.show()