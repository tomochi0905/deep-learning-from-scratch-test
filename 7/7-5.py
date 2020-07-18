import os, sys
sys.path.append(os.pardir)
import numpy as np
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist
from simple_convnet import SimpleConvNet
from common.trainer import Trainer

# データの読み込み
(x_train, t_train), (x_test, t_test) = load_mnist(flatten=False)

# 処理に時間がかかる場合はデータを削除
# TRAIN_DATA_SIZE = 5000
# TEST_DATA_SIZE = 1000
# x_train, t_train = x_train[:TRAIN_DATA_SIZE], t_train[:TRAIN_DATA_SIZE]
# x_test, t_test = x_test[:TEST_DATA_SIZE], t_test[:TEST_DATA_SIZE]

max_epochs = 20

network = SimpleConvNet(input_dim=(1, 28, 28), conv_param={'filter_num' : 30, 'filter_size' : 5, 'pad' : 0, 'stride' : 1}, hidden_size=100, output_size=10, weight_init_std=0.1)
trainer = Trainer(network, x_train, t_train, x_test, t_test, epochs=max_epochs, mini_batch_size=100, optimizer='Adam', optimizer_param={'lr' : 0.001}, evaluate_sample_num_per_epoch=1000)
trainer.train()

# パラメータの保存
network.save_params("params.pkl")
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