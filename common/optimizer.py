import numpy as np

# 確率的勾配降下法 (Stochastic Gradient Descent : SGD)
class SGD:
    def __init__(self, lr=0.01):
        self.lr = lr
    
    def update(self, params, grads):
        for key in params.key():
            params[key] -= self.lr * grads[key]