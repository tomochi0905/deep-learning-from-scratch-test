import numpy as np
import matplotlib.pyplot as plt

# def step_function(x):
#     y = x > 0
#     return y.astype(np.int)

def sigmoid(x):
    return 1 / (1 + np.exp(-x)) 

x = np.arange(-5.0, 5.0, 0.1)
y = sigmoid(x)
plt.plot(x, y)
plt.ylim(-0.1, 1.1)
plt.show()