import numpy as np
import matplotlib.pyplot as plt

def step_function(x):
    y = x > 0
    return y.astype(np.int)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

x = np.arange(-5.0, 5.0, 0.1)
y1 = step_function(x)
y2 = sigmoid(x)

plt.plot(x, y1, label="step_function", linestyle="--")
plt.plot(x, y2, label="sigmoid")
plt.xlabel("x")
plt.ylabel("y")
plt.title("step_function & sigmoid")
plt.ylim(-0.1, 1.1)
plt.legend()

plt.show()