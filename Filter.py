import numpy as np
import math

class Filter(object):
    def __init__(self, width, height, depth):
        # self.weights = np.random.uniform(-1e-3, 1e-3, (depth, height, width))
        self.weights = np.random.randn(depth, height, width) * 0.01
        self.bias = 0.0001
        self.weights_grad = np.zeros(self.weights.shape)
        self.bias_grad = 0

    def get_weights(self):
        return self.weights

    def get_bias(self):
        return self.bias

    def update(self, learning_rate):
        self.weights -= learning_rate * self.weights_grad
        self.bias -= learning_rate * self.bias_grad
