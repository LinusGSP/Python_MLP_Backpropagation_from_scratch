import numpy as np


def tanh(x): return ((x1 := np.exp(x)) - (x2 := np.exp(-x))) / (x1 + x2)
def tanh_dx(x): return 1 - tanh(x) ** 2
def sigmoid(x): return 1 / (1 + np.exp(-x))
def sigmoid_dx(x): return (x1 := sigmoid(x)) * (1 - x1)
def ReLu(x): return np.where(x < 0, 0, x)
def ReLu_dx(x): return np.where(0 < x, 1, 0)
def LReLu(x): return np.where(0 < x, 0.1 * x, x)
def LReLu_dx(x): return np.where(0 < x, 1, 0.1)
def fast_tanh(x): return x / (1 + abs(x))
def fast_tanh_dx(x): return (1 + abs(x) - (x**2 / abs(x))) / (1 + abs(x))**2

