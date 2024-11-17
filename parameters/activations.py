import numpy as np

def sigmoid(z):
    A1 = 1/(1+np.exp(-z))
    return A1

def tanh(z):
    A1 = (np.exp(z)-np.exp(-z))/(np.exp(z)+np.exp(-z))
    return A1

def relu(z):
    A1 = max(0, z)
    return A1

def leaky_relu(z):
    A1 = max((0.001 * z), z)
    return A1