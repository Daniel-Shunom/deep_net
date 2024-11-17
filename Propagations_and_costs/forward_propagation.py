import numpy as np
from Parameters.weights_biases import intialize_parameters
from Parameters.activations import tanh

def forward_propagation(X, parameters):
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    Z1 = np.dot(W1,X) + b1
    A1 = tanh(Z1)
    Z2 = np.dot(W1,A1) + b2
    A2 = tanh(Z2)

    cache = {
        "Z1":Z1,
        "A1":A1,
        "Z2":Z2,
        "A2":A2,
    }

    return A2, cache