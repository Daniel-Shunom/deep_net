import numpy as np

def intialize_parameters():
    W1 = np.random.randn(n_h, n_x) * 0.01
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_y, n_x) * 0.01
    b2 = np.zeros((n_y, 1))

    parameters = {
        "W1":W1,
        "b1":b1,
        "W2":W2,
        "b2":b2,
    }