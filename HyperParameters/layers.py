import numpy as np

def layer_size(X, Y):
    n_x = X.shape[0]
    n_h = 6
    n_y = Y.shape[0]

    return (n_x, n_h, n_y)