import numpy as np
input_layer = null
X = np.random.randn(input_layer.shape[0], input_layer.shape[1])
def layer_size(X, Y):
    n_x = X.shape[0]
    n_h = 6
    n_y = Y.shape[0]

    return (n_x, n_h, n_y)