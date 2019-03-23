import numpy as np

def softmax(x, beta):
    return np.exp(x * beta) / np.sum(np.exp(x * beta), axis=0)