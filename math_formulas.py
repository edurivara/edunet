import numpy as np
# activation functions

def tanh(x):
    return np.tanh(x)

def tanh_prime(x):
    return 1-np.tanh(x)**2

def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_prime(x):
    return np.exp(-x)/(np.exp(-x)+1)**2
    # sigmoid(x) * (1.0 - sigmoid(x))

def relu(x):
    return np.maximum(0., x)

def relu_prime(x):
    return np.where(x > 0, 1.0, 0.0)

# loss functions

def mse(y_true, y_pred):
    return np.mean(np.power((y_pred - y_true),2))

def mse_prime(y_true, y_pred):
    return 2*(y_pred-y_true)/y_true.size
