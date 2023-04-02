import numpy as np

# loss function and its derivative
def mse(y_true, y_pred): # mean squared error
    return np.mean(np.power(y_true-y_pred, 2))

def mse_prime(y_true, y_pred): # mean squared error derivative
    return 2*(y_pred-y_true)/y_true.size

def sse(y_true, y_pred): # sum of squared errors
    e= np.sum((y_pred- y_true)**2)
    return e

def sse_prime(y_true, y_pred): #
    return 2*(y_pred-y_true)