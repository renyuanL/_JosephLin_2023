'''
Created on 2023-03-29, 04-03
Author: Renyuan Lyu
Description: A simple MLP 
Task: xor problem
Path: ryMLP002.py
'''
#%% Activation function and its derivative
import numpy as np

def tanh(x):
    '''
    hyperbolic tangent activation function
    tanh(x) = (exp(x)-exp(-x))/(exp(x)+exp(-x))
    '''
    return np.tanh(x)

def tanh_prime(x):
    '''
    derivative of tanh
    tanh'(x) = 1 - tanh(x)^2
    '''
    return 1-tanh(x)**2

def sigmoid(x):
    '''
    sigmoid activation function
    sigmoid(x) = 1/(1+exp(-x))
    '''
    return 1/(1+np.exp(-x))

def sigmoid_prime(x):
    '''
    derivative of sigmoid
    sigmoid'(x) = sigmoid(x)*(1-sigmoid(x))
    '''
    return sigmoid(x)*(1-sigmoid(x))

#%% Loss function and its derivative (prime)

def sse(y_true, y_pred):
    '''
    sum squared error
    sse(y_true, y_pred) = sum((y_true-y_pred)^2)
    '''
    return np.sum((y_true-y_pred)**2)

def sse_prime(y_true, y_pred):
    '''
    derivative of sse
    sse'(y_true, y_pred) = 2 * (y_pred-y_true)
    '''
    return 2*(y_pred-y_true)


#%% Layer
class Layer:
    def __init__(self):
        self.x= None
        self.y= None
    def forward(self, x):
        raise NotImplementedError
    def backward(self, de_y, eta= .001):
        raise NotImplementedError

class ActivationLayer(Layer):
    def __init__(self, f, f_prime):
        self.f= f
        self.f_prime= f_prime

    def forward(self, z):
        self.z= z
        self.y= self.f(z)
        return self.y
    
    def backward(self, de_y, eta= None):
        
        # de_z= de_y * self.f_prime(self.z)
        
        # element-wise multiplication 
        # is equivalent to 
        # matrix multiplication with a diagonal matrix
        # which is faster?
        # https://stackoverflow.com/questions/32109319/how-to-efficiently-calculate-a-hadamard-product-in-numpy
        

        de_z= de_y @ np.diag(
            self.f_prime(self.y))

        return de_z

class FCLayer(Layer):
    def __init__(self, I, J):
        '''
        I: number of input neurons
        J: number of output neurons
        '''
        # initialize weights and bias, 
        # with random values in [-0.5, 0.5]
        self.W= np.random.rand(I, J) - 0.5
        self.b= np.random.rand(1, J) - 0.5

    def forward(self, x):
        '''
        x: input,  shape=(N, I)
        z: output, shape=(N, J)
        '''
        self.x=  x
        self.z= self.x @ self.W + self.b
        return self.z
    
    def backward(self, de_y, eta):
        '''
        de_y: derivative of loss function w.r.t. y
        eta: learning rate
        de_x: derivative of loss function w.r.t. x
        de_W: derivative of loss function w.r.t. W
        de_b: derivative of loss function w.r.t. b
        '''
        de_x= de_y @ self.W.T 
        de_W= self.x.T @ de_y
        de_b= de_y

        self.W -= de_W * eta

        #self.b -= de_b * eta # this is wrong, why?
        self.b= self.b - de_b * eta # this is correct, why?

        return de_x

#%% MLP
class MLP:
    def __init__(self, loss, loss_prime, layers= []):
        '''
        loss: loss function
        loss_prime: derivative of loss function
        layers: list of layers

        example:
        model= MLP(
            sse, sse_prime,
            layers= [
                FCLayer(2, 3), ActivationLayer(tanh, tanh_prime),
                FCLayer(3, 1), ActivationLayer(tanh, tanh_prime)
            ])
        '''
        self.layers= layers
        self.loss= loss
        self.loss_prime= loss_prime
    def add(self, layer):
        self.layers.append(layer)
    def forward(self, x):
        for layer in self.layers:
            x= layer.forward(x)
        return x
    def backward(self, de_y, eta):
        for layer in reversed(self.layers):
            de_y= layer.backward(de_y, eta)
    def train(self, x, y_true, eta= .001, epochs= 1000):
        for epoch in range(epochs):
            y_pred= self.forward(x)
            loss= self.loss(y_true, y_pred)
            de_y= self.loss_prime(y_true, y_pred)
            self.backward(de_y, eta)
            if epoch % 100 == 0:
                print(f'{epoch= }, {loss= }')
    def predict(self, x):
        return self.forward(x)
    
#%% Main

def do_xor():
    #%% Data
    x= np.array([
        [+1, +1], 
        [+1, -1], 
        [-1, +1], 
        [-1, -1]])
    y_true= np.array([
        [+1], 
        [-1], 
        [-1], 
        [+1]])
    
    #%% Model
    model= MLP(
        sse, sse_prime,
        layers= [
            FCLayer(2, 4), ActivationLayer(tanh, tanh_prime),
            FCLayer(4, 3), ActivationLayer(tanh, tanh_prime),
            FCLayer(3, 1), ActivationLayer(tanh, tanh_prime)
        ])
    
    #%% Predict, before train
    y_pred= model.predict(x)
    print(f'{y_pred= }')
    print(f'{y_true= }')
    
    #%% Train
    model.train(
        x, 
        y_true, 
        eta= .01, 
        epochs= 1000)
    
    #%% Predict, after train
    y_pred= model.predict(x)
    print(f'{y_pred= }')
    print(f'{y_true= }')
    

if __name__ == '__main__':
    do_xor()
