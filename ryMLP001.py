# %% [markdown]
# <a href="https://colab.research.google.com/github/renyuanL/_JosephLin_2023/blob/main/ryMLP001_py.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# %%
'''
Created on 2023-03-29
Author: Renyuan Lyu
Description: A simple MLP with 2 hidden layers
Task: xor problem
Path: Medium-Python-Neural-Network-master\ryMLP001.py
'''
#%% activation function and its derivative
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

#%% loss function and its derivative (prime)
def mse(y_true, y_pred):
    '''
    mean squared error
    mse(y_true, y_pred) = 1/n * sum((y_true-y_pred)^2)
    '''
    return np.mean(np.power(y_true-y_pred, 2))

def mse_prime(y_true, y_pred):
    '''
    derivative of mse
    mse'(y_true, y_pred) = 2/n * (y_pred-y_true)
    '''
    return 2*(y_pred-y_true)/y_true.size

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

def cross_entropy(y_true, y_pred):
    '''
    cross entropy
    cross_entropy(y_true, y_pred) = -sum(y_true*log(y_pred))
    '''
    return -np.sum(y_true*np.log(y_pred))

def cross_entropy_prime(y_true, y_pred):
    '''
    derivative of cross_entropy
    cross_entropy'(y_true, y_pred) = -y_true/y_pred
    '''
    return -y_true/y_pred

#%%
# Base class
class Layer:
    def __init__(self):
        self.x= None
        self.y= None
    # computes the output y of a layer for a given input x
    def forward(self, x):
        '''
        x: a row vector
        '''
        raise NotImplementedError

    # computes de/dx for a given de/dy (and update parameters if any)
    def backward(self, de_y, eta= .001):
        '''
        de_y= de/dy: error w.r.t. the output y
        eta: learning rate
        '''
        raise NotImplementedError

class ActivationLayer(Layer):
    def __init__(self, f, f_prime):
        self.f= f
        self.f_prime= f_prime

    # returns the activated input
    def forward(self, x):
        self.x= x
        self.y= self.f(x)
        return self.y

    # Returns de/dx for a given de/dy.
    # learning_rate (eta) is not used 
    # because there is no "learnable" parameters.
    def backward(self, de_y, eta= None):
                    
        de_x= de_y * self.f_prime(self.x) 
        # element-wise multiplication 

        return de_x

class FCLayer(Layer):
    def __init__(self, I, J):
        '''
        I: number of input neurons
        J: number of output neurons
        '''
        self.W= np.random.rand(I, J) - 0.5
        self.b= np.random.rand(1, J) - 0.5

    # returns output for a given input
    def forward(self, x):
        '''
        x: a row vector
        '''
        self.x=  x
        self.y= self.x @ self.W + self.b
        return self.y

    # computes de/dW, de/db 
    # for a given output_error= de/dz= (de/dy) (dy/dz). 
    # Returns input_error= de/dx.
    def backward(self, de_y, eta):
        '''
        de_y: de/dy
        eta: learning rate
        '''
        
        de_x= de_y @ self.W.T 
        de_W= self.x.T @ de_y
        de_b= de_y

        # update parameters (gradient descent)
        self.W -= de_W * eta
        self.b -= de_b * eta

        return de_x

# Base class
class Layer_000:
    def __init__(self):
        self.input=  None
        self.output= None

    # computes the output Y of a layer for a given input X
    def forward(self, input):
        raise NotImplementedError

    # computes dE/dX for a given dE/dY (and update parameters if any)
    def backward(self, output_error, learning_rate):
        raise NotImplementedError

# inherit from base class Layer
class FCLayer_000(Layer_000):
    # input_size = number of input neurons
    # output_size = number of output neurons
    def __init__(self, input_size, output_size):
        self.weights= np.random.rand(
                        input_size, output_size
                        ) - 0.5
        self.bias=    np.random.rand(
                        1, output_size
                        ) - 0.5

    # returns output for a given input
    def forward(self, input_data):
        self.input=  input_data # a row vector
        self.output= self.input @ self.weights + self.bias
        return self.output

    # computes de/dW, de/db 
    # for a given output_error= de/dz= (de/dy) (dy/dz). 
    # Returns input_error= de/dx.
    def backward(self, output_error, 
                             learning_rate):
        
        input_error=   output_error @ self.weights.T 
        weights_error= self.input.T @ output_error
        bias_error=    output_error

        # update parameters (gradient descent)
        self.weights -= weights_error * learning_rate
        self.bias    -= bias_error    * learning_rate

        return input_error

# inherit from base class Layer
class ActivationLayer_000(Layer_000):
    def __init__(self, activation, activation_prime):
        self.activation=       activation
        self.activation_prime= activation_prime

    # returns the activated input
    def forward(self, input_data):
        self.input=  input_data
        self.output= self.activation(self.input)
        return self.output

    # Returns input_error= de/dx for a given output_error= de/dy.
    # learning_rate is not used because there is no "learnable" parameters.
    def backward(self, output_error, 
                             learning_rate= None):
        input_error= output_error * self.activation_prime(self.input) 
        # element-wise multiplication 
        
        return input_error



class Network:
    def __init__(self, layers= []):
        self.layers= layers
        self.loss=   None
        self.loss_prime= None

        self.fig= plt.figure()
        self.ax=  self.fig.add_subplot(1,1,1, title= 'network weights')

    # add layer to network
    def add(self, layers):
        self.layers += layers
        return self
    
    def __add__(self, layers):
        return self.add(layers)
    
    def __repr__(self):
        return str(self.layers)
    
    def showLayerParams(self):
        k= 0
        for (l, layer) in enumerate(self.layers):
            if isinstance(layer, FCLayer):
                print(f'{k= }, {l= }\n{layer.W= },\n{layer.b= }\n')
            
                # plot layer weights in line plot
                '''
                self.ax=  self.fig.add_subplot(1,2,k%(1*2)+1, 
                                               title= f'Layer {l= }')
                self.ax.plot(layer.weights)
                '''
                k += 1
                #plt.show()


    # set loss to use
    def setLoss(self, loss, loss_prime):
        self.loss= loss
        self.loss_prime= loss_prime

    # predict output for given input
    def predict(self, input_data):
        # sample dimension first
        samples= len(input_data)
        results= []

        # run network over all samples
        for i in range(samples):
            
            # forward propagation
            output= input_data[i]
            for layer in self.layers:
                output= layer.forward(
                    output
                    )

            results += [output]

        return results

    # train the network
    def _shuffle(self, x_train, y_train):
        # sample dimension first
        samples= len(x_train)
        indices= np.arange(samples)
        np.random.shuffle(indices)

        x_train= x_train[indices]
        y_train= y_train[indices]

        return x_train, y_train
    
    def fit(self, x_train, y_train, epochs, learning_rate):
        # sample dimension first
        samples = len(x_train)

        # training loop
        for i in range(epochs):
            
            x_train, y_train= self._shuffle(x_train, y_train)

            err = 0
            for j in range(samples):
                # forward propagation
                output= x_train[j]
                for layer in self.layers:
                    output= layer.forward(output)

                # compute loss (for display purpose only)
                err += self.loss(y_train[j], output)

                # backward propagation
                error= self.loss_prime(y_train[j], output)
                for layer in reversed(self.layers):
                    error= layer.backward(
                        error, learning_rate
                        )

            # calculate average error on all samples
            err /= samples

            print(f'{i= }, {err= }')

import matplotlib.pyplot as plt

# decision boundary plot
fig= plt.figure()
#ax=  fig.add_subplot(111, projection="3d")

subplots=0

def decision_boundary_plot(net, show= True):
    #global fig, ax
    #ax.clear()
    global subplots
    
    
    points= []
    for x in np.linspace(-1, 1, 20):
        for y in np.linspace(-1, 1, 20):
            z= net.predict([[x, y]])
            points += [(x, y, z[0][0,0])]

    points= np.array(points)

    #fig= plt.figure()
    

    ax= fig.add_subplot(
        3, 4, 
        subplots%(3*4)+1, 
        projection= "3d",
        title= f'epoch {subplots}'
        )
    subplots += 1

    ax.scatter(
        points[:, 0], 
        points[:, 1], 
        points[:, 2], 
        c= points[:, 2], 
        cmap= "rainbow")
    
    ax.view_init(azim= -100, elev= 20)

    if show==True:
        plt.show()
    

def XY_plot(X, Y, show= True):
    '''
    plot X,Y similar to that in decision_boundary_plot
    '''

    points= []
    for x,y in zip(X,Y):
        for i in range(len(x)):
            points += [(x[i][0], x[i][1], y[i][0])]

    points= np.array(points)

    fig= plt.figure()

    ax= fig.add_subplot(
        1, 1, 1, 
        projection= "3d",
        title= 'X,Y plot')
    
    ax.scatter(
        points[:, 0], 
        points[:, 1], 
        points[:, 2], 
        c= points[:, 2], 
        cmap= "rainbow")
    
    ax.view_init(azim= -100, elev= +20)

    if show==True:
        plt.show()





def main():

    # training data
    X= [[[-1, -1]], 
        [[-1, +1]], 
        [[+1, -1]], 
        [[+1, +1]]]
    
    Y= [[[+1]], 
        [[-1]], 
        [[-1]], 
        [[+1]]]
    
    X+=[[[-.1, -.1]], 
        [[-.1, +.1]], 
        [[+.1, -.1]], 
        [[+.1, +.1]]]
    
    Y+=[[[+1]], 
        [[-1]], 
        [[-1]], 
        [[+1]]]
    

    def generate_data(nsamples= 1_000, seed= 0):
        
        np.random.seed(seed)

        X= []
        Y= []
        for i in range(nsamples):
            x= np.random.rand(2) * 2 - 1
            y= np.sign(x[0] * x[1])
            y= np.array([y])
            X += [[x]]
            Y += [[y]]
        
        return X, Y
    
    X,Y= generate_data()

    

    X= np.array(X)
    Y= np.array(Y)

    XY_plot(X, Y, show= False)

    # mlp network

    mlp= Network([
        FCLayer(2,5),  ActivationLayer(tanh,tanh_prime),
        FCLayer(5,1),  ActivationLayer(tanh,tanh_prime)
    ])
    
    mlp.showLayerParams()
    decision_boundary_plot(mlp, show= False)
    
    


    # train (fit)
    mlp.setLoss(mse, mse_prime)
    for i in range(10):
        mlp.fit(
            X, Y, 
            epochs= 20, 
            learning_rate= 0.02)

        decision_boundary_plot(mlp, show= False)
        #mlp.showLayerParams()
    
    mlp.showLayerParams()
    decision_boundary_plot(mlp, show= True)
    

#%%

#from   mpl_toolkits.mplot3d import Axes3D

if __name__ == '__main__':
    main()

#%%