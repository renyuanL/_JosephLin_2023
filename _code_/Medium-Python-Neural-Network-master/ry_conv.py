#%%
import numpy as np
from   scipy import signal

'''
from network            import Network
from conv_layer         import ConvLayer
from activation_layer   import ActivationLayer
from activations        import tanh, tanh_prime
from losses             import mse, mse_prime
'''

#%% activation function and its derivative
def tanh(x):
    return np.tanh(x)

def tanh_prime(x):
    '''
    tanh'(x) = 1 - tanh(x)^2
    '''
    return 1-np.tanh(x)**2



#%% loss function and its derivative
def mse(y_true, y_pred): # mean squared error
    return np.mean(np.power(y_true-y_pred, 2))

def mse_prime(y_true, y_pred): # mean squared error derivative
    return 2*(y_pred-y_true)/y_true.size


#%% CNN Network
class Network:
    def __init__(self):
        self.layers = []
        self.loss = None
        self.loss_prime = None

    # add layer to network
    def add(self, layer):
        self.layers.append(layer)

    # set loss to use
    def use(self, loss, loss_prime):
        self.loss = loss
        self.loss_prime = loss_prime

    # predict output for given input
    def predict(self, input_data):
        # sample dimension first
        samples = len(input_data)
        result = []

        # run network over all samples
        for i in range(samples):
            # forward propagation
            output = input_data[i]
            for layer in self.layers:
                output = layer.forward_propagation(output)
            result.append(output)

        # joint all samples into a single batch
        result= np.array(result)
        result= np.squeeze(result)

        return result

    # train the network
    def fit(self, x_train, y_train, epochs, learning_rate):
        # sample dimension first
        samples = len(x_train)

        # training loop
        for i in range(epochs):
            err = 0
            for j in range(samples):
                # forward propagation
                output = x_train[j]
                for layer in self.layers:
                    output = layer.forward_propagation(output)

                # compute loss (for display purpose only)
                err += self.loss(y_train[j], output)

                # backward propagation
                error = self.loss_prime(y_train[j], output)
                for layer in reversed(self.layers):
                    error = layer.backward_propagation(error, learning_rate)

            # calculate average error on all samples
            err /= samples
            print('epoch %d/%d   error=%f' % (i+1, epochs, err))


# Base class
class Layer:
    def __init__(self):
        self.input = None
        self.output = None

    # computes the output Y of a layer for a given input X
    def forward_propagation(self, input):
        raise NotImplementedError

    # computes dE/dX for a given dE/dY (and update parameters if any)
    def backward_propagation(self, output_error, learning_rate):
        raise NotImplementedError

# inherit from base class Layer
class ActivationLayer(Layer):
    def __init__(self, activation, activation_prime):
        self.activation = activation
        self.activation_prime = activation_prime

    # returns the activated input
    def forward_propagation(self, input_data):
        self.input = input_data
        self.output = self.activation(self.input)
        return self.output

    # Returns input_error=dE/dX for a given output_error=dE/dY.
    # learning_rate is not used because there is no "learnable" parameters.
    def backward_propagation(self, output_error, learning_rate):
        return self.activation_prime(self.input) * output_error

## Math behind this layer can found at : 
## https://medium.com/@2017csm1006/forward-and-backpropagation-in-convolutional-neural-network-4dfa96d7b37e

# inherit from base class Layer
# This convolutional layer is always with stride 1
class ConvLayer(Layer):
    # input_shape = (i,j,d)
    # kernel_shape = (m,n)
    # layer_depth = output_depth
    def __init__(self, 
                 input_shape, 
                 kernel_shape, 
                 layer_depth):
        self.input_shape= input_shape
        
        self.input_depth=  input_shape[2]
        self.kernel_shape= kernel_shape
        self.layer_depth=  layer_depth
        
        self.output_shape= (
            input_shape[0]-kernel_shape[0]+1, #-kernel_shape[0]+1, 
            input_shape[1]-kernel_shape[1]+1, # -kernel_shape[1]+1, 
            layer_depth)
        
        self.weights= np.random.rand(
            kernel_shape[0], 
            kernel_shape[1], 
            self.input_depth, 
            layer_depth) - 0.5
        
        self.bias= np.random.rand(layer_depth) - 0.5

    # returns output for a given input
    def forward_propagation(self, input):
        self.input=  input
        self.output= np.zeros(self.output_shape)

        for k in range(self.layer_depth):
            for d in range(self.input_depth):
                self.output[:,:,k] += signal.correlate2d(
                    self.input[:,:,d], 
                    self.weights[:,:,d,k], 
                    'valid' #'same' #'valid'
                    ) + self.bias[k]

        return self.output

    # computes dE/dW, dE/dB for a given output_error=dE/dY. Returns input_error=dE/dX.
    def backward_propagation(self, 
                             output_error, 
                             learning_rate):
        
        in_error= np.zeros(self.input_shape)
        dWeights= np.zeros((
            self.kernel_shape[0], 
            self.kernel_shape[1], 
            self.input_depth, 
            self.layer_depth))
        dBias=   np.zeros(self.layer_depth)

        for k in range(self.layer_depth):
            for d in range(self.input_depth):
                in_error[:,:,d] += signal.convolve2d(
                    output_error[:,:,k], 
                    self.weights[:,:,d,k], 
                    'full' #'same'#'full'
                    )
                dWeights[:,:,d,k] = signal.correlate2d(
                    self.input[:,:,d], 
                    output_error[:,:,k], 
                    'valid' #'same' #'valid'
                    )
            dBias[k]= self.layer_depth * np.sum(output_error[:,:,k])

        self.weights -= dWeights * learning_rate
        self.bias    -= dBias    * learning_rate
        
        return in_error


# inherit from base class Layer
class FCLayer(Layer):
    # input_size = number of input neurons
    # output_size = number of output neurons
    def __init__(self, input_size, output_size):
        self.weights = np.random.rand(input_size, output_size) - 0.5
        self.bias = np.random.rand(1, output_size) - 0.5

    # returns output for a given input
    def forward_propagation(self, input_data):
        self.input = input_data
        self.output = np.dot(self.input, self.weights) + self.bias
        return self.output

    # computes dE/dW, dE/dB for a given output_error=dE/dY. Returns input_error=dE/dX.
    def backward_propagation(self, output_error, learning_rate):
        input_error = np.dot(output_error, self.weights.T)
        weights_error = np.dot(self.input.T, output_error)
        # dBias = output_error

        # update parameters
        self.weights -= learning_rate * weights_error
        self.bias -= learning_rate * output_error
        return input_error


# inherit from base class Layer
class FlattenLayer(Layer):
    # returns the flattened input
    def forward_propagation(self, 
                            input_data):
        self.input=  input_data
        self.output= input_data.flatten().reshape((1,-1))
        return self.output

    # Returns input_error=dE/dX for a given output_error=dE/dY.
    # learning_rate is not used because there is no "learnable" parameters.
    def backward_propagation(self, 
                             output_error, 
                             learning_rate):
        return output_error.reshape(self.input.shape)




def test_cnn():
    # training data
    x_train= [np.random.rand(10,10,1)]

    # network
    net= Network()

    # (height, width, depth), 
    # (filter_height, filter_width), 
    # num_filters

    net.add(ConvLayer((10,10,1),(3,3), 4));    net.add(ActivationLayer(tanh, tanh_prime))
    net.add(ConvLayer((8,8,4),  (3,3), 5));    net.add(ActivationLayer(tanh, tanh_prime))
    net.add(ConvLayer((6,6,5),  (3,3), 2));    net.add(ActivationLayer(tanh, tanh_prime))

    y_train= [np.random.rand(4,4,2)]

    # test
    out0= net.predict(x_train)
    print("predicted = ", out0)
    print("expected = ", y_train)
    #%%
    # train
    net.use(mse, mse_prime)
    net.fit(
        x_train, 
        y_train, 
        epochs= 1000, 
        learning_rate= 0.1)

    # test
    out= net.predict(x_train)
    print("predicted = ", out)
    print("expected = ", y_train)


from keras.datasets import mnist
from keras.utils    import np_utils

def do_mnist(network='mlp'):

    # load MNIST from server
    (x_train, y_train), (x_test, y_test)= mnist.load_data()

    # training data : 60000 samples
    # reshape and normalize input data 
    x_train = x_train.reshape(x_train.shape[0], 1, 28*28)
    x_train = x_train.astype('float32')
    x_train /= 255

    # encode output which is a number in range [0,9] into a vector of size 10
    # e.g. number 3 will become [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
    y_train = np_utils.to_categorical(y_train)

    # same for test data : 10000 samples
    x_test = x_test.reshape(x_test.shape[0], 1, 28*28)
    x_test = x_test.astype('float32')
    x_test /= 255
    y_test = np_utils.to_categorical(y_test)

    if network=='mlp':
        net= Network()                 
        net.add(FCLayer(28*28*1, 100)); net.add(ActivationLayer(tanh, tanh_prime))
        net.add(FCLayer(100,     10));  net.add(ActivationLayer(tanh, tanh_prime))
    

    elif network=='cnn':
        x_train= x_train.reshape(x_train.shape[0], 28, 28, 1)
        x_test= x_test.reshape(x_test.shape[0], 28, 28, 1)

        net= Network()
        net.add(ConvLayer((28, 28, 1), (3, 3), 5)); net.add(ActivationLayer(tanh, tanh_prime))
        net.add(ConvLayer((26, 26, 5), (3, 3), 2)); net.add(ActivationLayer(tanh, tanh_prime))
        net.add(FlattenLayer())                     
        net.add(FCLayer(24*24*2, 100)); net.add(ActivationLayer(tanh, tanh_prime))
        net.add(FCLayer(100,     10));  net.add(ActivationLayer(tanh, tanh_prime))
    else:
        print(f'{network} not implemented!!')
        return
    


    # train on 1000 samples
    # as we didn't implemented mini-batch GD, training will be pretty slow if we update at each iteration on 60000 samples...
    net.use(mse, mse_prime)
    net.fit(
        x_train[0:1000], 
        y_train[0:1000], 
        epochs= 100, 
        learning_rate= .01)

    # test on 100 samples
    X= x_test[0:100]
    Y= y_test[0:100]

    Y_pred= net.predict(X)
    #Y_pred= Y_pred[0]
    
    '''
    print(f'{X= }')
    print(f'{Y= }')
    print(f'{Y_pred= }')
    '''

    # print predictions
    print("Predicted = ", np.argmax(Y_pred, axis=1))
    print("Expected = ",  np.argmax(Y, axis=1))

    # get accuracy on test data
    accuracy= np.mean(
        np.argmax(Y_pred, axis=1) == np.argmax(Y, axis=1)
        )
    print("Accuracy = ", accuracy)


if __name__ == "__main__":
    test_cnn()
    do_mnist('mlp')
    do_mnist('cnn')

