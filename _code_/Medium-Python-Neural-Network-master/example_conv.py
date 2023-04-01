#%%
import numpy as np

from network import Network
from conv_layer import ConvLayer
from activation_layer import ActivationLayer
from activations import * #tanh, tanh_prime
from losses import mse, mse_prime

# training data
x_train= [np.random.rand(10,10,1)]


# network
net= Network()

# (height, width, depth), 
# (filter_height, filter_width), 
# num_filters

net.add(ConvLayer((10,10,1), (3,3), 4));   
net.add(ActivationLayer(tanh, tanh_prime))

net.add(ConvLayer((8,8,4),  (3,3), 5));   
net.add(ActivationLayer(tanh, tanh_prime))

net.add(ConvLayer((6,6,5),  (3,3), 2));   
net.add(ActivationLayer(tanh, tanh_prime))

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
