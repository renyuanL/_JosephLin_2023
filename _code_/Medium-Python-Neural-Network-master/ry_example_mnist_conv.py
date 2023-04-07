#%% CNN on MNIST from scratch

import numpy as np

from network            import Network
from fc_layer           import FCLayer
from conv_layer         import ConvLayer
from flatten_layer      import FlattenLayer
from activation_layer   import ActivationLayer
from activations        import tanh, tanh_prime
from losses             import mse, mse_prime

from keras.datasets import mnist
from keras.utils    import np_utils

# load MNIST from server
(x_train, y_train), (x_test, y_test)= mnist.load_data()

# training data : 60_000 samples
# reshape and normalize input data 
x_train= x_train.reshape(x_train.shape[0], 28, 28, 1)
x_train= x_train.astype('float32')
x_train /= 255

# encode output which is a number in range [0,9] into a vector of size 10
# e.g. number 3 will become [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
y_train= np_utils.to_categorical(y_train)

# same for test data : 10_000 samples
x_test= x_test.reshape(x_test.shape[0], 28, 28, 1)
x_test= x_test.astype('float32')
x_test /= 255
y_test= np_utils.to_categorical(y_test)

# Network
net= Network()

net.add(ConvLayer((28, 28, 1), (3, 3), 1))  # input_shape=(28, 28, 1)   ;   output_shape=(26, 26, 1) 
net.add(ActivationLayer(tanh, tanh_prime))
net.add(FlattenLayer())                     # input_shape=(26, 26, 1)   ;   output_shape=(1, 26*26*1)

net.add(FCLayer(26*26*1, 100))              # input_shape=(1, 26*26*1)  ;   output_shape=(1, 100)
net.add(ActivationLayer(tanh, tanh_prime))
net.add(FCLayer(100, 10))                   # input_shape=(1, 100)      ;   output_shape=(1, 10)
net.add(ActivationLayer(tanh, tanh_prime))

# train on 1_000 samples
# as we didn't implemented mini-batch GD, 
# training will be pretty slow if we update at each iteration on 60_000 samples...

net.use(mse, mse_prime)
net.fit(x_train[0:1000], y_train[0:1000], epochs=100, learning_rate=0.1)

# test on 3 samples
out= net.predict(x_test[0:3])

print("\n")
print("predicted values : ")
print(out, end="\n")
print("true values : ")
print(y_test[0:3])

#%% CNN on MNIST using PyTorch and GPU

# Here is an example of how you can modify the script to use PyTorch 
# and run on a GPU:

import torch
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# check if GPU is available
device= torch.device("cuda" if torch.cuda.is_available() else 
                     "cpu")

# define data transformations
transform= transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5,), (0.5,))])

# load MNIST dataset
trainset= datasets.MNIST('~/.pytorch/MNIST_data/', 
                         download=True, train=True, transform=transform)
testset=  datasets.MNIST('~/.pytorch/MNIST_data/', 
                         download=True, train=False, transform=transform)

# create data loaders
trainloader= DataLoader(trainset, batch_size=64, shuffle=True)
testloader=  DataLoader(testset, batch_size=64, shuffle=True)

# define the network architecture
class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 1, 3)
        self.fc1 = nn.Linear(26*26*1, 100)
        self.fc2 = nn.Linear(100, 10)

    def forward(self, x):
        x = torch.tanh(self.conv1(x))
        x = x.view(-1, 26*26*1)
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        return x

# create an instance of the network and move it to the GPU
net = Network().to(device)

# define loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.1)

# train the network
epochs = 100
for e in range(epochs):
    running_loss = 0
    for images, labels in trainloader:
        # move data to GPU
        images, labels = images.to(device), labels.to(device)
        
        # flatten labels into one-hot encoded vectors
        labels = nn.functional.one_hot(labels, num_classes=10).float()
        
        # forward pass
        output = net(images)
        
        # calculate loss
        loss = criterion(output, labels)
        
        # backward pass and optimization step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    print(f"Epoch: {e+1}/{epochs}.. Training loss: {running_loss/len(trainloader)}")

# test the network on a small number of samples
dataiter = iter(testloader)
images, labels = dataiter.next()

# move data to GPU
images, labels = images.to(device), labels.to(device)

# get predictions from the network
output = net(images[:3])

print("\n")
print("predicted values : ")
print(torch.argmax(output, dim=1))
print("true values : ")
print(labels[:3])


#This script uses PyTorch to define a neural network 
# with a similar architecture as the original script. 
# The training data is loaded using PyTorch's built-in dataset 
# and data loader classes. 
# The network is trained on a GPU if one is available. 
# At the end of training, the network is tested on a small number 
# of test samples and the predicted and true values are printed.

#Please note that this is just an example and you may need to adjust it 
# according to your specific needs. 
# Also make sure that you have PyTorch installed 
# and that your machine has a CUDA-capable GPU 
# if you want to run the script on a GPU.
# %%
