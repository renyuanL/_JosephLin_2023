# %% [markdown]
# # Deep Learning with PyTorch, 9-day mini-course
# 
# - Contents
# - Before We Get Started... 1
# - Lesson 01: Introduction to PyTorch 4
# - Lesson 02: Build Your First Multilayer Perceptron Model 6
# - Lesson 03: Training a PyTorch Model 8
# - Lesson 04: Using a PyTorch Model for Inference 10
# - Lesson 05: Loading Data from Torchvision 12
# - Lesson 06: Using PyTorch DataLoader 14
# - Lesson 07: Convolutional Neural Network 16
# - Lesson 08: Train an Image Classifier 18
# - Lesson 09: Train with GPU 20
# - Final Word Before You Go...

# %%

# Example of PyTorch library
import torch
import torch.nn.functional as F

# declare two symbolic floating-point scalars
x= torch.tensor(
    [[[1,2,3,4,5,6,7,8,9]]]
    )
w= torch.tensor(
    [[[1,1,1]],[[1,-1,1]]]
    )
# compute the 1d convolution of x with w
y= F.conv1d(x,w)

x,w,y



# %%
b=   1
c=   1
k=   2

tx= 10

tw=  3

import numpy as np
x= np.arange(0,b*c*tx).reshape(b,c,tx)
w= np.arange(0,k*c*tw).reshape(k,c,tw)

x= torch.tensor(x)
w= torch.tensor(w)
y= F.conv1d(x,w)

x.shape, w.shape, y.shape
x,w,y

# %%
#torch.__version__ # '2.0.0+cu117'


# %%
import torch.nn as nn
model= nn.Sequential(
    nn.Linear(8, 12), nn.ReLU(),
    nn.Linear(12, 8), nn.ReLU(),
    nn.Linear(8, 1),  nn.Sigmoid()
    )

model



# %%
model.state_dict()

# %%
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

dataset= np.loadtxt(
    'pima-indians-diabetes.csv', 
    delimiter=',')

X= dataset[:,0:8]
y= dataset[:,8]

X= torch.tensor(X, dtype=torch.float32)
y= torch.tensor(y, dtype=torch.float32).reshape(-1, 1)

loss_fn= nn.BCELoss() # binary cross-entropy


# %%
optimizer = optim.Adam(model.parameters(), lr=0.001)

n_epochs=  100
batch_size= 10

for epoch in range(n_epochs):
    for i in range(0, len(X), batch_size):
        
        Xbatch= X[i:i+batch_size]
        y_pred= model(Xbatch)
        ybatch= y[i:i+batch_size]
        loss=   loss_fn(y_pred, ybatch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print(f'Finished epoch {epoch}, latest loss {loss}')



# %%
i = 5
X_sample= X[i:i+1]
y_pred= model(X_sample)

print(f"{X_sample[0]} -> {y_pred[0]}")

# %%
x= X[0]
y_pred= model(x)
x, y_pred

# %%
model.eval()
with torch.no_grad():
    y_pred = model(X)

# %%
accuracy= (y_pred.round() == y).float().mean()
print(f"Accuracy {accuracy}")


# %%
import matplotlib.pyplot as plt
import torchvision

trainset= torchvision.datasets.CIFAR10(root='./data', train=True,  download=True)
testset=  torchvision.datasets.CIFAR10(root='./data', train=False, download=True)
fig, ax= plt.subplots(4, 6, 
                      sharex=True, 
                      sharey=True, 
                      figsize=(12,8))
for i in range(0, 24):
    row, col = i//6, i%6
    ax[row][col].imshow(trainset.data[i])
plt.show()


# %%
trainset.data.shape, testset.data.shape

# %%
import matplotlib.pyplot as plt
import torchvision
import torch
from torchvision.datasets import CIFAR10

transform= torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
trainset=  CIFAR10(root='./data', train=True, download=True, transform=transform)
testset=   CIFAR10(root='./data', train=False, download=True, transform=transform)

batch_size= 24
trainloader= torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
testloader=  torch.utils.data.DataLoader(testset,  batch_size=batch_size, shuffle=True)

fig, ax = plt.subplots(4, 6, sharex=True, sharey=True, figsize=(12,8))

for images, labels in trainloader:
    for i in range(batch_size):
        row, col = i//6, i%6
        ax[row][col].imshow(images[i].numpy().transpose([1,2,0]))
    break # take only the first batch
plt.show()

# %%
import torch.nn as nn

model= nn.Sequential(
    nn.Conv2d(3,  32, kernel_size=(3,3), stride=1, padding=1), nn.ReLU(), nn.Dropout(0.3),
    nn.Conv2d(32, 32, kernel_size=(3,3), stride=1, padding=1), nn.ReLU(), nn.MaxPool2d(kernel_size=(2, 2)),
    nn.Flatten(),
    nn.Linear(8192, 512), nn.ReLU(), nn.Dropout(0.5),
    nn.Linear(512, 10)
    )

model

# %%
import torch.nn as nn
import torch.optim as optim

loss_fn=    nn.CrossEntropyLoss()
optimizer= optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# this is for demonstration purposes only, 
# because it takes a long time to train
# you should use at least 10 epochs
# and a GPU for training

n_epochs= 2 # 10
for epoch in range(n_epochs):
    model.train()
    for inputs, labels in trainloader:
        y_pred = model(inputs)
        loss = loss_fn(y_pred, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    acc = 0
    count = 0
    model.eval()
    with torch.no_grad():
        for inputs, labels in testloader:
            y_pred = model(inputs)
            acc += (torch.argmax(y_pred, 1) == labels).float().sum()
            count += len(labels)
    acc /= count
    print("Epoch %d: model accuracy %.2f%%" % (epoch, acc*100))



# %%
'''
The model training you did in the previous lesson should take a while. 
If you have a supported GPU, you can speed up the training a lot.
The way to use GPU in PyTorch is to send the model and data to GPU before execution.
Then you have an option to send back the result from GPU, 
or perform the evaluation in GPU directly.
It is not difficult to modify the code from the previous lesson to use GPU. 
Below is what it should be done:
'''

import torch.nn as nn
import torch.optim as optim

device= torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
print("Using device:", device)

loss_fn=    nn.CrossEntropyLoss()
optimizer=  optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

n_epochs= 20
for epoch in range(n_epochs):
    model.train()
    for inputs, labels in trainloader:
        inputs= inputs.to(device)
        labels= labels.to(device)
        
        y_pred= model(inputs)
        
        loss= loss_fn(y_pred, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    acc= 0
    count= 0
    model.eval()
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs= inputs.to(device)
            labels= labels.to(device)
            y_pred = model(inputs)
            acc += (torch.argmax(y_pred, 1) == labels).float().sum()
            count += len(labels)
    acc /= count
    print("Epoch %d: model accuracy %.2f%%" % (epoch, acc*100))

# Epoch 19: model accuracy 70.10%




