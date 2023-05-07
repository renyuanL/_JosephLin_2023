#%%
# !wget https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv
#%%
import numpy as np
import torch

# load the dataset
dataset= np.loadtxt(
    'pima-indians-diabetes.csv', 
    delimiter=',')

X= dataset[:,0:8]
y= dataset[:,8]
X= torch.tensor(X, dtype=torch.float32)
y= torch.tensor(y, dtype=torch.float32).reshape(-1, 1)

# split the dataset into training and test sets
Xtrain= X[:700]
ytrain= y[:700]
Xtest=  X[700:]
ytest=  y[700:]

''''
This dataset is small — only 768 samples. Here, 
it takes the first 700 as the training set and
the rest as the test set.
It is not the focus of this chapter, 
but you can reuse the model, 
the loss function, and
the optimizer from a previous chapter:
'''
#%%
import torch.nn as nn
import torch.optim as optim

# %%
model= nn.Sequential(
    nn.Linear( 8, 12), nn.ReLU(),
    nn.Linear(12,  8), nn.ReLU(),
    nn.Linear( 8,  1), nn.Sigmoid()
    )
print(model)

# loss function and optimizer
loss_fn= nn.BCELoss() # binary cross entropy
optimizer= optim.Adam(model.parameters(), lr=0.001)

# %%
n_epochs = 100 # number of epochs to run
batch_size = 10 # size of each batch
batches_per_epoch = len(Xtrain) // batch_size
for epoch in range(n_epochs):
    for i in range(batches_per_epoch):
        start = i * batch_size
        # take a batch
        Xbatch = Xtrain[start:start+batch_size]
        ybatch = ytrain[start:start+batch_size]
        # forward pass
        y_pred = model(Xbatch)
        loss = loss_fn(y_pred, ybatch)
        # backward pass
        optimizer.zero_grad()
        loss.backward()
        # update weights
        optimizer.step()

# %%
# evaluate trained model with test set
with torch.no_grad():
    y_pred = model(Xtest)
accuracy = (y_pred.round() == ytest).float().mean()
print("Accuracy {:.2f}".format(accuracy * 100))


# %%

#from numpy import loadtxt
import numpy as np
from keras.models import Sequential
from keras.layers import Dense

import warnings
warnings.filterwarnings('ignore')

path=    'pima-indians-diabetes.csv'
dataset= np.loadtxt(path, delimiter=',')

X= dataset[:,0:8]
y= dataset[:,8]

# split into train and test sets
Xtrain= X[:700]
ytrain= y[:700]
Xtest=  X[700:]
ytest=  y[700:]

# %%
model= Sequential()
model.add(Dense(12, input_dim= 8, activation='relu'))
model.add(Dense(8,  activation='relu'))
model.add(Dense(1,  activation='sigmoid'))

model.compile(
     loss='binary_crossentropy', 
     optimizer='adam', 
     metrics=['accuracy'])

model.fit(Xtrain, ytrain, 
          epochs= 100, 
          batch_size= 10, 
          verbose=0)
# %%
predictions= model.predict(Xtest)

for i in range(5):
	print('%s => %d (expected %d)'%
       (X[i].tolist(), 
        predictions[i], 
        y[i]))

# %%
res= model.evaluate(Xtest,ytest, return_dict=True)
print(res)


# %%
