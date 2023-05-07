'''
Training a PyTorch Model with
DataLoader and Dataset
'''


#http://archive.ics.uci.edu/ml/datasets/connectionist+bench+(sonar,+mines+vs.+rocks)
#%%
import pandas as pd

import torch
from torch.utils.data      import DataLoader

from sklearn.preprocessing import LabelEncoder

# Read data, convert to NumPy arrays
data= pd.read_csv("sonar.csv", header=None)
X= data.iloc[:, 0:60].values
y= data.iloc[:, 60].values
#%%
# encode class values as integers
encoder= LabelEncoder()
encoder.fit(y)
y= encoder.transform(y)

# convert into PyTorch tensors
X= torch.tensor(X, dtype=torch.float32)
y= torch.tensor(y, dtype=torch.float32).reshape(-1, 1)

# create DataLoader, then take one batch
loader= DataLoader(
    list(zip(X,y)), 
    shuffle=True, 
    batch_size= 10)

for X_batch, y_batch in loader:
    print(X_batch, y_batch)
    break
# %%
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split

# train-test split for evaluation of the model
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, shuffle=True)
# setup DataLoader for training set
loader = DataLoader(list(zip(X_train, y_train)), shuffle=True, batch_size=16)

# create model
model= nn.Sequential(
    nn.Linear(60, 60), nn.ReLU(),
    nn.Linear(60, 30), nn.ReLU(),
    nn.Linear(30, 1),  nn.Sigmoid()
    )

# Train the model
n_epochs= 200
loss_fn=  nn.BCELoss()
optimizer= optim.SGD(model.parameters(), lr=0.1)

model.train()

for epoch in range(n_epochs):
    for X_batch, y_batch in loader:
        y_pred = model(X_batch)
        loss = loss_fn(y_pred, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
# evaluate accuracy after training
model.eval()

y_pred= model(X_test)

acc= (y_pred.round() == y_test).float().mean()
acc= float(acc)

print("Model accuracy: %.2f%%" % (acc*100))

# %%
