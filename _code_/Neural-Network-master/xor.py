#%%
import numpy as np
import matplotlib.pyplot as plt
from   mpl_toolkits.mplot3d import Axes3D

from dense       import Dense, Softmax
from activations import Tanh, Sigmoid, Relu
from losses      import mse, mse_prime
from network     import train, predict

# training data
X= np.array(
    [[0, 0], 
     [0, 1], 
     [1, 0], 
     [1, 1]])

Y= np.array(
    [[0], 
     [1], 
     [1], 
     [0]])
#%%
Dense(2,3).forward(X[0])
Softmax().forward(X[2])

#%%

# add one more dimension to X, Y
#X= np.expand_dims(X, axis= -1)
#Y= np.expand_dims(Y, axis= -1)

# network
network= [
    Dense(2, 3),
    Tanh(),   #Relu(),
    Dense(3, 1),
    Sigmoid() #Softmax()
]

predict(network, X[0])

#%%
# train
train(network, 
      mse, 
      mse_prime, 
      X, 
      Y, 
      epochs= 10_000, 
      learning_rate= 0.01)
#%%
# decision boundary plot
points= []
for x0 in np.linspace(0, 1, 20):
    for x1 in np.linspace(0, 1, 20):
        y= predict(network, [x0, x1])
        points += [(x0, x1, y[0,0])]

points= np.array(points)

fig= plt.figure()
ax=  fig.add_subplot(111, projection="3d")
ax.scatter(points[:, 0], 
           points[:, 1], 
           points[:, 2], 
           c= points[:, 2], 
           cmap= "rainbow" #"winter"
           )
plt.show()

# %%
