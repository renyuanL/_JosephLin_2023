# https://learn.microsoft.com/zh-tw/training/modules/intro-computer-vision-pytorch/4-convolutional-networks

#!/usr/bin/env python
# coding: utf-8

# # Convolutional neural networks
# 
# In the previous unit we have learned how to define a multi-layered neural network using class definition, but those networks were generic, and not specialized for computer vision tasks. In this unit we will learn about **Convolutional Neural Networks (CNNs)**, which are specifically designed for computer vision.
# 
# Computer vision is different from generic classification, because when we are trying to find a certain object in the picture, we are scanning the image looking for some specific **patterns** and their combinations. For example, when looking for a cat, we first may look for horizontal lines, which can form whiskers, and then certain combination of whiskers can tell us that it is actually a picture of a cat. Relative position and presence of certain patterns is important, and not their exact position on the image. 
# 
# To extract patterns, we will use the notion of **convolutional filters**. But first, let us load all dependencies and functions that we have defined in the previous units.

# In[1]:


#get_ipython().system('
#wget https://raw.githubusercontent.com/MicrosoftDocs/pytorchfundamentals/main/computer-vision-pytorch/pytorchcv.py
#')



# In[2]:


import torch
import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt
from torchinfo import summary
import numpy as np

from pytorchcv import (
    load_mnist, 
    train, 
    plot_results, 
    plot_convolution, 
    display_dataset)

load_mnist(batch_size=128)


# ## Convolutional filters
# 
# Convolutional filters are small windows that run over each pixel of the image and compute weighted average of the neighboring pixels.
# 
# <img alt="Sliging window over 28x28 digit image" src="images/4-convolutional-networks-1.png" width="50%"/>
# 
# They are defined by matrices of weight coefficients. Let's see the examples of applying two different convolutional filters over our MNIST handwritten digits:

# In[3]:


plot_convolution(torch.tensor([[-1.,0.,1.],[-1.,0.,1.],[-1.,0.,1.]]),'Vertical edge filter')
plot_convolution(torch.tensor([[-1.,-1.,-1.],[0.,0.,0.],[1.,1.,1.]]),'Horizontal edge filter')


# First filter is called a **vertical edge filter**, and it is defined by the following matrix:
# $$
# \left(
#     \begin{matrix}
#      -1 & 0 & 1 \cr
#      -1 & 0 & 1 \cr
#      -1 & 0 & 1 \cr
#     \end{matrix}
# \right)
# $$
# When this filter goes over relatively uniform pixel field, all values add up to 0. However, when it encounters a vertical edge in the image, high spike value is generated. That's why in the images above you can see vertical edges represented by high and low values, while horizontal edges are averaged out.
# 
# An opposite thing happens when we apply horizontal edge filter - horizontal lines are amplified, and vertical are averaged out.
# 
# > If we apply $3\times3$ filter to an image of size $28\times28$ - the size of the image will become $26\times26$, because the filter does not go over the image boundaries. In some cases, however, we may want to keep the size of the image the same, in which case image is padded with zeros on each side.
# 
# In classical computer vision, multiple filters were applied to the image to generate features, which then were used by machine learning algorithm to build a classifier. However, in deep learning we construct networks that **learn** best convolutional filters to solve classification problem.
# 
# To do that, we introduce **convolutional layers**.

# ## Covolutional layers
# 
# Convolutional layers are defined using `nn.Conv2d` construction. We need to specify the following:
# * `in_channels` - number of input channels. In our case we are dealing with a grayscale image, thus number of input channels is 1. Color image has 3 channels (RGB).
# * `out_channels` - number of filters to use. We will use 9 different filters, which will give the network plenty of opportunities to explore which filters work best for our scenario.
# * `kernel_size` is the size of the sliding window. Usually 3x3 or 5x5 filters are used. The choice of filter size is usually chosen by experiment, that is by trying out different filter sizes and comparing resulting accuracy.
# 
# Simplest CNN will contain one convolutional layer. Given the input size 28x28, after applying nine 5x5 filters we will end up with a tensor of 9x24x24 (the spatial size is smaller, because there are only 24 positions where a sliding interval of length 5 can fit into 28 pixels). Here the result of each filter is represented by a different channel in the image (thus the first dimension 9 corresponds to the number of filters).
# 
# After convolution, we flatten 9x24x24 tensor into one vector of size 5184, and then add linear layer, to produce 10 classes. We also use `relu` activation function in between layers. 

# In[4]:


class OneConv(nn.Module):
    def __init__(self):
        super(OneConv, self).__init__()
        self.conv = nn.Conv2d(in_channels=1,out_channels=9,kernel_size=(5,5))
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(5184,10)

    def forward(self, x):
        x = nn.functional.relu(self.conv(x))
        x = self.flatten(x)
        x = nn.functional.log_softmax(self.fc(x),dim=1)
        return x

net = OneConv()

summary(net,input_size=(1,1,28,28))


# You can see that this network contains around 50k trainable parameters, compared to around 80k in fully-connected multi-layered networks. This allows us to achieve good results even on smaller datasets, because convolutional networks generalize much better.
# 
# > Note that the number of parameters of convolutional layer is quite small, and it *does not depend on the resolution of the image*! In our case, we were using 9 filters of dimension $5\times5$, thus the number of parameters is $9\times5\times5+9=234$. Even though we missed this in our discussion above, but convolutional filter also has bias. Most of the parameters of our network comes from the final `Dense` layer.

# In[5]:


hist = train(net,train_loader,test_loader,epochs=5)
plot_results(hist)


# As you can see, we are able to achieve higher accuracy, and much faster, compared to the fully-connected networks from previous unit.
# 
# We can also visualize the weights of our trained convolutional layers, to try and make some more sense of what is going on:

# In[ ]:


fig,ax = plt.subplots(1,9)
with torch.no_grad():
    p = next(net.conv.parameters())
    for i,x in enumerate(p):
        ax[i].imshow(x.detach().cpu()[0,...])
        ax[i].axis('off')


# You can see that some of those filters look like they can recognize some oblique strokes, while others look pretty random. 
# 
# ## Takeaway
# 
# Convolutional layer allows us to extract certain image patterns from the image, so that final classifier is based on top of those features. However, we can use the same approach of extracting patterns inside the feature space, by stacking another convolutional layer on top of the first one. We will learn about multi-layer convolutional networks in the next unit.
