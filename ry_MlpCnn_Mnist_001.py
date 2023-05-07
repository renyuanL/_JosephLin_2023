# ry Implementing 
# a Multilayer Perceptron (MLP) 
# and a Convolutional Neural Network (CNN) 
# for MNIST classification
# for tutorial purpose
# 2023/04/22

# https://paperswithcode.com/method/lenet



# %%  References
# https://github.com/patrickloeber/pytorchTutorial/blob/master/13_feedforward.py

import torch
import torch.nn             as nn
import torch.nn.functional  as F

import torchvision
import torchvision.transforms   as transforms

import matplotlib.pyplot        as plt

# Device configuration
device= torch.device(
    'cuda' if torch.cuda.is_available() else 
    'cpu')

print(f'{device= }')

##%% View data

# MNIST dataset 
train_dataset= torchvision.datasets.MNIST(
    root=       './data', 
    train=      True, 
    transform=  transforms.ToTensor(),  
    download=   True)

test_dataset= torchvision.datasets.MNIST(
    root=       './data', 
    train=      False, 
    transform=  transforms.ToTensor())

# Data loader

# batch size, not too large due to memory constraint
# not too small due to speed consideration
# batch_size=   100  samples per batch
# 60_000/100= 600 batches in train_loader
# 10_000/100= 100 batches in test_loader

batch_size=     100 

train_loader= torch.utils.data.DataLoader(
    dataset=    train_dataset, 
    batch_size= batch_size, 
    shuffle=    True)

test_loader= torch.utils.data.DataLoader(
    dataset=    test_dataset, 
    batch_size= batch_size, 
    shuffle= False)

# %% take 1 batch of data

examples= iter(test_loader)
example_data, example_targets= next(examples)

# view the images in the batch

for i in range(100):
    plt.subplot(10, 10, i+1)
    if i<10:
        plt.title(example_targets[i].item())
    # not show axes
    plt.axis('off')
    plt.imshow(example_data[i][0])
plt.show()

for i in range(100):
    lab= example_targets[i].item()
    print(lab, end=' ')
    if (i+1)%10==0:
        print('')

#%% Mlp class

# Hyper-parameters 
h0= input_size=  28*28 # 784
h1= h0//4              # 196
h2= h1//4              # 49
h3= num_classes= 10

num_epochs=      10
learning_rate= 0.01

# Fully connected neural network with 2 hidden layers
class Mlp(nn.Module):
    def __init__(self, 
                 input_size=  28*28, 
                 h1=          28*28//4,
                 h2=          28*28//4//4, 
                 num_classes= 10):
        
        super(Mlp, self).__init__()
        
        self.f= nn.ReLU()
        self.g= nn.Softmax(dim= -1)

        self.l1= nn.Linear(input_size, 
                           h1) 
        self.l2= nn.Linear(h1,         
                           h2)
        self.l3= nn.Linear(h2,         
                           num_classes)  
    
    def forward(self, x):

        y= self.l1(x);  y= self.f(y)
        y= self.l2(y);  y= self.f(y)
        y= self.l3(y);  y= self.g(y)  # softmax at the end
        return y
    
    def predict(self, x):
        
        y= self.forward(x)
        y_pred= torch.argmax(y, axis= -1)

        return y_pred
    
    def train(self, train_loader, num_epochs= 10, learning_rate= .01):

        loss=  nn.CrossEntropyLoss()
        optim= torch.optim.Adam(
            self.parameters(), 
            lr= learning_rate)  

        # Train the model
        num_batches= len(train_loader)
        for epoch in range(num_epochs):
            b= 0
            for x, y_tgt in train_loader:
                
                # origin shape: [100, 1, 28, 28]
                # reshape:      [100, 784]  
                x=     x.reshape(    -1, 1*28*28).to(device)
        
                # Forward pass
                y= self.forward(x)
                
                # compute loss
                y_tgt= y_tgt.to(device)
                e= loss(y, y_tgt)
                
                # Backward and optimize
                optim.zero_grad()
                e.backward()
                optim.step()
                b += 1
                # show progress
                if b % 100 == 0:
                    print (f'{epoch}/{num_epochs}, {b}/{num_batches}, e: {e.item():.4f}')

    def test(self, test_loader):
        # Test the model
        # In test phase, we don't need to compute gradients 
        # (for memory efficiency)
        with torch.no_grad():
            n_correct= 0
            n_samples= 0
            for x, y_tgt in test_loader:
                x=      x.reshape(-1, 1*28*28).to(device)
                y_pred= self.predict(x)
                
                y_tgt= y_tgt.to(device)
                n_correct += (y_pred== y_tgt).sum().item()
                n_samples += len(y_tgt)
            
            acc= n_correct / n_samples
            print(f'{acc= :.4f}')
            
        return {'acc': acc, 'n_correct': n_correct, 'n_samples': n_samples}

mlp= Mlp(input_size, h1, h2, num_classes).to(device)
mlp.train(train_loader)
mlp.test(test_loader)

# %% LeNet class

class Cnn(nn.Module):

    def __init__(self):
        super(Cnn, self).__init__()
        
        self.convolutional_layer= nn.Sequential(
            nn.Conv2d(in_channels=  1, 
                      out_channels= 16, 
                      kernel_size= (3,3), 
                      padding= 'same'),
            nn.ReLU(),

            nn.Conv2d(in_channels=  16, 
                      out_channels= 16, 
                      kernel_size=(3, 3),
                      padding= 'same'),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size= (2,2), 
                         stride=      (2,2)),

            nn.Conv2d(in_channels=  16, 
                      out_channels= 16, 
                      kernel_size= (3,3), 
                      padding= 'same'),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size= (2,2), 
                         stride=      (2,2)),
        )

        height= 28//2//2 # 7
        width=  28//2//2 # 7

        flatten_size= 16*height*width # 784

        h1= flatten_size//4 # 196
        h2= h1//4 # 49
        

        self.linear_layer= nn.Sequential(
            nn.Linear(in_features= flatten_size, 
                      out_features= h1),
            nn.ReLU(),
            nn.Linear(in_features= h1, 
                      out_features=h2),
            nn.ReLU(),
            nn.Linear(in_features= h2, 
                      out_features=10),
        )

    def forward(self, x):
        x= self.convolutional_layer(x)
        
        x= torch.flatten(x, 1)

        x= self.linear_layer(x)
        y= F.softmax(x, dim=-1)
        return y
    
    def predict(self, x):
        
        y= self.forward(x)
        y_pred= torch.argmax(y, axis= -1)

        return y_pred
    
    def train(self, train_loader, num_epochs= 10, learning_rate= .001):
            
        optimizer= torch.optim.Adam(self.parameters(), lr= learning_rate)
        criterion= nn.CrossEntropyLoss()

        epochs= num_epochs
        train_loss, val_loss = [], []

        for epoch in range(epochs):
        
            total_train_loss = 0
            total_val_loss = 0

            # self.train()
            #### super(LeNet5, self).train() # same as self.train() 
            # 
            # and self.eval() will turn on and off the training mode 
            # of our model.
            
            # training our model
            for idx, (image, label) in enumerate(train_loader):

                image, label = image.to(device), label.to(device)
                optimizer.zero_grad()
                pred = self.forward(image)

                loss = criterion(pred, label)
                total_train_loss += loss.item()

                loss.backward()
                optimizer.step()

            total_train_loss = total_train_loss / (idx + 1)
            train_loss.append(total_train_loss)
            
            # validating our model
            #### model.eval()

            total = 0
            for idx, (image, label) in enumerate(test_loader):
                image, label = image.to(device), label.to(device)
                
                pred = self.forward(image)

                loss = criterion(pred, label)
                total_val_loss += loss.item()

                pred = torch.nn.functional.softmax(pred, dim=1)
                for i, p in enumerate(pred):
                    if label[i] == torch.max(p.data, 0)[1]:
                        total = total + 1

            accuracy = total / len(test_loader.dataset)

            total_val_loss = total_val_loss / (idx + 1)
            val_loss.append(total_val_loss)

            if epoch % 5 == 0:
                print('\nEpoch: {}/{}, Train Loss: {:.4f}, Val Loss: {:.4f}, Val Acc: {:.4f}'
                        .format(epoch, epochs, total_train_loss, total_val_loss, accuracy))
        plt.plot(train_loss)
        plt.plot(val_loss)

        return train_loss, val_loss
    
    def test(self, test_loader):

        model= self

        testiter = iter(test_loader)
        images, labels = next(testiter)

        with torch.no_grad():
            images, labels = images.to(device), labels.to(device)
            preds = model(images)

        images_np = [i.mean(dim=0).cpu().numpy() for i in images]

        class_names= ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

        fig = plt.figure(figsize=(20,20))
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.02, wspace=0.02)

        for i in range(100):
            ax = fig.add_subplot(10, 10, i + 1, xticks=[], yticks=[])
            ax.imshow(images_np[i], cmap=plt.cm.gray_r, interpolation='nearest')

            if labels[i] == torch.max(preds[i], 0)[1]:
                ax.text(0, 5, class_names[torch.max(preds[i], 0)[1]], 
                        color='green', fontsize=20)
                # set the text font larger

            else:
                ax.text(0, 5, class_names[torch.max(preds[i], 0)[1]], 
                        color='red',  fontsize=30, fontweight='bold')
                ax.text(20, 5, class_names[labels[i]], 
                        color='blue',  fontsize=30, fontweight='bold')
# %% Train the model

cnn= Cnn().to(device)
cnn.train(train_loader)
cnn.test(test_loader)

#%%
# %%
class LeNet(nn.Module):

  def __init__(self):
    
    super(LeNet, self).__init__()

    self.conv1= nn.Conv2d(   1, 6, 5) 
    self.pool1= nn.MaxPool2d(2, 2) 
    self.conv2= nn.Conv2d(   6,16, 5) 
    self.pool2= nn.MaxPool2d(2, 2) 
    
    self.fc1= nn.Linear(     16*4*4, 120) 
    # 16*4*4= 256
    # why 4*4? 
    # this is the output size of the conv2 layer
    # 28-5+1= 24, 24/2= 12, 12-5+1= 8, 8/2= 4 
    
    self.fc2= nn.Linear(     120,84)
    self.fc3= nn.Linear(     84, 10) 


  def forward(self,x):
    
    x= self.pool1(F.relu(self.conv1(x))) 
    x= self.pool2(F.relu(self.conv2(x))) 
    
    x= torch.flatten(x,1) 
    # flatten all dimensions except batch
    # x= x.view(-1, 16*4*4) # same as above
    # x= x.reshape(-1, 16*4*4) # same as above
    # x.shape= (64, 256) # 64= batch_size, 256= 16*4*4

    x= F.relu(self.fc1(x))
    x= F.relu(self.fc2(x)) 
    logits= self.fc3(x)

    # logits= output of the last layer 
    # logits.shape= (64, 10) # 64= batch_size, 10= number of classes
    # logits are the raw values before the softmax function
    # y= F.softmax(logits, dim=1)
    # y will be the probability distribution over each class

    return logits

leNet= LeNet() 
print(leNet)


opt=  optimizer= torch.optim.Adam(leNet.parameters(), lr= learning_rate)
loss= criterion= nn.CrossEntropyLoss()

import numpy as np

def train(dataloader, model, loss, opt, 
          device= torch.device('cpu')):
  
  model= model.to(device)

  size= len(train_loader.dataset)
  for b, (x, y_tgt) in enumerate(dataloader):
    x, y_tgt= x.to(device), y_tgt.to(device) 
    y_pred= model(x) 
    l=      loss(y_pred, y_tgt) 

    opt.zero_grad()
    l.backward()
    opt.step()

    if b % 100 ==0:
      l, current = l.item(), b * len(x) 
      perplexity= np.exp(l)
      print(f"loss= {l:.6f},  perplexity= {perplexity:.6f}, [{current} / {size}]")

def test(dataloader, model, loss, 
         device= torch.device('cpu')):

  model= model.to(device)
  size = len(dataloader.dataset)
  num_batches = len(dataloader) 
  correct, test_loss = 0,0 
  with torch.no_grad():
    for x,y in dataloader:
      x, y = x.to(device), y.to(device)
      y_pred = model(x) 
      test_loss += loss(y_pred, y).item() 
      correct += (y_pred.argmax(1) == y).type(torch.float).sum().item() 
  test_loss /= num_batches 
  correct /= size 
  print(f"accuracy is {correct:.4f}, test_loss= {test_loss:.6f}, test_perplexity= {np.exp(test_loss):.6f}")


'''
device= torch.device(
  "cuda" if torch.cuda.is_available() else 
  "cpu")
'''
#%%
print(f'{device= }')

EPOCHS= num_epochs #= 10 

for epochs in range(EPOCHS):
  print(f"epoch: {epochs+1} ---------------------------")
  train(train_loader, leNet, loss, opt, device)
  test(test_loader, leNet, loss, device) 

print('DONE') 
# %%
# the parameters of the model
for name, param in leNet.named_parameters():
    print(f'{name= }, {param.shape= }')

# %%
# the total number of parameters
total_number_of_parameters= sum(
    p.numel() for p in leNet.parameters())

print(f'{total_number_of_parameters= }')
# %%
'''
name= 'conv1.weight', param.shape= torch.Size([6, 1, 5, 5])
name= 'conv1.bias', param.shape= torch.Size([6])
name= 'conv2.weight', param.shape= torch.Size([16, 6, 5, 5])
name= 'conv2.bias', param.shape= torch.Size([16])
name= 'fc1.weight', param.shape= torch.Size([120, 256])
name= 'fc1.bias', param.shape= torch.Size([120])
name= 'fc2.weight', param.shape= torch.Size([84, 120])
name= 'fc2.bias', param.shape= torch.Size([84])
name= 'fc3.weight', param.shape= torch.Size([10, 84])
name= 'fc3.bias', param.shape= torch.Size([10])
total_number_of_parameters= 44_426
'''
qq= (
      6*( 1*5*5+1) +
     16*( 6*5*5+1) + 
    120*(16*4*4+1) + 
     84*(   120+1) + 
     10*(    84+1)
    ) 
print(f'{qq= }')
# %%
