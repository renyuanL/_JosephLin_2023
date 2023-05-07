#%%
import torch 
import torchvision 
from torch.utils.data import DataLoader
from torchvision import datasets 
from torchvision import transforms
from torch import nn 
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

trans= transforms.Compose([
    transforms.ToTensor(), 
    #### transforms.Normalize(mean= 0.5, std= 0.5) 
    # mean and std of the dataset
    ])

training_data= datasets.MNIST(
    root= "data", 
    train= True, 
    download= True, 
    transform= trans)

validation_data= datasets.MNIST(
    root= "data",
    train= False, 
    download= True, 
    transform= trans)


#%%
training_data
#%%
training_data.class_to_idx

image, label= training_data[0] 
classes= training_data.classes

print(image.shape)
print(image.dtype)
print(classes[label])
#%%
X= ( training_data
    .data
    .numpy()
    .astype(np.float32))

X.shape, X.max(), X.min(), X.mean(), X.std()

#%%
plt.imshow(image.reshape(28,28)) 
plt.title(f'{label= }')
plt.show()

#%%
training_dataloader= DataLoader(
    dataset=    training_data, 
    batch_size= 64, 
    shuffle=    True)

test_dataloader= DataLoader(
    dataset=    validation_data, 
    batch_size= 64, 
    shuffle=    False) 
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

net= LeNet() 
print(net)
# %%
# the number of parameters in each layer
for name, param in net.named_parameters():
    print(name, param.numel())

# the total number of parameters
sum(p.numel() 
    for p in net.parameters() 
    if p.requires_grad)
# 44_426


# %%
# loss function and optimizer
loss=  nn.CrossEntropyLoss() 

# study the loss function, the CrossEntropyLoss
# https://en.wikipedia.org/wiki/Cross_entropy
# https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html

N= 1000 # number of classes

p= np.ones(N)/N        # uniform distribution
q= np.random.random(N) # random distribution
p /= p.sum()
q /= q.sum()

log= np.log
pq= p @ log(1/q)
pp= p @ log(1/p)
qp= q @ log(1/p)
qq= q @ log(1/q)

np.exp(pq), np.exp(pp), np.exp(qp), np.exp(qq)

#%%
opt= torch.optim.Adam(
   net.parameters(), 
   lr= 1e-3)
 
# %%

def train(dataloader, model, loss, opt, 
          device= torch.device('cpu')):
  
  model= model.to(device)

  size= len(training_dataloader.dataset)
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



device= torch.device(
  "cuda" if torch.cuda.is_available() else 
  "cpu")

print(f'{device= }')


EPOCHS= 20 

for epochs in range(EPOCHS):
  print(f"epoch: {epochs+1} ---------------------------")
  train(training_dataloader, net, loss, opt, device)
  test(test_dataloader, net, loss, device) 

print('DONE') 
# %%
