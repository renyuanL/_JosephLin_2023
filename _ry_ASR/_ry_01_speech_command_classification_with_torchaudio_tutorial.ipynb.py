#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchaudio
import sys

import matplotlib.pyplot as plt
import IPython.display as ipd

from tqdm import tqdm


# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


# %%
from torchaudio.datasets import SPEECHCOMMANDS
import os

data_path= "L:/_ryDatasets"
# check if the dircetory exists, if not, make it
if not os.path.isdir(data_path):
    os.mkdir(data_path)


class SubsetSC(SPEECHCOMMANDS):
    def __init__(self, subset: str = None):
        super().__init__(data_path, download=True)

        def load_list(filename):
            filepath = os.path.join(self._path, filename)
            with open(filepath) as fileobj:
                return [os.path.normpath(os.path.join(self._path, line.strip())) for line in fileobj]

        if subset == "validation":
            self._walker = load_list("validation_list.txt")
        elif subset == "testing":
            self._walker = load_list("testing_list.txt")
        elif subset == "training":
            excludes = load_list("validation_list.txt") + load_list("testing_list.txt")
            excludes = set(excludes)
            self._walker = [w for w in self._walker if w not in excludes]


# Create training and testing split of the data. We do not use validation in this tutorial.
train_set = SubsetSC("training")
test_set = SubsetSC("testing")

waveform, sample_rate, label, speaker_id, utterance_number = train_set[0]

labels= [
 'backward', 'bed',     'bird',     'cat',      'dog',
 'down',    'eight',    'five',     'follow',   'forward',
 'four',    'go',       'happy',    'house',    'learn',
 'left',    'marvin',   'nine',     'no',       'off',
 'on',      'one',      'right',    'seven',    'sheila',
 'six',     'stop',     'three',    'tree',     'two',
 'up',      'visual',   'wow',      'yes',      'zero'
]

# %%
def label_to_index(word):
    # Return the position of the word in labels
    return torch.tensor(labels.index(word))


def index_to_label(index):
    # Return the word corresponding to the index in labels
    # This is the inverse of label_to_index
    return labels[index]


word_start = "yes"
index = label_to_index(word_start)
word_recovered = index_to_label(index)

print(word_start, "-->", index, "-->", word_recovered)


# %%
def pad_sequence(batch):
    # Make all tensor in a batch the same length by padding with zeros
    batch = [item.t() for item in batch]
    batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0.)
    return batch.permute(0, 2, 1)


def collate_fn(batch):

    # A data tuple has the form:
    # waveform, sample_rate, label, speaker_id, utterance_number

    tensors, targets = [], []

    # Gather in lists, and encode labels as indices
    for waveform, _, label, *_ in batch:
        tensors += [waveform]
        targets += [label_to_index(label)]

    # Group the list of tensors into a batched tensor
    tensors = pad_sequence(tensors)
    targets = torch.stack(targets)

    return tensors, targets


batch_size = 1024 #1000 #1024 #256

if device == "cuda":
    num_workers = 1
    pin_memory = True
else:
    num_workers = 0
    pin_memory = False

train_loader = torch.utils.data.DataLoader(
    train_set,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=collate_fn,
    num_workers=num_workers,
    pin_memory=pin_memory,
)
test_loader = torch.utils.data.DataLoader(
    test_set,
    batch_size=batch_size,
    shuffle=False,
    drop_last=False,
    collate_fn=collate_fn,
    num_workers=num_workers,
    pin_memory=pin_memory,
)

#%%

#%%

class ryM(nn.Module):
    def __init__(self, 
                 in_chs=   1,  #  1 channel, mono waveform
                 out_cls= 35,  # 35 words as output classes
                 sample_rate=  16_000 # sample rate of the audio file
                 ):
        
        super().__init__()

        new_sample_rate= sample_rate//2 #8_000

        self.transform= torchaudio.transforms.Resample(
            orig_freq= sample_rate, 
            new_freq=  new_sample_rate)

        self.act=  nn.ReLU()
        self.flat= nn.Flatten()
        self.out=  nn.LogSoftmax(dim=-1)
        #self.out=  nn.Softmax(dim=-1)

        k1= int(.02* new_sample_rate) # 160 # 20ms
        s1= int(.01* new_sample_rate) #  80 # 10ms
        ch1= 64 # 64 channels in 1st convolution layer

        k2= 4 # kernel size in the other conv layer
        s2= 2 # stride in the other conv layer

        self.conv1= nn.Conv1d(in_chs, ch1,   kernel_size= k1, stride= s1) 
        self.bn1=   nn.BatchNorm1d(ch1)

        self.conv2= nn.Conv1d(ch1,  ch1 *2,  kernel_size= k2, stride= s2)
        self.bn2=   nn.BatchNorm1d(ch1 *2)

        self.conv3= nn.Conv1d(ch1 *2, ch1 *4, kernel_size= k2, stride= s2)
        self.bn3=   nn.BatchNorm1d(ch1 *4)

        self.conv4= nn.Conv1d(ch1 *4, ch1 *4, kernel_size= k2, stride= s2)
        self.bn4=   nn.BatchNorm1d(ch1 *4)

        self.conv5= nn.Conv1d(ch1 *4, ch1 *2, kernel_size= k2, stride= s2)
        self.bn5=   nn.BatchNorm1d(ch1 *2)
        
        self.fc1= nn.Linear(ch1 *2, ch1)
        self.fc2= nn.Linear(ch1,    out_cls)

    def forward(self, x):
        
        x= self.transform(x) # (1,16000) -> (1,8000) # downsample by factor of 2

        #  CNNs
        x= self.conv1(x) # (1, 8000) -> (64, 99)
        x= self.bn1(x)   
        x= self.act(x)   
        
        x= self.conv2(x) # (64, 99) -> (128, 48)
        x= self.bn2(x)   
        x= self.act(x)   
        
        x= self.conv3(x) # (128, 48) -> (256, 23)
        x= self.bn3(x)   
        x= self.act(x)   
       
        x= self.conv4(x) # (256, 23) -> (256, 10)
        x= self.bn4(x)   
        x= self.act(x)

        x= self.conv5(x) # (256, 10) -> (128, 4)
        x= self.bn5(x)   
        x= self.act(x)   
        
        # global average pooling
        x= F.avg_pool1d(x, x.shape[-1])  # -> (128, 1)
        x= self.flat(x) # -> (128)

        # MLPs
        x= self.fc1(x)  # -> (64)
        x= self.act(x)  # -> (64)

        x= self.fc2(x)  # -> (35)
        y= self.out(x)  # -> (35)

        return y

model= ryM(in_chs= 1, out_cls=35)

# ryM, Test@epoch= 13, acc=【0.8706】, [9581/11005]

# Train@epoch= 15, Loss: 0.205410
# ryM,  Test@epoch= 15, acc= 0.8642, [9510/11005]

# ryM,  Test@epoch= 15, acc= 0.8531, [9388/11005]
# M6,   Test Epoch: 24  Accuracy: 9362/11005 (85%)
# M5_1, Test Epoch: 21  Accuracy: 8905/11005 (81%)

#%%
#%%
# load the weights
# check the availability of "model.pt"
#'''
if os.path.isfile('model.pt'):
    model.load_state_dict(torch.load('model.pt'))
#'''
#%%

model.to(device)
print(model)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

n = count_parameters(model)
print("Number of parameters: %s" % n)


# %%
# using CrossEntropyLoss as our loss function
# criterion= nn.CrossEntropyLoss()

# using negative log likelihood loss as the loss function
loss_fn=   nn.NLLLoss()

optimizer= optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0001)
scheduler= optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)  
# reduce the learning after 10 epochs by a factor of 10

# %%
def train(model, epoch, log_interval, lossL= []):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):

        data=   data.to(device)
        target= target.to(device)

        output= model(data)

        loss= loss_fn(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        lossL += [loss.item()]

        # update progress bar
        pbar.update(pbar_update)

    print(f"\nTrain@{epoch= }, Loss: {lossL[-1]:.6f}")
    return lossL


# %%
def number_of_correct(pred, target):
    # count number of correct predictions
    return pred.squeeze().eq(target).sum().item()

def get_likely_index(tensor):
    # find most likely label index for each element in the batch
    return tensor.argmax(dim=-1)

def test(model, epoch=1):
    model.eval()
    correct = 0
    for data, target in test_loader:

        data=   data.to(device)
        target= target.to(device)

        # apply transform and model on whole batch directly on device
        # data = transform(data)
        output= model(data)

        pred = get_likely_index(output)
        correct += number_of_correct(pred, target)

        # update progress bar
        pbar.update(pbar_update)

    acc= correct/len(test_loader.dataset)
    print(f"\nTest@{epoch= }, acc=【{acc:.4f}】, [{correct}/{len(test_loader.dataset)}]\n")
    
    return acc

# %%
log_interval= 100
test_interval= 5
n_epoch=       30

pbar_update = 1 / (len(train_loader) + len(test_loader))
lossL= []
accL= []

with tqdm(total=n_epoch) as pbar:
    for epoch in range(1, n_epoch + 1):

        lossL= train(model, epoch, log_interval, lossL)
        
        if (epoch % test_interval == 0) or (epoch == 1):
            acc= test(model, epoch)
            accL += [acc]
            if acc >= max(accL):
                torch.save(model.state_dict(), "model.pt")
            

        scheduler.step()
#%%
# finally, test the model on the test set
# save the trained model
acc= test(model)
accL += [acc]
if acc >= max(accL):
    torch.save(model.state_dict(), "model.pt")


# %%

# plot the loss
plt.figure()
plt.subplot(1,2,1)
plt.plot(lossL)
plt.xlabel("batch")
plt.ylabel("loss")
#plt.show()

# plot the accuracy
plt.subplot(1,2,2)
plt.plot(accL)
plt.xlabel(f"epoch/{test_interval}")
plt.ylabel("accuracy")
plt.show()

# %%

# initialize the model
mdl= ryM(in_chs= 1, out_cls=35)
mdl.to(device)

# load the weights

# get the directory of the current file
# dir= os.path.dirname(os.path.abspath(__file__))
# get the path to the file
path= 'model.pt' #os.path.join(dir, 'model.pt')

mdl.load_state_dict(torch.load(path))

# %%
# test the model
mdl.eval()
correct = 0
with torch.no_grad():
    for data, target in test_loader:

        data=   data.to(device)
        target= target.to(device)

        # apply transform and model on whole batch directly on device
        # data = transform(data)
        output= mdl(data)

        pred = get_likely_index(output)
        correct += number_of_correct(pred, target)

acc= correct/len(test_loader.dataset)
print(f"\n{acc= :.4f}, [{correct}/{len(test_loader.dataset)}]\n")

