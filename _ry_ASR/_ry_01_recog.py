
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

#%%

# get data from google drive
#from google.colab import drive
#drive.mount('/content/drive')

# add path to sys
#sys.path.append('/content/drive/MyDrive/Colab Notebooks/_ry_ASR')

#"https://drive.google.com/file/d/1-3JF7rhFBpfajaIM-_NjKgg8WXHJ_fP9/view?usp=share_link"

#"https://drive.google.com/file/d/1-3_AWSuw9m195PKgixouOR_2LDr_bAEE/view?usp=share_link"


# get data from link above
# !gdown --id 1-3JF7rhFBpfajaIM-_NjKgg8WXHJ_fP9
# !gdown --id 1-3_AWSuw9m195PKgixouOR_2LDr_bAEE



# %%
from torchaudio.datasets import SPEECHCOMMANDS
import os




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

#model= ryM(in_chs= 1, out_cls=35)

# ryM, Test@epoch= 13, acc=【0.8706】, [9581/11005]

# Train@epoch= 15, Loss: 0.205410
# ryM,  Test@epoch= 15, acc= 0.8642, [9510/11005]

# ryM,  Test@epoch= 15, acc= 0.8531, [9388/11005]
# M6,   Test Epoch: 24  Accuracy: 9362/11005 (85%)
# M5_1, Test Epoch: 21  Accuracy: 8905/11005 (81%)
# %%


def get_likely_index(tensor):
    # find most likely label index for each element in the batch
    return tensor.argmax(dim=-1)


# initialize the model
mdl= ryM(in_chs= 1, out_cls=35)
mdl.to(device)

# load the weights

mdl.load_state_dict(torch.load('model.pt'))

# only in inference mode
mdl.eval()


# %%
# get the test data set to test the model on

# get the test data set
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
test_set= SubsetSC("testing")
waveform, sample_rate, label, speaker_id, utterance_number = test_set[0]
len(test_set) # 11_005

# %%
# put the test data into a data loader

from torch.utils.data import DataLoader

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


batch_size= 2048


test_loader = torch.utils.data.DataLoader(
    test_set,
    batch_size=batch_size,
    shuffle=False,
    collate_fn=collate_fn,
    pin_memory= True # CUDA only, much faster
)

len(test_loader)

# using the test data loader, test the model

def number_of_correct(pred, target):
    # count number of correct predictions
    return pred.squeeze().eq(target).sum().item()

def get_likely_index(tensor):
    # find most likely label index for each element in the batch
    return tensor.argmax(dim=-1)

def test(model, test_loader, epoch=0):
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
        #pbar.update(pbar_update)

    acc= correct/len(test_loader.dataset)
    print(f"\nTest@{epoch= }, acc=【{acc:.4f}】, [{correct}/{len(test_loader.dataset)}]\n")
    
    return acc

#%%
# check the speed of the model

import time

t0= time.time()
acc= test(mdl, test_loader)
t1= time.time()

print(f"{t1-t0= :.4f} seconds")
print(f'{acc= :.4f}')

#%%

x, _, y, *_= test_loader.dataset[1001]
x= x.squeeze()
x,y

# %%
# get the waveforms from the currrent dircetory
# and test the model on them

# get the waveform from ryTest.wav
# and test the model on it
# ryTest.wav is a recording of the words of several words
# the words are: ""zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", ...

import torchaudio

# load the waveform

# get the directory of the current file
dir= os.path.dirname(os.path.abspath(__file__))

# get the path to the file
fn= "ryTest.wav"
path= os.path.join(dir, fn)


# %%

#%%
# %%
# get the waveforms from the currrent dircetory
# and test the model on them

# get the waveform from ryTest.wav
# and test the model on it
# ryTest.wav is a recording of the words of several words
# the words are: 
# "zero", "one", "two", "three", "four", 
# "five", "six", "seven", "eight", "nine", 
# "forward", "backward", 
# "up", "down", "left", "right",



# load the waveform
waveform, sample_rate= torchaudio.load(path)

# plot the waveform
plt.figure()
plt.plot(waveform.t().numpy())
plt.show()

# %%
# segment the waveform into words
# segment the waveform into 1 second segments

# get the number of samples in 1 second
n_samples= sample_rate

# get the number of samples in the waveform
n_samples_waveform= waveform.shape[1]

# get the number of segments
n_segments= n_samples_waveform//n_samples

# get the segments
segments= torch.split(waveform, n_samples, dim=1)

# plot the segments
plt.figure()
for i in range(n_segments):
    plt.subplot(n_segments, 1, i+1)
    plt.plot(segments[i].t().numpy())
plt.show()

# %%
# use the model to predict the words in the segments

# %%
# stack the segments into a batch of tensors
# discard the last segment if it is not of size n_samples
# or pad it with zeros if it is smaller than n_samples

# get the number of samples in the last segment
n_samples_last_segment= segments[-1].shape[1]

# check if the last segment is of size n_samples
if n_samples_last_segment != n_samples:
    # pad the last segment with zeros
    seg= torch.cat((
        segments[-1], torch.zeros(
            (1, n_samples-n_samples_last_segment))), 
        dim=1 
        )
#%%
# stack the segments into a batch of tensors
sss= torch.stack([*segments[:-1], seg], dim=0)

# %%
# use the model to predict the words in the segs

sss_cuda= sss.to(device)

# get the prediction
pred= mdl(sss_cuda)
# get the most likely index
pred= get_likely_index(pred)
# get the word
result= [labels[p] for p in pred]

print(f'{result= }')

# plot the segments
plt.figure()
for i in range(n_segments):
    plt.subplot(n_segments//2+1, 2, i+1)
    plt.plot(sss[i].t().numpy())
    plt.title(result[i])
plt.show()
# %%









# %%
