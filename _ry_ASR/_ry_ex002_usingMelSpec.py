'''
SPEECH COMMAND CLASSIFICATION WITH TORCHAUDIO

ry modify from:
https://pytorch.org/tutorials/intermediate/speech_command_classification_with_torchaudio_tutorial.html

for the purpose of learning pytorch and torchaudio
'''

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

device= torch.device("cuda" if torch.cuda.is_available() else 
                     "cpu")
print(device)
#%%
from torchaudio.datasets import SPEECHCOMMANDS
import os

data_path= "L:\\_ryDatasets\\_expData_"
# check if the data is already downloaded
if not os.path.isdir(data_path):
    print("Downloading data...")
    os.makedirs(data_path)
    train_set= SPEECHCOMMANDS(data_path, download= True)
    test_set= SPEECHCOMMANDS(data_path, download= True)
    print("Downloading complete!")
else:
    print("Data already downloaded!")


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
train_set= SubsetSC("training")
test_set=  SubsetSC("testing")
waveform, sample_rate, label, speaker_id, utterance_number= train_set[0]

# show the size of training and testing split
print("Training set size:", len(train_set))
print("Test set size:", len(test_set))

# show infomation about the data
print("Shape of waveform: {}".format(waveform.size()))
print("Sample rate of waveform: {}".format(sample_rate))

#plt.plot(waveform.t().numpy())
#plt.show()

print("Label: {}".format(label))
print("Speaker ID: {}".format(speaker_id))
print("Utterance number: {}".format(utterance_number))

#%% put info of the randomly chosed 10 data into a dataframe

import pandas as pd
waveform_pre10= []
sample_rate_pre10= []
label_pre10= []
speaker_id_pre10= []
utterance_number_pre10= []

for j in range(10):
    
    # randomly choose one data from the training set
    i= torch.randint(len(train_set), size= (1,)).item()

    waveform_pre10.append(train_set[i][0].shape)
    sample_rate_pre10.append(train_set[i][1])
    label_pre10.append(train_set[i][2])
    speaker_id_pre10.append(train_set[i][3])
    utterance_number_pre10.append(train_set[i][4])

df= pd.DataFrame({"waveform.shape": waveform_pre10,
                    "sample_rate": sample_rate_pre10,
                    "label": label_pre10,
                    "speaker_id": speaker_id_pre10,
                    "utterance_number": utterance_number_pre10})

print(df)
#%%
# find all labels
# it takes a while to run this cell, so we save the result to disk 
# or put it here as a comment

# labels = sorted(list(set(datapoint[2] for datapoint in train_set)))
# save the list of labels to disk for prediction later in the notebook
# with open("labels.txt", "w") as fp:
#    fp.write("\n".join(labels))

labels= [
 'backward', 'bed',     'bird',     'cat',      'dog',
 'down',    'eight',    'five',     'follow',   'forward',
 'four',    'go',       'happy',    'house',    'learn',
 'left',    'marvin',   'nine',     'no',       'off',
 'on',      'one',      'right',    'seven',    'sheila',
 'six',     'stop',     'three',    'tree',     'two',
 'up',      'visual',   'wow',      'yes',      'zero'
]

#%%
# downsampling the data to 8000Hz
# why do we need to do this?
# because the original sample rate is 16000Hz, which is too high for our model
# we can downsample the data to 8000Hz to make the model run faster
# and we can still hear the sound clearly

waveform, sample_rate, label, speaker_id, utterance_number= train_set[0]
waveform.shape, sample_rate, label

#%% downSample the data to 8_000Hz

new_sample_rate= 8_000

transform= \
transform_reSample= torchaudio.transforms.Resample(
    orig_freq= sample_rate, 
    new_freq=  new_sample_rate)

waveform= waveform.squeeze()
transformed_waveform= transform(waveform)

ipd.display(ipd.Audio(
    waveform.numpy(),    
    rate=sample_rate))

ipd.display(ipd.Audio(
    transformed_waveform.numpy(), 
    rate=new_sample_rate))

#%% transform the data to mel spectrogram
#transform= \
transform_melSpec= torchaudio.transforms.MelSpectrogram(
    sample_rate= sample_rate,
    n_fft= 512,      #1024,
    hop_length= 161, #512,
    n_mels= 80       #64
)

waveform= waveform.to('cuda')
transform.to('cuda')

waveform= waveform.squeeze()
transformed_waveform= transform(waveform)
transformed_waveform.shape


# %% encode/decode a label

def label_to_index(word):
    # Return the position of the word in labels
    return torch.tensor(labels.index(word))

def index_to_label(index):
    # Return the word corresponding to the index in labels
    # This is the inverse of label_to_index
    return labels[index]

lbl=  "yes"
idx=  label_to_index(lbl)
lbl2= index_to_label(idx)

print(lbl, "-->", idx, "-->", lbl2)
# %%

# collate function
# we need to collate the data to make sure all the data have the same size
# we will pad the data with zeros to make them have the same size

def pad_sequence(batch):
    # Make all tensor in a batch the same length by padding with zeros
    batch= [item.t() for item in batch]
    batch= torch.nn.utils.rnn.pad_sequence(
        batch, 
        batch_first=True, 
        padding_value=0.)
    return batch.permute(0, 2, 1)



def collate_fn(batch):

    # A data tuple has the form:
    # waveform, sample_rate, label, speaker_id, utterance_number

    tensors, targets = [], []

    # Gather in lists, and encode labels as indices
    # put the data into cuda

    for waveform, _, label, *_ in batch:

        tensors += [waveform]        
        targets += [label_to_index(label)]

    # Group the list of tensors into a batched tensor
    tensors = pad_sequence(tensors)
    targets = torch.stack(targets)

    #tensors= tensors.to(device)
    #targets= targets.to(device)

    #tensors= tensors.squeeze()
    #tensors= transform(tensors)

    return tensors, targets

batch_size = 1000 #1024 #256

if device == "cuda":
    num_workers = 1
    pin_memory = True
else:
    num_workers = 0
    pin_memory = False

#%%
train_loader= torch.utils.data.DataLoader(
    train_set,
    batch_size=batch_size,
    shuffle=True,
    collate_fn= collate_fn,
    num_workers=num_workers,
    pin_memory=pin_memory,
)

test_loader= torch.utils.data.DataLoader(
    test_set,
    batch_size=batch_size,
    shuffle=False,
    drop_last=False,
    collate_fn=collate_fn,
    num_workers=num_workers,
    pin_memory=pin_memory
)

# %%
# we will put the data into the model in batches
# each batch contains 1000 data
# we will simulate the process here 
# (without training the model, 
# just to see how the data be pulled out from the dataloader)
# we will print out the shape of the data in each batch
# and the first 5 and last 5 labels in each batch
#'''
for n, data in enumerate(train_loader):
    w,l= data
    print(n, w.shape, l[:5], l[-5:])

for n, data in enumerate(test_loader):
    w,l= data
    print(n, w.shape, l[:5], l[-5:])
#'''


#%% Using MelSpectrogram from whisper library

import os.path
import numpy as np
from whisper.audio import load_audio, log_mel_spectrogram, mel_filters

from whisper.audio import SAMPLE_RATE, N_MELS, N_FFT, HOP_LENGTH

# hard-coded audio hyperparameters in whisper library
'''
SAMPLE_RATE= 16000
N_FFT= 400      # 400/16000= 25ms
N_MELS= 80 # min freq= 
HOP_LENGTH= 160 # 160/16000= 10ms
'''

audio_fn=   "backward.wav"
audio_path= os.path.join(os.path.dirname(__file__), audio_fn)

audio=          load_audio(audio_path)

mel_from_audio= log_mel_spectrogram(audio)
mel_from_file=  log_mel_spectrogram(audio_path)

audio.shape, mel_from_audio.shape, mel_from_file.shape

# %%  plot the mel spectrogram
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 4))
plt.imshow(mel_from_audio, 
           cmap='rainbow', 
           interpolation='nearest', 
           origin='lower')
plt.colorbar(format='%+2.0f dB')
plt.title('Mel spectrogram')
plt.tight_layout()
plt.show()

#%%

import torchaudio

audio_fn=   "backward.wav"
audio_path= os.path.join(os.path.dirname(__file__), audio_fn)

waveform, sample_rate= torchaudio.load(audio_path, normalize=True)
melSpecTransform= torchaudio.transforms.MelSpectrogram(
    sample_rate= sample_rate,
    n_fft= N_FFT,           # 400 # 400/16000= 25ms
    hop_length= HOP_LENGTH, # 160 # 160/16000= 10ms
    n_mels= N_MELS          # 80
    )
mel_specgram= melSpecTransform(waveform)    # (channel, n_mels, time)
mel_specgram= mel_specgram[..., :-1] 
# remove the last column 
# (channel, n_mels, time-1)
# 這是 whisper library 的 作法，讓 1 sec= 100 frames 而非 101 frames


# %% ## Define the Network

## Define the Network

'''
For this tutorial we will use a convolutional neural network to process the raw audio data. 

Usually more advanced transforms are applied to the audio data, 
however CNNs can be used to accurately process the raw data.

The specific architecture is modeled after the M5 network architecture
described in [this paper](https://arxiv.org/pdf/1610.00087.pdf)_. 
An important aspect of models processing raw audio data is the receptive
field of their first layer’s filters. 

Our model’s first filter is length 80 so when processing audio sampled at 8kHz 
the receptive field is around 10ms (and at 4kHz, around 20 ms). 

This size is similar to speech processing applications 
that often use receptive fields ranging from 20ms to 40ms.
'''
#%%
class M5_ori(nn.Module):
    def __init__(self, 
                 n_input=   1, # number of input channel 
                 n_output= 35, # number of output classes
                 kernel_size= 80, # length of the convolutional kernel
                 stride=      16, # stride of the first convolutional layer
                 n_channel=   32  # number of channels of the first convolutional layer
                 ):
        super().__init__()
        
        self.conv1= nn.Conv1d(
            n_input, 
            n_channel, 
            kernel_size= kernel_size, 
            stride=      stride)
        
        self.bn1=   nn.BatchNorm1d(n_channel)
        self.pool1= nn.MaxPool1d(4)

        self.conv2= nn.Conv1d(
            n_channel, 
            n_channel, 
            kernel_size=3)
        
        self.bn2=   nn.BatchNorm1d(n_channel)
        self.pool2= nn.MaxPool1d(4)
        
        self.conv3= nn.Conv1d(
            n_channel, 
            2 * n_channel, 
            kernel_size=3)
        
        self.bn3=   nn.BatchNorm1d(2 * n_channel)
        self.pool3= nn.MaxPool1d(4)
        
        self.conv4= nn.Conv1d(
            2 * n_channel, 
            2 * n_channel, 
            kernel_size=3)
        
        self.bn4=   nn.BatchNorm1d(2 * n_channel)
        self.pool4= nn.MaxPool1d(4)
        
        self.fc1=   nn.Linear(2 * n_channel, n_output)

    def forward(self, x):
        
        x= self.conv1(x)        # 1,8000 --> 32, 496
        x= F.relu(self.bn1(x))  # 32, 496 --> 32, 496
        x= self.pool1(x)        # 32, 496 --> 32, 124

        x= self.conv2(x)        # 32, 124 --> 32, 122
        x= F.relu(self.bn2(x))  # 32, 122 --> 32, 122
        x= self.pool2(x)        # 32, 122 --> 32, 30
        
        x= self.conv3(x)        # 32,30 --> 64, 28
        x= F.relu(self.bn3(x))  # 64, 28 --> 64, 28
        x= self.pool3(x)        # 64, 28 --> 64, 7
        
        x= self.conv4(x)        # 64, 7 --> 64, 5
        x= F.relu(self.bn4(x))  # 64, 5 --> 64, 5
        x= self.pool4(x)        # 64, 5 --> 64, 1
        
        x= F.avg_pool1d(x, x.shape[-1]) # 64, 1 
        x= x.permute(0, 2, 1)           # 1, 64
        x= self.fc1(x)                  # 1,64 --> 1,35
        
        y= F.log_softmax(x, dim=2)      # 1,35 --> 1,35

        return y    # 1,35 


# ry_Edit
class M5_ver01(nn.Module):
    def __init__(self, 
                 n_input=   1, # number of input channel 
                 n_output= 35, # number of output classes
                 kernel_size= 80, # length of the convolutional kernel
                 stride=      16, # stride of the first convolutional layer
                 n_channel=   32  # number of channels of the first convolutional layer
                 ):
        super().__init__()
        
        self.conv1= nn.Conv1d(
            n_input, 
            n_channel, 
            kernel_size= kernel_size, 
            stride=      stride,
            #padding='same'
            )
        
        self.bn1=   nn.BatchNorm1d(n_channel)
        self.pool1= nn.MaxPool1d(4)

        self.conv2= nn.Conv1d(
            n_channel, 
            n_channel, 
            kernel_size=3,
            padding='same')
        
        self.bn2=   nn.BatchNorm1d(n_channel)
        self.pool2= nn.MaxPool1d(4)
        
        self.conv3= nn.Conv1d(
            n_channel, 
            2 * n_channel, 
            kernel_size=3,
            padding='same')
        
        self.bn3=   nn.BatchNorm1d(2 * n_channel)
        self.pool3= nn.MaxPool1d(4)
        
        self.conv4= nn.Conv1d(
            2 * n_channel, 
            2 * n_channel, 
            kernel_size=3, 
            padding='same')
        
        self.bn4=   nn.BatchNorm1d(2 * n_channel)
        self.pool4= nn.MaxPool1d(4)
        
        self.fc1=   nn.Linear(2 * n_channel, n_output)

    def forward(self, x):
        
        x= self.conv1(x)        # 1,8000 --> 32, 496
        x= F.relu(self.bn1(x))  # 32, 496 --> 32, 496
        x= self.pool1(x)        # 32, 496 --> 32, 124

        x= self.conv2(x)        # 32, 124 --> 32, 124
        x= F.relu(self.bn2(x))  # 32, 124 --> 32, 124
        x= self.pool2(x)        # 32, 124 --> 32, 31
        
        x= self.conv3(x)        # 32,31 --> 64, 31
        x= F.relu(self.bn3(x))  # 64, 31 --> 64, 31
        x= self.pool3(x)        # 64, 31 --> 64, 7
        
        x= self.conv4(x)        # 64, 7 --> 64, 7
        x= F.relu(self.bn4(x))  # 64, 7 --> 64, 7
        x= self.pool4(x)        # 64, 7 --> 64, 1
        
        x= F.avg_pool1d(x, x.shape[-1]) # 64, 1 
        x= x.permute(0, 2, 1)           # 1, 64
        x= self.fc1(x)                  # 1,64 --> 1,35
        
        y= F.log_softmax(x, dim=2)      # 1,35 --> 1,35

        return y    # 1,35 



#%%
class M5(nn.Module):
    def __init__(self, 
                 n_input=  80, # number of input channel 
                 n_output= 35, # number of output classes
                 #kernel_size= 80, # length of the convolutional kernel
                 #stride=      16, # stride of the first convolutional layer
                 n_channel=   32  # number of channels of the first convolutional layer
                 ):
        super().__init__()
        
        '''
        self.conv1= nn.Conv1d(
            n_input, 
            n_channel, 
            kernel_size= kernel_size, 
            stride=      stride,
            #padding='same'
            )
        
        self.bn1=   nn.BatchNorm1d(n_channel)
        self.pool1= nn.MaxPool1d(4)
        '''

        self.conv2= nn.Conv1d(
            n_input, #n_channel, 
            n_channel, 
            kernel_size=3,
            padding='same')
        
        self.bn2=   nn.BatchNorm1d(n_channel)
        self.pool2= nn.MaxPool1d(4)
        
        self.conv3= nn.Conv1d(
            n_channel, 
            2 * n_channel, 
            kernel_size=3,
            padding='same')
        
        self.bn3=   nn.BatchNorm1d(2 * n_channel)
        self.pool3= nn.MaxPool1d(4)
        
        self.conv4= nn.Conv1d(
            2 * n_channel, 
            2 * n_channel, 
            kernel_size=3, 
            padding='same')
        
        self.bn4=   nn.BatchNorm1d(2 * n_channel)
        self.pool4= nn.MaxPool1d(4)
        
        self.fc1=   nn.Linear(2 * n_channel, n_output)

    def forward(self, x):
        
        '''
        x= self.conv1(x)        # 1,8000 --> 32, 496
        x= F.relu(self.bn1(x))  # 32, 496 --> 32, 496
        x= self.pool1(x)        # 32, 496 --> 32, 124
        '''

        

        x= x.squeeze(1)         # 1, 80,101 --> 80, 101

        x= self.conv2(x)        # 32, 124 --> 32, 124  # 1, 80,101
        x= F.relu(self.bn2(x))  # 32, 124 --> 32, 124
        x= self.pool2(x)        # 32, 124 --> 32, 31
        
        x= self.conv3(x)        # 32,31 --> 64, 31
        x= F.relu(self.bn3(x))  # 64, 31 --> 64, 31
        x= self.pool3(x)        # 64, 31 --> 64, 7
        
        x= self.conv4(x)        # 64, 7 --> 64, 7
        x= F.relu(self.bn4(x))  # 64, 7 --> 64, 7
        x= self.pool4(x)        # 64, 7 --> 64, 1
        
        x= F.avg_pool1d(x, x.shape[-1]) # 64, 1 
        x= x.permute(0, 2, 1)           # 1, 64
        x= self.fc1(x)                  # 1,64 --> 1,35
        
        y= F.log_softmax(x, dim=2)      # 1,35 --> 1,35

        return y    # 1,35 



#%%

model= M5(n_input= transformed_waveform.shape[-2], 
          n_output=len(labels))

model.to(device)
print(model)


def count_parameters(model):
    return sum(p.numel() 
               for p in model.parameters() 
               if p.requires_grad)

n= count_parameters(model)
print("Number of parameters: %s" % n)
# %%
paras= model.parameters()

# %%
'''
We will use the same optimization technique used in the paper, 
an Adam optimizer with weight decay set to 0.0001. 

At first, we will train with a learning rate of 0.01, 
but we will use a ``scheduler`` to decrease it to 0.001 
during training after 20 epochs.
'''

optimizer= optim.Adam(
    model.parameters(), 
    lr=0.01, 
    weight_decay=0.0001)

scheduler= optim.lr_scheduler.StepLR(
    optimizer, 
    step_size=20, 
    gamma= 0.1)  

# reduce the learning after 20 epochs by a factor of 10


# %%
## Training and Testing the Network
'''
Now let’s define a training function that will feed our training data
into the model and perform the backward pass and optimization steps. For
training, the loss we will use is the negative log-likelihood. The
network will then be tested after each epoch to see how the accuracy
varies during the training.
'''

def train(model, epoch, log_interval):
    model.train()
    
    losses= []
    for batch_idx, (data, target) in enumerate(train_loader):

        data = data.to(device)
        target = target.to(device)

        # apply transform and model on whole batch directly on device
        data = transform(data)
        output = model(data)

        # negative log-likelihood for a tensor of size (batch x 1 x n_output)
        loss = F.nll_loss(output.squeeze(), target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print training stats
        if True: #batch_idx % log_interval == 0:
            print(f"""
            Train Epoch: {epoch} 
            [{batch_idx * len(data)}/{len(train_loader.dataset)} 
            ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}
            """)

        # update progress bar
        pbar.update(pbar_update)
        # record loss
        losses.append(loss.item())
    
    return losses

def number_of_correct(pred, target):
    # count number of correct predictions
    return pred.squeeze().eq(target).sum().item()


def get_likely_index(tensor):
    # find most likely label index for each element in the batch
    return tensor.argmax(dim=-1)


def test(model, epoch):
    model.eval()
    correct = 0
    for data, target in test_loader:

        data = data.to(device)
        target = target.to(device)

        # apply transform and model on whole batch directly on device
        data = transform(data)
        output = model(data)

        pred = get_likely_index(output)
        correct += number_of_correct(pred, target)

        # update progress bar
        pbar.update(pbar_update)

    acc= correct / len(test_loader.dataset)

    print(f"""
    Test Epoch: {epoch}
    Accuracy: {acc} ({correct}/{len(test_loader.dataset)}) 
    """)

    return acc

# %%
log_interval= 100
n_epoch=       2

pbar_update = 1 / (len(train_loader) + len(test_loader))

losses= []

# The transform needs to live on the same device as the model and the data.
transform= transform.to(device)

with tqdm(total=n_epoch) as pbar:
    for epoch in range(1, n_epoch + 1):
        loss1= train(model, epoch, log_interval)
        losses += [loss1]
        test( model, epoch)
        scheduler.step()
print(losses)

# %%
# save model
torch.save(model.state_dict(), "model.pth")
#%%
# load model
m5= M5(
    n_input=  transformed_waveform.shape[0], 
    n_output= len(labels))
m5.load_state_dict(torch.load("model.pth"))

# put the model to the device
m5= m5.to(device)

# test the model
test(m5, 1)


# %%
# train one more epoch and then test again

train(m5, 1, log_interval)
test( m5, 1)


# %%
