'''
file: _ry_02_models.py
date: 2023-05-09
author: Renyuan Lyu

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

import os
#os.system('gdown --id 1-3JF7rhFBpfajaIM-_NjKgg8WXHJ_fP9')
#os.system('gdown --id 1-3_AWSuw9m195PKgixouOR_2LDr_bAEE')

#  display the data
ipd.display(ipd.Audio('ryTest.wav'))


#%%

#%%  Visualize the speech data

# show the waveform and spectrogram
waveform, sample_rate= torchaudio.load('ryTest.wav')
waveform= waveform.squeeze()
T= waveform.shape[0] / sample_rate
print(f'T= {T} seconds')

# padding to 20 seconds
if T < 20:
    waveform= torch.cat([waveform, torch.zeros(int(20*sample_rate - T*sample_rate))])
    T= waveform.shape[0] / sample_rate

print(f'T= {T} seconds')

specgram= torchaudio.transforms.Spectrogram()(waveform)
melsgram= torchaudio.transforms.MelSpectrogram()(waveform)

plt.subplot(3,1,1)
plt.plot(waveform.numpy())
plt.subplot(3,1,2)
plt.imshow(specgram.log().numpy())
plt.subplot(3,1,3)
plt.imshow(melsgram.log().numpy())


#%%

x= waveform[:sample_rate*10] # 10 seconds

x.shape
#%% 嘗試 新的建模方式


# get the number of parameters in the model
def get_n_params(model):
    np= 0
    for p in model.parameters():
        np += p.numel()
    return np

ryMelsgram= torchaudio.transforms.MelSpectrogram(
    sample_rate= 16_000,   # 16 kHz
    hop_length=  160,      # 10 ms
    n_fft=       160*2,    # 20 ms
    n_mels=      64, 
)

class ryMelsgram1d(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        x= ryMelsgram(x)
        return x

class ryMelsgram2d(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        x= ryMelsgram(x)   
        x= x.unsqueeze(dim=-3) 
        return x

# a torch layer to average the output of the last layer

class ryAvgPool1d(nn.Module):
    def __init__(self, o_size= 1):
        super().__init__()
        self.pool= nn.AdaptiveAvgPool1d(o_size) # output size= 1
    def forward(self, x):
        x= self.pool(x)
        x= x.squeeze(dim=-1)
        return x

class ryAvgPool2d(nn.Module):
    def __init__(self, o_size= 1):
        super().__init__()
        self.pool= nn.AdaptiveAvgPool2d(o_size) # output size= 1
    def forward(self, x):
        x= self.pool(x)
        x= x.squeeze(dim=(-2,-1))
        return x

# Using Sequential to create a small model. When `model` is run,
#%%
import collections

layers= collections.OrderedDict([
    ('c1', nn.Conv1d(1,    64, kernel_size= 320, stride= 160)),
    ('c2', nn.Conv1d(64,  128, kernel_size=   4, stride=   2)),
    ('c3', nn.Conv1d(128, 256, kernel_size=   4, stride=   2)),
    ('c4', nn.Conv1d(256, 256, kernel_size=   4, stride=   2)),
    ('c5', nn.Conv1d(256, 256, kernel_size=   4, stride=   2)),
    ('p1', ryAvgPool1d()), 
    ('l1', nn.Linear(256, 128)),
    ('l2', nn.Linear(128, 35))
    ])


model= nn.Sequential(layers)

# get the number of parameters
n_params= sum(p.numel() for p in model.parameters())
print(f'{n_params= }')

xB= x.view(-1,1,16000)
model(xB).shape


# %%

#%%


#%%
import collections

layers= collections.OrderedDict([
    #('c1', nn.Conv1d(1,    64, kernel_size= 320, stride= 160)),
    ('f1', ryMelsgram1d()),
    ('c2', nn.Conv1d( 64,128, kernel_size= 4, stride=  2)),
    ('c3', nn.Conv1d(128,256, kernel_size= 4, stride=  2)),
    ('c4', nn.Conv1d(256,256, kernel_size= 4, stride=  2)),
    ('c5', nn.Conv1d(256,256, kernel_size= 4, stride=  2)),
    ('p7', ryAvgPool1d()), 
    ('l8', nn.Linear(256, 128)),
    ('l9', nn.Linear(128,  35))
    ])


model= nn.Sequential(layers)
get_n_params(model)
#%%
xB= x.view(-1,16000)
model(xB).shape

# %%
layers= collections.OrderedDict([
    #('c1', nn.Conv1d(1,    64, kernel_size= 320, stride= 160)),
    ('f1', ryMelsgram2d()),
    ('c2', nn.Conv2d(   1,  64, kernel_size= 4, stride=  2)),
    ('c3', nn.Conv2d(  64, 128, kernel_size= 4, stride=  2)),
    ('c4', nn.Conv2d( 128, 256, kernel_size= 4, stride=  2)),
    ('c5', nn.Conv2d( 256, 256, kernel_size= 4, stride=  2)),
    ('p7', ryAvgPool2d()),
    ('l8', nn.Linear(256, 128)),
    ('l9', nn.Linear(128,  35))
    ])

model= nn.Sequential(layers)
get_n_params(model)
#%%
xB= x.view((-1,16_000))

model(xB).shape


# %%
