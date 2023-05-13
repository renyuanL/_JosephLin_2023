'''
file: _ry_02_recog.py
date: 2023-05-08
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

# %%

# 1. load the pre-trained model, and a test audio file by Renyuan Lyu
# 2. visualize the audio file
# 3. recognize the audio file
# 4. visualize the recognition result
# 5. do some analysis


import os

#  display the data
fn_wav= 'ry35words.wav'



#%%  Visualize the speech data

# show the waveform and spectrogram
waveform, sample_rate= torchaudio.load(fn_wav)
specgram= torchaudio.transforms.Spectrogram()(waveform)

plt.subplot(2,1,1)
plt.plot(waveform.t().numpy())
plt.subplot(2,1,2)
plt.imshow(specgram.log2()[0,:,:].numpy(), cmap='rainbow')


#%%

x= waveform[:, 0:48_000] # 3 seconds
x= x.squeeze() # (1,48000) -> (48000)

#  display the data
ipd.display(ipd.Audio(x, rate= sample_rate))

# show the waveform and spectrogram
X_ft= specgram= torchaudio.transforms.Spectrogram()(x)

plt.figure(figsize=(10,5))
#plt.subplot(3,1,1)
plt.plot(x.numpy())

plt.figure(figsize=(10,5))
#plt.subplot(3,1,2)
plt.imshow(X_ft.log()[:,:].numpy(), cmap='rainbow')

# mel scale spectrogram
X_mt= melsgram=  torchaudio.transforms.MelSpectrogram()(x)

# show the mel scale spectrogram
plt.figure(figsize=(10,5))
#plt.subplot(3,1,3)
plt.imshow(X_mt.log()[:,:].numpy(), cmap='rainbow')


# %% The Speech Commands Dataset, 
# 
# about 105,000 audio files, 35 classes, 1 second each
# how many speakers?  how many utterances per speaker?
#
# https://arxiv.org/pdf/1804.03209.pdf
# https://www.tensorflow.org/datasets/catalog/speech_commands
# April 2018, 5 years ago (from 2023)
#

from ryModels import theLabels, label_to_index, index_to_label 
from ryModels import ryM10 as ryM

labels= theLabels

print(f'{labels= }')


#%%
device = torch.device("cuda" if torch.cuda.is_available() else 
                      "cpu")
print(device)

# initialize the model and load the weights
mdl= ryM(in_chs= 1, out_cls=35)

model_fn= 'ryM.pt' # 'ryM10.pt'

mdl.load_state_dict(
    torch.load(model_fn, map_location=device)
    )

mdl.eval() # only in inference mode
mdl.to(device)

print(mdl)
#%%
# ry Example
# load the waveform

t0= 0 #16_000*10
x= waveform[:, t0:t0+16_000*5] # 5 seconds
x= x.squeeze() # (1,80000) -> (80000)

xB= x.reshape(-1,1,16000)
xB= xB.cuda()
y=  mdl(xB)
y=  y.squeeze()
y=  y.argmax(dim=-1)
y=  [index_to_label(q) for q in y]
y

#%%
X_mt= melsgram= \
torchaudio\
.transforms\
.MelSpectrogram().to('cuda')(xB).squeeze()
X_mt.shape # torch.Size([3, 128, 81])



#%%
# load the waveform
#fn= fn0= "backward.wav"
fn= fn1= fn_wav
waveform, sample_rate= torchaudio.load(fn)

print(waveform.shape, sample_rate)

# plot the waveform
plt.figure()
plt.plot(waveform.t().numpy())
plt.show()

#%%
def get_likely_index(tensor):
    # find most likely label index for each element in the batch
    return tensor.argmax(dim=-1)

def get_likely_score(tensor):
    # find most likely label index for each element in the batch
    return tensor.max(axis=-1).values

#%%
x=  waveform
xB= x[None,...]
xB= xB.to(device)

# recognize the waveform
with torch.no_grad():
    yB= mdl(xB)
    yB= get_likely_index(yB)
    yB= index_to_label(yB)
    print(f'{yB= }')

#%%
# split the waveform into chunks, and recognize each chunk
# 這一段是為了讓模型可以接受任意長度的音訊檔案
# 照理說應該按照某種規則來切，但這裡只是簡單地把音訊檔案切成固定長度 
# (1sec = 16_000 點)的小塊
# 這裡有一些變化可以嘗試： ....
# 1. 用不同的長度來切
# 2. 用不同的 overlap 來切
# 3. ...

#
# 1. 先以 16_000 點為單位切，不足的補 0，彼此間沒有 overlap
#
chunk_size= 16_000
chunks= waveform.t().split(chunk_size)

print(f'{len(chunks)= }')

chunks= list(chunks)
if chunks[-1].shape[0] < chunk_size:
    n_zero= chunk_size - chunks[-1].shape[0]
    chunks[-1]= torch.stack(( 
        *chunks[-1], 
        *torch.zeros((n_zero,1)) 
        ),
        axis=0)

print(f'{chunks[-1].shape= }')

xB= torch.stack(chunks, axis=0)
xB= xB.permute(0,2,1)
xB= xB.to(device)

# recognize the waveform
with torch.no_grad():
    yB= mdl(xB)
    yB= get_likely_index(yB)
    #yB= index_to_label(yB)
    yL= [index_to_label(y) for y in yB]
    print(f'{yL= }')

#%%
#
# 2. 嘗試用 16_000 點來切，但彼此間有 overlap 8_000 點
#
# padding zeros to the end of the waveform to make it be a multiple of chunk_size
n_zero= chunk_size - waveform.shape[1] % chunk_size
w= torch.concat(
    (waveform, torch.zeros(1, n_zero)), 
    axis= 1
    )

print(f'{w.shape= }')

w2= w.view(-1,chunk_size//2)

w3= torch.stack((
        torch.concat((w2[0], w2[1])),
        torch.concat((w2[1], w2[2])),
        torch.concat((w2[2], w2[3])),
        torch.concat((w2[3], w2[4]))
        ), 
        axis=0)
w3.shape

I= w2.shape[0] # 0..I-1

w3= torch.stack([
        torch.concat((w2[i], w2[i+1]))
        #torch.concat((w2[1], w2[2])),
        #torch.concat((w2[2], w2[3])),
        #torch.concat((w2[3], w2[4]))
        for i in range(I-1)
        ], 
        axis=0)
w3.shape

w_I= torch.concat((w2[I-1], torch.zeros(chunk_size//2)))

w4= torch.stack((
    *w3,
    w_I), 
    axis=0)

xB= w4.unsqueeze(1)
xB.shape

xB= xB.to(device)

# recognize the waveform
with torch.no_grad():
    yB= mdl(xB)
    yB= get_likely_index(yB)
    #yB= index_to_label(yB)
    yL= [index_to_label(y) for y in yB]
    print(f'{yL= }')


#%%
# 3.
# 使用固定長度 16000 點來切，但彼此間有 overlap
# 可以使用任意長度 shift 的方式位移
# 這裡使用 shift= 16000//10= 1600 點 (0.1 sec)
#
width= 16_000
shift= width//10

x= waveform.squeeze()

i=0
xL= []
while i*shift+width < x.shape[0]:
    xL += [x[i*shift:i*shift+width]]
    i+=1

# 最後一個 chunk 不足 width，補 0
n_zero= width- x[i*shift:].shape[0]
xL += [torch.cat((x[i*shift:], torch.zeros(n_zero)))]

xB= torch.stack(xL).unsqueeze(1)
xB.shape
xB= xB.to(device)

# recognize the waveform
with torch.no_grad():
    zB= mdl(xB)
    yB= get_likely_index(zB)
    pB= get_likely_score(zB)
    #yB= index_to_label(yB)
    yL= [(index_to_label(y), f'{p.item():.2f}') 
         for (y,p) in zip(yB,pB)]
    print(f'{yL= }')


#%%
# view the output of the first layer

mdl
# %%
zB
# %%  view the results
#
# 4. 畫圖觀察，還不錯。
#
import numpy as np

plt.figure(figsize=(20,10))

for i in range(0,360,10):

    x= xB[i:i+1,:,:] # x.shape= (1,1,16000) # 16000=  1 sec
    xx= x.cpu().detach().numpy().squeeze()

    z= mdl(x) # z.shape= (35,) # 35= n_labels, 這裡是 35 個log機率值

    z= z.cpu().detach().numpy().squeeze()

    y= np.exp(z) # y.shape= (35,) # 35= n_labels, 這裡是 35 個機率值
    y_index= y.argmax()
    y_label= index_to_label(y_index)

    plt.subplot(211)
    plt.plot(xx+0.01*i)
    plt.subplot(212)
    plt.stem(y)
    #plt.title(f'{y_index= }, {y_label= }')
    plt.text(y_index, 0.005*i, f'{y_label}', fontsize=12)

plt.show()

# %%
