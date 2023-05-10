'''
file: _ry_01_recog_01.py
date: 2023-05-06
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

#"https://drive.google.com/file/d/1-3JF7rhFBpfajaIM-_NjKgg8WXHJ_fP9/view?usp=share_link"
#"https://drive.google.com/file/d/1-3_AWSuw9m195PKgixouOR_2LDr_bAEE/view?usp=share_link"
# get data from link above
#### !gdown --id 1-3JF7rhFBpfajaIM-_NjKgg8WXHJ_fP9
#### !gdown --id 1-3_AWSuw9m195PKgixouOR_2LDr_bAEE

import os
os.system('gdown --id 1-3JF7rhFBpfajaIM-_NjKgg8WXHJ_fP9')
os.system('gdown --id 1-3_AWSuw9m195PKgixouOR_2LDr_bAEE')

#  display the data
ipd.Audio('ryTest.wav')


# %%

labels= [
 'backward', 'bed',     'bird',     'cat',      'dog',
 'down',    'eight',    'five',     'follow',   'forward',
 'four',    'go',       'happy',    'house',    'learn',
 'left',    'marvin',   'nine',     'no',       'off',
 'on',      'one',      'right',    'seven',    'sheila',
 'six',     'stop',     'three',    'tree',     'two',
 'up',      'visual',   'wow',      'yes',      'zero'
]

def label_to_index(label):
    return torch.tensor(labels.index(label))

def index_to_label(index):
    return labels[index]

#%%
class ryM(nn.Module):
    def __init__(self, 
                 in_chs=   1,  #  1 channel, mono waveform
                 out_cls= 35,  # 35 words as output classes
                 sample_rate=  16_000 # sample rate of the audio file
                 ):
        
        super().__init__()

        new_sample_rate= sample_rate //2 #8_000

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



#%%
device = torch.device("cuda" if torch.cuda.is_available() else 
                      "cpu")
print(device)

# initialize the model and load the weights
mdl= ryM(in_chs= 1, out_cls=35)

mdl.load_state_dict(
    torch.load('model.pt', map_location=device)
    )

mdl.eval() # only in inference mode
mdl.to(device)

print(mdl)
#%%
# load the waveform
#fn= fn0= "backward.wav"
fn= fn1= "ryTest.wav"
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
# the result of the above cell looks like this:
# Quite interesting, isn't it?
#
yL= [
     #'up', 'off', 
     'zero', 'zero', 'zero', 'zero', 'zero', 'zero', 'zero', 'zero', 'zero', 'zero', 'zero', 'zero', 
     #'go', 'up', 'no', 'wow', 
     'one', 'one', 'one', 'one', 'one', 'one', 'one', 'one', 
     #'three', 
     'two', 'two', 'tree', 'two', 'two', 'two', 'two', 'two', 'two', 'two', 'two', 
     #'up', 'up', 'cat', 'seven', 
     'three', 'three', 'three', 'three', 'three', 'three', 'three', 'three', 'three', 
     #'two', 'two', 
     'four', 'four', 'four', 'forward', 'four', 'forward', 'forward', 'forward', 'four', 'forward', 'four', 
     'five', 'five', 'five', 'five', 'five', 'five', 'five', 'five', 'five', 'five', 'five', 'five', 
     'six', 'six', 'six', 'six', 'six', 'six', 'six', 'six', 'six', 'six', 'six', 
     #'three', 'six', 
     'seven', 'seven', 'seven', 'seven', 'seven', 'seven', 'seven', 'seven', 'seven', 'seven', 
     'eight', 'eight', 'eight', 'eight', 'eight', 'eight', 'eight', 'eight', 'eight', 
     #'up', 'marvin', 
     'nine', 'nine', 'nine', 'nine', 'nine', 'nine', 'nine', 'nine', 
     #'tree', 'six', 'four', 'four', 
     'forward', 'forward', 'forward', 'forward', 'forward', 'forward', 'forward', 
     #'three', 'forward', 'go', 'down', 
     'backward', 'backward', 'backward', 'backward', 'backward', 'backward', 'backward', 
     #'three', 'three', 'three', 
     'up', 'up', 'up', 'up', 'up', 'up', 'up', 'up', 'up', 
     #'cat', 'stop', 
     'down', 'down', 'down', 'down', 'down', 'down', 'down', 'down', 
     #'no', 
     'left', 'left', 'left', 'left', 'left', 'left', 'left', 'left', 'left', 'left', 
     'right', 'right', 'right', 'right', 'right', 'right', 'right', 'right', 'right', 'right', 
     #'six', 'off', 'up', 'up', 'up', 'up', 'up', 'four'
     ]


yL= [
    #('up', '-2.81'), ('off', '-2.94'),

    ('zero', '-2.09'), ('zero', '-1.08'), ('zero', '-0.20'), ('zero', '-0.05'), ('zero', '-0.00'), ('zero', '-0.00'), ('zero', '-0.00'), 
    ('zero', '-0.00'), ('zero', '-0.00'), ('zero', '-0.00'), ('zero', '-0.01'), ('zero', '-0.46'), 
    
    #('go', '-0.69'), ('up', '-2.04'), ('no', '-2.24'), ('wow', '-0.40'), 
    
    ('one', '-0.12'), ('one', '-0.00'), ('one', '-0.00'), ('one', '-0.00'), ('one', '-0.00'), ('one', '-0.00'), ('one', '-0.00'), ('one', '-0.00'), 
    
    #('three', '-2.03'), 
    
    ('two', '-2.75'), ('two', '-2.49'), ('tree', '-0.65'), ('two', '-0.00'), ('two', '-0.00'), ('two', '-0.00'), ('two', '-0.00'), ('two', '-0.00'), 
    ('two', '-0.00'), ('two', '-0.00'), ('two', '-0.13'), 
    
    #('up', '-2.39'), ('up', '-2.60'), ('cat', '-2.48'), ('seven', '-1.81'), 
    
    ('three', '-0.90'), ('three', '-0.09'), ('three', '-0.17'), ('three', '-0.04'), ('three', '-0.05'), ('three', '-0.09'), ('three', '-0.05'), 
    ('three', '-0.16'), ('three', '-1.15'), 
    
    #('two', '-2.66'), ('two', '-2.70'), 
    
    ('four', '-2.51'), ('four', '-1.00'), ('four', '-0.56'), ('forward', '-0.64'), ('four', '-0.24'), ('forward', '-0.52'), ('forward', '-0.41'), 
    ('forward', '-0.55'), ('four', '-0.48'), ('forward', '-0.44'), ('four', '-0.82'), 
    
    ('five', '-1.56'), ('five', '-1.67'), ('five', '-0.21'), ('five', '-0.01'), ('five', '-0.00'), ('five', '0.00'), ('five', '0.00'), 
    ('five', '-0.00'), ('five', '-0.00'), ('five', '-0.00'), ('five', '-0.19'), ('five', '-0.72'), 
    
    ('six', '-1.42'), ('six', '-0.44'), ('six', '-0.02'), ('six', '-0.00'), ('six', '-0.00'), ('six', '0.00'), ('six', '-0.00'), ('six', '-0.00'), 
    ('six', '-0.00'), ('six', '-0.48'), ('six', '-1.54'), ('three', '-2.37'), ('six', '-0.44'), 
    
    ('seven', '-0.82'), ('seven', '-0.06'), ('seven', '-0.01'), ('seven', '-0.00'), ('seven', '-0.00'), ('seven', '-0.00'), ('seven', '-0.00'), 
    ('seven', '-0.00'), ('seven', '-0.04'), ('seven', '-2.44'), 
    
    ('eight', '-0.53'), ('eight', '-0.01'), ('eight', '-0.00'), ('eight', '-0.00'), ('eight', '-0.00'), ('eight', '-0.00'), ('eight', '-0.00'), 
    ('eight', '-0.00'), ('eight', '-0.02'), 
    
    #('up', '-2.70'), ('marvin', '-1.26'), 
    
    ('nine', '-0.19'), ('nine', '-0.00'), ('nine', '-0.00'), ('nine', '-0.00'), ('nine', '-0.00'), ('nine', '-0.00'), ('nine', '-0.00'), 
    ('nine', '-0.04'), 
    
    #('tree', '-1.49'), ('six', '-1.93'), 
    
    ('four', '-0.99'), ('four', '-0.50'), ('forward', '-0.09'), ('forward', '-0.02'), ('forward', '-0.00'), ('forward', '-0.00'), ('forward', '-0.00'), 
    ('forward', '-0.00'), ('forward', '-0.00'), ('three', '-0.31'), ('forward', '-0.97'), 
    
    #('go', '-1.52'), ('down', '-0.82'), 
    
    ('backward', '-0.14'), ('backward', '-0.00'), ('backward', '0.00'), ('backward', '0.00'), ('backward', '0.00'), ('backward', '-0.00'), 
    ('backward', '-0.00'), 
    
    #('three', '-0.51'), ('three', '-0.21'), ('three', '-0.99'), 
    
    ('up', '-0.09'), ('up', '-0.01'), ('up', '-0.01'), ('up', '-0.00'), ('up', '-0.00'), ('up', '-0.03'), ('up', '-0.01'), 
    ('up', '-0.00'), ('up', '-2.37'), 
    
    #('cat', '-2.47'), ('stop', '-1.30'), 
    
    ('down', '-0.29'), ('down', '-0.00'), ('down', '-0.01'), ('down', '-0.00'), ('down', '-0.00'), ('down', '-0.03'), ('down', '-0.02'), 
    ('down', '-1.26'), 
    
    #('no', '-2.08'), 
    
    ('left', '-0.57'), ('left', '-0.24'), ('left', '-0.55'), ('left', '-0.00'), ('left', '-0.00'), ('left', '-0.00'), 
    ('left', '-0.00'), ('left', '-0.00'), ('left', '-0.03'), ('left', '-0.68'), 
    
    ('right', '-0.23'), ('right', '-0.00'), ('right', '-0.00'), ('right', '-0.00'), ('right', '0.00'), ('right', '0.00'), ('right', '0.00'), 
    ('right', '0.00'), ('right', '-0.00'), ('right', '-0.06'), 
    
    #('six', '-1.68'), ('off', '-2.36'), ('up', '-2.37'), ('up', '-2.83'), ('up', '-2.91'), ('up', '-2.92'), ('up', '-2.95'), ('four', '-2.93')
    ]






# %%
