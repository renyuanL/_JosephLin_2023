
#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

theLabels= [
 'backward', 'bed',     'bird',     'cat',      'dog',
 'down',    'eight',    'five',     'follow',   'forward',
 'four',    'go',       'happy',    'house',    'learn',
 'left',    'marvin',   'nine',     'no',       'off',
 'on',      'one',      'right',    'seven',    'sheila',
 'six',     'stop',     'three',    'tree',     'two',
 'up',      'visual',   'wow',      'yes',      'zero'
]

漢字表= [
 '零', '一', '二', '三', '四',
 '五', '六', '七', '八', '九',
 '是', '否', '去',  '停', '哇', 
 '開機', '關機', '前進', '後退', 
 '向上', '向下', '向左', '向右',
 '樹', '屋', '床', '貓', '狗', '鳥', 
 '看見', '跟隨', '學習', '快樂', 
 '馬文', '席拉' 
]

def label_to_index(label):
    return torch.tensor(theLabels.index(label))

def index_to_label(index):
    return theLabels[index]

#%%

#def get_likely_index(tensor):
#    # find most likely label index for each element in the batch
#    return tensor.argmax(dim=-1)

def get_likely_index(tensor):
    # find most likely label index for each element in the batch
    return tensor.max(axis=-1).indices

def get_likely_score(tensor):
    # find most likely label index for each element in the batch
    return tensor.max(axis=-1).values



class ryM2(nn.Module):
    def __init__(self, 
                 in_chs=   1,  #  1 channel, mono waveform
                 out_cls= 35,  # 35 words as output classes
                 sample_rate=  16_000 # sample rate of the audio file
                 ):
        
        super().__init__()

        new_sample_rate= sample_rate #//2 #8_000

        #self.transform= torchaudio.transforms.Resample(
        #    orig_freq= sample_rate, 
        #    new_freq=  new_sample_rate)

        self.act=  nn.ReLU()
        self.flat= nn.Flatten()
        self.out=  nn.LogSoftmax(dim=-1)
        #self.out=  nn.Softmax(dim=-1)

        k1= int(.02 *new_sample_rate) # 320 # 20ms
        s1= int(.01 *new_sample_rate) # 160 # 10ms
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
        
        #x= self.transform(x) # (1,16000) -> (1,8000) # downsample by factor of 2

        #  CNNs
        x= self.conv1(x) #  -> ( 64, 99)
        x= self.bn1(x)   
        x= self.act(x)   
        
        x= self.conv2(x) #  -> (128, 48)
        x= self.bn2(x)   
        x= self.act(x)   
        
        x= self.conv3(x) #  -> (256, 23)
        x= self.bn3(x)   
        x= self.act(x)   
       
        x= self.conv4(x) #  -> (256, 10)
        x= self.bn4(x)   
        x= self.act(x)

        x= self.conv5(x) #  -> (128, 4)
        x= self.bn5(x)   
        x= self.act(x)   
        
        # global average pooling
        x= F.avg_pool1d(x, x.shape[-1])  # -> (128, 1)
        x= self.flat(x) # -> (128)

        # MLPs
        x= self.fc1(x)  # -> (64)
        x= self.act(x)

        x= self.fc2(x)  # -> (35)
        y= self.out(x)  # -> (35)

        return y

# raw waveform -> CNNs -> MLPs -> output
# 1D convolutional neural network


## summary of the model
'''
ryM2(
  (act): ReLU()
  (flat): Flatten(start_dim=1, end_dim=-1)
  (out): LogSoftmax(dim=-1)
  (conv1): Conv1d(1, 64, kernel_size=(320,), stride=(160,))
  (bn1): BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)     
  (conv2): Conv1d(64, 128, kernel_size=(4,), stride=(2,))
  (bn2): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)    
  (conv3): Conv1d(128, 256, kernel_size=(4,), stride=(2,))
  (bn3): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)    
  (conv4): Conv1d(256, 256, kernel_size=(4,), stride=(2,))
  (bn4): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)    
  (conv5): Conv1d(256, 128, kernel_size=(4,), stride=(2,))
  (bn5): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)    
  (fc1): Linear(in_features=128, out_features=64, bias=True)
  (fc2): Linear(in_features=64, out_features=35, bias=True)
)
'''
# Number of parameters: 590_563 vs 1_500_000_000 (1.5G) for VGG16
'''
Test@epoch= 30, acc=【0.8782】, [8765/9981]
 88%|███████████████████████████████████▎    | 26.48936170212732/30 [15:59<02:07, 36.24s/it] 
Test@epoch= 1, acc=【0.8782】, [8765/9981]
Test@epoch= 1, acc=【0.8872】, [8855/9981]
Val accuracy: 0.8872
Test@epoch= 1, acc=【0.8745】, [9624/11005]
Test accuracy: 0.8745
'''

'''
Test@epoch= 1, acc=【0.8861】, [8844/9981]
Val accuracy: 0.8861
Test@epoch= 1, acc=【0.8709】, [9584/11005]
Test accuracy: 0.8709

Training set size: 84843
Test@epoch= 1, acc=【0.9612】, [81549/84843]
'''





#%%
'''
xB= torch.randn((10, 16_000))
xB= xB.cuda()
X_mt= melsgram= \
torchaudio\
.transforms\
.MelSpectrogram().to('cuda')(xB)
X_mt.shape # torch.Size([10, 128, 81])
'''
# %%

class ryM3(nn.Module):
    def __init__(self, 
                 in_chs=   1,  #  1 channel, mono waveform
                 out_cls= 35,  # 35 words as output classes
                 sample_rate=  16_000 # sample rate of the audio file
                 ):
        
        super().__init__()

        new_sample_rate= sample_rate #//2 #8_000

        n_fft=  int(.02*sample_rate) #320 #400
        n_mels=  64 #128
        self.transform= torchaudio.transforms.MelSpectrogram(
            sample_rate= sample_rate,
            n_fft= n_fft,
            n_mels= n_mels)


        self.act=  nn.ReLU()
        self.flat= nn.Flatten()
        self.out=  nn.LogSoftmax(dim=-1)
        #self.out=  nn.Softmax(dim=-1)

        '''
        k1= int(.02 *new_sample_rate) # 320 # 20ms
        s1= int(.01 *new_sample_rate) # 160 # 10ms
        ch1= 64 # 64 channels in 1st convolution layer
        '''
        k1= n_fft
        s1= k1//2
        ch1= n_mels #64 # 64 channels in 1st convolution layer

        k2= 4 # kernel size in the other conv layer
        s2= 2 # stride in the other conv layer

        ####self.conv1= nn.Conv1d(in_chs, ch1,   kernel_size= k1, stride= s1) 
        #self.bn1=   nn.BatchNorm1d(ch1)

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
        
        #x= self.transform(x) # (1,16000) -> (1,8000) # downsample by factor of 2

        #  CNNs
        x= self.transform(x) #  -> (, 1, 64, 101)
        x= x.squeeze(1) # -> (, 64, 101)
        #x= self.bn1(x)   
        #x= self.act(x)   
        
        x= self.conv2(x) #  -> (,128, 49)
        x= self.bn2(x)   
        x= self.act(x)   
        
        x= self.conv3(x) #  -> (,256, 23)
        x= self.bn3(x)   
        x= self.act(x)   
       
        x= self.conv4(x) #  -> (,256, 10)
        x= self.bn4(x)   
        x= self.act(x)

        x= self.conv5(x) #  -> (,128, 4)
        x= self.bn5(x)   
        x= self.act(x)   
        
        # global average pooling
        x= F.avg_pool1d(x, x.shape[-1])  # -> (,128, 1)
        x= self.flat(x) # -> (,128)

        # MLPs
        x= self.fc1(x)  # -> (,64)
        x= self.act(x)

        x= self.fc2(x)  # -> (,35)
        y= self.out(x)  # -> (,35)

        return y

##
## MelSpectrogram + CNNs + MLPs
## 1. 1st conv layer: 64 channels, kernel size= 320, stride= 160
## 2. 2nd conv layer: 128 channels, kernel size= 4, stride= 2
## 3. 3rd conv layer: 256 channels, kernel size= 4, stride= 2
## 4. 4th conv layer: 256 channels, kernel size= 4, stride= 2
## 5. 5th conv layer: 128 channels, kernel size= 4, stride= 2
## 6. 1st fc layer: 64 neurons
## 7. 2nd fc layer: 35 neurons
## 8. output layer: 35 neurons
##

# summary of the model

'''
ryM3(
  (transform): MelSpectrogram(
    (spectrogram): Spectrogram()
    (mel_scale): MelScale()
  )
  (act): ReLU()
  (flat): Flatten(start_dim=1, end_dim=-1)
  (out): LogSoftmax(dim=-1)
  (conv2): Conv1d(64, 128, kernel_size=(4,), stride=(2,))
  (bn2): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (conv3): Conv1d(128, 256, kernel_size=(4,), stride=(2,))
  (bn3): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (conv4): Conv1d(256, 256, kernel_size=(4,), stride=(2,))
  (bn4): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (conv5): Conv1d(256, 128, kernel_size=(4,), stride=(2,))
  (bn5): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (fc1): Linear(in_features=128, out_features=64, bias=True)
  (fc2): Linear(in_features=64, out_features=35, bias=True)
)
'''

# Number of parameters: 569_891

'''
Test@epoch= 30, acc=【0.7740】, [7725/9981]
 88%|███████████████████████████████████████████████████████████████████         | 26.48936170212732/30 [15:50<02:06, 35.89s/it]
Test@epoch= 1, acc=【0.7740】, [7725/9981]
Test@epoch= 1, acc=【0.7740】, [7725/9981]
Val accuracy: 0.7740
Test@epoch= 1, acc=【0.7662】, [8432/11005]
Test accuracy: 0.7662
'''

#%%
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

# check the availability of GPU
device= torch.device('cuda' if torch.cuda.is_available() else 
                     'cpu')
ryMelsgram= ryMelsgram.to(device)

class ryMelsgram1d(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        x= ryMelsgram(x)
        x= x.squeeze(dim=-3) 
        return x

class ryMelsgram2d(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        x= ryMelsgram(x)   
        #x= x.unsqueeze(dim=-3) # add channel dim，可能不需要
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

#%%

import collections

class ryM10(nn.Module):
    def __init__(self, in_chs= 1, out_cls= 35):
        super(ryM10, self).__init__()
        
        layers= collections.OrderedDict([

            ('c1', nn.Conv1d(in_chs,  64, kernel_size= 320, stride= 160)),
            ('b1', nn.BatchNorm1d(64)),
            ('r1', nn.ReLU()),
            #('mels', ryMelsgram1d()),
            
            ('c2', nn.Conv1d(64,  128, kernel_size= 4, stride= 2)),
            ('b2', nn.BatchNorm1d(128)),
            ('r2', nn.ReLU()),
            
            ('c3', nn.Conv1d(128, 256, kernel_size= 4, stride= 2)),
            ('b3', nn.BatchNorm1d(256)),
            ('r3', nn.ReLU()),
            
            ('c4', nn.Conv1d(256, 256, kernel_size= 4, stride= 2)),
            ('b4', nn.BatchNorm1d(256)),
            ('r4', nn.ReLU()),
            
            ('c5', nn.Conv1d(256, 256, kernel_size= 4, stride= 2)),
            ('b5', nn.BatchNorm1d(256)),
            ('r5', nn.ReLU()),
            
            ('p1', ryAvgPool1d()), 
            
            ('l1', nn.Linear(256, 128)),
            ('t1', nn.Tanh()),
            
            ('l2', nn.Linear(128, out_cls)),
            ('out',nn.LogSoftmax(dim=-1))
            ])
        
        self.model= nn.Sequential(layers)

    def forward(self, x):
        x= self.model(x)
        return x

#
# Simply use cov1d to replace melspectrogram
# # 
# Number of parameters: 748_899
# Test@epoch= 15, acc=【0.8876】, [8859/9981]
# Test@epoch= 1, acc=【0.8685】, [9558/11005]
# Test@epoch= 1, acc=【0.9608】, [81519/84843]
# Test@epoch= 1, acc=【0.8943】, [8926/9981]
# Val accuracy: 0.8943
# Test@epoch= 1, acc=【0.8802】, [9687/11005]
# Test accuracy: 0.8802
# Test@epoch= 1, acc=【0.9688】, [82196/84843]

#
# Simple is beautiful
#
'''
Test@epoch= 1, acc=【0.8897】, [8880/9981]
Test@epoch= 1, acc=【0.8946】, [8929/9981]
Val accuracy: 0.8946
Test@epoch= 1, acc=【0.8806】, [9691/11005]
Test accuracy: 0.8806
Test@epoch= 1, acc=【0.9696】, [82264/84843]
Train accuracy: 0.9696
'''

class ryM11(nn.Module):
    def __init__(self, in_chs= 1, out_cls= 35):
        super(ryM11, self).__init__()
        
        layers= collections.OrderedDict([

            #('c1', nn.Conv1d(in_chs,  64, kernel_size= 320, stride= 160)),
            #('b1', nn.BatchNorm1d(64)),
            #('r1', nn.ReLU()),
            ('mels', ryMelsgram1d()),
            
            ('c2', nn.Conv1d(64,  128, kernel_size= 4, stride= 2)),
            ('b2', nn.BatchNorm1d(128)),
            ('r2', nn.ReLU()),
            
            ('c3', nn.Conv1d(128, 256, kernel_size= 4, stride= 2)),
            ('b3', nn.BatchNorm1d(256)),
            ('r3', nn.ReLU()),
            
            ('c4', nn.Conv1d(256, 256, kernel_size= 4, stride= 2)),
            ('b4', nn.BatchNorm1d(256)),
            ('r4', nn.ReLU()),
            
            ('c5', nn.Conv1d(256, 256, kernel_size= 4, stride= 2)),
            ('b5', nn.BatchNorm1d(256)),
            ('r5', nn.ReLU()),
            
            ('p1', ryAvgPool1d()), 
            
            ('l1', nn.Linear(256, 128)),
            ('t1', nn.Tanh()),
            
            ('l2', nn.Linear(128, out_cls)),
            ('out',nn.LogSoftmax(dim=-1))
            ])
        
        self.model= nn.Sequential(layers)

    def forward(self, x):
        x= self.model(x)
        return x


# use melspectrogram
# Number of parameters: 728_227
# Test@epoch= 15, acc=【0.7686】, [7671/9981]
'''
Test@epoch= 1, acc=【0.7949】, [7934/9981]
Test@epoch= 1, acc=【0.7949】, [7934/9981]
Val accuracy: 0.7949
Test@epoch= 1, acc=【0.7873】, [8664/11005]
Test accuracy: 0.7873
Test@epoch= 1, acc=【0.8570】, [72709/84843]
Train accuracy: 0.8570
'''

#%%
class ryM12(nn.Module):
    def __init__(self, in_chs= 1, out_cls= 35):
        super(ryM12, self).__init__()
        
        layers= collections.OrderedDict([

            #('c1', nn.Conv1d(in_chs,  64, kernel_size= 320, stride= 160)),
            #('b1', nn.BatchNorm1d(64)),
            #('r1', nn.ReLU()),
            ('mels', ryMelsgram2d()), #
            
            ('c2', nn.Conv2d(1,  64, kernel_size= 4, stride= 2)),
            ('b2', nn.BatchNorm2d(64)),
            ('r2', nn.ReLU()),
            
            ('c3', nn.Conv2d(64, 128, kernel_size= 4, stride= 2)),
            ('b3', nn.BatchNorm2d(128)),
            ('r3', nn.ReLU()),
            
            ('c4', nn.Conv2d(128, 256, kernel_size= 4, stride= 2)),
            ('b4', nn.BatchNorm2d(256)),
            ('r4', nn.ReLU()),
            
            ('c5', nn.Conv2d(256, 256, kernel_size= 4, stride= 2)),
            ('b5', nn.BatchNorm2d(256)),
            ('r5', nn.ReLU()),
            
            ('p1', ryAvgPool2d()), 
            
            ('l1', nn.Linear(256, 128)),
            ('t1', nn.Tanh()),
            
            ('l2', nn.Linear(128, out_cls)),
            ('out',nn.LogSoftmax(dim=-1))
            ])
        
        self.model= nn.Sequential(layers)

    def forward(self, x):
        x= self.model(x)
        return x

# Number of parameters: 727_491
'''
Test@epoch= 1, acc=【0.8166】, [8150/9981]
Test@epoch= 1, acc=【0.8463】, [8447/9981]
Val accuracy: 0.8463
Test@epoch= 1, acc=【0.8353】, [9193/11005]
Test accuracy: 0.8353
Test@epoch= 1, acc=【0.9042】, [76716/84843]
Train accuracy: 0.9042
'''

# Number of parameters: 1_744_483
'''
Test@epoch= 1, acc=【0.8393】, [8377/9981]
Test@epoch= 1, acc=【0.8393】, [8377/9981]
Val accuracy: 0.8393
Test@epoch= 1, acc=【0.8275】, [9107/11005]
Test accuracy: 0.8275
Test@epoch= 1, acc=【0.8931】, [75775/84843]
Train accuracy: 0.8931
'''