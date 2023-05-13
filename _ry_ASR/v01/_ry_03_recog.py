'''
file: _ry_03_recog.py
date: 2023-05-10
author: Renyuan Lyu

'''

#%%
import torch
import torchaudio
import matplotlib.pyplot as plt

from ryModels import (
    theLabels, 
    label_to_index, 
    index_to_label,
    get_likely_index,
    get_likely_score)

from ryModels import ryM10 as ryM

labels= theLabels
print(f'{labels= }')

#%%
device= torch.device("cuda" if torch.cuda.is_available() else 
                     "cpu")
print(device)

#%%
# initialize the model and load the weights
model_fn= 'ryM.pt' # 'ryM10.pt'
model= torch.load(model_fn, map_location= device)
# [(k, model.get(k).shape) for k in model.keys()]

# calculate the number of parameters
n_params= 0
for k in model.keys():
    n_params += model.get(k).numel()
print(f'{n_params= }')
#%%

mdl= ryM(in_chs= 1, out_cls=35)
mdl.load_state_dict(model)
#%%
#print(mdl)

mdl.eval() # only in inference mode
mdl.to(device)

#%%

#  get the wave data

fn_wav= 'ry35words.wav'
waveform, sample_rate= torchaudio.load(fn_wav)

plt.plot(waveform.T.numpy())
plt.show()

x=  waveform
xB= x[None,...]
xB= xB.to(device)

#%%
# split the waveform into chunks, and recognize each chunk
# 這一段是為了讓模型可以接受任意長度的音訊檔案
# 照理說應該按照某種規則來切，但這裡只是簡單地把音訊檔案切成固定長度 
# (1sec = 16_000 點)的小塊
# 這裡有一些變化可以嘗試： ....
# 1. 用不同的長度來切
# 2. 用不同的 overlap 來切
# 3. ...


#%%
# 3.
# 使用固定長度 16000 點來切，但彼此間有 overlap
# 可以使用任意長度 shift 的方式位移
# 這裡使用 shift= 16000//10= 1600 點 (0.1 sec)
#
#%%

width= 16_000 # 1 sec
shift= width//10 # 0.1 sec

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


# %%  view the results
#
# 4. 畫圖觀察，還不錯。
#
import numpy as np

plt.figure(figsize=(20,10))


step= 10
T= xB.shape[0] 

for i in range(0, T, step):

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
    plt.text(y_index, i/T, f'{y_label}', fontsize=12)

plt.show()

# %%
