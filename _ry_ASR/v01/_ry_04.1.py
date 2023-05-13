'''
file: _ry_04.1.py
date: 2023-05-10
author: Renyuan Lyu
'''

#%%
import torch
import torchaudio
import matplotlib.pyplot as plt

from ryModels import (
    index_to_label,
    get_likely_index,
    get_likely_score)

#%%
device= torch.device("cuda" if torch.cuda.is_available() else 
                     "cpu")
device= torch.device("cpu") # 強迫使用 cpu

mdl_fn= 'ryJitM.pt'
mdl= torch.jit.load(mdl_fn)
mdl.eval()
mdl.to(device)

print(device)
print(mdl)
#%%

#  get the wave data

fn_wav= 'ry35words.wav'
waveform, sample_rate= torchaudio.load(fn_wav)
waveform= waveform.squeeze()
T= waveform.shape[0] # 580160 #/sample_rate = 36.26 sec
#%%

width= sample_rate*1 # 1 sec
shift= width//2      # 0.5 sec

i=0
xL= []
while i*shift+width < T:
    xL += [waveform[i*shift:i*shift+width]]
    i += 1

# 最後一個 chunk 不足 width，補 0
n_zero= i*shift + width - T
xL += [ torch.cat((
    waveform[i*shift:], 
    torch.zeros(n_zero)
    ))]

xB= torch.stack(xL).unsqueeze(1)
xB= xB.to(device)
#%%

# recognize the waveform
with torch.no_grad():
    
    zB= mdl(xB[0:10])
    
    yB= get_likely_index(zB)
    pB= get_likely_score(zB)

    yL= [index_to_label(y) for y in yB]
    
    yL2= [(index_to_label(y), f'{p.item():.2f}') 
         for (y, p) in zip(yB,pB)]
    
    print(f'{yL= }')
    print(f'{yL2= }')

# %%

# %%

def fwd(xB):
    with torch.no_grad():
        c1= zB= mdl.model.c1(xB)
        b1= zB= mdl.model.b1(zB)
        r1= zB= mdl.model.r1(zB)
        c2= zB= mdl.model.c2(zB)
        b2= zB= mdl.model.b2(zB)
        r2= zB= mdl.model.r2(zB)
        c3= zB= mdl.model.c3(zB)
        b3= zB= mdl.model.b3(zB)
        r3= zB= mdl.model.r3(zB)
        c4= zB= mdl.model.c4(zB)
        b4= zB= mdl.model.b4(zB)
        r4= zB= mdl.model.r4(zB)
        c5= zB= mdl.model.c5(zB)
        b5= zB= mdl.model.b5(zB)
        r5= zB= mdl.model.r5(zB)
        p1= zB= mdl.model.p1(zB)
        l1= zB= mdl.model.l1(zB)
        t1= zB= mdl.model.t1(zB)
        l2= zB= mdl.model.l2(zB)
        out= zB= mdl.model.out(zB)
        layerout= (
            r1,r2,r3,r4,r5,
            p1,l1,t1,l2,out
            )
        #layerout= torch.stack(layerout)
        return layerout

# %%
fwd(xB)[-1].shape
# %%
