'''
file: _ry_04.2_whisper.py
date: 2023-05-10
author: Renyuan Lyu
'''

#%%
import torch
import torchaudio
import matplotlib.pyplot as plt
#%%


#%% using openAI whisper model

device= torch.device("cuda" if torch.cuda.is_available() else 
                     "cpu")
#%%
import whisper

whisperM= whisper.load_model('tiny')

whisperM.eval()
whisperM.to(device)

# get the number of parameters in whisperM
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

count_parameters(whisperM) # 37_184_640

#%%
#  get the wave data
fn_wav= 'ry35words.wav'
waveform, sample_rate= torchaudio.load(fn_wav)
waveform= waveform.squeeze()
T= waveform.shape[0] # 580160 #/sample_rate = 36.26 sec
#%%
# 一口氣辨識整段音檔 (35 sec)
x= waveform
trans= whisperM.transcribe(x)
print(f'{x.shape=}, {trans= }')

#%%
# 逐句辨識，每句 1 sec，不重疊
x= waveform[0:16_000*35].view(-1,16_000)
for xi in x:
    trans= whisperM.transcribe(xi)
    print(f'{xi.shape=}, {trans= }')
#%%
# 逐句辨識，每句 1 sec，重疊 0.5 sec

width= sample_rate*1 # 1 sec
shift= width//2      # 0.5 sec

i=0
xL= []
while i*shift+width < T:
    xL += [waveform[i*shift:i*shift+width]]
    i += 1
for xi in xL:
    trans= whisperM.transcribe(xi)
    print(f'{xi.shape=}, {trans= }')

#%%
#
# 開始 剝洋蔥式的解析 ....
#
from whisper import audio

x30sec= waveform[0:480_000] # 30 sec

xmel= audio.log_mel_spectrogram(x30sec)
x= xmel.cuda()[None,...]
z= whisperM.encoder(x)
y= whisperM.decode(z)

x.shape, \
z.shape, \
y[0].audio_features.shape, \
y[0].text

#%%
xmel.shape # torch.Size([80, 3000])
x.shape    # torch.Size([1, 80, 3000])

from whisper import decoding
init_tokens= decoding.DecodingTask(
    whisperM,
    decoding.DecodingOptions()
    )._get_initial_tokens()

init_tokens # (50258, 50259, 50359)

decoding.DecodingTask(
    whisperM, 
    decoding.DecodingOptions()
    ).run(x)

#%%
from whisper import tokenizer

tokenizer.get_encoding('gpt2').decode([50258, 50259, 50359])

tokenizer.get_encoding('gpt2').encode("a") # [64]
tokenizer.get_encoding('gpt2').encode("abc") # [39305]
tokenizer.get_encoding('gpt2').encode("abcd") # [397, 10210]

tokenizer.get_encoding('gpt2').decode([397, 10210]) #'abcd'

#%%
# 運用 whisperM 來 decode
# 先把 waveform pad 到 30 sec
# 然後轉成 mel spectrogram
# 再轉成 tensor
# 最後用 whisperM 的 encoder/decoder 來 轉換

self= whisperM

# decoder 的 input 須提供初始 token
tks= (50258, 50259, 50359)
i_tokens= torch.tensor(tks).cuda()[None,...]#[0:1,0:3]

# 輸入的 waveform 須為 30 sec，不足的補 0
Tsec= 5 # 小於 30 sec
xTsec= waveform[0:16_000*Tsec] # T sec
# pad zero to 30 sec
x30sec= torch.cat((xTsec, torch.zeros(16_000*(30-Tsec))))
xmel= audio.log_mel_spectrogram(x30sec)
x= xmel.cuda()[None,...]

# encoder
enc= self.encoder(x)

# decoder
dec= self.decoder(i_tokens, enc) 

x.shape, enc.shape, i_tokens.shape, dec.shape

# 一次 decode 一個 token，
# 必須用 for loop 才能 decode 完整的句子！
#%%
decoding.DecodingTask(
    whisperM, 
    decoding.DecodingOptions()
    ).run(x)

#%%

import torch.nn.functional as F

x.shape # torch.Size([1, 80, 3000])

c1= xx= whisperM.encoder.conv1(x)
c1= xx= F.gelu(xx)
c2= xx= whisperM.encoder.conv2(xx)
c2= xx= F.gelu(xx)

c2= xx= xx.permute(0,2,1)
c2.shape

a0= xx = (xx + whisperM.encoder.positional_embedding).to(xx.dtype)
a0.shape # torch.Size([1, 1500, 384])

a0= xx= whisperM.encoder.blocks[0](xx)
a1= xx= whisperM.encoder.blocks[1](xx)
a2= xx= whisperM.encoder.blocks[2](xx)
a3= xx= whisperM.encoder.blocks[3](xx)

a3= xx= whisperM.encoder.ln_post(xx)
a3.shape

# x.shape, c2.shape, a3.shape
'''
(torch.Size([1, 80, 3000]),
 torch.Size([1, 1500, 384]),
 torch.Size([1, 1500, 384]))
'''


# %%
whisperM.encoder(x)[0]

# %%
a3[0][0]
# %%
whisperM.decode(x)[0].audio_features[0]
# %%
whisperM.decode(x)[0].text
# %%
enc= a3

# decoder
tks= (50258, 50259, 50359)
i_tokens= torch.tensor(tks).cuda()[None,...]#[0:1,0:3]
dec= whisperM.decoder(i_tokens, enc) 

x.shape, enc.shape, i_tokens.shape, dec.shape

torch.argmax(dec[0][0], dim=-1)

# %%
whisperM.decode(x)[0].tokens

# %%
whisperM.decode(x)[0].text

# %%
whisperM.decode(x)[0].audio_features.shape
# %%
x.shape
# %%
# 輸入的 waveform 須為 30 sec，不足的補 0
Tsec= 10 # 小於 30 sec
tsec= 0
xTsec= waveform[tsec*16_000:(tsec+Tsec)*16_000] # T sec
# pad zero in the front of signal to 30 sec
x30sec= torch.cat(
    (torch.zeros(16_000*(30-Tsec)), 
     xTsec) 
    )

xmel= audio.log_mel_spectrogram(x30sec)

# show mel spectrogram
plt.imshow(xmel.cpu().numpy(), 
           aspect='auto', 
           origin='lower')
#%%
x= xmel.cuda()[None,...]
# %%
whisperM.decode(x)[0].text

# %%

Tsec= 10 # 小於 30 sec

for tsec in range(0,25,1):
    xTsec= waveform[tsec*16_000:(tsec+Tsec)*16_000] # T sec
    # pad zero in the front of signal to 30 sec
    x30sec= torch.cat(
        (torch.zeros(16_000*(30-Tsec)), 
        xTsec) 
        )

    xmel= audio.log_mel_spectrogram(x30sec)

    # show mel spectrogram
    plt.imshow(xmel.cpu().numpy(), 
            aspect='auto', 
            origin='lower')
    plt.show()
    
    x= xmel.cuda()[None,...]
    y=whisperM.decode(x)[0].text
    #y= whisperM.transcribe(x30sec, language='en')['text']
    print(y)




# %%
whisperM.transcribe(x30sec)

# %%
whisperM.transcribe(x30sec, language='en')
# %%

x30sec.shape
# %%
audio.log_mel_spectrogram(x30sec).shape
# %%
# hard-coded audio hyperparameters
SAMPLE_RATE = 16000
N_FFT = 400
N_MELS = 80
HOP_LENGTH = 160
CHUNK_LENGTH = 30
N_SAMPLES = CHUNK_LENGTH * SAMPLE_RATE  # 480000 samples in a 30-second chunk

from whisper.utils import exact_div
N_FRAMES = exact_div(N_SAMPLES, HOP_LENGTH)  # 3000 frames in a mel spectrogram input


import torchaudio
torchaudio.transforms.MelSpectrogram(
    sample_rate=SAMPLE_RATE,
    n_fft=N_FFT,
    hop_length=HOP_LENGTH,
    n_mels=N_MELS
    )(x30sec)[:,:-1].shape

# %%

tsec= 0
Tsec= 30 # 小於 30 sec
xTsec= waveform[tsec*16_000:(tsec+Tsec)*16_000] # T sec
# pad zero in the front of signal to 30 sec
x30sec= torch.cat(
    (torch.zeros(16_000*(30-Tsec)), 
    xTsec) 
    )
xmel0= audio.log_mel_spectrogram(x30sec)

# 這個 log_mel_spectrogram 是一個 不漂亮的東西
# 它會讓系統不易移植。每一家研究單位都有自己的 log_mel_spectrogram
# 如能廢棄不用，就不要用它。

# 所以，使用 end-to-end 的模型，
# 直接使用 waveform 有其實用性、必要性。

'''
xmel1= torchaudio.transforms.MelSpectrogram(
    sample_rate=SAMPLE_RATE,
    n_fft=N_FFT,
    hop_length=HOP_LENGTH,
    n_mels=N_MELS
    )(x30sec)[:,:-1]

log_spec= torch.clamp(xmel1, min=1e-10).log10()
log_spec= torch.maximum(log_spec, log_spec.max() - 8.0)
log_spec= (log_spec + 4.0) / 4.0

xmel1= log_spec
'''

x= xmel0.cuda()[None,...]
y=whisperM.decode(x)[0].text
y

#%%
whisperM.transcribe(x30sec)
# %%
