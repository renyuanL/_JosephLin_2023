'''
file: _ry_04.3_whisper.py
date: 2023-05-11
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

whisperM= whisper.load_model('tiny.en')

whisperM.eval()
whisperM.to(device)

# get the number of parameters in whisperM
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

count_parameters(whisperM) # 37_184_640 (tiny), 37_184_256 (tiny.en)

#%%
#  get the wave data
fn_wav= 'ry35words.wav'
waveform, sample_rate= torchaudio.load(fn_wav)
waveform= waveform.squeeze()
T= waveform.shape[0] # 580_160 #/sample_rate = 36.26 sec
T
#%%
# 一口氣辨識整段音檔 (35 sec)
def rcg01(waveform):
    x= waveform
    trans= whisperM.transcribe(x)
    print(f'{x.shape=}, {trans= }')

#%%
# 逐句辨識，每句 1 sec，不重疊
def rcg02(waveform):
    x= waveform[0:16_000*35].view(-1,16_000)
    for xi in x:
        trans= whisperM.transcribe(xi)
        print(f'{xi.shape=}, {trans= }')
#%%
# 逐句辨識，每句 1 sec，重疊 0.5 sec
def rcg03(waveform):
    width= 16_000 # 1 sec
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
#rcg01(waveform)
#rcg02(waveform)
#rcg03(waveform)

#%%
#
# 開始 剝洋蔥式的解析 ....
#

x30sec= waveform[:16_000*30]
x30sec.shape # torch.Size([480000])

#y= whisperM.transcribe(x30sec)
#y

x= whisper.audio.log_mel_spectrogram(x30sec).to('cuda')
y= whisperM.decode(x, language='en')
y.text, y.tokens

# %%
y_text= whisper.tokenizer.get_encoding('gpt2').decode(y.tokens)
y_text
# %%


# %%


# %%
whisperM.dims
#
'''
ModelDimensions(
    n_mels=         80, 
    n_audio_ctx=    1500, # 3000//2 # 3000 = 30sec

    n_audio_state=  384, 
    n_audio_head=   6, 
    n_audio_layer=  4,

    n_vocab=        51864, 
    n_text_ctx=     448, # 最大字數224 *2

    n_text_state=   384, 
    n_text_head=    6, 
    n_text_layer=   4)
'''

# %%
'''
mdl= whisper.Whisper(whisperM.dims)
mdl.load_state_dict(whisperM.state_dict())
mdl.eval()
mdl= mdl.to('cpu')
mdl.transcribe(x30sec[0:16000*2])
mdl.transcribe(x30sec[0:16000*10])
'''

# %%
'''
Whisper= (
  (encoder): AudioEncoder(
    (conv1): Conv1d(80, 384, kernel_size=(3,), stride=(1,), padding=(1,))
    (conv2): Conv1d(384, 384, kernel_size=(3,), stride=(2,), padding=(1,))
    (blocks): ModuleList(
      (0-3): 4 x ResidualAttentionBlock(
        (attn): MultiHeadAttention(
          (query): Linear(in_features=384, out_features=384, bias=True)
          (key): Linear(in_features=384, out_features=384, bias=False)
          (value): Linear(in_features=384, out_features=384, bias=True)
          (out): Linear(in_features=384, out_features=384, bias=True)
        )
        (attn_ln): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
        (mlp): Sequential(
          (0): Linear(in_features=384, out_features=1536, bias=True)
          (1): GELU(approximate='none')
          (2): Linear(in_features=1536, out_features=384, bias=True)
        )
        (mlp_ln): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
      )
    )
    (ln_post): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
  )
  (decoder): TextDecoder(
    (token_embedding): Embedding(51864, 384)
    (blocks): ModuleList(
      (0-3): 4 x ResidualAttentionBlock(
        (attn): MultiHeadAttention(
          (query): Linear(in_features=384, out_features=384, bias=True)
          (key): Linear(in_features=384, out_features=384, bias=False)
          (value): Linear(in_features=384, out_features=384, bias=True)
          (out): Linear(in_features=384, out_features=384, bias=True)
        )
        (attn_ln): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
        (cross_attn): MultiHeadAttention(
          (query): Linear(in_features=384, out_features=384, bias=True)
          (key): Linear(in_features=384, out_features=384, bias=False)
          (value): Linear(in_features=384, out_features=384, bias=True)
          (out): Linear(in_features=384, out_features=384, bias=True)
        )
        (cross_attn_ln): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
        (mlp): Sequential(
          (0): Linear(in_features=384, out_features=1536, bias=True)
          (1): GELU(approximate='none')
          (2): Linear(in_features=1536, out_features=384, bias=True)
        )
        (mlp_ln): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
      )
    )
    (ln): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
  )
)
'''
# %%
import torch.nn as nn
import torch.nn.functional as F
import torch

embed_dim= 4
num_heads= 1

x= [
  [0, 0, 0, 1], # Seq 1
  [0, 0, 1, 0], # Seq 2
  [0, 0, 1, 1], # Seq 3
  [0, 1, 0, 0], # Seq 4
  [0, 1, 0, 1]  # Seq 5
]
x= torch.tensor(x, dtype=torch.float32)

wk= [
  [0, 0, 1, 1],
  [1, 1, 0, 1],
  [0, 1, 0, 1],
  [1, 1, 0, 1]
]

wq= [
  [1, 0, 1, 1],
  [1, 0, 0, 1],
  [0, 0, 1, 1],
  [0, 1, 1, 1]
]

wv= [
  [0, 2, 0, 1],
  [0, 3, 0, 1],
  [1, 0, 3, 1],
  [1, 1, 0, 1]
]

wk= torch.tensor(wk, dtype=torch.float32)
wq= torch.tensor(wq, dtype=torch.float32)
wv= torch.tensor(wv, dtype=torch.float32)


k= (x @ wk).unsqueeze(0)     # to batch mode
q= (x @ wq).unsqueeze(0)
v= (x @ wv).unsqueeze(0)

m_attn= nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
attn_o, attn_ow= m_attn(q, k, v)

attn_o.shape, attn_ow.shape
# %%
attn= whisper.model.MultiHeadAttention(
    n_state= 4, n_head= 2
)

attn.named_parameters()

# %%
