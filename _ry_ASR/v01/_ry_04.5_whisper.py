
#%%
fn_wav= fn_wav_00= 'ry35words.wav'
fn_wav_01= 'ry35words_zh.wav' # change the wav to Chinese
fn_wav_02= 'ry35wordsH.wav'   # change the wav to Higer pitch

md_fn=  "md.pkl"  # the same as 'tiny'

# load the variable dims
import pickle
import torch
import torchaudio
import whisper
import whisper as wp

def get_md(md_fn=  "md.pkl", model_size= 'tiny'):
    try:
      with open(md_fn, 'rb') as f:
          md= pickle.load(f)
      # md.pkl is not portable between Windows (OK) and Linux (not OK)
      # so we need to re-load the model, if run in Colab (Linux)
    except:
      md= whisper.load_model(model_size)
    return md

def get_wav(fn_wav= 'ry35words.wav'):
    wav, sr= torchaudio.load(fn_wav)
    wav= wav.squeeze()
    return wav

def save_md(md, md_fn=  "md.pkl", device= 'cpu'):
    md= (md.cuda() if device in ['cuda','gpu'] else 
         md.cuda())
    with open(md_fn, 'wb') as f:
        pickle.dump(md, f)

md= get_md()
#%% try different prompt

x=  get_wav(fn_wav)
y=  md.transcribe(
   x, 
   initial_prompt= 'english, 台灣華語, ')
print(f'{y= }')

# %%
# %%

x0= get_wav(fn_wav_00)
x1= get_wav(fn_wav_01)
x2= get_wav(fn_wav_02)

# cat the wav files
x= x0
x= torch.cat([x,x1], dim= -1)
x= torch.cat([x,x2], dim= -1)

x.shape

# %%
y=  md.transcribe(
   x, 
   initial_prompt= 'English, 華語, ', 
   task= 'transcribe'
   )

print(f'{y= }')
# %% try the other pre-trained models
#
# you can test the other pre-trained models to check the performance
# try cuda if you have a gpu, because the model is large, and spends a lot of time
#

# check the model size
# get the number of parameters
def count_parameters(model):
  return sum(p.numel() 
             for p in model.parameters() 
             if p.requires_grad)

x= x.cuda()
for m in [
    #'large',
    #'medium',
    #'small', 
    #'base',
    'tiny'
    ]:
    md= whisper.load_model(m)
    md= md.cuda()
 
    n_params= count_parameters(md)

    y=  md.transcribe(
      x, 
      initial_prompt= 'English, 華語, ', 
      task= 'transcribe'
      )
    print(f'{m= } {n_params= }, {y= }')
    #md= md.cpu()
    del md                   # release the memory
    torch.cuda.empty_cache() # release the memory
  
theModelSize= '''
  m= 'large'  n_params= 1_541_384_960,
  m= 'medium' n_params=   762_321_920,
  m= 'small'  n_params=   240_582_912,  
  m= 'base'   n_params=    71_825_920,  
  m= 'tiny'   n_params=    37_184_640, 
  '''
# %% outout the result to excel
import pandas as pd
# pd.DataFrame([y0,y1,y2,y3,y4]).to_excel('whisper.xlsx')

#%%

#%% 研究一下 tokenizer ...

# (50258, 50259, 50359)
# y= '<|startoftranscript|><|en|><|transcribe|>'

tok= torch.tensor(
   [[50258, 50259, 50359]]
   )

def txt4tok(tok):
  txt= wp.tokenizer.get_encoding(
     'multilingual'
     ).decode(tok[0].tolist())
  return txt

def tok4txt(txt):
  tok= wp.tokenizer.get_encoding(
     'multilingual'
     ).encode(txt)
  tok= torch.tensor(tok)[None, ...]
  return tok

txt2tok= tok4txt # txt->tok, tok<-txt
tok2txt= txt4tok # tok->txt, txt<-tok

txt= '0, 1, 2, 3, a, b, c, d, 一, 乙, 丁, 七.'
tok=  tok4txt(txt)
txt1= txt4tok(tok)


txt, tok, txt1
'''
tensor([[   
  15,     11,   
  502,    11,   
  568,    11,  
  805,    11,   
  
  257,    11,
  272,    11,   
  269,    11,   
  274,    11, 
  
  26923,          11,   220,  
  2930,   247,    11,   220,   
  940,    223,    11,   220, 
  29920,          13
]]),
'''

txt= 'a, b, c, d, ant, bug, cat, dog, 蟻, 蟲, 貓, 狗.'
tok=  tok4txt(txt)
tok

'''
tensor([[   
  64,     11,   
  272,    11,   
  269,    11,   
  274,    11,  
  
  2511,    11,
  7426,    11,  
  3857,    11,  
  3000,    11,   220,   
  
  164,     253,   119,  11,  220,   
  164,     253,   110,  11,  220, 
  11561,   241,         11,  220, 
  18637,   245,         13
  ]])
'''

#%% The simplest way to use the model

md= get_md()

print(f'{md.dims= }, {md= }')

md.dims
'''
ModelDimensions(
  n_mels= 80, # 聲音的mel頻譜的維度， sampleRate= 16KHz 
  n_audio_ctx=1500, # 固定的音長，1500 = 30 sec 
  
  n_audio_state=384, # 384 = 6*64, 6 heads, 1 head = 64 維
  n_audio_head=6, 
  n_audio_layer=4, 
  
  n_vocab= 51_865,  # 詞彙表的大小 # 是 相異 tok 的數量 
  n_text_ctx=448,   # 30 sec 內，最大的 tok 長度，448 = 224*2, 224 似乎是「詞」數量的上限

  n_text_state=384, 
  n_text_head=6, 
  n_text_layer=4
'''                

wav= get_wav()
y= md.transcribe(wav)

print(f'{wav.shape= }, {y= }')

#%% %% dive into the model

#%%
a30sec=  wp.pad_or_trim(wav)              # -->torch.Size([480_000])
mel= wp.audio.log_mel_spectrogram(a30sec) # -->torch.Size([80, 3000])
mel= mel[None, ...]                       # -->torch.Size([1, 80, 3000])

Xa= enc= md.encoder(mel)                  # -->torch.Size([1, 1500, 384])

X= tok= torch.tensor(
   [[50_258, 50_259, 50_359]]
   ) # torch.Size([1, 3])

for i in range(100):
  
  x1= md.decoder(X, Xa)                # torch.Size([1, 3, 51865])

  x1= x1[..., -1,:]                    # torch.Size([1, 51865])
  y1= x1.argmax(dim=-1, keepdims=True) # torch.Size([1, 1])
  X=  torch.cat([X, y1], axis=-1)      # torch.Size([1, 4])
  
  y_txt= txt4tok(y1)           # '<|0.00|> Backward, bed, bird, ...'
  print(f'{i= }, {y_txt= }')
#%%
X.shape # torch.Size([1, 103])

txt= txt4tok(X)  
print(f'{i= }, {txt= }')

# %%
