
#%%
fn_wav= fn_wav_00= 'ry35words.wav'

import torch
import torchaudio
import whisper
import whisper as wp

device= ('cuda' if torch.cuda.is_available() else 
         'cpu')

def get_md(model_size= 'tiny'):
    md= whisper.load_model(model_size)
    md= md.to(device)
    return md

def get_wav(fn_wav= 'ry35words.wav'):
    wav, sr= torchaudio.load(fn_wav)
    wav= wav.squeeze()
    wav= wav.to(device)
    return wav
#%% the simplest way to run the model
md= get_md()


x=  get_wav()
y=  md.transcribe(x)
print(f'{y= }')

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

#%% dive into the model

print(f'{md.dims= }, {md= }')

theDims= '''
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

# 發現 whisper 原始的 model 有一個 bug, 可能不影響辨認結果，或者影響很小，但是還是要修正一下。
#
# 它原來的 code 是這樣的：這應是paper 上的公式，
# 但是實際上，它在 forward 的時候，卻把 attn_ln 放在了 attn 之前先做。
# 這樣的話，就不是原來的公式了。
# 由於 training 和 recog 必須一致，因此，我們不能改他們的 forward，
# 只能將錯就錯，改他們的 model definition。
# 他們的 forward 也許不算大錯，可能還是可以收斂，但畢竟不是原來的公式，
# 寫在 model __init__，裡面的畢竟是原來的公式。因此我們把它在 __init__ 裡面改回來(反映真正程式的運作)。
# 再仔細看一下，我們發現，其實都是 LayerNorm， 只差開頭和結尾的位置，可能少做或多做一次。
# 而 LayerNorm 在最後一層重複做，或最前層少做，應該不會有太大的影響。
#
# 總之，發現原程式有 bug，也是功德一件，找個時間在與作者討論一下。
#

theMd_ORI= ''' Whisper(
  (encoder): AudioEncoder(
  
    (conv1): Conv1d( 80, 384, kernel_size=(3,), stride=(1,), padding=(1,))
    (conv2): Conv1d(384, 384, kernel_size=(3,), stride=(2,), padding=(1,))

    (blocks): ModuleList(
      (0-3): 4 x ResidualAttentionBlock(
        (attn): MultiHeadAttention(
          (query):  Linear(in_features=384, out_features=384, bias=True)
          (key):    Linear(in_features=384, out_features=384, bias=False)
          (value):  Linear(in_features=384, out_features=384, bias=True)
          (out):    Linear(in_features=384, out_features=384, bias=True)
        )
        (attn_ln):  LayerNorm((384,), eps=1e-05, elementwise_affine=True) ## BUG 在此

        (mlp): Sequential(
          (0): Linear(in_features=384, out_features=1536, bias=True)
          (1): GELU(approximate='none')
          (2): Linear(in_features=1536, out_features=384, bias=True)
        )
        (mlp_ln): LayerNorm((384,), eps=1e-05, elementwise_affine=True) ## BUG 在此
      )
    )
    (ln_post): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
  )

  (decoder): TextDecoder(
  
    (token_embedding): Embedding(51_865, 384)

    (blocks): ModuleList(
      (0-3): 4 x ResidualAttentionBlock(
        (attn): MultiHeadAttention(
          (query):  Linear(in_features=384, out_features=384, bias=True)
          (key):    Linear(in_features=384, out_features=384, bias=False)
          (value):  Linear(in_features=384, out_features=384, bias=True)
          (out):    Linear(in_features=384, out_features=384, bias=True)
        )
        (attn_ln):  LayerNorm((384,), eps=1e-05, elementwise_affine=True) ## BUG 在此

        (cross_attn): MultiHeadAttention(
          (query):  Linear(in_features=384, out_features=384, bias=True)
          (key):    Linear(in_features=384, out_features=384, bias=False)
          (value):  Linear(in_features=384, out_features=384, bias=True)
          (out):    Linear(in_features=384, out_features=384, bias=True)
        )
        (cross_attn_ln): LayerNorm((384,), eps=1e-05, elementwise_affine=True) ## BUG 在此

        (mlp): Sequential(
          (0): Linear(in_features=384, out_features=1_536, bias=True)
          (1): GELU(approximate='none')
          (2): Linear(in_features=1_536, out_features=384, bias=True)
        )
        (mlp_ln): LayerNorm((384,), eps=1e-05, elementwise_affine=True) ## BUG 在此
      )
    )
    (ln): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
  )
)'''


theModel_ryEdit= '''md= Whisper(

  (encoder): AudioEncoder(
  
    (conv1): Conv1d( 80, 384, kernel_size=(3,), stride=(1,), padding=(1,))
    (conv2): Conv1d(384, 384, kernel_size=(3,), stride=(2,), padding=(1,))
    
    (blocks): ModuleList(
      (0-3): 4 x ResidualAttentionBlock(
        
        (attn_ln): LayerNorm((384,), eps=1e-05, elementwise_affine=True) #### ryEdit
        
        (attn): MultiHeadAttention(
          (query):  Linear(in_features=384, out_features=384, bias=True)
          (key):    Linear(in_features=384, out_features=384, bias=False)
          (value):  Linear(in_features=384, out_features=384, bias=True)
          (out):    Linear(in_features=384, out_features=384, bias=True)
        )
        
        (mlp_ln): LayerNorm((384,), eps=1e-05, elementwise_affine=True) #### ryEdit
        
        (mlp): Sequential(
          (0): Linear(in_features=384, out_features=1_536, bias=True)
          (1): GELU(approximate='none')
          (2): Linear(in_features=1_536, out_features=384, bias=True)
        )
      )
    )
    
    (ln_post): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
  )

  (decoder): TextDecoder(
    
    (token_embedding): Embedding(51_865, 384)
    
    (blocks): ModuleList(
      (0-3): 4 x ResidualAttentionBlock(
        
        (attn_ln): LayerNorm((384,), eps=1e-05, elementwise_affine=True) #### ryEdit
        
        (attn): MultiHeadAttention(
          (query):  Linear(in_features=384, out_features=384, bias=True)
          (key):    Linear(in_features=384, out_features=384, bias=False)
          (value):  Linear(in_features=384, out_features=384, bias=True)
          (out):    Linear(in_features=384, out_features=384, bias=True)
        )
        
        (cross_attn_ln): LayerNorm((384,), eps=1e-05, elementwise_affine=True) #### ryEdit
        
        (cross_attn): MultiHeadAttention(
          (query):  Linear(in_features=384, out_features=384, bias=True)
          (key):    Linear(in_features=384, out_features=384, bias=False)
          (value):  Linear(in_features=384, out_features=384, bias=True)
          (out):    Linear(in_features=384, out_features=384, bias=True)
        )
        
        (mlp_ln): LayerNorm((384,), eps=1e-05, elementwise_affine=True) #### ryEdit
        
        (mlp): Sequential(
          (0): Linear(in_features=384, out_features=1_536, bias=True)
          (1): GELU(approximate='none')
          (2): Linear(in_features=1_536, out_features=384, bias=True)
        )
      )
    )
    
    (ln): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
  )
)'''


#%% %% dive into the model

wav= get_wav() # torch.Size([580_160]) # 36.26 sec

#%%
t0=0 #sec
dt=5 #sec
w5sec= wav[t0*16_000:(t0+dt)*16_000] # torch.Size([240_000]) # 15 sec

a30sec=  wp.pad_or_trim(w5sec)              # -->torch.Size([480_000])
mel= wp.audio.log_mel_spectrogram(a30sec) # -->torch.Size([80, 3000])
mel= mel[None, ...]                       # -->torch.Size([1, 80, 3000])


Xa= enc= md.encoder(mel)                  # -->torch.Size([1, 1500, 384])

# 給定開頭的文字 X，
# 在上面聲音(特徵) Xa 壓陣的條件下，
# 不斷預測接下來的文字、、

X0= tok= torch.tensor(
   [[50_258, 50_259, 50_359]]
   ) # torch.Size([1, 3])
X0= X0.to(device)

X0_txt= txt4tok(X0)  
print(f'{X0_txt= }')

for i in range(100):
  
  X1= md.decoder(X0, Xa)               # torch.Size([1, 3, 51865])

  x_next= X1[..., -1,:]                    # torch.Size([1, 51865])
  y_next= x_next.argmax(dim=-1, keepdims=True) # torch.Size([1, 1])
  X0=  torch.cat([X0, y_next], axis=-1)     # torch.Size([1, 4])
  
  y_next= txt4tok(y_next)           # '<|0.00|> Backward, bed, bird, ...'
  print(f'{i= }, {y_next= }')
  
  if y_next in ['<|endoftext|>']:
    break

X0.shape # torch.Size([1, 103])

X0_txt= txt4tok(X0)  
print(f'{i= }, {X0_txt= }')

# %%
