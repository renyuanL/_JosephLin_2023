#%%
import whisper

model=  whisper.load_model("tiny")
result= model.transcribe("ry35wordsH.wav")
print(result["text"])

#%%
import whisper

fn_wav= 'ry35words.wav'

model= whisper.load_model("tiny")

# get audio 
audio= whisper.load_audio(fn_wav)
#audio= audio[10*16_000:20*16_000]

# pad/trim it to fit 30 seconds
audio= whisper.pad_or_trim(audio)

# make log-Mel spectrogram and move to the same device as the model
mel= whisper.log_mel_spectrogram(audio).to(model.device)

# detect the spoken language
_, probs= model.detect_language(mel)
lang= max(probs, key=probs.get)

# decode the audio
options= whisper.DecodingOptions()
result=  whisper.decode(model, mel, options)

# print the recognized text
print(f"Detected language: {lang}")
print(result.text)
# %%

whisper.decoding.DecodingTask(model, options).run(
    mel[None,...]
)
# %%
# using pytorch directly to get mel spectrogram
import torch
import torchaudio


#fn_wav= 'ry35words.wav'
#audio, sample_rate= torchaudio.load(fn_wav)
sample_rate= 16_000

melsgram= torchaudio.transforms.MelSpectrogram(
    sample_rate= sample_rate,
    n_fft= 400,
    hop_length= 160,
    n_mels= 80
)(torch.from_numpy(audio))[:,:-1].cuda()
melsgram.shape
result1=  whisper.decode(model, melsgram, options)
result1.text

#%%
log_spec = torch.clamp(melsgram, min=1e-10).log10()
log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
log_spec = (log_spec + 4.0) / 4.0

result2=  whisper.decode(model, log_spec, options)
result2.text

#%% Audio_Encoder

xa= model.encoder(mel[None,...]) #(1, 80, 3000) -> (1,1500,384)
xa.shape
# %%


task= whisper.decoding.DecodingTask(model, options)
y= task.run(xa)
y
#%%

x= torch.tensor([task.initial_tokens]).cuda()

qq= task._main_loop(xa, x)

aL= [ q.item() for q in qq[0][0]]
whisper.tokenizer.get_encoding('multilingual').decode(aL)
# %%
from torch import Tensor
import numpy as np

self= task

def _main_loop(self, audio_features: Tensor, tokens: Tensor):

    assert audio_features.shape[0] == tokens.shape[0]
    n_batch = tokens.shape[0]
    sum_logprobs: Tensor = torch.zeros(n_batch, device=audio_features.device)
    #no_speech_probs = [np.nan] * n_batch

    try:
        for i in range(self.sample_len):
            logits = self.inference.logits(tokens, audio_features)

            #if (
            #    i == 0 and self.tokenizer.no_speech is not None
            #):  # save no_speech_probs
            #    probs_at_sot = logits[:, self.sot_index].float().softmax(dim=-1)
            #    no_speech_probs = probs_at_sot[:, self.tokenizer.no_speech].tolist()

            # now we need to consider the logits at the last token only
            logits = logits[:, -1]

            # apply the logit filters, e.g. for suppressing or applying penalty to
            #for logit_filter in self.logit_filters:
            #    logit_filter.apply(logits, tokens)

            # expand the tokens tensor with the selected next tokens
            tokens, completed = self.decoder.update(tokens, logits, sum_logprobs)

            if completed or tokens.shape[-1] > self.n_ctx:
                break
    finally:
        self.inference.cleanup_caching()

    return tokens, sum_logprobs, #no_speech_probs

self.decoder.reset()
q1= _main_loop(self, xa, x)
q1

#%%

def tokens_to_text(tokens):
    aL= [ q.item() for q in tokens]
    y= whisper.tokenizer.get_encoding('multilingual').decode(aL)
    #print(f'{y= }')
    return y

sum_logprobs: Tensor = torch.zeros(1).cuda()

x= torch.tensor([task.initial_tokens]).cuda() 
y= tokens_to_text(x[0])
print(f'{y= }')

# (50258, 50259, 50359)
# y= '<|startoftranscript|><|en|><|transcribe|>'

tokens= x
for i in range(self.sample_len):
    
    logits= self.inference.logits(tokens, xa)

    logits= logits[:, -1]
    tokens, completed= self.decoder.update(tokens, logits, sum_logprobs)

    if completed or tokens.shape[-1] > self.n_ctx:
        break
self.inference.cleanup_caching()

y= tokens_to_text(tokens[0])
print(f'{y= }')

#'''
#y= '<|startoftranscript|><|en|><|transcribe|>
#<|0.00|> Backward, bed, bird, cat, dog, down, 
#eight, five, follow, forward, forward, go, 
#happy, house,<|15.60|><|15.60|><|endoftext|>'
#'''

# %%

# %%
# get the number of parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

count_parameters(model) 
# 37_184_640
# save the model in jit format

# %%
'''
model_jit_fn= 'whisper_tiny_jit.pt'
model_jit= torch.jit.script(model) # Export to TorchScript
model_jit.save(model_jit_fn) # Save the TorchScript model

# load the trained model
md2= torch.jit.load(model_jit_fn)
md2.eval()
md2.to(device)

md2
'''

#%%
# save the model
model_fn= 'whisper_tiny.pt'
torch.save(model.state_dict(), model_fn)

#%% 
# load the model from model_fn
dims= model.dims
md= whisper.model.Whisper(dims)
md.load_state_dict(torch.load(model_fn))


# %%
device= ( 'cuda' if torch.cuda.is_available() else 
          'cpu')
md.eval()
md.to(device)


# %%
# save the variable dims
import pickle
with open('dims.pkl', 'wb') as f:
    pickle.dump(dims, f)

# %%

# %%
# load the variable dims
import pickle
with open('dims.pkl', 'rb') as f:
    dims= pickle.load(f)

# %%
