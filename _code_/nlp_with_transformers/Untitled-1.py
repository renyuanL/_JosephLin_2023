# %%
# Uncomment and run this cell if you're on Colab or Kaggle
# !git clone https://github.com/nlp-with-transformers/notebooks.git
# %cd notebooks
# from install import *
# install_requirements()

# %%
#hide
from utils import *
setup_chapter()

# %% [markdown]
# # Hello Transformers

# %% [markdown]
# <img alt="transformer-timeline" caption="The transformers timeline" src="images/chapter01_timeline.png" id="transformer-timeline"/>

# %% [markdown]
# ## The Encoder-Decoder Framework

# %% [markdown]
# <img alt="rnn" caption="Unrolling an RNN in time." src="images/chapter01_rnn.png" id="rnn"/>

# %% [markdown]
# <img alt="enc-dec" caption="Encoder-decoder architecture with a pair of RNNs. In general, there are many more recurrent layers than those shown." src="images/chapter01_enc-dec.png" id="enc-dec"/>

# %% [markdown]
# ## Attention Mechanisms

# %% [markdown]
# <img alt="enc-dec-attn" caption="Encoder-decoder architecture with an attention mechanism for a pair of RNNs." src="images/chapter01_enc-dec-attn.png" id="enc-dec-attn"/> 

# %% [markdown]
# <img alt="attention-alignment" width="500" caption="RNN encoder-decoder alignment of words in English and the generated translation in French (courtesy of Dzmitry Bahdanau)." src="images/chapter02_attention-alignment.png" id="attention-alignment"/> 

# %% [markdown]
# <img alt="transformer-self-attn" caption="Encoder-decoder architecture of the original Transformer." src="images/chapter01_self-attention.png" id="transformer-self-attn"/> 

# %% [markdown]
# ## Transfer Learning in NLP

# %% [markdown]
# <img alt="transfer-learning" caption="Comparison of traditional supervised learning (left) and transfer learning (right)." src="images/chapter01_transfer-learning.png" id="transfer-learning"/>  

# %% [markdown]
# <img alt="ulmfit" width="500" caption="The ULMFiT process (courtesy of Jeremy Howard)." src="images/chapter01_ulmfit.png" id="ulmfit"/>

# %% [markdown]
# ## Hugging Face Transformers: Bridging the Gap

# %% [markdown]
# ## A Tour of Transformer Applications

# %%
text = """Dear Amazon, last week I ordered an Optimus Prime action figure \
from your online store in Germany. Unfortunately, when I opened the package, \
I discovered to my horror that I had been sent an action figure of Megatron \
instead! As a lifelong enemy of the Decepticons, I hope you can understand my \
dilemma. To resolve the issue, I demand an exchange of Megatron for the \
Optimus Prime figure I ordered. Enclosed are copies of my records concerning \
this purchase. I expect to hear from you soon. Sincerely, Bumblebee."""

# %% [markdown]
# ### Text Classification

# %%
#hide_output
from transformers import pipeline

classifier = pipeline("text-classification")

# %%
import pandas as pd

outputs = classifier(text)
pd.DataFrame(outputs)    

# %% [markdown]
# ### Named Entity Recognition

# %%
ner_tagger = pipeline("ner", aggregation_strategy="simple")
outputs = ner_tagger(text)
pd.DataFrame(outputs)    

# %% [markdown]
# ### Question Answering 

# %%
reader = pipeline("question-answering")
question = "What does the customer want?"
outputs = reader(question=question, context=text)
pd.DataFrame([outputs])    

# %% [markdown]
# ### Summarization

# %%
summarizer = pipeline("summarization")
outputs = summarizer(
    text, 
    max_length= 100, #45, 
    clean_up_tokenization_spaces=True)
print(outputs[0]['summary_text'])

# %% [markdown]
# ### Translation

# %%
translator = pipeline("translation_en_to_de", 
                      model="Helsinki-NLP/opus-mt-en-de")
outputs = translator(text, clean_up_tokenization_spaces=True, min_length=100)
print(outputs[0]['translation_text'])

# %% [markdown]
# ### Text Generation

# %%
#hide
from transformers import set_seed
set_seed(42) # Set the seed to get reproducible results

# %%
generator = pipeline("text-generation")
response = "Dear Bumblebee, I am sorry to hear that your order was mixed up."
prompt = text + "\n\nCustomer service response:\n" + response
outputs = generator(prompt, max_length=200)
print(outputs[0]['generated_text'])

# %% [markdown]
# ## The Hugging Face Ecosystem

# %% [markdown]
# <img alt="ecosystem" width="500" caption="An overview of the Hugging Face ecosystem of libraries and the Hub." src="images/chapter01_hf-ecosystem.png" id="ecosystem"/>

# %% [markdown]
# ### The Hugging Face Hub

# %% [markdown]
# <img alt="hub-overview" width="1000" caption="The models page of the Hugging Face Hub, showing filters on the left and a list of models on the right." src="images/chapter01_hub-overview.png" id="hub-overview"/> 

# %% [markdown]
# <img alt="hub-model-card" width="1000" caption="A example model card from the Hugging Face Hub. The inference widget is shown on the right, where you can interact with the model." src="images/chapter01_hub-model-card.png" id="hub-model-card"/> 

# %% [markdown]
# ### Hugging Face Tokenizers

# %% [markdown]
# ### Hugging Face Datasets

# %% [markdown]
# ### Hugging Face Accelerate

# %% [markdown]
# ## Main Challenges with Transformers

# %% [markdown]
# ## Conclusion

# %%



