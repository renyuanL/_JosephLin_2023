# %% [markdown]
# # ry do some modification for tutoring, 2023/04/20
# ## using CNN to classify audio of 1 second
# - the output is 8 classes
# 
# |  |  |
# |----|----|
# |yes    | no   |
# |go     | stop |
# |up     | down |
# |left   | right|
# 
# 

# %% [markdown]
# 
# ##### Copyright 2020 The TensorFlow Authors.

# %%
#@title Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# %% [markdown]
# # Simple audio recognition: Recognizing keywords

# %% [markdown]
# <table class="tfo-notebook-buttons" align="left">
#   <td>
#     <a target="_blank" href="https://www.tensorflow.org/tutorials/audio/simple_audio">
#     <img src="https://www.tensorflow.org/images/tf_logo_32px.png" />
#     View on TensorFlow.org</a>
#   </td>
#   <td>
#     <a target="_blank" href="https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/audio/simple_audio.ipynb">
#     <img src="https://www.tensorflow.org/images/colab_logo_32px.png" />
#     Run in Google Colab</a>
#   </td>
#   <td>
#     <a target="_blank" href="https://github.com/tensorflow/docs/blob/master/site/en/tutorials/audio/simple_audio.ipynb">
#     <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
#     View source on GitHub</a>
#   </td>
#   <td>
#     <a href="https://storage.googleapis.com/tensorflow_docs/docs/site/en/tutorials/audio/simple_audio.ipynb"><img src="https://www.tensorflow.org/images/download_logo_32px.png" />Download notebook</a>
#   </td>
# </table>

# %% [markdown]
# This tutorial demonstrates how to preprocess audio files in the WAV format and build and train a basic [automatic speech recognition](https://en.wikipedia.org/wiki/Speech_recognition) (ASR) model for recognizing ten different words. You will use a portion of the [Speech Commands dataset](https://www.tensorflow.org/datasets/catalog/speech_commands) ([Warden, 2018](https://arxiv.org/abs/1804.03209)), which contains short (one-second or less) audio clips of commands, such as "down", "go", "left", "no", "right", "stop", "up" and "yes".
# 
# Real-world speech and audio recognition [systems](https://ai.googleblog.com/search/label/Speech%20Recognition) are complex. But, like [image classification with the MNIST dataset](../quickstart/beginner.ipynb), this tutorial should give you a basic understanding of the techniques involved.

# %% [markdown]
# ## Setup
# 
# Import necessary modules and dependencies. You'll be using `tf.keras.utils.audio_dataset_from_directory` (introduced in TensorFlow 2.10), which helps generate audio classification datasets from directories of `.wav` files. You'll also need [seaborn](https://seaborn.pydata.org) for visualization in this tutorial.

# %%
#!pip install -U -q tensorflow tensorflow_datasets
#!apt install --allow-change-held-packages libcudnn8=8.1.0.77-1+cuda11.2

# %%
import os
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras import models
from IPython import display

# Set the seed value for experiment reproducibility.
seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)

# %% [markdown]
# ## Import the mini Speech Commands dataset
# 
# To save time with data loading, you will be working with a smaller version of the Speech Commands dataset. The [original dataset](https://www.tensorflow.org/datasets/catalog/speech_commands) consists of over 105,000 audio files in the [WAV (Waveform) audio file format](https://www.aelius.com/njh/wavemetatools/doc/riffmci.pdf) of people saying 35 different words. This data was collected by Google and released under a CC BY license.
# 
# Download and extract the `mini_speech_commands.zip` file containing the smaller Speech Commands datasets with `tf.keras.utils.get_file`:

# %%
DATASET_PATH = 'data/mini_speech_commands'

data_dir = pathlib.Path(DATASET_PATH)
if not data_dir.exists():
  tf.keras.utils.get_file(
      'mini_speech_commands.zip',
      origin="http://storage.googleapis.com/download.tensorflow.org/data/mini_speech_commands.zip",
      extract=True,
      cache_dir='.', cache_subdir='data')

# %% [markdown]
# The dataset's audio clips are stored in eight folders corresponding to each speech command: `no`, `yes`, `down`, `go`, `left`, `up`, `right`, and `stop`:

# %%
qqq= commands = np.array(tf.io.gfile.listdir(str(data_dir)))
commands = commands[(commands != 'README.md') & (commands != '.DS_Store')]
print('Commands:', commands)
qqq

# %% [markdown]
# Divided into directories this way, you can easily load the data using `keras.utils.audio_dataset_from_directory`. 
# 
# The audio clips are 1 second or less at 16kHz. The `output_sequence_length=16000` pads the short ones to exactly 1 second (and would trim longer ones) so that they can be easily batched.

# %%
train_ds, val_ds = tf.keras.utils.audio_dataset_from_directory(
    directory= data_dir,
    batch_size= 64,
    validation_split= 0.2,
    seed= 0,
    output_sequence_length= 16_000,
    subset='both')

label_names = np.array(train_ds.class_names)
print()
print("label names:", label_names)
_label_names_= [
    'down', 
    'go', 
    'left', 
    'no', 
    'right', 
    'stop', 
    'up', 
    'yes']

_rearranged_label_names_= '''
yes,    no,
go,     stop,
up,     down, 
left,   right,
'''
# on,      off 
# forward, backward
# marvin,  sheila
# dog,     cat
# happy,   ?sad?
# bed,     bird
# house,   tree

# 0, 1, 2, 3, 4, 5, 6, 7, 8, 9


# %% [markdown]
# The dataset now contains batches of audio clips and integer labels. The audio clips have a shape of `(batch, samples, channels)`. 

# %%
train_ds.element_spec

# %% [markdown]
# This dataset only contains single channel audio, so use the `tf.squeeze` function to drop the extra axis:

# %%
def squeeze(audio, labels):
  audio= tf.squeeze(audio, axis=-1)
  return audio, labels

train_ds= train_ds.map(squeeze, tf.data.AUTOTUNE)
val_ds=   val_ds.map(squeeze, tf.data.AUTOTUNE)

# %% [markdown]
# The `utils.audio_dataset_from_directory` function only returns up to two splits. It's a good idea to keep a test set separate from your validation set.
# Ideally you'd keep it in a separate directory, but in this case you can use `Dataset.shard` to split the validation set into two halves. Note that iterating over **any** shard will load **all** the data, and only keep its fraction. 

# %%
list(train_ds.take(1)) # 1 batch

# %%
test_ds= val_ds.shard(num_shards=2, index=0)
val_ds=  val_ds.shard(num_shards=2, index=1)

# %%
list(test_ds.take(1)) # 1 batch

# %%
for example_audio, example_labels in train_ds.take(1):  
  print(example_audio.shape)
  print(example_labels.shape)

# %% [markdown]
# Let's plot a few audio waveforms:

# %%
label_names

# %%
rows = 3
cols = 3
n = rows * cols
fig, axes = plt.subplots(rows, cols, figsize=(16, 9))

for i in range(n):
  if i>=n:
    break
  r = i // cols
  c = i % cols
  
  label=     example_labels[i]
  labelName= label_names[label]

  label= f'{labelName}({label})'

  ax = axes[r][c]
  ax.plot(example_audio[i].numpy())
  ax.set_yticks(np.arange(-1.2, 1.2, 0.2))
  
  ax.set_title(label)
  ax.set_ylim([-1.1,1.1])

plt.show()

# %% [markdown]
# ## Convert waveforms to spectrograms
# 
# The waveforms in the dataset are represented in the time domain. Next, you'll transform the waveforms from the time-domain signals into the time-frequency-domain signals by computing the [short-time Fourier transform (STFT)](https://en.wikipedia.org/wiki/Short-time_Fourier_transform) to convert the waveforms to as [spectrograms](https://en.wikipedia.org/wiki/Spectrogram), which show frequency changes over time and can be represented as 2D images. You will feed the spectrogram images into your neural network to train the model.
# 
# A Fourier transform (`tf.signal.fft`) converts a signal to its component frequencies, but loses all time information. In comparison, STFT (`tf.signal.stft`) splits the signal into windows of time and runs a Fourier transform on each window, preserving some time information, and returning a 2D tensor that you can run standard convolutions on.
# 
# Create a utility function for converting waveforms to spectrograms:
# 
# - The waveforms need to be of the same length, so that when you convert them to spectrograms, the results have similar dimensions. This can be done by simply zero-padding the audio clips that are shorter than one second (using `tf.zeros`).
# - When calling `tf.signal.stft`, choose the `frame_length` and `frame_step` parameters such that the generated spectrogram "image" is almost square. For more information on the STFT parameters choice, refer to [this Coursera video](https://www.coursera.org/lecture/audio-signal-processing/stft-2-tjEQe) on audio signal processing and STFT.
# - The STFT produces an array of complex numbers representing magnitude and phase. However, in this tutorial you'll only use the magnitude, which you can derive by applying `tf.abs` on the output of `tf.signal.stft`.

# %%
def get_spectrogram(waveform):
  # Convert the waveform to a spectrogram via a STFT.
  
  # padding n zeros to the end of the signal

  frame_length= 256 # 255 # 256/16000= 0.016 sec
  frame_step=   160 # 160/16000= 0.01 sec
  padding_size= frame_length-frame_step # 16000= 1 sec
  

  #waveform_0= tf.pad( waveform, [[0, padding_size]]) # 16000= 1 sec
  ##[[0, padding_size]]) # 16000= 1 sec

  spectrogram = tf.signal.stft(
      waveform, 
      frame_length= frame_length, # 255, # 256/16000= 0.016 sec
      frame_step=   frame_step,  # 160/16000= 0.01 sec
      #padding= 'SAME'
      )
  # Obtain the magnitude of the STFT.
  spectrogram = tf.abs(spectrogram)
  # Add a `channels` dimension, so that the spectrogram can be used
  # as image-like input data with convolution layers (which expect
  # shape (`batch_size`, `height`, `width`, `channels`).
  spectrogram = spectrogram[..., tf.newaxis]
  return spectrogram

def get_mel_scaled_spectrogram(waveform):
  # Convert the waveform to a spectrogram via a STFT.
  spectrogram= tf.signal.stft(
      waveform, 
      frame_length= 1024, 
      frame_step= 117 # 160*94//128= 117 
                      # to generrate 128 frames exactly
      )
  # spectrogram.shape == (batch_size, 100, 513)
  # 160*94//128= 117
  # 160*94//100= 150
  # 128*128= 16_384 ~ 16_000

  # Obtain the magnitude of the STFT.
  spectrogram= tf.abs(spectrogram)

  # Compute the mel-scale spectrogram.
  num_spectrogram_bins= spectrogram.shape[-1]
  
  lower_edge_hertz, upper_edge_hertz, num_mel_bins= 80.0, 7600.0, 128 #100 #128 #80

  linear_to_mel_weight_matrix= tf.signal.linear_to_mel_weight_matrix(
      num_mel_bins, 
      num_spectrogram_bins, 
      16_000, 
      lower_edge_hertz,
      upper_edge_hertz
      )
  
  mel_spectrogram= tf.tensordot(
      spectrogram,
      linear_to_mel_weight_matrix,  
      1 
      )
  
  mel_spectrogram= mel_spectrogram[..., tf.newaxis]

  '''
  # Compute a stabilized log to get log-magnitude mel-scale spectrograms.
  log_mel_spectrogram = tf.math.log(mel_spectrogram + 1.e-6)
  # Add a `channels` dimension, so that the spectrogram can be used
  # as image-like input data with convolution layers (which expect
(`batch_size`, `height`, `width`, `channels`).
  log_mel_spectrogram= log_mel_spectrogram[..., tf.newaxis]
  mel_spectrogram=     mel_spectrogram[..., tf.newaxis]
  
  spectrogram = log_mel_spectrogram
  '''
  spectrogram = mel_spectrogram
  
  return spectrogram

get_spectrogram= get_mel_scaled_spectrogram


# %% [markdown]
# Next, start exploring the data. Print the shapes of one example's tensorized waveform and the corresponding spectrogram, and play the original audio:

# %%
for i in range(3):
  label=    label_names[example_labels[i]]
  waveform= example_audio[i]

  spectrogram= get_spectrogram(waveform)

  print('Label:', label)
  print('Waveform shape:', waveform.shape)
  print('Spectrogram shape:', spectrogram.shape)
  print('Audio playback')
  display.display(display.Audio(waveform, rate=16000))

# %% [markdown]
# Now, define a function for displaying a spectrogram:

# %%
def plot_spectrogram_ori(spectrogram, ax):
  if len(spectrogram.shape) > 2:
    assert len(spectrogram.shape) == 3
    spectrogram = np.squeeze(spectrogram, axis=-1)
  # Convert the frequencies to log scale and transpose, so that the time is
  # represented on the x-axis (columns).
  # Add an epsilon to avoid taking a log of zero.
  log_spec= np.log(spectrogram.T + np.finfo(float).eps)
  height=   log_spec.shape[0]
  width=    log_spec.shape[1]
  X= np.linspace(0, np.size(spectrogram), num=width, dtype=int)
  Y= range(height)
  ax.pcolormesh(X, Y, log_spec)

def plot_spectrogram(spectrogram, ax):
  if len(spectrogram.shape) > 2:
    assert len(spectrogram.shape) == 3
    spectrogram = np.squeeze(spectrogram, axis=-1)

  # Convert the frequencies to log scale and transpose, so that the time is
  # represented on the x-axis (columns).
  # Add an epsilon to avoid taking a log of zero.
  log_spec= np.log(
    spectrogram.T # time in x-axis, freq in y-axis
    + np.finfo(float).eps # a small number to avoid log(0)
    )
  height=   log_spec.shape[0]
  width=    log_spec.shape[1]
  
  X= np.linspace(
    0, 
    width, #np.size(spectrogram), 
    num= width, 
    dtype=int)

  Y= range(height)

  ax.pcolormesh(
    X, 
    Y, 
    log_spec
    )

# %% [markdown]
# Plot the example's waveform over time and the corresponding spectrogram (frequencies over time):

# %%
waveform.shape, spectrogram.shape

# %%

fig, axes = plt.subplots(2, figsize=(12, 8))
#waveform= waveform[:16000]
timescale = np.arange(waveform.shape[0])
axes[0].plot(timescale, waveform.numpy())
axes[0].set_title('Waveform')
axes[0].set_xlim([0, 16_000])

plot_spectrogram(spectrogram.numpy(), axes[1])
axes[1].set_title('Spectrogram (mel-scaled)')
plt.suptitle(label.title())
plt.ylabel('freq in mel-scale')
plt.xlabel('time in frames (.01 sec/frame)')
plt.show()

# %% [markdown]
# Now, create spectrogramn datasets from the audio datasets:

# %%
def make_spec_ds(ds):
  return ds.map(
      map_func=lambda audio,label: (get_spectrogram(audio), label),
      num_parallel_calls=tf.data.AUTOTUNE)

# %%
train_spectrogram_ds= make_spec_ds(train_ds)
val_spectrogram_ds=   make_spec_ds(val_ds)
test_spectrogram_ds=  make_spec_ds(test_ds)

# %% [markdown]
# Examine the spectrograms for different examples of the dataset:

# %%
for example_spectrograms, example_spect_labels in train_spectrogram_ds.take(1):
  break

# %%
rows = 4
cols = 4
n = rows*cols
fig, axes = plt.subplots(rows, cols, figsize=(16, 16))

#np.random.seed(0)

ryLabels= [
  'yes',  'yes',
  'no',   'no',
  'go',   'go',
  'stop', 'stop',
  'up',   'up',
  'down', 'down',
  'left', 'left',
  'right','right'
]

k=0
for lab in ryLabels:
    for i in range(n):
        r = k // cols
        c = k % cols
        ax = axes[r][c]
        
        j= np.random.randint(0, len(example_spectrograms))
        #j=i

        x= example_spectrograms[j].numpy()
        y= label_names[example_spect_labels[j].numpy()]
        if y==lab:
            plot_spectrogram(x, ax)
            ax.set_title(y)
            k+=1
            break
plt.show()

# %% [markdown]
# ## Build and train the model

# %% [markdown]
# Add `Dataset.cache` and `Dataset.prefetch` operations to reduce read latency while training the model:

# %%
train_spectrogram_ds= train_spectrogram_ds.cache().shuffle(10_000).prefetch(tf.data.AUTOTUNE)
val_spectrogram_ds=   val_spectrogram_ds.cache().prefetch(tf.data.AUTOTUNE)
test_spectrogram_ds=  test_spectrogram_ds.cache().prefetch(tf.data.AUTOTUNE)

# %% [markdown]
# For the model, you'll use a simple convolutional neural network (CNN), since you have transformed the audio files into spectrogram images.
# 
# Your `tf.keras.Sequential` model will use the following Keras preprocessing layers:
# 
# - `tf.keras.layers.Resizing`: to downsample the input to enable the model to train faster.
# - `tf.keras.layers.Normalization`: to normalize each pixel in the image based on its mean and standard deviation.
# 
# For the `Normalization` layer, its `adapt` method would first need to be called on the training data in order to compute aggregate statistics (that is, the mean and the standard deviation).

# %%
input_shape= example_spectrograms.shape[1:]
num_labels=  len(label_names)

print(f'{input_shape= }\n{num_labels= }')

# Instantiate the `tf.keras.layers.Normalization` layer.
norm_layer= layers.Normalization()
# Fit the state of the layer to the spectrograms
# with `Normalization.adapt`.
norm_layer.adapt(
    data= train_spectrogram_ds.map(
        map_func= lambda spec, label: spec))

model= models.Sequential([
    
    layers.Input(shape= input_shape), 
    
    # Downsample the input.
    # (100, 100) --> (32, 32)

    layers.Resizing(32, 32), 
    # 32*32= 1024 
    # each spectrogram becomes an image of 32x32 pixels 
    
    # Normalize.
    norm_layer,
    
    layers.Conv2D(
        32,    # filters 
        (5,5), # kernel_size
        activation='relu',
        padding='same'),
    
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Dropout(1/2),

    layers.Conv2D(
        64,    # filters
        (3,3), # kernel_size
        activation='relu',
        padding='same'),
    
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Dropout(1/2),

    layers.Conv2D(
        128,    # filters
        (2,2), # kernel_size
        activation='relu',
        padding='same'),
    
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Dropout(1/2),
    
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(1/2),
    layers.Dense(64, activation='relu'),
    layers.Dropout(1/2),
    
    layers.Dense(num_labels, activation='softmax'),
])

print(model.summary())


# %% [markdown]
# ```python
# Model: "sequential"
# _________________________________________________________________
#  Layer (type)    Output Shape          Param #   
# =================================================================
# (Resizing)       (None, 32, 32,   1)   0      # (100,100,1) --> (32,32,1)
# (Normalization)  (None, 32, 32,   1)   3      # why 3?  
# (Conv2D)         (None, 32, 32,  20)   200    # 1*20*3*3 + 20 =   200
# (MaxPooling2D)   (None, 16, 16,  20)   0      
# (Dropout)        (None, 16, 16,  20)   0      
# (Conv2D)         (None, 16, 16,  50)   9_050  # 20*50*3*3 + 50 = 9,050
# (MaxPooling2D)   (None,  8,  8,  50)   0       
# (Dropout)        (None,  8,  8,  50)   0       
# (Conv2D)         (None,  8,  8, 100)   45_100 # 50*100*3*3 + 100 = 45,100
# (MaxPooling2D)   (None,  4,  4, 100)   0         
# (Dropout)        (None,  4,  4, 100)   0         
# (Flatten)        (None, 1_600)       0 # (32/2/2/2)*(32/2/2/2)*100= 1,600 
# (Dense)          (None, 256)         409_856 # 1600*256 + 256   = 409,856  
# (Dropout)        (None, 256)         0         
# (Dense)          (None, 64)          16_448  # 256*64 + 64 = 16,448
# (Dropout)        (None, 64)          0         
# (Dense)          (None, 8)           520     # 64*8 + 8 = 520
# =================================================================
# Total params: 481,177  
# Trainable params: 481,174  # 481,177 - 3 = 481,174
# Non-trainable params: 3
# _________________________________________________________________
# ```

# %% [markdown]
# Configure the Keras model with the Adam optimizer and the cross-entropy loss:

# %%
model.compile(
    optimizer= tf.keras.optimizers.Adam(),
    loss= tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True
    ),
    metrics=['accuracy']
)

# %% [markdown]
# Train the model over 10 epochs for demonstration purposes:

# %%
EPOCHS= 100
history= model.fit(
    train_spectrogram_ds,
    validation_data= val_spectrogram_ds,
    epochs= EPOCHS,
    callbacks= tf.keras.callbacks.EarlyStopping(
        verbose=  1, 
        patience= 10,
        restore_best_weights= True,
        monitor="val_accuracy",
        
        )
    )

# '''
# Epoch 32/50
# 100/100 [==============================] - 1s 9ms/step 
# - loss: 0.1749 - accuracy: 0.9430 
# - val_loss: 0.3872 - val_accuracy: 0.8932
# Epoch 32: early stopping
# '''


# %% [markdown]
# Let's plot the training and validation loss curves to check how your model has improved during training:

# %%
metrics = history.history
plt.figure(figsize=(16,6))
plt.subplot(1,2,1)
plt.plot(history.epoch, metrics['loss'], metrics['val_loss'])
plt.legend(['loss', 'val_loss'])
plt.ylim([0, max(plt.ylim())])
plt.xlabel('Epoch')
plt.ylabel('Loss [CrossEntropy]')

plt.subplot(1,2,2)
plt.plot(history.epoch, 100*np.array(metrics['accuracy']), 100*np.array(metrics['val_accuracy']))
plt.legend(['accuracy', 'val_accuracy'])
plt.ylim([0, 100])
plt.xlabel('Epoch')
plt.ylabel('Accuracy [%]')

# %% [markdown]
# ## Evaluate the model performance
# 
# Run the model on the test set and check the model's performance:

# %%
print(model.summary())


# %% [markdown]
# ```python
# Model: "sequential_3"
# _________________________________________________________________
#  Layer (type)                Output Shape              Param #   
# =================================================================
# (Resizing)       (None, 32, 32, 1)         0         
# 
# (Normalization)  (None, 32, 32, 1)        3         
#                                                             
# 
# (Conv2D)           (None, 32, 32, 20)        200       
# 
# (MaxPooling2D)  (None, 16, 16, 20)       0         
#                                                              
# 
# (Dropout)        (None, 16, 16, 20)        0         
# 
# (Conv2D)          (None, 16, 16, 50)        9050      
# 
# (MaxPooling2D)  (None, 8, 8, 50)         0         
#                                                              
# 
# (Dropout)        (None, 8, 8, 50)          0         
# 
# (Conv2D)          (None, 8, 8, 100)         45100     
# 
# (MaxPooling2D)  (None, 4, 4, 100)        0         
#                                                             
# 
# (Dropout)        (None, 4, 4, 100)         0         
# 
# (Flatten)         (None, 1600)              0         
# 
# (Dense)             (None, 256)               409856    
# 
# (Dropout)        (None, 256)               0         
# 
# (Dense)            (None, 64)                16448     
# 
# (Dropout)        (None, 64)                0         
# 
# (Dense)            (None, 8)                 520       
# 
# =================================================================
# Total params: 481,177
# Trainable params: 481,174
# Non-trainable params: 3
# _________________________________________________________________
# ```

# %%

model.evaluate(test_spectrogram_ds, return_dict=True)

# %% [markdown]
# ### Display a confusion matrix
# 
# Use a [confusion matrix](https://developers.google.com/machine-learning/glossary#confusion-matrix) to check how well the model did classifying each of the commands in the test set:
# 

# %%
y_pred = model.predict(test_spectrogram_ds)

# %%
y_pred = tf.argmax(y_pred, axis=1)

# %%
y_true = tf.concat(list(test_spectrogram_ds.map(lambda s,lab: lab)), axis=0)

# %%
confusion_mtx= tf.math.confusion_matrix(y_true, y_pred)

recall=    confusion_mtx/np.sum(confusion_mtx, axis=1)
precision= confusion_mtx/np.sum(confusion_mtx, axis=0)
f1= 2*precision*recall/(precision+recall)



# %%
# Confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(confusion_mtx,
            xticklabels=label_names,
            yticklabels=label_names,
            annot=True, fmt='g') #, cmap='rainbow')
plt.xlabel('Prediction')
plt.ylabel('Label')
plt.show()

# %%

# Recall rate
plt.figure(figsize=(10, 8))
sns.heatmap(recall,
            xticklabels=label_names,
            yticklabels=label_names,
            annot=True, fmt='g') #, cmap='rainbow')
plt.xlabel('Prediction')
plt.ylabel('Label')
plt.show()


# %%
# Precision rate

plt.figure(figsize=(10, 8))
sns.heatmap(precision,
            xticklabels=label_names,
            yticklabels=label_names,
            annot=True, fmt='g') #, cmap='rainbow')
plt.xlabel('Prediction')
plt.ylabel('Label')
plt.show()


# %%
# F1 score

plt.figure(figsize=(10, 8))
sns.heatmap(f1,
            xticklabels=label_names,
            yticklabels=label_names,
            annot=True, fmt='g') #, cmap='rainbow')
plt.xlabel('Prediction')
plt.ylabel('Label')
plt.show()

# %% [markdown]
# ## Run inference on an audio file
# 
# Finally, verify the model's prediction output using an input audio file of someone saying "no". How well does your model perform?

# %%
#import glob
#glob.glob(f"{data_dir}/stop/*.wav")

# %%
inputWavL= [
    'yes/0ab3b47d_nohash_0.wav',
    'no/01bb6a2a_nohash_0.wav',
    'go/0a9f9af7_nohash_0.wav',
    'stop/012c8314_nohash_0.wav',
    'up/0ab3b47d_nohash_0.wav',
    'down/00b01445_nohash_1.wav',
    'left/00b01445_nohash_0.wav',
    'right/0ab3b47d_nohash_0.wav'
    ]

def rec_1_file(inputWav):
    tgtLabel= inputWav.split('/')[0]

    x= data_dir/inputWav

    x= tf.io.read_file(str(x))
    x, sample_rate= tf.audio.decode_wav(
        x, 
        desired_channels=1, 
        desired_samples= 16_000)

    x= tf.squeeze(x, axis=-1)
    waveform= x
    x= get_spectrogram(x)
    x= x[tf.newaxis,...]

    y_pred= model(x)
    y_pred= y_pred[0]

    x_labels= label_names

    plt.bar(x_labels, y_pred)

    plt.title(f'{tgtLabel= }')
    plt.show()

    display.display(display.Audio(waveform, rate=16_000))
    return x, y_pred, tgtLabel


# %%

for inputWav in inputWavL:
    rec_1_file(inputWav)
    

# %%
#model.save('model.h5')
x= inputWavL[0]
x, y, tgtLabel= rec_1_file(x)
x, y, tgtLabel


# %%
# find the index of the max value

x0= x
aL= [x0]
for layer in model.layers:
    x1= layer(x0)
    aL += [x1]
    x0= x1


# %%
[l.shape for l in aL]

# %% [markdown]
# As the output suggests, your model should have recognized the audio command as "no".

# %% [markdown]
# ## Export the model with preprocessing

# %% [markdown]
# The model's not very easy to use if you have to apply those preprocessing steps before passing data to the model for inference. So build an end-to-end version:

# %%
class ExportModel(tf.Module):
  def __init__(self, model):
    self.model = model

    # Accept either a string-filename or a batch of waveforms.
    # YOu could add additional signatures for a single wave, or a ragged-batch. 
    self.__call__.get_concrete_function(
        x=tf.TensorSpec(shape=(), dtype=tf.string))
    self.__call__.get_concrete_function(
       x=tf.TensorSpec(shape=[None, 16000], dtype=tf.float32))


  @tf.function
  def __call__(self, x):
    # If they pass a string, load the file and decode it. 
    if x.dtype == tf.string:
      x = tf.io.read_file(x)
      x, _ = tf.audio.decode_wav(x, desired_channels=1, desired_samples=16_000,)
      x = tf.squeeze(x, axis=-1)
      x = x[tf.newaxis, :]
    
    x = get_spectrogram(x)  
    result = self.model(x, training=False)
    
    class_ids = tf.argmax(result, axis=-1)
    class_names = tf.gather(label_names, class_ids)
    return {'predictions':result,
            'class_ids': class_ids,
            'class_names': class_names}

# %% [markdown]
# Test run the "export" model:

# %%
export= ExportModel(model)
#export(tf.constant(str(data_dir/'no/01bb6a2a_nohash_0.wav')))
aL= []
for x in inputWavL:
    
    #x= tf.constant(str(data_dir/f'{x}'))
    x= f'{data_dir}/{x}'
    x= tf.constant(x)
    y= export(x)
    y= y['class_names'].numpy()[0]
    aL += [(x, y)]
aL

# %% [markdown]
# Save and reload the model, the reloaded model gives identical output:

# %%
tf.saved_model.save(export, "saved")
imported= tf.saved_model.load("saved")
#imported(waveform[tf.newaxis, :])
#imported(inputWavL[0])
aL= []
for x in inputWavL:
    
    #x= tf.constant(str(data_dir/f'{x}'))
    x= f'{data_dir}/{x}'
    x= tf.constant(x)
    y= imported(x)
    y= y['class_names'].numpy()[0]
    aL += [(x, y)]
aL

# %% [markdown]
# ## Next steps
# 
# This tutorial demonstrated how to carry out simple audio classification/automatic speech recognition using a convolutional neural network with TensorFlow and Python. To learn more, consider the following resources:
# 
# - The [Sound classification with YAMNet](https://www.tensorflow.org/hub/tutorials/yamnet) tutorial shows how to use transfer learning for audio classification.
# - The notebooks from [Kaggle's TensorFlow speech recognition challenge](https://www.kaggle.com/c/tensorflow-speech-recognition-challenge/overview).
# - The 
# [TensorFlow.js - Audio recognition using transfer learning codelab](https://codelabs.developers.google.com/codelabs/tensorflowjs-audio-codelab/index.html#0) teaches how to build your own interactive web app for audio classification.
# - [A tutorial on deep learning for music information retrieval](https://arxiv.org/abs/1709.04396) (Choi et al., 2017) on arXiv.
# - TensorFlow also has additional support for [audio data preparation and augmentation](https://www.tensorflow.org/io/tutorials/audio) to help with your own audio-based projects.
# - Consider using the [librosa](https://librosa.org/) library for music and audio analysis.


