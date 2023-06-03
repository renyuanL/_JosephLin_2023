#%%
import sounddevice as sd
import numpy as np
import queue
import sys

#%%
fs= 16_000
x1= np.random.randn(fs*1)
t=  np.linspace(0, 1, fs*1)
x2= np.sin(2*np.pi*440*t)

# %%
sd.default.samplerate= fs
sd.default.channels=   1

sd.play(x1)
sd.wait()

sd.play(x2)
sd.wait()
# %%

duration= 10  # seconds
ryRec= sd.rec(int(duration * fs), 
              #samplerate=fs, 
              #channels=1
              )
sd.wait()  # Wait until recording is finished
#sd.play(ryRec)
#sd.wait()
# %%

duration= 10  # seconds
#sd.wait()  # Wait until recording is finished
ryRec1= sd.playrec(ryRec)
sd.wait()
# %%
sd.play(ryRec1+ryRec)
sd.wait()
#%%
sd.query_devices()
# %%
sd.get_status()
# %%
sd.get_stream()
# %%
import sounddevice as sd
duration= 10  # seconds

def callback(indata, outdata, frames, time, status):
    if status:
        print(status)
    outdata[:]= indata

with sd.Stream(
    callback= callback
    ) as sdStrm:
    sd.sleep(int(duration * 1000))
# %%
'''
Play a Very Long Sound File
play_long_file.py
https://python-sounddevice.readthedocs.io/en/0.4.6/examples.html
'''
import argparse
import queue
import sys
import threading

import sounddevice as sd
import soundfile as sf

class Args:
    def __init__(self):
        self.blocksize=  1024
        self.buffersize=   10
        self.filename= 'ry35words.wav'
        self.channels= 1
        self.samplerate= 16_000
        self.device= 0
args= Args()

q=     queue.Queue(maxsize= args.buffersize)
event= threading.Event()

def callback(outdata, frames, time, status):
    assert frames == args.blocksize
    if status.output_underflow:
        print('Output underflow: increase blocksize?', file=sys.stderr)
        raise sd.CallbackAbort
    assert not status
    try:
        data = q.get_nowait()
    except queue.Empty as e:
        print('Buffer is empty: increase buffersize?', file=sys.stderr)
        raise sd.CallbackAbort from e
    if len(data) < len(outdata):
        outdata[:len(data)] = data
        outdata[len(data):].fill(0)
        raise sd.CallbackStop
    else:
        outdata[:] = data

with sf.SoundFile(args.filename) as f:

    for _ in range(args.buffersize):
        data = f.read(args.blocksize)
        if not len(data):
            break
        q.put_nowait(data)  # Pre-fill queue
    
    stream= sd.OutputStream(
        samplerate= f.samplerate, 
        blocksize=  args.blocksize,
        device=     args.device, 
        channels=   f.channels,
        callback= callback, 
        finished_callback= event.set)
    
    with stream:
        timeout = args.blocksize * args.buffersize / f.samplerate
        while len(data):
            data= f.read(args.blocksize)
            q.put(data, timeout= timeout)


# %%
