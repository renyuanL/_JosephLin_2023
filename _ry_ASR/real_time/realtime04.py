#!/usr/bin/env python3
"""Create a recording with arbitrary duration.

The soundfile module (https://PySoundFile.readthedocs.io/) has to be installed!

"""

#%%

import queue
import sys

import sounddevice as sd
import threading


#%%
# 自己設定一些參數
class Arg:
    def __init__(self):
        self.samplerate= 16_000
        self.channels= 1
        self.subtype=  'PCM_16'
        self.filename= 'test.wav'
        self.device= 0

args= Arg()


soundQ=    queue.Queue(maxsize= 1024) 
soundPool= queue.Queue(maxsize= 1024*10)

def Mic2Q(indata, frames, time, status):
    """This is called (from a separate thread) for each audio block."""
    if status:
        print(status, file=sys.stderr)
    
    x= indata.copy()
    
    soundQ.put(x)

def Q2Pool():
    t=0
    while True:
        x= soundQ.get()
        soundPool.put(x)
        
        soundQ.task_done()
        soundPool.task_done()

        if t%10==0 and t>0:
            yL=[]
            for i in range(10):
                y= soundPool.get()
                yL.extend(y)
            
            print(f'{t}.', end='', flush=True)
            print(yL[::10], flush=True)
        t+=1
    return

theMic= sd.InputStream(
    samplerate= args.samplerate,
    blocksize=  args.samplerate // 10, 
    device=     args.device,
    channels=   args.channels, 
    callback=   Mic2Q
    )
    
thePool= threading.Thread(
    target=     Q2Pool, 
    daemon=     True
    )

theMic.start()
thePool.start()

#%%
