#!/usr/bin/env python3
"""Create a recording with arbitrary duration.

The soundfile module (https://PySoundFile.readthedocs.io/) has to be installed!

"""
#%%
import argparse
import tempfile
import queue
import sys

import sounddevice as sd
import soundfile as sf
import numpy  # Make sure NumPy is loaded before it is used in the callback
assert numpy  # avoid "imported but unused" message (W0611)

import time
import threading


def int_or_str(text):
    """Helper function for argument parsing."""
    try:
        return int(text)
    except ValueError:
        return text

#%%
parser= argparse.ArgumentParser(add_help=False)

parser.add_argument(
    '-l', '--list-devices', action='store_true',
    help='show list of audio devices and exit')

args, remaining= parser.parse_known_args()

if args.list_devices:
    print(sd.query_devices())
    parser.exit(0)

parser= argparse.ArgumentParser(
    description=__doc__,
    formatter_class=argparse.RawDescriptionHelpFormatter,
    parents=[parser])
parser.add_argument(
    'filename', nargs='?', metavar='FILENAME',
    help='audio file to store recording to')
parser.add_argument(
    '-d', '--device', type=int_or_str,
    help='input device (numeric ID or substring)')
parser.add_argument(
    '-r', '--samplerate', type=int, help='sampling rate')
parser.add_argument(
    '-c', '--channels', type=int, default=1, help='number of input channels')
parser.add_argument(
    '-t', '--subtype', type=str, help='sound file subtype (e.g. "PCM_24")')

#args = parser.parse_args(remaining) # 這裡會出錯，因為 remaining 是空的

# 自己設定一些參數
args.samplerate= 16_000
args.channels= 1
args.subtype=  'PCM_16'
args.filename= 'test.wav'
args.device= 0


#%%

soundQ=    queue.Queue(maxsize= 1024) 
soundPool= queue.Queue(maxsize= 1024*10)

# 這個 queue 是用來存放錄音的資料的，
# 最多存放 1024 個
# 這個數字可以自己設定，但是不要設太大，不然會爆記憶體
# 這個 queue 的資料型態是 numpy.ndarray
# 1024 個 ndarray 會佔用多少記憶體呢？
# 這個 ndarray 的 shape 是 (16000, 1)，
# 也就是 16000 個 float32，每個 float32 佔 4 bytes，
# 所以 1024 個 ndarray 佔用的記憶體是 
# 1024*16000*4 bytes = 64_000_000 bytes = 64 MB

def Mic2Q(indata, frames, time, status):
    """This is called (from a separate thread) for each audio block."""
    if status:
        print(status, file=sys.stderr)
    
    x= indata.copy()
    
    soundQ.put(x)

#%%
# using a thread to save the data
import threading

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



#%%
# 這裡開始錄音

try:
    if args.samplerate is None:
        device_info= sd.query_devices(args.device, 'input')
        # soundfile expects an int, sounddevice provides a float:
        args.samplerate= int(device_info['default_samplerate'])
    
    if args.filename is None:
        args.filename= tempfile.mktemp(prefix='delme_rec_unlimited_',
                                       suffix='.wav', dir='')


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
    
    #move2Pool()

except KeyboardInterrupt:
    print('\nRecording finished: ' + repr(args.filename))
    parser.exit(0)

except Exception as e:
    parser.exit(type(e).__name__ + ': ' + str(e))
# %%
# %%
