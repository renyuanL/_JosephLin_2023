#%%
#!/usr/bin/env python3
"""Plot the live microphone signal(s) with matplotlib.

Matplotlib and NumPy have to be installed.

"""
import argparse
import queue
import sys

from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import numpy as np

import sounddevice as sd # pip install sounddevice


def int_or_str(text):
    """Helper function for argument parsing."""
    try:
        return int(text)
    except ValueError:
        return text


parser = argparse.ArgumentParser(add_help=False)
parser.add_argument(
    '-l', '--list-devices', action='store_true',
    help='show list of audio devices and exit')
args, remaining = parser.parse_known_args()
if args.list_devices:
    print(sd.query_devices())
    parser.exit(0)
parser = argparse.ArgumentParser(
    description=__doc__,
    formatter_class=argparse.RawDescriptionHelpFormatter,
    parents=[parser])
parser.add_argument(
    'channels', type=int, default=[1], nargs='*', metavar='CHANNEL',
    help='input channels to plot (default: the first)')
parser.add_argument(
    '-d', '--device', type=int_or_str,
    help='input device (numeric ID or substring)')
parser.add_argument(
    '-w', '--window', type=float, default=1_000, metavar='DURATION',
    help='visible time slot (default: %(default)s ms)')
parser.add_argument(
    '-i', '--interval', type=float, default=30,
    help='minimum time between plot updates (default: %(default)s ms)')
parser.add_argument(
    '-b', '--blocksize', type=int, help='block size (in samples)')
parser.add_argument(
    '-r', '--samplerate', type=float, default= 16_000 ,help='sampling rate of audio device')
parser.add_argument(
    '-n', '--downsample', type=int, default= 1, metavar='N',
    help='display every Nth sample (default: %(default)s)')
#%%
#args = parser.parse_args(remaining)
args, remaining = parser.parse_known_args(remaining)

#%%
if any(c < 1 for c in args.channels):
    parser.error('argument CHANNEL: must be >= 1')
mapping = [c - 1 for c in args.channels]  # Channel numbers start with 1


#from ryModels import get_likely_index 

#fn_wav= fn_wav_00= 'ry35words.wav'

import torch
import torchaudio
import whisper
import whisper as wp

device= ('cuda' if torch.cuda.is_available() else 
         'cpu')

device= 'cpu'

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


#x=  get_wav()
#y=  md.transcribe(x)
#print(f'{y= }')




theQ= q= queue.Queue(maxsize= 16_000)
theResult= None


def audio_callback(indata, frames, time, status):
    """This is called (from a separate thread) for each audio block."""
    global theResult

    if status:
        print(status, file=sys.stderr)
    # Fancy indexing with mapping creates a (necessary!) copy:
    
    x= indata[::args.downsample, mapping]

    q.put(x)
    
    # TODO: speech recognition
    # theResult= speech_recognition(x)

    #xx= q.get()

    #theResult= md.transcribe(xx)



def update_plot(frame):
    """This is called by matplotlib for each plot update.

    Typically, audio callbacks happen more frequently than plot updates,
    therefore the queue tends to contain multiple blocks of audio data.

    """
    global plotdata
    while True:
        try:
            data = q.get_nowait()
        except queue.Empty:
            break
        shift = len(data)
        plotdata = np.roll(plotdata, -shift, axis=0)
        plotdata[-shift:, :] = data
    for column, line in enumerate(lines):
        
        xx= plotdata[:, column]

        #xx2= xx[::10]
        #line.set_ydata(xx2)
        
        xx= torch.from_numpy(xx).float().to(device)

        #
        # I should put longer audio into the model
        #

        theResult= md.transcribe(xx)

        print(f'{xx.shape= }, {theResult["text"]= }')


    return lines


try:
    if args.samplerate is None:
        device_info = sd.query_devices(args.device, 'input')
        args.samplerate = device_info['default_samplerate']

    length = int(args.window * args.samplerate / (1000 * args.downsample))
    plotdata = np.zeros((length, len(args.channels)))

    fig, ax = plt.subplots()
    lines = ax.plot(plotdata)
    if len(args.channels) > 1:
        ax.legend([f'channel {c}' for c in args.channels],
                  loc='lower left', ncol=len(args.channels))
    ax.axis((0, len(plotdata), -1, 1))
    ax.set_yticks([0])
    ax.yaxis.grid(True)
    ax.tick_params(bottom=False, top=False, labelbottom=False,
                   right=False, left=False, labelleft=False)
    fig.tight_layout(pad=0)

    
    stream= sd.InputStream(
        device= args.device, 
        channels= max(args.channels),
        samplerate= args.samplerate, # 16_000

        callback= audio_callback # a "callback" function
        
        )
    

    ani= FuncAnimation(
        fig, 
        update_plot, # a "callback" function
        interval= args.interval, 
        blit=     True)
    
    print('before plot.show()..')

    with stream:
        plt.show()

        '''
        for i in range(1000):
            xx= q.get()
            
            #theResult= md.transcribe(xx)
            theResult= xx.flatten()
            maxResult= theResult.max(axis=-1)

            print(f'{theResult.shape= }, {maxResult= }')
        '''

except Exception as e:
    parser.exit(type(e).__name__ + ': ' + str(e))

print('Bye....')
# %%
