# import os
# import ctypes

# dll_path = os.path.join(os.path.dirname(__file__), "opus.dll")
# ctypes.WinDLL(dll_path)  # load the DLL

# print("DLL loaded successfully")

import sounddevice as sd
import numpy as np
from opuslib import Encoder, Decoder

samplerate = 16000
channels = 1
frame_size = 160  # 20 ms

encoder = Encoder(samplerate, channels, 'voip')
encoder.bitrate = 12000  # 12 kbps

decoder = Decoder(samplerate, channels)

def callback(indata, outdata, frames, time, status):
    if status:
        print(status)
    pcm = (indata * 32767).astype(np.int16)
    encoded = encoder.encode(pcm.tobytes(), frame_size)
    decoded = decoder.decode(encoded, frame_size)
    outdata[:] = np.frombuffer(decoded, dtype=np.int16).reshape(-1, channels) / 32768.0

with sd.Stream(samplerate=samplerate, channels=channels, dtype='float32',
               blocksize=frame_size, callback=callback):
    print("Loopback test running at 12 kbps...")
    print("Press Ctrl+C to stop.")
    try:
        while True:
            sd.sleep(1000)
    except KeyboardInterrupt:
        print("\nStopped.")
