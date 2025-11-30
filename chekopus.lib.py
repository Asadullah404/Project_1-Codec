import os
import ctypes

# Add current directory to DLL search path so opuslib can find opus.dll
current_dir = os.path.dirname(os.path.abspath(__file__))
if hasattr(os, 'add_dll_directory'):
    # Python 3.8+ - recommended way
    os.add_dll_directory(current_dir)
else:
    # Python < 3.8 - add to PATH
    os.environ['PATH'] = current_dir + os.pathsep + os.environ.get('PATH', '')

# Also try to preload the DLL explicitly
dll_path = os.path.join(current_dir, "opus.dll")
if os.path.exists(dll_path):
    try:
        ctypes.WinDLL(dll_path)  # load the DLL
        print("DLL loaded successfully")
    except Exception as e:
        print(f"Warning: Could not preload opus.dll: {e}")
else:
    # Try libopus-0.dll as alternative
    alt_dll_path = os.path.join(current_dir, "libopus-0.dll")
    if os.path.exists(alt_dll_path):
        try:
            ctypes.WinDLL(alt_dll_path)
            print("Alternative DLL (libopus-0.dll) loaded successfully")
        except Exception as e:
            print(f"Warning: Could not preload libopus-0.dll: {e}")

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
