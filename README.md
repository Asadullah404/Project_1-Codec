Ultra Low-Latency Audio Codec Suite



This application is a complete suite for testing, evaluating, and deploying neural and traditional audio codecs. It features a custom-trained Tiny Transformer model capable of real-time, low-latency audio streaming.



The suite is built with PyQt5 and PyTorch and provides a user-friendly interface for real-time streaming, objective model evaluation, and detailed comparative analysis.



Core Features



Automatic Setup: On first run, the application automatically:



Installs all required Python packages (PyQt5, PyTorch, Librosa, etc.) using pip.



Downloads the necessary pre-trained model files (.pt and .pth) from Google Drive using gdown.



Modular Tab-Based Interface:



Real-Time Streaming



Model Evaluation



Detailed Comparative Analysis



1\. Real-Time Streaming



This tab allows for a full-duplex, low-latency audio stream between two peers on a local network.



Peer Discovery: Automatically discovers other users on the network running the app.



Stream Handshake: Before streaming begins, the initiator must send a request, which the receiver can Accept or Decline via a pop-up.



Codec Support: Stream using:



Uncompressed (256 kbps)



μ-Law \& A-Law (128 kbps)



Simulated AMR-WB (~12.65 kbps)



Custom Tiny Transformer Codec (~9.33 kbps)



DAC (Descript Audio Codec)



Controls: Includes mute, file playback (streaming an audio file instead of the mic), and a real-time log.



2\. Model Evaluation



This tab allows you to run a single audio file through a selected codec and get objective quality metrics.



Streaming-Mode Simulation: Accurately simulates the real-time Overlap-and-Save (OaS) streaming logic to provide metrics that match real-world performance.



Metrics: Calculates:



PESQ (wb): Perceptual Evaluation of Speech Quality



STOI: Short-Time Objective Intelligibility



RTF: Real-Time Factor (to verify the codec is faster than real-time).



Visual \& Audio Feedback:



Displays spectrograms of the original and reconstructed audio.



Allows for playback of both original and reconstructed audio.



Provides a button to download the reconstructed audio as a .wav file.



FFmpeg Integration: Includes support for file-based codecs like Opus and AAC for comparison.



3\. Detailed Comparative Analysis



This tab batch-processes a single audio file through all available codecs and presents the results in a unified dashboard.



Comparative Table: Displays all key metrics (PESQ, STOI, RTF, Bitrate, Latency) in a sortable table.



Data Visualization: Generates bar charts to visually compare codec performance on quality (PESQ/STOI) and speed (RTF).



Data Export: Allows you to export the complete results table to a .txt file.



Getting Started



Ensure Python is installed: This application requires Python 3.8 or newer.



Run the App:



python app.py





Wait for Setup: On the first launch, the script will automatically install all required packages and download the three pre-trained model files. This may take a few minutes.



Ready: Once setup is complete, the application window will open.



Training Your Own Model



This project includes training script.py (designed for Google Colab) to train your own version of the TinyTransformerCodec.



The script:



Installs all training dependencies.



Mounts Google Drive to save and load checkpoints.



Downloads and prepares the LibriTTS dataset.



Trains the model using a Multi-Resolution STFT Loss.



Validates using the exact same OaS streaming logic as the main application to ensure metrics are accurate.



The final trained model (tiny\_transformer\_best.pt) is the exact file used by the main application.

