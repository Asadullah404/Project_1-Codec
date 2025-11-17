"""
Real-Time Neural Codec for Teleconferencing - ENHANCED WITH LIVE SIMULATION
Latency: ~60-80ms (acceptable for conferencing)
Bitrate: 6-12 kbps (your target range)

NEW FEATURES:
- Live teleconferencing simulation mode
- Real-time chunk-by-chunk processing with delays
- Visual streaming progress
- Per-chunk latency tracking
- Comparison: Full vs Real-Time modes
"""

print("="*80)
print("Real-Time Neural Codec for Teleconferencing (ENHANCED)")
print("="*80)

################################################################################
# STEP 1: INSTALL WITH PYTORCH 2.6+ (FIXED)
################################################################################
print("\n--- Installing dependencies (PyTorch 2.6+) ---")

import subprocess
import sys

# Uninstall old PyTorch AND torchvision
print("Uninstalling old torch, torchaudio, and torchvision...")
subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "torch", "torchaudio", "torchvision"],
               capture_output=True)

# Install PyTorch 2.6+ AND compatible torchvision
print("Installing PyTorch 2.6+, torchaudio, and torchvision...")
subprocess.check_call([
    sys.executable, "-m", "pip", "install", "-q",
    "torch>=2.6.0", "torchaudio>=2.6.0", "torchvision", # <-- ADDED torchvision here
    "--index-url", "https://download.pytorch.org/whl/cu118"
])

# Install transformers and other deps
print("Installing transformers and other dependencies...")
subprocess.check_call([
    sys.executable, "-m", "pip", "install", "-q",
    "transformers>=4.50.0", "accelerate", "safetensors"
])

# Install metrics and visualization
print("Installing metrics and visualization...")
subprocess.check_call([
    sys.executable, "-m", "pip", "install", "-q",
    "pesq", "pystoi", "matplotlib", "IPython"
])

print("✓ All packages installed")

################################################################################
# STEP 2: IMPORTS
################################################################################
print("\n--- Importing libraries ---")
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchaudio
import torchaudio.transforms as T

from transformers import EncodecModel, EncodecFeatureExtractor
from pathlib import Path
import numpy as np
import random
import time
from tqdm import tqdm
import os
import warnings
from datetime import datetime
from collections import deque

warnings.filterwarnings('ignore')

try:
    from pystoi import stoi
    from pesq import pesq
    METRICS_AVAILABLE = True
    print("✓ Loaded metrics (pesq, pystoi)")
except:
    METRICS_AVAILABLE = False
    print("⚠ Could not load metrics (pesq, pystoi). Try `pip install pesq pystoi`")

try:
    import matplotlib.pyplot as plt
    from IPython.display import Audio, display, HTML, clear_output
    VISUALIZATION_AVAILABLE = True
    print("✓ Loaded visualization tools")
except:
    VISUALIZATION_AVAILABLE = False
    print("⚠ Could not load visualization tools")

print(f"\n✓ PyTorch: {torch.__version__}")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✓ Device: {DEVICE}")

################################################################################
# STEP 3: MOUNT DRIVE
################################################################################
print("\n--- Mounting Drive ---")
try:
    from google.colab import drive
    drive.mount('/content/drive', force_remount=True)
    DRIVE_AVAILABLE = True
    print("✓ Drive mounted")
except:
    DRIVE_AVAILABLE = False
    print("⚠ Drive not available. Training (Mode 1) will be disabled.")

################################################################################
# STEP 4: CONFIGURATION
################################################################################
print("\n--- Configuration ---")

DRIVE_FOLDER = "/content/drive/MyDrive/dev"
OUTPUT_DIR = "/content/drive/MyDrive/finetuned_realtime_codec" if DRIVE_AVAILABLE else "/content/finetuned_codec"

# Model configuration
TARGET_SR = 24000
CHUNK_DURATION = 1.5  # 1.5 second chunks for training

# ⚡ CRITICAL FOR REAL-TIME: Small streaming windows
STREAMING_FRAME_MS = 80  # 80ms frames (acceptable latency for conferencing)
STREAMING_FRAME_SAMPLES = int(STREAMING_FRAME_MS * TARGET_SR / 1000)
OVERLAP_MS = 20  # 20ms overlap for smooth transitions
OVERLAP_SAMPLES = int(OVERLAP_MS * TARGET_SR / 1000)

# Training config
NUM_SAMPLES = 300
NUM_EPOCHS = 15
BATCH_SIZE = 4
LEARNING_RATE = 3e-5
GRAD_ACCUM = 2

# Target bitrate
TARGET_BANDWIDTH = 12.0  # 12 kbps (in your 8-15 kbps range)

print(f"Sample rate: {TARGET_SR} Hz")
print(f"Streaming frame: {STREAMING_FRAME_MS}ms")
print(f"Overlap: {OVERLAP_MS}ms")
print(f"Target bitrate: {TARGET_BANDWIDTH} kbps")
print(f"Effective latency: ~{STREAMING_FRAME_MS}ms")

################################################################################
# STEP 5: LOAD ENCODEC MODEL
################################################################################
print("\n--- Loading EnCodec Model ---")

# Load pretrained EnCodec 24kHz
model = EncodecModel.from_pretrained("facebook/encodec_24khz")
feature_extractor = EncodecFeatureExtractor.from_pretrained("facebook/encodec_24khz")

print("✓ Loaded pretrained EnCodec 24kHz")
print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"  Supported bandwidths: {model.config.target_bandwidths}")

model = model.to(DEVICE)

################################################################################
# STEP 6: STREAMING WRAPPER (FIXED VERSION)
################################################################################
print("\n--- Creating Real-Time Streaming Wrapper ---")

class RealTimeEncodecWrapper(nn.Module):
    """
    Wrapper that makes EnCodec work in real-time streaming mode.
    """

    def __init__(self, encodec_model, frame_samples, overlap_samples):
        super().__init__()
        self.encodec = encodec_model
        self.frame_samples = frame_samples
        self.overlap_samples = overlap_samples
        self.hop_samples = frame_samples - overlap_samples
        
        # Crossfade windows for smooth overlap blending
        self.register_buffer(
            'fade_in',
            torch.linspace(0, 1, overlap_samples).reshape(1, 1, -1)
        )
        self.register_buffer(
            'fade_out',
            torch.linspace(1, 0, overlap_samples).reshape(1, 1, -1)
        )
    
    def forward(self, audio, bandwidth=None):
        """
        Standard forward pass for training.
        """
        # Encode
        encoded = self.encodec.encode(audio, bandwidth=bandwidth)
        
        # Decode
        decoded_output = self.encodec.decode(
            encoded.audio_codes,
            encoded.audio_scales
        )
        
        # Extract the actual audio tensor
        decoded = decoded_output.audio_values
        
        # Match lengths
        min_len = min(audio.shape[-1], decoded.shape[-1])
        audio = audio[..., :min_len]
        decoded = decoded[..., :min_len]
        
        return decoded, audio
    
    def streaming_inference(self, audio_stream, bandwidth=None, simulate_realtime=False, verbose=False):
        """
        Real-time streaming inference with overlap-add.
        
        Args:
            audio_stream: [1, 1, samples] - full audio
            bandwidth: target bandwidth in kbps
            simulate_realtime: if True, adds actual time delays to simulate live streaming
            verbose: if True, prints detailed per-chunk stats
            
        Yields:
            dict with:
                - 'audio': output_frame [1, 1, hop_samples]
                - 'latency_ms': processing time for this chunk
                - 'chunk_idx': chunk number
                - 'timestamp': when chunk was processed
        """
        self.eval()
        total_samples = audio_stream.shape[-1]
        
        # State for overlap-add
        overlap_buffer = None
        chunk_idx = 0
        
        # For real-time simulation
        start_time = time.time()
        expected_time = 0  # Expected time if truly real-time
        
        with torch.no_grad():
            for start_idx in range(0, total_samples, self.hop_samples):
                chunk_start = time.time()
                
                # Extract frame with overlap
                end_idx = min(start_idx + self.frame_samples, total_samples)
                frame = audio_stream[..., start_idx:end_idx]
                
                # Pad if needed (last frame)
                if frame.shape[-1] < self.frame_samples:
                    pad_amount = self.frame_samples - frame.shape[-1]
                    frame = F.pad(frame, (0, pad_amount))
                    is_last_frame = True
                else:
                    is_last_frame = False
                
                # Process frame through EnCodec
                encoded = self.encodec.encode(frame, bandwidth=bandwidth)
                decoded_output = self.encodec.decode(
                    encoded.audio_codes,
                    encoded.audio_scales
                )
                
                # Extract audio tensor
                decoded_frame = decoded_output.audio_values
                
                # Match frame size
                if decoded_frame.shape[-1] > self.frame_samples:
                    decoded_frame = decoded_frame[..., :self.frame_samples]
                elif decoded_frame.shape[-1] < self.frame_samples:
                    pad = self.frame_samples - decoded_frame.shape[-1]
                    decoded_frame = F.pad(decoded_frame, (0, pad))
                
                # Overlap-add processing
                if overlap_buffer is not None:
                    overlap_region_current = decoded_frame[..., :self.overlap_samples]
                    blended = (overlap_buffer * self.fade_out + 
                             overlap_region_current * self.fade_in)
                    output = torch.cat([
                        blended,
                        decoded_frame[..., self.overlap_samples:self.hop_samples]
                    ], dim=-1)
                else:
                    output = decoded_frame[..., :self.hop_samples]
                
                # Save overlap buffer for next frame
                if not is_last_frame:
                    overlap_buffer = decoded_frame[..., self.hop_samples:self.hop_samples + self.overlap_samples]
                else:
                    # Handle the very last bit of audio
                    remaining_samples_in_original = end_idx - start_idx
                    if remaining_samples_in_original < self.frame_samples:
                        # We are on the last frame, and it was shorter than a full frame
                        # We only want the valid part of the output
                        
                        # Calculate how many samples from the 'hop' part we need
                        valid_hop_samples = max(0, remaining_samples_in_original - self.overlap_samples)
                        
                        if overlap_buffer is not None:
                             # We still have the blended part
                            valid_blended = blended[..., :self.overlap_samples]
                            valid_hop = decoded_frame[..., self.overlap_samples:self.overlap_samples + valid_hop_samples]
                            output = torch.cat([valid_blended, valid_hop], dim=-1)
                        else:
                            # This is the *only* frame, and it's short
                            output = decoded_frame[..., :remaining_samples_in_original]
                    
                    overlap_buffer = None # No more overlap


                # Calculate latency
                chunk_latency = time.time() - chunk_start
                chunk_latency_ms = chunk_latency * 1000
                
                # Simulate real-time delay
                if simulate_realtime:
                    # Calculate how long this chunk SHOULD take in real-time
                    chunk_duration = self.hop_samples / TARGET_SR  # seconds
                    expected_time += chunk_duration
                    
                    # Calculate how much time has actually passed
                    actual_elapsed = time.time() - start_time
                    
                    # If we're ahead of real-time, wait
                    if actual_elapsed < expected_time:
                        wait_time = expected_time - actual_elapsed
                        if verbose:
                            print(f"  [Chunk {chunk_idx}] Waiting {wait_time*1000:.1f}ms to simulate real-time...")
                        time.sleep(wait_time)
                
                chunk_idx += 1
                
                yield {
                    'audio': output,
                    'latency_ms': chunk_latency_ms,
                    'chunk_idx': chunk_idx - 1,
                    'timestamp': time.time() - start_time,
                    'is_realtime_capable': chunk_latency_ms < (self.hop_samples / TARGET_SR * 1000)
                }
    
    def process_streaming(self, audio_stream, bandwidth=None):
        """
        Process entire audio in streaming mode and return full output.
        """
        chunks = []
        for result in self.streaming_inference(audio_stream, bandwidth, simulate_realtime=False):
            chunks.append(result['audio'].cpu())
        
        return torch.cat(chunks, dim=-1)

# Create wrapper
streaming_model = RealTimeEncodecWrapper(
    model,
    STREAMING_FRAME_SAMPLES,
    OVERLAP_SAMPLES
)
streaming_model = streaming_model.to(DEVICE)

print("✓ Real-time streaming wrapper created")
print(f"  Frame size: {STREAMING_FRAME_MS}ms")
print(f"  Overlap: {OVERLAP_MS}ms")
print(f"  Hop size: {STREAMING_FRAME_MS - OVERLAP_MS}ms")
print(f"  Expected latency: ~{STREAMING_FRAME_MS}ms per frame")

################################################################################
# STEP 7: REAL-TIME TELECONFERENCING SIMULATOR (NEW!)
################################################################################
print("\n--- Creating Teleconferencing Simulator ---")

class TeleconferencingSimulator:
    """
    Simulates a live teleconferencing call with real-time streaming.
    """
    
    def __init__(self, model, sample_rate, frame_samples, hop_samples):
        self.model = model
        self.sample_rate = sample_rate
        self.frame_samples = frame_samples
        self.hop_samples = hop_samples
        self.hop_duration = hop_samples / sample_rate
        
    def simulate_live_call(self, audio, bandwidth, visualize=True):
        """
        Simulate a live teleconferencing call.
        
        Processes audio in real-time with actual delays, as if streaming
        from a microphone to a speaker.
        
        Args:
            audio: [1, 1, samples] - the "microphone input"
            bandwidth: bitrate in kbps
            visualize: show live progress
            
        Returns:
            dict with results and statistics
        """
        print("\n" + "="*80)
        print("🎙️  LIVE TELECONFERENCING SIMULATION")
        print("="*80)
        print(f"Simulating real-time call with {STREAMING_FRAME_MS}ms frames...")
        print(f"(Processing will take the same time as audio duration)")
        print("="*80 + "\n")
        
        audio_duration = audio.shape[-1] / self.sample_rate
        num_chunks = int(np.ceil(audio.shape[-1] / self.hop_samples))
        
        # Statistics tracking
        latencies = []
        chunk_times = []
        output_chunks = []
        
        # Real-time visualization setup
        if visualize and VISUALIZATION_AVAILABLE:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6))
            latency_history = deque(maxlen=50)
        else:
            print("Visualization disabled (matplotlib/IPython not found)")

        print(f"📊 Call info:")
        print(f"   Duration: {audio_duration:.2f}s")
        print(f"   Chunks: {num_chunks}")
        print(f"   Chunk size: {self.hop_duration*1000:.1f}ms")
        print(f"   Bitrate: {bandwidth} kbps\n")
        
        print("▶️  Call started...\n")
        
        # Simulate real-time streaming
        call_start = time.time()
        
        for result in self.model.streaming_inference(
            audio, 
            bandwidth=bandwidth, 
            simulate_realtime=True,  # ← KEY: Adds real-time delays
            verbose=False
        ):
            chunk_idx = result['chunk_idx']
            latency_ms = result['latency_ms']
            timestamp = result['timestamp']
            is_realtime = result['is_realtime_capable']
            
            output_chunks.append(result['audio'].cpu())
            latencies.append(latency_ms)
            chunk_times.append(timestamp)
            
            # Live console output
            status = "✅" if is_realtime else "⚠️"
            print(f"{status} Chunk {chunk_idx+1}/{num_chunks} | "
                  f"Latency: {latency_ms:.1f}ms | "
                  f"Time: {timestamp:.2f}s | "
                  f"{'REAL-TIME' if is_realtime else 'DELAYED'}")
            
            # Live visualization
            if visualize and VISUALIZATION_AVAILABLE:
                latency_history.append(latency_ms)
                
                if (chunk_idx + 1) % 5 == 0 or chunk_idx == num_chunks - 1:
                    clear_output(wait=True)
                    
                    # Plot 1: Latency over time
                    ax1.clear()
                    ax1.plot(latencies, 'b-', linewidth=2, label='Chunk Latency')
                    ax1.axhline(y=self.hop_duration*1000, color='g', linestyle='--', 
                               label=f'Real-time threshold ({self.hop_duration*1000:.1f}ms)')
                    ax1.fill_between(range(len(latencies)), 0, self.hop_duration*1000, 
                                     alpha=0.2, color='green')
                    ax1.set_xlabel('Chunk Number')
                    ax1.set_ylabel('Latency (ms)')
                    ax1.set_title(f'Live Teleconferencing Latency (Chunk {chunk_idx+1}/{num_chunks})')
                    ax1.legend()
                    ax1.grid(True, alpha=0.3)
                    
                    # Plot 2: Waveform progress
                    ax2.clear()
                    reconstructed_so_far = torch.cat(output_chunks, dim=-1).squeeze().numpy()
                    time_axis = np.arange(len(reconstructed_so_far)) / self.sample_rate
                    ax2.plot(time_axis, reconstructed_so_far, 'r-', linewidth=0.5)
                    ax2.axvline(x=timestamp, color='b', linestyle='--', 
                               label=f'Current position: {timestamp:.2f}s')
                    ax2.set_xlabel('Time (s)')
                    ax2.set_ylabel('Amplitude')
                    ax2.set_title('Reconstructed Audio (Real-time Stream)')
                    ax2.legend()
                    ax2.grid(True, alpha=0.3)
                    
                    plt.tight_layout()
                    display(fig)
        
        # Close plot after loop
        if visualize and VISUALIZATION_AVAILABLE:
            plt.close(fig)

        call_duration = time.time() - call_start
        
        print(f"\n▶️  Call ended.\n")
        print("="*80)
        print("📊 CALL STATISTICS")
        print("="*80)
        
        # Concatenate output
        reconstructed = torch.cat(output_chunks, dim=-1)
        
        # Calculate statistics
        avg_latency = np.mean(latencies)
        max_latency = np.max(latencies)
        min_latency = np.min(latencies)
        std_latency = np.std(latencies)
        
        realtime_capable_count = sum(1 for l in latencies if l < self.hop_duration * 1000)
        realtime_percentage = (realtime_capable_count / len(latencies)) * 100
        
        print(f"Total call duration:      {audio_duration:.2f}s")
        print(f"Processing time:          {call_duration:.2f}s")
        print(f"Real-time factor:         {call_duration/audio_duration:.2f}x")
        print(f"Chunks processed:         {len(latencies)}")
        print("-"*80)
        print(f"Average latency:          {avg_latency:.2f}ms")
        print(f"Min latency:              {min_latency:.2f}ms")
        print(f"Max latency:              {max_latency:.2f}ms")
        print(f"Std deviation:            {std_latency:.2f}ms")
        print(f"Target latency:           {self.hop_duration*1000:.2f}ms")
        print("-"*80)
        print(f"Real-time capable:        {realtime_capable_count}/{len(latencies)} chunks ({realtime_percentage:.1f}%)")
        
        if realtime_percentage >= 95:
            print(f"✅ EXCELLENT - Fully real-time capable for teleconferencing!")
        elif realtime_percentage >= 80:
            print(f"✅ GOOD - Suitable for teleconferencing with occasional delays")
        elif realtime_percentage >= 60:
            print(f"⚠️  FAIR - May experience noticeable delays")
        else:
            print(f"❌ POOR - Not suitable for real-time conferencing")
        
        print("="*80)
        
        return {
            'reconstructed': reconstructed,
            'latencies': latencies,
            'avg_latency_ms': avg_latency,
            'max_latency_ms': max_latency,
            'min_latency_ms': min_latency,
            'realtime_percentage': realtime_percentage,
            'call_duration': call_duration,
            'audio_duration': audio_duration
        }

teleconf_simulator = TeleconferencingSimulator(
    streaming_model,
    TARGET_SR,
    STREAMING_FRAME_SAMPLES,
    STREAMING_FRAME_SAMPLES - OVERLAP_SAMPLES
)

print("✓ Teleconferencing simulator ready")

################################################################################
# STEP 7b: DATASET (keeping original)
################################################################################
print("\n--- Initializing Dataset Class ---")
class AudioDataset(Dataset):
    def __init__(self, files, chunk_samples, target_sr):
        self.files = files
        self.chunk_samples = chunk_samples
        self.target_sr = target_sr
        
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        try:
            audio, sr = torchaudio.load(self.files[idx])
            
            if sr != self.target_sr:
                audio = T.Resample(sr, self.target_sr)(audio)
            
            if audio.shape[0] > 1:
                audio = audio.mean(0, keepdim=True)
            
            chunk_samples = int(self.chunk_samples)
            
            if audio.shape[1] >= chunk_samples:
                start = random.randint(0, audio.shape[1] - chunk_samples)
                audio = audio[:, start:start+chunk_samples]
            else:
                audio = F.pad(audio, (0, chunk_samples - audio.shape[1]))
            
            return audio
        except Exception as e:
            # print(f"Warning: Failed to load {self.files[idx]}: {e}")
            return torch.zeros(1, int(self.chunk_samples))

def collate_fn(batch):
    return torch.stack(batch)

print("✓ Dataset class defined")

################################################################################
# STEP 8: MODE SELECTION (ENHANCED)
################################################################################
print("\n" + "="*80)
print("MODE SELECTION")
print("="*80)
print("1 = TRAIN (fine-tune for your audio)")
print("2 = TEST - FULL FILE MODE (process entire file at once)")
print("3 = TEST - LIVE TELECONFERENCING SIMULATION (real-time streaming)")  # ← NEW!
print("="*80)

mode = input("\nEnter 1, 2, or 3: ").strip()

################################################################################
# TRAIN MODE (unchanged)
################################################################################
if mode == '1':
    print("\n" + "="*80)
    print("TRAINING MODE")
    print("="*80)
    
    if not DRIVE_AVAILABLE:
        print("❌ Drive required for training. Please mount Google Drive.")
        import sys
        sys.exit(1)
    
    # Find audio files
    print(f"\nScanning {DRIVE_FOLDER}...")
    audio_files = []
    for ext in ['.wav', '.mp3', '.flac']:
        audio_files.extend(Path(DRIVE_FOLDER).rglob(f"*{ext}"))
    
    print(f"✓ Found {len(audio_files)} audio files")
    
    if len(audio_files) == 0:
        print(f"❌ No audio files in {DRIVE_FOLDER}. Add files to train.")
        import sys
        sys.exit(1)
    
    selected = random.sample(audio_files, min(NUM_SAMPLES, len(audio_files)))
    print(f"✓ Using {len(selected)} files")
    
    # Create dataset
    chunk_samples = int(CHUNK_DURATION * TARGET_SR)
    dataset = AudioDataset([str(f) for f in selected], chunk_samples, TARGET_SR)
    dataloader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True
    )
    
    print(f"✓ Dataset ready: {len(dataset)} samples")
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        streaming_model.parameters(),
        lr=LEARNING_RATE,
        betas=(0.9, 0.999),
        weight_decay=0.01
    )
    
    # Scheduler
    total_steps = len(dataloader) * NUM_EPOCHS
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_steps
    )
    
    # Loss function
    def compute_loss(recon, target):
        """Time domain + simple spectral loss"""
        time_loss = F.l1_loss(recon, target)
        
        # Simple spectral loss
        stft_loss = 0.0
        for n_fft in [2048, 1024]:
            stft = T.Spectrogram(n_fft=n_fft, hop_length=n_fft//4, power=2).to(recon.device)
            
            spec_r = stft(recon.squeeze(1))
            spec_t = stft(target.squeeze(1))
            
            stft_loss += F.l1_loss(
                torch.log(spec_r + 1e-5),
                torch.log(spec_t + 1e-5)
            )
        
        return time_loss + 0.3 * stft_loss
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("\n" + "="*80)
    print("STARTING TRAINING")
    print("="*80)
    
    streaming_model.train()
    global_step = 0
    
    for epoch in range(NUM_EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{NUM_EPOCHS} ---")
        
        epoch_loss = 0.0
        progress = tqdm(dataloader, desc=f"Epoch {epoch+1}")
        
        for batch_idx, audio in enumerate(progress):
            audio = audio.to(DEVICE)
            
            # Forward pass
            recon, target = streaming_model(audio, bandwidth=TARGET_BANDWIDTH)
            
            # Compute loss
            loss = compute_loss(recon, target)
            
            if torch.isnan(loss):
                print("\n⚠ NaN loss, skipping batch")
                continue
            
            # Backward
            loss = loss / GRAD_ACCUM
            loss.backward()
            
            if (batch_idx + 1) % GRAD_ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(streaming_model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
            
            epoch_loss += loss.item() * GRAD_ACCUM
            
            progress.set_postfix({
                'loss': f'{loss.item() * GRAD_ACCUM:.4f}',
                'lr': f'{scheduler.get_last_lr()[0]:.2e}'
            })
        
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1} - Avg Loss: {avg_loss:.4f}")
        
        # Save checkpoint
        if (epoch + 1) % 5 == 0:
            checkpoint = f"{OUTPUT_DIR}/checkpoint_epoch_{epoch+1}.pt"
            torch.save(streaming_model.state_dict(), checkpoint)
            print(f"✓ Saved: {checkpoint}")
    
    # Save final model
    final_path = f"{OUTPUT_DIR}/final_realtime_codec.pt"
    torch.save(streaming_model.state_dict(), final_path)
    print(f"\n✓ Training complete! Saved to: {final_path}")

################################################################################
# TEST MODE 2: FULL FILE (original test mode)
################################################################################
elif mode == '2':
    print("\n" + "="*80)
    print("TEST MODE: FULL FILE PROCESSING")
    print("="*80)
    
    # Try to load fine-tuned model
    model_path = f"{OUTPUT_DIR}/final_realtime_codec.pt"
    
    if os.path.exists(model_path):
        print(f"Loading fine-tuned model from: {model_path}")
        streaming_model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        print("✓ Fine-tuned model loaded")
    else:
        print("⚠ No fine-tuned model found, using pretrained EnCodec")
    
    streaming_model.eval()
    
    # Upload test file
    print("\n--- Upload Test Audio File ---")
    try:
        from google.colab import files
        uploaded = files.upload()
        
        if not uploaded:
            print("❌ No file uploaded")
            import sys
            sys.exit(0)
        
        input_file = list(uploaded.keys())[0]
        print(f"✓ Uploaded: {input_file}")

    except:
        print("❌ Failed to upload file (files.upload() only works in Colab)")
        import sys
        sys.exit(0)
    
    # Load and preprocess
    print("\n--- Loading Audio ---")
    audio, sr = torchaudio.load(input_file)
    print(f"Original: {sr} Hz, {audio.shape}")
    
    if sr != TARGET_SR:
        print(f"Resampling to {TARGET_SR} Hz...")
        audio = T.Resample(sr, TARGET_SR)(audio)
    
    if audio.shape[0] > 1:
        print("Converting to mono...")
        audio = audio.mean(0, keepdim=True)
    
    audio = audio.unsqueeze(0).to(DEVICE)
    duration = audio.shape[-1] / TARGET_SR
    
    print(f"✓ Preprocessed: {audio.shape}")
    print(f"  Duration: {duration:.2f} seconds")
    
    # Full file processing
    print(f"\n--- Full File Processing (Non Real-Time) ---")
    
    start_time = time.time()
    reconstructed = streaming_model.process_streaming(audio, bandwidth=TARGET_BANDWIDTH)
    total_latency = time.time() - start_time
    
    num_frames = int(np.ceil(audio.shape[-1] / (STREAMING_FRAME_SAMPLES - OVERLAP_SAMPLES)))
    latency_per_frame = total_latency / num_frames
    
    print(f"\n✓ Processing complete!")
    print(f"  Total latency: {total_latency:.4f}s")
    print(f"  Audio duration: {duration:.2f}s")
    print(f"  Real-time factor: {total_latency/duration:.2f}x")
    print(f"  Avg frame latency: {latency_per_frame*1000:.1f}ms")
    
    # Quality metrics
    if METRICS_AVAILABLE:
        print("\n--- Quality Metrics ---")
        
        orig_np = audio.squeeze().cpu().numpy()
        recon_np = reconstructed.squeeze().numpy()
        
        min_len = min(len(orig_np), len(recon_np))
        orig_np = orig_np[:min_len]
        recon_np = recon_np[:min_len]
        
        max_val = max(np.abs(orig_np).max(), np.abs(recon_np).max())
        if max_val > 0:
            orig_np = orig_np / max_val * 0.95
            recon_np = recon_np / max_val * 0.95
        
        try:
            resampler_16k = T.Resample(TARGET_SR, 16000)
            orig_16k = resampler_16k(torch.from_numpy(orig_np)).numpy()
            recon_16k = resampler_16k(torch.from_numpy(recon_np)).numpy()
            pesq_score = pesq(16000, orig_16k, recon_16k, 'wb')
            print(f"  PESQ: {pesq_score:.4f}")
        except Exception as e:
            pesq_score = 0.0
            print(f"  PESQ: Failed ({e})")
        
        try:
            resampler_10k = T.Resample(TARGET_SR, 10000)
            orig_10k = resampler_10k(torch.from_numpy(orig_np)).numpy()
            recon_10k = resampler_10k(torch.from_numpy(recon_np)).numpy()
            stoi_score = stoi(orig_10k, recon_10k, 10000)
            print(f"  STOI: {stoi_score:.4f}")
        except Exception as e:
            stoi_score = 0.0
            print(f"  STOI: Failed ({e})")

    else:
        print("\n--- Quality Metrics (Skipped) ---")
        print("Install `pesq` and `pystoi` for metrics.")
    
    # Save output
    output_file = f"full_output_{TARGET_BANDWIDTH}kbps.wav"
    output_audio = reconstructed.squeeze(0)
    if output_audio.dim() == 1:
        output_audio = output_audio.unsqueeze(0)
    
    torchaudio.save(output_file, output_audio.cpu(), TARGET_SR)
    print(f"\n✓ Saved: {output_file}")
    
    try:
        files.download(output_file)
    except:
        pass # Fails if not in Colab

################################################################################
# TEST MODE 3: LIVE TELECONFERENCING SIMULATION (NEW!)
################################################################################
elif mode == '3':
    print("\n" + "="*80)
    print("TEST MODE: LIVE TELECONFERENCING SIMULATION")
    print("="*80)
    
    # Try to load fine-tuned model
    model_path = f"{OUTPUT_DIR}/final_realtime_codec.pt"
    
    if os.path.exists(model_path):
        print(f"Loading fine-tuned model from: {model_path}")
        streaming_model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        print("✓ Fine-tuned model loaded")
    else:
        print("⚠ No fine-tuned model found, using pretrained EnCodec")
    
    streaming_model.eval()
    
    # Upload test file
    print("\n--- Upload Test Audio File ---")
    try:
        from google.colab import files
        uploaded = files.upload()
        
        if not uploaded:
            print("❌ No file uploaded")
            import sys
            sys.exit(0)
        
        input_file = list(uploaded.keys())[0]
        print(f"✓ Uploaded: {input_file}")

    except:
        print("❌ Failed to upload file (files.upload() only works in Colab)")
        import sys
        sys.exit(0)
    
    # Load and preprocess
    print("\n--- Loading Audio ---")
    audio, sr = torchaudio.load(input_file)
    print(f"Original: {sr} Hz, {audio.shape}")
    
    if sr != TARGET_SR:
        print(f"Resampling to {TARGET_SR} Hz...")
        audio = T.Resample(sr, TARGET_SR)(audio)
    
    if audio.shape[0] > 1:
        print("Converting to mono...")
        audio = audio.mean(0, keepdim=True)
    
    audio_full = audio.to(DEVICE).unsqueeze(0) # Keep a CPU copy for playback
    duration = audio_full.shape[-1] / TARGET_SR
    
    print(f"✓ Preprocessed: {audio_full.shape}")
    print(f"  Duration: {duration:.2f} seconds")
    
    # Run live simulation
    results = teleconf_simulator.simulate_live_call(
        audio_full,
        bandwidth=TARGET_BANDWIDTH,
        visualize=VISUALIZATION_AVAILABLE
    )
    
    reconstructed = results['reconstructed']
    
    # Quality metrics
    if METRICS_AVAILABLE:
        print("\n--- Quality Metrics ---")
        
        orig_np = audio_full.squeeze().cpu().numpy()
        recon_np = reconstructed.squeeze().numpy()
        
        min_len = min(len(orig_np), len(recon_np))
        orig_np = orig_np[:min_len]
        recon_np = recon_np[:min_len]
        
        max_val = max(np.abs(orig_np).max(), np.abs(recon_np).max())
        if max_val > 0:
            orig_np = orig_np / max_val * 0.95
            recon_np = recon_np / max_val * 0.95
        
        try:
            resampler_16k = T.Resample(TARGET_SR, 16000)
            orig_16k = resampler_16k(torch.from_numpy(orig_np)).numpy()
            recon_16k = resampler_16k(torch.from_numpy(recon_np)).numpy()
            pesq_score = pesq(16000, orig_16k, recon_16k, 'wb')
            print(f"  PESQ: {pesq_score:.4f} (>3.0 = good)")
        except Exception as e:
            pesq_score = 0.0
            print(f"  PESQ: Failed ({e})")
        
        try:
            resampler_10k = T.Resample(TARGET_SR, 10000)
            orig_10k = resampler_10k(torch.from_numpy(orig_np)).numpy()
            recon_10k = resampler_10k(torch.from_numpy(recon_np)).numpy()
            stoi_score = stoi(orig_10k, recon_10k, 10000)
            print(f"  STOI: {stoi_score:.4f} (>0.85 = good)")
        except Exception as e:
            stoi_score = 0.0
            print(f"  STOI: Failed ({e})")
            
    else:
        print("\n--- Quality Metrics (Skipped) ---")
        print("Install `pesq` and `pystoi` for metrics.")

    
    # Save output
    print("\n--- Saving Output ---")
    output_file = f"live_teleconf_output_{TARGET_BANDWIDTH}kbps.wav"
    output_audio = reconstructed.squeeze(0)
    if output_audio.dim() == 1:
        output_audio = output_audio.unsqueeze(0)
    
    torchaudio.save(output_file, output_audio.cpu(), TARGET_SR)
    print(f"✓ Saved: {output_file}")
    
    try:
        files.download(output_file)
    except:
        pass # Fails if not in Colab
    
    # Play audio in notebook
    if VISUALIZATION_AVAILABLE:
        print("\n--- Audio Playback ---")
        print("Original:")
        display(Audio(audio_full.squeeze().cpu().numpy(), rate=TARGET_SR))
        print("\nReconstructed (after live simulation):")
        display(Audio(reconstructed.squeeze().numpy(), rate=TARGET_SR))
    
    # Final summary
    print("\n" + "="*80)
    print("📊 TELECONFERENCING SIMULATION SUMMARY")
    print("="*80)
    print(f"✅ Mode:               Live Real-Time Simulation")
    print(f"✅ Bitrate:            {TARGET_BANDWIDTH} kbps")
    print(f"✅ Frame size:         {STREAMING_FRAME_MS}ms")
    print(f"✅ Avg latency:        {results['avg_latency_ms']:.2f}ms")
    print(f"✅ Real-time capable:  {results['realtime_percentage']:.1f}% of chunks")
    if METRICS_AVAILABLE:
        print(f"✅ PESQ:               {pesq_score:.4f}")
        print(f"✅ STOI:               {stoi_score:.4f}")
    print("="*80)

else:
    print(f"\n❌ Invalid mode '{mode}'. Please enter 1, 2, or 3.")


print("\n🎉 Script complete!")