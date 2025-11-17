"""
Core Real-Time Neural Codec Model
- Contains the RealTimeEncodecWrapper
- Contains the model loading logic
- This file is intended to be imported by other applications.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import EncodecModel
import time
import os
import warnings

warnings.filterwarnings('ignore')

# --- Configuration ---
TARGET_SR = 24000
# ⚡ CRITICAL FOR REAL-TIME: Small streaming windows
STREAMING_FRAME_MS = 80  # 80ms frames (acceptable latency for conferencing)
STREAMING_FRAME_SAMPLES = int(STREAMING_FRAME_MS * TARGET_SR / 1000)
OVERLAP_MS = 20  # 20ms overlap for smooth transitions
OVERLAP_SAMPLES = int(OVERLAP_MS * TARGET_SR / 1000)
TARGET_BANDWIDTH = 12.0  # 12 kbps

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Codec Config: SR={TARGET_SR}, Frame={STREAMING_FRAME_MS}ms, Overlap={OVERLAP_MS}ms")


class RealTimeEncodecWrapper(nn.Module):
    """
    Wrapper that makes EnCodec work in real-time streaming mode.
    This is the core "engine" for your application.
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
    
    @torch.no_grad()
    def streaming_inference(self, audio_stream, bandwidth=None, simulate_realtime=False, verbose=False):
        """
        Real-time streaming inference with overlap-add.
        
        Args:
            audio_stream: [1, 1, samples] - full audio
            bandwidth: target bandwidth in kbps
            simulate_realtime: if True, adds actual time delays (FOR TESTING ONLY)
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
                    valid_hop_samples = max(0, remaining_samples_in_original - self.overlap_samples)
                    
                    if overlap_buffer is not None:
                        valid_blended = blended[..., :self.overlap_samples]
                        valid_hop = decoded_frame[..., self.overlap_samples:self.overlap_samples + valid_hop_samples]
                        output = torch.cat([valid_blended, valid_hop], dim=-1)
                    else:
                        output = decoded_frame[..., :remaining_samples_in_original]
                
                overlap_buffer = None # No more overlap

            # Calculate latency
            chunk_latency = time.time() - chunk_start
            chunk_latency_ms = chunk_latency * 1000
            
            # Simulate real-time delay (FOR TESTING)
            if simulate_realtime:
                chunk_duration = self.hop_samples / TARGET_SR  # seconds
                expected_time += chunk_duration
                actual_elapsed = time.time() - start_time
                
                if actual_elapsed < expected_time:
                    wait_time = expected_time - actual_elapsed
                    if verbose:
                        print(f"  [Chunk {chunk_idx}] Waiting {wait_time*1000:.1f}ms...")
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
        Helper function to process entire audio in streaming mode and return full output.
        """
        chunks = []
        for result in self.streaming_inference(audio_stream, bandwidth, simulate_realtime=False):
            chunks.append(result['audio'].cpu())
        
        return torch.cat(chunks, dim=-1)


def load_codec_model(custom_model_path=None):
    """
    Loads the EnCodec model and wraps it.
    
    Args:
        custom_model_path (str, optional): Path to a fine-tuned .pt checkpoint.
                                          If None, loads the pretrained model.
    
    Returns:
        RealTimeEncodecWrapper: The wrapped, ready-to-use model.
    """
    print("\n--- Loading EnCodec Model ---")
    
    # Load pretrained EnCodec 24kHz
    base_model = EncodecModel.from_pretrained("facebook/encodec_24khz")
    print("✓ Loaded pretrained EnCodec 24kHz")
    
    # Create wrapper
    streaming_model = RealTimeEncodecWrapper(
        base_model,
        STREAMING_FRAME_SAMPLES,
        OVERLAP_SAMPLES
    )
    
    # Load fine-tuned weights if provided
    if custom_model_path and os.path.exists(custom_model_path):
        print(f"Loading fine-tuned weights from: {custom_model_path}")
        try:
            streaming_model.load_state_dict(torch.load(custom_model_path, map_location=DEVICE))
            print("✓ Fine-tuned model loaded")
        except Exception as e:
            print(f"⚠ Warning: Failed to load custom weights: {e}. Using pretrained model.")
    elif custom_model_path:
        print(f"⚠ Warning: Custom model path not found: {custom_model_path}. Using pretrained model.")
    else:
        print("✓ Using pretrained model (no fine-tuned path provided)")
        
    streaming_model = streaming_model.to(DEVICE)
    streaming_model.eval()
    
    print("✓ Real-time streaming wrapper created")
    print(f"  Frame size: {STREAMING_FRAME_MS}ms")
    print(f"  Overlap: {OVERLAP_MS}ms")
    print(f"  Hop size: {STREAMING_FRAME_MS - OVERLAP_MS}ms")
    
    return streaming_model

if __name__ == '__main__':
    print("\n--- Codec Model Module Test ---")
    print("This file is a module and is meant to be imported.")
    print("Testing model loading...")
    
    model = load_codec_model()
    
    print("\nTesting inference with random noise...")
    # Create 2 seconds of noise
    test_audio = torch.randn(1, 1, TARGET_SR * 2).to(DEVICE)
    print(f"Input shape: {test_audio.shape}")
    
    # Process in streaming mode
    output_audio = model.process_streaming(test_audio, bandwidth=TARGET_BANDWIDTH)
    print(f"Output shape: {output_audio.shape}")
    
    print("\n✓ Module test complete.")