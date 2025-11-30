# # import torch
# # import torch.nn as nn
# # import numpy as np
# # import torch.nn.functional as F
# # import os
# # import struct
# # import subprocess
# # import sys
# # import math

# # # --- START: New Imports ---
# # import torchaudio
# # import torchaudio.transforms as T
# # from codec_model import (
# #     load_codec_model as load_encodec_model_24k,
# #     RealTimeEncodecWrapper as EncodecStreamingWrapper,
# #     TARGET_SR as ENCODEC_SR_24K,
# #     TARGET_BANDWIDTH as ENCODEC_BANDWIDTH,
# #     STREAMING_FRAME_SAMPLES as ENCODEC_FRAME_SAMPLES,
# #     OVERLAP_SAMPLES as ENCODEC_OVERLAP_SAMPLES
# # )
# # # --- END: New Imports ---

# # # --- START: OpusLib Import ---
# # try:
# #     import opuslib
# #     OPUSLIB_AVAILABLE = True
# # except (ImportError, OSError) as e:
# #     OPUSLIB_AVAILABLE = False
# #     print(f"WARNING: opuslib not installed or libopus not found (Error: {e}). Opus (opuslib) codec will be unavailable.")
# # # --- END: OpusLib Import ---


# # # --- Global Configuration (Synchronized with ~9.3kbps, 15ms LATENCY Trainer) ---
# # SR = 16000
# # CHANNELS = 1
# # LATENT_DIM = 64  
# # BLOCKS = 4
# # HEADS = 8        
# # KERNEL_SIZE = 3

# # # Synchronized: 2 * 2 * 2 * 3 = 24x downsampling (Unchanged)
# # STRIDES = [2, 2, 2, 3] 
# # DOWN_FACTOR = np.prod(STRIDES) # 24

# # # Synchronized: 2 Codebooks (128 entries each)
# # NUM_CODEBOOKS = 2
# # CODEBOOK_SIZE = 128 
# # COMMITMENT_COST = 0.25  
# # TRANSFORMER_BLOCKS = 4  

# # # --- **** CRITICAL CHANGE **** ---
# # # HOP size changed to 20ms (320 samples) to support Opus
# # # This is now the app's fundamental chunk size.
# # HOP_SIZE = int(SR / (1000 / 20)) # 320 samples (20ms)
# # # --- **** END CRITICAL CHANGE **** ---

# # # Models cache directory
# # MODELS_CACHE_DIR = os.path.join(os.path.expanduser("~"), ".audio_codec_models")
# # if not os.path.exists(MODELS_CACHE_DIR):
# #     os.makedirs(MODELS_CACHE_DIR)

# # # --- DAC INTEGRATION (Unchanged) ---
# # try:
# #     import dac
# #     DAC_AVAILABLE = True
# # except ImportError:
# #     DAC_AVAILABLE = False


# # # --- START: New OpusLib Streaming Codec ---
# # class OpusLibStreamingCodec:
# #     """
# #     A stateless streaming Opus codec using 'opuslib'.
# #     It's designed to process 20ms (320-sample) chunks,
# #     matching the application's streaming buffer.
# #     """
# #     def __init__(self, bitrate_kbps=12):
# #         if not OPUSLIB_AVAILABLE:
# #             raise ImportError("opuslib not found. Please install it with 'pip install opuslib' and ensure libopus is installed.")
        
# #         self.sample_rate = SR
# #         self.channels = CHANNELS
# #         self.bitrate_bps = int(bitrate_kbps * 1000)
        
# #         # App's frame size is 20ms (320 samples)
# #         self.samples_per_frame = HOP_SIZE # 320
        
# #         try:
# #             # --- Encoder ---
# #             self._opus_encoder = opuslib.Encoder(
# #                 self.sample_rate, 
# #                 self.channels, 
# #                 opuslib.APPLICATION_VOIP
# #             )
# #             self._opus_encoder.bitrate = self.bitrate_bps
            
# #             # --- Decoder ---
# #             self._opus_decoder = opuslib.Decoder(
# #                 self.sample_rate, 
# #                 self.channels
# #             )
# #         except Exception as e:
# #             raise RuntimeError(f"Failed to initialize opuslib Encoder/Decoder: {e}")
            
# #         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# #     def encode(self, audio_tensor):
# #         """
# #         Encodes a float tensor chunk into Opus bytes.
# #         Input: audio_tensor (torch.Tensor, shape [1, 1, 320])
# #         Output: encoded_bytes (bytes)
# #         """
# #         # 1. Convert float tensor [-1.0, 1.0] to int16 bytes
# #         pcm_int16 = (audio_tensor.squeeze().cpu().numpy() * 32767.0).astype(np.int16)
        
# #         # Ensure correct frame size
# #         if len(pcm_int16) != self.samples_per_frame:
# #              pcm_int16 = np.pad(
# #                  pcm_int16, 
# #                  (0, self.samples_per_frame - len(pcm_int16)), 
# #                  'constant'
# #              )[:self.samples_per_frame]
             
# #         pcm_bytes = pcm_int16.tobytes()
        
# #         # 2. Encode the bytes
# #         try:
# #             encoded_bytes = self._opus_encoder.encode(pcm_bytes, self.samples_per_frame)
# #             return encoded_bytes
# #         except Exception as e:
# #             print(f"OpusLib encode error: {e}")
# #             return b''

# #     def decode(self, encoded_bytes):
# #         """
# #         Decodes Opus bytes into a float tensor chunk.
# #         Input: encoded_bytes (bytes)
# #         Output: audio_tensor (torch.Tensor, shape [1, 1, 320])
# #         """
# #         try:
# #             # 1. Decode bytes
# #             decoded_bytes = self._opus_decoder.decode(encoded_bytes, self.samples_per_frame)
# #         except Exception as e:
# #             # Packet loss or corruption
# #             decoded_bytes = self._opus_decoder.decode_missing_packet(self.samples_per_frame)
            
# #         # 2. Convert int16 bytes back to float tensor
# #         decoded_np = np.frombuffer(decoded_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        
# #         # 3. Add batch/channel dims and return
# #         return torch.from_numpy(decoded_np).unsqueeze(0).unsqueeze(0).to(self.device)
# # # --- END: New OpusLib Streaming Codec ---


# # # --- START: New Fine-Tuned Encodec 24kHz Wrapper ---
# # class FineTunedEncodecWrapper:
# #     """
# #     Wraps the 24kHz Fine-Tuned Encodec model to make it compatible
# #     with the 16kHz streaming and evaluation pipeline of the app.
    
# #     It handles all necessary resampling and buffering.
# #     """
# #     def __init__(self, model_path):
# #         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
# #         # 1. Load the 24kHz model from codec_model.py
# #         self.model_24k = load_encodec_model_24k(model_path)
# #         self.model_24k.to(self.device)
# #         self.model_24k.eval()
        
# #         self.target_sr = SR # App's sample rate (16kHz)
# #         self.model_sr = ENCODEC_SR_24K # Model's sample rate (24kHz)
        
# #         # 2. Create resamplers
# #         self.resampler_16_to_24 = T.Resample(self.target_sr, self.model_sr).to(self.device)
# #         self.resampler_24_to_16 = T.Resample(self.model_sr, self.target_sr).to(self.device)
        
# #         # 3. Define model's frame parameters (at 24kHz)
# #         self.frame_samples_24k = ENCODEC_FRAME_SAMPLES
# #         self.hop_samples_24k = ENCODEC_FRAME_SAMPLES - ENCODEC_OVERLAP_SAMPLES
# #         self.overlap_samples_24k = ENCODEC_OVERLAP_SAMPLES
        
# #         # 4. Internal buffers for streaming
# #         # These buffers will hold 24kHz audio
# #         self.encode_buffer_24k = torch.tensor([], device=self.device)
# #         self.decode_buffer_24k = torch.tensor([], device=self.device)
        
# #         # This buffer holds 16kHz decoded audio
# #         self.output_buffer_16k = torch.tensor([], device=self.device)

# #     @torch.no_grad()
# #     def encode(self, audio_chunk_16k_tensor):
# #         """
# #         Accepts a 16kHz audio chunk (NOW 320 SAMPLES), adds it to a buffer, 
# #         and processes it if the buffer is full.
# #         Returns bytes if a frame was processed, else None.
# #         """
# #         # 1. Resample incoming 16kHz chunk to 24kHz and add to buffer
# #         audio_chunk_24k = self.resampler_16_to_24(audio_chunk_16k_tensor.to(self.device))
# #         self.encode_buffer_24k = torch.cat([self.encode_buffer_24k, audio_chunk_24k.squeeze(0)])

# #         # 2. Check if we have enough data to process at least one frame
# #         if self.encode_buffer_24k.shape[0] < self.frame_samples_24k:
# #             return None # Not enough data yet

# #         # 3. Extract the frame to process
# #         frame_to_process = self.encode_buffer_24k[:self.frame_samples_24k].unsqueeze(0).unsqueeze(0)
        
# #         # 4. Update the buffer (Overlap-Add)
# #         # Keep the last 'overlap' samples for the next frame
# #         self.encode_buffer_24k = self.encode_buffer_24k[self.hop_samples_24k:]
        
# #         # 5. Run the model's encode function
# #         # We need to get the *raw codes* (indices), not the float tensor
# #         encoded_output = self.model_24k.encodec.encode(
# #             frame_to_process, 
# #             bandwidth=ENCODEC_BANDWIDTH
# #         )
        
# #         # (B, n_codebooks, T_latent)
# #         codes = encoded_output.audio_codes 
# #         codes_np = codes.cpu().numpy().astype(np.int32)
        
# #         # 6. Pack and return bytes
# #         shape = codes_np.shape
# #         header = struct.pack('I', len(shape))
# #         for s in shape:
# #             header += struct.pack('I', s)
# #         payload = header + codes_np.tobytes()
# #         return payload

# #     @torch.no_grad()
# #     def decode(self, payload_bytes):
# #         """
# #         Accepts the byte payload, decodes it to 24kHz, resamples to 16kHz,
# #         and returns a 16kHz audio chunk.
# #         """
# #         # 1. Unpack bytes
# #         offset = 0
# #         num_dims = struct.unpack('I', payload_bytes[offset:offset + 4])[0]
# #         offset += 4
# #         shape = []
# #         for i in range(num_dims):
# #             shape.append(struct.unpack('I', payload_bytes[offset + 4 * i:offset + 4 * (i + 1)])[0])
# #         offset += 4 * num_dims
        
# #         codes_bytes = payload_bytes[offset:]
# #         codes_np = np.frombuffer(codes_bytes, dtype=np.int32).reshape(shape)
# #         codes_tensor = torch.from_numpy(codes_np).to(self.device)
        
# #         # 2. Decode using the 24k model
# #         # We pass dummy scales. The model wrapper's .decode handles it.
# #         dummy_scales = torch.ones(shape[0], 1, shape[2], device=self.device)
# #         decoded_output_24k = self.model_24k.encodec.decode(codes_tensor, [dummy_scales])
# #         decoded_frame_24k = decoded_output_24k.audio_values.squeeze() # (L_frame_24k)

# #         # 3. Add to internal 24k buffer
# #         if self.decode_buffer_24k.shape[0] > 0:
# #             # Overlap-add logic
# #             overlap_region = self.decode_buffer_24k
# #             current_region = decoded_frame_24k[:self.overlap_samples_24k]
            
# #             # Blend
# #             fade_out = torch.linspace(1, 0, self.overlap_samples_24k, device=self.device)
# #             fade_in = torch.linspace(0, 1, self.overlap_samples_24k, device=self.device)
# #             blended = overlap_region * fade_out + current_region * fade_in
            
# #             # New audio is the blended part + the rest of the hop
# #             new_audio_24k = torch.cat([
# #                 blended,
# #                 decoded_frame_24k[self.overlap_samples_24k:self.hop_samples_24k]
# #             ])
# #         else:
# #             # First frame, just take the hop part
# #             new_audio_24k = decoded_frame_24k[:self.hop_samples_24k]
            
# #         # 4. Save the new overlap buffer
# #         self.decode_buffer_24k = decoded_frame_24k[self.hop_samples_24k:]

# #         # 5. Resample the *new audio* back to 16kHz and add to output buffer
# #         new_audio_16k = self.resampler_24_to_16(new_audio_24k)
# #         self.output_buffer_16k = torch.cat([self.output_buffer_16k, new_audio_16k])
        
# #         # 6. Return 16kHz chunks to the app
# #         # --- *** CRITICAL CHANGE: Return 320 samples now *** ---
# #         app_hop_samples = HOP_SIZE # 320 samples
# #         if self.output_buffer_16k.shape[0] >= app_hop_samples:
# #             output_chunk_16k = self.output_buffer_16k[:app_hop_samples]
# #             self.output_buffer_16k = self.output_buffer_16k[app_hop_samples:]
# #             return (output_chunk_16k.cpu().numpy() * 32767.0).astype(np.int16).tobytes()
# #         else:
# #             return None # Not enough 16k audio to play yet

# #     @torch.no_grad()
# #     def process_full(self, audio_16k_tensor):
# #         """
# #         Processes an entire 16kHz audio tensor for evaluation.
# #         Returns a full 16kHz audio tensor.
# #         """
# #         # 1. Resample full audio to 24kHz
# #         audio_24k = self.resampler_16_to_24(audio_16k_tensor.to(self.device))
        
# #         # 2. Run the 24kHz model's built-in streaming processor
# #         if audio_24k.dim() == 1:
# #             audio_24k = audio_24k.unsqueeze(0)
# #         if audio_24k.dim() == 2:
# #             audio_24k = audio_24k.unsqueeze(1) # (B, C, L)
            
# #         # process_streaming is a helper from *your* codec_model.py
# #         reconstructed_24k = self.model_24k.process_streaming(
# #             audio_24k, 
# #             bandwidth=ENCODEC_BANDWIDTH
# #         ).squeeze()
        
# #         # 3. Resample the full reconstructed audio back to 16kHz
# #         reconstructed_16k = self.resampler_24_to_16(reconstructed_24k.cpu())
        
# #         return reconstructed_16k.squeeze().numpy()
# # # --- END: New Fine-Tuned Encodec 24kHz Wrapper ---


# # # --- START: New Opus Fine-Tuning Model Definitions ---
# # # These classes are copied directly from your Colab script
# # # to define the enhancement model's architecture.

# # class CausalConv1d(nn.Module):
# #     """Causal convolution with better initialization"""
    
# #     def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
# #         super().__init__()
# #         self.padding = (kernel_size - 1) * dilation
# #         self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, 
# #                               padding=self.padding, dilation=dilation)
        
# #         # Xavier initialization for stability
# #         nn.init.xavier_uniform_(self.conv.weight)
# #         nn.init.zeros_(self.conv.bias)
    
# #     def forward(self, x):
# #         x = self.conv(x)
# #         return x[:, :, :-self.padding] if self.padding != 0 else x


# # class StreamingEnhancementNet(nn.Module):
# #     """FIXED: More stable training with dropout"""
    
# #     def __init__(self, hidden_size=48, num_layers=2, kernel_size=3, dropout=0.1):
# #         super().__init__()
        
# #         self.hidden_size = hidden_size
# #         self.dropout = dropout
        
# #         # Input projection
# #         self.input_conv = CausalConv1d(1, hidden_size, kernel_size=1)
# #         self.input_norm = nn.BatchNorm1d(hidden_size)
        
# #         # Causal blocks with dropout
# #         self.conv_blocks = nn.ModuleList()
# #         for i in range(num_layers):
# #             dilation = 2 ** i
# #             self.conv_blocks.append(nn.ModuleList([
# #                 CausalConv1d(hidden_size, hidden_size, kernel_size, dilation),
# #                 nn.PReLU(hidden_size, init=0.25),
# #                 nn.BatchNorm1d(hidden_size, momentum=0.01, eps=1e-3),
# #                 nn.Dropout(dropout),  # Add dropout
# #             ]))
        
# #         # Output projection
# #         self.output_conv = nn.Sequential(
# #             CausalConv1d(hidden_size, hidden_size // 2, kernel_size=1),
# #             nn.PReLU(hidden_size // 2, init=0.25),
# #             nn.BatchNorm1d(hidden_size // 2, momentum=0.01, eps=1e-3),
# #             CausalConv1d(hidden_size // 2, 1, kernel_size=1),
# #         )
        
# #         # Initialize output layer to near-zero (start with identity)
# #         nn.init.xavier_uniform_(self.output_conv[-1].conv.weight, gain=0.001)
# #         nn.init.zeros_(self.output_conv[-1].conv.bias)
        
# #         # Learnable residual weight (start even closer to identity)
# #         self.alpha = nn.Parameter(torch.tensor(0.02))
    
# #     def forward(self, x):
# #         identity = x
        
# #         # Input projection
# #         x = self.input_conv(x)
# #         x = self.input_norm(x)
# #         x = F.relu(x)
        
# #         # Causal blocks with dropout
# #         for conv, activation, norm, dropout in self.conv_blocks:
# #             residual = x
# #             x = conv(x)
# #             x = activation(x)
# #             x = norm(x)
# #             x = dropout(x)  # Apply dropout
# #             x = x + residual
        
# #         # Output
# #         x = self.output_conv(x)
        
# #         # Adaptive residual (clamped for stability)
# #         alpha = torch.clamp(self.alpha, 0.0, 0.5)
# #         output = alpha * x + (1 - alpha) * identity
        
# #         # Ensure valid output range
# #         output = torch.clamp(output, -1.0, 1.0)
        
# #         return output
# # # --- END: New Opus Fine-Tuning Model Definitions ---


# # # --- START: New Opus Fine-Tuned Codec Wrapper ---
# # class OpusFineTunedCodec:
# #     """
# #     Wrapper for the Opus Fine-Tuned (post-filter) model.
    
# #     - encode(): Uses standard Opus 12kbps encode.
# #     - decode(): Uses standard Opus 12kbps decode, THEN runs the
# #                 result through the trained enhancement network.
# #     """
# #     def __init__(self, model_path, bitrate_kbps=12):
# #         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
# #         if not OPUSLIB_AVAILABLE:
# #             raise ImportError("opuslib not found. Please install it with 'pip install opuslib' and ensure libopus is installed.")
            
# #         # 1. Load the underlying Opus codec
# #         self.opus_codec = OpusLibStreamingCodec(bitrate_kbps=bitrate_kbps)
        
# #         # 2. Load the trained enhancement model
# #         try:
# #             # Define the model architecture from the Colab script
# #             # CONFIG: hidden_size=48, num_layers=2, kernel_size=3, dropout=0.1
# #             self.enhancement_model = StreamingEnhancementNet(
# #                 hidden_size=48,
# #                 num_layers=2,
# #                 kernel_size=3,
# #                 dropout=0.1
# #             ).to(self.device)
            
# #             # Load the state dict
# #             checkpoint = torch.load(model_path, map_location=self.device)
# #             state_dict = checkpoint.get('model_state_dict', checkpoint)
# #             self.enhancement_model.load_state_dict(state_dict)
# #             self.enhancement_model.eval()
# #             print(f"OpusFineTunedCodec: Enhancement model loaded from {model_path}")
# #         except Exception as e:
# #             print(f"ERROR loading Opus enhancement model from {model_path}: {e}")
# #             raise
            
# #     def encode(self, audio_tensor):
# #         """
# #         Encodes the audio using the standard Opus codec.
# #         Input: audio_tensor (torch.Tensor, shape [1, 1, 320])
# #         Output: encoded_bytes (bytes)
# #         """
# #         return self.opus_codec.encode(audio_tensor)

# #     @torch.no_grad()
# #     def decode(self, encoded_bytes):
# #         """
# #         Decodes the Opus bytes AND applies the enhancement filter.
# #         Input: encoded_bytes (bytes)
# #         Output: enhanced_audio_tensor (torch.Tensor, shape [1, 1, 320])
# #         """
# #         # 1. Decode the raw 12kbps Opus stream
# #         # This gives the "degraded" audio
# #         degraded_tensor = self.opus_codec.decode(encoded_bytes)
        
# #         # 2. Pass the degraded audio through the enhancement model
# #         # The model expects (B, C, L), which matches degraded_tensor
# #         enhanced_tensor = self.enhancement_model(degraded_tensor.to(self.device))
        
# #         # 3. Return the enhanced audio
# #         return enhanced_tensor
# # # --- END: New Opus Fine-Tuned Codec Wrapper ---


# # # --- CUSTOM TINY VQ-CODEC COMPONENTS (Causal Architecture) ---
# # class CausalConvBlock(nn.Module):
# #     """Causal Conv1d block matching the architecture from the trainer script."""
# #     def __init__(self, in_channels, out_channels, kernel_size, stride):
# #         super().__init__()
# #         self.padding_amount = kernel_size - 1
# #         self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding=0)
# #         self.norm = nn.GroupNorm(1, out_channels)
# #         self.activation = nn.GELU() # Synchronized
# #         self.stride = stride

# #     def forward(self, x):
# #         # Apply padding only to the left (causal)
# #         x = F.pad(x, (self.padding_amount, 0), mode='constant', value=0)
# #         x = self.activation(self.norm(self.conv(x)))
# #         return x

# # class OptimizedTransformerBlock(nn.Module):
# #     """Optimized transformer block matching the exact 4-module FFN sequence of the trainer script."""
# #     def __init__(self, dim, heads):
# #         super().__init__()
# #         self.attn = nn.MultiheadAttention(dim, heads, batch_first=True, dropout=0.1)
# #         self.norm1 = nn.LayerNorm(dim)
# #         self.norm2 = nn.LayerNorm(dim)
        
# #         # FFN sequence must match the 4 layers in the checkpoint
# #         self.ffn = nn.Sequential(
# #             nn.Linear(dim, dim * 2),
# #             nn.GELU(),
# #             nn.Dropout(0.1),
# #             nn.Linear(dim * 2, dim)
# #         )

# #     def forward(self, x):
# #         B, C, T = x.shape
# #         x_attn = x.transpose(1, 2)
        
# #         # Causal mask ensures future tokens are not attended to
# #         attn_mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1)
        
# #         attn_output, _ = self.attn(x_attn, x_attn, x_attn, attn_mask=attn_mask, is_causal=False)
# #         x_attn = self.norm1(x_attn + attn_output)
        
# #         ffn_output = self.ffn(x_attn)
# #         x_attn = self.norm2(x_attn + ffn_output)
        
# #         return x_attn.transpose(1, 2)

# # class ImprovedVectorQuantizer(nn.Module):
# #     """
# #     Vector Quantization layer, synchronized with the training script's class definition 
# #     to correctly load EMA buffers.
# #     """
# #     def __init__(self, num_embeddings=CODEBOOK_SIZE, embedding_dim=LATENT_DIM, commitment_cost=COMMITMENT_COST):
# #         super().__init__()
# #         self.embedding_dim = embedding_dim
# #         self.num_embeddings = num_embeddings
# #         self.commitment_cost = commitment_cost
        
# #         self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)
        
# #         # Include EMA buffers/parameters for compatibility
# #         self.register_buffer('ema_cluster_size', torch.zeros(num_embeddings))
# #         self.register_buffer('ema_w', self.embedding.weight.data.clone())
# #         self.ema_decay = 0.99


# #     def forward(self, inputs):
# #         # Flatten input (B, C, T) -> (B*T, C)
# #         flat_input = inputs.transpose(1, 2).contiguous().view(-1, self.embedding_dim)
        
# #         # Calculate distances
# #         distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
# #                       + torch.sum(self.embedding.weight**2, dim=1)
# #                       - 2 * torch.matmul(flat_input, self.embedding.weight.t()))
            
# #         # Find the nearest codebook vector index
# #         encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        
# #         # Create one-hot vectors
# #         encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=inputs.device)
# #         encodings.scatter_(1, encoding_indices, 1)
        
# #         # Quantize vector
# #         quantized = torch.matmul(encodings, self.embedding.weight).view(inputs.shape[0], inputs.shape[2], -1).transpose(1, 2)
        
# #         # Apply STE
# #         quantized = inputs + (quantized - inputs).detach()
        
# #         # Loss logic is removed (only needed for training)
        
# #         return quantized, encoding_indices

# # class TinyTransformerCodec(nn.Module):
# #     def __init__(self, latent_dim=LATENT_DIM, blocks=BLOCKS, heads=HEADS, sr=SR):
# #         super().__init__()
# #         self.latent_dim = latent_dim
# #         self.sr = sr
# #         self.downsampling_factor = DOWN_FACTOR
# #         self.num_codebooks = NUM_CODEBOOKS

# #         # --- Encoder ---
# #         self.encoder_convs = nn.ModuleList()
# #         in_c = CHANNELS
# #         encoder_channels = []
# #         for i in range(blocks):
# #             # OPTIMIZED: Scaled channels based on new LATENT_DIM
# #             out_c = min(latent_dim, 16 * (2**i)) # Start with 16, up to 64
# #             encoder_channels.append(out_c)
# #             stride = STRIDES[i]
# #             self.encoder_convs.append(CausalConvBlock(in_c, out_c, KERNEL_SIZE, stride))
# #             in_c = out_c
        
# #         self.pre_quant = CausalConvBlock(in_c, LATENT_DIM * NUM_CODEBOOKS, KERNEL_SIZE, 1)

# #         # --- Vector Quantization (Using the compatible class) ---
# #         self.quantizers = nn.ModuleList([
# #             ImprovedVectorQuantizer(CODEBOOK_SIZE, LATENT_DIM, commitment_cost=COMMITMENT_COST)
# #             for _ in range(NUM_CODEBOOKS)
# #         ])
        
# #         # --- Transformer (Using the compatible block) ---
# #         self.transformer = nn.Sequential(*[
# #             OptimizedTransformerBlock(latent_dim * NUM_CODEBOOKS, heads)
# #             for _ in range(TRANSFORMER_BLOCKS)
# #         ])
# #         self.post_transformer = nn.Conv1d(latent_dim * NUM_CODEBOOKS, latent_dim * NUM_CODEBOOKS, 1)

# #         # --- Decoder ---
# #         self.decoder_tconvs = nn.ModuleList()
# #         self.skip_convs = nn.ModuleList()
        
# #         in_c = latent_dim * NUM_CODEBOOKS
# #         for i in range(blocks):
# #             idx = blocks - 1 - i
# #             stride = STRIDES[idx]
            
# #             if idx > 0:
# #                 out_c = encoder_channels[idx - 1]
# #             else:
# #                 out_c = 16 # Final channel before output
            
# #             self.decoder_tconvs.append(
# #                 nn.ConvTranspose1d(in_c, out_c, KERNEL_SIZE, stride, padding=KERNEL_SIZE//2)
# #             )
            
# #             if idx > 0:
# #                 skip_in_channels = encoder_channels[idx - 1]
# #                 # This skip_conv layer still needs to be defined to match checkpoint keys
# #                 self.skip_convs.append(
# #                     nn.Conv1d(out_c + skip_in_channels, out_c, kernel_size=1)
# #                 )
# #             in_c = out_c
        
# #         self.post_decoder_final = nn.Conv1d(in_c, CHANNELS, 1)

# #     @classmethod
# #     def load_model(cls, model_path):
# #         """Loads the model weights and returns the initialized model."""
# #         model = cls()
# #         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
# #         if not os.path.exists(model_path):
# #             raise FileNotFoundError(f"Trained model not found at path: {model_path}")
            
# #         try:
# #             checkpoint = torch.load(model_path, map_location=device)
# #             state_dict = checkpoint.get('model_state_dict', checkpoint)
# #             model.load_state_dict(state_dict)
# #             model.to(device)
# #             model.eval()
# #             print(f"TinyTransformerCodec (VQ-Codec) loaded successfully from {model_path}.")
# #             return model
# #         except Exception as e:
# #             print(f"Error loading model state dict: {e}")
# #             raise

# #     def encode(self, x):
# #         """
# #         Encodes audio into quantized latent codes/indices.
# #         """
# #         x = x.view(x.size(0), CHANNELS, -1)
# #         input_length = x.shape[-1]
        
# #         encoder_outputs = []
        
# #         # Encoder
# #         for layer in self.encoder_convs:
# #             x = layer(x)
# #             encoder_outputs.append(x)
        
# #         # Pre-quantization
# #         z_e = self.pre_quant(x)
        
# #         # Vector Quantization
# #         z_q_list = []
# #         indices_list = []
# #         z_e_split = z_e.chunk(self.num_codebooks, dim=1)
        
# #         for i in range(self.num_codebooks):
# #             z_q, indices = self.quantizers[i](z_e_split[i]) # VQ layer returns quantized and indices
# #             z_q_list.append(z_q)
# #             indices_list.append(indices)
        
# #         # Concatenated quantized latent features
# #         quantized_codes = torch.cat(z_q_list, dim=1)
        
# #         # Pass through transformer
# #         codes = self.transformer(quantized_codes)
# #         codes = self.post_transformer(codes)
        
# #         return codes, indices_list, input_length, encoder_outputs

# #     def decode(self, codes_or_indices_list, input_length=None, encoder_outputs=None):
# #         """
# #         Decodes from VQ-quantized float codes (for evaluation) OR integer indices (for streaming).
# #         """
        
# #         if isinstance(codes_or_indices_list, list):
# #             # Case 1: Decoding from raw integer indices (streaming receiver)
# #             indices_list = codes_or_indices_list
# #             z_q_list = []
            
# #             for i, indices in enumerate(indices_list):
# #                 # indices is (T_latent, 1). Squeeze to (T_latent) for embedding lookup
# #                 quantized = self.quantizers[i].embedding(indices.squeeze(1))
                
# #                 # Add batch dim: (T_latent, C_latent) -> (1, T_latent, C_latent)
# #                 quantized = quantized.unsqueeze(0)
# #                 # Transpose to (B, C, T): (1, T_latent, C_latent) -> (1, C_latent, T_latent)
# #                 quantized = quantized.transpose(1, 2)
                
# #                 z_q_list.append(quantized)
            
# #             x_pre_transformer = torch.cat(z_q_list, dim=1)
            
# #             # Streaming path must also pass through transformer
# #             x = self.transformer(x_pre_transformer)
# #             x = self.post_transformer(x)
            
# #             # In streaming, we have no skip connections
# #             encoder_outputs = None 
            
# #         elif isinstance(codes_or_indices_list, torch.Tensor):
# #             # Case 2: Decoding from the quantized float tensor (evaluation)
# #             # This tensor 'codes' is already post-transformer from the encode() step.
# #             x = codes_or_indices_list
# #             # 'encoder_outputs' is passed from the function arguments
# #         else:
# #             raise ValueError("Decoding input must be a torch.Tensor (quantized codes) or a list of Tensors (indices).")

# #         # Decoder
# #         for i, tconv in enumerate(self.decoder_tconvs):
# #             x = F.gelu(tconv(x)) # Synchronized activation
            
# #             # --- SKIP CONNECTION LOGIC (Intentionally disabled to match training) ---
# #             #
# #             # if encoder_outputs and i < len(self.skip_convs):
# #             # ...
# #             # --- END SKIP LOGIC ---
        
# #         # Final output
# #         x = torch.tanh(self.post_decoder_final(x))
        
# #         # Match input length
# #         if input_length is not None:
# #             if x.shape[-1] > input_length:
# #                 x = x[..., :input_length]
# #             elif x.shape[-1] < input_length:
# #                 x = F.pad(x, (0, input_length - x.shape[-1]))
        
# #         return x.view(x.size(0), CHANNELS, -1)


# # # --- TRADITIONAL CODECS (Unchanged) ---

# # class MuLawCodec:
# #     """μ-law codec for baseline comparison"""
# #     def __init__(self, quantization_channels=256):
# #         self.mu = float(quantization_channels - 1)
    
# #     def encode(self, x):
# #         mu_t = torch.tensor(self.mu, dtype=torch.float32, device=x.device)
# #         encoded = torch.sign(x) * torch.log1p(mu_t * torch.abs(x)) / torch.log1p(mu_t)
# #         return (((encoded + 1) / 2 * self.mu) + 0.5).to(torch.uint8)
    
# #     def decode(self, z):
# #         z_float = z.to(torch.float32)
# #         mu_t = torch.tensor(self.mu, dtype=torch.float32, device=z.device)
# #         y = (z_float / self.mu) * 2.0 - 1.0
# #         return (torch.sign(y) * (1.0 / self.mu) * (torch.pow(1.0 + self.mu, torch.abs(y)) - 1.0)).unsqueeze(1)

# # class ALawCodec:
# #     """A-law codec for baseline comparison"""
# #     def __init__(self):
# #         self.A = 87.6
    
# #     def encode(self, x):
# #         a_t = torch.tensor(self.A, dtype=torch.float32, device=x.device)
# #         abs_x = torch.abs(x)
# #         encoded = torch.zeros_like(x)
# #         cond = abs_x < (1 / self.A)
# #         encoded[cond] = torch.sign(x[cond]) * (a_t * abs_x[cond]) / (1 + torch.log(a_t))
# #         encoded[~cond] = torch.sign(x[~cond]) * (1 + torch.log(a_t * abs_x[~cond])) / (1 + torch.log(a_t))
# #         return (((encoded + 1) / 2 * 255) + 0.5).to(torch.uint8)
    
# #     def decode(self, z):
# #         z_float = z.to(torch.float32)
# #         a_t = torch.tensor(self.A, dtype=torch.float32, device=z.device)
# #         y = (z_float / 127.5) - 1.0
# #         abs_y = torch.abs(y)
# #         decoded = torch.zeros_like(y)
# #         cond = abs_y < (1 / (1 + torch.log(a_t)))
# #         decoded[cond] = torch.sign(y[cond]) * (abs_y[cond] * (1 + torch.log(a_t))) / a_t
# #         decoded[~cond] = torch.sign(y[~cond]) * torch.exp(abs_y[~cond] * (1 + torch.log(a_t)) - 1) / a_t
# #         return decoded.unsqueeze(1)

# # class AMRWBCodec:
# #     """Simulated AMR-WB (12.65 kbps mode, 20ms frame) using Mu-Law for quantization."""
# #     def __init__(self, mode=6): # Use mode 6 (12.65 kbps)
# #         self.mu_codec = MuLawCodec()
# #         self.sample_rate = 16000
# #         # --- **** CRITICAL CHANGE **** ---
# #         self.frame_size = 320 # 20ms at 16kHz
# #         # --- **** END CRITICAL CHANGE **** ---
# #         self.codec_bitrate = 12.65
        
# #     def encode(self, x):
# #         return self.mu_codec.encode(x)
    
# #     def decode(self, z):
# #         return self.mu_codec.decode(z)

# # # --- DAC CODEC (MODIFIED) ---
# # class DACCodec:
# #     """Wrapper for Descript Audio Codec (DAC)"""
# #     def __init__(self, model_path=None, model_type="16khz"):
# #         if not DAC_AVAILABLE:
# #             raise ImportError("DAC is not installed. Install with: pip install descript-audio-codec")
# #         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# #         self.model_type = model_type
# #         print(f"Loading DAC {model_type} model...")
# #         try:
# #             model_path = dac.utils.download(model_type=model_type)
# #             self.model = dac.DAC.load(model_path)
# #             print(f"DAC model loaded successfully")
# #         except Exception as e:
# #             print(f"Error loading DAC model: {e}")
# #             raise
# #         self.model.to(self.device)
# #         self.model.eval()
# #         self.sample_rate = 16000 if "16khz" in model_type else 44100
        
# #         # --- **** CRITICAL CHANGE **** ---
# #         # This model's hop size IS 320! It works perfectly now.
# #         self.hop_size = 320 # 20ms @ 16kHz
# #         self.chunk_size = 320 # Process 20ms chunks
# #         # --- **** END CRITICAL CHANGE **** ---
        
# #     def encode(self, audio_tensor):
# #         with torch.no_grad():
# #             if audio_tensor.dim() == 2: audio_tensor = audio_tensor.unsqueeze(1)
# #             audio_tensor = audio_tensor.to(self.device, dtype=torch.float32)
# #             original_length = audio_tensor.shape[-1]
# #             if original_length % self.hop_size != 0:
# #                 pad_length = self.hop_size - (original_length % self.hop_size)
# #                 audio_tensor = F.pad(audio_tensor, (0, pad_length))
# #             _, codes, _, _, _ = self.model.encode(audio_tensor)
# #             return codes, original_length # Returns the integer codes tensor and original length
    
# #     def decode(self, codes, original_length=None):
# #         with torch.no_grad():
# #             if not isinstance(codes, torch.Tensor): codes = torch.tensor(codes, dtype=torch.long)
# #             codes = codes.to(self.device)
# #             z = self.model.quantizer.from_codes(codes)[0]
# #             audio_recon = self.model.decode(z)
# #             if original_length is not None and audio_recon.shape[-1] > original_length:
# #                 audio_recon = audio_recon[..., :original_length]
# #             return audio_recon


# # # --- UTILITY FUNCTIONS (Unchanged) ---
# # def get_model_cache_info():
# #     """Returns information about cached models"""
# #     cache_info = {}
# #     if os.path.exists(MODELS_CACHE_DIR):
# #         for file in os.listdir(MODELS_CACHE_DIR):
# #             file_path = os.path.join(MODELS_CACHE_DIR, file)
# #             if os.path.isfile(file_path):
# #                 file_size = os.path.getsize(file_path) / (1024 * 1024)  # Size in MB
# #                 cache_info[file] = f"{file_size:.2f} MB"
# #     return cache_info

# # def clear_model_cache():
# #     """Clears all cached models"""
# #     import shutil
# #     if os.path.exists(MODELS_CACHE_DIR):
# #         shutil.rmtree(MODELS_CACHE_DIR)
# #         os.makedirs(MODELS_CACHE_DIR)
# #         print("Model cache cleared.")


# import torch
# import torch.nn as nn
# import numpy as np
# import torch.nn.functional as F
# import os
# import struct
# import subprocess
# import sys
# import math

# # --- START: New Imports ---
# import torchaudio
# import torchaudio.transforms as T
# from codec_model import (
#     load_codec_model as load_encodec_model_24k,
#     RealTimeEncodecWrapper as EncodecStreamingWrapper,
#     TARGET_SR as ENCODEC_SR_24K,
#     TARGET_BANDWIDTH as ENCODEC_BANDWIDTH,
#     STREAMING_FRAME_SAMPLES as ENCODEC_FRAME_SAMPLES,
#     OVERLAP_SAMPLES as ENCODEC_OVERLAP_SAMPLES
# )
# # --- END: New Imports ---

# # --- START: OpusLib Import ---
# try:
#     import opuslib
#     OPUSLIB_AVAILABLE = True
# except (ImportError, OSError) as e:
#     OPUSLIB_AVAILABLE = False
#     print(f"WARNING: opuslib not installed or libopus not found (Error: {e}). Opus (opuslib) codec will be unavailable.")
# # --- END: OpusLib Import ---


# # --- Global Configuration (Synchronized with ~9.3kbps, 15ms LATENCY Trainer) ---
# SR = 16000
# CHANNELS = 1
# LATENT_DIM = 64  
# BLOCKS = 4
# HEADS = 8        
# KERNEL_SIZE = 3

# # Synchronized: 2 * 2 * 2 * 3 = 24x downsampling (Unchanged)
# STRIDES = [2, 2, 2, 3] 
# DOWN_FACTOR = np.prod(STRIDES) # 24

# # Synchronized: 2 Codebooks (128 entries each)
# NUM_CODEBOOKS = 2
# CODEBOOK_SIZE = 128 
# COMMITMENT_COST = 0.25  
# TRANSFORMER_BLOCKS = 4  

# # --- **** CRITICAL CHANGE **** ---
# # HOP size changed to 20ms (320 samples) to support Opus
# # This is now the app's fundamental chunk size.
# HOP_SIZE = int(SR / (1000 / 20)) # 320 samples (20ms)
# # --- **** END CRITICAL CHANGE **** ---

# # Models cache directory
# MODELS_CACHE_DIR = os.path.join(os.path.expanduser("~"), ".audio_codec_models")
# if not os.path.exists(MODELS_CACHE_DIR):
#     os.makedirs(MODELS_CACHE_DIR)

# # --- DAC INTEGRATION (Unchanged) ---
# try:
#     import dac
#     DAC_AVAILABLE = True
# except ImportError:
#     DAC_AVAILABLE = False


# # --- START: New OpusLib Streaming Codec ---
# class OpusLibStreamingCodec:
#     """
#     A stateless streaming Opus codec using 'opuslib'.
#     It's designed to process 20ms (320-sample) chunks,
#     matching the application's streaming buffer.
#     """
#     def __init__(self, bitrate_kbps=12):
#         if not OPUSLIB_AVAILABLE:
#             raise ImportError("opuslib not found. Please install it with 'pip install opuslib' and ensure libopus is installed.")
        
#         self.sample_rate = SR
#         self.channels = CHANNELS
#         self.bitrate_bps = int(bitrate_kbps * 1000)
        
#         # App's frame size is 20ms (320 samples)
#         self.samples_per_frame = HOP_SIZE # 320
        
#         try:
#             # --- Encoder ---
#             self._opus_encoder = opuslib.Encoder(
#                 self.sample_rate, 
#                 self.channels, 
#                 opuslib.APPLICATION_VOIP
#             )
#             self._opus_encoder.bitrate = self.bitrate_bps
#             # Set complexity to 0 (matches training script)
#             self._opus_encoder.complexity = 0 
            
#             # --- Decoder ---
#             self._opus_decoder = opuslib.Decoder(
#                 self.sample_rate, 
#                 self.channels
#             )
#         except Exception as e:
#             raise RuntimeError(f"Failed to initialize opuslib Encoder/Decoder: {e}")
            
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     def encode(self, audio_tensor):
#         """
#         Encodes a float tensor chunk into Opus bytes.
#         Input: audio_tensor (torch.Tensor, shape [1, 1, 320])
#         Output: encoded_bytes (bytes)
#         """
#         # 1. Convert float tensor [-1.0, 1.0] to int16 bytes
#         pcm_int16 = (audio_tensor.squeeze().cpu().numpy() * 32767.0).astype(np.int16)
        
#         # Ensure correct frame size
#         if len(pcm_int16) != self.samples_per_frame:
#              pcm_int16 = np.pad(
#                  pcm_int16, 
#                  (0, self.samples_per_frame - len(pcm_int16)), 
#                  'constant'
#              )[:self.samples_per_frame]
             
#         pcm_bytes = pcm_int16.tobytes()
        
#         # 2. Encode the bytes
#         try:
#             encoded_bytes = self._opus_encoder.encode(pcm_bytes, self.samples_per_frame)
#             return encoded_bytes
#         except Exception as e:
#             print(f"OpusLib encode error: {e}")
#             return b''

#     def decode(self, encoded_bytes):
#         """
#         Decodes Opus bytes into a float tensor chunk.
#         Input: encoded_bytes (bytes)
#         Output: audio_tensor (torch.Tensor, shape [1, 1, 320])
#         """
#         try:
#             # 1. Decode bytes
#             decoded_bytes = self._opus_decoder.decode(encoded_bytes, self.samples_per_frame)
#         except Exception as e:
#             # Packet loss or corruption
#             decoded_bytes = self._opus_decoder.decode_missing_packet(self.samples_per_frame)
            
#         # 2. Convert int16 bytes back to float tensor
#         decoded_np = np.frombuffer(decoded_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        
#         # 3. Add batch/channel dims and return
#         return torch.from_numpy(decoded_np).unsqueeze(0).unsqueeze(0).to(self.device)
# # --- END: New OpusLib Streaming Codec ---


# # --- START: New Fine-Tuned Encodec 24kHz Wrapper ---
# class FineTunedEncodecWrapper:
#     """
#     Wraps the 24kHz Fine-Tuned Encodec model to make it compatible
#     with the 16kHz streaming and evaluation pipeline of the app.
    
#     It handles all necessary resampling and buffering.
#     """
#     def __init__(self, model_path):
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
#         # 1. Load the 24kHz model from codec_model.py
#         self.model_24k = load_encodec_model_24k(model_path)
#         self.model_24k.to(self.device)
#         self.model_24k.eval()
        
#         self.target_sr = SR # App's sample rate (16kHz)
#         self.model_sr = ENCODEC_SR_24K # Model's sample rate (24kHz)
        
#         # 2. Create resamplers
#         self.resampler_16_to_24 = T.Resample(self.target_sr, self.model_sr).to(self.device)
#         self.resampler_24_to_16 = T.Resample(self.model_sr, self.target_sr).to(self.device)
        
#         # 3. Define model's frame parameters (at 24kHz)
#         self.frame_samples_24k = ENCODEC_FRAME_SAMPLES
#         self.hop_samples_24k = ENCODEC_FRAME_SAMPLES - ENCODEC_OVERLAP_SAMPLES
#         self.overlap_samples_24k = ENCODEC_OVERLAP_SAMPLES
        
#         # 4. Internal buffers for streaming
#         # These buffers will hold 24kHz audio
#         self.encode_buffer_24k = torch.tensor([], device=self.device)
#         self.decode_buffer_24k = torch.tensor([], device=self.device)
        
#         # This buffer holds 16kHz decoded audio
#         self.output_buffer_16k = torch.tensor([], device=self.device)

#     @torch.no_grad()
#     def encode(self, audio_chunk_16k_tensor):
#         """
#         Accepts a 16kHz audio chunk (NOW 320 SAMPLES), adds it to a buffer, 
#         and processes it if the buffer is full.
#         Returns bytes if a frame was processed, else None.
#         """
#         # 1. Resample incoming 16kHz chunk to 24kHz and add to buffer
#         audio_chunk_24k = self.resampler_16_to_24(audio_chunk_16k_tensor.to(self.device))
#         self.encode_buffer_24k = torch.cat([self.encode_buffer_24k, audio_chunk_24k.squeeze(0)])

#         # 2. Check if we have enough data to process at least one frame
#         if self.encode_buffer_24k.shape[0] < self.frame_samples_24k:
#             return None # Not enough data yet

#         # 3. Extract the frame to process
#         frame_to_process = self.encode_buffer_24k[:self.frame_samples_24k].unsqueeze(0).unsqueeze(0)
        
#         # 4. Update the buffer (Overlap-Add)
#         # Keep the last 'overlap' samples for the next frame
#         self.encode_buffer_24k = self.encode_buffer_24k[self.hop_samples_24k:]
        
#         # 5. Run the model's encode function
#         # We need to get the *raw codes* (indices), not the float tensor
#         encoded_output = self.model_24k.encodec.encode(
#             frame_to_process, 
#             bandwidth=ENCODEC_BANDWIDTH
#         )
        
#         # (B, n_codebooks, T_latent)
#         codes = encoded_output.audio_codes 
#         codes_np = codes.cpu().numpy().astype(np.int32)
        
#         # 6. Pack and return bytes
#         shape = codes_np.shape
#         header = struct.pack('I', len(shape))
#         for s in shape:
#             header += struct.pack('I', s)
#         payload = header + codes_np.tobytes()
#         return payload

#     @torch.no_grad()
#     def decode(self, payload_bytes):
#         """
#         Accepts the byte payload, decodes it to 24kHz, resamples to 16kHz,
#         and returns a 16kHz audio chunk.
#         """
#         # 1. Unpack bytes
#         offset = 0
#         num_dims = struct.unpack('I', payload_bytes[offset:offset + 4])[0]
#         offset += 4
#         shape = []
#         for i in range(num_dims):
#             shape.append(struct.unpack('I', payload_bytes[offset + 4 * i:offset + 4 * (i + 1)])[0])
#         offset += 4 * num_dims
        
#         codes_bytes = payload_bytes[offset:]
#         codes_np = np.frombuffer(codes_bytes, dtype=np.int32).reshape(shape)
#         codes_tensor = torch.from_numpy(codes_np).to(self.device)
        
#         # 2. Decode using the 24k model
#         # We pass dummy scales. The model wrapper's .decode handles it.
#         dummy_scales = torch.ones(shape[0], 1, shape[2], device=self.device)
#         decoded_output_24k = self.model_24k.encodec.decode(codes_tensor, [dummy_scales])
#         decoded_frame_24k = decoded_output_24k.audio_values.squeeze() # (L_frame_24k)

#         # 3. Add to internal 24k buffer
#         if self.decode_buffer_24k.shape[0] > 0:
#             # Overlap-add logic
#             overlap_region = self.decode_buffer_24k
#             current_region = decoded_frame_24k[:self.overlap_samples_24k]
            
#             # Blend
#             fade_out = torch.linspace(1, 0, self.overlap_samples_24k, device=self.device)
#             fade_in = torch.linspace(0, 1, self.overlap_samples_24k, device=self.device)
#             blended = overlap_region * fade_out + current_region * fade_in
            
#             # New audio is the blended part + the rest of the hop
#             new_audio_24k = torch.cat([
#                 blended,
#                 decoded_frame_24k[self.overlap_samples_24k:self.hop_samples_24k]
#             ])
#         else:
#             # First frame, just take the hop part
#             new_audio_24k = decoded_frame_24k[:self.hop_samples_24k]
            
#         # 4. Save the new overlap buffer
#         self.decode_buffer_24k = decoded_frame_24k[self.hop_samples_24k:]

#         # 5. Resample the *new audio* back to 16kHz and add to output buffer
#         new_audio_16k = self.resampler_24_to_16(new_audio_24k)
#         self.output_buffer_16k = torch.cat([self.output_buffer_16k, new_audio_16k])
        
#         # 6. Return 16kHz chunks to the app
#         # --- *** CRITICAL CHANGE: Return 320 samples now *** ---
#         app_hop_samples = HOP_SIZE # 320 samples
#         if self.output_buffer_16k.shape[0] >= app_hop_samples:
#             output_chunk_16k = self.output_buffer_16k[:app_hop_samples]
#             self.output_buffer_16k = self.output_buffer_16k[app_hop_samples:]
#             return (output_chunk_16k.cpu().numpy() * 32767.0).astype(np.int16).tobytes()
#         else:
#             return None # Not enough 16k audio to play yet

#     @torch.no_grad()
#     def process_full(self, audio_16k_tensor):
#         """
#         Processes an entire 16kHz audio tensor for evaluation.
#         Returns a full 16kHz audio tensor.
#         """
#         # 1. Resample full audio to 24kHz
#         audio_24k = self.resampler_16_to_24(audio_16k_tensor.to(self.device))
        
#         # 2. Run the 24kHz model's built-in streaming processor
#         if audio_24k.dim() == 1:
#             audio_24k = audio_24k.unsqueeze(0)
#         if audio_24k.dim() == 2:
#             audio_24k = audio_24k.unsqueeze(1) # (B, C, L)
            
#         # process_streaming is a helper from *your* codec_model.py
#         reconstructed_24k = self.model_24k.process_streaming(
#             audio_24k, 
#             bandwidth=ENCODEC_BANDWIDTH
#         ).squeeze()
        
#         # 3. Resample the full reconstructed audio back to 16kHz
#         reconstructed_16k = self.resampler_24_to_16(reconstructed_24k.cpu())
        
#         return reconstructed_16k.squeeze().numpy()
# # --- END: New Fine-Tuned Encodec 24kHz Wrapper ---


# # --- START: New Opus Fine-Tuning Model Definition (from user script) ---
# class UltraLightEnhancer(nn.Module):
#     """Ultra-conservative model that barely touches the signal"""
#     def __init__(self):
#         super().__init__()
        
#         # Hardcode config from the training script
#         HIDDEN_SIZE = 32
#         KERNEL_SIZE = 3
#         INITIAL_ALPHA = 0.01
#         MAX_ALPHA = 0.1

#         self.enhance = nn.Sequential(
#             nn.Conv1d(1, HIDDEN_SIZE, kernel_size=KERNEL_SIZE, padding=KERNEL_SIZE//2),
#             nn.PReLU(HIDDEN_SIZE, init=0.01),
#             nn.Conv1d(HIDDEN_SIZE, HIDDEN_SIZE//2, kernel_size=KERNEL_SIZE, padding=KERNEL_SIZE//2),
#             nn.PReLU(HIDDEN_SIZE//2, init=0.01),
#             nn.Conv1d(HIDDEN_SIZE//2, 1, kernel_size=1),
#         )

#         for m in self.modules():
#             if isinstance(m, nn.Conv1d):
#                 nn.init.xavier_uniform_(m.weight, gain=0.01) # Very small initialization
#                 if m.bias is not None:
#                     nn.init.zeros_(m.bias)
        
#         self.alpha = nn.Parameter(torch.tensor(INITIAL_ALPHA))
#         self.MAX_ALPHA = MAX_ALPHA # Store max alpha

#     def forward(self, x):
#         identity = x
#         enhanced_residual = self.enhance(x)
        
#         # Very conservative mixing (max 10% enhancement)
#         alpha = torch.clamp(self.alpha, 0.0, self.MAX_ALPHA)
        
#         # Additive enhancement
#         output = (1 - alpha) * identity + alpha * enhanced_residual
        
#         return torch.clamp(output, -1.0, 1.0)
# # --- END: New Opus Fine-Tuning Model Definition ---


# # --- START: New Opus Fine-Tuned Codec Wrapper ---
# class OpusFineTunedCodec:
#     """
#     Wrapper for the Opus Fine-Tuned (post-filter) model.
    
#     - encode(): Uses standard Opus 12kbps encode.
#     - decode(): Uses standard Opus 12kbps decode, THEN runs the
#                 result through the trained enhancement network.
#     """
#     def __init__(self, model_path, bitrate_kbps=12):
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
#         if not OPUSLIB_AVAILABLE:
#             raise ImportError("opuslib not found. Please install it with 'pip install opuslib' and ensure libopus is installed.")
            
#         # 1. Load the underlying Opus codec
#         # It will inherit the 12kbps bitrate and 0 complexity
#         self.opus_codec = OpusLibStreamingCodec(bitrate_kbps=bitrate_kbps)
        
#         # 2. Load the trained enhancement model
#         try:
#             # Define the new model architecture from the user's script
#             self.enhancement_model = UltraLightEnhancer().to(self.device)
            
#             # Load the state dict
#             checkpoint = torch.load(model_path, map_location=self.device)
#             state_dict = checkpoint.get('model_state_dict', checkpoint)
#             self.enhancement_model.load_state_dict(state_dict)
#             self.enhancement_model.eval()
#             print(f"OpusFineTunedCodec: Enhancement model (UltraLightEnhancer) loaded from {model_path}")
#         except Exception as e:
#             print(f"ERROR loading Opus enhancement model from {model_path}: {e}")
#             raise
            
#     def encode(self, audio_tensor):
#         """
#         Encodes the audio using the standard Opus codec.
#         Input: audio_tensor (torch.Tensor, shape [1, 1, 320])
#         Output: encoded_bytes (bytes)
#         """
#         return self.opus_codec.encode(audio_tensor)

#     @torch.no_grad()
#     def decode(self, encoded_bytes):
#         """
#         Decodes the Opus bytes AND applies the enhancement filter.
#         Input: encoded_bytes (bytes)
#         Output: enhanced_audio_tensor (torch.Tensor, shape [1, 1, 320])
#         """
#         # 1. Decode the raw 12kbps Opus stream
#         # This gives the "degraded" audio
#         degraded_tensor = self.opus_codec.decode(encoded_bytes)
        
#         # 2. Pass the degraded audio through the enhancement model
#         # The model expects (B, C, L), which matches degraded_tensor
#         enhanced_tensor = self.enhancement_model(degraded_tensor.to(self.device))
        
#         # 3. Return the enhanced audio
#         return enhanced_tensor
# # --- END: New Opus Fine-Tuned Codec Wrapper ---


# # --- CUSTOM TINY VQ-CODEC COMPONENTS (Causal Architecture) ---
# class CausalConvBlock(nn.Module):
#     """Causal Conv1d block matching the architecture from the trainer script."""
#     def __init__(self, in_channels, out_channels, kernel_size, stride):
#         super().__init__()
#         self.padding_amount = kernel_size - 1
#         self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding=0)
#         self.norm = nn.GroupNorm(1, out_channels)
#         self.activation = nn.GELU() # Synchronized
#         self.stride = stride

#     def forward(self, x):
#         # Apply padding only to the left (causal)
#         x = F.pad(x, (self.padding_amount, 0), mode='constant', value=0)
#         x = self.activation(self.norm(self.conv(x)))
#         return x

# class OptimizedTransformerBlock(nn.Module):
#     """Optimized transformer block matching the exact 4-module FFN sequence of the trainer script."""
#     def __init__(self, dim, heads):
#         super().__init__()
#         self.attn = nn.MultiheadAttention(dim, heads, batch_first=True, dropout=0.1)
#         self.norm1 = nn.LayerNorm(dim)
#         self.norm2 = nn.LayerNorm(dim)
        
#         # FFN sequence must match the 4 layers in the checkpoint
#         self.ffn = nn.Sequential(
#             nn.Linear(dim, dim * 2),
#             nn.GELU(),
#             nn.Dropout(0.1),
#             nn.Linear(dim * 2, dim)
#         )

#     def forward(self, x):
#         B, C, T = x.shape
#         x_attn = x.transpose(1, 2)
        
#         # Causal mask ensures future tokens are not attended to
#         attn_mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1)
        
#         attn_output, _ = self.attn(x_attn, x_attn, x_attn, attn_mask=attn_mask, is_causal=False)
#         x_attn = self.norm1(x_attn + attn_output)
        
#         ffn_output = self.ffn(x_attn)
#         x_attn = self.norm2(x_attn + ffn_output)
        
#         return x_attn.transpose(1, 2)

# class ImprovedVectorQuantizer(nn.Module):
#     """
#     Vector Quantization layer, synchronized with the training script's class definition 
#     to correctly load EMA buffers.
#     """
#     def __init__(self, num_embeddings=CODEBOOK_SIZE, embedding_dim=LATENT_DIM, commitment_cost=COMMITMENT_COST):
#         super().__init__()
#         self.embedding_dim = embedding_dim
#         self.num_embeddings = num_embeddings
#         self.commitment_cost = commitment_cost
        
#         self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)
        
#         # Include EMA buffers/parameters for compatibility
#         self.register_buffer('ema_cluster_size', torch.zeros(num_embeddings))
#         self.register_buffer('ema_w', self.embedding.weight.data.clone())
#         self.ema_decay = 0.99


#     def forward(self, inputs):
#         # Flatten input (B, C, T) -> (B*T, C)
#         flat_input = inputs.transpose(1, 2).contiguous().view(-1, self.embedding_dim)
        
#         # Calculate distances
#         distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
#                       + torch.sum(self.embedding.weight**2, dim=1)
#                       - 2 * torch.matmul(flat_input, self.embedding.weight.t()))
            
#         # Find the nearest codebook vector index
#         encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        
#         # Create one-hot vectors
#         encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=inputs.device)
#         encodings.scatter_(1, encoding_indices, 1)
        
#         # Quantize vector
#         quantized = torch.matmul(encodings, self.embedding.weight).view(inputs.shape[0], inputs.shape[2], -1).transpose(1, 2)
        
#         # Apply STE
#         quantized = inputs + (quantized - inputs).detach()
        
#         # Loss logic is removed (only needed for training)
        
#         return quantized, encoding_indices

# class TinyTransformerCodec(nn.Module):
#     def __init__(self, latent_dim=LATENT_DIM, blocks=BLOCKS, heads=HEADS, sr=SR):
#         super().__init__()
#         self.latent_dim = latent_dim
#         self.sr = sr
#         self.downsampling_factor = DOWN_FACTOR
#         self.num_codebooks = NUM_CODEBOOKS

#         # --- Encoder ---
#         self.encoder_convs = nn.ModuleList()
#         in_c = CHANNELS
#         encoder_channels = []
#         for i in range(blocks):
#             # OPTIMIZED: Scaled channels based on new LATENT_DIM
#             out_c = min(latent_dim, 16 * (2**i)) # Start with 16, up to 64
#             encoder_channels.append(out_c)
#             stride = STRIDES[i]
#             self.encoder_convs.append(CausalConvBlock(in_c, out_c, KERNEL_SIZE, stride))
#             in_c = out_c
        
#         self.pre_quant = CausalConvBlock(in_c, LATENT_DIM * NUM_CODEBOOKS, KERNEL_SIZE, 1)

#         # --- Vector Quantization (Using the compatible class) ---
#         self.quantizers = nn.ModuleList([
#             ImprovedVectorQuantizer(CODEBOOK_SIZE, LATENT_DIM, commitment_cost=COMMITMENT_COST)
#             for _ in range(NUM_CODEBOOKS)
#         ])
        
#         # --- Transformer (Using the compatible block) ---
#         self.transformer = nn.Sequential(*[
#             OptimizedTransformerBlock(latent_dim * NUM_CODEBOOKS, heads)
#             for _ in range(TRANSFORMER_BLOCKS)
#         ])
#         self.post_transformer = nn.Conv1d(latent_dim * NUM_CODEBOOKS, latent_dim * NUM_CODEBOOKS, 1)

#         # --- Decoder ---
#         self.decoder_tconvs = nn.ModuleList()
#         self.skip_convs = nn.ModuleList()
        
#         in_c = latent_dim * NUM_CODEBOOKS
#         for i in range(blocks):
#             idx = blocks - 1 - i
#             stride = STRIDES[idx]
            
#             if idx > 0:
#                 out_c = encoder_channels[idx - 1]
#             else:
#                 out_c = 16 # Final channel before output
            
#             self.decoder_tconvs.append(
#                 nn.ConvTranspose1d(in_c, out_c, KERNEL_SIZE, stride, padding=KERNEL_SIZE//2)
#             )
            
#             if idx > 0:
#                 skip_in_channels = encoder_channels[idx - 1]
#                 # This skip_conv layer still needs to be defined to match checkpoint keys
#                 self.skip_convs.append(
#                     nn.Conv1d(out_c + skip_in_channels, out_c, kernel_size=1)
#                 )
#             in_c = out_c
        
#         self.post_decoder_final = nn.Conv1d(in_c, CHANNELS, 1)

#     @classmethod
#     def load_model(cls, model_path):
#         """Loads the model weights and returns the initialized model."""
#         model = cls()
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
#         if not os.path.exists(model_path):
#             raise FileNotFoundError(f"Trained model not found at path: {model_path}")
            
#         try:
#             checkpoint = torch.load(model_path, map_location=device)
#             state_dict = checkpoint.get('model_state_dict', checkpoint)
#             model.load_state_dict(state_dict)
#             model.to(device)
#             model.eval()
#             print(f"TinyTransformerCodec (VQ-Codec) loaded successfully from {model_path}.")
#             return model
#         except Exception as e:
#             print(f"Error loading model state dict: {e}")
#             raise

#     def encode(self, x):
#         """
#         Encodes audio into quantized latent codes/indices.
#         """
#         x = x.view(x.size(0), CHANNELS, -1)
#         input_length = x.shape[-1]
        
#         encoder_outputs = []
        
#         # Encoder
#         for layer in self.encoder_convs:
#             x = layer(x)
#             encoder_outputs.append(x)
        
#         # Pre-quantization
#         z_e = self.pre_quant(x)
        
#         # Vector Quantization
#         z_q_list = []
#         indices_list = []
#         z_e_split = z_e.chunk(self.num_codebooks, dim=1)
        
#         for i in range(self.num_codebooks):
#             z_q, indices = self.quantizers[i](z_e_split[i]) # VQ layer returns quantized and indices
#             z_q_list.append(z_q)
#             indices_list.append(indices)
        
#         # Concatenated quantized latent features
#         quantized_codes = torch.cat(z_q_list, dim=1)
        
#         # Pass through transformer
#         codes = self.transformer(quantized_codes)
#         codes = self.post_transformer(codes)
        
#         return codes, indices_list, input_length, encoder_outputs

#     def decode(self, codes_or_indices_list, input_length=None, encoder_outputs=None):
#         """
#         Decodes from VQ-quantized float codes (for evaluation) OR integer indices (for streaming).
#         """
        
#         if isinstance(codes_or_indices_list, list):
#             # Case 1: Decoding from raw integer indices (streaming receiver)
#             indices_list = codes_or_indices_list
#             z_q_list = []
            
#             for i, indices in enumerate(indices_list):
#                 # indices is (T_latent, 1). Squeeze to (T_latent) for embedding lookup
#                 quantized = self.quantizers[i].embedding(indices.squeeze(1))
                
#                 # Add batch dim: (T_latent, C_latent) -> (1, T_latent, C_latent)
#                 quantized = quantized.unsqueeze(0)
#                 # Transpose to (B, C, T): (1, T_latent, C_latent) -> (1, C_latent, T_latent)
#                 quantized = quantized.transpose(1, 2)
                
#                 z_q_list.append(quantized)
            
#             x_pre_transformer = torch.cat(z_q_list, dim=1)
            
#             # Streaming path must also pass through transformer
#             x = self.transformer(x_pre_transformer)
#             x = self.post_transformer(x)
            
#             # In streaming, we have no skip connections
#             encoder_outputs = None 
            
#         elif isinstance(codes_or_indices_list, torch.Tensor):
#             # Case 2: Decoding from the quantized float tensor (evaluation)
#             # This tensor 'codes' is already post-transformer from the encode() step.
#             x = codes_or_indices_list
#             # 'encoder_outputs' is passed from the function arguments
#         else:
#             raise ValueError("Decoding input must be a torch.Tensor (quantized codes) or a list of Tensors (indices).")

#         # Decoder
#         for i, tconv in enumerate(self.decoder_tconvs):
#             x = F.gelu(tconv(x)) # Synchronized activation
            
#             # --- SKIP CONNECTION LOGIC (Intentionally disabled to match training) ---
#             #
#             # if encoder_outputs and i < len(self.skip_convs):
#             # ...
#             # --- END SKIP LOGIC ---
        
#         # Final output
#         x = torch.tanh(self.post_decoder_final(x))
        
#         # Match input length
#         if input_length is not None:
#             if x.shape[-1] > input_length:
#                 x = x[..., :input_length]
#             elif x.shape[-1] < input_length:
#                 x = F.pad(x, (0, input_length - x.shape[-1]))
        
#         return x.view(x.size(0), CHANNELS, -1)


# # --- TRADITIONAL CODECS (Unchanged) ---

# class MuLawCodec:
#     """μ-law codec for baseline comparison"""
#     def __init__(self, quantization_channels=256):
#         self.mu = float(quantization_channels - 1)
    
#     def encode(self, x):
#         mu_t = torch.tensor(self.mu, dtype=torch.float32, device=x.device)
#         encoded = torch.sign(x) * torch.log1p(mu_t * torch.abs(x)) / torch.log1p(mu_t)
#         return (((encoded + 1) / 2 * self.mu) + 0.5).to(torch.uint8)
    
#     def decode(self, z):
#         z_float = z.to(torch.float32)
#         mu_t = torch.tensor(self.mu, dtype=torch.float32, device=z.device)
#         y = (z_float / self.mu) * 2.0 - 1.0
#         return (torch.sign(y) * (1.0 / self.mu) * (torch.pow(1.0 + self.mu, torch.abs(y)) - 1.0)).unsqueeze(1)

# class ALawCodec:
#     """A-law codec for baseline comparison"""
#     def __init__(self):
#         self.A = 87.6
    
#     def encode(self, x):
#         a_t = torch.tensor(self.A, dtype=torch.float32, device=x.device)
#         abs_x = torch.abs(x)
#         encoded = torch.zeros_like(x)
#         cond = abs_x < (1 / self.A)
#         encoded[cond] = torch.sign(x[cond]) * (a_t * abs_x[cond]) / (1 + torch.log(a_t))
#         encoded[~cond] = torch.sign(x[~cond]) * (1 + torch.log(a_t * abs_x[~cond])) / (1 + torch.log(a_t))
#         return (((encoded + 1) / 2 * 255) + 0.5).to(torch.uint8)
    
#     def decode(self, z):
#         z_float = z.to(torch.float32)
#         a_t = torch.tensor(self.A, dtype=torch.float32, device=z.device)
#         y = (z_float / 127.5) - 1.0
#         abs_y = torch.abs(y)
#         decoded = torch.zeros_like(y)
#         cond = abs_y < (1 / (1 + torch.log(a_t)))
#         decoded[cond] = torch.sign(y[cond]) * (abs_y[cond] * (1 + torch.log(a_t))) / a_t
#         decoded[~cond] = torch.sign(y[~cond]) * torch.exp(abs_y[~cond] * (1 + torch.log(a_t)) - 1) / a_t
#         return decoded.unsqueeze(1)

# class AMRWBCodec:
#     """Simulated AMR-WB (12.65 kbps mode, 20ms frame) using Mu-Law for quantization."""
#     def __init__(self, mode=6): # Use mode 6 (12.65 kbps)
#         self.mu_codec = MuLawCodec()
#         self.sample_rate = 16000
#         # --- **** CRITICAL CHANGE **** ---
#         self.frame_size = 320 # 20ms at 16kHz
#         # --- **** END CRITICAL CHANGE **** ---
#         self.codec_bitrate = 12.65
        
#     def encode(self, x):
#         return self.mu_codec.encode(x)
    
#     def decode(self, z):
#         return self.mu_codec.decode(z)

# # --- DAC CODEC (MODIFIED) ---
# class DACCodec:
#     """Wrapper for Descript Audio Codec (DAC)"""
#     def __init__(self, model_path=None, model_type="16khz"):
#         if not DAC_AVAILABLE:
#             raise ImportError("DAC is not installed. Install with: pip install descript-audio-codec")
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         self.model_type = model_type
#         print(f"Loading DAC {model_type} model...")
#         try:
#             model_path = dac.utils.download(model_type=model_type)
#             self.model = dac.DAC.load(model_path)
#             print(f"DAC model loaded successfully")
#         except Exception as e:
#             print(f"Error loading DAC model: {e}")
#             raise
#         self.model.to(self.device)
#         self.model.eval()
#         self.sample_rate = 16000 if "16khz" in model_type else 44100
        
#         # --- **** CRITICAL CHANGE **** ---
#         # This model's hop size IS 320! It works perfectly now.
#         self.hop_size = 320 # 20ms @ 16kHz
#         self.chunk_size = 320 # Process 20ms chunks
#         # --- **** END CRITICAL CHANGE **** ---
        
#     def encode(self, audio_tensor):
#         with torch.no_grad():
#             if audio_tensor.dim() == 2: audio_tensor = audio_tensor.unsqueeze(1)
#             audio_tensor = audio_tensor.to(self.device, dtype=torch.float32)
#             original_length = audio_tensor.shape[-1]
#             if original_length % self.hop_size != 0:
#                 pad_length = self.hop_size - (original_length % self.hop_size)
#                 audio_tensor = F.pad(audio_tensor, (0, pad_length))
#             _, codes, _, _, _ = self.model.encode(audio_tensor)
#             return codes, original_length # Returns the integer codes tensor and original length
    
#     def decode(self, codes, original_length=None):
#         with torch.no_grad():
#             if not isinstance(codes, torch.Tensor): codes = torch.tensor(codes, dtype=torch.long)
#             codes = codes.to(self.device)
#             z = self.model.quantizer.from_codes(codes)[0]
#             audio_recon = self.model.decode(z)
#             if original_length is not None and audio_recon.shape[-1] > original_length:
#                 audio_recon = audio_recon[..., :original_length]
#             return audio_recon


# # --- UTILITY FUNCTIONS (Unchanged) ---
# def get_model_cache_info():
#     """Returns information about cached models"""
#     cache_info = {}
#     if os.path.exists(MODELS_CACHE_DIR):
#         for file in os.listdir(MODELS_CACHE_DIR):
#             file_path = os.path.join(MODELS_CACHE_DIR, file)
#             if os.path.isfile(file_path):
#                 file_size = os.path.getsize(file_path) / (1024 * 1024)  # Size in MB
#                 cache_info[file] = f"{file_size:.2f} MB"
#     return cache_info

# def clear_model_cache():
#     """Clears all cached models"""
#     import shutil
#     if os.path.exists(MODELS_CACHE_DIR):
#         shutil.rmtree(MODELS_CACHE_DIR)
#         os.makedirs(MODELS_CACHE_DIR)
#         print("Model cache cleared.")






import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import os
import struct
import subprocess
import sys
import math

# --- START: New Imports ---
import torchaudio
import torchaudio.transforms as T
from codec_model import (
    load_codec_model as load_encodec_model_24k,
    RealTimeEncodecWrapper as EncodecStreamingWrapper,
    TARGET_SR as ENCODEC_SR_24K,
    TARGET_BANDWIDTH as ENCODEC_BANDWIDTH,
    STREAMING_FRAME_SAMPLES as ENCODEC_FRAME_SAMPLES,
    OVERLAP_SAMPLES as ENCODEC_OVERLAP_SAMPLES
)
# --- END: New Imports ---

# --- START: OpusLib Import ---
try:
    import opuslib
    OPUSLIB_AVAILABLE = True
except (ImportError, OSError) as e:
    OPUSLIB_AVAILABLE = False
    print(f"WARNING: opuslib not installed or libopus not found (Error: {e}). Opus (opuslib) codec will be unavailable.")
# --- END: OpusLib Import ---


# --- Global Configuration (Synchronized with ~9.3kbps, 15ms LATENCY Trainer) ---
SR = 16000
CHANNELS = 1
LATENT_DIM = 64  
BLOCKS = 4
HEADS = 8        
KERNEL_SIZE = 3

# Synchronized: 2 * 2 * 2 * 3 = 24x downsampling (Unchanged)
STRIDES = [2, 2, 2, 3] 
DOWN_FACTOR = np.prod(STRIDES) # 24

# Synchronized: 2 Codebooks (128 entries each)
NUM_CODEBOOKS = 2
CODEBOOK_SIZE = 128 
COMMITMENT_COST = 0.25  
TRANSFORMER_BLOCKS = 4  

# --- **** CRITICAL CHANGE **** ---
# HOP size changed to 20ms (320 samples) to support Opus
# This is now the app's fundamental chunk size.
HOP_SIZE = int(SR / (1000 / 20)) # 320 samples (20ms)
# --- **** END CRITICAL CHANGE **** ---

# Models cache directory
MODELS_CACHE_DIR = os.path.join(os.path.expanduser("~"), ".audio_codec_models")
if not os.path.exists(MODELS_CACHE_DIR):
    os.makedirs(MODELS_CACHE_DIR)

# --- DAC INTEGRATION (Unchanged) ---
try:
    import dac
    DAC_AVAILABLE = True
except ImportError:
    DAC_AVAILABLE = False


# --- START: New OpusLib Streaming Codec ---
class OpusLibStreamingCodec:
    """
    A stateless streaming Opus codec using 'opuslib'.
    It's designed to process 20ms (320-sample) chunks,
    matching the application's streaming buffer.
    """
    def __init__(self, bitrate_kbps=12):
        if not OPUSLIB_AVAILABLE:
            raise ImportError("opuslib not found. Please install it with 'pip install opuslib' and ensure libopus is installed.")
        
        self.sample_rate = SR
        self.channels = CHANNELS
        self.bitrate_bps = int(bitrate_kbps * 1000)
        
        # App's frame size is 20ms (320 samples)
        self.samples_per_frame = HOP_SIZE # 320
        
        try:
            # --- Encoder ---
            self._opus_encoder = opuslib.Encoder(
                self.sample_rate, 
                self.channels, 
                opuslib.APPLICATION_VOIP
            )
            self._opus_encoder.bitrate = self.bitrate_bps
            # Set complexity to 0 (matches training script)
            self._opus_encoder.complexity = 0 
            
            # --- Decoder ---
            self._opus_decoder = opuslib.Decoder(
                self.sample_rate, 
                self.channels
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize opuslib Encoder/Decoder: {e}")
            
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def encode(self, audio_tensor):
        """
        Encodes a float tensor chunk into Opus bytes.
        Input: audio_tensor (torch.Tensor, shape [1, 1, 320])
        Output: encoded_bytes (bytes)
        """
        # 1. Convert float tensor [-1.0, 1.0] to int16 bytes
        pcm_int16 = (audio_tensor.squeeze().cpu().numpy() * 32767.0).astype(np.int16)
        
        # Ensure correct frame size
        if len(pcm_int16) != self.samples_per_frame:
             pcm_int16 = np.pad(
                 pcm_int16, 
                 (0, self.samples_per_frame - len(pcm_int16)), 
                 'constant'
             )[:self.samples_per_frame]
             
        pcm_bytes = pcm_int16.tobytes()
        
        # 2. Encode the bytes
        try:
            encoded_bytes = self._opus_encoder.encode(pcm_bytes, self.samples_per_frame)
            return encoded_bytes
        except Exception as e:
            print(f"OpusLib encode error: {e}")
            return b''

    def decode(self, encoded_bytes):
        """
        Decodes Opus bytes into a float tensor chunk.
        Input: encoded_bytes (bytes)
        Output: audio_tensor (torch.Tensor, shape [1, 1, 320])
        """
        try:
            # 1. Decode bytes
            decoded_bytes = self._opus_decoder.decode(encoded_bytes, self.samples_per_frame)
        except Exception as e:
            # Packet loss or corruption
            decoded_bytes = self._opus_decoder.decode_missing_packet(self.samples_per_frame)
            
        # 2. Convert int16 bytes back to float tensor
        decoded_np = np.frombuffer(decoded_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        
        # 3. Add batch/channel dims and return
        return torch.from_numpy(decoded_np).unsqueeze(0).unsqueeze(0).to(self.device)
# --- END: New OpusLib Streaming Codec ---


# --- START: New Fine-Tuned Encodec 24kHz Wrapper ---
class FineTunedEncodecWrapper:
    """
    Wraps the 24kHz Fine-Tuned Encodec model to make it compatible
    with the 16kHz streaming and evaluation pipeline of the app.
    
    It handles all necessary resampling and buffering.
    """
    def __init__(self, model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 1. Load the 24kHz model from codec_model.py
        self.model_24k = load_encodec_model_24k(model_path)
        self.model_24k.to(self.device)
        self.model_24k.eval()
        
        self.target_sr = SR # App's sample rate (16kHz)
        self.model_sr = ENCODEC_SR_24K # Model's sample rate (24kHz)
        
        # 2. Create resamplers
        self.resampler_16_to_24 = T.Resample(self.target_sr, self.model_sr).to(self.device)
        self.resampler_24_to_16 = T.Resample(self.model_sr, self.target_sr).to(self.device)
        
        # 3. Define model's frame parameters (at 24kHz)
        self.frame_samples_24k = ENCODEC_FRAME_SAMPLES
        self.hop_samples_24k = ENCODEC_FRAME_SAMPLES - ENCODEC_OVERLAP_SAMPLES
        self.overlap_samples_24k = ENCODEC_OVERLAP_SAMPLES
        
        # 4. Internal buffers for streaming
        # These buffers will hold 24kHz audio
        self.encode_buffer_24k = torch.tensor([], device=self.device)
        self.decode_buffer_24k = torch.tensor([], device=self.device)
        
        # This buffer holds 16kHz decoded audio
        self.output_buffer_16k = torch.tensor([], device=self.device)

    @torch.no_grad()
    def encode(self, audio_chunk_16k_tensor):
        """
        Accepts a 16kHz audio chunk (NOW 320 SAMPLES), adds it to a buffer, 
        and processes it if the buffer is full.
        Returns bytes if a frame was processed, else None.
        """
        # 1. Resample incoming 16kHz chunk to 24kHz and add to buffer
        audio_chunk_24k = self.resampler_16_to_24(audio_chunk_16k_tensor.to(self.device))
        self.encode_buffer_24k = torch.cat([self.encode_buffer_24k, audio_chunk_24k.squeeze(0)])

        # 2. Check if we have enough data to process at least one frame
        if self.encode_buffer_24k.shape[0] < self.frame_samples_24k:
            return None # Not enough data yet

        # 3. Extract the frame to process
        frame_to_process = self.encode_buffer_24k[:self.frame_samples_24k].unsqueeze(0).unsqueeze(0)
        
        # 4. Update the buffer (Overlap-Add)
        # Keep the last 'overlap' samples for the next frame
        self.encode_buffer_24k = self.encode_buffer_24k[self.hop_samples_24k:]
        
        # 5. Run the model's encode function
        # We need to get the *raw codes* (indices), not the float tensor
        encoded_output = self.model_24k.encodec.encode(
            frame_to_process, 
            bandwidth=ENCODEC_BANDWIDTH
        )
        
        # (B, n_codebooks, T_latent)
        codes = encoded_output.audio_codes 
        codes_np = codes.cpu().numpy().astype(np.int32)
        
        # 6. Pack and return bytes
        shape = codes_np.shape
        header = struct.pack('I', len(shape))
        for s in shape:
            header += struct.pack('I', s)
        payload = header + codes_np.tobytes()
        return payload

    @torch.no_grad()
    def decode(self, payload_bytes):
        """
        Accepts the byte payload, decodes it to 24kHz, resamples to 16kHz,
        and returns a 16kHz audio chunk.
        """
        # 1. Unpack bytes
        offset = 0
        num_dims = struct.unpack('I', payload_bytes[offset:offset + 4])[0]
        offset += 4
        shape = []
        for i in range(num_dims):
            shape.append(struct.unpack('I', payload_bytes[offset + 4 * i:offset + 4 * (i + 1)])[0])
        offset += 4 * num_dims
        
        codes_bytes = payload_bytes[offset:]
        codes_np = np.frombuffer(codes_bytes, dtype=np.int32).reshape(shape)
        codes_tensor = torch.from_numpy(codes_np).to(self.device)
        
        # 2. Decode using the 24k model
        # We pass dummy scales. The model wrapper's .decode handles it.
        dummy_scales = torch.ones(shape[0], 1, shape[2], device=self.device)
        decoded_output_24k = self.model_24k.encodec.decode(codes_tensor, [dummy_scales])
        decoded_frame_24k = decoded_output_24k.audio_values.squeeze() # (L_frame_24k)

        # 3. Add to internal 24k buffer
        if self.decode_buffer_24k.shape[0] > 0:
            # Overlap-add logic
            overlap_region = self.decode_buffer_24k
            current_region = decoded_frame_24k[:self.overlap_samples_24k]
            
            # Blend
            fade_out = torch.linspace(1, 0, self.overlap_samples_24k, device=self.device)
            fade_in = torch.linspace(0, 1, self.overlap_samples_24k, device=self.device)
            blended = overlap_region * fade_out + current_region * fade_in
            
            # New audio is the blended part + the rest of the hop
            new_audio_24k = torch.cat([
                blended,
                decoded_frame_24k[self.overlap_samples_24k:self.hop_samples_24k]
            ])
        else:
            # First frame, just take the hop part
            new_audio_24k = decoded_frame_24k[:self.hop_samples_24k]
            
        # 4. Save the new overlap buffer
        self.decode_buffer_24k = decoded_frame_24k[self.hop_samples_24k:]

        # 5. Resample the *new audio* back to 16kHz and add to output buffer
        new_audio_16k = self.resampler_24_to_16(new_audio_24k)
        self.output_buffer_16k = torch.cat([self.output_buffer_16k, new_audio_16k])
        
        # 6. Return 16kHz chunks to the app
        # --- *** CRITICAL CHANGE: Return 320 samples now *** ---
        app_hop_samples = HOP_SIZE # 320 samples
        if self.output_buffer_16k.shape[0] >= app_hop_samples:
            output_chunk_16k = self.output_buffer_16k[:app_hop_samples]
            self.output_buffer_16k = self.output_buffer_16k[app_hop_samples:]
            return (output_chunk_16k.cpu().numpy() * 32767.0).astype(np.int16).tobytes()
        else:
            return None # Not enough 16k audio to play yet

    @torch.no_grad()
    def process_full(self, audio_16k_tensor):
        """
        Processes an entire 16kHz audio tensor for evaluation.
        Returns a full 16kHz audio tensor.
        """
        # 1. Resample full audio to 24kHz
        audio_24k = self.resampler_16_to_24(audio_16k_tensor.to(self.device))
        
        # 2. Run the 24kHz model's built-in streaming processor
        if audio_24k.dim() == 1:
            audio_24k = audio_24k.unsqueeze(0)
        if audio_24k.dim() == 2:
            audio_24k = audio_24k.unsqueeze(1) # (B, C, L)
            
        # process_streaming is a helper from *your* codec_model.py
        reconstructed_24k = self.model_24k.process_streaming(
            audio_24k, 
            bandwidth=ENCODEC_BANDWIDTH
        ).squeeze()
        
        # 3. Resample the full reconstructed audio back to 16kHz
        reconstructed_16k = self.resampler_24_to_16(reconstructed_24k.cpu())
        
        return reconstructed_16k.squeeze().numpy()
# --- END: New Fine-Tuned Encodec 24kHz Wrapper ---


# --- START: New Opus Fine-Tuning Model Definition (from user script) ---
class UltraLightEnhancer(nn.Module):
    """Ultra-conservative model that barely touches the signal"""
    def __init__(self):
        super().__init__()
        
        # Hardcode config from the training script
        HIDDEN_SIZE = 32
        KERNEL_SIZE = 3
        INITIAL_ALPHA = 0.01
        MAX_ALPHA = 0.1

        self.enhance = nn.Sequential(
            nn.Conv1d(1, HIDDEN_SIZE, kernel_size=KERNEL_SIZE, padding=KERNEL_SIZE//2),
            nn.PReLU(HIDDEN_SIZE, init=0.01),
            nn.Conv1d(HIDDEN_SIZE, HIDDEN_SIZE//2, kernel_size=KERNEL_SIZE, padding=KERNEL_SIZE//2),
            nn.PReLU(HIDDEN_SIZE//2, init=0.01),
            nn.Conv1d(HIDDEN_SIZE//2, 1, kernel_size=1),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.xavier_uniform_(m.weight, gain=0.01) # Very small initialization
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
        self.alpha = nn.Parameter(torch.tensor(INITIAL_ALPHA))
        self.MAX_ALPHA = MAX_ALPHA # Store max alpha

    def forward(self, x):
        identity = x
        enhanced_residual = self.enhance(x)
        
        # Very conservative mixing (max 10% enhancement)
        alpha = torch.clamp(self.alpha, 0.0, self.MAX_ALPHA)
        
        # Additive enhancement
        output = (1 - alpha) * identity + alpha * enhanced_residual
        
        return torch.clamp(output, -1.0, 1.0)
# --- END: New Opus Fine-Tuning Model Definition ---


# --- START: New Opus Fine-Tuned Codec Wrapper ---
class OpusFineTunedCodec:
    """
    Wrapper for the Opus Fine-Tuned (post-filter) model.
    
    - encode(): Uses standard Opus 12kbps encode.
    - decode(): Uses standard Opus 12kbps decode, THEN runs the
                result through the trained enhancement network.
    """
    def __init__(self, model_path, bitrate_kbps=12):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if not OPUSLIB_AVAILABLE:
            raise ImportError("opuslib not found. Please install it with 'pip install opuslib' and ensure libopus is installed.")
            
        # 1. Load the underlying Opus codec
        # It will inherit the 12kbps bitrate and 0 complexity
        self.opus_codec = OpusLibStreamingCodec(bitrate_kbps=bitrate_kbps)
        
        # 2. Load the trained enhancement model
        try:
            # Define the new model architecture from the user's script
            self.enhancement_model = UltraLightEnhancer().to(self.device)
            
            # Load the state dict
            checkpoint = torch.load(model_path, map_location=self.device)
            state_dict = checkpoint.get('model_state_dict', checkpoint)
            self.enhancement_model.load_state_dict(state_dict)
            self.enhancement_model.eval()
            print(f"OpusFineTunedCodec: Enhancement model (UltraLightEnhancer) loaded from {model_path}")
        except Exception as e:
            print(f"ERROR loading Opus enhancement model from {model_path}: {e}")
            raise
            
    def encode(self, audio_tensor):
        """
        Encodes the audio using the standard Opus codec.
        Input: audio_tensor (torch.Tensor, shape [1, 1, 320])
        Output: encoded_bytes (bytes)
        """
        return self.opus_codec.encode(audio_tensor)

    @torch.no_grad()
    def decode(self, encoded_bytes):
        """
        Decodes the Opus bytes AND applies the enhancement filter.
        Input: encoded_bytes (bytes)
        Output: enhanced_audio_tensor (torch.Tensor, shape [1, 1, 320])
        """
        # 1. Decode the raw 12kbps Opus stream
        # This gives the "degraded" audio
        degraded_tensor = self.opus_codec.decode(encoded_bytes)
        
        # 2. Pass the degraded audio through the enhancement model
        # The model expects (B, C, L), which matches degraded_tensor
        enhanced_tensor = self.enhancement_model(degraded_tensor.to(self.device))
        
        # 3. Return the enhanced audio
        return enhanced_tensor
# --- END: New Opus Fine-Tuned Codec Wrapper ---


# --- CUSTOM TINY VQ-CODEC COMPONENTS (Causal Architecture) ---
class CausalConvBlock(nn.Module):
    """Causal Conv1d block matching the architecture from the trainer script."""
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        self.padding_amount = kernel_size - 1
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding=0)
        self.norm = nn.GroupNorm(1, out_channels)
        self.activation = nn.GELU() # Synchronized
        self.stride = stride

    def forward(self, x):
        # Apply padding only to the left (causal)
        x = F.pad(x, (self.padding_amount, 0), mode='constant', value=0)
        x = self.activation(self.norm(self.conv(x)))
        return x

class OptimizedTransformerBlock(nn.Module):
    """Optimized transformer block matching the exact 4-module FFN sequence of the trainer script."""
    def __init__(self, dim, heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True, dropout=0.1)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
        # FFN sequence must match the 4 layers in the checkpoint
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(dim * 2, dim)
        )

    def forward(self, x):
        B, C, T = x.shape
        x_attn = x.transpose(1, 2)
        
        # Causal mask ensures future tokens are not attended to
        attn_mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1)
        
        attn_output, _ = self.attn(x_attn, x_attn, x_attn, attn_mask=attn_mask, is_causal=False)
        x_attn = self.norm1(x_attn + attn_output)
        
        ffn_output = self.ffn(x_attn)
        x_attn = self.norm2(x_attn + ffn_output)
        
        return x_attn.transpose(1, 2)

class ImprovedVectorQuantizer(nn.Module):
    """
    Vector Quantization layer, synchronized with the training script's class definition 
    to correctly load EMA buffers.
    """
    def __init__(self, num_embeddings=CODEBOOK_SIZE, embedding_dim=LATENT_DIM, commitment_cost=COMMITMENT_COST):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost
        
        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)
        
        # Include EMA buffers/parameters for compatibility
        self.register_buffer('ema_cluster_size', torch.zeros(num_embeddings))
        self.register_buffer('ema_w', self.embedding.weight.data.clone())
        self.ema_decay = 0.99


    def forward(self, inputs):
        # Flatten input (B, C, T) -> (B*T, C)
        flat_input = inputs.transpose(1, 2).contiguous().view(-1, self.embedding_dim)
        
        # Calculate distances
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                      + torch.sum(self.embedding.weight**2, dim=1)
                      - 2 * torch.matmul(flat_input, self.embedding.weight.t()))
            
        # Find the nearest codebook vector index
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        
        # Create one-hot vectors
        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        # Quantize vector
        quantized = torch.matmul(encodings, self.embedding.weight).view(inputs.shape[0], inputs.shape[2], -1).transpose(1, 2)
        
        # Apply STE
        quantized = inputs + (quantized - inputs).detach()
        
        # Loss logic is removed (only needed for training)
        
        return quantized, encoding_indices

class TinyTransformerCodec(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM, blocks=BLOCKS, heads=HEADS, sr=SR):
        super().__init__()
        self.latent_dim = latent_dim
        self.sr = sr
        self.downsampling_factor = DOWN_FACTOR
        self.num_codebooks = NUM_CODEBOOKS

        # --- Encoder ---
        self.encoder_convs = nn.ModuleList()
        in_c = CHANNELS
        encoder_channels = []
        for i in range(blocks):
            # OPTIMIZED: Scaled channels based on new LATENT_DIM
            out_c = min(latent_dim, 16 * (2**i)) # Start with 16, up to 64
            encoder_channels.append(out_c)
            stride = STRIDES[i]
            self.encoder_convs.append(CausalConvBlock(in_c, out_c, KERNEL_SIZE, stride))
            in_c = out_c
        
        self.pre_quant = CausalConvBlock(in_c, LATENT_DIM * NUM_CODEBOOKS, KERNEL_SIZE, 1)

        # --- Vector Quantization (Using the compatible class) ---
        self.quantizers = nn.ModuleList([
            ImprovedVectorQuantizer(CODEBOOK_SIZE, LATENT_DIM, commitment_cost=COMMITMENT_COST)
            for _ in range(NUM_CODEBOOKS)
        ])
        
        # --- Transformer (Using the compatible block) ---
        self.transformer = nn.Sequential(*[
            OptimizedTransformerBlock(latent_dim * NUM_CODEBOOKS, heads)
            for _ in range(TRANSFORMER_BLOCKS)
        ])
        self.post_transformer = nn.Conv1d(latent_dim * NUM_CODEBOOKS, latent_dim * NUM_CODEBOOKS, 1)

        # --- Decoder ---
        self.decoder_tconvs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        
        in_c = latent_dim * NUM_CODEBOOKS
        for i in range(blocks):
            idx = blocks - 1 - i
            stride = STRIDES[idx]
            
            if idx > 0:
                out_c = encoder_channels[idx - 1]
            else:
                out_c = 16 # Final channel before output
            
            self.decoder_tconvs.append(
                nn.ConvTranspose1d(in_c, out_c, KERNEL_SIZE, stride, padding=KERNEL_SIZE//2)
            )
            
            if idx > 0:
                skip_in_channels = encoder_channels[idx - 1]
                # This skip_conv layer still needs to be defined to match checkpoint keys
                self.skip_convs.append(
                    nn.Conv1d(out_c + skip_in_channels, out_c, kernel_size=1)
                )
            in_c = out_c
        
        self.post_decoder_final = nn.Conv1d(in_c, CHANNELS, 1)

    @classmethod
    def load_model(cls, model_path):
        """Loads the model weights and returns the initialized model."""
        model = cls()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Trained model not found at path: {model_path}")
            
        try:
            checkpoint = torch.load(model_path, map_location=device)
            state_dict = checkpoint.get('model_state_dict', checkpoint)
            model.load_state_dict(state_dict)
            model.to(device)
            model.eval()
            print(f"TinyTransformerCodec (VQ-Codec) loaded successfully from {model_path}.")
            return model
        except Exception as e:
            print(f"Error loading model state dict: {e}")
            raise

    def encode(self, x):
        """
        Encodes audio into quantized latent codes/indices.
        """
        x = x.view(x.size(0), CHANNELS, -1)
        input_length = x.shape[-1]
        
        encoder_outputs = []
        
        # Encoder
        for layer in self.encoder_convs:
            x = layer(x)
            encoder_outputs.append(x)
        
        # Pre-quantization
        z_e = self.pre_quant(x)
        
        # Vector Quantization
        z_q_list = []
        indices_list = []
        z_e_split = z_e.chunk(self.num_codebooks, dim=1)
        
        for i in range(self.num_codebooks):
            z_q, indices = self.quantizers[i](z_e_split[i]) # VQ layer returns quantized and indices
            z_q_list.append(z_q)
            indices_list.append(indices)
        
        # Concatenated quantized latent features
        quantized_codes = torch.cat(z_q_list, dim=1)
        
        # Pass through transformer
        codes = self.transformer(quantized_codes)
        codes = self.post_transformer(codes)
        
        return codes, indices_list, input_length, encoder_outputs

    def decode(self, codes_or_indices_list, input_length=None, encoder_outputs=None):
        """
        Decodes from VQ-quantized float codes (for evaluation) OR integer indices (for streaming).
        """
        
        if isinstance(codes_or_indices_list, list):
            # Case 1: Decoding from raw integer indices (streaming receiver)
            indices_list = codes_or_indices_list
            z_q_list = []
            
            for i, indices in enumerate(indices_list):
                # indices is (T_latent, 1). Squeeze to (T_latent) for embedding lookup
                quantized = self.quantizers[i].embedding(indices.squeeze(1))
                
                # Add batch dim: (T_latent, C_latent) -> (1, T_latent, C_latent)
                quantized = quantized.unsqueeze(0)
                # Transpose to (B, C, T): (1, T_latent, C_latent) -> (1, C_latent, T_latent)
                quantized = quantized.transpose(1, 2)
                
                z_q_list.append(quantized)
            
            x_pre_transformer = torch.cat(z_q_list, dim=1)
            
            # Streaming path must also pass through transformer
            x = self.transformer(x_pre_transformer)
            x = self.post_transformer(x)
            
            # In streaming, we have no skip connections
            encoder_outputs = None 
            
        elif isinstance(codes_or_indices_list, torch.Tensor):
            # Case 2: Decoding from the quantized float tensor (evaluation)
            # This tensor 'codes' is already post-transformer from the encode() step.
            x = codes_or_indices_list
            # 'encoder_outputs' is passed from the function arguments
        else:
            raise ValueError("Decoding input must be a torch.Tensor (quantized codes) or a list of Tensors (indices).")

        # Decoder
        for i, tconv in enumerate(self.decoder_tconvs):
            x = F.gelu(tconv(x)) # Synchronized activation
            
            # --- SKIP CONNECTION LOGIC (Intentionally disabled to match training) ---
            #
            # if encoder_outputs and i < len(self.skip_convs):
            # ...
            # --- END SKIP LOGIC ---
        
        # Final output
        x = torch.tanh(self.post_decoder_final(x))
        
        # Match input length
        if input_length is not None:
            if x.shape[-1] > input_length:
                x = x[..., :input_length]
            elif x.shape[-1] < input_length:
                x = F.pad(x, (0, input_length - x.shape[-1]))
        
        return x.view(x.size(0), CHANNELS, -1)


# --- TRADITIONAL CODECS (Unchanged) ---

class MuLawCodec:
    """μ-law codec for baseline comparison"""
    def __init__(self, quantization_channels=256):
        self.mu = float(quantization_channels - 1)
    
    def encode(self, x):
        mu_t = torch.tensor(self.mu, dtype=torch.float32, device=x.device)
        encoded = torch.sign(x) * torch.log1p(mu_t * torch.abs(x)) / torch.log1p(mu_t)
        return (((encoded + 1) / 2 * self.mu) + 0.5).to(torch.uint8)
    
    def decode(self, z):
        z_float = z.to(torch.float32)
        mu_t = torch.tensor(self.mu, dtype=torch.float32, device=z.device)
        y = (z_float / self.mu) * 2.0 - 1.0
        return (torch.sign(y) * (1.0 / self.mu) * (torch.pow(1.0 + self.mu, torch.abs(y)) - 1.0)).unsqueeze(1)

class ALawCodec:
    """A-law codec for baseline comparison"""
    def __init__(self):
        self.A = 87.6
    
    def encode(self, x):
        a_t = torch.tensor(self.A, dtype=torch.float32, device=x.device)
        abs_x = torch.abs(x)
        encoded = torch.zeros_like(x)
        cond = abs_x < (1 / self.A)
        encoded[cond] = torch.sign(x[cond]) * (a_t * abs_x[cond]) / (1 + torch.log(a_t))
        encoded[~cond] = torch.sign(x[~cond]) * (1 + torch.log(a_t * abs_x[~cond])) / (1 + torch.log(a_t))
        return (((encoded + 1) / 2 * 255) + 0.5).to(torch.uint8)
    
    def decode(self, z):
        z_float = z.to(torch.float32)
        a_t = torch.tensor(self.A, dtype=torch.float32, device=z.device)
        y = (z_float / 127.5) - 1.0
        abs_y = torch.abs(y)
        decoded = torch.zeros_like(y)
        cond = abs_y < (1 / (1 + torch.log(a_t)))
        decoded[cond] = torch.sign(y[cond]) * (abs_y[cond] * (1 + torch.log(a_t))) / a_t
        decoded[~cond] = torch.sign(y[~cond]) * torch.exp(abs_y[~cond] * (1 + torch.log(a_t)) - 1) / a_t
        return decoded.unsqueeze(1)

class AMRWBCodec:
    """Simulated AMR-WB (12.65 kbps mode, 20ms frame) using Mu-Law for quantization."""
    def __init__(self, mode=6): # Use mode 6 (12.65 kbps)
        self.mu_codec = MuLawCodec()
        self.sample_rate = 16000
        # --- **** CRITICAL CHANGE **** ---
        self.frame_size = 320 # 20ms at 16kHz
        # --- **** END CRITICAL CHANGE **** ---
        self.codec_bitrate = 12.65
        
    def encode(self, x):
        return self.mu_codec.encode(x)
    
    def decode(self, z):
        return self.mu_codec.decode(z)

# --- DAC CODEC (MODIFIED) ---
class DACCodec:
    """Wrapper for Descript Audio Codec (DAC)"""
    def __init__(self, model_path=None, model_type="16khz"):
        if not DAC_AVAILABLE:
            raise ImportError("DAC is not installed. Install with: pip install descript-audio-codec")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_type = model_type
        print(f"Loading DAC {model_type} model...")
        try:
            model_path = dac.utils.download(model_type=model_type)
            self.model = dac.DAC.load(model_path)
            print(f"DAC model loaded successfully")
        except Exception as e:
            print(f"Error loading DAC model: {e}")
            raise
        self.model.to(self.device)
        self.model.eval()
        self.sample_rate = 16000 if "16khz" in model_type else 44100
        
        # --- **** CRITICAL CHANGE **** ---
        # This model's hop size IS 320! It works perfectly now.
        self.hop_size = 320 # 20ms @ 16kHz
        self.chunk_size = 320 # Process 20ms chunks
        # --- **** END CRITICAL CHANGE **** ---
        
    def encode(self, audio_tensor):
        with torch.no_grad():
            if audio_tensor.dim() == 2: audio_tensor = audio_tensor.unsqueeze(1)
            audio_tensor = audio_tensor.to(self.device, dtype=torch.float32)
            original_length = audio_tensor.shape[-1]
            if original_length % self.hop_size != 0:
                pad_length = self.hop_size - (original_length % self.hop_size)
                audio_tensor = F.pad(audio_tensor, (0, pad_length))
            _, codes, _, _, _ = self.model.encode(audio_tensor)
            return codes, original_length # Returns the integer codes tensor and original length
    
    def decode(self, codes, original_length=None):
        with torch.no_grad():
            if not isinstance(codes, torch.Tensor): codes = torch.tensor(codes, dtype=torch.long)
            codes = codes.to(self.device)
            z = self.model.quantizer.from_codes(codes)[0]
            audio_recon = self.model.decode(z)
            if original_length is not None and audio_recon.shape[-1] > original_length:
                audio_recon = audio_recon[..., :original_length]
            return audio_recon

    def process_full(self, audio_tensor):
        """Processes the full audio tensor in one pass (no manual chunking)."""
        with torch.no_grad():
            if audio_tensor.dim() == 2: audio_tensor = audio_tensor.unsqueeze(1)
            audio_tensor = audio_tensor.to(self.device, dtype=torch.float32)
            
            # Use DAC's internal preprocess to handle padding automatically
            x = self.model.preprocess(audio_tensor, self.sample_rate)
            
            # Encode
            z, codes, latents, _, _ = self.model.encode(x)
            
            # Decode
            audio_recon = self.model.decode(z)
            
            # Trim potential padding
            if audio_recon.shape[-1] > audio_tensor.shape[-1]:
                audio_recon = audio_recon[..., :audio_tensor.shape[-1]]
                
            return audio_recon


# --- UTILITY FUNCTIONS (Unchanged) ---
def get_model_cache_info():
    """Returns information about cached models"""
    cache_info = {}
    if os.path.exists(MODELS_CACHE_DIR):
        for file in os.listdir(MODELS_CACHE_DIR):
            file_path = os.path.join(MODELS_CACHE_DIR, file)
            if os.path.isfile(file_path):
                file_size = os.path.getsize(file_path) / (1024 * 1024)  # Size in MB
                cache_info[file] = f"{file_size:.2f} MB"
    return cache_info

def clear_model_cache():
    """Clears all cached models"""
    import shutil
    if os.path.exists(MODELS_CACHE_DIR):
        shutil.rmtree(MODELS_CACHE_DIR)
        os.makedirs(MODELS_CACHE_DIR)
        print("Model cache cleared.")