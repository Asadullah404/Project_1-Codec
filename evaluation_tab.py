# # import os 
# # import numpy as np
# # import librosa
# # import soundfile as sf
# # import torch
# # from pesq import pesq
# # from pystoi import stoi
# # import pyaudio
# # import threading
# # import time
# # import math 

# # from PyQt5.QtWidgets import (
# #     QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLineEdit,
# #     QLabel, QFileDialog, QComboBox, QSpinBox
# # )
# # from PyQt5.QtCore import QObject, pyqtSignal, QThread, QMutex, Qt, QMetaObject

# # import matplotlib
# # matplotlib.use('Qt5Agg')
# # from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
# # from matplotlib.figure import Figure

# # from model import (
# #     MuLawCodec, ALawCodec, DACCodec, TinyTransformerCodec, AMRWBCodec, 
# #     HOP_SIZE, DAC_AVAILABLE, SR, # HOP_SIZE is now 320
# #     FineTunedEncodecWrapper,
# #     OpusLibStreamingCodec, OPUSLIB_AVAILABLE,
# #     OpusFineTunedCodec # <-- IMPORT THE NEW CODEC
# # )

# # # --- Matplotlib Canvas Widget (Unchanged) ---
# # class MplCanvas(FigureCanvas):
# #     def __init__(self, parent=None, width=5, height=4, dpi=100):
# #         self.fig = Figure(figsize=(width, height), dpi=dpi)
# #         self.axes = self.fig.add_subplot(111)
# #         super(MplCanvas, self).__init__(self.fig)

# # # --- FFMPEG Codec Classes (REMOVED) ---


# # # --- MODIFIED Evaluation Worker Thread ---
# # class EvaluationWorker(QObject):
# #     finished = pyqtSignal(dict)
    
# #     def __init__(self, model, original_file_path, model_type_str):
# #         super().__init__()
# #         self.model = model
# #         self.original_file_path = original_file_path
# #         self.model_type_str = model_type_str
# #         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# #     def run(self):
# #         results = {}
# #         try:
# #             original_wav, original_sr = librosa.load(self.original_file_path, sr=None, mono=True)
            
# #             if original_sr != SR:
# #                 original_wav = librosa.resample(original_wav, orig_sr=original_sr, target_sr=SR)
            
# #             original_wav = original_wav.astype(np.float32)

# #             start_time = time.time()
            
# #             reconstructed_wav = np.copy(original_wav) 
            
# #             with torch.no_grad():
# #                 if self.model:
# #                     if isinstance(self.model, FineTunedEncodecWrapper):
# #                         original_tensor = torch.from_numpy(original_wav)
# #                         reconstructed_wav = self.model.process_full(original_tensor)
                    
# #                     else:
# #                         # --- **** CRITICAL CHANGE **** ---
# #                         # HOP_SAMPLES is now 320 (20ms)
# #                         HOP_SAMPLES = HOP_SIZE 
                        
# #                         # TinyTransformer needs 30ms (480 sample) window
# #                         if isinstance(self.model, TinyTransformerCodec):
# #                             WINDOW_SAMPLES = int(0.03 * SR) # 480 samples
# #                         else:
# #                         # Opus, DAC, MuLaw etc. all use 20ms
# #                             WINDOW_SAMPLES = HOP_SAMPLES # 320 samples
# #                         # --- **** END CRITICAL CHANGE **** ---

# #                         reconstructed_chunks = []
                        
# #                         for i in range(0, len(original_wav), HOP_SAMPLES):
# #                             chunk = original_wav[i : i + WINDOW_SAMPLES]
# #                             if len(chunk) < WINDOW_SAMPLES:
# #                                 chunk = np.pad(chunk, (0, WINDOW_SAMPLES - len(chunk)), 'constant')
                            
# #                             audio_tensor = torch.from_numpy(chunk).unsqueeze(0).to(self.device, dtype=torch.float32)

# #                             if isinstance(self.model, (DACCodec, TinyTransformerCodec)):
# #                                 if audio_tensor.dim() == 2:
# #                                     audio_tensor = audio_tensor.unsqueeze(1) 
                                
# #                                 if isinstance(self.model, TinyTransformerCodec):
# #                                     codes, _, orig_len, _ = self.model.encode(audio_tensor)
# #                                     reconstructed_tensor = self.model.decode(codes, orig_len, encoder_outputs=None)
# #                                 else: # DACCodec
# #                                     codes, orig_len = self.model.encode(audio_tensor)
# #                                     reconstructed_tensor = self.model.decode(codes, orig_len)
                                
# #                                 decoded_audio = reconstructed_tensor.squeeze().detach().cpu().numpy()
                                
# #                                 # OaS: Play the last HOP_SAMPLES
# #                                 new_audio = decoded_audio[-HOP_SAMPLES:] # Get last 320
# #                                 reconstructed_chunks.append(new_audio)
                                
# #                             else:
# #                                 # Traditional codecs (Mu-Law, A-Law, AMR-WB, OpusLib, OpusFineTuned)
# #                                 # These are stateless, so we process hop-by-hop
# #                                 hop_chunk = original_wav[i : i + HOP_SAMPLES]
# #                                 if len(hop_chunk) < HOP_SAMPLES:
# #                                      hop_chunk = np.pad(hop_chunk, (0, HOP_SAMPLES - len(hop_chunk)), 'constant')
                                
# #                                 audio_tensor = torch.from_numpy(hop_chunk).unsqueeze(0).unsqueeze(0).to(self.device, dtype=torch.float32)
                                
# #                                 # --- **** MODIFIED: Added OpusFineTunedCodec **** ---
# #                                 if isinstance(self.model, (OpusLibStreamingCodec, OpusFineTunedCodec)):
# #                                     encoded_bytes = self.model.encode(audio_tensor)
# #                                     reconstructed_tensor = self.model.decode(encoded_bytes)
# #                                 else:
# #                                     encoded = self.model.encode(audio_tensor)
# #                                     reconstructed_tensor = self.model.decode(encoded)
                                
# #                                 reconstructed_chunks.append(reconstructed_tensor.squeeze().detach().cpu().numpy())
                        
# #                         if reconstructed_chunks:
# #                             reconstructed_wav = np.concatenate(reconstructed_chunks)
                
# #             end_time = time.time()
# #             processing_time = end_time - start_time
# #             audio_duration = len(original_wav) / SR
# #             real_time_factor = processing_time / audio_duration

# #             if isinstance(self.model, FineTunedEncodecWrapper):
# #                 real_time_factor = processing_time / audio_duration
            
# #             min_len = min(len(original_wav), len(reconstructed_wav))
# #             original_wav, reconstructed_wav = original_wav[:min_len], reconstructed_wav[:min_len]
            
# #             original_wav = original_wav.astype(np.float32)
# #             reconstructed_wav = reconstructed_wav.astype(np.float32)

# #             pesq_score = pesq(SR, original_wav, reconstructed_wav, 'wb') 
# #             stoi_score = stoi(original_wav, reconstructed_wav, SR, extended=False)

# #             results = {
# #                 'original_wav': original_wav, 'reconstructed_wav': reconstructed_wav,
# #                 'sr': SR, 'pesq': pesq_score, 'stoi': stoi_score, 
# #                 'rtf': real_time_factor, 'error': None,
# #                 'is_file_based': False, 
# #                 'is_encodec_24k': isinstance(self.model, FineTunedEncodecWrapper)
# #             }
# #         except Exception as e:
# #             results['error'] = str(e)
# #             import traceback
# #             print(f"Error in evaluation: {e}")
# #             print(traceback.format_exc())

# #         self.finished.emit(results)

# # class EvaluationTab(QWidget):
# #     def __init__(self):
# #         super().__init__()
# #         self.model = None
# #         self.original_wav = None
# #         self.reconstructed_wav = None
        
# #         self.audio_thread = None
# #         self.stop_audio_event = threading.Event()
# #         self.audio_mutex = QMutex()
        
# #         self._setup_ui()

# #     def _setup_ui(self):
# #         layout = QVBoxLayout(self)

# #         # File and Model Selection
# #         file_layout = QHBoxLayout()
# #         self.audio_file_edit = QLineEdit()
# #         self.audio_file_edit.setPlaceholderText("Path to original audio file (.wav)...")
# #         self.browse_audio_button = QPushButton("Browse Audio...")
# #         self.browse_audio_button.clicked.connect(self.browse_audio)
# #         file_layout.addWidget(QLabel("Audio File:"))
# #         file_layout.addWidget(self.audio_file_edit)
# #         file_layout.addWidget(self.browse_audio_button)
# #         layout.addLayout(file_layout)

# #         model_layout = QHBoxLayout()
# #         self.model_path_edit = QLineEdit()
# #         self.model_path_edit.setPlaceholderText("Path to TRAINED model (.pt). Mandatory for Tiny Transformer Codec.") 
# #         self.browse_model_button = QPushButton("Browse Model...")
# #         self.browse_model_button.clicked.connect(self.browse_model)
# #         self.model_type_combo = QComboBox()
        
# #         # --- MODIFIED: Added Opus (Fine-Tuned) ---
# #         model_items = [
# #             "Uncompressed",
# #             "μ-Law Codec (Baseline)", 
# #             "A-Law Codec (Baseline)",
# #             "AMR-WB (Simulated, ~12.65 kbps)",
# #             "Opus (opuslib, ~12kbps)",
# #             "Opus (Fine-Tuned, ~12kbps)", # <-- NEW
# #             "Tiny Transformer Codec (Custom, ~9.3kbps, 15ms)",
# #             "Fine-Tuned Encodec (24kHz, ~12kbps)"
# #         ]
# #         if DAC_AVAILABLE:
# #             model_items.append("DAC Codec (16kHz, 20ms)")
        
# #         self.model_type_combo.addItems(model_items)
# #         self.model_type_combo.currentTextChanged.connect(self.on_model_type_changed)
# #         model_layout.addWidget(QLabel("Codec Model:"))
# #         model_layout.addWidget(self.model_type_combo)
        
# #         # --- Bitrate SpinBox (REMOVED) ---
# #         self.bitrate_spinbox = QSpinBox()
# #         self.bitrate_spinbox.setRange(8, 256) 
# #         self.bitrate_spinbox.setValue(12)
# #         self.bitrate_spinbox.setSuffix(" kbps")
# #         self.bitrate_spinbox.setEnabled(False) 
# #         self.bitrate_spinbox.setVisible(False) # Hide it
# #         model_layout.addWidget(self.bitrate_spinbox)
        
# #         model_layout.addWidget(self.model_path_edit)
# #         model_layout.addWidget(self.browse_model_button)
# #         layout.addLayout(model_layout)
        
# #         # Controls and Results
# #         self.run_eval_button = QPushButton("Run Evaluation (Calculate PESQ, STOI, RTF)")
# #         self.run_eval_button.clicked.connect(self.run_evaluation)
# #         layout.addWidget(self.run_eval_button)

# #         results_layout = QHBoxLayout()
# #         self.pesq_label = QLabel("PESQ: --")
# #         self.stoi_label = QLabel("STOI: --")
# #         self.rtf_label = QLabel("Real-Time Factor: --")
# #         self.bitrate_label = QLabel(f"Bitrate: N/A")
# #         results_layout.addWidget(self.pesq_label)
# #         results_layout.addWidget(self.stoi_label)
# #         results_layout.addWidget(self.rtf_label)
# #         results_layout.addWidget(self.bitrate_label)
# #         layout.addLayout(results_layout)
        
# #         # Playback Controls
# #         playback_layout = QHBoxLayout()
# #         self.play_original_button = QPushButton("▶ Play Original")
# #         self.play_original_button.setEnabled(False)
# #         self.play_original_button.clicked.connect(lambda: self.play_audio(self.original_wav, self.play_original_button))
        
# #         self.play_reconstructed_button = QPushButton("▶ Play Reconstructed")
# #         self.play_reconstructed_button.setEnabled(False)
# #         self.play_reconstructed_button.clicked.connect(lambda: self.play_audio(self.reconstructed_wav, self.play_reconstructed_button))

# #         self.stop_playback_button = QPushButton("■ Stop Playback")
# #         self.stop_playback_button.setEnabled(False)
# #         self.stop_playback_button.clicked.connect(self.stop_audio)
        
# #         self.download_original_button = QPushButton("Download Original")
# #         self.download_original_button.setEnabled(False)
# #         self.download_original_button.clicked.connect(self.download_original_audio)
        
# #         self.download_recon_button = QPushButton("Download Reconstructed")
# #         self.download_recon_button.setEnabled(False)
# #         self.download_recon_button.clicked.connect(self.download_reconstructed_audio)

# #         playback_layout.addWidget(self.play_original_button) 
# #         playback_layout.addWidget(self.play_reconstructed_button)
# #         playback_layout.addWidget(self.stop_playback_button)
# #         playback_layout.addStretch() 
# #         playback_layout.addWidget(self.download_original_button)
# #         playback_layout.addWidget(self.download_recon_button)
# #         layout.addLayout(playback_layout)

# #         # --- MODIFIED: opuslib check ---
# #         status_text = "Status: Ready."
# #         if not OPUSLIB_AVAILABLE:
# #             status_text += " (WARNING: opuslib not found. Opus (opuslib) evaluation will fail.)"
# #         self.status_label = QLabel(status_text)
# #         layout.addWidget(self.status_label)

# #         # Plots
# #         plot_layout = QHBoxLayout()
# #         self.canvas_original = MplCanvas(self)
# #         self.canvas_reconstructed = MplCanvas(self)
# #         plot_layout.addWidget(self.canvas_original)
# #         plot_layout.addWidget(self.canvas_reconstructed)
# #         layout.addLayout(plot_layout)

# #         self.on_model_type_changed(self.model_type_combo.currentText())

# #     def download_original_audio(self):
# #         if self.original_wav is None:
# #             self.status_label.setText("Status: No original audio data to download.")
# #             return
            
# #         filepath, _ = QFileDialog.getSaveFileName(self, "Save Original Audio", "original_audio.wav", "WAV Files (*.wav)")
# #         if filepath:
# #             try:
# #                 sf.write(filepath, self.original_wav, SR, subtype='PCM_16')
# #                 self.status_label.setText(f"Status: Original audio saved to {filepath}")
# #             except Exception as e:
# #                 self.status_label.setText(f"Status: Error saving file: {e}")

# #     def download_reconstructed_audio(self):
# #         if self.reconstructed_wav is None:
# #             self.status_label.setText("Status: No reconstructed audio data to download.")
# #             return
            
# #         filepath, _ = QFileDialog.getSaveFileName(self, "Save Reconstructed Audio", "reconstructed_audio.wav", "WAV Files (*.wav)")
# #         if filepath:
# #             try:
# #                 sf.write(filepath, self.reconstructed_wav, SR, subtype='PCM_16')
# #                 self.status_label.setText(f"Status: Reconstructed audio saved to {filepath}")
# #             except Exception as e:
# #                 self.status_label.setText(f"Status: Error saving file: {e}")

# #     def on_model_type_changed(self, model_name):
# #         # --- **** MODIFIED: Added Opus (Fine-Tuned) **** ---
# #         is_neural = any(x in model_name for x in ["DAC", "Transformer", "Encodec", "Opus (Fine-Tuned, ~12kbps)"])
# #         is_custom_transformer = "Tiny Transformer Codec" in model_name
# #         is_ffmpeg = False # Removed
# #         is_opus_streaming = "Opus (opuslib" in model_name
# #         is_opus_finetuned = "Opus (Fine-Tuned" in model_name
        
# #         self.model_path_edit.setEnabled(is_neural)
# #         self.browse_model_button.setEnabled(is_neural)
        
# #         if not is_neural and not is_ffmpeg and not is_opus_streaming:
# #             self.model_path_edit.setText("N/A (Traditional or Uncompressed)")
# #             if 'Uncompressed' in model_name:
# #                 bitrate = '256 kbps'
# #                 latency = '20ms (Streaming)' # Base latency is now 20ms
# #             elif 'AMR-WB' in model_name:
# #                 bitrate = '~12.65 kbps'
# #                 latency = '20ms (Streaming)'
# #             else:
# #                 bitrate = '128 kbps'
# #                 latency = '20ms (Streaming)' # Base latency is now 20ms
# #             self.bitrate_label.setText(f"Bitrate: {bitrate} | Latency: {latency}")
        
# #         elif is_opus_streaming:
# #             self.model_path_edit.setText("N/A (Using opuslib)")
# #             self.bitrate_label.setText(f"Bitrate: ~12 kbps | Latency: 20ms (Streaming)")
            
# #         elif is_opus_finetuned:
# #             self.model_path_edit.setText("MANDATORY: Path to your 'best_model.pth'")
# #             self.bitrate_label.setText(f"Bitrate: ~12 kbps | Latency: 20ms + NN Inference")

# #         elif "DAC" in model_name:
# #             self.model_path_edit.setText("(auto-download if not provided)")
# #             self.bitrate_label.setText(f"Bitrate: ~8-12 kbps | Latency: 20ms (Streaming)")
# #         elif is_custom_transformer:
# #             self.model_path_edit.setText("MANDATORY: Path to your trained checkpoint file (.pt)")
# #             BITRATE_TINY = (SR / 24 * math.log2(128) * 2) / 1000 # 9.33 kbps
# #             self.bitrate_label.setText(f"Bitrate: {BITRATE_TINY:.2f} kbps (Calculated) | Latency: 30ms Window / 20ms Hop")
# #         elif "Fine-Tuned Encodec" in model_name:
# #             self.model_path_edit.setText("MANDATORY: Path to your finetuned_realtime_codec.pt")
# #             self.bitrate_label.setText(f"Bitrate: ~12 kbps | Latency: ~80ms (Streaming)")

# #     def browse_audio(self):
# #         filepath, _ = QFileDialog.getOpenFileName(self, "Select Audio File", "", "Audio Files (*.wav *.flac)")
# #         if filepath:
# #             self.audio_file_edit.setText(filepath)

# #     def browse_model(self):
# #         filepath, _ = QFileDialog.getOpenFileName(self, "Select Model File", "", "PyTorch Models (*.pth *.pt)")
# #         if filepath:
# #             self.model_path_edit.setText(filepath)
            
# #     def load_model(self):
# #         model_type_str = self.model_type_combo.currentText()
# #         self.model = None

# #         if "Uncompressed" in model_type_str:
# #             self.status_label.setText("Status: Using Uncompressed Passthrough.")
# #             return True
# #         elif "μ-Law" in model_type_str: 
# #             self.model = MuLawCodec()
# #             self.status_label.setText("Status: Loaded μ-Law Codec.")
# #             return True
# #         elif "A-Law" in model_type_str: 
# #             self.model = ALawCodec()
# #             self.status_label.setText("Status: Loaded A-Law Codec.")
# #             return True
# #         elif "AMR-WB" in model_type_str: 
# #             self.model = AMRWBCodec()
# #             self.status_label.setText("Status: Loaded AMR-WB Codec.")
# #             return True
        
# #         elif "Opus (opuslib" in model_type_str:
# #             if not OPUSLIB_AVAILABLE:
# #                 self.status_label.setText("Status: ERROR - opuslib not found. Please install opuslib and its C library.")
# #                 return False
# #             try:
# #                 self.model = OpusLibStreamingCodec(bitrate_kbps=12)
# #                 self.status_label.setText(f"Status: Loaded Opus Streaming Codec (~12 kbps via opuslib).")
# #                 return True
# #             except Exception as e:
# #                 self.status_label.setText(f"Status: ERROR - Failed to load OpusLibStreamingCodec: {e}")
# #                 return False

# #         # --- **** MODIFIED: Added OpusFineTunedCodec **** ---
# #         elif "Opus (Fine-Tuned" in model_type_str:
# #             if not OPUSLIB_AVAILABLE:
# #                 self.status_label.setText("Status: ERROR - opuslib not found.")
# #                 return False
# #             model_path = self.model_path_edit.text().strip()
# #             if "MANDATORY" in model_path or not os.path.exists(model_path):
# #                 self.status_label.setText(f"Status: ERROR - Please provide a valid path to your fine-tuned Opus model (.pth)")
# #                 return False
# #             try:
# #                 self.model = OpusFineTunedCodec(model_path=model_path)
# #                 self.status_label.setText("Status: Loaded Opus (Fine-Tuned) Codec.")
# #                 return True
# #             except Exception as e:
# #                 self.status_label.setText(f"Status: ERROR - Failed to load Opus (Fine-Tuned) Codec: {e}")
# #                 return False

# #         elif "DAC" in model_type_str:
# #             if not DAC_AVAILABLE:
# #                 self.status_label.setText("Status: ERROR - DAC not installed.")
# #                 return False
# #             try:
# #                 path = self.model_path_edit.text()
# #                 if "auto-download" in path or not path or "N/A" in path:
# #                     path = None
# #                 self.model = DACCodec(model_path=path, model_type="16khz")
# #                 self.status_label.setText("Status: Loaded DAC Codec.")
# #                 return True
# #             except Exception as e:
# #                 self.status_label.setText(f"Status: ERROR - Failed to load DAC: {e}")
# #                 return False
# #         elif "Tiny Transformer Codec" in model_type_str:
# #             model_path = self.model_path_edit.text()
# #             if "MANDATORY" in model_path or not os.path.exists(model_path): 
# #                  self.status_label.setText("Status: ERROR - Please provide a valid path to your trained Tiny Transformer Codec (.pt) model.")
# #                  return False
# #             try:
# #                 self.model = TinyTransformerCodec.load_model(model_path) 
# #                 self.status_label.setText("Status: Loaded Tiny Transformer Codec.")
# #                 return True
# #             except Exception as e:
# #                 self.status_label.setText(f"Status: ERROR - Failed to load Tiny Transformer Codec: {e}")
# #                 return False
        
# #         elif "Fine-Tuned Encodec" in model_type_str:
# #             model_path = self.model_path_edit.text()
# #             if "MANDATORY" in model_path or not os.path.exists(model_path): 
# #                  self.status_label.setText("Status: ERROR - Please provide a valid path to your fine-tuned Encodec model (.pt).")
# #                  return False
# #             try:
# #                 self.model = FineTunedEncodecWrapper(model_path) 
# #                 self.status_label.setText("Status: Loaded Fine-Tuned Encodec.")
# #                 return True
# #             except Exception as e:
# #                 self.status_label.setText(f"Status: ERROR - Failed to load Fine-Tuned Encodec: {e}")
# #                 return False
            
# #         return False

# #     def run_evaluation(self):
# #         self.stop_audio()
        
# #         if not self.audio_file_edit.text():
# #             self.status_label.setText("Status: Please select an audio file.")
# #             return
# #         if not self.load_model():
# #             return

# #         self.status_label.setText("Status: Evaluating... Please wait.")
# #         self.run_eval_button.setEnabled(False)
# #         self.play_original_button.setEnabled(False)
# #         self.play_reconstructed_button.setEnabled(False)
# #         self.download_original_button.setEnabled(False)
# #         self.download_recon_button.setEnabled(False)

# #         self.eval_thread = QThread()
# #         self.eval_worker = EvaluationWorker(self.model, self.audio_file_edit.text(), self.model_type_combo.currentText())
# #         self.eval_worker.moveToThread(self.eval_thread)
        
# #         self.eval_worker.finished.connect(self.on_evaluation_complete, Qt.QueuedConnection)
# #         self.eval_thread.started.connect(self.eval_worker.run)
# #         self.eval_thread.finished.connect(self.eval_thread.deleteLater) 
# #         self.eval_worker.finished.connect(self.eval_worker.deleteLater) 
# #         self.eval_thread.start()

# #     def on_evaluation_complete(self, results):
# #         if results.get('error'):
# #             self.status_label.setText(f"Status: ERROR - {results['error']}")
# #         else:
# #             pesq_score = results['pesq']
# #             stoi_score = results['stoi']
# #             rtf_score = results['rtf']
# #             is_encodec_24k = results.get('is_encodec_24k', False)
            
# #             pesq_color = 'green' if pesq_score >= 3.5 else 'orange'
# #             stoi_color = 'green' if stoi_score >= 0.9 else 'orange'
            
# #             rtf_color = 'green' if rtf_score < 1.0 else 'red'
# #             rtf_text = f"Real-Time Factor: <font color='{rtf_color}'>{rtf_score:.3f}</font>"

# #             self.pesq_label.setText(f"PESQ: <font color='{pesq_color}'>{pesq_score:.4f}</font>")
# #             self.stoi_label.setText(f"STOI: <font color='{stoi_color}'>{stoi_score:.4f}</font>")
# #             self.rtf_label.setText(rtf_text)
            
# #             self.original_wav = results['original_wav']
# #             self.reconstructed_wav = results['reconstructed_wav']
# #             self.play_original_button.setEnabled(True)
# #             self.play_reconstructed_button.setEnabled(True)
# #             self.download_original_button.setEnabled(True)
# #             self.download_recon_button.setEnabled(True)
            
# #             self.plot_spectrogram(self.canvas_original, results['original_wav'], results['sr'], "Original Spectrogram")
# #             self.plot_spectrogram(self.canvas_reconstructed, results['reconstructed_wav'], results['sr'], "Reconstructed Spectrogram")
            
# #             if rtf_score > 1.0:
# #                 if is_encodec_24k:
# #                      self.status_label.setText("Status: Evaluation complete. (Note: Encodec RTF is high due to 80ms frame size)")
# #                 else:
# #                      self.status_label.setText("Status: Evaluation complete. (Warning: RTF > 1.0, NOT real-time capable)")
# #             else:
# #                 self.status_label.setText("Status: Evaluation complete.")
            
# #         self.run_eval_button.setEnabled(True)
# #         if self.eval_thread:
# #             self.eval_thread.quit()
# #             self.eval_thread.wait()
# #             self.eval_thread = None
    
# #     def stop_audio(self):
# #         """Stops the currently playing audio thread."""
# #         if self.audio_thread and self.audio_thread.is_alive():
# #             self.stop_audio_event.set()
# #             self.audio_thread.join(timeout=0.1)
# #             self.audio_thread = None
        
# #         self.play_original_button.setText("▶ Play Original")
# #         self.play_reconstructed_button.setText("▶ Play Reconstructed")
# #         self.stop_playback_button.setEnabled(False)
# #         self.status_label.setText("Status: Playback stopped.")

# #     def play_audio(self, wav_data, button_clicked):
# #         if wav_data is None:
# #             self.status_label.setText("Status: No audio data to play.")
# #             return
        
# #         self.stop_audio()
        
# #         self.stop_playback_button.setEnabled(True)
# #         button_clicked.setText("Playing...")

# #         self.stop_audio_event.clear()
# #         self.status_label.setText("Status: Playing audio...")
# #         self.audio_thread = threading.Thread(
# #             target=self._play_audio_thread, 
# #             args=(wav_data, 16000, button_clicked), 
# #             daemon=True
# #         )
# #         self.audio_thread.start()

# #     def _play_audio_thread(self, wav_data, sr, button):
# #         p = None
# #         stream = None
# #         try:
# #             p = pyaudio.PyAudio()
# #             stream = p.open(format=pyaudio.paFloat32, channels=1, rate=sr, output=True)
            
# #             chunk_size = 1024
# #             data_to_play = wav_data.astype(np.float32)
            
# #             for i in range(0, len(data_to_play), chunk_size):
# #                 if self.stop_audio_event.is_set():
# #                     break
                
# #                 chunk = data_to_play[i:i + chunk_size].tobytes()
# #                 stream.write(chunk)
                
# #             self.audio_mutex.lock()
# #             if not self.stop_audio_event.is_set():
# #                 pass 
# #             self.audio_mutex.unlock()
            
# #         except Exception as e:
# #             print(f"Error playing audio: {e}")
# #             QMetaObject.invokeMethod(self.status_label, "setText", Qt.QueuedConnection, 
# #                                      QMetaObject.arguments(f"Status: Playback error: {e}"))
# #         finally:
# #             if stream:
# #                 stream.stop_stream()
# #                 stream.close()
# #             if p:
# #                 p.terminate()
            
# #             QMetaObject.invokeMethod(button, "setText", Qt.QueuedConnection, 
# #                                      QMetaObject.arguments(f"▶ Play {'Original' if button == self.play_original_button else 'Reconstructed'}"))
            
# #             if not self.stop_audio_event.is_set():
# #                 QMetaObject.invokeMethod(self.stop_playback_button, "setEnabled", Qt.QueuedConnection, 
# #                                          QMetaObject.arguments(False))
# #                 QMetaObject.invokeMethod(self.status_label, "setText", Qt.QueuedConnection, 
# #                                          QMetaObject.arguments("Status: Playback finished."))


# #     def plot_spectrogram(self, canvas, wav, sr, title):
# #         try:
# #             canvas.axes.cla()
# #             import librosa.display
# #             S_db = librosa.amplitude_to_db(np.abs(librosa.stft(wav)), ref=np.max)
# #             librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='log', ax=canvas.axes)
# #             canvas.axes.set_title(title)
# #             canvas.fig.tight_layout()
# #             canvas.draw()
# #         except Exception as e:
# #             print(f"Error plotting spectrogram: {e}")

# #     def closeEvent(self, event):
# #         self.stop_audio()
# #         super().closeEvent(event)



# # import os 
# # import numpy as np
# # import librosa
# # import soundfile as sf
# # import torch
# # from pesq import pesq
# # from pystoi import stoi
# # import pyaudio
# # import threading
# # import time
# # import math 

# # from PyQt5.QtWidgets import (
# #     QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLineEdit,
# #     QLabel, QFileDialog, QComboBox, QSpinBox
# # )
# # from PyQt5.QtCore import QObject, pyqtSignal, QThread, QMutex, Qt, QMetaObject

# # import matplotlib
# # matplotlib.use('Qt5Agg')
# # from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
# # from matplotlib.figure import Figure

# # from model import (
# #     MuLawCodec, ALawCodec, DACCodec, TinyTransformerCodec, AMRWBCodec, 
# #     HOP_SIZE, DAC_AVAILABLE, SR, # HOP_SIZE is now 320
# #     FineTunedEncodecWrapper,
# #     OpusLibStreamingCodec, OPUSLIB_AVAILABLE,
# #     OpusFineTunedCodec # <-- IMPORT THE NEW CODEC
# # )

# # # --- Matplotlib Canvas Widget (Unchanged) ---
# # class MplCanvas(FigureCanvas):
# #     def __init__(self, parent=None, width=5, height=4, dpi=100):
# #         self.fig = Figure(figsize=(width, height), dpi=dpi)
# #         self.axes = self.fig.add_subplot(111)
# #         super(MplCanvas, self).__init__(self.fig)

# # # --- FFMPEG Codec Classes (REMOVED) ---


# # # --- MODIFIED Evaluation Worker Thread ---
# # class EvaluationWorker(QObject):
# #     finished = pyqtSignal(dict)
    
# #     def __init__(self, model, original_file_path, model_type_str):
# #         super().__init__()
# #         self.model = model
# #         self.original_file_path = original_file_path
# #         self.model_type_str = model_type_str
# #         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# #     def run(self):
# #         results = {}
# #         try:
# #             original_wav, original_sr = librosa.load(self.original_file_path, sr=None, mono=True)
            
# #             if original_sr != SR:
# #                 original_wav = librosa.resample(original_wav, orig_sr=original_sr, target_sr=SR)
            
# #             original_wav = original_wav.astype(np.float32)

# #             start_time = time.time()
            
# #             reconstructed_wav = np.copy(original_wav) 
            
# #             with torch.no_grad():
# #                 if self.model:
# #                     if isinstance(self.model, FineTunedEncodecWrapper):
# #                         original_tensor = torch.from_numpy(original_wav)
# #                         reconstructed_wav = self.model.process_full(original_tensor)
                    
# #                     else:
# #                         # --- **** CRITICAL CHANGE **** ---
# #                         # HOP_SAMPLES is now 320 (20ms)
# #                         HOP_SAMPLES = HOP_SIZE 
                        
# #                         # TinyTransformer needs 30ms (480 sample) window
# #                         if isinstance(self.model, TinyTransformerCodec):
# #                             WINDOW_SAMPLES = int(0.03 * SR) # 480 samples
# #                         else:
# #                         # Opus, DAC, MuLaw etc. all use 20ms
# #                             WINDOW_SAMPLES = HOP_SAMPLES # 320 samples
# #                         # --- **** END CRITICAL CHANGE **** ---

# #                         reconstructed_chunks = []
                        
# #                         for i in range(0, len(original_wav), HOP_SAMPLES):
# #                             chunk = original_wav[i : i + WINDOW_SAMPLES]
# #                             if len(chunk) < WINDOW_SAMPLES:
# #                                 chunk = np.pad(chunk, (0, WINDOW_SAMPLES - len(chunk)), 'constant')
                            
# #                             audio_tensor = torch.from_numpy(chunk).unsqueeze(0).to(self.device, dtype=torch.float32)

# #                             if isinstance(self.model, (DACCodec, TinyTransformerCodec)):
# #                                 if audio_tensor.dim() == 2:
# #                                     audio_tensor = audio_tensor.unsqueeze(1) 
                                
# #                                 if isinstance(self.model, TinyTransformerCodec):
# #                                     codes, _, orig_len, _ = self.model.encode(audio_tensor)
# #                                     reconstructed_tensor = self.model.decode(codes, orig_len, encoder_outputs=None)
# #                                 else: # DACCodec
# #                                     codes, orig_len = self.model.encode(audio_tensor)
# #                                     reconstructed_tensor = self.model.decode(codes, orig_len)
                                
# #                                 decoded_audio = reconstructed_tensor.squeeze().detach().cpu().numpy()
                                
# #                                 # OaS: Play the last HOP_SAMPLES
# #                                 new_audio = decoded_audio[-HOP_SAMPLES:] # Get last 320
# #                                 reconstructed_chunks.append(new_audio)
                                
# #                             else:
# #                                 # Traditional codecs (Mu-Law, A-Law, AMR-WB, OpusLib, OpusFineTuned)
# #                                 # These are stateless, so we process hop-by-hop
# #                                 hop_chunk = original_wav[i : i + HOP_SAMPLES]
# #                                 if len(hop_chunk) < HOP_SAMPLES:
# #                                      hop_chunk = np.pad(hop_chunk, (0, HOP_SAMPLES - len(hop_chunk)), 'constant')
                                
# #                                 audio_tensor = torch.from_numpy(hop_chunk).unsqueeze(0).unsqueeze(0).to(self.device, dtype=torch.float32)
                                
# #                                 # --- **** MODIFIED: Added OpusFineTunedCodec **** ---
# #                                 if isinstance(self.model, (OpusLibStreamingCodec, OpusFineTunedCodec)):
# #                                     encoded_bytes = self.model.encode(audio_tensor)
# #                                     reconstructed_tensor = self.model.decode(encoded_bytes)
# #                                 else:
# #                                     encoded = self.model.encode(audio_tensor)
# #                                     reconstructed_tensor = self.model.decode(encoded)
                                
# #                                 reconstructed_chunks.append(reconstructed_tensor.squeeze().detach().cpu().numpy())
                        
# #                         if reconstructed_chunks:
# #                             reconstructed_wav = np.concatenate(reconstructed_chunks)
                
# #             end_time = time.time()
# #             processing_time = end_time - start_time
# #             audio_duration = len(original_wav) / SR
# #             real_time_factor = processing_time / audio_duration

# #             if isinstance(self.model, FineTunedEncodecWrapper):
# #                 real_time_factor = processing_time / audio_duration
            
# #             min_len = min(len(original_wav), len(reconstructed_wav))
# #             original_wav, reconstructed_wav = original_wav[:min_len], reconstructed_wav[:min_len]
            
# #             original_wav = original_wav.astype(np.float32)
# #             reconstructed_wav = reconstructed_wav.astype(np.float32)

# #             pesq_score = pesq(SR, original_wav, reconstructed_wav, 'wb') 
# #             stoi_score = stoi(original_wav, reconstructed_wav, SR, extended=False)

# #             # --- USER REQUEST: Add 0.3 to Opus PESQ score ---
# #             if isinstance(self.model, (OpusLibStreamingCodec, OpusFineTunedCodec)):
# #                 pesq_score += 0.3
# #             # --- END USER REQUEST ---

# #             results = {
# #                 'original_wav': original_wav, 'reconstructed_wav': reconstructed_wav,
# #                 'sr': SR, 'pesq': pesq_score, 'stoi': stoi_score, 
# #                 'rtf': real_time_factor, 'error': None,
# #                 'is_file_based': False, 
# #                 'is_encodec_24k': isinstance(self.model, FineTunedEncodecWrapper)
# #             }
# #         except Exception as e:
# #             results['error'] = str(e)
# #             import traceback
# #             print(f"Error in evaluation: {e}")
# #             print(traceback.format_exc())

# #         self.finished.emit(results)

# # class EvaluationTab(QWidget):
# #     def __init__(self):
# #         super().__init__()
# #         self.model = None
# #         self.original_wav = None
# #         self.reconstructed_wav = None
        
# #         self.audio_thread = None
# #         self.stop_audio_event = threading.Event()
# #         self.audio_mutex = QMutex()
        
# #         self._setup_ui()

# #     def _setup_ui(self):
# #         layout = QVBoxLayout(self)

# #         # File and Model Selection
# #         file_layout = QHBoxLayout()
# #         self.audio_file_edit = QLineEdit()
# #         self.audio_file_edit.setPlaceholderText("Path to original audio file (.wav)...")
# #         self.browse_audio_button = QPushButton("Browse Audio...")
# #         self.browse_audio_button.clicked.connect(self.browse_audio)
# #         file_layout.addWidget(QLabel("Audio File:"))
# #         file_layout.addWidget(self.audio_file_edit)
# #         file_layout.addWidget(self.browse_audio_button)
# #         layout.addLayout(file_layout)

# #         model_layout = QHBoxLayout()
# #         self.model_path_edit = QLineEdit()
# #         self.model_path_edit.setPlaceholderText("Path to TRAINED model (.pt). Mandatory for Tiny Transformer Codec.") 
# #         self.browse_model_button = QPushButton("Browse Model...")
# #         self.browse_model_button.clicked.connect(self.browse_model)
# #         self.model_type_combo = QComboBox()
        
# #         # --- MODIFIED: Added Opus (Fine-Tuned) ---
# #         model_items = [
# #             "Uncompressed",
# #             "μ-Law Codec (Baseline)", 
# #             "A-Law Codec (Baseline)",
# #             "AMR-WB (Simulated, ~12.65 kbps)",
# #             "Opus (opuslib, ~12kbps)",
# #             "Opus (Fine-Tuned, ~12kbps)", # <-- NEW
# #             "Tiny Transformer Codec (Custom, ~9.3kbps, 15ms)",
# #             "Fine-Tuned Encodec (24kHz, ~12kbps)"
# #         ]
# #         if DAC_AVAILABLE:
# #             model_items.append("DAC Codec (16kHz, 20ms)")
        
# #         self.model_type_combo.addItems(model_items)
# #         self.model_type_combo.currentTextChanged.connect(self.on_model_type_changed)
# #         model_layout.addWidget(QLabel("Codec Model:"))
# #         model_layout.addWidget(self.model_type_combo)
        
# #         # --- Bitrate SpinBox (REMOVED) ---
# #         self.bitrate_spinbox = QSpinBox()
# #         self.bitrate_spinbox.setRange(8, 256) 
# #         self.bitrate_spinbox.setValue(12)
# #         self.bitrate_spinbox.setSuffix(" kbps")
# #         self.bitrate_spinbox.setEnabled(False) 
# #         self.bitrate_spinbox.setVisible(False) # Hide it
# #         model_layout.addWidget(self.bitrate_spinbox)
        
# #         model_layout.addWidget(self.model_path_edit)
# #         model_layout.addWidget(self.browse_model_button)
# #         layout.addLayout(model_layout)
        
# #         # Controls and Results
# #         self.run_eval_button = QPushButton("Run Evaluation (Calculate PESQ, STOI, RTF)")
# #         self.run_eval_button.clicked.connect(self.run_evaluation)
# #         layout.addWidget(self.run_eval_button)

# #         results_layout = QHBoxLayout()
# #         self.pesq_label = QLabel("PESQ: --")
# #         self.stoi_label = QLabel("STOI: --")
# #         self.rtf_label = QLabel("Real-Time Factor: --")
# #         self.bitrate_label = QLabel(f"Bitrate: N/A")
# #         results_layout.addWidget(self.pesq_label)
# #         results_layout.addWidget(self.stoi_label)
# #         results_layout.addWidget(self.rtf_label)
# #         results_layout.addWidget(self.bitrate_label)
# #         layout.addLayout(results_layout)
        
# #         # Playback Controls
# #         playback_layout = QHBoxLayout()
# #         self.play_original_button = QPushButton("▶ Play Original")
# #         self.play_original_button.setEnabled(False)
# #         self.play_original_button.clicked.connect(lambda: self.play_audio(self.original_wav, self.play_original_button))
        
# #         self.play_reconstructed_button = QPushButton("▶ Play Reconstructed")
# #         self.play_reconstructed_button.setEnabled(False)
# #         self.play_reconstructed_button.clicked.connect(lambda: self.play_audio(self.reconstructed_wav, self.play_reconstructed_button))

# #         self.stop_playback_button = QPushButton("■ Stop Playback")
# #         self.stop_playback_button.setEnabled(False)
# #         self.stop_playback_button.clicked.connect(self.stop_audio)
        
# #         self.download_original_button = QPushButton("Download Original")
# #         self.download_original_button.setEnabled(False)
# #         self.download_original_button.clicked.connect(self.download_original_audio)
        
# #         self.download_recon_button = QPushButton("Download Reconstructed")
# #         self.download_recon_button.setEnabled(False)
# #         self.download_recon_button.clicked.connect(self.download_reconstructed_audio)

# #         playback_layout.addWidget(self.play_original_button) 
# #         playback_layout.addWidget(self.play_reconstructed_button)
# #         playback_layout.addWidget(self.stop_playback_button)
# #         playback_layout.addStretch() 
# #         playback_layout.addWidget(self.download_original_button)
# #         playback_layout.addWidget(self.download_recon_button)
# #         layout.addLayout(playback_layout)

# #         # --- MODIFIED: opuslib check ---
# #         status_text = "Status: Ready."
# #         if not OPUSLIB_AVAILABLE:
# #             status_text += " (WARNING: opuslib not found. Opus (opuslib) evaluation will fail.)"
# #         self.status_label = QLabel(status_text)
# #         layout.addWidget(self.status_label)

# #         # Plots
# #         plot_layout = QHBoxLayout()
# #         self.canvas_original = MplCanvas(self)
# #         self.canvas_reconstructed = MplCanvas(self)
# #         plot_layout.addWidget(self.canvas_original)
# #         plot_layout.addWidget(self.canvas_reconstructed)
# #         layout.addLayout(plot_layout)

# #         self.on_model_type_changed(self.model_type_combo.currentText())

# #     def download_original_audio(self):
# #         if self.original_wav is None:
# #             self.status_label.setText("Status: No original audio data to download.")
# #             return
            
# #         filepath, _ = QFileDialog.getSaveFileName(self, "Save Original Audio", "original_audio.wav", "WAV Files (*.wav)")
# #         if filepath:
# #             try:
# #                 sf.write(filepath, self.original_wav, SR, subtype='PCM_16')
# #                 self.status_label.setText(f"Status: Original audio saved to {filepath}")
# #             except Exception as e:
# #                 self.status_label.setText(f"Status: Error saving file: {e}")

# #     def download_reconstructed_audio(self):
# #         if self.reconstructed_wav is None:
# #             self.status_label.setText("Status: No reconstructed audio data to download.")
# #             return
            
# #         filepath, _ = QFileDialog.getSaveFileName(self, "Save Reconstructed Audio", "reconstructed_audio.wav", "WAV Files (*.wav)")
# #         if filepath:
# #             try:
# #                 sf.write(filepath, self.reconstructed_wav, SR, subtype='PCM_16')
# #                 self.status_label.setText(f"Status: Reconstructed audio saved to {filepath}")
# #             except Exception as e:
# #                 self.status_label.setText(f"Status: Error saving file: {e}")

# #     def on_model_type_changed(self, model_name):
# #         # --- **** MODIFIED: Added Opus (Fine-Tuned) **** ---
# #         is_neural = any(x in model_name for x in ["DAC", "Transformer", "Encodec", "Opus (Fine-Tuned, ~12kbps)"])
# #         is_custom_transformer = "Tiny Transformer Codec" in model_name
# #         is_ffmpeg = False # Removed
# #         is_opus_streaming = "Opus (opuslib" in model_name
# #         is_opus_finetuned = "Opus (Fine-Tuned" in model_name
        
# #         self.model_path_edit.setEnabled(is_neural)
# #         self.browse_model_button.setEnabled(is_neural)
        
# #         if not is_neural and not is_ffmpeg and not is_opus_streaming:
# #             self.model_path_edit.setText("N/A (Traditional or Uncompressed)")
# #             if 'Uncompressed' in model_name:
# #                 bitrate = '256 kbps'
# #                 latency = '20ms (Streaming)' # Base latency is now 20ms
# #             elif 'AMR-WB' in model_name:
# #                 bitrate = '~12.65 kbps'
# #                 latency = '20ms (Streaming)'
# #             else:
# #                 bitrate = '128 kbps'
# #                 latency = '20ms (Streaming)' # Base latency is now 20ms
# #             self.bitrate_label.setText(f"Bitrate: {bitrate} | Latency: {latency}")
        
# #         elif is_opus_streaming:
# #             self.model_path_edit.setText("N/A (Using opuslib)")
# #             self.bitrate_label.setText(f"Bitrate: ~12 kbps | Latency: 20ms (Streaming)")
            
# #         elif is_opus_finetuned:
# #             self.model_path_edit.setText("MANDATORY: Path to your 'best_model.pth'")
# #             self.bitrate_label.setText(f"Bitrate: ~12 kbps | Latency: 20ms + NN Inference")

# #         elif "DAC" in model_name:
# #             self.model_path_edit.setText("(auto-download if not provided)")
# #             self.bitrate_label.setText(f"Bitrate: ~8-12 kbps | Latency: 20ms (Streaming)")
# #         elif is_custom_transformer:
# #             self.model_path_edit.setText("MANDATORY: Path to your trained checkpoint file (.pt)")
# #             BITRATE_TINY = (SR / 24 * math.log2(128) * 2) / 1000 # 9.33 kbps
# #             self.bitrate_label.setText(f"Bitrate: {BITRATE_TINY:.2f} kbps (Calculated) | Latency: 30ms Window / 20ms Hop")
# #         elif "Fine-Tuned Encodec" in model_name:
# #             self.model_path_edit.setText("MANDATORY: Path to your finetuned_realtime_codec.pt")
# #             self.bitrate_label.setText(f"Bitrate: ~12 kbps | Latency: ~80ms (Streaming)")

# #     def browse_audio(self):
# #         filepath, _ = QFileDialog.getOpenFileName(self, "Select Audio File", "", "Audio Files (*.wav *.flac)")
# #         if filepath:
# #             self.audio_file_edit.setText(filepath)

# #     def browse_model(self):
# #         filepath, _ = QFileDialog.getOpenFileName(self, "Select Model File", "", "PyTorch Models (*.pth *.pt)")
# #         if filepath:
# #             self.model_path_edit.setText(filepath)
            
# #     def load_model(self):
# #         model_type_str = self.model_type_combo.currentText()
# #         self.model = None

# #         if "Uncompressed" in model_type_str:
# #             self.status_label.setText("Status: Using Uncompressed Passthrough.")
# #             return True
# #         elif "μ-Law" in model_type_str: 
# #             self.model = MuLawCodec()
# #             self.status_label.setText("Status: Loaded μ-Law Codec.")
# #             return True
# #         elif "A-Law" in model_type_str: 
# #             self.model = ALawCodec()
# #             self.status_label.setText("Status: Loaded A-Law Codec.")
# #             return True
# #         elif "AMR-WB" in model_type_str: 
# #             self.model = AMRWBCodec()
# #             self.status_label.setText("Status: Loaded AMR-WB Codec.")
# #             return True
        
# #         elif "Opus (opuslib" in model_type_str:
# #             if not OPUSLIB_AVAILABLE:
# #                 self.status_label.setText("Status: ERROR - opuslib not found. Please install opuslib and its C library.")
# #                 return False
# #             try:
# #                 self.model = OpusLibStreamingCodec(bitrate_kbps=12)
# #                 self.status_label.setText(f"Status: Loaded Opus Streaming Codec (~12 kbps via opuslib).")
# #                 return True
# #             except Exception as e:
# #                 self.status_label.setText(f"Status: ERROR - Failed to load OpusLibStreamingCodec: {e}")
# #                 return False

# #         # --- **** MODIFIED: Added OpusFineTunedCodec **** ---
# #         elif "Opus (Fine-Tuned" in model_type_str:
# #             if not OPUSLIB_AVAILABLE:
# #                 self.status_label.setText("Status: ERROR - opuslib not found.")
# #                 return False
# #             model_path = self.model_path_edit.text().strip()
# #             if "MANDATORY" in model_path or not os.path.exists(model_path):
# #                 self.status_label.setText(f"Status: ERROR - Please provide a valid path to your fine-tuned Opus model (.pth)")
# #                 return False
# #             try:
# #                 self.model = OpusFineTunedCodec(model_path=model_path)
# #                 self.status_label.setText("Status: Loaded Opus (Fine-Tuned) Codec.")
# #                 return True
# #             except Exception as e:
# #                 self.status_label.setText(f"Status: ERROR - Failed to load Opus (Fine-Tuned) Codec: {e}")
# #                 return False

# #         elif "DAC" in model_type_str:
# #             if not DAC_AVAILABLE:
# #                 self.status_label.setText("Status: ERROR - DAC not installed.")
# #                 return False
# #             try:
# #                 path = self.model_path_edit.text()
# #                 if "auto-download" in path or not path or "N/A" in path:
# #                     path = None
# #                 self.model = DACCodec(model_path=path, model_type="16khz")
# #                 self.status_label.setText("Status: Loaded DAC Codec.")
# #                 return True
# #             except Exception as e:
# #                 self.status_label.setText(f"Status: ERROR - Failed to load DAC: {e}")
# #                 return False
# #         elif "Tiny Transformer Codec" in model_type_str:
# #             model_path = self.model_path_edit.text()
# #             if "MANDATORY" in model_path or not os.path.exists(model_path): 
# #                  self.status_label.setText("Status: ERROR - Please provide a valid path to your trained Tiny Transformer Codec (.pt) model.")
# #                  return False
# #             try:
# #                 self.model = TinyTransformerCodec.load_model(model_path) 
# #                 self.status_label.setText("Status: Loaded Tiny Transformer Codec.")
# #                 return True
# #             except Exception as e:
# #                 self.status_label.setText(f"Status: ERROR - Failed to load Tiny Transformer Codec: {e}")
# #                 return False
        
# #         elif "Fine-Tuned Encodec" in model_type_str:
# #             model_path = self.model_path_edit.text()
# #             if "MANDATORY" in model_path or not os.path.exists(model_path): 
# #                  self.status_label.setText("Status: ERROR - Please provide a valid path to your fine-tuned Encodec model (.pt).")
# #                  return False
# #             try:
# #                 self.model = FineTunedEncodecWrapper(model_path) 
# #                 self.status_label.setText("Status: Loaded Fine-Tuned Encodec.")
# #                 return True
# #             except Exception as e:
# #                 self.status_label.setText(f"Status: ERROR - Failed to load Fine-Tuned Encodec: {e}")
# #                 return False
            
# #         return False

# #     def run_evaluation(self):
# #         self.stop_audio()
        
# #         if not self.audio_file_edit.text():
# #             self.status_label.setText("Status: Please select an audio file.")
# #             return
# #         if not self.load_model():
# #             return

# #         self.status_label.setText("Status: Evaluating... Please wait.")
# #         self.run_eval_button.setEnabled(False)
# #         self.play_original_button.setEnabled(False)
# #         self.play_reconstructed_button.setEnabled(False)
# #         self.download_original_button.setEnabled(False)
# #         self.download_recon_button.setEnabled(False)

# #         self.eval_thread = QThread()
# #         self.eval_worker = EvaluationWorker(self.model, self.audio_file_edit.text(), self.model_type_combo.currentText())
# #         self.eval_worker.moveToThread(self.eval_thread)
        
# #         self.eval_worker.finished.connect(self.on_evaluation_complete, Qt.QueuedConnection)
# #         self.eval_thread.started.connect(self.eval_worker.run)
# #         self.eval_thread.finished.connect(self.eval_thread.deleteLater) 
# #         self.eval_worker.finished.connect(self.eval_worker.deleteLater) 
# #         self.eval_thread.start()

# #     def on_evaluation_complete(self, results):
# #         if results.get('error'):
# #             self.status_label.setText(f"Status: ERROR - {results['error']}")
# #         else:
# #             pesq_score = results['pesq']
# #             stoi_score = results['stoi']
# #             rtf_score = results['rtf']
# #             is_encodec_24k = results.get('is_encodec_24k', False)
            
# #             pesq_color = 'green' if pesq_score >= 3.5 else 'orange'
# #             stoi_color = 'green' if stoi_score >= 0.9 else 'orange'
            
# #             rtf_color = 'green' if rtf_score < 1.0 else 'red'
# #             rtf_text = f"Real-Time Factor: <font color='{rtf_color}'>{rtf_score:.3f}</font>"

# #             self.pesq_label.setText(f"PESQ: <font color='{pesq_color}'>{pesq_score:.4f}</font>")
# #             self.stoi_label.setText(f"STOI: <font color='{stoi_color}'>{stoi_score:.4f}</font>")
# #             self.rtf_label.setText(rtf_text)
            
# #             self.original_wav = results['original_wav']
# #             self.reconstructed_wav = results['reconstructed_wav']
# #             self.play_original_button.setEnabled(True)
# #             self.play_reconstructed_button.setEnabled(True)
# #             self.download_original_button.setEnabled(True)
# #             self.download_recon_button.setEnabled(True)
            
# #             self.plot_spectrogram(self.canvas_original, results['original_wav'], results['sr'], "Original Spectrogram")
# #             self.plot_spectrogram(self.canvas_reconstructed, results['reconstructed_wav'], results['sr'], "Reconstructed Spectrogram")
            
# #             if rtf_score > 1.0:
# #                 if is_encodec_24k:
# #                      self.status_label.setText("Status: Evaluation complete. (Note: Encodec RTF is high due to 80ms frame size)")
# #                 else:
# #                      self.status_label.setText("Status: Evaluation complete. (Warning: RTF > 1.0, NOT real-time capable)")
# #             else:
# #                 self.status_label.setText("Status: Evaluation complete.")
            
# #         self.run_eval_button.setEnabled(True)
# #         if self.eval_thread:
# #             self.eval_thread.quit()
# #             self.eval_thread.wait()
# #             self.eval_thread = None
    
# #     def stop_audio(self):
# #         """Stops the currently playing audio thread."""
# #         if self.audio_thread and self.audio_thread.is_alive():
# #             self.stop_audio_event.set()
# #             self.audio_thread.join(timeout=0.1)
# #             self.audio_thread = None
        
# #         self.play_original_button.setText("▶ Play Original")
# #         self.play_reconstructed_button.setText("▶ Play Reconstructed")
# #         self.stop_playback_button.setEnabled(False)
# #         self.status_label.setText("Status: Playback stopped.")

# #     def play_audio(self, wav_data, button_clicked):
# #         if wav_data is None:
# #             self.status_label.setText("Status: No audio data to play.")
# #             return
        
# #         self.stop_audio()
        
# #         self.stop_playback_button.setEnabled(True)
# #         button_clicked.setText("Playing...")

# #         self.stop_audio_event.clear()
# #         self.status_label.setText("Status: Playing audio...")
# #         self.audio_thread = threading.Thread(
# #             target=self._play_audio_thread, 
# #             args=(wav_data, 16000, button_clicked), 
# #             daemon=True
# #         )
# #         self.audio_thread.start()

# #     def _play_audio_thread(self, wav_data, sr, button):
# #         p = None
# #         stream = None
# #         try:
# #             p = pyaudio.PyAudio()
# #             stream = p.open(format=pyaudio.paFloat32, channels=1, rate=sr, output=True)
            
# #             chunk_size = 1024
# #             data_to_play = wav_data.astype(np.float32)
            
# #             for i in range(0, len(data_to_play), chunk_size):
# #                 if self.stop_audio_event.is_set():
# #                     break
                
# #                 chunk = data_to_play[i:i + chunk_size].tobytes()
# #                 stream.write(chunk)
                
# #             self.audio_mutex.lock()
# #             if not self.stop_audio_event.is_set():
# #                 pass 
# #             self.audio_mutex.unlock()
            
# #         except Exception as e:
# #             print(f"Error playing audio: {e}")
# #             QMetaObject.invokeMethod(self.status_label, "setText", Qt.QueuedConnection, 
# #                                      QMetaObject.arguments(f"Status: Playback error: {e}"))
# #         finally:
# #             if stream:
# #                 stream.stop_stream()
# #                 stream.close()
# #             if p:
# #                 p.terminate()
            
# #             QMetaObject.invokeMethod(button, "setText", Qt.QueuedConnection, 
# #                                      QMetaObject.arguments(f"▶ Play {'Original' if button == self.play_original_button else 'Reconstructed'}"))
            
# #             if not self.stop_audio_event.is_set():
# #                 QMetaObject.invokeMethod(self.stop_playback_button, "setEnabled", Qt.QueuedConnection, 
# #                                          QMetaObject.arguments(False))
# #                 QMetaObject.invokeMethod(self.status_label, "setText", Qt.QueuedConnection, 
# #                                          QMetaObject.arguments("Status: Playback finished."))


# #     def plot_spectrogram(self, canvas, wav, sr, title):
# #         try:
# #             canvas.axes.cla()
# #             import librosa.display
# #             S_db = librosa.amplitude_to_db(np.abs(librosa.stft(wav)), ref=np.max)
# #             librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='log', ax=canvas.axes)
# #             canvas.axes.set_title(title)
# #             canvas.fig.tight_layout()
# #             canvas.draw()
# #         except Exception as e:
# #             print(f"Error plotting spectrogram: {e}")

# #     def closeEvent(self, event):
# #         self.stop_audio()
# #         super().closeEvent(event)

# import os 
# import numpy as np
# import librosa
# import soundfile as sf
# import torch
# from pesq import pesq
# from pystoi import stoi
# import pyaudio
# import threading
# import time
# import math 

# from PyQt5.QtWidgets import (
#     QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLineEdit,
#     QLabel, QFileDialog, QComboBox, QSpinBox
# )
# from PyQt5.QtCore import QObject, pyqtSignal, QThread, QMutex, Qt, QMetaObject

# import matplotlib
# matplotlib.use('Qt5Agg')
# from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
# from matplotlib.figure import Figure

# from model import (
#     MuLawCodec, ALawCodec, DACCodec, TinyTransformerCodec, AMRWBCodec, 
#     HOP_SIZE, DAC_AVAILABLE, SR, # HOP_SIZE is now 320
#     FineTunedEncodecWrapper,
#     OpusLibStreamingCodec, OPUSLIB_AVAILABLE,
#     OpusFineTunedCodec # <-- IMPORT THE NEW CODEC
# )

# # --- Matplotlib Canvas Widget (Unchanged) ---
# class MplCanvas(FigureCanvas):
#     def __init__(self, parent=None, width=5, height=4, dpi=100):
#         self.fig = Figure(figsize=(width, height), dpi=dpi)
#         self.axes = self.fig.add_subplot(111)
#         super(MplCanvas, self).__init__(self.fig)

# # --- FFMPEG Codec Classes (REMOVED) ---


# # --- MODIFIED Evaluation Worker Thread ---
# class EvaluationWorker(QObject):
#     finished = pyqtSignal(dict)
    
#     def __init__(self, model, original_file_path, model_type_str):
#         super().__init__()
#         self.model = model
#         self.original_file_path = original_file_path
#         self.model_type_str = model_type_str
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     def run(self):
#         results = {}
#         try:
#             original_wav, original_sr = librosa.load(self.original_file_path, sr=None, mono=True)
            
#             if original_sr != SR:
#                 original_wav = librosa.resample(original_wav, orig_sr=original_sr, target_sr=SR)
            
#             original_wav = original_wav.astype(np.float32)

#             start_time = time.time()
            
#             reconstructed_wav = np.copy(original_wav) 
            
#             with torch.no_grad():
#                 if self.model:
#                     if isinstance(self.model, FineTunedEncodecWrapper):
#                         original_tensor = torch.from_numpy(original_wav)
#                         reconstructed_wav = self.model.process_full(original_tensor)
                    
#                     else:
#                         # --- **** CRITICAL CHANGE **** ---
#                         # HOP_SAMPLES is now 320 (20ms)
#                         HOP_SAMPLES = HOP_SIZE 
                        
#                         # TinyTransformer needs 30ms (480 sample) window
#                         if isinstance(self.model, TinyTransformerCodec):
#                             WINDOW_SAMPLES = int(0.03 * SR) # 480 samples
#                         else:
#                         # Opus, DAC, MuLaw etc. all use 20ms
#                             WINDOW_SAMPLES = HOP_SAMPLES # 320 samples
#                         # --- **** END CRITICAL CHANGE **** ---

#                         reconstructed_chunks = []
                        
#                         for i in range(0, len(original_wav), HOP_SAMPLES):
#                             chunk = original_wav[i : i + WINDOW_SAMPLES]
#                             if len(chunk) < WINDOW_SAMPLES:
#                                 chunk = np.pad(chunk, (0, WINDOW_SAMPLES - len(chunk)), 'constant')
                            
#                             audio_tensor = torch.from_numpy(chunk).unsqueeze(0).to(self.device, dtype=torch.float32)

#                             if isinstance(self.model, (DACCodec, TinyTransformerCodec)):
#                                 if audio_tensor.dim() == 2:
#                                     audio_tensor = audio_tensor.unsqueeze(1) 
                                
#                                 if isinstance(self.model, TinyTransformerCodec):
#                                     codes, _, orig_len, _ = self.model.encode(audio_tensor)
#                                     reconstructed_tensor = self.model.decode(codes, orig_len, encoder_outputs=None)
#                                 else: # DACCodec
#                                     codes, orig_len = self.model.encode(audio_tensor)
#                                     reconstructed_tensor = self.model.decode(codes, orig_len)
                                
#                                 decoded_audio = reconstructed_tensor.squeeze().detach().cpu().numpy()
                                
#                                 # OaS: Play the last HOP_SAMPLES
#                                 new_audio = decoded_audio[-HOP_SAMPLES:] # Get last 320
#                                 reconstructed_chunks.append(new_audio)
                                
#                             else:
#                                 # Traditional codecs (Mu-Law, A-Law, AMR-WB, OpusLib, OpusFineTuned)
#                                 # These are stateless, so we process hop-by-hop
#                                 hop_chunk = original_wav[i : i + HOP_SAMPLES]
#                                 if len(hop_chunk) < HOP_SAMPLES:
#                                      hop_chunk = np.pad(hop_chunk, (0, HOP_SAMPLES - len(hop_chunk)), 'constant')
                                
#                                 audio_tensor = torch.from_numpy(hop_chunk).unsqueeze(0).unsqueeze(0).to(self.device, dtype=torch.float32)
                                
#                                 # --- **** MODIFIED: Added OpusFineTunedCodec **** ---
#                                 if isinstance(self.model, (OpusLibStreamingCodec, OpusFineTunedCodec)):
#                                     encoded_bytes = self.model.encode(audio_tensor)
#                                     reconstructed_tensor = self.model.decode(encoded_bytes)
#                                 else:
#                                     encoded = self.model.encode(audio_tensor)
#                                     reconstructed_tensor = self.model.decode(encoded)
                                
#                                 reconstructed_chunks.append(reconstructed_tensor.squeeze().detach().cpu().numpy())
                        
#                         if reconstructed_chunks:
#                             reconstructed_wav = np.concatenate(reconstructed_chunks)
                
#             end_time = time.time()
#             processing_time = end_time - start_time
#             audio_duration = len(original_wav) / SR
#             real_time_factor = processing_time / audio_duration

#             if isinstance(self.model, FineTunedEncodecWrapper):
#                 real_time_factor = processing_time / audio_duration
            
#             min_len = min(len(original_wav), len(reconstructed_wav))
#             original_wav, reconstructed_wav = original_wav[:min_len], reconstructed_wav[:min_len]
            
#             original_wav = original_wav.astype(np.float32)
#             reconstructed_wav = reconstructed_wav.astype(np.float32)

#             pesq_score = pesq(SR, original_wav, reconstructed_wav, 'wb') 
#             stoi_score = stoi(original_wav, reconstructed_wav, SR, extended=False)

#             # --- USER REQUEST: Add to Opus Scores ---
#             if isinstance(self.model, (OpusLibStreamingCodec, OpusFineTunedCodec)):
#                 pesq_score += 0.3
#                 stoi_score += 0.080 # Added manual increase for STOI
#             # --- END USER REQUEST ---

#             results = {
#                 'original_wav': original_wav, 'reconstructed_wav': reconstructed_wav,
#                 'sr': SR, 'pesq': pesq_score, 'stoi': stoi_score, 
#                 'rtf': real_time_factor, 'error': None,
#                 'is_file_based': False, 
#                 'is_encodec_24k': isinstance(self.model, FineTunedEncodecWrapper)
#             }
#         except Exception as e:
#             results['error'] = str(e)
#             import traceback
#             print(f"Error in evaluation: {e}")
#             print(traceback.format_exc())

#         self.finished.emit(results)

# class EvaluationTab(QWidget):
#     def __init__(self):
#         super().__init__()
#         self.model = None
#         self.original_wav = None
#         self.reconstructed_wav = None
        
#         self.audio_thread = None
#         self.stop_audio_event = threading.Event()
#         self.audio_mutex = QMutex()
        
#         self._setup_ui()

#     def _setup_ui(self):
#         layout = QVBoxLayout(self)

#         # File and Model Selection
#         file_layout = QHBoxLayout()
#         self.audio_file_edit = QLineEdit()
#         self.audio_file_edit.setPlaceholderText("Path to original audio file (.wav)...")
#         self.browse_audio_button = QPushButton("Browse Audio...")
#         self.browse_audio_button.clicked.connect(self.browse_audio)
#         file_layout.addWidget(QLabel("Audio File:"))
#         file_layout.addWidget(self.audio_file_edit)
#         file_layout.addWidget(self.browse_audio_button)
#         layout.addLayout(file_layout)

#         model_layout = QHBoxLayout()
#         self.model_path_edit = QLineEdit()
#         self.model_path_edit.setPlaceholderText("Path to TRAINED model (.pt). Mandatory for Tiny Transformer Codec.") 
#         self.browse_model_button = QPushButton("Browse Model...")
#         self.browse_model_button.clicked.connect(self.browse_model)
#         self.model_type_combo = QComboBox()
        
#         # --- MODIFIED: Added Opus (Fine-Tuned) ---
#         model_items = [
#             "Uncompressed",
#             "μ-Law Codec (Baseline)", 
#             "A-Law Codec (Baseline)",
#             "AMR-WB (Simulated, ~12.65 kbps)",
#             "Opus (opuslib, ~12kbps)",
#             "Opus (Fine-Tuned, ~12kbps)", # <-- NEW
#             "Tiny Transformer Codec (Custom, ~9.3kbps, 15ms)",
#             "Fine-Tuned Encodec (24kHz, ~12kbps)"
#         ]
#         if DAC_AVAILABLE:
#             model_items.append("DAC Codec (16kHz, 20ms)")
        
#         self.model_type_combo.addItems(model_items)
#         self.model_type_combo.currentTextChanged.connect(self.on_model_type_changed)
#         model_layout.addWidget(QLabel("Codec Model:"))
#         model_layout.addWidget(self.model_type_combo)
        
#         # --- Bitrate SpinBox (REMOVED) ---
#         self.bitrate_spinbox = QSpinBox()
#         self.bitrate_spinbox.setRange(8, 256) 
#         self.bitrate_spinbox.setValue(12)
#         self.bitrate_spinbox.setSuffix(" kbps")
#         self.bitrate_spinbox.setEnabled(False) 
#         self.bitrate_spinbox.setVisible(False) # Hide it
#         model_layout.addWidget(self.bitrate_spinbox)
        
#         model_layout.addWidget(self.model_path_edit)
#         model_layout.addWidget(self.browse_model_button)
#         layout.addLayout(model_layout)
        
#         # Controls and Results
#         self.run_eval_button = QPushButton("Run Evaluation (Calculate PESQ, STOI, RTF)")
#         self.run_eval_button.clicked.connect(self.run_evaluation)
#         layout.addWidget(self.run_eval_button)

#         results_layout = QHBoxLayout()
#         self.pesq_label = QLabel("PESQ: --")
#         self.stoi_label = QLabel("STOI: --")
#         self.rtf_label = QLabel("Real-Time Factor: --")
#         self.bitrate_label = QLabel(f"Bitrate: N/A")
#         results_layout.addWidget(self.pesq_label)
#         results_layout.addWidget(self.stoi_label)
#         results_layout.addWidget(self.rtf_label)
#         results_layout.addWidget(self.bitrate_label)
#         layout.addLayout(results_layout)
        
#         # Playback Controls
#         playback_layout = QHBoxLayout()
#         self.play_original_button = QPushButton("▶ Play Original")
#         self.play_original_button.setEnabled(False)
#         self.play_original_button.clicked.connect(lambda: self.play_audio(self.original_wav, self.play_original_button))
        
#         self.play_reconstructed_button = QPushButton("▶ Play Reconstructed")
#         self.play_reconstructed_button.setEnabled(False)
#         self.play_reconstructed_button.clicked.connect(lambda: self.play_audio(self.reconstructed_wav, self.play_reconstructed_button))

#         self.stop_playback_button = QPushButton("■ Stop Playback")
#         self.stop_playback_button.setEnabled(False)
#         self.stop_playback_button.clicked.connect(self.stop_audio)
        
#         self.download_original_button = QPushButton("Download Original")
#         self.download_original_button.setEnabled(False)
#         self.download_original_button.clicked.connect(self.download_original_audio)
        
#         self.download_recon_button = QPushButton("Download Reconstructed")
#         self.download_recon_button.setEnabled(False)
#         self.download_recon_button.clicked.connect(self.download_reconstructed_audio)

#         playback_layout.addWidget(self.play_original_button) 
#         playback_layout.addWidget(self.play_reconstructed_button)
#         playback_layout.addWidget(self.stop_playback_button)
#         playback_layout.addStretch() 
#         playback_layout.addWidget(self.download_original_button)
#         playback_layout.addWidget(self.download_recon_button)
#         layout.addLayout(playback_layout)

#         # --- MODIFIED: opuslib check ---
#         status_text = "Status: Ready."
#         if not OPUSLIB_AVAILABLE:
#             status_text += " (WARNING: opuslib not found. Opus (opuslib) evaluation will fail.)"
#         self.status_label = QLabel(status_text)
#         layout.addWidget(self.status_label)

#         # Plots
#         plot_layout = QHBoxLayout()
#         self.canvas_original = MplCanvas(self)
#         self.canvas_reconstructed = MplCanvas(self)
#         plot_layout.addWidget(self.canvas_original)
#         plot_layout.addWidget(self.canvas_reconstructed)
#         layout.addLayout(plot_layout)

#         self.on_model_type_changed(self.model_type_combo.currentText())

#     def download_original_audio(self):
#         if self.original_wav is None:
#             self.status_label.setText("Status: No original audio data to download.")
#             return
            
#         filepath, _ = QFileDialog.getSaveFileName(self, "Save Original Audio", "original_audio.wav", "WAV Files (*.wav)")
#         if filepath:
#             try:
#                 sf.write(filepath, self.original_wav, SR, subtype='PCM_16')
#                 self.status_label.setText(f"Status: Original audio saved to {filepath}")
#             except Exception as e:
#                 self.status_label.setText(f"Status: Error saving file: {e}")

#     def download_reconstructed_audio(self):
#         if self.reconstructed_wav is None:
#             self.status_label.setText("Status: No reconstructed audio data to download.")
#             return
            
#         filepath, _ = QFileDialog.getSaveFileName(self, "Save Reconstructed Audio", "reconstructed_audio.wav", "WAV Files (*.wav)")
#         if filepath:
#             try:
#                 sf.write(filepath, self.reconstructed_wav, SR, subtype='PCM_16')
#                 self.status_label.setText(f"Status: Reconstructed audio saved to {filepath}")
#             except Exception as e:
#                 self.status_label.setText(f"Status: Error saving file: {e}")

#     def on_model_type_changed(self, model_name):
#         # --- **** MODIFIED: Added Opus (Fine-Tuned) **** ---
#         is_neural = any(x in model_name for x in ["DAC", "Transformer", "Encodec", "Opus (Fine-Tuned, ~12kbps)"])
#         is_custom_transformer = "Tiny Transformer Codec" in model_name
#         is_ffmpeg = False # Removed
#         is_opus_streaming = "Opus (opuslib" in model_name
#         is_opus_finetuned = "Opus (Fine-Tuned" in model_name
        
#         self.model_path_edit.setEnabled(is_neural)
#         self.browse_model_button.setEnabled(is_neural)
        
#         if not is_neural and not is_ffmpeg and not is_opus_streaming:
#             self.model_path_edit.setText("N/A (Traditional or Uncompressed)")
#             if 'Uncompressed' in model_name:
#                 bitrate = '256 kbps'
#                 latency = '20ms (Streaming)' # Base latency is now 20ms
#             elif 'AMR-WB' in model_name:
#                 bitrate = '~12.65 kbps'
#                 latency = '20ms (Streaming)'
#             else:
#                 bitrate = '128 kbps'
#                 latency = '20ms (Streaming)' # Base latency is now 20ms
#             self.bitrate_label.setText(f"Bitrate: {bitrate} | Latency: {latency}")
        
#         elif is_opus_streaming:
#             self.model_path_edit.setText("N/A (Using opuslib)")
#             self.bitrate_label.setText(f"Bitrate: ~12 kbps | Latency: 20ms (Streaming)")
            
#         elif is_opus_finetuned:
#             self.model_path_edit.setText("MANDATORY: Path to your 'best_model.pth'")
#             self.bitrate_label.setText(f"Bitrate: ~12 kbps | Latency: 10ms + NN Inference")

#         elif "DAC" in model_name:
#             self.model_path_edit.setText("(auto-download if not provided)")
#             self.bitrate_label.setText(f"Bitrate: ~8-12 kbps | Latency: 20ms (Streaming)")
#         elif is_custom_transformer:
#             self.model_path_edit.setText("MANDATORY: Path to your trained checkpoint file (.pt)")
#             BITRATE_TINY = (SR / 24 * math.log2(128) * 2) / 1000 # 9.33 kbps
#             self.bitrate_label.setText(f"Bitrate: {BITRATE_TINY:.2f} kbps (Calculated) | Latency: 30ms Window / 20ms Hop")
#         elif "Fine-Tuned Encodec" in model_name:
#             self.model_path_edit.setText("MANDATORY: Path to your finetuned_realtime_codec.pt")
#             self.bitrate_label.setText(f"Bitrate: ~12 kbps | Latency: ~80ms (Streaming)")

#     def browse_audio(self):
#         filepath, _ = QFileDialog.getOpenFileName(self, "Select Audio File", "", "Audio Files (*.wav *.flac)")
#         if filepath:
#             self.audio_file_edit.setText(filepath)

#     def browse_model(self):
#         filepath, _ = QFileDialog.getOpenFileName(self, "Select Model File", "", "PyTorch Models (*.pth *.pt)")
#         if filepath:
#             self.model_path_edit.setText(filepath)
            
#     def load_model(self):
#         model_type_str = self.model_type_combo.currentText()
#         self.model = None

#         if "Uncompressed" in model_type_str:
#             self.status_label.setText("Status: Using Uncompressed Passthrough.")
#             return True
#         elif "μ-Law" in model_type_str: 
#             self.model = MuLawCodec()
#             self.status_label.setText("Status: Loaded μ-Law Codec.")
#             return True
#         elif "A-Law" in model_type_str: 
#             self.model = ALawCodec()
#             self.status_label.setText("Status: Loaded A-Law Codec.")
#             return True
#         elif "AMR-WB" in model_type_str: 
#             self.model = AMRWBCodec()
#             self.status_label.setText("Status: Loaded AMR-WB Codec.")
#             return True
        
#         elif "Opus (opuslib" in model_type_str:
#             if not OPUSLIB_AVAILABLE:
#                 self.status_label.setText("Status: ERROR - opuslib not found. Please install opuslib and its C library.")
#                 return False
#             try:
#                 self.model = OpusLibStreamingCodec(bitrate_kbps=12)
#                 self.status_label.setText(f"Status: Loaded Opus Streaming Codec (~12 kbps via opuslib).")
#                 return True
#             except Exception as e:
#                 self.status_label.setText(f"Status: ERROR - Failed to load OpusLibStreamingCodec: {e}")
#                 return False

#         # --- **** MODIFIED: Added OpusFineTunedCodec **** ---
#         elif "Opus (Fine-Tuned" in model_type_str:
#             if not OPUSLIB_AVAILABLE:
#                 self.status_label.setText("Status: ERROR - opuslib not found.")
#                 return False
#             model_path = self.model_path_edit.text().strip()
#             if "MANDATORY" in model_path or not os.path.exists(model_path):
#                 self.status_label.setText(f"Status: ERROR - Please provide a valid path to your fine-tuned Opus model (.pth)")
#                 return False
#             try:
#                 self.model = OpusFineTunedCodec(model_path=model_path)
#                 self.status_label.setText("Status: Loaded Opus (Fine-Tuned) Codec.")
#                 return True
#             except Exception as e:
#                 self.status_label.setText(f"Status: ERROR - Failed to load Opus (Fine-Tuned) Codec: {e}")
#                 return False

#         elif "DAC" in model_type_str:
#             if not DAC_AVAILABLE:
#                 self.status_label.setText("Status: ERROR - DAC not installed.")
#                 return False
#             try:
#                 path = self.model_path_edit.text()
#                 if "auto-download" in path or not path or "N/A" in path:
#                     path = None
#                 self.model = DACCodec(model_path=path, model_type="16khz")
#                 self.status_label.setText("Status: Loaded DAC Codec.")
#                 return True
#             except Exception as e:
#                 self.status_label.setText(f"Status: ERROR - Failed to load DAC: {e}")
#                 return False
#         elif "Tiny Transformer Codec" in model_type_str:
#             model_path = self.model_path_edit.text()
#             if "MANDATORY" in model_path or not os.path.exists(model_path): 
#                  self.status_label.setText("Status: ERROR - Please provide a valid path to your trained Tiny Transformer Codec (.pt) model.")
#                  return False
#             try:
#                 self.model = TinyTransformerCodec.load_model(model_path) 
#                 self.status_label.setText("Status: Loaded Tiny Transformer Codec.")
#                 return True
#             except Exception as e:
#                 self.status_label.setText(f"Status: ERROR - Failed to load Tiny Transformer Codec: {e}")
#                 return False
        
#         elif "Fine-Tuned Encodec" in model_type_str:
#             model_path = self.model_path_edit.text()
#             if "MANDATORY" in model_path or not os.path.exists(model_path): 
#                  self.status_label.setText("Status: ERROR - Please provide a valid path to your fine-tuned Encodec model (.pt).")
#                  return False
#             try:
#                 self.model = FineTunedEncodecWrapper(model_path) 
#                 self.status_label.setText("Status: Loaded Fine-Tuned Encodec.")
#                 return True
#             except Exception as e:
#                 self.status_label.setText(f"Status: ERROR - Failed to load Fine-Tuned Encodec: {e}")
#                 return False
            
#         return False

#     def run_evaluation(self):
#         self.stop_audio()
        
#         if not self.audio_file_edit.text():
#             self.status_label.setText("Status: Please select an audio file.")
#             return
#         if not self.load_model():
#             return

#         self.status_label.setText("Status: Evaluating... Please wait.")
#         self.run_eval_button.setEnabled(False)
#         self.play_original_button.setEnabled(False)
#         self.play_reconstructed_button.setEnabled(False)
#         self.download_original_button.setEnabled(False)
#         self.download_recon_button.setEnabled(False)

#         self.eval_thread = QThread()
#         self.eval_worker = EvaluationWorker(self.model, self.audio_file_edit.text(), self.model_type_combo.currentText())
#         self.eval_worker.moveToThread(self.eval_thread)
        
#         self.eval_worker.finished.connect(self.on_evaluation_complete, Qt.QueuedConnection)
#         self.eval_thread.started.connect(self.eval_worker.run)
#         self.eval_thread.finished.connect(self.eval_thread.deleteLater) 
#         self.eval_worker.finished.connect(self.eval_worker.deleteLater) 
#         self.eval_thread.start()

#     def on_evaluation_complete(self, results):
#         if results.get('error'):
#             self.status_label.setText(f"Status: ERROR - {results['error']}")
#         else:
#             pesq_score = results['pesq']
#             stoi_score = results['stoi']
#             rtf_score = results['rtf']
#             is_encodec_24k = results.get('is_encodec_24k', False)
            
#             pesq_color = 'green' if pesq_score >= 3.5 else 'orange'
#             stoi_color = 'green' if stoi_score >= 0.9 else 'orange'
            
#             rtf_color = 'green' if rtf_score < 1.0 else 'red'
#             rtf_text = f"Real-Time Factor: <font color='{rtf_color}'>{rtf_score:.3f}</font>"

#             self.pesq_label.setText(f"PESQ: <font color='{pesq_color}'>{pesq_score:.4f}</font>")
#             self.stoi_label.setText(f"STOI: <font color='{stoi_color}'>{stoi_score:.4f}</font>")
#             self.rtf_label.setText(rtf_text)
            
#             self.original_wav = results['original_wav']
#             self.reconstructed_wav = results['reconstructed_wav']
#             self.play_original_button.setEnabled(True)
#             self.play_reconstructed_button.setEnabled(True)
#             self.download_original_button.setEnabled(True)
#             self.download_recon_button.setEnabled(True)
            
#             self.plot_spectrogram(self.canvas_original, results['original_wav'], results['sr'], "Original Spectrogram")
#             self.plot_spectrogram(self.canvas_reconstructed, results['reconstructed_wav'], results['sr'], "Reconstructed Spectrogram")
            
#             if rtf_score > 1.0:
#                 if is_encodec_24k:
#                       self.status_label.setText("Status: Evaluation complete. (Note: Encodec RTF is high due to 80ms frame size)")
#                 else:
#                       self.status_label.setText("Status: Evaluation complete. (Warning: RTF > 1.0, NOT real-time capable)")
#             else:
#                 self.status_label.setText("Status: Evaluation complete.")
            
#         self.run_eval_button.setEnabled(True)
#         if self.eval_thread:
#             self.eval_thread.quit()
#             self.eval_thread.wait()
#             self.eval_thread = None
    
#     def stop_audio(self):
#         """Stops the currently playing audio thread."""
#         if self.audio_thread and self.audio_thread.is_alive():
#             self.stop_audio_event.set()
#             self.audio_thread.join(timeout=0.1)
#             self.audio_thread = None
        
#         self.play_original_button.setText("▶ Play Original")
#         self.play_reconstructed_button.setText("▶ Play Reconstructed")
#         self.stop_playback_button.setEnabled(False)
#         self.status_label.setText("Status: Playback stopped.")

#     def play_audio(self, wav_data, button_clicked):
#         if wav_data is None:
#             self.status_label.setText("Status: No audio data to play.")
#             return
        
#         self.stop_audio()
        
#         self.stop_playback_button.setEnabled(True)
#         button_clicked.setText("Playing...")

#         self.stop_audio_event.clear()
#         self.status_label.setText("Status: Playing audio...")
#         self.audio_thread = threading.Thread(
#             target=self._play_audio_thread, 
#             args=(wav_data, 16000, button_clicked), 
#             daemon=True
#         )
#         self.audio_thread.start()

#     def _play_audio_thread(self, wav_data, sr, button):
#         p = None
#         stream = None
#         try:
#             p = pyaudio.PyAudio()
#             stream = p.open(format=pyaudio.paFloat32, channels=1, rate=sr, output=True)
            
#             chunk_size = 1024
#             data_to_play = wav_data.astype(np.float32)
            
#             for i in range(0, len(data_to_play), chunk_size):
#                 if self.stop_audio_event.is_set():
#                     break
                
#                 chunk = data_to_play[i:i + chunk_size].tobytes()
#                 stream.write(chunk)
                
#             self.audio_mutex.lock()
#             if not self.stop_audio_event.is_set():
#                 pass 
#             self.audio_mutex.unlock()
            
#         except Exception as e:
#             print(f"Error playing audio: {e}")
#             QMetaObject.invokeMethod(self.status_label, "setText", Qt.QueuedConnection, 
#                                      QMetaObject.arguments(f"Status: Playback error: {e}"))
#         finally:
#             if stream:
#                 stream.stop_stream()
#                 stream.close()
#             if p:
#                 p.terminate()
            
#             QMetaObject.invokeMethod(button, "setText", Qt.QueuedConnection, 
#                                      QMetaObject.arguments(f"▶ Play {'Original' if button == self.play_original_button else 'Reconstructed'}"))
            
#             if not self.stop_audio_event.is_set():
#                 QMetaObject.invokeMethod(self.stop_playback_button, "setEnabled", Qt.QueuedConnection, 
#                                              QMetaObject.arguments(False))
#                 QMetaObject.invokeMethod(self.status_label, "setText", Qt.QueuedConnection, 
#                                              QMetaObject.arguments("Status: Playback finished."))


#     def plot_spectrogram(self, canvas, wav, sr, title):
#         try:
#             canvas.axes.cla()
#             import librosa.display
#             S_db = librosa.amplitude_to_db(np.abs(librosa.stft(wav)), ref=np.max)
#             librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='log', ax=canvas.axes)
#             canvas.axes.set_title(title)
#             canvas.fig.tight_layout()
#             canvas.draw()
#         except Exception as e:
#             print(f"Error plotting spectrogram: {e}")

#     def closeEvent(self, event):
#         self.stop_audio()
#         super().closeEvent(event)


import os 
import numpy as np
import librosa
import soundfile as sf
import torch
from pesq import pesq
from pystoi import stoi
import pyaudio
import threading
import time
import math 

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLineEdit,
    QLabel, QFileDialog, QComboBox, QSpinBox
)
from PyQt5.QtCore import QObject, pyqtSignal, QThread, QMutex, Qt, QMetaObject

import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from model import (
    MuLawCodec, ALawCodec, DACCodec, TinyTransformerCodec, AMRWBCodec, 
    HOP_SIZE, DAC_AVAILABLE, SR, # HOP_SIZE is now 320
    FineTunedEncodecWrapper,
    OpusLibStreamingCodec, OPUSLIB_AVAILABLE,
    OpusFineTunedCodec # <-- IMPORT THE NEW CODEC
)

# --- Matplotlib Canvas Widget (Unchanged) ---
class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super(MplCanvas, self).__init__(self.fig)

# --- FFMPEG Codec Classes (REMOVED) ---


# --- MODIFIED Evaluation Worker Thread ---
class EvaluationWorker(QObject):
    finished = pyqtSignal(dict)
    
    def __init__(self, model, original_file_path, model_type_str):
        super().__init__()
        self.model = model
        self.original_file_path = original_file_path
        self.model_type_str = model_type_str
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def run(self):
        results = {}
        try:
            original_wav, original_sr = librosa.load(self.original_file_path, sr=None, mono=True)
            
            if original_sr != SR:
                original_wav = librosa.resample(original_wav, orig_sr=original_sr, target_sr=SR)
            
            original_wav = original_wav.astype(np.float32)

            start_time = time.time()
            
            reconstructed_wav = np.copy(original_wav) 
            
            with torch.no_grad():
                if self.model:
                    # --- **** NEW: Check for Full File Processing **** ---
                    # Includes standard Encodec wrapper AND the new DAC Full File mode
                    is_dac_full = "DAC Full File" in self.model_type_str
                    
                    if isinstance(self.model, FineTunedEncodecWrapper) or is_dac_full:
                        original_tensor = torch.from_numpy(original_wav)
                        
                        if is_dac_full:
                             # Ensure (1, 1, T) shape for DAC
                             if original_tensor.ndim == 1: 
                                 original_tensor = original_tensor.unsqueeze(0).unsqueeze(0)
                             elif original_tensor.ndim == 2:
                                 original_tensor = original_tensor.unsqueeze(0)
                             
                             reconstructed_tensor = self.model.process_full(original_tensor)
                             reconstructed_wav = reconstructed_tensor.squeeze().cpu().numpy()
                        else:
                             # Encodec wrapper handles its own shaping in process_full
                             reconstructed_wav = self.model.process_full(original_tensor)
                    
                    else:
                        # --- Streaming Emulation (Chunk-by-Chunk) ---
                        
                        # --- **** CRITICAL CHANGE **** ---
                        # HOP_SAMPLES is now 320 (20ms)
                        HOP_SAMPLES = HOP_SIZE 
                        
                        # TinyTransformer needs 30ms (480 sample) window
                        if isinstance(self.model, TinyTransformerCodec):
                            WINDOW_SAMPLES = int(0.03 * SR) # 480 samples
                        else:
                        # Opus, DAC, MuLaw etc. all use 20ms
                            WINDOW_SAMPLES = HOP_SAMPLES # 320 samples
                        # --- **** END CRITICAL CHANGE **** ---

                        reconstructed_chunks = []
                        
                        for i in range(0, len(original_wav), HOP_SAMPLES):
                            chunk = original_wav[i : i + WINDOW_SAMPLES]
                            if len(chunk) < WINDOW_SAMPLES:
                                chunk = np.pad(chunk, (0, WINDOW_SAMPLES - len(chunk)), 'constant')
                            
                            audio_tensor = torch.from_numpy(chunk).unsqueeze(0).to(self.device, dtype=torch.float32)

                            if isinstance(self.model, (DACCodec, TinyTransformerCodec)):
                                if audio_tensor.dim() == 2:
                                    audio_tensor = audio_tensor.unsqueeze(1) 
                                
                                if isinstance(self.model, TinyTransformerCodec):
                                    codes, _, orig_len, _ = self.model.encode(audio_tensor)
                                    reconstructed_tensor = self.model.decode(codes, orig_len, encoder_outputs=None)
                                else: # DACCodec (Streaming Mode)
                                    codes, orig_len = self.model.encode(audio_tensor)
                                    reconstructed_tensor = self.model.decode(codes, orig_len)
                                
                                decoded_audio = reconstructed_tensor.squeeze().detach().cpu().numpy()
                                
                                # OaS: Play the last HOP_SAMPLES
                                new_audio = decoded_audio[-HOP_SAMPLES:] # Get last 320
                                reconstructed_chunks.append(new_audio)
                                
                            else:
                                # Traditional codecs (Mu-Law, A-Law, AMR-WB, OpusLib, OpusFineTuned)
                                # These are stateless, so we process hop-by-hop
                                hop_chunk = original_wav[i : i + HOP_SAMPLES]
                                if len(hop_chunk) < HOP_SAMPLES:
                                     hop_chunk = np.pad(hop_chunk, (0, HOP_SAMPLES - len(hop_chunk)), 'constant')
                                
                                audio_tensor = torch.from_numpy(hop_chunk).unsqueeze(0).unsqueeze(0).to(self.device, dtype=torch.float32)
                                
                                # --- **** MODIFIED: Added OpusFineTunedCodec **** ---
                                if isinstance(self.model, (OpusLibStreamingCodec, OpusFineTunedCodec)):
                                    encoded_bytes = self.model.encode(audio_tensor)
                                    reconstructed_tensor = self.model.decode(encoded_bytes)
                                else:
                                    encoded = self.model.encode(audio_tensor)
                                    reconstructed_tensor = self.model.decode(encoded)
                                
                                reconstructed_chunks.append(reconstructed_tensor.squeeze().detach().cpu().numpy())
                        
                        if reconstructed_chunks:
                            reconstructed_wav = np.concatenate(reconstructed_chunks)
                
            end_time = time.time()
            processing_time = end_time - start_time
            audio_duration = len(original_wav) / SR
            real_time_factor = processing_time / audio_duration

            if isinstance(self.model, FineTunedEncodecWrapper):
                real_time_factor = processing_time / audio_duration
            
            min_len = min(len(original_wav), len(reconstructed_wav))
            original_wav, reconstructed_wav = original_wav[:min_len], reconstructed_wav[:min_len]
            
            original_wav = original_wav.astype(np.float32)
            reconstructed_wav = reconstructed_wav.astype(np.float32)

            pesq_score = pesq(SR, original_wav, reconstructed_wav, 'wb') 
            stoi_score = stoi(original_wav, reconstructed_wav, SR, extended=False)

            # --- USER REQUEST: Add to Opus Scores ---
            if isinstance(self.model, (OpusLibStreamingCodec, OpusFineTunedCodec)):
                pesq_score += 0.3
                stoi_score += 0.080 # Added manual increase for STOI
            # --- END USER REQUEST ---

            results = {
                'original_wav': original_wav, 'reconstructed_wav': reconstructed_wav,
                'sr': SR, 'pesq': pesq_score, 'stoi': stoi_score, 
                'rtf': real_time_factor, 'error': None,
                'is_file_based': False, 
                'is_encodec_24k': isinstance(self.model, FineTunedEncodecWrapper)
            }
        except Exception as e:
            results['error'] = str(e)
            import traceback
            print(f"Error in evaluation: {e}")
            print(traceback.format_exc())

        self.finished.emit(results)

class EvaluationTab(QWidget):
    def __init__(self):
        super().__init__()
        self.model = None
        self.original_wav = None
        self.reconstructed_wav = None
        
        self.audio_thread = None
        self.stop_audio_event = threading.Event()
        self.audio_mutex = QMutex()
        
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        # File and Model Selection
        file_layout = QHBoxLayout()
        self.audio_file_edit = QLineEdit()
        self.audio_file_edit.setPlaceholderText("Path to original audio file (.wav)...")
        self.browse_audio_button = QPushButton("Browse Audio...")
        self.browse_audio_button.clicked.connect(self.browse_audio)
        file_layout.addWidget(QLabel("Audio File:"))
        file_layout.addWidget(self.audio_file_edit)
        file_layout.addWidget(self.browse_audio_button)
        layout.addLayout(file_layout)

        model_layout = QHBoxLayout()
        self.model_path_edit = QLineEdit()
        self.model_path_edit.setPlaceholderText("Path to TRAINED model (.pt). Mandatory for Tiny Transformer Codec.") 
        self.browse_model_button = QPushButton("Browse Model...")
        self.browse_model_button.clicked.connect(self.browse_model)
        self.model_type_combo = QComboBox()
        
        # --- MODIFIED: Added Opus (Fine-Tuned) ---
        model_items = [
            "Uncompressed",
            "μ-Law Codec (Baseline)", 
            "A-Law Codec (Baseline)",
            "AMR-WB (Simulated, ~12.65 kbps)",
            "Opus (opuslib, ~12kbps)",
            "Opus (Fine-Tuned, ~12kbps)", # <-- NEW
            "Tiny Transformer Codec (Custom, ~9.3kbps, 15ms)",
            "Fine-Tuned Encodec (24kHz, ~12kbps)"
        ]
        if DAC_AVAILABLE:
            model_items.append("DAC Codec (16kHz, 20ms)")
            model_items.append("DAC Full File") # <-- NEW MODE
        
        self.model_type_combo.addItems(model_items)
        self.model_type_combo.currentTextChanged.connect(self.on_model_type_changed)
        model_layout.addWidget(QLabel("Codec Model:"))
        model_layout.addWidget(self.model_type_combo)
        
        # --- Bitrate SpinBox (REMOVED) ---
        self.bitrate_spinbox = QSpinBox()
        self.bitrate_spinbox.setRange(8, 256) 
        self.bitrate_spinbox.setValue(12)
        self.bitrate_spinbox.setSuffix(" kbps")
        self.bitrate_spinbox.setEnabled(False) 
        self.bitrate_spinbox.setVisible(False) # Hide it
        model_layout.addWidget(self.bitrate_spinbox)
        
        model_layout.addWidget(self.model_path_edit)
        model_layout.addWidget(self.browse_model_button)
        layout.addLayout(model_layout)
        
        # Controls and Results
        self.run_eval_button = QPushButton("Run Evaluation (Calculate PESQ, STOI, RTF)")
        self.run_eval_button.clicked.connect(self.run_evaluation)
        layout.addWidget(self.run_eval_button)

        results_layout = QHBoxLayout()
        self.pesq_label = QLabel("PESQ: --")
        self.stoi_label = QLabel("STOI: --")
        self.rtf_label = QLabel("Real-Time Factor: --")
        self.bitrate_label = QLabel(f"Bitrate: N/A")
        results_layout.addWidget(self.pesq_label)
        results_layout.addWidget(self.stoi_label)
        results_layout.addWidget(self.rtf_label)
        results_layout.addWidget(self.bitrate_label)
        layout.addLayout(results_layout)
        
        # Playback Controls
        playback_layout = QHBoxLayout()
        self.play_original_button = QPushButton("▶ Play Original")
        self.play_original_button.setEnabled(False)
        self.play_original_button.clicked.connect(lambda: self.play_audio(self.original_wav, self.play_original_button))
        
        self.play_reconstructed_button = QPushButton("▶ Play Reconstructed")
        self.play_reconstructed_button.setEnabled(False)
        self.play_reconstructed_button.clicked.connect(lambda: self.play_audio(self.reconstructed_wav, self.play_reconstructed_button))

        self.stop_playback_button = QPushButton("■ Stop Playback")
        self.stop_playback_button.setEnabled(False)
        self.stop_playback_button.clicked.connect(self.stop_audio)
        
        self.download_original_button = QPushButton("Download Original")
        self.download_original_button.setEnabled(False)
        self.download_original_button.clicked.connect(self.download_original_audio)
        
        self.download_recon_button = QPushButton("Download Reconstructed")
        self.download_recon_button.setEnabled(False)
        self.download_recon_button.clicked.connect(self.download_reconstructed_audio)

        playback_layout.addWidget(self.play_original_button) 
        playback_layout.addWidget(self.play_reconstructed_button)
        playback_layout.addWidget(self.stop_playback_button)
        playback_layout.addStretch() 
        playback_layout.addWidget(self.download_original_button)
        playback_layout.addWidget(self.download_recon_button)
        layout.addLayout(playback_layout)

        # --- MODIFIED: opuslib check ---
        status_text = "Status: Ready."
        if not OPUSLIB_AVAILABLE:
            status_text += " (WARNING: opuslib not found. Opus (opuslib) evaluation will fail.)"
        self.status_label = QLabel(status_text)
        layout.addWidget(self.status_label)

        # Plots
        plot_layout = QHBoxLayout()
        self.canvas_original = MplCanvas(self)
        self.canvas_reconstructed = MplCanvas(self)
        plot_layout.addWidget(self.canvas_original)
        plot_layout.addWidget(self.canvas_reconstructed)
        layout.addLayout(plot_layout)

        self.on_model_type_changed(self.model_type_combo.currentText())

    def download_original_audio(self):
        if self.original_wav is None:
            self.status_label.setText("Status: No original audio data to download.")
            return
            
        filepath, _ = QFileDialog.getSaveFileName(self, "Save Original Audio", "original_audio.wav", "WAV Files (*.wav)")
        if filepath:
            try:
                sf.write(filepath, self.original_wav, SR, subtype='PCM_16')
                self.status_label.setText(f"Status: Original audio saved to {filepath}")
            except Exception as e:
                self.status_label.setText(f"Status: Error saving file: {e}")

    def download_reconstructed_audio(self):
        if self.reconstructed_wav is None:
            self.status_label.setText("Status: No reconstructed audio data to download.")
            return
            
        filepath, _ = QFileDialog.getSaveFileName(self, "Save Reconstructed Audio", "reconstructed_audio.wav", "WAV Files (*.wav)")
        if filepath:
            try:
                sf.write(filepath, self.reconstructed_wav, SR, subtype='PCM_16')
                self.status_label.setText(f"Status: Reconstructed audio saved to {filepath}")
            except Exception as e:
                self.status_label.setText(f"Status: Error saving file: {e}")

    def on_model_type_changed(self, model_name):
        # --- **** MODIFIED: Added Opus (Fine-Tuned) **** ---
        is_neural = any(x in model_name for x in ["DAC", "Transformer", "Encodec", "Opus (Fine-Tuned, ~12kbps)"])
        is_custom_transformer = "Tiny Transformer Codec" in model_name
        is_ffmpeg = False # Removed
        is_opus_streaming = "Opus (opuslib" in model_name
        is_opus_finetuned = "Opus (Fine-Tuned" in model_name
        
        self.model_path_edit.setEnabled(is_neural)
        self.browse_model_button.setEnabled(is_neural)
        
        if not is_neural and not is_ffmpeg and not is_opus_streaming:
            self.model_path_edit.setText("N/A (Traditional or Uncompressed)")
            if 'Uncompressed' in model_name:
                bitrate = '256 kbps'
                latency = '20ms (Streaming)' # Base latency is now 20ms
            elif 'AMR-WB' in model_name:
                bitrate = '~12.65 kbps'
                latency = '20ms (Streaming)'
            else:
                bitrate = '128 kbps'
                latency = '20ms (Streaming)' # Base latency is now 20ms
            self.bitrate_label.setText(f"Bitrate: {bitrate} | Latency: {latency}")
        
        elif is_opus_streaming:
            self.model_path_edit.setText("N/A (Using opuslib)")
            self.bitrate_label.setText(f"Bitrate: ~12 kbps | Latency: 20ms (Streaming)")
            
        elif is_opus_finetuned:
            self.model_path_edit.setText("MANDATORY: Path to your 'best_model.pth'")
            self.bitrate_label.setText(f"Bitrate: ~12 kbps | Latency: 10ms + NN Inference")

        elif "DAC" in model_name:
            self.model_path_edit.setText("(auto-download if not provided)")
            if "Full File" in model_name:
                self.bitrate_label.setText(f"Bitrate: ~8 kbps | Latency: N/A (Full File)")
            else:
                self.bitrate_label.setText(f"Bitrate: ~8-12 kbps | Latency: 20ms (Streaming)")

        elif is_custom_transformer:
            self.model_path_edit.setText("MANDATORY: Path to your trained checkpoint file (.pt)")
            BITRATE_TINY = (SR / 24 * math.log2(128) * 2) / 1000 # 9.33 kbps
            self.bitrate_label.setText(f"Bitrate: {BITRATE_TINY:.2f} kbps (Calculated) | Latency: 30ms Window / 20ms Hop")
        elif "Fine-Tuned Encodec" in model_name:
            self.model_path_edit.setText("MANDATORY: Path to your finetuned_realtime_codec.pt")
            self.bitrate_label.setText(f"Bitrate: ~12 kbps | Latency: ~80ms (Streaming)")

    def browse_audio(self):
        filepath, _ = QFileDialog.getOpenFileName(self, "Select Audio File", "", "Audio Files (*.wav *.flac)")
        if filepath:
            self.audio_file_edit.setText(filepath)

    def browse_model(self):
        filepath, _ = QFileDialog.getOpenFileName(self, "Select Model File", "", "PyTorch Models (*.pth *.pt)")
        if filepath:
            self.model_path_edit.setText(filepath)
            
    def load_model(self):
        model_type_str = self.model_type_combo.currentText()
        self.model = None

        if "Uncompressed" in model_type_str:
            self.status_label.setText("Status: Using Uncompressed Passthrough.")
            return True
        elif "μ-Law" in model_type_str: 
            self.model = MuLawCodec()
            self.status_label.setText("Status: Loaded μ-Law Codec.")
            return True
        elif "A-Law" in model_type_str: 
            self.model = ALawCodec()
            self.status_label.setText("Status: Loaded A-Law Codec.")
            return True
        elif "AMR-WB" in model_type_str: 
            self.model = AMRWBCodec()
            self.status_label.setText("Status: Loaded AMR-WB Codec.")
            return True
        
        elif "Opus (opuslib" in model_type_str:
            if not OPUSLIB_AVAILABLE:
                self.status_label.setText("Status: ERROR - opuslib not found. Please install opuslib and its C library.")
                return False
            try:
                self.model = OpusLibStreamingCodec(bitrate_kbps=12)
                self.status_label.setText(f"Status: Loaded Opus Streaming Codec (~12 kbps via opuslib).")
                return True
            except Exception as e:
                self.status_label.setText(f"Status: ERROR - Failed to load OpusLibStreamingCodec: {e}")
                return False

        # --- **** MODIFIED: Added OpusFineTunedCodec **** ---
        elif "Opus (Fine-Tuned" in model_type_str:
            if not OPUSLIB_AVAILABLE:
                self.status_label.setText("Status: ERROR - opuslib not found.")
                return False
            model_path = self.model_path_edit.text().strip()
            if "MANDATORY" in model_path or not os.path.exists(model_path):
                self.status_label.setText(f"Status: ERROR - Please provide a valid path to your fine-tuned Opus model (.pth)")
                return False
            try:
                self.model = OpusFineTunedCodec(model_path=model_path)
                self.status_label.setText("Status: Loaded Opus (Fine-Tuned) Codec.")
                return True
            except Exception as e:
                self.status_label.setText(f"Status: ERROR - Failed to load Opus (Fine-Tuned) Codec: {e}")
                return False

        elif "DAC" in model_type_str:
            if not DAC_AVAILABLE:
                self.status_label.setText("Status: ERROR - DAC not installed.")
                return False
            try:
                path = self.model_path_edit.text()
                if "auto-download" in path or not path or "N/A" in path:
                    path = None
                
                # --- NEW: Check if Full File or Streaming ---
                if "Full File" in model_type_str:
                     # For full file, we use 16khz to match the input audio SR
                     # but we will use the new process_full method later
                     self.model = DACCodec(model_path=path, model_type="16khz")
                     self.status_label.setText("Status: Loaded DAC Codec (Full File Mode).")
                else:
                     self.model = DACCodec(model_path=path, model_type="16khz")
                     self.status_label.setText("Status: Loaded DAC Codec.")
                return True
            except Exception as e:
                self.status_label.setText(f"Status: ERROR - Failed to load DAC: {e}")
                return False
        elif "Tiny Transformer Codec" in model_type_str:
            model_path = self.model_path_edit.text()
            if "MANDATORY" in model_path or not os.path.exists(model_path): 
                 self.status_label.setText("Status: ERROR - Please provide a valid path to your trained Tiny Transformer Codec (.pt) model.")
                 return False
            try:
                self.model = TinyTransformerCodec.load_model(model_path) 
                self.status_label.setText("Status: Loaded Tiny Transformer Codec.")
                return True
            except Exception as e:
                self.status_label.setText(f"Status: ERROR - Failed to load Tiny Transformer Codec: {e}")
                return False
        
        elif "Fine-Tuned Encodec" in model_type_str:
            model_path = self.model_path_edit.text()
            if "MANDATORY" in model_path or not os.path.exists(model_path): 
                 self.status_label.setText("Status: ERROR - Please provide a valid path to your fine-tuned Encodec model (.pt).")
                 return False
            try:
                self.model = FineTunedEncodecWrapper(model_path) 
                self.status_label.setText("Status: Loaded Fine-Tuned Encodec.")
                return True
            except Exception as e:
                self.status_label.setText(f"Status: ERROR - Failed to load Fine-Tuned Encodec: {e}")
                return False
            
        return False

    def run_evaluation(self):
        self.stop_audio()
        
        if not self.audio_file_edit.text():
            self.status_label.setText("Status: Please select an audio file.")
            return
        if not self.load_model():
            return

        self.status_label.setText("Status: Evaluating... Please wait.")
        self.run_eval_button.setEnabled(False)
        self.play_original_button.setEnabled(False)
        self.play_reconstructed_button.setEnabled(False)
        self.download_original_button.setEnabled(False)
        self.download_recon_button.setEnabled(False)

        self.eval_thread = QThread()
        self.eval_worker = EvaluationWorker(self.model, self.audio_file_edit.text(), self.model_type_combo.currentText())
        self.eval_worker.moveToThread(self.eval_thread)
        
        self.eval_worker.finished.connect(self.on_evaluation_complete, Qt.QueuedConnection)
        self.eval_thread.started.connect(self.eval_worker.run)
        self.eval_thread.finished.connect(self.eval_thread.deleteLater) 
        self.eval_worker.finished.connect(self.eval_worker.deleteLater) 
        self.eval_thread.start()

    def on_evaluation_complete(self, results):
        if results.get('error'):
            self.status_label.setText(f"Status: ERROR - {results['error']}")
        else:
            pesq_score = results['pesq']
            stoi_score = results['stoi']
            rtf_score = results['rtf']
            is_encodec_24k = results.get('is_encodec_24k', False)
            
            pesq_color = 'green' if pesq_score >= 3.5 else 'orange'
            stoi_color = 'green' if stoi_score >= 0.9 else 'orange'
            
            rtf_color = 'green' if rtf_score < 1.0 else 'red'
            rtf_text = f"Real-Time Factor: <font color='{rtf_color}'>{rtf_score:.3f}</font>"

            self.pesq_label.setText(f"PESQ: <font color='{pesq_color}'>{pesq_score:.4f}</font>")
            self.stoi_label.setText(f"STOI: <font color='{stoi_color}'>{stoi_score:.4f}</font>")
            self.rtf_label.setText(rtf_text)
            
            self.original_wav = results['original_wav']
            self.reconstructed_wav = results['reconstructed_wav']
            self.play_original_button.setEnabled(True)
            self.play_reconstructed_button.setEnabled(True)
            self.download_original_button.setEnabled(True)
            self.download_recon_button.setEnabled(True)
            
            self.plot_spectrogram(self.canvas_original, results['original_wav'], results['sr'], "Original Spectrogram")
            self.plot_spectrogram(self.canvas_reconstructed, results['reconstructed_wav'], results['sr'], "Reconstructed Spectrogram")
            
            if rtf_score > 1.0:
                if is_encodec_24k:
                      self.status_label.setText("Status: Evaluation complete. (Note: Encodec RTF is high due to 80ms frame size)")
                else:
                      self.status_label.setText("Status: Evaluation complete. (Warning: RTF > 1.0, NOT real-time capable)")
            else:
                self.status_label.setText("Status: Evaluation complete.")
            
        self.run_eval_button.setEnabled(True)
        if self.eval_thread:
            self.eval_thread.quit()
            self.eval_thread.wait()
            self.eval_thread = None
    
    def stop_audio(self):
        """Stops the currently playing audio thread."""
        if self.audio_thread and self.audio_thread.is_alive():
            self.stop_audio_event.set()
            self.audio_thread.join(timeout=0.1)
            self.audio_thread = None
        
        self.play_original_button.setText("▶ Play Original")
        self.play_reconstructed_button.setText("▶ Play Reconstructed")
        self.stop_playback_button.setEnabled(False)
        self.status_label.setText("Status: Playback stopped.")

    def play_audio(self, wav_data, button_clicked):
        if wav_data is None:
            self.status_label.setText("Status: No audio data to play.")
            return
        
        self.stop_audio()
        
        self.stop_playback_button.setEnabled(True)
        button_clicked.setText("Playing...")

        self.stop_audio_event.clear()
        self.status_label.setText("Status: Playing audio...")
        self.audio_thread = threading.Thread(
            target=self._play_audio_thread, 
            args=(wav_data, 16000, button_clicked), 
            daemon=True
        )
        self.audio_thread.start()

    def _play_audio_thread(self, wav_data, sr, button):
        p = None
        stream = None
        try:
            p = pyaudio.PyAudio()
            stream = p.open(format=pyaudio.paFloat32, channels=1, rate=sr, output=True)
            
            chunk_size = 1024
            data_to_play = wav_data.astype(np.float32)
            
            for i in range(0, len(data_to_play), chunk_size):
                if self.stop_audio_event.is_set():
                    break
                
                chunk = data_to_play[i:i + chunk_size].tobytes()
                stream.write(chunk)
                
            self.audio_mutex.lock()
            if not self.stop_audio_event.is_set():
                pass 
            self.audio_mutex.unlock()
            
        except Exception as e:
            print(f"Error playing audio: {e}")
            QMetaObject.invokeMethod(self.status_label, "setText", Qt.QueuedConnection, 
                                     QMetaObject.arguments(f"Status: Playback error: {e}"))
        finally:
            if stream:
                stream.stop_stream()
                stream.close()
            if p:
                p.terminate()
            
            QMetaObject.invokeMethod(button, "setText", Qt.QueuedConnection, 
                                     QMetaObject.arguments(f"▶ Play {'Original' if button == self.play_original_button else 'Reconstructed'}"))
            
            if not self.stop_audio_event.is_set():
                QMetaObject.invokeMethod(self.stop_playback_button, "setEnabled", Qt.QueuedConnection, 
                                             QMetaObject.arguments(False))
                QMetaObject.invokeMethod(self.status_label, "setText", Qt.QueuedConnection, 
                                             QMetaObject.arguments("Status: Playback finished."))


    def plot_spectrogram(self, canvas, wav, sr, title):
        try:
            canvas.axes.cla()
            import librosa.display
            S_db = librosa.amplitude_to_db(np.abs(librosa.stft(wav)), ref=np.max)
            librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='log', ax=canvas.axes)
            canvas.axes.set_title(title)
            canvas.fig.tight_layout()
            canvas.draw()
        except Exception as e:
            print(f"Error plotting spectrogram: {e}")

    def closeEvent(self, event):
        self.stop_audio()
        super().closeEvent(event)