import sys
import subprocess
import importlib.util
import os

# Create models cache directory
MODELS_CACHE_DIR = os.path.join(os.path.expanduser("~"), ".audio_codec_models")
if not os.path.exists(MODELS_CACHE_DIR):
    os.makedirs(MODELS_CACHE_DIR)
    print(f"Created models cache directory: {MODELS_CACHE_DIR}")

# --- Package Installation ---
REQUIRED_PACKAGES = {
    'PyQt5': 'PyQt5',
    'torch': 'torch==2.1.0',
    'torchaudio': 'torchaudio==2.1.0', 
    'numpy': 'numpy',
    'pyaudio': 'pyaudio',
    'scipy': 'scipy',
    'librosa': 'librosa',
    'matplotlib': 'matplotlib',
    'pystoi': 'pystoi',
    'pesq': 'pesq[speechmetrics]', 
    'soundfile': 'soundfile',
    'dac': 'descript-audio-codec', 
    'einops': 'einops',
    'pandas': 'pandas',
    'gdown': 'gdown',
    'transformers': 'transformers==4.35.0', 
    'accelerate': 'accelerate==0.24.1', 
    'safetensors': 'safetensors',
    'tokenizers': 'tokenizers==0.14.1', 
    'huggingface-hub': 'huggingface-hub==0.17.3', 
    'opuslib': 'opuslib', # <-- Using opuslib as requested
    # sounddevice is not needed by pyaudio
}


def check_and_install_packages():
    """Checks and installs required packages, ensuring specific versions."""
    print("Checking and installing required packages...")
    
    for module_name, install_string in REQUIRED_PACKAGES.items():
        
        print(f"Ensuring package '{install_string}' is installed...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", install_string])
            print(f"Successfully processed '{install_string}'.")
        except subprocess.CalledProcessError as e:
            print(f"ERROR: Failed to install '{install_string}'. Error: {e}")
            if module_name not in ['einops']: 
                sys.exit(1)
        except Exception as e:
            print(f"ERROR: An unexpected error occurred during installation of '{install_string}'. Error: {e}")
            if module_name not in ['einops']:
                sys.exit(1)

# --- Model Downloader ---
def check_and_download_models():
    """
    Checks for required model files and downloads them from Google Drive
    using gdown if they are missing.
    """
    print("Checking for required model files...")
    
    models_to_check = {
        # "best_model.pth": "1IOggsjQ-AtmBGUhXVrrmgerb45zGHxFk",
        # "latest_model.pth": "1ru7ixDdkZiEDgQy5ss8s8gSoPupAxZj-",
        "tiny_transformer_best.pt": "16ChGzs6MR8PcGHYmFKkW_WsaWEUjegsn",
        "opus-realtime-10-framesize.pth.pth" : "1YGV3Smt-h6qR-cguMbLVZp9wZeUKUhMV",
        "libopus-0.dll" : "1gSPw2fqC5P5VWU1yl3wNVEn_VyWmhXvG",
        "opus.dll" : "1apJ6vx4gj4Wm5ZjrExR0bMRSaImyBL6D",
        "final_model_realtime.pt" : "1VSe1T5dmaN0yUSMWnESmSPtb7cFT4GEO"

    }

    for filename, file_id in models_to_check.items():
        if not os.path.exists(filename):
            print(f"Model file '{filename}' not found. Downloading from Google Drive...")
            print(f"(This may take a moment... ID: {file_id})")
            
            try:
                subprocess.check_call([
                    sys.executable, "-m", "gdown",
                    file_id, "-O", filename
                ])
                print(f"Successfully downloaded '{filename}'.")
            except subprocess.CalledProcessError as e:
                print(f"ERROR: Failed to download '{filename}'. Error: {e}")
                sys.exit(1) 
            except FileNotFoundError:
                print("ERROR: The 'gdown' package was not found.")
                sys.exit(1)
        else:
            print(f"Found existing model: '{filename}'")

# --- END: New Model Downloader ---
    

# Run the package check
print("="*70)
print("Audio Codec Suite - Automatic Setup")
print("="*70)
check_and_install_packages()

# --- NEW: Run the model downloader ---
print("="*70)
check_and_download_models()
# ---
 
print("="*70)
print("Setup complete! Starting application...")
print("="*70 + "\n")


# --- REMOVED: PYOGG PATH HACK ---


try:
    from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QTabWidget
    # These imports will now happen *after* the path is fixed
    from streaming_tab import StreamingTab
    from evaluation_tab import EvaluationTab
    from detailed_analysis_tab import DetailedAnalysisTab 
except ImportError as e:
    print(f"Failed to import a required module: {e}")
    print("Please ensure all packages from REQUIRED_PACKAGES are installed.")
    sys.exit(1)


class MainWindow(QMainWindow):
    """The main application window which holds the tabbed interface."""
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Ultra Low-Latency Audio Codec Suite")
        self.setGeometry(100, 100, 1050, 800) 

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)

        self.tabs = QTabWidget()
        self.layout.addWidget(self.tabs)

        self.streaming_tab = StreamingTab()
        self.evaluation_tab = EvaluationTab()
        self.detailed_analysis_tab = DetailedAnalysisTab() 

        self.tabs.addTab(self.streaming_tab, "Real-Time Streaming")
        self.tabs.addTab(self.evaluation_tab, "Model Evaluation")
        self.tabs.addTab(self.detailed_analysis_tab, "Detailed Comparative Analysis") 
        
        # Show available codecs in status bar
        self.statusBar().showMessage(self._get_codec_status())
        
    def _get_codec_status(self):
        """Check which codecs are available"""
        status_parts = []
        
        try:
            import dac
            status_parts.append("DAC ✓")
        except:
            status_parts.append("DAC ✗")
            
        try:
            import pandas
            status_parts.append("Pandas ✓")
        except:
            status_parts.append("Pandas ✗")
        
        # --- MODIFIED OPUSLIB CHECK ---
        try:
            import opuslib
            status_parts.append("Opus(opuslib) ✓")
        except Exception: # Catch any error
            status_parts.append("Opus(opuslib) ✗")
        # --- END MODIFIED OPUSLIB CHECK ---
            
        return "System Status: " + " | ".join(status_parts)
        
    def closeEvent(self, event):
        """Ensures background threads are terminated when the application is closed."""
        print("Closing application...")
        self.streaming_tab.stop_all_threads()
        event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())