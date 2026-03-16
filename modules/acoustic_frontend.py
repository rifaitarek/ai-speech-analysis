import librosa
import numpy as np

class AcousticFrontend:
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate

    def process_audio(self, file_path):
        """Loads audio and extracts Mel-frequency cepstral coefficients (MFCCs)."""
        print("[1] Acoustic Frontend: Loading audio and applying Fourier/Mel transforms...")
        
        # Load raw waveform
        y, sr = librosa.load(file_path, sr=self.sample_rate)
        
        # 1. Short-Time Fourier Transform (STFT)
        stft = np.abs(librosa.stft(y))
        
        # 2. Mel Spectrogram mapping
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, S=stft**2)
        
        # 3. Extract MFCCs (The actual features used by HMMs)
        mfccs = librosa.feature.mfcc(S=librosa.power_to_db(mel_spec), n_mfcc=13)
        
        print(f"    -> Extracted MFCC shape: {mfccs.shape} (Features x Time Frames)")
        return mfccs.T # Transpose for hmmlearn (Samples x Features)