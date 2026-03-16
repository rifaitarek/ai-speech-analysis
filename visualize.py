import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os

def generate_visualizations(audio_path):
    if not os.path.exists(audio_path):
        print(f"Error: {audio_path} not found.")
        return

    # 1. Load the audio
    y, sr = librosa.load(audio_path)
    
    # 2. Compute transformations
    # Short-Time Fourier Transform (Spectrogram)
    D = librosa.stft(y)
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    
    # Mel Spectrogram
    M = librosa.feature.melspectrogram(y=y, sr=sr)
    M_db = librosa.power_to_db(M, ref=np.max)
    
    # MFCCs (What the HMM sees)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

    # 3. Plotting
    fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(10, 12))
    fig.suptitle(f"ChronoVoice Audio Analysis: {os.path.basename(audio_path)}", fontsize=16)

    # Panel 1: Waveform
    librosa.display.waveshow(y, sr=sr, ax=axes[0], color='steelblue')
    axes[0].set(title="1. Raw Waveform (Time vs Amplitude)")

    # Panel 2: Spectrogram
    img2 = librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='hz', ax=axes[1])
    axes[1].set(title="2. Linear Spectrogram (Frequency vs Time)")
    fig.colorbar(img2, ax=axes[1], format="%+2.0f dB")

    # Panel 3: Mel Spectrogram
    img3 = librosa.display.specshow(M_db, sr=sr, x_axis='time', y_axis='mel', ax=axes[2])
    axes[2].set(title="3. Mel Spectrogram (Human-Perceived Frequency)")
    fig.colorbar(img3, ax=axes[2], format="%+2.0f dB")

    # Panel 4: MFCCs
    img4 = librosa.display.specshow(mfccs, sr=sr, x_axis='time', ax=axes[3])
    axes[3].set(title="4. MFCCs (The 'Fingerprint' used by HMM)")
    fig.colorbar(img4, ax=axes[3])

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    output_path = "data/analysis.png"
    plt.savefig(output_path)
    print(f"Success! Analysis image saved to {output_path}")

if __name__ == "__main__":
    generate_visualizations("data/test_audio.wav")