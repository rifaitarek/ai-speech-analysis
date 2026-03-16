# ChronoVoice

An evolutionary speech-to-speech pipeline bridging classical speech processing with modern AI. 

## Concepts Covered
1. **Acoustic Frontend:** Fast Fourier Transforms (FFT) and Mel-frequency cepstral coefficients (MFCCs).
2. **Classical Speech Recognition:** Hidden Markov Models (HMM) and Viterbi decoding.
3. **Generative AI:** Modern LLM prompting.
4. **Speech Synthesis:** Text-to-Speech generation.

## How to run
1. Clone the repository.
2. Install requirements: `pip install -r requirements.txt`
3. Place any short `.wav` file in the `data/` folder and name it `test_audio.wav`.
4. Run the pipeline: `python main.py`