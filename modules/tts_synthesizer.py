from gtts import gTTS
import os

class TTSSynthesizer:
    def __init__(self, output_path="data/output_audio.wav"):
        self.output_path = output_path

    def synthesize(self, text):
        print(f"[4] TTS Synthesizer: Converting text back to audio...")
        
        # Using gTTS (Google TTS) for immediate, out-of-the-box results.
        # You can later swap this for Bark or Coqui TTS to see neural vocoders in action.
        tts = gTTS(text=text, lang='en', slow=False)
        tts.save(self.output_path)
        
        print(f"    -> Audio saved successfully to {self.output_path}")
        return self.output_path