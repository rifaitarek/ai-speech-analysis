import os
from dotenv import load_dotenv

# 1. New Import
from visualize import generate_visualizations 

# Load environment variables
load_dotenv() 

from modules.acoustic_frontend import AcousticFrontend
from modules.hmm_recognizer import HMMRecognizer
from modules.llm_brain import LLMBrain
from modules.tts_synthesizer import TTSSynthesizer

def main():
    print("=== Starting ChronoVoice Pipeline ===")
    
    input_audio = "data/test_audio.wav"
    if not os.path.exists(input_audio):
        print(f"ERROR: Please place a .wav file at '{input_audio}' before running.")
        return

    # --- New Visualization Step ---
    print("[0] Visualization: Generating signal analysis plots...")
    generate_visualizations(input_audio)
    # ------------------------------

    # Initialize modules
    frontend = AcousticFrontend()
    hmm = HMMRecognizer(n_components=5)
    llm = LLMBrain()
    tts = TTSSynthesizer()

    # Step 1: Raw Audio -> Mel Features
    features = frontend.process_audio(input_audio)

    # Step 2: Mel Features -> Text via HMM
    transcription = hmm.decode(features)

    # Step 3: Text -> Response via LLM
    response_text = llm.generate_response(transcription)

    # Step 4: Text Response -> Synthetic Speech
    output_audio_path = tts.synthesize(response_text)

    print(f"=== Pipeline Complete! View the plots at data/analysis.png ===")

if __name__ == "__main__":
    main()