import os
from google import genai
from dotenv import load_dotenv

# Ensure environment variables are loaded if not already
load_dotenv()

class LLMBrain:
    def __init__(self, model_type="gemini"):
        self.model_type = model_type
        
        if self.model_type == "gemini":
            api_key = os.environ.get("GEMINI_API_KEY")
            
            if not api_key:
                print("    -> WARNING: GEMINI_API_KEY not found. Using mock mode.")
                self.model_type = "mock"
            else:
                print("    -> Initializing Modern Gemini API...")
                # Modern Client Syntax
                self.client = genai.Client(api_key=api_key)
                self.model_id = "gemini-2.0-flash"

    def generate_response(self, text_prompt):
        print(f"[3] LLM Brain: Processing prompt: '{text_prompt}'")
        
        if self.model_type == "gemini":
            try:
                # Optimized for speed
                response = self.client.models.generate_content(
                    model='gemini-2.0-flash', 
                    contents=f"Keep it short: {text_prompt}"
                )
                output = response.text.strip()
            except Exception as e:
                print(f"    -> API Error: {e}")
                output = "Connection to the neural hub failed. Reverting to local fallback."
        else:
            output = "Did you know that audio sampling at 44.1kHz was chosen because of early digital video tape compatibility?"
            
        print(f"    -> LLM Generated Response: '{output}'")
        return output