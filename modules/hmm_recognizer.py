from hmmlearn import hmm
import numpy as np

class HMMRecognizer:
    def __init__(self, n_components=3):
        # n_components represents hidden states (e.g., phonemes in a word)
        self.model = hmm.GaussianHMM(n_components=n_components, covariance_type="diag", n_iter=100)
        self.is_trained = False

    def train_mock_model(self, features):
        """Trains the HMM on the provided features (mocking the training phase)."""
        print("[2] HMM Recognizer: Training Gaussian HMM on acoustic features...")
        self.model.fit(features)
        self.is_trained = True
        print("    -> HMM Transition Matrix Learned:")
        print(np.round(self.model.transmat_, 2))

    def decode(self, features):
        """Decodes features into hidden states and maps to text."""
        if not self.is_trained:
            self.train_mock_model(features)
            
        # Viterbi algorithm to find the most likely sequence of hidden states
        logprob, state_sequence = self.model.decode(features, algorithm="viterbi")
        print(f"    -> Decoded hidden state sequence (first 10 frames): {state_sequence[:10]}")
        
        # MOCK TRANSLATION: In a real system, state sequences map to a dictionary.
        # We return a hardcoded string here to keep the pipeline moving.
        mock_transcription = "Tell me a fun fact about audio."
        print(f"    -> HMM Predicted Text: '{mock_transcription}'")
        return mock_transcription