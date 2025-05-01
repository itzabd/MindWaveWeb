import numpy as np
import pandas as pd
from scipy import signal
from sklearn.preprocessing import StandardScaler
import pygdf  # For GDF file support
import json
from datetime import datetime
import os


class EEGProcessor:
    def __init__(self, localai_client):
        self.localai = localai_client

    def load_gdf(self, filepath):
        """Load GDF EEG data file"""
        try:
            # Using pygdf to load GDF files
            eeg_data = pygdf.read_gdf(filepath)
            return eeg_data
        except Exception as e:
            raise Exception(f"Error loading GDF file: {str(e)}")

    def preprocess(self, eeg_data):
        """Preprocess EEG data"""
        # Convert to numpy array if not already
        data = eeg_data.to_numpy() if hasattr(eeg_data, 'to_numpy') else eeg_data

        # 1. Filtering (bandpass 8-30 Hz)
        b, a = signal.butter(4, [8, 30], btype='bandpass', fs=250)
        filtered = signal.filtfilt(b, a, data, axis=0)

        # 2. Normalization
        scaler = StandardScaler()
        normalized = scaler.fit_transform(filtered)

        # 3. Feature extraction (simple statistical features)
        features = {
            'mean': np.mean(normalized, axis=1),
            'std': np.std(normalized, axis=1),
            'max': np.max(normalized, axis=1),
            'min': np.min(normalized, axis=1)
        }

        return features

    def classify(self, features):
        """Classify EEG data using LocalAI"""
        prompt = {
            "input": features,
            "instruction": "Classify this EEG data into one of four categories: "
                           "1. left hand, 2. right hand, 3. tongue, 4. foot movement. "
                           "Return only the classification number (1-4) and confidence percentage."
        }

        try:
            response = self.localai.generate(
                model=os.getenv("EEG_MODEL_NAME"),
                prompt=json.dumps(prompt),
                max_tokens=50
            )
            return self._parse_response(response)
        except Exception as e:
            raise Exception(f"LocalAI classification error: {str(e)}")

    def _parse_response(self, response):
        """Parse LocalAI response"""
        try:
            result = json.loads(response['choices'][0]['text'])
            return {
                'classification': result.get('classification', 'unknown'),
                'confidence': result.get('confidence', 0),
                'raw_response': response
            }
        except:
            return {
                'classification': 'unknown',
                'confidence': 0,
                'raw_response': response
            }