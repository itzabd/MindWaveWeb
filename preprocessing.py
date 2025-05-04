"""
EEG Preprocessing Pipeline (CSP-Free Version)
1. Loads EEG data
2. Extracts frequency band features
3. Saves in OpenAI-compatible JSON/JSONL format
"""

import os
import json
import pandas as pd
import numpy as np
from feature_extraction import load_eeg_data
from csv_to_json_4o import csv_to_json, json_to_jsonl

def pipeline(csv_path, json_path, jsonl_path, window_size=1000, selected_columns=None):
    """
    Simplified pipeline without CSP
    """
    try:
        # Load data
        eeg_data, labels = load_eeg_data(csv_path)

        # Default frequency bands if not specified
        if selected_columns is None:
            selected_columns = [
                [0, [(10, 12), (12, 14)]],  # FCz
                [2, [(20, 22), (22, 24)]],  # C3
                [3, [(8, 10)]],              # Cz
                [4, [(20, 22), (22, 24)]],  # C4
                [5, [(28, 30)]]              # CP3
            ]

        # Convert to JSON format (pass None for CSP)
        json_data = csv_to_json(eeg_data, None, window_size, selected_columns, labels)

        # Ensure output directories exist
        os.makedirs(os.path.dirname(json_path), exist_ok=True)
        os.makedirs(os.path.dirname(jsonl_path), exist_ok=True)

        # Save outputs
        with open(json_path, 'w') as f:
            json.dump(json_data, f, indent=4)
        json_to_jsonl(json_path, jsonl_path)

        print(f"\nSuccessfully processed {os.path.basename(csv_path)}")
        print(f"- JSON output: {json_path}")
        print(f"- JSONL output: {jsonl_path}")
        print(f"- Processed trials: {len(eeg_data) // window_size}")

    except Exception as e:
        print(f"\nError processing {csv_path}:")
        print(f"Type: {type(e).__name__}")
        print(f"Details: {str(e)}")
        raise

def main():
    """Configuration and execution"""
    BASE_PATH = ''

    datasets = {
        'train': {
            'csv': os.path.join(BASE_PATH, 'k3b_laf_eeg_data_ch9_all_labels_train.csv'),
            'json': os.path.join(BASE_PATH, 'json/k3b_train.json'),
            'jsonl': os.path.join(BASE_PATH, 'jsonl/k3b_train.jsonl')
        },
        'val': {
            'csv': os.path.join(BASE_PATH, 'k3b_laf_eeg_data_ch9_all_labels_val.csv'),
            'json': os.path.join(BASE_PATH, 'json/k3b_val.json'),
            'jsonl': os.path.join(BASE_PATH, 'jsonl/k3b_val.jsonl')
        }
    }

    for name, paths in datasets.items():
        print(f"\n{'='*40}")
        print(f"Processing {name} dataset")
        print(f"{'='*40}")

        pipeline(
            csv_path=paths['csv'],
            json_path=paths['json'],
            jsonl_path=paths['jsonl'],
            window_size=1000
        )

if __name__ == '__main__':
    main()