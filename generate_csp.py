# generate_csp.py
import numpy as np
import pandas as pd
from mne.decoding import CSP
import os


def generate_csp(eeg_path, output_path, class_a=3, class_b=4):  # Changed default classes to match your data
    # Load data
    try:
        df = pd.read_csv(eeg_path)
        print(f"Loaded data shape: {df.shape}")
    except FileNotFoundError:
        print(f"Error: File {eeg_path} not found")
        return

    # Verify label column exists
    if 'label' not in df.columns:
        print("Error: 'label' column not found in the data")
        return

    # Separate features and labels
    X = df.drop(columns=['label']).values
    y = df['label'].values.astype(int)  # Convert to integers

    # Calculate number of complete trials (1000 samples each)
    n_trials = len(X) // 1000
    if n_trials == 0:
        print("Error: Not enough samples for even one complete trial (need 1000 samples per trial)")
        return

    X = X[:n_trials * 1000]
    y = y[:n_trials * 1000]

    # Reshape to trials x channels x time
    X_reshaped = X.reshape(n_trials, 1000, -1).transpose(0, 2, 1)

    # Get most common label for each trial
    trial_labels = np.array([np.argmax(np.bincount(y[i * 1000:(i + 1) * 1000]))
                             for i in range(n_trials)])

    # Check available classes
    unique_classes = np.unique(trial_labels)
    print(f"Available classes in data: {unique_classes}")

    # Verify our target classes exist
    if class_a not in unique_classes or class_b not in unique_classes:
        print(f"Error: Requested classes {class_a} and/or {class_b} not found in data")
        print(f"Available classes: {unique_classes}")
        print("Try one of these class combinations:")
        for i in range(len(unique_classes)):
            for j in range(i + 1, len(unique_classes)):
                print(f"- Class {unique_classes[i]} vs Class {unique_classes[j]}")
        return

    # Create mask for selected classes
    mask = (trial_labels == class_a) | (trial_labels == class_b)
    X_filtered = X_reshaped[mask]
    y_filtered = trial_labels[mask]

    # Check we have enough trials
    if len(X_filtered) < 2:
        print(f"Error: Not enough trials for CSP (need ≥2, got {len(X_filtered)})")
        print(f"Try different class combinations from: {unique_classes}")
        return

    print(f"Processing {len(X_filtered)} trials for classes {class_a} vs {class_b}")

    # Generate CSP features
    csp = CSP(n_components=min(4, len(X_filtered)), reg=None, log=True, norm_trace=False)
    try:
        csp_features = csp.fit_transform(X_filtered, y_filtered)
    except ValueError as e:
        print(f"CSP Error: {str(e)}")
        return

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save features
    pd.DataFrame(csp_features).to_csv(output_path, index=False)
    print(f"Successfully generated {output_path} with {len(csp_features)} features")


if __name__ == "__main__":
    # Modified to use classes that exist in your data (1-4)
    generate_csp(
        eeg_path='laf_eeg_data_9ch_360000_train.csv',
        output_path='csp4/class_3_vs_4_train_features.csv',  # Changed output filename
        class_a=3,  # Using classes that exist
        class_b=4  # Using classes that exist
    )

    generate_csp(
        eeg_path='laf_eeg_data_9ch_360000_val.csv',
        output_path='csp4/class_3_vs_4_val_features.csv',  # Changed output filename
        class_a=3,  # Using classes that exist
        class_b=4  # Using classes that exist
    )