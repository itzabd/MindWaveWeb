import numpy as np
import pandas as pd
from mne.decoding import CSP
from itertools import combinations
import os
from joblib import dump, load

# Configuration
WINDOW_SIZE = 1000  # 4 seconds at 250Hz
N_COMPONENTS = 2  # CSP components per pair
CLASSES = [1, 2, 3, 4, 5]  # Your class labels
CLASS_PAIRS = list(combinations(CLASSES, 2))  # 10 pairs for 5 classes


def generate_csp_features(input_csv, output_dir, split_name, csp_models_dir=None):
    """
    Generate both merged and individual 1vs1 CSP features.

    Args:
        input_csv: Path to EEG CSV
        output_dir: Directory to save features
        split_name: 'train', 'val', or 'test'
        csp_models_dir: Directory for CSP models (required for val/test)
    """
    # Load data
    data = pd.read_csv(input_csv)
    eeg = data.iloc[:, :-1].values  # EEG channels (n_samples, 9)
    labels = data.iloc[:, -1].values  # Labels (n_samples,)

    # Reshape into trials (n_trials, n_channels, n_samples)
    n_trials = len(eeg) // WINDOW_SIZE
    eeg_reshaped = eeg[:n_trials * WINDOW_SIZE].reshape(n_trials, WINDOW_SIZE, -1)
    eeg_reshaped = np.transpose(eeg_reshaped, (0, 2, 1))  # (n_trials, 9, 1000)
    trial_labels = labels[::WINDOW_SIZE]

    # Prepare directories
    os.makedirs(output_dir, exist_ok=True)
    merged_features = pd.DataFrame()

    for class_a, class_b in CLASS_PAIRS:
        pair_name = f"{class_a}v{class_b}"
        csp_path = os.path.join(csp_models_dir, f"csp_{pair_name}.joblib") if csp_models_dir else None

        # Filter trials for current pair
        mask = np.isin(trial_labels, [class_a, class_b])
        eeg_pair = eeg_reshaped[mask]
        labels_pair = trial_labels[mask]

        # Fit or load CSP
        if split_name == 'train':
            csp = CSP(n_components=N_COMPONENTS, reg=None, log=True)
            csp_features = csp.fit_transform(eeg_pair, labels_pair)
            if csp_models_dir:
                os.makedirs(csp_models_dir, exist_ok=True)
                dump(csp, csp_path)  # Save trained CSP
        else:
            if not csp_models_dir or not os.path.exists(csp_path):
                raise FileNotFoundError(f"CSP model not found at {csp_path}. Train first!")
            csp = load(csp_path)  # Load pre-trained CSP
            csp_features = csp.transform(eeg_pair)

        # Save individual 1vs1 CSP file
        pair_df = pd.DataFrame(csp_features,
                               columns=[f"CSP1_{pair_name}", f"CSP2_{pair_name}"])
        pair_df.to_csv(os.path.join(output_dir, f"csp_{pair_name}.csv"), index=False)

        # Add to merged features
        merged_features = pd.concat([merged_features, pair_df], axis=1)

    # Save merged features
    merged_features.to_csv(os.path.join(output_dir, "l1b_merged_csp_features.csv"), index=False)
    print(f"Generated CSP features in {output_dir}")
    print(f"- Merged features: {merged_features.shape} (all pairs)")
    print(f"- Individual files: {len(CLASS_PAIRS)} 1vs1 CSP files")


if __name__ == "__main__":
    # Path configuration
    CSP_MODELS_DIR = "csp_models"  # Where to save trained CSP objects
    os.makedirs(CSP_MODELS_DIR, exist_ok=True)

    # Process training data (fit CSP)
    print("\n" + "=" * 50)
    print("Processing TRAINING data (fitting CSP)")
    generate_csp_features(
        input_csv="l1b_laf_eeg_data_ch9_all_labels_train.csv",
        output_dir="csp_output/train",
        split_name="train",
        csp_models_dir=CSP_MODELS_DIR
    )

    # Process validation data (transform with pre-trained CSP)
    print("\n" + "=" * 50)
    print("Processing VALIDATION data (using pre-trained CSP)")
    generate_csp_features(
        input_csv="l1b_laf_eeg_data_ch9_all_labels_val.csv",
        output_dir="csp_output/val",
        split_name="val",
        csp_models_dir=CSP_MODELS_DIR
    )

    # Process test data (transform with pre-trained CSP)
    print("\n" + "=" * 50)
    print("Processing TEST data (using pre-trained CSP)")
    generate_csp_features(
        input_csv="l1b_laf_eeg_data_ch9_all_labels_test.csv",
        output_dir="csp_output/test",
        split_name="test",
        csp_models_dir=CSP_MODELS_DIR
    )