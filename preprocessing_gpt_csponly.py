#%%
import mne
import numpy as np
import pandas as pd
import json
#%%

"""
Additional filtering is not required as the data is already preprocessed.
"""


def load_eeg_data(file_path):
    """
    Load EEG data from a csv file and separate data and label.
    :param file_path: File path of the EEG data
    :return: EEG data (DataFrame), label
    """
    data_src = pd.read_csv(file_path)
    data = data_src.iloc[:, :-1]  # Exclude the last column as it is a label
    label = data_src.iloc[:, -1]  # Use the last column as a label
    return data, label


def compute_band_power(raw, band):
    """
    Compute the power in a specific frequency band.
    :param raw: MNE Raw object
    :param band: Frequency band of interest (tuple)
    :return: Power in the frequency band
    """
    fmin, fmax = band  # Setting frequency band
    data = raw.get_data()
    sfreq = raw.info['sfreq']
    psds, freqs = mne.time_frequency.psd_array_welch(data, sfreq=sfreq, fmin=fmin, fmax=fmax, n_fft=128)  # Compute PSD
    # Compute power in the frequency band
    band_power = np.sum(psds, axis=-1)
    return band_power
#%%
def csv_to_json_csp(df, csp, window_size, selected_columns, labels):
    """
    Convert a DataFrame of EEG data into a JSON format suitable for GPT-3 davinci.
    :param df: Data converted to pandas DataFrame from the original csv file
    :param window_size: Window size to divide EEG data
    :param selected_columns: EEG channel to use (provide a list with frequency bands)
    :param labels: Label for each window (provide a list, left, right, top, bottom)
    :return: List of data in JSON format
    """
    json_array = []

    # EEG 채널 이름을 selected_columns에 매핑합니다.
    channel_names = ['FCz', 'C3', 'Cz', 'C4', 'CP3']  # 각각 0, 1, 2, 3에 대응

    for start in range(0, len(df) - window_size + 1, window_size):
        # window_data = df.iloc[start:start + window_size, :]  # 전체 데이터를 가져옴
        label = str(int(labels[start]))  # Assuming labels are provided for each window

        # Extract features using the updated extract_features function
        # features = extract_features(window_data, selected_columns)  # feature extraction
        features = pd.DataFrame(csp[int(start / 1000)]).T  # cspdata 가져옴
        # features와 cspdata 가로 방향으로 합침
        # features = pd.concat([features, cspdata], axis=1)
        features_dict = features.to_dict('index')[0]  # DataFrame to dictionary

        # Generate features_dict_with_keys
        features_dict_with_keys = {}
        """
        for i, (channel_idx, bands) in enumerate(selected_columns):
            key = f"at channel {channel_names[i]}"
            features_list = []
            for band in bands if isinstance(bands, list) else [bands]:
                band_key = f"{channel_names[i]}_{band[0]}-{band[1]}Hz"
                power_value = features_dict[band_key]

                # Flatten the power value if it's an array
                if isinstance(power_value, np.ndarray):
                    power_value = power_value.item()  # Convert array to scalar if it's 1D
                features_list.append(f"Power in {band[0]}-{band[1]} Hz: {power_value}")
            features_dict_with_keys[key] = features_list
            
        """
        # Set the CSP 값을 라벨에 맞게 프롬프트에 추가
        csp_key = f"CSP values: 0: {features.values[0][0]}, 1: {features.values[0][1]}"

        # Set the GPT's role
        system_message = "Look at the feature values of a given EEG electrode and determine which label the data belongs to. The result should always provide only integer label values."

        # Prompt explaining the feature information
        prompt = f"Quantitative EEG: In a {window_size / 250} second period,"
        features_str = ""
        """
        for key, value in features_dict_with_keys.items():
            features_str += f"{key}:\n"
            features_str += "\n".join([f"  {v}" for v in value])
            features_str += "\n"
        """

        # CSP 값을 프롬프트에 포함
        combined_prompt = f"{prompt}\n{features_str}\n{csp_key}\n"

        # Convert the data to JSON format
        json_entry = {
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": combined_prompt},
                {"role": "assistant", "content": label}
            ]
        }

        json_array.append(json_entry)

    return json_array
#%%
def json_to_jsonl(json_dir, jsonl_dir):
    """
    Convert JSON file to JSONL file
    :param json_dir: JSON file path to load
    :param jsonl_dir: JSONL file path to save
    """
    json_data = load_json(json_dir)
    save_to_jsonl(json_data, jsonl_dir)

    print(f"Converted {json_dir} to {jsonl_dir}")


# Load JSON file
def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)


# Function to save as JSONL file (convert completion value to string)
def save_to_jsonl(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as jsonl_file:
        for entry in data:
            jsonl_line = json.dumps(entry, ensure_ascii=False)
            jsonl_file.write(jsonl_line + '\n')
#%%
def pipeline(csv_path, csp_path, json_path, jsonl_path, window_size, selected_columns):
    """
    Load the EEG data csv file, convert the preprocessed data to json format, and convert the json to jsonl format and save it.
    :param csv_path:  EEG data csv file path
    :param json_path:  json file path to save the preprocessed data
    :param jsonl_path:  jsonl file path to save the preprocessed data
    :param window_size:  window size of EEG data
    :param selected_columns:  EEG channel to use
    """
    # EEG(csv) load
    data, label = load_eeg_data(csv_path)

    # Preprocess the loaded data and convert it to json format
    json_data = csv_to_json_csp(data, csp_path, window_size, selected_columns, label)

    # Save the converted data to the specified path
    with open(json_path, 'w') as json_file:
        json.dump(json_data, json_file, indent=4)
    print(f"Data has been successfully saved to {json_path}")

    # Convert the preprocessed json to jsonl format and save it
    json_to_jsonl(json_path, jsonl_path)
#%%
base_path = '/Users/imdohyeon/Library/CloudStorage/GoogleDrive-dhlim1598@gmail.com/공유 드라이브/4N_PKNU/Project/EEG-LLM/Dataset/subject 1 data (k3b)/down sampling X ver/label45/'

train_csv_path = base_path + 'train.csv'
val_csv_path = base_path + 'val.csv'

train_json_path = base_path + 'csponly/json/train_csponly.json'
train_jsonl_path = base_path + 'csponly/jsonl/train_csponly.jsonl'

val_json_path = base_path + 'csponly/json/val_csponly.json'
val_jsonl_path = base_path + 'csponly/jsonl/val_csponly.jsonl'

train_csp_path = base_path + 'csp4/class_4_vs_5_train_features.csv'
val_csp_path = base_path + 'csp4/class_4_vs_5_val_features.csv'

train_csp, train_csp_label = load_eeg_data(train_csp_path)
val_csp, val_csp_label = load_eeg_data(val_csp_path)
train_csp = train_csp.to_numpy()
val_csp = val_csp.to_numpy()

window_size = 1000
# FCz=0, C3=2, Cz=3, C4=4
# selected_columns = [0, 2, 3, 4]  # EEG channels to use, selected by fisher ratio
selected_columns = [
    [0, [(10, 12), (12, 14)]],  # FCz
    [2, [(20, 22), (22, 24)]],  # C3
    [3, [(8, 10)]],  # Cz
    [4, [(20, 22), (22, 24)]],  # C4
    [5, [(28, 30)]],  # CP3
]

pipeline(train_csv_path, train_csp, train_json_path, train_jsonl_path, window_size, selected_columns)
pipeline(val_csv_path, val_csp, val_json_path, val_jsonl_path, window_size, selected_columns)
#%%
