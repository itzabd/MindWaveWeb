#%%
import mne
import numpy as np
import pandas as pd

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


def extract_features(data, selected_columns, sfreq=250):
    """
    Extract features from EEG data. Furthermore, the data is downsampled to the target sampling frequency.
    :param data: EEG data (DataFrame)
    :param selected_columns: List of tuples containing channel index and frequency bands
    :param sfreq: Sampling frequency of the data
    :param target_sfreq: Target sampling frequency
    :return: Extracted features (DataFrame)
    """
    feature_dict = {}  # 결과를 저장할 딕셔너리

    for item in selected_columns:
        channel_idx = item[0]  # 채널 인덱스
        bands = item[1]  # 해당 채널에서 추출할 주파수 대역 리스트

        # 주파수 대역이 하나만 주어졌을 때도 리스트로 처리
        if isinstance(bands, tuple):
            bands = [bands]

        # 채널의 데이터 추출
        eeg_data = data.iloc[:, channel_idx].values  # 특정 채널의 데이터를 가져옴
        ch_name = data.columns[channel_idx]  # 채널 이름

        # mne RawArray 객체 생성
        info = mne.create_info(ch_names=[ch_name], sfreq=sfreq, ch_types='eeg')
        raw = mne.io.RawArray(eeg_data[np.newaxis, :], info)  # 2D array 필요

        # 주파수 대역별로 PSD 계산
        for band in bands:
            band_power = compute_band_power(raw, band)
            # 열 이름 생성 (예: Channel_1_10-12Hz)
            column_name = f'{ch_name}_{band[0]}-{band[1]}Hz'
            feature_dict[column_name] = band_power

    # 최종 데이터프레임 생성
    features = pd.DataFrame([feature_dict])

    return features

#%%
base_path = 'your_path/'

train_csv_path = "laf_eeg_data_9ch_360000_train.csv"
val_csv_path = "laf_eeg_data_9ch_360000_val.csv"
test_csv_path = "laf_eeg_data_9ch_360000_test.csv"

train_json_path = base_path + 'json/l1b_train.json'
train_jsonl_path = base_path + 'jsonl/l1b_train.jsonl'

val_json_path = base_path + 'json/l1b_val.json'
val_jsonl_path = base_path + 'jsonl/l1b_val.jsonl'

# csp_train_path = base_path + 'csp1/class_1_vs_5_train_features.csv'
# csp_val_path = base_path + 'csp1/class_1_vs_5_val_features.csv'
# csp_test_path = base_path + 'csp1/class_1_vs_5_test_features.csv'
csp_train_path = 'class_1_vs_5_train_features.csv'
csp_val_path = 'class_1_vs_5_val_features.csv'
csp_test_path = 'class_1_vs_5_test_features.csv'
#%%
from feature_extraction import *
dftrain, labeltrain = load_eeg_data(train_csv_path)
dfval, labelval = load_eeg_data(val_csv_path)
dftest, labeltest = load_eeg_data(test_csv_path)
print(dftrain.shape)
print(labeltrain.shape)
#%%
csptrain, csptrainlabel = load_eeg_data(csp_train_path)
csptrain = csptrain.to_numpy()
cspval, cspvallabel = load_eeg_data(csp_val_path)
cspval = cspval.to_numpy()
csptest, csptestlabel = load_eeg_data(csp_test_path)
csptest = csptest.to_numpy()
#%%
selected_columns = [
        [0, [(10, 12), (12, 14)]],  # FCz
        [2, [(20, 22), (22, 24)]],  # C3
        [3, [(8, 10)]],  # Cz
        [4, [(20, 22), (22, 24)]],  # C4
        [5, [(28, 30)]],  # CP3
]
#%%
def csv_to_df(df, csp, window_size, selected_columns, labels):
    """
    Convert a DataFrame of EEG data into a JSON format suitable for GPT-3 davinci.
    :param df: Data converted to pandas DataFrame from the original csv file
    :param window_size: Window size to divide EEG data
    :param selected_columns: EEG channel to use (provide a list with frequency bands)
    :param labels: Label for each window (provide a list, left, right, top, bottom)
    :return: List of data in JSON format
    """
    df_array = pd.DataFrame()

    # EEG 채널 이름을 selected_columns에 매핑합니다.
    channel_names = ['FCz', 'C3', 'Cz', 'C4', 'CP3']  # 각각 0, 1, 2, 3에 대응

    for start in range(0, len(df) - window_size + 1, window_size):
        window_data = df.iloc[start:start + window_size, :]  # 전체 데이터를 가져옴
        label = str(int(labels[start]))  # Assuming labels are provided for each window

        # Extract features using the updated extract_features function
        features = extract_features(window_data, selected_columns)  # feature extraction
        cspdata = pd.DataFrame(csp[int(start / 1000)]).T  # cspdata 가져옴
        # features와 cspdata 가로 방향으로 합침
        features = pd.concat([features, cspdata], axis=1)
        # 라벨까지 합치기, 라벨의 column명은 'Label'
        features['Label'] = label
        
        # 최종 데이터프레임 생성
        df_array = pd.concat([df_array, features], axis=0)
    return df_array
#%%
train4ml = csv_to_df(dftrain, csptrain, 1000, selected_columns, labeltrain)
train4ml
#%%
val4ml = csv_to_df(dfval, cspval, 1000, selected_columns, labelval)
val4ml
#%%
test4ml = csv_to_df(dftest, csptest, 1000, selected_columns, labeltest)
test4ml
#%%
train4ml.to_csv(base_path+'train4ml.csv')
val4ml.to_csv(base_path+'val4ml.csv')
test4ml.to_csv(base_path+'test4ml.csv')
#%%
