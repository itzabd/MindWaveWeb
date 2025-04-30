#%%
import argparse
import mne
import numpy as np
import pandas as pd
import os
import sys
#%%
# EEG data 파일 경로
file_path = "gdf_dataset/l1b.gdf"
# GDF EEG data
raw = mne.io.read_raw_gdf(file_path, preload=True)
#%%
raw
#%%
raw.ch_names
#%%
raw.plot()
#%%
# 데이터를 DataFrame으로 변환
raw_df = raw.to_data_frame()

raw_df.to_csv('eeg_data.csv', index=False)
#%%
raw_df.head(15)
#%% md
# 'FC3', 'FCz', 'FC4', 'C3', 'Cz', 'C4', 'CP3', 'CPz', 'CP4' 채널 매핑
#%%
# 채널 매핑 정의
channel_mapping = {
    10: 'FC3',
    13: 'FCz',
    16: 'FC4',
    28: 'C3',
    31: 'Cz',
    34: 'C4',
    46: 'CP3',
    49: 'CPz',
    52: 'CP4'
}
print(channel_mapping)
#%%
# 채널 이름을 매핑된 채널 번호로 변환
def map_channel_names(raw, channel_mapping):
    # 기존 채널 번호를 가져옴
    original_channel_names = raw.info['ch_names']
    
    # 새로운 채널 이름 목록을 초기화
    new_channel_names = []
    
    for ch in range(len(original_channel_names)):
        if ch + 1 in channel_mapping:
            new_channel_names.append(channel_mapping[ch + 1])
        else:
            new_channel_names.append(original_channel_names[ch])
    
    # 채널 이름을 새로 설정
    raw.rename_channels(dict(zip(original_channel_names, new_channel_names)))

# 채널 이름 매핑 적용
map_channel_names(raw, channel_mapping)
#%%
raw.ch_names

#%% md
# Based on the code context, the error is occurring because the `raw` object has been converted to a pandas DataFrame earlier in the notebook. You need to use the MNE Raw object to plot the PSD.
# 
# Here's the corrected code cell:
# 
# 
#%%
raw.plot_psd()
#%%
# 데이터를 DataFrame으로 변환
raw = raw.to_data_frame()
raw
#%% md
# 0-1 Scaling (MinMaxScaler)
#%%
from sklearn.preprocessing import MinMaxScaler


# time 컬럼을 제외한 나머지 데이터만 스케일링
features = raw.drop(columns=['time'])

# Min-Max 스케일러 초기화
scaler = MinMaxScaler()

# 스케일링 적용
scaled_features = scaler.fit_transform(features)

# 스케일링된 데이터를 DataFrame으로 변환하고 time 컬럼 추가
scaled_df = pd.DataFrame(scaled_features, columns=features.columns)
scaled_df['time'] = raw['time']

# 결과 확인
scaled_df
#%% md
# Large Laplacian Filter 적용
#%%
import mne
import numpy as np

# 10-20 시스템의 표준 전극 위치 파일 로드
montage = mne.channels.make_standard_montage('standard_1020')

# 60채널에 대한 모든 전극 위치 정보
all_positions = montage.get_positions()['ch_pos']

# 적용할 채널들
target_channels = ['FC3', 'FCz', 'FC4', 'C3', 'Cz', 'C4', 'CP3', 'CPz', 'CP4']

# 각 채널의 위치 추출
channel_positions = {ch: all_positions[ch] for ch in target_channels}
channel_positions
#%%


def calculate_distance(pos1, pos2):
    return np.linalg.norm(np.array(pos1) - np.array(pos2))

def apply_large_laplacian_filter(scaled_df, target_channels, neighbor_dict, electrode_positions):
    filtered_df = scaled_df.copy()  # 함수 내부에서 결과를 저장할 데이터프레임

    for ch in target_channels:
        # 인접 채널 설정
        neighbors = neighbor_dict.get(ch, [])
        
        # 존재하는 인접 채널만 선택
        neighbors = [n for n in neighbors if n in scaled_df.columns and n in electrode_positions]
        
        if not neighbors:
            continue
        
        # 현재 채널과 인접 채널 데이터 추출
        ch_data = scaled_df[ch].values
        neighbor_data = scaled_df[neighbors].values
        
        # 거리 계산 및 가중치 설정
        distances = np.array([calculate_distance(electrode_positions[ch], electrode_positions[n]) for n in neighbors])
        weights = 1 / distances
        weights /= weights.sum()  # 정규화
        
        # 가중치를 사용하여 인접 채널 데이터의 가중합 계산
        weighted_avg_neighbor_data = np.sum(weights[:, np.newaxis] * neighbor_data.T, axis=0)
        
        # 라플라시안 필터 적용
        filtered_df[ch] = ch_data - weighted_avg_neighbor_data

    return filtered_df

# 10-20 시스템의 표준 전극 위치 파일 로드
montage = mne.channels.make_standard_montage('standard_1020')

# 적용할 채널 목록
target_channels = ['FC3', 'FCz', 'FC4', 'C3', 'Cz', 'C4', 'CP3', 'CPz', 'CP4']

# 60채널에 대한 모든 전극 위치 정보
all_positions = montage.get_positions()['ch_pos']

# 각 채널의 위치 추출 (존재하는 채널만)
electrode_positions = {ch: all_positions[ch] for ch in target_channels if ch in all_positions}

# 각 채널의 인접 채널을 정의
neighbor_dict = {
    'FC3': ['FCz', 'C3'],
    'FCz': ['FC3', 'FC4', 'Cz'],
    'FC4': ['FCz', 'C4'],
    'C3': ['FC3', 'Cz', 'CP3'],
    'Cz': ['C3', 'C4', 'FCz', 'CPz'],
    'C4': ['FC4', 'Cz', 'CP4'],
    'CP3': ['C3', 'CPz'],
    'CPz': ['CP3', 'CP4', 'Cz'],
    'CP4': ['C4', 'CPz']
}

# 라플라시안 필터 적용
# scaled_df는 함수 호출 전에 준비된 데이터프레임이어야 합니다.
filtered_df = apply_large_laplacian_filter(scaled_df, target_channels, neighbor_dict, electrode_positions)

filtered_df
#%%
# CSV 파일로 저장
csv_file_path = "laf_eeg_data.csv"
filtered_df.to_csv(csv_file_path, index=False)

print("CSV File path:", csv_file_path)
#%%
filtered_df
#%% md
# HDR.TRIG / HDR.Classlabel 로드
#%%
# CSV 파일 읽기
eeg_d = pd.read_csv('laf_eeg_data.csv')
trig_df = pd.read_csv('csv_trig/k6b_HDR_TRIG.csv', header=None)
label_df = pd.read_csv('csv_Classlabels/k6b_HDR_Classlabel.csv', header=None)


#%%
eeg_d
#%%
# 각 trial의 시작
trig_df
#%%
# trigger dataframe에 index 컬럼 이름 지정
trig_df.columns = ['index']
#%%
trig_df
#%%
# label의 값
label_df
#%%
# label dataframe에 label 컬럼 이름 지정
label_df.columns = ['label']
#%%
label_df
#%%
# eeg_d에 label column 추가
eeg_d['label'] = 0
eeg_d
#%% md
# trig_df, label_df를 통해 eeg_d에 label 값 정하기
#%%
# trigger 인덱스와 레이블 가져오기
trigger_indices = trig_df['index'].values
labels = label_df['label'].values
#%%
# 트리거 구간에 따라 레이블 지정
for i in range(len(trigger_indices) - 1):
    start_idx = trigger_indices[i]
    end_idx = trigger_indices[i + 1]
    label = labels[i]
    eeg_d.loc[start_idx:end_idx-1, 'label'] = label
#%%
# 마지막 트리거 이후 구간 레이블 지정
if len(trigger_indices) > 0:
    start_idx = trigger_indices[-1]
    label = labels[-1]
    eeg_d.loc[start_idx:, 'label'] = label
#%%
eeg_d[2500:3100]
#%%
eeg_d[4945:5045]
#%%
# 각 레이블의 개수 출력
print(eeg_d['label'].value_counts())
#%%
eeg_d
#%% md
# epoch로 자르기 (rest : 1~2s / label : 3~7s 값들만)
#%%
# label이 0인 eeg_d 행 출력
eeg_d[eeg_d['label'] == 0]
#%%
# 샘플링 주파수
fs = 250
start_offset = 3 * fs
end_offset = 7 * fs

# Epochs를 저장할 리스트 초기화
epochs = []

trigger_indices = trig_df['index'].values.astype(int)  # 정수로 변환
labels = label_df['label'].values

# Epoch 추출
for idx, label in zip(trigger_indices, labels):
    start_idx = int(idx + start_offset)  # 정수로 변환
    end_idx = int(idx + end_offset)  # 정수로 변환
    
    if end_idx <= len(eeg_d):
        epoch = eeg_d.iloc[start_idx:end_idx].copy()
        epoch['label'] = label
        epochs.append(epoch)

# epochs 리스트를 단일 DataFrame으로 결합
epochs_df = pd.concat(epochs, ignore_index=True)
#%%
epochs_df
#%%
epochs[0]
#%%
epochs[37]
#%%
len(epochs)
#%% md
# rest 상태 epoch 추출
#%%
# 샘플링 주파수
fs = 250
start_offset = 1 * fs
end_offset = 2 * fs

# Epochs를 저장할 리스트 초기화
rest_epochs = []

trigger_indices = trig_df['index'].values.astype(int)  # 정수로 변환
labels = label_df['label'].values

# Epoch 추출
for idx, label in zip(trigger_indices, labels):
    start_idx = int(idx + start_offset)  # 정수로 변환
    end_idx = int(idx + end_offset)  # 정수로 변환
    
    if end_idx <= len(eeg_d):
        epoch = eeg_d.iloc[start_idx:end_idx].copy()
        epoch['label'] = label
        rest_epochs.append(epoch)

# rest_epochs 리스트를 단일 DataFrame으로 결합
rest_epochs_df = pd.concat(rest_epochs, ignore_index=True)
#%%
rest_epochs_df
#%% md
# NaN 값을 가진 epoch 제거
#%%
# NaN 값을 가진 행을 제거
label_epochs_df = epochs_df.dropna(subset=['label'])
rest_epochs_df = rest_epochs_df.dropna(subset=['label'])
#%%
# NaN 값을 가진 항목을 제거하여 rest_epochs 리스트를 업데이트
rest_epochs = [epoch for epoch in rest_epochs if not epoch['label'].isnull().any()]

# NaN 값을 가진 항목을 제거하여 epochs 리스트를 업데이트
epochs = [epoch for epoch in epochs if not epoch['label'].isnull().any()]
#%%
label_epochs_df
#%%
rest_epochs_df
#%%
# rest 상태일 때 label 컬럼의 모든 값을 '5'로 변경
rest_epochs_df['label'] = '5'
rest_epochs_df
#%% md
# label_epochs_df에서 time 열 제거하기
#%%
# 'time' 열 제거
label_epochs_df = label_epochs_df.drop(columns=['time'])
label_epochs_df
#%%
# CSV 파일로 저장
csv_file_path = "laf_eeg_data_label_360000.csv"
label_epochs_df.to_csv(csv_file_path, index=False)

print("CSV 파일이 저장되었습니다:", csv_file_path)
#%%
# CSV 파일 불러오기
csv_file_path = "laf_eeg_data_label_360000.csv"
df = pd.read_csv(csv_file_path)

# 특정 채널들만 남기기+++++++++++))))))))))))))KKKKKKKKKKKKKKKKK
df = df[['FC3', 'FCz', 'FC4', 'C3', 'Cz', 'C4', 'CP3', 'CPz', 'CP4', 'label']]

# 결과 출력
df
#%%
# CSV 파일로 저장
csv_file_path = "laf_eeg_data_9ch_360000.csv"
df.to_csv(csv_file_path, index=False)

print("CSV 파일이 저장되었습니다:", csv_file_path)
#%%
# Load the CSV file
csv_file_path = "laf_eeg_data_label_360000.csv"
df = pd.read_csv(csv_file_path)

# Filter the dataframe to include only rows where label is 1
df_label_1 = df[df['label'] == 1]
df_label_1
#%%
# Reset the index of the filtered dataframe
df_label_1 = df_label_1.reset_index(drop=True)
df_label_1
#%%
# Load the CSV file
csv_file_path = "laf_eeg_data_label_360000.csv"
df = pd.read_csv(csv_file_path)

# Filter the dataframe to include only rows where label is 2
df_label_2 = df[df['label'] == 2]
df_label_2
#%%
# Load the CSV file
csv_file_path = "laf_eeg_data_label_360000.csv"
df = pd.read_csv(csv_file_path)

# Filter the dataframe to include only rows where label is 3
df_label_3 = df[df['label'] == 3]
df_label_3
#%%
# Load the CSV file
csv_file_path = "laf_eeg_data_label_360000.csv"
df = pd.read_csv(csv_file_path)

# Filter the dataframe to include only rows where label is 4
df_label_4 = df[df['label'] == 4]
df_label_4
#%%
# Save the filtered and reset dataframe to a new CSV file
csv_file_path = "laf_eeg_data_label_1.csv"
df_label_1.to_csv(csv_file_path, index=False)
#%%
# Save the filtered and reset dataframe to a new CSV file
csv_file_path = "laf_eeg_data_label_2.csv"
df_label_2.to_csv(csv_file_path, index=False)
#%%
# Save the filtered and reset dataframe to a new CSV file
csv_file_path = "laf_eeg_data_label_3.csv"
df_label_3.to_csv(csv_file_path, index=False)
#%%
# Save the filtered dataframe to a new CSV file
csv_file_path = "laf_eeg_data_label_4.csv"
df_label_4.to_csv(csv_file_path, index=False)
#%%
rest_epochs_df = rest_epochs_df.drop(columns=['time'])
#%%
# Save the filtered dataframe to a new CSV file
csv_file_path = "laf_eeg_data_label_5.csv"
rest_epochs_df.to_csv(csv_file_path, index=False)
#%%
# CSV 파일 불러오기
csv_file_path = "laf_eeg_data_label_1.csv"
df = pd.read_csv(csv_file_path)

# 특정 채널들만 남기기 _==================================++++++++++++++=kkkkkkkkk
df = df[['FC3', 'FCz', 'FC4', 'C3', 'Cz', 'C4', 'CP3', 'CPz', 'CP4', 'label']]

# 결과 출력
df
#%%
# Save the filtered dataframe to a new CSV file
csv_file_path = "laf_eeg_data_ch9_label1.csv"
df.to_csv(csv_file_path, index=False)
#%%
# CSV 파일 불러오기
csv_file_path = "laf_eeg_data_label_2.csv"
df = pd.read_csv(csv_file_path)

# 특정 채널들만 남기기
df = df[['FC3', 'FCz', 'FC4', 'C3', 'Cz', 'C4', 'CP3', 'CPz', 'CP4', 'label']]

# 결과 출력
df
#%%
# Save the filtered dataframe to a new CSV file
csv_file_path = "laf_eeg_data_ch9_label2.csv"
df.to_csv(csv_file_path, index=False)
#%%
# CSV 파일 불러오기
csv_file_path = "laf_eeg_data_label_3.csv"
df = pd.read_csv(csv_file_path)

# 특정 채널들만 남기기
df = df[['FC3', 'FCz', 'FC4', 'C3', 'Cz', 'C4', 'CP3', 'CPz', 'CP4', 'label']]

# 결과 출력
df
#%%
# Save the filtered dataframe to a new CSV file
csv_file_path = "laf_eeg_data_ch9_label3.csv"
df.to_csv(csv_file_path, index=False)
#%%
# CSV 파일 불러오기
csv_file_path = "laf_eeg_data_label_4.csv"
df = pd.read_csv(csv_file_path)

# 특정 채널들만 남기기
df = df[['FC3', 'FCz', 'FC4', 'C3', 'Cz', 'C4', 'CP3', 'CPz', 'CP4', 'label']]

# 결과 출력
df
#%%
# Save the filtered dataframe to a new CSV file
csv_file_path = "laf_eeg_data_ch9_label4.csv"
df.to_csv(csv_file_path, index=False)
#%%
# CSV 파일 불러오기
csv_file_path = "laf_eeg_data_label_5.csv"
df = pd.read_csv(csv_file_path)

# 특정 채널들만 남기기
df = df[['FC3', 'FCz', 'FC4', 'C3', 'Cz', 'C4', 'CP3', 'CPz', 'CP4', 'label']]

# 결과 출력
df
#%%
# Save the filtered dataframe to a new CSV file
csv_file_path = "laf_eeg_data_ch9_label5.csv"
df.to_csv(csv_file_path, index=False)