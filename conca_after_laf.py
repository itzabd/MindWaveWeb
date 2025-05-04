#%%
# laf_eeg_data_ch9_label1.csv ~ laf_eeg_data_ch9_label4.csv 까지의 EEG를 받아, row 방향으로 합침
# 결과는 laf_eeg_data_180000.csv로 저장

import pandas as pd
#%%
# 데이터 불러오기
base_dir = 'your_path'
data1 = pd.read_csv(base_dir + 'laf_eeg_data_ch9_label1.csv')
data2 = pd.read_csv(base_dir + 'laf_eeg_data_ch9_label2.csv')
data3 = pd.read_csv(base_dir + 'laf_eeg_data_ch9_label3.csv')
data4 = pd.read_csv(base_dir + 'laf_eeg_data_ch9_label4.csv')
#%%
# 데이터 합치기
data = pd.concat([data1, data2, data3, data4], axis=0)
#%%
data
#%%
# 데이터 저장
data.to_csv('/Users/imdohyeon/Library/CloudStorage/GoogleDrive-dhlim1598@gmail.com/공유 드라이브/4N_PKNU/BXAI/EEG-LLM/Dataset//laf_eeg_data_180000.csv', index=True)
#%%
