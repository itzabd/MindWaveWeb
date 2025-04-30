
import pandas as pd
import random


def data_split(data, train_dir, val_dir, test_dir):

    # Save the data in an array in units of 1000
    print('Scanned data length : ', len(data))

    data_list = []
    for i in range(0, len(data), 1000):
        data_list.append(data.iloc[i:i + 1000])

    random.shuffle(data_list)

    # Use 60% of the shuffled indices as training data, 20% as validation data and 20% as test data
    chunk_data = int(len(data) / 10000)  # 1000 * 10(0.6, 0.8, 1.0 -> 6, 8, 10 // 소수점 연산 미지원)
    train_data = pd.DataFrame()
    test_data = pd.DataFrame()
    val_data = pd.DataFrame()
    for i in range(0, chunk_data*6):
        train_data = pd.concat([train_data, data_list[i]])
    for i in range(chunk_data*6, chunk_data*8):
        val_data = pd.concat([val_data, data_list[i]])
    for i in range(chunk_data*8, chunk_data*10):
        test_data = pd.concat([test_data, data_list[i]])

    # Drop the first column (index)
    train_data = train_data.iloc[:, 1:]
    val_data = val_data.iloc[:, 1:]
    test_data = test_data.iloc[:, 1:]

    # Save into csv files
    train_data.to_csv(train_dir, index=False)
    val_data.to_csv(val_dir, index=False)
    test_data.to_csv(test_dir, index=False)

    train_df = pd.read_csv(train_dir)
    val_df = pd.read_csv(val_dir)
    test_df = pd.read_csv(test_dir)

    # Display their lengths (number of rows)
    print(f"Train data length: {len(train_df)} rows")
    print(f"Validation data length: {len(val_df)} rows")
    print(f"Test data length: {len(test_df)} rows")

def main():
    # Assuming the input file 'laf_eeg_data_9ch_360000.csv' is in the same directory as the script
    data_name = 'laf_eeg_data_9ch_360000'
    data_dir = data_name + '.csv'

    # Define output file names (basic)
    train_dir = data_name + '_train.csv'
    val_dir = data_name + '_val.csv'
    test_dir = data_name + '_test.csv'

    # Create empty CSV files
    for file_path in [train_dir, val_dir, test_dir]:
        with open(file_path, 'w') as f:
            pass  # Just creates an empty file

    df = pd.read_csv(data_dir)
    data_split(df, train_dir, val_dir, test_dir)


if __name__ == '__main__':
    main()