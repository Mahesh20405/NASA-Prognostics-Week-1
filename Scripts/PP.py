import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib

def preprocess_data():
    # Load the data
    column_names = ['unit_number', 'time_in_cycles'] + [f'op_setting_{i}' for i in range(1, 4)] + [f'sensor_measurement_{i}' for i in range(1, 22)]
    train_data = pd.read_csv(r'F:\PJ\Pred\Data\train_FD001.txt', sep=r'\s+', header=None, names=column_names)
    test_data = pd.read_csv(r'F:\PJ\Pred\Data\test_FD001.txt', sep=r'\s+', header=None, names=column_names)
    rul_data = pd.read_csv(r'F:\PJ\Pred\Data\RUL_FD001.txt', sep=r'\s+', header=None, names=['RUL'])

    # Add RUL column
    rul_train = pd.DataFrame(train_data.groupby('unit_number')['time_in_cycles'].max()).reset_index()
    rul_train.columns = ['unit_number', 'max_cycles']
    train_data = train_data.merge(rul_train, on=['unit_number'], how='left')
    train_data['RUL'] = train_data['max_cycles'] - train_data['time_in_cycles']
    train_data.drop(['max_cycles'], axis=1, inplace=True)

    # Check for NaNs and drop if any
    if train_data['RUL'].isnull().any():
        print("NaNs found in RUL column of train data. Dropping rows with NaNs.")
        train_data.dropna(subset=['RUL'], inplace=True)
    if test_data.isnull().any().any():
        print("NaNs found in test data. Dropping rows with NaNs.")
        test_data.dropna(inplace=True)

    # Normalize the data
    scaler = MinMaxScaler()
    numeric_features = train_data.columns[2:-1]  # Adjust based on the actual feature columns
    train_data[numeric_features] = scaler.fit_transform(train_data[numeric_features])
    test_data[numeric_features] = scaler.transform(test_data[numeric_features])

    # Prepare training and test datasets
    X_train = train_data.iloc[:, 2:-1].values
    y_train = train_data['RUL'].values

    # Group by unit_number and take the last record
    X_test = test_data.groupby('unit_number').last().reset_index().iloc[:, 2:].values
    y_test = rul_data['RUL'].values

    # Save preprocessed data to files
    np.save('X_train.npy', X_train)
    np.save('y_train.npy', y_train)
    np.save('X_test.npy', X_test)
    np.save('y_test.npy', y_test)

if __name__ == "__main__":
    preprocess_data()
