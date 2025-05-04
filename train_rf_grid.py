#%%
import mne
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
import matplotlib.pyplot as plt
from Preprocessing.feature_extraction import load_eeg_data, compute_band_power, extract_features
mne.set_log_level('error')
#%%
def pipeline(base_path):
    train_dir = base_path + 'train4ml.csv'
    test_dir = base_path + 'test4ml.csv'
    val_dir = base_path + 'val4ml.csv'
    data_train, label_train = load_eeg_data(train_dir)
    data_val, label_val = load_eeg_data(val_dir)   
    data_test, label_test = load_eeg_data(test_dir)
    
    train_X = data_train
    train_y = label_train
    val_X = data_val
    val_y = label_val
    test_X = data_test
    test_y = label_test
    
    # Scaling
    scaler = StandardScaler()
    train_X = scaler.fit_transform(train_X)
    val_X = scaler.transform(val_X)
    test_X = scaler.transform(test_X)
    
    # Scaling
    scaler = StandardScaler()
    train_X = scaler.fit_transform(train_X)
    val_X = scaler.transform(val_X)
    test_X = scaler.transform(test_X)
    
    # Train through GridSearchCV
    rf = RandomForestClassifier()
    param_grid = {
        'n_estimators': [100, 200, 300],  # 트리의 개수
        'max_depth': [10, 20, None],  # 트리의 최대 깊이
        'min_samples_split': [2, 5, 10],  # 분할을 위한 최소 샘플 수
        'min_samples_leaf': [1, 2, 4],  # 리프 노드의 최소 샘플 수
        'max_features': ['sqrt', 'log2', None]  # 고려할 최대 피처 수
    }

    grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='accuracy', verbose=2, n_jobs=-1)
    grid_search.fit(train_X, train_y)  # Fit the model on the training data

    # Print the best parameters and the best score from the validation process
    print("Best parameters found: ", grid_search.best_params_)
    print("Best cross-validation accuracy: {:.2f}%".format(grid_search.best_score_ * 100))
    
    # (Validation) Use the best model to make predictions on the validation set
    best_model = grid_search.best_estimator_
    val_predictions = best_model.predict(val_X)

    # Evaluate the model on the validation set
    print("\nValidation Set Performance:")
    print("Validation Accuracy: {:.2f}%".format(accuracy_score(val_y, val_predictions) * 100))
    print("Validation ROC-AUC Score: {:.2f}".format(roc_auc_score(val_y, val_predictions)))
    print("\nValidation Classification Report:")
    print(classification_report(val_y, val_predictions))
    
    # (Test) After validation, use the best model to predict on the test set
    test_predictions = best_model.predict(test_X)

    # Evaluate the model on the test set
    print("\nTest Set Performance:")
    print("Test Accuracy: {:.2f}%".format(accuracy_score(test_y, test_predictions) * 100))
    print("Test ROC-AUC Score: {:.2f}".format(roc_auc_score(test_y, test_predictions)))
    print("\nTest Classification Report:")
    print(classification_report(test_y, test_predictions))
#%%
# Load data
base_path_1 = 'your_path'
base_path_2 = 'your_path'
base_path_3 = 'your_path'
base_path_4 = 'your_path'
window_size = 1000
#%%
train_rf_1 = pipeline(base_path_1)
#%%
train_rf_2 = pipeline(base_path_2)
#%%
train_rf_3 = pipeline(base_path_3)
#%%
train_rf_4 = pipeline(base_path_4)