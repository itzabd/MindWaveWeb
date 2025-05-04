# mlp_eeg_training_single_file.py
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
import warnings

warnings.filterwarnings('ignore')


def load_eeg_data(file_path):
    """Load EEG data from CSV file"""
    df = pd.read_csv(file_path)
    data = df.drop('label', axis=1).values
    labels = df['label'].values
    return data, labels


def run_pipeline(file_path):
    """
    Complete MLP training pipeline for EEG data from a single file
    Returns: Best model and evaluation metrics
    """
    # Load dataset and split into train/val/test
    data, labels = load_eeg_data(file_path)

    # First split: 70% train, 30% temp (val+test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        data, labels, test_size=0.3, random_state=42)

    # Second split: 50% val, 50% test of temp (15% each of total)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42)

    # Feature Scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # MLP Model Configuration
    mlp = MLPClassifier(early_stopping=True)
    param_grid = {
        'hidden_layer_sizes': [(100,), (50, 50), (30, 30, 30)],
        'activation': ['relu', 'tanh'],
        'solver': ['adam', 'sgd'],
        'alpha': [0.0001, 0.001, 0.01],
        'learning_rate': ['constant', 'adaptive'],
        'max_iter': [200, 300, 500],
        'batch_size': [32, 64]
    }

    # Hyperparameter Tuning
    grid_search = GridSearchCV(
        mlp,
        param_grid,
        cv=5,
        scoring='accuracy',
        verbose=2,
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train)

    # Best Model Evaluation
    best_model = grid_search.best_estimator_

    print("\n" + "=" * 50)
    print("Best parameters found:")
    print(grid_search.best_params_)
    print(f"Best validation accuracy: {grid_search.best_score_:.2%}")

    # Validation Set Performance
    val_pred = best_model.predict(X_val)
    print("\nValidation Results:")
    print(f"Accuracy: {accuracy_score(y_val, val_pred):.2%}")
    print(f"ROC AUC: {roc_auc_score(y_val, val_pred):.2f}")
    print("Classification Report:")
    print(classification_report(y_val, val_pred))

    # Test Set Performance
    test_pred = best_model.predict(X_test)
    print("\nTest Results:")
    print(f"Accuracy: {accuracy_score(y_test, test_pred):.2%}")
    print(f"ROC AUC: {roc_auc_score(y_test, test_pred):.2f}")
    print("Classification Report:")
    print(classification_report(y_test, test_pred))

    return best_model


if __name__ == "__main__":
    # Path to your EEG data file
    eeg_file = "concat_files.csv"

    print(f"\n{'#' * 20} Processing EEG Data File {'#' * 20}")
    model = run_pipeline(eeg_file)
    print(f"\n{'#' * 20} Completed Processing {'#' * 20}\n")