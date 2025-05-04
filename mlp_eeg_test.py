# mlp_eeg_test.py
import pandas as pd
from joblib import load
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
import warnings

warnings.filterwarnings('ignore')


def load_data(path):
    df = pd.read_csv(path)
    X = df.drop('label', axis=1).values
    y = df['label'].values
    return X, y


def main():
    # ---->>>>> CHANGE THESE TWO PATHS AS NEEDED <<<<<<----
    test_csv_path = 'test_mlp.csv'  # Path to your test CSV file
    model_file_path = 'best_model.joblib'  # Path to your saved trained model

    # Load model and test data
    model = load(model_file_path)
    X_test, y_test = load_data(test_csv_path)

    # Predict and evaluate
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    roc = roc_auc_score(y_test, preds)

    print(f"Test Accuracy: {acc:.2%}")
    print(f"Test ROC AUC: {roc:.2f}")
    print("Classification Report:")
    print(classification_report(y_test, preds))


if __name__ == '__main__':
    main()
