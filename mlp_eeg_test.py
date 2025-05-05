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
    # ---->>>>> PATHS TO UPDATE <<<<<<----
    test_csv_path = 'data/test_mlp.csv'  # Path to test CSV
    model_file_path = 'models/best_model.joblib'  # Path to saved model

    # Load model and data
    model = load(model_file_path)
    X_test, y_test = load_data(test_csv_path)

    # Get predictions and probabilities
    preds = model.predict(X_test)
    preds_proba = model.predict_proba(X_test)  # For ROC AUC

    # Calculate metrics
    acc = accuracy_score(y_test, preds)

    # Handle multi-class ROC AUC
    try:
        roc = roc_auc_score(y_test,
                            preds_proba,
                            multi_class='ovr',
                            average='weighted')
    except Exception as e:
        print(f"Error calculating ROC AUC: {str(e)}")
        roc = None

    # Display results
    print("\n" + "=" * 50)
    print(f"Test Accuracy: {acc:.2%}")
    if roc is not None:
        print(f"Test ROC AUC: {roc:.2f}")
    print("\nClassification Report:")
    print(classification_report(y_test, preds))
    print("=" * 50 + "\n")


if __name__ == '__main__':
    main()