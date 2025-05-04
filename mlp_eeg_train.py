# mlp_eeg_train.py
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
from joblib import dump
import warnings

warnings.filterwarnings('ignore')

def load_data(path):
    df = pd.read_csv(path)
    X = df.drop('label', axis=1).values
    y = df['label'].values
    return X, y

def main():
    # 👇 Provide your input CSV and output model path here
    input_csv = 'test_mlp.csv'               # <-- Set your CSV file path here
    output_model = 'best_model.joblib'       # <-- Set your desired output model path here

    # Load and split data
    X, y = load_data(input_csv)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    # Build pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('mlp', MLPClassifier(early_stopping=True, random_state=42))
    ])

    param_grid = {
        'mlp__hidden_layer_sizes': [(100,), (50,50), (30,30,30)],
        'mlp__activation': ['relu', 'tanh'],
        'mlp__solver': ['adam', 'sgd'],
        'mlp__alpha': [0.0001, 0.001, 0.01],
        'mlp__learning_rate': ['constant', 'adaptive'],
        'mlp__max_iter': [200, 300, 500],
        'mlp__batch_size': [32, 64]
    }

    # Hyperparameter search
    grid = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', verbose=2, n_jobs=-1)
    grid.fit(X_train, y_train)

    best = grid.best_estimator_
    print("Best parameters:", grid.best_params_)
    print(f"Validation accuracy (CV): {grid.best_score_:.2%}")

    # Evaluate on held-out validation and test
    for name, X_part, y_part in [('Validation', X_val, y_val), ('Test', X_test, y_test)]:
        preds = best.predict(X_part)
        preds_proba = best.predict_proba(X_part)  # <-- ADD THIS LINE
        acc = accuracy_score(y_part, preds)
        roc = roc_auc_score(
            y_part,
            preds_proba,  # Use probabilities, not labels
            multi_class='ovr',  # Set strategy
            average='weighted'  # Set averaging method
        )
        print(f"\n{name} set:")
        print(f"  Accuracy: {acc:.2%}")
        print(f"  ROC AUC: {roc:.2f}")
        print(classification_report(y_part, preds))

    # Save pipeline
    dump(best, output_model)
    print(f"Model saved to {output_model}")

if __name__ == '__main__':
    main()
