import pandas as pd
import numpy as np
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

def train_and_evaluate(input_path='data/model_ready.csv', output_dir='outputs', test_size=0.2):
    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found")
        return None
    
    df = pd.read_csv(input_path)
    print(f"Loaded {len(df)} records")
    
    feature_cols = [c for c in df.columns if c != 'risk_level']
    X = df[feature_cols]
    y = df['risk_level']
    
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=test_size, random_state=RANDOM_STATE, stratify=y_encoded
    )
    
    os.makedirs(output_dir, exist_ok=True)
    
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight='balanced'
    )
    
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)
    
    test_acc_rf = accuracy_score(y_test, y_pred_rf)
    test_f1_rf = f1_score(y_test, y_pred_rf, average='macro')
    
    print(f"Random Forest - Accuracy: {test_acc_rf:.4f}, F1: {test_f1_rf:.4f}")
    
    with open(os.path.join(output_dir, 'random_forest_model.pkl'), 'wb') as f:
        pickle.dump(rf_model, f)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    lr_model = LogisticRegression(
        max_iter=1000,
        random_state=RANDOM_STATE,
        class_weight='balanced',
        multi_class='multinomial',
        solver='lbfgs'
    )
    
    lr_model.fit(X_train_scaled, y_train)
    y_pred_lr = lr_model.predict(X_test_scaled)
    
    test_acc_lr = accuracy_score(y_test, y_pred_lr)
    test_f1_lr = f1_score(y_test, y_pred_lr, average='macro')
    
    print(f"Logistic Regression - Accuracy: {test_acc_lr:.4f}, F1: {test_f1_lr:.4f}")
    
    with open(os.path.join(output_dir, 'logistic_regression_model.pkl'), 'wb') as f:
        pickle.dump(lr_model, f)
    with open(os.path.join(output_dir, 'scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)
    with open(os.path.join(output_dir, 'label_encoder.pkl'), 'wb') as f:
        pickle.dump(label_encoder, f)
    
    cm_rf = confusion_matrix(y_test, y_pred_rf)
    cm_lr = confusion_matrix(y_test, y_pred_lr)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    class_names = label_encoder.classes_
    
    sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names, ax=axes[0])
    axes[0].set_title('Random Forest')
    
    sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Greens',
                xticklabels=class_names, yticklabels=class_names, ax=axes[1])
    axes[1].set_title('Logistic Regression')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    with open(os.path.join(output_dir, 'results.txt'), 'w') as f:
        f.write("MODEL EVALUATION RESULTS\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Total samples: {len(X)}\n")
        f.write(f"Training samples: {len(X_train)}\n")
        f.write(f"Test samples: {len(X_test)}\n\n")
        f.write("RANDOM FOREST\n")
        f.write(f"Test Accuracy: {test_acc_rf:.4f}\n")
        f.write(f"Test F1-score: {test_f1_rf:.4f}\n\n")
        f.write("LOGISTIC REGRESSION\n")
        f.write(f"Test Accuracy: {test_acc_lr:.4f}\n")
        f.write(f"Test F1-score: {test_f1_lr:.4f}\n")
    
    print("Training completed")

if __name__ == "__main__":
    train_and_evaluate()
