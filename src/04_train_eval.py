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
    
    feature_cols = [c for c in df.columns if c not in ['risk_level', 'crime_type', 'most_common_crime_type']]
    X = df[feature_cols]
    y_risk = df['risk_level']
    y_crime = df['crime_type'].fillna('Unknown')
    
    risk_encoder = LabelEncoder()
    crime_encoder = LabelEncoder()
    y_risk_encoded = risk_encoder.fit_transform(y_risk)
    y_crime_encoded = crime_encoder.fit_transform(y_crime)
    
    X_train, X_test, y_risk_train, y_risk_test, y_crime_train, y_crime_test = train_test_split(
        X, y_risk_encoded, y_crime_encoded, test_size=test_size, random_state=RANDOM_STATE, stratify=y_risk_encoded
    )
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("\nTraining Risk Level Model...")
    rf_risk = RandomForestClassifier(
        n_estimators=100, max_depth=20, min_samples_split=5,
        min_samples_leaf=2, random_state=RANDOM_STATE, n_jobs=-1, class_weight='balanced'
    )
    rf_risk.fit(X_train, y_risk_train)
    risk_pred = rf_risk.predict(X_test)
    risk_acc = accuracy_score(y_risk_test, risk_pred)
    risk_f1 = f1_score(y_risk_test, risk_pred, average='macro')
    print(f"Risk Model - Accuracy: {risk_acc:.4f}, F1: {risk_f1:.4f}")
    
    print("\nTraining Crime Type Model...")
    rf_crime = RandomForestClassifier(
        n_estimators=100, max_depth=20, min_samples_split=5,
        min_samples_leaf=2, random_state=RANDOM_STATE, n_jobs=-1, class_weight='balanced'
    )
    rf_crime.fit(X_train, y_crime_train)
    crime_pred = rf_crime.predict(X_test)
    crime_acc = accuracy_score(y_crime_test, crime_pred)
    crime_f1 = f1_score(y_crime_test, crime_pred, average='macro')
    print(f"Crime Type Model - Accuracy: {crime_acc:.4f}, F1: {crime_f1:.4f}")
    
    with open(os.path.join(output_dir, 'risk_model.pkl'), 'wb') as f:
        pickle.dump(rf_risk, f)
    with open(os.path.join(output_dir, 'crime_model.pkl'), 'wb') as f:
        pickle.dump(rf_crime, f)
    with open(os.path.join(output_dir, 'risk_encoder.pkl'), 'wb') as f:
        pickle.dump(risk_encoder, f)
    with open(os.path.join(output_dir, 'crime_encoder.pkl'), 'wb') as f:
        pickle.dump(crime_encoder, f)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    cm_risk = confusion_matrix(y_risk_test, risk_pred)
    sns.heatmap(cm_risk, annot=True, fmt='d', cmap='Blues',
                xticklabels=risk_encoder.classes_, yticklabels=risk_encoder.classes_, ax=axes[0])
    axes[0].set_title('Risk Level Model')
    
    top_crimes = pd.Series(crime_encoder.classes_).value_counts().head(10)
    crime_cm = confusion_matrix(y_crime_test, crime_pred)
    sns.heatmap(crime_cm[:10, :10], annot=True, fmt='d', cmap='Greens', ax=axes[1])
    axes[1].set_title('Crime Type Model (Top 10)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    with open(os.path.join(output_dir, 'results.txt'), 'w') as f:
        f.write("MODEL EVALUATION RESULTS\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Total samples: {len(X)}\n")
        f.write(f"Training samples: {len(X_train)}\n")
        f.write(f"Test samples: {len(X_test)}\n\n")
        f.write("RISK LEVEL MODEL\n")
        f.write(f"Test Accuracy: {risk_acc:.4f}\n")
        f.write(f"Test F1-score: {risk_f1:.4f}\n\n")
        f.write("CRIME TYPE MODEL\n")
        f.write(f"Test Accuracy: {crime_acc:.4f}\n")
        f.write(f"Test F1-score: {crime_f1:.4f}\n")
    
    print("Training completed")

if __name__ == "__main__":
    train_and_evaluate()
