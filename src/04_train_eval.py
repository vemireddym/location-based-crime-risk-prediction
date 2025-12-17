"""
Model Training and Evaluation Script
Trains Random Forest and Logistic Regression models, evaluates performance, and saves results.
"""

import pandas as pd
import numpy as np
import os
import sys
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Add parent directory to path for imports if needed
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set random seed for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

def train_and_evaluate(input_path='data/model_ready.csv', 
                      output_dir='outputs',
                      test_size=0.2):
    """
    Train and evaluate Random Forest and Logistic Regression models.
    
    Args:
        input_path: Path to model-ready dataset
        output_dir: Directory to save results and models
        test_size: Proportion of data for testing (default: 0.2)
    """
    print("=" * 60)
    print("Step 4: Model Training and Evaluation")
    print("=" * 60)
    
    # Check if input file exists
    if not os.path.exists(input_path):
        print(f"Error: Input file '{input_path}' not found.")
        print("Please run 03_labels.py first.")
        return None
    
    # Load model-ready data
    print(f"\nLoading model-ready data from {input_path}...")
    try:
        df = pd.read_csv(input_path)
        print(f"Loaded {len(df)} records")
        print(f"Features: {list(df.columns)}")
    except Exception as e:
        print(f"Error loading data: {e}")
        return None
    
    # Prepare features and target
    target_col = 'risk_level'
    
    if target_col not in df.columns:
        print(f"Error: Target column '{target_col}' not found in dataset.")
        return None
    
    # Separate features and target
    feature_cols = [col for col in df.columns if col != target_col]
    X = df[feature_cols].copy()
    y = df[target_col].copy()
    
    print(f"\nDataset info:")
    print(f"  - Features: {len(feature_cols)}")
    print(f"  - Samples: {len(X)}")
    print(f"  - Target distribution:")
    print(y.value_counts())
    
    # Encode target labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    class_names = label_encoder.classes_
    
    print(f"\nEncoded classes: {dict(zip(class_names, range(len(class_names))))}")
    
    # Train-test split
    print(f"\nSplitting data (train: {1-test_size:.0%}, test: {test_size:.0%})...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=test_size, random_state=RANDOM_STATE, stratify=y_encoded
    )
    
    print(f"  - Training set: {len(X_train)} samples")
    print(f"  - Test set: {len(X_test)} samples")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize results dictionary
    results = {}
    
    # Train Random Forest (Main Model)
    print("\n" + "=" * 60)
    print("Training Random Forest Classifier (Main Model)")
    print("=" * 60)
    
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight='balanced'  # Handle class imbalance
    )
    
    print("\nTraining...")
    rf_model.fit(X_train, y_train)
    
    # Predictions
    y_train_pred_rf = rf_model.predict(X_train)
    y_test_pred_rf = rf_model.predict(X_test)
    
    # Evaluate Random Forest
    train_acc_rf = accuracy_score(y_train, y_train_pred_rf)
    test_acc_rf = accuracy_score(y_test, y_test_pred_rf)
    train_f1_rf = f1_score(y_train, y_train_pred_rf, average='macro')
    test_f1_rf = f1_score(y_test, y_test_pred_rf, average='macro')
    
    print(f"\nRandom Forest Results:")
    print(f"  Training Accuracy: {train_acc_rf:.4f}")
    print(f"  Test Accuracy: {test_acc_rf:.4f}")
    print(f"  Training F1-score (macro): {train_f1_rf:.4f}")
    print(f"  Test F1-score (macro): {test_f1_rf:.4f}")
    
    # Per-class F1 scores
    f1_per_class_rf = f1_score(y_test, y_test_pred_rf, average=None)
    print(f"\n  Per-class F1-scores:")
    for i, class_name in enumerate(class_names):
        print(f"    {class_name}: {f1_per_class_rf[i]:.4f}")
    
    # Confusion matrix
    cm_rf = confusion_matrix(y_test, y_test_pred_rf)
    print(f"\n  Confusion Matrix:")
    print(cm_rf)
    
    results['Random_Forest'] = {
        'train_accuracy': train_acc_rf,
        'test_accuracy': test_acc_rf,
        'train_f1': train_f1_rf,
        'test_f1': test_f1_rf,
        'f1_per_class': dict(zip(class_names, f1_per_class_rf)),
        'confusion_matrix': cm_rf.tolist()
    }
    
    # Save Random Forest model
    rf_model_path = os.path.join(output_dir, 'random_forest_model.pkl')
    with open(rf_model_path, 'wb') as f:
        pickle.dump(rf_model, f)
    print(f"\n  ✓ Model saved to {rf_model_path}")
    
    # Train Logistic Regression (Comparison Model)
    print("\n" + "=" * 60)
    print("Training Logistic Regression (Comparison Model)")
    print("=" * 60)
    
    # Scale features for Logistic Regression
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
    
    print("\nTraining...")
    lr_model.fit(X_train_scaled, y_train)
    
    # Predictions
    y_train_pred_lr = lr_model.predict(X_train_scaled)
    y_test_pred_lr = lr_model.predict(X_test_scaled)
    
    # Evaluate Logistic Regression
    train_acc_lr = accuracy_score(y_train, y_train_pred_lr)
    test_acc_lr = accuracy_score(y_test, y_test_pred_lr)
    train_f1_lr = f1_score(y_train, y_train_pred_lr, average='macro')
    test_f1_lr = f1_score(y_test, y_test_pred_lr, average='macro')
    
    print(f"\nLogistic Regression Results:")
    print(f"  Training Accuracy: {train_acc_lr:.4f}")
    print(f"  Test Accuracy: {test_acc_lr:.4f}")
    print(f"  Training F1-score (macro): {train_f1_lr:.4f}")
    print(f"  Test F1-score (macro): {test_f1_lr:.4f}")
    
    # Per-class F1 scores
    f1_per_class_lr = f1_score(y_test, y_test_pred_lr, average=None)
    print(f"\n  Per-class F1-scores:")
    for i, class_name in enumerate(class_names):
        print(f"    {class_name}: {f1_per_class_lr[i]:.4f}")
    
    # Confusion matrix
    cm_lr = confusion_matrix(y_test, y_test_pred_lr)
    print(f"\n  Confusion Matrix:")
    print(cm_lr)
    
    results['Logistic_Regression'] = {
        'train_accuracy': train_acc_lr,
        'test_accuracy': test_acc_lr,
        'train_f1': train_f1_lr,
        'test_f1': test_f1_lr,
        'f1_per_class': dict(zip(class_names, f1_per_class_lr)),
        'confusion_matrix': cm_lr.tolist()
    }
    
    # Save Logistic Regression model and scaler
    lr_model_path = os.path.join(output_dir, 'logistic_regression_model.pkl')
    scaler_path = os.path.join(output_dir, 'scaler.pkl')
    
    with open(lr_model_path, 'wb') as f:
        pickle.dump(lr_model, f)
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    
    print(f"\n  ✓ Model saved to {lr_model_path}")
    print(f"  ✓ Scaler saved to {scaler_path}")
    
    # Save label encoder
    encoder_path = os.path.join(output_dir, 'label_encoder.pkl')
    with open(encoder_path, 'wb') as f:
        pickle.dump(label_encoder, f)
    print(f"  ✓ Label encoder saved to {encoder_path}")
    
    # Create confusion matrix visualization
    print("\nCreating confusion matrix visualization...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Random Forest confusion matrix
    sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                ax=axes[0])
    axes[0].set_title('Random Forest - Confusion Matrix')
    axes[0].set_ylabel('True Label')
    axes[0].set_xlabel('Predicted Label')
    
    # Logistic Regression confusion matrix
    sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Greens',
                xticklabels=class_names, yticklabels=class_names,
                ax=axes[1])
    axes[1].set_title('Logistic Regression - Confusion Matrix')
    axes[1].set_ylabel('True Label')
    axes[1].set_xlabel('Predicted Label')
    
    plt.tight_layout()
    confusion_matrix_path = os.path.join(output_dir, 'confusion_matrix.png')
    plt.savefig(confusion_matrix_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Confusion matrix saved to {confusion_matrix_path}")
    plt.close()
    
    # Save results to text file
    print("\nSaving results to text file...")
    results_path = os.path.join(output_dir, 'results.txt')
    
    with open(results_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("MODEL EVALUATION RESULTS\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("Dataset Information:\n")
        f.write(f"  Total samples: {len(X)}\n")
        f.write(f"  Training samples: {len(X_train)}\n")
        f.write(f"  Test samples: {len(X_test)}\n")
        f.write(f"  Features: {len(feature_cols)}\n")
        f.write(f"  Classes: {list(class_names)}\n\n")
        
        f.write("=" * 60 + "\n")
        f.write("RANDOM FOREST (Main Model)\n")
        f.write("=" * 60 + "\n")
        f.write(f"Training Accuracy: {train_acc_rf:.4f}\n")
        f.write(f"Test Accuracy: {test_acc_rf:.4f}\n")
        f.write(f"Training F1-score (macro): {train_f1_rf:.4f}\n")
        f.write(f"Test F1-score (macro): {test_f1_rf:.4f}\n\n")
        
        f.write("Per-class F1-scores:\n")
        for class_name, f1 in zip(class_names, f1_per_class_rf):
            f.write(f"  {class_name}: {f1:.4f}\n")
        f.write("\n")
        
        f.write("Confusion Matrix:\n")
        f.write(str(cm_rf))
        f.write("\n\n")
        
        f.write("=" * 60 + "\n")
        f.write("LOGISTIC REGRESSION (Comparison Model)\n")
        f.write("=" * 60 + "\n")
        f.write(f"Training Accuracy: {train_acc_lr:.4f}\n")
        f.write(f"Test Accuracy: {test_acc_lr:.4f}\n")
        f.write(f"Training F1-score (macro): {train_f1_lr:.4f}\n")
        f.write(f"Test F1-score (macro): {test_f1_lr:.4f}\n\n")
        
        f.write("Per-class F1-scores:\n")
        for class_name, f1 in zip(class_names, f1_per_class_lr):
            f.write(f"  {class_name}: {f1:.4f}\n")
        f.write("\n")
        
        f.write("Confusion Matrix:\n")
        f.write(str(cm_lr))
        f.write("\n\n")
        
        f.write("=" * 60 + "\n")
        f.write("MODEL COMPARISON\n")
        f.write("=" * 60 + "\n")
        f.write(f"Test Accuracy:\n")
        f.write(f"  Random Forest: {test_acc_rf:.4f}\n")
        f.write(f"  Logistic Regression: {test_acc_lr:.4f}\n")
        f.write(f"  Difference: {abs(test_acc_rf - test_acc_lr):.4f}\n\n")
        
        f.write(f"Test F1-score (macro):\n")
        f.write(f"  Random Forest: {test_f1_rf:.4f}\n")
        f.write(f"  Logistic Regression: {test_f1_lr:.4f}\n")
        f.write(f"  Difference: {abs(test_f1_rf - test_f1_lr):.4f}\n\n")
        
        if test_acc_rf > test_acc_lr:
            f.write("Winner: Random Forest (higher accuracy and F1-score)\n")
        else:
            f.write("Winner: Logistic Regression (higher accuracy and F1-score)\n")
    
    print(f"  ✓ Results saved to {results_path}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Random Forest - Test Accuracy: {test_acc_rf:.4f}, F1: {test_f1_rf:.4f}")
    print(f"Logistic Regression - Test Accuracy: {test_acc_lr:.4f}, F1: {test_f1_lr:.4f}")
    
    if test_acc_rf > test_acc_lr:
        print("\n✓ Random Forest performs better")
    else:
        print("\n✓ Logistic Regression performs better")
    
    return results

if __name__ == "__main__":
    results = train_and_evaluate()
    
    if results is not None:
        print("\n" + "=" * 60)
        print("Model training and evaluation completed successfully!")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("Model training failed. Please check the error messages above.")
        print("=" * 60)
        sys.exit(1)

