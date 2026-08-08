import os
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, roc_auc_score
)
import utils

def main():
    print("=" * 60)
    print("  Customer Churn Prediction System - Model Training Script  ")
    print("=" * 60)
    
    # 1. Load and clean the dataset
    data_path = 'customer_churn.csv'
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found. Please place the dataset in the working directory.")
        return
        
    print(f"Loading and cleaning data from '{data_path}'...")
    df = utils.load_and_clean_data(data_path)
    print(f"Dataset loaded. Shape: {df.shape}")
    
    # 2. Split Features and Target
    num_features, cat_features, target_col = utils.get_features_lists()
    
    # Verify columns exist
    missing_cols = [col for col in (num_features + cat_features + [target_col]) if col not in df.columns]
    if missing_cols:
        print(f"Error: Missing columns in dataset: {missing_cols}")
        return
        
    X = df[num_features + cat_features]
    y = df[target_col]
    
    # 3. Train/Test Split (80% Train, 20% Test)
    print("Splitting dataset into train and test sets (80% train, 20% test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train set size: {X_train.shape[0]} samples")
    print(f"Test set size: {X_test.shape[0]} samples")
    
    # 4. Model Building (Pipeline containing Preprocessing + Logistic Regression)
    print("Building model preprocessing and classifier pipeline...")
    preprocessor = utils.get_preprocessing_pipeline()
    
    # Create the complete pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'))
    ])
    
    # 5. Train the Model
    print("Training Logistic Regression model...")
    pipeline.fit(X_train, y_train)
    print("Model training complete.")
    
    # 6. Evaluate Model Performance
    print("Evaluating model...")
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    print("\n" + "-"*40)
    print("MODEL PERFORMANCE METRICS (TEST SET):")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"ROC AUC:   {roc_auc:.4f}")
    print("\nConfusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(report)
    print("-"*40)
    
    # Compute ROC Curve for serialization
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    
    # 7. Save Model Pipeline
    model_output_path = 'churn_model.pkl'
    print(f"Saving serialized pipeline to '{model_output_path}'...")
    joblib.dump(pipeline, model_output_path)
    print("Model saved successfully.")
    
    # 8. Save Metrics for Streamlit display
    metrics_path = 'model_metrics.json'
    print(f"Saving model evaluation metrics to '{metrics_path}'...")
    
    # Convert types to serializable format
    metrics_data = {
        'accuracy': float(acc),
        'precision': float(prec),
        'recall': float(rec),
        'f1_score': float(f1),
        'roc_auc': float(roc_auc),
        'confusion_matrix': cm.tolist(),
        'classification_report_dict': classification_report(y_test, y_pred, output_dict=True),
        'roc_curve': {
            # Subsample to keep JSON size reasonable
            'fpr': list(fpr[::max(1, len(fpr)//100)]),
            'tpr': list(tpr[::max(1, len(tpr)//100)])
        }
    }
    
    with open(metrics_path, 'w') as f:
        json.dump(metrics_data, f, indent=4)
    print("Metrics saved successfully.")
    print("=" * 60)

if __name__ == '__main__':
    main()
