import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def split_features_target(df, features_list, target_col='Loan_Status'):
    """
    Split the dataframe into features and target.
    """
    X = df[features_list]
    y = df[target_col] if target_col in df.columns else None
    return X, y

def train_and_compare_models(X, y):
    """
    Split data, train multiple classifiers, and compare their performance.
    Returns a dictionary of models, their evaluations, and the name of the best model.
    """
    # 1. Stratified Train-Test Split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 2. Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 3. Define candidate models
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100, max_depth=8),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42, n_estimators=100, learning_rate=0.05, max_depth=4)
    }
    
    results = {}
    best_f1 = -1
    best_model_name = None
    
    # 4. Train and evaluate
    for name, model in models.items():
        # Train
        model.fit(X_train_scaled, y_train)
        
        # Predict
        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:, 1]
        
        # Calculate Metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_prob)
        
        results[name] = {
            'model_object': model,
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1_score': f1,
            'roc_auc': auc,
            'y_test': y_test,
            'y_pred': y_pred,
            'y_prob': y_prob
        }
        
        # Use F1-score and Accuracy to pick the best model
        # Prioritize F1-score to handle potential class imbalance
        if f1 > best_f1:
            best_f1 = f1
            best_model_name = name
            
    return results, best_model_name, scaler, X_train_scaled, X_test_scaled, y_train, y_test

def save_pipeline(preprocessor, scaler, model, features_list, file_path='models/best_loan_model.joblib'):
    """
    Save preprocessor, scaler, trained model, and feature list as a single joblib file.
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    pipeline_dict = {
        'preprocessor': preprocessor,
        'scaler': scaler,
        'model': model,
        'features': features_list
    }
    joblib.dump(pipeline_dict, file_path)
    print(f"Model pipeline successfully saved to {file_path}")
