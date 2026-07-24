import os
import pandas as pd
from src.preprocessing import load_data, LoanDataPreprocessor
from src.model import split_features_target, train_and_compare_models, save_pipeline
from src.utils import (
    plot_correlation_heatmap,
    plot_distributions,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_feature_importance
)

def run_pipeline():
    print("=" * 60)
    print("Starting Loan Approval Prediction Machine Learning Pipeline")
    print("=" * 60)
    
    # 1. Load Data
    data_path = 'data/loan_approval_dataset.csv'
    print(f"Loading data from: {data_path}")
    raw_df = load_data(data_path)
    print(f"Dataset loaded. Dimensions: {raw_df.shape[0]} rows, {raw_df.shape[1]} columns.")
    
    # Print initial summary of missing values
    missing = raw_df.isnull().sum()
    print("\nInitial Missing Values:")
    print(missing[missing > 0])
    
    # 2. Fit and Transform Data
    print("\nFitting preprocessor and cleaning data...")
    preprocessor = LoanDataPreprocessor()
    preprocessor.fit(raw_df)
    preprocessed_df = preprocessor.transform(raw_df)
    print("Data cleaning & feature engineering complete.")
    
    # 3. Define Features and Target
    # List of features we want to feed into the model
    features_list = [
        'Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area', 'Credit_History', 
        'Dependents_Numeric', 'TotalIncome_Log', 'LoanAmount_Log', 'Loan_Amount_Term', 
        'Loan_Income_Ratio', 'Income_Per_Dependent'
    ]
    target_col = 'Loan_Status'
    
    X, y = split_features_target(preprocessed_df, features_list, target_col)
    
    # 4. Train and Compare Models
    print("\nTraining models and comparing performance...")
    results, best_model_name, scaler, X_train_scaled, X_test_scaled, y_train, y_test = train_and_compare_models(X, y)
    
    # Print comparison table
    print("\nModel Comparison Summary:")
    print(f"{'Model Name':<25} | {'Accuracy':<10} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'ROC-AUC':<10}")
    print("-" * 85)
    for name, metrics in results.items():
        print(f"{name:<25} | {metrics['accuracy']:.4f}     | {metrics['precision']:.4f}     | {metrics['recall']:.4f}     | {metrics['f1_score']:.4f}     | {metrics['roc_auc']:.4f}")
        
    print(f"\n>>> Selected Best Model: {best_model_name}")
    best_metrics = results[best_model_name]
    
    # 5. Generate and Save Visual Reports
    print("\nGenerating visual reports...")
    
    # Correlation heatmap of numeric features
    numeric_cols = [
        'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 
        'TotalIncome', 'Loan_Income_Ratio', 'Income_Per_Dependent', 'Dependents_Numeric'
    ]
    # Use preprocessed dataframe columns
    plot_correlation_heatmap(preprocessed_df, numeric_cols, 'reports/correlation_heatmap.png')
    
    # Distributions plot
    plot_distributions(preprocessed_df, 'reports/distribution_plots.png')
    
    # Confusion Matrix for best model
    plot_confusion_matrix(best_metrics['y_test'], best_metrics['y_pred'], 'reports/confusion_matrix.png')
    
    # ROC Curve for best model
    plot_roc_curve(best_metrics['y_test'], best_metrics['y_prob'], best_metrics['roc_auc'], best_model_name, 'reports/roc_curve.png')
    
    # Feature Importance for best model
    plot_feature_importance(best_metrics['model_object'], features_list, best_model_name, 'reports/feature_importance.png')
    
    # 6. Save Pipeline
    best_model = best_metrics['model_object']
    model_save_path = 'models/best_loan_model.joblib'
    save_pipeline(preprocessor, scaler, best_model, features_list, model_save_path)
    
    print("\n" + "=" * 60)
    print("ML Pipeline Execution Complete!")
    print("Check 'reports/' for visualizations and 'models/' for saved pipeline.")
    print("=" * 60)

if __name__ == '__main__':
    run_pipeline()
