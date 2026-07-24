import os
import sys
import joblib
import pandas as pd
import numpy as np

def load_saved_pipeline(model_path='models/best_loan_model.joblib'):
    """Loads the pre-trained preprocessing and model pipeline."""
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found.")
        print("Please run 'python main.py' first to train and save the model.")
        sys.exit(1)
    return joblib.load(model_path)

def make_prediction(pipeline, applicant_dict):
    """
    Takes a dictionary of applicant details, preprocesses them,
    and runs the prediction.
    """
    preprocessor = pipeline['preprocessor']
    scaler = pipeline['scaler']
    model = pipeline['model']
    features_list = pipeline['features']
    
    # 1. Convert single applicant dictionary to a pandas DataFrame
    df_raw = pd.DataFrame([applicant_dict])
    
    # 2. Apply preprocessing (Handles imputation, feature engineering, label encoding)
    df_processed = preprocessor.transform(df_raw)
    
    # 3. Extract the exact list of features expected by the model
    X = df_processed[features_list]
    
    # 4. Apply feature scaling
    X_scaled = scaler.transform(X)
    
    # 5. Predict
    prediction = model.predict(X_scaled)[0]
    probability = model.predict_proba(X_scaled)[0][1] # Probability of loan approval (class 1)
    
    return prediction, probability

def get_interactive_input():
    """Prompts the user in the CLI to enter applicant details."""
    print("\n" + "=" * 40)
    print("      Enter Applicant Information")
    print("=" * 40)
    
    details = {}
    
    # Categorical fields with validation
    details['Gender'] = input("Gender (Male/Female) [Male]: ").strip() or "Male"
    details['Married'] = input("Married (Yes/No) [Yes]: ").strip() or "Yes"
    details['Dependents'] = input("Number of Dependents (0/1/2/3+) [0]: ").strip() or "0"
    details['Education'] = input("Education (Graduate/Not Graduate) [Graduate]: ").strip() or "Graduate"
    details['Self_Employed'] = input("Self Employed (Yes/No) [No]: ").strip() or "No"
    
    # Numerical fields with validation
    try:
        details['ApplicantIncome'] = float(input("Applicant Monthly Income ($) [5000]: ") or 5000)
    except ValueError:
        details['ApplicantIncome'] = 5000.0
        
    try:
        details['CoapplicantIncome'] = float(input("Co-applicant Monthly Income ($) [0]: ") or 0)
    except ValueError:
        details['CoapplicantIncome'] = 0.0
        
    try:
        details['LoanAmount'] = float(input("Requested Loan Amount (in thousands $) [120]: ") or 120)
    except ValueError:
        details['LoanAmount'] = 120.0
        
    try:
        details['Loan_Amount_Term'] = float(input("Loan Term in Days (e.g. 360) [360]: ") or 360)
    except ValueError:
        details['Loan_Amount_Term'] = 360.0
        
    try:
        details['Credit_History'] = float(input("Credit History meets guidelines? (1 = Yes, 0 = No) [1]: ") or 1)
    except ValueError:
        details['Credit_History'] = 1.0
        
    details['Property_Area'] = input("Property Area (Urban/Semiurban/Rural) [Semiurban]: ").strip() or "Semiurban"
    
    return details

def main():
    pipeline = load_saved_pipeline()
    
    # Samples for demonstration
    approved_sample = {
        'Gender': 'Male', 'Married': 'Yes', 'Dependents': '1', 'Education': 'Graduate',
        'Self_Employed': 'No', 'ApplicantIncome': 8000.0, 'CoapplicantIncome': 3000.0,
        'LoanAmount': 150.0, 'Loan_Amount_Term': 360.0, 'Credit_History': 1.0,
        'Property_Area': 'Semiurban'
    }
    
    rejected_sample = {
        'Gender': 'Male', 'Married': 'No', 'Dependents': '0', 'Education': 'Not Graduate',
        'Self_Employed': 'No', 'ApplicantIncome': 2000.0, 'CoapplicantIncome': 0.0,
        'LoanAmount': 300.0, 'Loan_Amount_Term': 360.0, 'Credit_History': 0.0,
        'Property_Area': 'Rural'
    }
    
    print("=" * 60)
    print("               Loan Approval Prediction CLI")
    print("=" * 60)
    print("\nRunning sample predictions for validation...")
    
    # 1. High Probability Approval Sample
    pred_app, prob_app = make_prediction(pipeline, approved_sample)
    print(f"\nSample 1 (High Income, Good Credit History, Semiurban):")
    status = "APPROVED" if pred_app == 1 else "REJECTED"
    print(f"  Result: {status} (Approval Probability: {prob_app * 100:.2f}%)")
    
    # 2. High Probability Rejection Sample
    pred_rej, prob_rej = make_prediction(pipeline, rejected_sample)
    print(f"\nSample 2 (Low Income, Large Loan request, No Credit History):")
    status = "APPROVED" if pred_rej == 1 else "REJECTED"
    print(f"  Result: {status} (Approval Probability: {prob_rej * 100:.2f}%)")
    
    # Interactive loop
    while True:
        choice = input("\nWould you like to test a custom applicant? (y/n) [n]: ").strip().lower()
        if choice not in ['y', 'yes']:
            break
            
        custom_input = get_interactive_input()
        pred, prob = make_prediction(pipeline, custom_input)
        
        print("\n" + "-" * 40)
        print("          PREDICTION RESULT")
        print("-" * 40)
        status = "APPROVED" if pred == 1 else "REJECTED"
        color_start = "\033[92m" if pred == 1 else "\033[91m"
        color_end = "\033[0m"
        print(f"Loan Status: {color_start}{status}{color_end}")
        print(f"Confidence Score (Approval Prob): {prob * 100:.2f}%")
        print("-" * 40)

if __name__ == '__main__':
    main()
