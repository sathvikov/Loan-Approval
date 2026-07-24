import os
import sys
import joblib
import pandas as pd
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# Load the trained model pipeline
model_path = 'models/best_loan_model.joblib'
if not os.path.exists(model_path):
    print(f"Error: Model file '{model_path}' not found.")
    print("Please run 'python main.py' first to train and save the model.")
    sys.exit(1)

# Loaded pipeline dictionary contains: preprocessor, scaler, model, features
pipeline = joblib.load(model_path)
preprocessor = pipeline['preprocessor']
scaler = pipeline['scaler']
model = pipeline['model']
features_list = pipeline['features']

@app.route('/')
def home():
    """Renders the dashboard page."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """
    Accepts JSON input for applicant details and returns predictions.
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({'status': 'error', 'message': 'No input data provided'}), 400
            
        # Required features checklist
        required_fields = [
            'Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 
            'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 
            'Credit_History', 'Property_Area'
        ]
        
        # Verify fields and map default/empty values if missing
        applicant = {}
        for field in required_fields:
            val = data.get(field)
            if val is None or val == '':
                # Provide reasonable fallbacks if omitted
                if field in ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term']:
                    applicant[field] = 0.0
                elif field == 'Credit_History':
                    applicant[field] = 1.0
                elif field == 'Dependents':
                    applicant[field] = '0'
                else:
                    applicant[field] = 'Unknown'
            else:
                # Convert numeric fields to correct types
                if field in ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History']:
                    applicant[field] = float(val)
                else:
                    applicant[field] = str(val)
                    
        # 1. Convert to DataFrame
        df_raw = pd.DataFrame([applicant])
        
        # 2. Transform raw features using the fitted preprocessor
        df_processed = preprocessor.transform(df_raw)
        
        # 3. Extract exact list of features expected by the model
        X = df_processed[features_list]
        
        # 4. Scale features
        X_scaled = scaler.transform(X)
        
        # 5. Get model prediction and probability
        prediction = int(model.predict(X_scaled)[0])
        probability = float(model.predict_proba(X_scaled)[0][1])
        
        # 6. Generate reasoning and advice
        advice = []
        loan_to_income = applicant['LoanAmount'] / ((applicant['ApplicantIncome'] + applicant['CoapplicantIncome']) or 1.0)
        
        if applicant['Credit_History'] == 0:
            advice.append("CRITICAL: Lack of credit history or poor repayment record is the primary risk factor. Rebuilding your credit score will yield the highest chance of approval.")
            
        if loan_to_income > 0.4:
            advice.append(f"HIGH RISK: The requested loan amount (${applicant['LoanAmount']:.1f}k) is high relative to your monthly household income (${(applicant['ApplicantIncome'] + applicant['CoapplicantIncome']):.1f}). A loan-to-income ratio of {loan_to_income*100:.1f}% indicates high debt burden. Try applying for a lower loan amount or adding a co-applicant.")
            
        if applicant['Self_Employed'] == 'Yes' and applicant['ApplicantIncome'] < 3000:
            advice.append("RISK FACTOR: Self-employed applicants with lower primary incomes are classified as higher risk. Providing secondary collateral or co-signers is highly recommended.")
            
        if prediction == 1:
            if not advice:
                advice.append("EXCELLENT PROFILE: Your financial profile, stable income, and positive credit history align perfectly with approval criteria.")
            else:
                advice.append("STABLE PROFILE: Despite minor risk factors, your overall profile remains strong enough for standard approval.")
        else:
            if not advice:
                advice.append("INSUFFICIENT MARGIN: Your overall income and requested loan size fall just short of safety thresholds. Consider reducing the loan amount.")
                
        return jsonify({
            'status': 'success',
            'prediction': prediction,
            'probability': probability,
            'loan_to_income_ratio': loan_to_income,
            'advice': advice
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(e)}), 500

if __name__ == '__main__':
    # Serve locally on port 5000
    app.run(host='127.0.0.1', port=5000, debug=True)
