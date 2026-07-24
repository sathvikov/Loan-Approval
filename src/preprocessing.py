import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

class LoanDataPreprocessor:
    def __init__(self):
        self.medians = {}
        self.modes = {}
        self.label_encoders = {}
        self.is_fitted = False
        
        # Categorical and Numerical features definition
        self.categorical_cols = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area', 'Credit_History']
        self.numerical_cols = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term']
        
    def fit(self, df):
        """
        Learn the imputation values and encoder mappings from the training data.
        """
        # Copy df to avoid modification warnings
        df_copy = df.copy()
        
        # Handle Dependents encoding for mode calculation
        if 'Dependents' in df_copy.columns:
            df_copy['Dependents'] = df_copy['Dependents'].astype(str).replace('nan', np.nan)
            
        # 1. Learn Imputation values
        # Median for continuous features
        for col in ['LoanAmount', 'Loan_Amount_Term', 'ApplicantIncome', 'CoapplicantIncome']:
            if col in df_copy.columns:
                self.medians[col] = df_copy[col].median()
                
        # Mode for categorical features
        for col in self.categorical_cols:
            if col in df_copy.columns:
                # Get the mode value (handles NaNs by taking the first non-null mode)
                mode_series = df_copy[col].mode()
                self.modes[col] = mode_series[0] if not mode_series.empty else 'Unknown'
        
        # Let's ensure Credit_History mode is numeric/float or int as present in data
        if 'Credit_History' in self.modes:
            # Usually 1.0 or 0.0, save it
            self.modes['Credit_History'] = float(self.modes['Credit_History'])
            
        # 2. Learn Label Encoders for categorical features
        # Note: We preprocess the column values before fitting encoders
        for col in ['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area']:
            if col in df_copy.columns:
                le = LabelEncoder()
                # Fill missing temporarily to fit the encoder
                non_null_val = df_copy[col].fillna(self.modes[col]).astype(str)
                le.fit(non_null_val)
                self.label_encoders[col] = le
                
        self.is_fitted = True
        return self
        
    def transform(self, df):
        """
        Clean, engineer features, and encode the data using learned parameters.
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor has not been fitted yet! Call fit() before transform().")
            
        df_clean = df.copy()
        
        # Remove Loan_ID as it is an identifier, not a feature
        if 'Loan_ID' in df_clean.columns:
            df_clean = df_clean.drop(columns=['Loan_ID'])
            
        # 1. Apply Imputations
        # Numerical Columns
        for col, median_val in self.medians.items():
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].fillna(median_val)
                
        # Categorical Columns
        for col, mode_val in self.modes.items():
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].fillna(mode_val)
                
        # 2. Feature Engineering
        # Map Dependents '3+' to 3 and convert to numeric
        if 'Dependents' in df_clean.columns:
            # Map values
            df_clean['Dependents'] = df_clean['Dependents'].astype(str).replace('3+', '3')
            df_clean['Dependents_Numeric'] = pd.to_numeric(df_clean['Dependents'], errors='coerce').fillna(0).astype(int)
        else:
            df_clean['Dependents_Numeric'] = 0
            
        # Total Household Income
        df_clean['TotalIncome'] = df_clean['ApplicantIncome'] + df_clean['CoapplicantIncome']
        
        # Loan to Income Ratio
        # Avoid division by zero by replacing zero incomes with a small value
        safe_total_income = df_clean['TotalIncome'].replace(0, 1.0)
        df_clean['Loan_Income_Ratio'] = df_clean['LoanAmount'] / safe_total_income
        
        # Income per Dependent
        df_clean['Income_Per_Dependent'] = df_clean['TotalIncome'] / (df_clean['Dependents_Numeric'] + 1)
        
        # Log Transformations for highly skewed variables
        df_clean['ApplicantIncome_Log'] = np.log1p(df_clean['ApplicantIncome'])
        df_clean['TotalIncome_Log'] = np.log1p(df_clean['TotalIncome'])
        df_clean['LoanAmount_Log'] = np.log1p(df_clean['LoanAmount'])
        
        # 3. Categorical encoding
        # Label encode binary/categorical variables
        for col, le in self.label_encoders.items():
            if col in df_clean.columns:
                # Use transform, handle unseen classes by mapping to first class
                # (in production we'd use a class or map unseen to a fallback, but here standard works)
                val_str = df_clean[col].astype(str)
                # Clip values to those present in encoder classes
                classes = set(le.classes_)
                val_str = val_str.apply(lambda x: x if x in classes else list(classes)[0])
                df_clean[col] = le.transform(val_str)
                
        # For Dependents, map directly to integer 'Dependents_Numeric' and drop string column
        if 'Dependents' in df_clean.columns:
            df_clean = df_clean.drop(columns=['Dependents'])
            
        # Make sure Credit_History is integer (0 or 1)
        if 'Credit_History' in df_clean.columns:
            df_clean['Credit_History'] = df_clean['Credit_History'].astype(int)
            
        # Target variable encoding (if it exists in the dataframe)
        if 'Loan_Status' in df_clean.columns:
            df_clean['Loan_Status'] = df_clean['Loan_Status'].map({'Y': 1, 'N': 0})
            
        return df_clean

def load_data(file_path):
    """Loads CSV dataset."""
    return pd.read_csv(file_path)
