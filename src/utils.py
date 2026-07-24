import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_curve

# Configure visual style for premium reports
plt.rcParams['figure.facecolor'] = '#fdfdfd'
plt.rcParams['axes.facecolor'] = '#fdfdfd'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['grid.color'] = '#e2e8f0'
plt.rcParams['grid.linestyle'] = '--'

def plot_correlation_heatmap(df, numeric_cols, output_path='reports/correlation_heatmap.png'):
    """
    Plots and saves a correlation heatmap for numerical columns.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.figure(figsize=(10, 8))
    
    # Calculate correlation matrix
    corr_matrix = df[numeric_cols].corr()
    
    # Create mask to hide upper triangle
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    
    # Draw heatmap
    sns.heatmap(
        corr_matrix, 
        mask=mask,
        annot=True, 
        fmt=".2f", 
        cmap='coolwarm', 
        vmin=-1, 
        vmax=1, 
        center=0,
        square=True, 
        linewidths=.5, 
        cbar_kws={"shrink": .8}
    )
    
    plt.title("Correlation Matrix of Numeric Features", fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved correlation heatmap to {output_path}")

def plot_distributions(df, output_path='reports/distribution_plots.png'):
    """
    Plots and saves the comparison of distributions of raw income/loan amounts
    versus their log-transformed values.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig, axes = plt.subplots(3, 2, figsize=(14, 15))
    
    palette = ['#4f46e5', '#0ea5e9'] # premium blue/indigo palette
    
    # Applicant Income raw vs log
    sns.histplot(df['ApplicantIncome'], kde=True, ax=axes[0, 0], color=palette[0], bins=30)
    axes[0, 0].set_title("Applicant Income Distribution (Raw)", fontweight='bold')
    axes[0, 0].set_xlabel("Income ($)")
    
    sns.histplot(df['ApplicantIncome_Log'], kde=True, ax=axes[0, 1], color=palette[1], bins=30)
    axes[0, 1].set_title("Applicant Income Distribution (Log-Transformed)", fontweight='bold')
    axes[0, 1].set_xlabel("Log(Income + 1)")
    
    # Total Income raw vs log
    sns.histplot(df['TotalIncome'], kde=True, ax=axes[1, 0], color=palette[0], bins=30)
    axes[1, 0].set_title("Total Household Income Distribution (Raw)", fontweight='bold')
    axes[1, 0].set_xlabel("Total Income ($)")
    
    sns.histplot(df['TotalIncome_Log'], kde=True, ax=axes[1, 1], color=palette[1], bins=30)
    axes[1, 1].set_title("Total Household Income Distribution (Log-Transformed)", fontweight='bold')
    axes[1, 1].set_xlabel("Log(Total Income + 1)")
    
    # Loan Amount raw vs log
    sns.histplot(df['LoanAmount'], kde=True, ax=axes[2, 0], color=palette[0], bins=30)
    axes[2, 0].set_title("Loan Amount Distribution (Raw)", fontweight='bold')
    axes[2, 0].set_xlabel("Loan Amount (Thousands $)")
    
    sns.histplot(df['LoanAmount_Log'], kde=True, ax=axes[2, 1], color=palette[1], bins=30)
    axes[2, 1].set_title("Loan Amount Distribution (Log-Transformed)", fontweight='bold')
    axes[2, 1].set_xlabel("Log(Loan Amount + 1)")
    
    plt.suptitle("Effect of Log Transformation on Skewed Features", fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved distribution plots to {output_path}")

def plot_confusion_matrix(y_true, y_pred, output_path='reports/confusion_matrix.png'):
    """
    Plots and saves the confusion matrix heatmap.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.figure(figsize=(7, 6))
    
    cm = confusion_matrix(y_true, y_pred)
    cm_df = pd.DataFrame(cm, index=['Actual Rejected (0)', 'Actual Approved (1)'], 
                         columns=['Predicted Rejected (0)', 'Predicted Approved (1)'])
    
    sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues', cbar=False, annot_kws={"size": 14})
    
    plt.title("Confusion Matrix", fontsize=14, fontweight='bold', pad=15)
    plt.ylabel("Actual Label", fontsize=11, fontweight='bold')
    plt.xlabel("Predicted Label", fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved confusion matrix plot to {output_path}")

def plot_roc_curve(y_true, y_prob, auc_score, model_name, output_path='reports/roc_curve.png'):
    """
    Plots and saves the ROC curve showing model performance.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.figure(figsize=(8, 6))
    
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    
    plt.plot(fpr, tpr, color='#4f46e5', lw=2.5, label=f'{model_name} (AUC = {auc_score:.3f})')
    plt.plot([0, 1], [0, 1], color='#94a3b8', lw=1.5, linestyle='--')
    
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    plt.grid(True, alpha=0.3)
    
    plt.title(f"Receiver Operating Characteristic (ROC) Curve", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
    plt.ylabel("True Positive Rate (Sensitivity)", fontsize=11)
    plt.legend(loc="lower right", fontsize=11, frameon=True, facecolor='white', edgecolor='#e2e8f0')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved ROC curve plot to {output_path}")

def plot_feature_importance(model, features_list, model_name, output_path='reports/feature_importance.png'):
    """
    Plots and saves the feature importances or coefficients of the model.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.figure(figsize=(10, 7))
    
    # Check if tree-based model (has feature_importances_) or linear model (has coef_)
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        title = f"Feature Importance ({model_name})"
        ylabel = "Relative Importance"
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_[0]) # Take absolute coefficients for importance
        title = f"Absolute Coefficients ({model_name})"
        ylabel = "Absolute Coefficient Value"
    else:
        print("Model does not have feature_importances_ or coef_ attribute. Skipping feature importance plot.")
        return
        
    # Create pandas Series
    feat_imp = pd.Series(importances, index=features_list).sort_values(ascending=True)
    
    # Plot horizontal bar chart
    feat_imp.plot(kind='barh', color='#0ea5e9', width=0.7)
    
    plt.title(title, fontsize=14, fontweight='bold', pad=15)
    plt.xlabel(ylabel, fontsize=11)
    plt.grid(True, axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved feature importance plot to {output_path}")
