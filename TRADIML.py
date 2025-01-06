import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import shap
from xgboost import XGBClassifier

# Step 1: Load the traditional dataset
file_path = r"/home/heisenberg/Desktop/UCI_Credit_Card.csv"  # Change this path if needed
df = pd.read_csv(file_path)

# Select relevant traditional columns
selected_columns = ['LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE',
                    'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6',
                    'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6',
                    'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6',
                    'default.payment.next.month']  # Target column is 'default.payment.next.month'

# Reduce dataset to selected columns
df_traditional = df[selected_columns]

# Step 2: Feature Engineering

def feature_engineering(df):
    # Binning 'AGE' into categories
    bins = [0, 25, 40, 60, np.inf]
    labels = ['young', 'middle-aged', 'old', 'senior']
    df['AGE_BIN'] = pd.cut(df['AGE'], bins=bins, labels=labels, right=False)
    
    # Binning 'LIMIT_BAL' into categories
    limit_bins = [0, 50000, 100000, 200000, np.inf]
    limit_labels = ['low', 'medium', 'high', 'very high']
    df['LIMIT_BAL_BIN'] = pd.cut(df['LIMIT_BAL'], bins=limit_bins, labels=limit_labels, right=False)
    
    # Adding interaction features
    df['PAY_AMT_SUM'] = df[['PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']].sum(axis=1)
    df['BILL_AMT_SUM'] = df[['BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6']].sum(axis=1)
    df['PAY_SUM'] = df[['PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']].sum(axis=1)
    
    return df

# Apply feature engineering
df_traditional = feature_engineering(df_traditional)

# Step 3: Preprocessing the data
def preprocess_data(df):
    # Handle missing values (drop rows with missing data)
    df = df.dropna()
    
    # Encode categorical variables (e.g., SEX, EDUCATION, MARRIAGE, AGE_BIN, LIMIT_BAL_BIN)
    label_encoders = {}
    for col in ["SEX", "EDUCATION", "MARRIAGE", "AGE_BIN", "LIMIT_BAL_BIN"]:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
    
    # Scale numerical features
    scaler = StandardScaler()
    numerical_cols = ['LIMIT_BAL', 'AGE', 'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6',
                      'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6',
                      'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6',
                      'PAY_AMT_SUM', 'BILL_AMT_SUM', 'PAY_SUM']
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    
    return df, scaler, label_encoders

# Apply preprocessing
df_preprocessed, scaler, label_encoders = preprocess_data(df_traditional)

# Step 4: Split the data into features (X) and target (y)
X = df_preprocessed.drop(columns=["default.payment.next.month"])  # Features
y = df_preprocessed["default.payment.next.month"]  # Target column

# Step 5: Cross-validation setup
kfold = KFold(n_splits=5, shuffle=True, random_state=42)  # 5-fold cross-validation

# Step 6: Initialize the XGBoost model
model = XGBClassifier(
    n_estimators=500,
    learning_rate=0.1,
    max_depth=6,
    scale_pos_weight=1,  # Already balanced due to SMOTE
    random_state=42
)

# Step 7: Initialize SMOTE
smote = SMOTE(random_state=42)

# Step 8: Perform cross-validation with SMOTE inside the loop
accuracies = []
for train_index, test_index in kfold.split(X):
    # Split data into train and test
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    # Apply SMOTE to training data only
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    
    # Fit the model
    model.fit(X_train_resampled, y_train_resampled)
    
    # Predict and evaluate on test data
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    accuracies.append(accuracy)

# Step 9: Print the average accuracy from cross-validation
print(f"Cross-validation Accuracy: {np.mean(accuracies) * 100:.2f}%")
print(f"Cross-validation Scores: {accuracies}")

# Step 10: Fit the model on the whole resampled data (after cross-validation evaluation)
model.fit(X, y)

# Step 11: Predict and evaluate on the test data (optional final evaluation)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=39)
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]  # Get predicted probabilities for the positive class (default)

# Step 12: Evaluate the model on the test set
print("Accuracy:", accuracy_score(y_test, y_pred) * 100, "%")
print("Classification Report:\n", classification_report(y_test, y_pred))

# Step 13: Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Plotting the Confusion Matrix
plt.figure(figsize=(6, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['No Default', 'Default'], yticklabels=['No Default', 'Default'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Step 14: Loan Approval Decision (using a threshold)
threshold = 0.5  # You can adjust this threshold depending on the bank's risk appetite
loan_approval = np.where(y_prob >= threshold, 0, 1)  # 0 for approve, 1 for reject

# Step 15: Create a decision DataFrame to show loan approval outcomes
decision_df = pd.DataFrame({
    'Customer_ID': X_test.index,
    'Predicted_Probability_of_Default': y_prob,
    'Loan_Approval_Decision': loan_approval
})

# Step 16: Save the loan approval decisions to a CSV file
decision_file_path = r"/home/heisenberg/Desktop/loan_approval_decision.csv"
decision_df.to_csv(decision_file_path, index=False)

# Step 17: Print the first few loan approval decisions to the console
print("Loan Approval Decisions (First 5 customers):")
print(decision_df.head())


# SHAP for model interpretability
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Visualize the SHAP summary plot
shap.summary_plot(shap_values, X_test)



