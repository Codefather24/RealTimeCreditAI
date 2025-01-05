import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Step 1: Load the traditional dataset
file_path = r"C:\Users\ritha\OneDrive\Desktop\archive\UCI_Credit_Card.csv"  # Change this path if needed
df = pd.read_csv(file_path)

# Select relevant traditional columns
selected_columns = ['LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE',
                    'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6',
                    'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6',
                    'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6',
                    'default.payment.next.month']  # Target column is 'default.payment.next.month'

# Reduce dataset to selected columns
df_traditional = df[selected_columns]

# Step 2: Preprocessing the data
def preprocess_data(df):
    # Handle missing values (drop rows with missing data)
    df = df.dropna()
    
    # Encode categorical variables (e.g., SEX, EDUCATION, MARRIAGE)
    label_encoders = {}
    for col in ["SEX", "EDUCATION", "MARRIAGE"]:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
    
    # Scale numerical features
    scaler = StandardScaler()
    numerical_cols = ['LIMIT_BAL', 'AGE', 'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6',
                      'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6',
                      'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    
    return df, scaler, label_encoders

# Apply preprocessing
df_preprocessed, scaler, label_encoders = preprocess_data(df_traditional)

# Step 3: Split the data into features (X) and target (y)
X = df_preprocessed.drop(columns=["default.payment.next.month"])  # Features
y = df_preprocessed["default.payment.next.month"]  # Target column

# Step 4: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Initialize and train the XGBoost model
model = XGBClassifier(n_estimators=500, learning_rate=0.1, max_depth=6, random_state=42)
model.fit(X_train, y_train)

# Step 6: Make predictions on the test set
y_pred = model.predict(X_test)

# Step 7: Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred)*100, "%")
print("Classification Report:\n", classification_report(y_test, y_pred))

# Step 8: Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Plotting the Confusion Matrix
plt.figure(figsize=(6, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['No Default', 'Default'], yticklabels=['No Default', 'Default'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
