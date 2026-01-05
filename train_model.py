import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib
import os

# Define file paths
import glob
from sklearn.impute import KNNImputer

MODEL_FILE = 'model.pkl'
SCALER_FILE = 'scaler.pkl'

# ... (imports remain the same)

def train():
    # Find all CSV files in the current directory
    csv_files = glob.glob('*.csv')
    
    if not csv_files:
        print("Error: No CSV files found in the current directory.")
        return

    dataframes = []
    expected_cols = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']

    print(f"Found {len(csv_files)} CSV files: {csv_files}")

    for file in csv_files:
        try:
            df_temp = pd.read_csv(file)
            # Check if columns match
            if all(col in df_temp.columns for col in expected_cols):
                # Keep only necessary columns to avoid mismatch errors if extra cols exist
                df_temp = df_temp[expected_cols]
                dataframes.append(df_temp)
                print(f"Loaded {file} ({len(df_temp)} rows)")
            else:
                print(f"Skipping {file}: Missing expected columns.")
        except Exception as e:
            print(f"Error loading {file}: {e}")

    if not dataframes:
        print("Error: No valid datasets loaded.")
        return

    # Concatenate all dataframes
    df = pd.concat(dataframes, ignore_index=True)
    print(f"Total data loaded: {len(df)} rows")

    # 1. Data Cleaning: Replace 0 with NaN in columns where 0 is invalid
    invalid_zero_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    df[invalid_zero_cols] = df[invalid_zero_cols].replace(0, np.nan)

    # 2. Outlier Removal using IQR (Interquartile Range)
    # We only check specific columns for outliers to avoid removing too much data
    outlier_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'BMI', 'Age']
    
    Q1 = df[outlier_cols].quantile(0.25)
    Q3 = df[outlier_cols].quantile(0.75)
    IQR = Q3 - Q1
    
    # Filter out rows that are outside bounds
    # Note: any() with axis=1 checks if *any* column in the row is an outlier
    # We use ~ to keep rows that are NOT outliers
    # We handle NaNs by ensuring the condition doesn't evaluate to True for them (pandas comparison with NaN is False)
    condition = ~((df[outlier_cols] < (Q1 - 1.5 * IQR)) | (df[outlier_cols] > (Q3 + 1.5 * IQR))).any(axis=1)
    
    initial_len = len(df)
    df = df[condition]
    print(f"Outlier Removal: Removed {initial_len - len(df)} rows. Remaining: {len(df)}")

    if len(df) == 0:
        print("Error: All data removed by outlier detection! Please check your data or relax the IQR threshold.")
        return

    # Separate features and target
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']

    # 3. Imputation Strategy
    # A. Use Median for Glucose, BloodPressure, BMI (Robust to outliers)
    for col in ['Glucose', 'BloodPressure', 'BMI']:
        X[col].fillna(X[col].median(), inplace=True)

    # B. Use KNN Imputation for Insulin and SkinThickness (and any remaining)
    # KNN works best when other features (like Age) are present, so we do this BEFORE categorizing Age
    imputer = KNNImputer(n_neighbors=5)
    X_imputed = imputer.fit_transform(X)
    X = pd.DataFrame(X_imputed, columns=X.columns)

    # 3. Feature Engineering: Age Categories
    # Young: < 30, Middle: 30-50, Senior: > 50
    X['Age_Young'] = (X['Age'] < 30).astype(int)
    X['Age_Middle'] = ((X['Age'] >= 30) & (X['Age'] <= 50)).astype(int)
    X['Age_Senior'] = (X['Age'] > 50).astype(int)
    X = X.drop('Age', axis=1)

    print("Features used:", X.columns.tolist())

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train Logistic Regression model
    # class_weight='balanced' adjusts weights inversely proportional to class frequencies
    model = LogisticRegression(class_weight='balanced')
    model.fit(X_train_scaled, y_train)

    # Evaluate
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy:.2f}")

    # Save model and scaler
    joblib.dump(model, MODEL_FILE)
    joblib.dump(scaler, SCALER_FILE)
    print(f"Model saved to {MODEL_FILE}")
    print(f"Scaler saved to {SCALER_FILE}")

    # Save accuracy for display in app if needed (simple text file)
    with open('model_accuracy.txt', 'w') as f:
        f.write(str(accuracy))

if __name__ == "__main__":
    train()
