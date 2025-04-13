"""
Module for Random Forest Classifier

This module provides ... 

Date: 
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import pandas as pd

# Load data
df_train = pd.read_parquet('train_df.parquet', engine='pyarrow')
df_test = pd.read_parquet('test_df.parquet', engine='pyarrow')

# Separate features X and target y (anomaly label 0 or 1)
X_train = df_train.drop('is_anomalous', axis=1)
y_train = df_train['is_anomalous']
X_test = df_test.drop('is_anomalous', axis=1)
y_test = df_test['is_anomalous']

# use label encoding for categorical/name columns (merchant info)
# label encoding is fine because tree-based models don’t assume ordering in the numbers

# Identify categorical columns
col_types = dict(X_train.dtypes)
label_cols = [col for col, dtype in col_types.items() if dtype == 'object']

# Label encode each categorical column (keeping consistency between train and test)
encoders = {}
for col in label_cols:
    encoder = LabelEncoder()
    combined = pd.concat([X_train[col], X_test[col]], axis=0).astype(str)
    encoder.fit(combined)
    X_train[col] = encoder.transform(X_train[col].astype(str))
    X_test[col] = encoder.transform(X_test[col].astype(str))
    encoders[col] = encoder  # Optional: store encoders for inverse_transform or future use

# remove datetime cols, only use important info (time between invoice date and due)
X_train['time_until_due'] = (X_train['due_date'] - X_train['invoice_date']).dt.days
X_test['time_until_due'] = (X_test['due_date'] - X_test['invoice_date']).dt.days
X_train = X_train.drop(columns=['invoice_date', 'due_date'])
X_test = X_test.drop(columns=['invoice_date', 'due_date'])

# initialize and train random forest classifier
model = RandomForestClassifier(n_estimators=500, 
                                random_state=42)
                                #class_weight = 'balanced')
                                #class_weight='balanced_subsample')
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# accuracy eval
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))