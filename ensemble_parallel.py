from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn import ensemble
from xgboost import XGBClassifier
import pandas as pd
import numpy as np
from collections import Counter

# Load data
df_train = pd.read_parquet('train_df.parquet', engine='pyarrow')
df_test = pd.read_parquet('test_df.parquet', engine='pyarrow')
df_test_raw = df_test.copy()

# Separate features X and target y (anomaly label 0 or 1)
X_train = df_train.drop(['is_anomalous', '_ANOMALY_TYPES_DROP_BEFORE_TRAINING_'], axis=1)
y_train = df_train['is_anomalous']
X_test = df_test.drop(['is_anomalous', '_ANOMALY_TYPES_DROP_BEFORE_TRAINING_'], axis=1)
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

# remove datetime cols, only use important info (time between invoice date and due: invoice_age)
X_train = X_train.drop(columns=['invoice_date', 'due_date'])
X_test = X_test.drop(columns=['invoice_date', 'due_date'])



# random forest
rf_model = ensemble.RandomForestClassifier(n_estimators=200, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

# xbgoost
xgboost_model = XGBClassifier(
    scale_pos_weight=(Counter(y_train)[0] / Counter(y_train)[1]),
    colsample_bytree=0.8,
    learning_rate=0.3,
    max_depth=5,
    subsample=1,
    n_estimators=200,
    eval_metric='logloss',
    random_state=42
)
xgboost_model.fit(X_train, y_train)
y_pred_xgb = xgboost_model.predict(X_test)

y_pred_final = np.logical_or(y_pred_rf, y_pred_xgb).astype(int)
print("Custom Ensemble Accuracy:", accuracy_score(y_test, y_pred_final))
print("\nClassification Report:\n", classification_report(y_test, y_pred_final))


# --- Build Result DataFrame ---
df_test_raw = df_test_raw.reset_index(drop=True)
result_df = df_test_raw.copy()

result_df['rf_pred'] = y_pred_rf
result_df['xgb_pred'] = y_pred_xgb
result_df['final_anomaly'] = y_pred_final

print(result_df[['merchant', 'po_number', 'invoice_date', 'final_anomaly']])