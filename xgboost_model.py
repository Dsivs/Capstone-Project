"""
This module builds and runs an XGBoost model to predict which 
invoices are anomalous. The input is a parquet file, which is the output from 
preprocess.py, containing invoice data after feature engineering preprocessing.

The output is...

"""


from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import make_scorer, recall_score
from collections import Counter
import pandas as pd

# Load data
df_train = pd.read_parquet('train_df.parquet', engine='pyarrow')
df_test = pd.read_parquet('test_df.parquet', engine='pyarrow')

# Separate features X and target y (anomaly label 0 or 1)
X_train = df_train.drop(['is_anomalous', '_ANOMALY_TYPES_DROP_BEFORE_TRAINING_'], axis=1)
y_train = df_train['is_anomalous']
X_test = df_test.drop(['is_anomalous', '_ANOMALY_TYPES_DROP_BEFORE_TRAINING_'], axis=1)
y_test = df_test['is_anomalous']

# use label encoding for categorical/name columns (po_number, payment_method, country, state, and currency)
# label encoding is fine because tree-based models don’t assume ordering in the numbers

# remove merchant, merchant chain, merchant branch, and merchant address
# remove datetime cols, only use important info (time between invoice date and due: invoice_age)
X_train = X_train.drop(columns=['merchant', 'merchant_branch', 'merchant_chain', 'merchant_address', 'invoice_date', 'due_date'])
X_test = X_test.drop(columns=['merchant', 'merchant_branch', 'merchant_chain', 'merchant_address', 'invoice_date', 'due_date'])

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


# model for the best set of hyper-paramters

model = XGBClassifier(
    scale_pos_weight=(Counter(y_train)[0] / Counter(y_train)[1]),
    colsample_bytree=0.8,
    learning_rate=0.3,
    max_depth=5,
    subsample=1,
    n_estimators=200,
    eval_metric='logloss',
    random_state=42
)

model.fit(X_train, y_train)

# adjust the threshold for is_anomaly
y_proba = model.predict_proba(X_test)[:, 1]
y_pred_thresh = (y_proba >= 0.4).astype(int)  # lower threshold = higher recall

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred_thresh))
