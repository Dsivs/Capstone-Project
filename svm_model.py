"""
This module builds and runs a Support Vector Machine model to predict which 
invoices are anomalous. The input is a parquet file, which is the output from 
preprocess.py, containing invoice data after feature engineering preprocessing.

The output is...

"""


from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
import pandas as pd

# Load data
df_train = pd.read_parquet('train_df.parquet', engine='pyarrow')
df_test = pd.read_parquet('test_df.parquet', engine='pyarrow')

X_train = df_train.drop(
    ["is_anomalous", "_ANOMALY_TYPES_DROP_BEFORE_TRAINING_"], axis=1, errors="ignore"
)
y_train = df_train["is_anomalous"]
X_test = df_test.drop(
    ["is_anomalous", "_ANOMALY_TYPES_DROP_BEFORE_TRAINING_"], axis=1, errors="ignore"
)
y_test = df_test["is_anomalous"]

# Identify categorical columns and label-encode them
col_types = dict(X_train.dtypes)
label_cols = [col for col, dtype in col_types.items() if dtype == 'object']

encoders = {}
for col in label_cols:
    encoder = LabelEncoder()
    combined = pd.concat([X_train[col], X_test[col]], axis=0).astype(str)
    encoder.fit(combined)
    X_train[col] = encoder.transform(X_train[col].astype(str))
    X_test[col] = encoder.transform(X_test[col].astype(str))
    encoders[col] = encoder

# drop date columns, remove merchant, merchant chain, merchant branch, and merchant address (not important info)
X_train = X_train.drop(columns=['merchant', 'merchant_branch', 'merchant_chain', 'merchant_address', 'invoice_date', 'due_date'])
X_test = X_test.drop(columns=['merchant', 'merchant_branch', 'merchant_chain', 'merchant_address', 'invoice_date', 'due_date'])

# standardize
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# dimensionality reduction
pca = PCA(n_components=10, random_state=42)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

# Train an SVM with RBF Kernel on the PCA features
model = SVC(kernel='rbf',
            C=1.0,
            gamma='scale',
            class_weight='balanced',
            probability=True,
            random_state=42)

model.fit(X_train_pca, y_train)
y_pred = model.predict(X_test_pca)

# Threshold adjustment for better recall results
y_proba = model.predict_proba(X_test_pca)[:, 1]
y_pred_thresh = (y_proba >= 0.3).astype(int)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred_thresh))