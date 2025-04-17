"""
This module builds and runs a Random Forest model to predict which 
invoices are anomalous. The input is a parquet file, which is the output from 
preprocess.py, containing invoice data after feature engineering preprocessing.

The output is...

"""

import pandas as pd
from feature_engine import datetime as fe_datetime
from sklearn import ensemble, metrics, preprocessing


def main() -> None:
    """Train random forest and evaluate performance."""
    # Load data
    df_train = pd.read_parquet("train_df.parquet", engine="pyarrow")
    df_test = pd.read_parquet("test_df.parquet", engine="pyarrow")

    # Separate features X and target y (anomaly label 0 or 1)
    # Drop anomaly types to avoid data leakage
    X_train = df_train.drop(
        ["is_anomalous", "_ANOMALY_TYPES_DROP_BEFORE_TRAINING_"],
        axis=1,
        errors="ignore",
    )
    y_train = df_train["is_anomalous"]
    X_test = df_test.drop(
        ["is_anomalous", "_ANOMALY_TYPES_DROP_BEFORE_TRAINING_"],
        axis=1,
        errors="ignore",
    )
    y_test = df_test["is_anomalous"]

    # Use label encoding for categorical/name columns (merchant info)
    # Label encoding is fine because tree-based models don’t assume
    # ordering in the numbers

    # Identify categorical and datetime columns
    label_cols = [col for col in X_train.columns if X_train[col].dtype == "object"]
    datetime_cols = [
        col for col in X_train.columns if X_train[col].dtype == "datetime64[ns]"
    ]

    # Label encode each categorical column (keeping consistency between
    # train and test)
    label_cols = [col for col in X_train.columns if X_train[col].dtype == "object"]
    for col in label_cols:
        encoder = preprocessing.OrdinalEncoder(
            handle_unknown="use_encoded_value", unknown_value=-1
        )
        X_train[[col]] = encoder.fit_transform(X_train[[col]].astype(str))
        X_test[[col]] = encoder.transform(X_test[[col]].astype(str))

    # Encode each datetime column into separate features
    datetime_cols = [
        col for col in X_train.columns if X_train[col].dtypes == "datetime64[ns]"
    ]
    for col in datetime_cols:
        dtfs = fe_datetime.DatetimeFeatures(
            features_to_extract=["month", "day_of_month"]
        )
        X_train = pd.concat(
            [
                X_train.drop(col, axis=1),
                pd.DataFrame(dtfs.fit_transform(X_train[[col]])),
            ],
            axis=1,
        )
        X_test = pd.concat(
            [X_test.drop(col, axis=1), pd.DataFrame(dtfs.transform(X_test[[col]]))],
            axis=1,
        )

    # Initialize and train random forest classifier
    model = ensemble.RandomForestClassifier(n_estimators=500, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Accuracy evaluation
    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", metrics.classification_report(y_test, y_pred))


if __name__ == "__main__":
    main()
