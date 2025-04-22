import collections
from typing import Any

import joblib
import numpy as np
import pandas as pd
import preprocess
import xgboost
from feature_engine import datetime as fe_datetime
from numpy.typing import ArrayLike
from sklearn import ensemble, metrics, preprocessing


def _encode(
    data: pd.DataFrame,
    encoders: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any] | None]:
    encoders_ = {}

    # Label encode each categorical column
    label_cols = [col for col in data.columns if data[col].dtype == "object"]
    for col in label_cols:
        if encoders is None:
            encoder = preprocessing.OrdinalEncoder(
                handle_unknown="use_encoded_value", unknown_value=-1
            )
            data[[col]] = encoder.fit_transform(data[[col]].astype(str))
            encoders_[col] = encoder
        else:
            data[[col]] = encoders[col].transform(data[[col]].astype(str))

    # Encode each datetime column into month and day features
    datetime_cols = [
        col for col in data.columns if data[col].dtypes == "datetime64[ns]"
    ]
    for col in datetime_cols:
        if encoders is None:
            dtfs = fe_datetime.DatetimeFeatures(
                features_to_extract=["month", "day_of_month"]
            )
            data = pd.concat(
                [
                    data.drop(col, axis=1),
                    pd.DataFrame(dtfs.fit_transform(data[[col]])),
                ],
                axis=1,
            )
            encoders_[col] = dtfs
        else:
            data = pd.concat(
                [
                    data.drop(col, axis=1),
                    pd.DataFrame(encoders[col].transform(data[[col]])),
                ],
                axis=1,
            )

    return (data, encoders_) if encoders is None else (data, None)


def _train_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame | pd.Series,
) -> ensemble.RandomForestClassifier:
    print("Training Random Forest Model...")
    rf_model = ensemble.RandomForestClassifier(n_estimators=500, random_state=42)
    rf_model.fit(X_train, y_train)
    return rf_model


def _train_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame | pd.Series,
) -> xgboost.XGBClassifier:
    print("Training XGBoost Model...")
    y_count = collections.Counter(y_train)
    xgb_model = xgboost.XGBClassifier(
        scale_pos_weight=(y_count[0] / y_count[1]),
        colsample_bytree=0.8,
        learning_rate=0.3,
        max_depth=5,
        subsample=1,
        n_estimators=200,
        eval_metric="logloss",
        random_state=42,
    )
    xgb_model.fit(X_train, y_train)
    return xgb_model


def _evaluate_model(model_name: str, y_true: ArrayLike, y_pred: ArrayLike) -> None:
    accuracy = metrics.accuracy_score(y_true, y_pred)
    report = metrics.classification_report(y_true, y_pred)
    print(f"\n{model_name} Accuracy: {accuracy:.4f}")
    print(f"{model_name} Classification Report:\n{report}")


class EnsembleModel:
    def __init__(self, preprocessor: preprocess.Preprocessor) -> None:
        self.preprocessor = preprocessor
        self.encoders = None
        self.rf_model = None
        self.xgb_model = None

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.DataFrame | pd.Series,
    ) -> None:
        X_train_encoded, self.encoders = _encode(X_train)
        self.rf_model = _train_random_forest(X_train_encoded, y_train)
        self.xgb_model = _train_xgboost(X_train_encoded, y_train)

    def encode(self, data: pd.DataFrame) -> pd.DataFrame:
        if self.encoders is None:
            raise RuntimeError("Model is not fitted yet. Call `fit` before `encode`.")
        data_encoded, _ = _encode(data, self.encoders)
        return data_encoded

    def predict(self, data: pd.DataFrame) -> tuple[ArrayLike, ArrayLike, ArrayLike]:
        if self.encoders is None or self.rf_model is None or self.xgb_model is None:
            raise RuntimeError("Model is not fitted yet. Call `fit` before `predict`.")

        data_encoded = self.encode(data)
        y_pred_rf = self.rf_model.predict(data_encoded)
        y_pred_xgb = self.xgb_model.predict(data_encoded)
        y_pred_ensemble = np.logical_or(y_pred_rf, y_pred_xgb).astype(int)

        return y_pred_ensemble, y_pred_rf, y_pred_xgb

    def predict_from_json(self, json_path: str) -> pd.DataFrame:
        data = self.preprocessor.process_json(json_path)
        data_unencoded = data.copy()
        y_pred, _, _ = self.predict(data)
        data_unencoded["is_anomalous"] = y_pred
        return data_unencoded[["merchant", "po_number", "invoice_date", "is_anomalous"]]


def train_and_save_model(
    json_path: str, save_path: str = "ensemble_model.pkl"
) -> EnsembleModel:
    print("Running Preprocessing...")
    preprocessor = preprocess.Preprocessor()
    train_df, test_df = preprocessor.split_and_fit(json_path)

    X_train = train_df.drop(columns="is_anomalous")
    y_train = train_df["is_anomalous"]
    X_test = test_df.drop(columns="is_anomalous")
    y_test = test_df["is_anomalous"]

    print("Training Ensemble Model...")

    model = EnsembleModel(preprocessor)
    model.fit(X_train, y_train)

    y_pred_ensemble, y_pred_rf, y_pred_xgb = model.predict(X_test)
    _evaluate_model("Random Forest", y_test, y_pred_rf)
    _evaluate_model("XGBoost", y_test, y_pred_xgb)
    _evaluate_model("Ensemble Model", y_test, y_pred_ensemble)

    with open(save_path, "wb") as f:
        joblib.dump(model, f)

    print(f"Model saved to '{save_path}'.\n")

    return model


def load_model(path: str = "ensemble_model.pkl") -> EnsembleModel:
    with open(path, "rb") as f:
        model = joblib.load(f)

    if not isinstance(model, EnsembleModel):
        raise TypeError(
            f"The loaded object is not an EnsembleModel. Got {type(model)} instead."
        )

    print(f"Model loaded from '{path}'.")

    return model
