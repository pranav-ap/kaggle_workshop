import json
import yaml
import os
import warnings

import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error, root_mean_squared_error, mean_absolute_error, r2_score

from common import (Config, load_train_split_data, load_val_split_data, plot_charts, save_model)

warnings.filterwarnings('ignore')


def train_model(X_train, y_train, X_val, y_val, params):
    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=params['n_estimators'],
        early_stopping_rounds=20,
        eval_metric='rmse',
        random_state=Config.RANDOM_SEED,
    )

    eval_set = [(X_train, y_train), (X_val, y_val)]

    model.fit(
        X_train, y_train,
        eval_set=eval_set,
        verbose=True,
    )

    return model


def validate_model(model, X_val, y_val):
    y_pred = model.predict(X_val)

    val_metrics = {
        "r2": r2_score(y_val, y_pred),
        "mse": mean_squared_error(y_val, y_pred),
        "rmse": root_mean_squared_error(y_val, y_pred),
        "mae": mean_absolute_error(y_val, y_pred),
        "mape": np.mean(np.abs((y_val - y_pred) / y_val)) * 100,
    }

    print("Validation metrics:", val_metrics)

    return val_metrics


def store_val_metrics(val_metrics):
    metrics_path = os.path.join(Config.REPO_PATH, "outputs", "s5e9", "metrics.json")
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump(val_metrics, f, indent=4)


def train():
    with open("params.yaml") as f:
        params = yaml.safe_load(f)
        params = params["train"]
        print('Training parameters:', params)

    X_train, y_train = load_train_split_data()
    X_val, y_val = load_val_split_data()

    model = train_model(X_train, y_train, X_val, y_val, params)
    plot_charts(model)
    val_metrics = validate_model(model, X_val, y_val)
    store_val_metrics(val_metrics)
    save_model(model)

    return model


if __name__ == "__main__":
    train()
