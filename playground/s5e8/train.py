import warnings

import numpy as np
import xgboost as xgb
from prefect import flow, task
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score, roc_auc_score)

from common import (Config, load_train_data, process_data, save_model, train_val_split, plot_charts)

warnings.filterwarnings('ignore')


@task(log_prints=True)
def train_model(X_train, y_train, X_val, y_val):
    pos_class_weight = (len(y_train) - np.sum(y_train)) / np.sum(y_train)

    model = xgb.XGBClassifier(
        objective='binary:logistic',
        n_estimators=250,
        scale_pos_weight=pos_class_weight,
        max_delta_step=1,
        early_stopping_rounds=20,
        eval_metric='logloss',
        random_state=Config.RANDOM_SEED,
    )

    eval_set = [(X_train, y_train), (X_val, y_val)]

    model.fit(
        X_train, y_train,
        eval_set=eval_set,
        verbose=True,
    )

    return model


@task(log_prints=True)
def validate_model(model, X_val, y_val):
    y_pred = model.predict(X_val)
    y_probs = model.predict_proba(X_val)[:, 1]

    val_metrics = {
        "accuracy": accuracy_score(y_val, y_pred),
        "precision": precision_score(y_val, y_pred),
        "recall": recall_score(y_val, y_pred),
        "f1": f1_score(y_val, y_pred),
        "roc_auc": roc_auc_score(y_val, y_probs)
    }

    print("Validation metrics:", val_metrics)

    return model, val_metrics


@flow
def train():
    data = load_train_data()
    train_split, val_split = train_val_split(data)
    train_split, val_split = process_data.map([train_split, val_split])

    X_train, y_train = train_split.result()
    X_val, y_val = val_split.result()

    model = train_model(X_train, y_train, X_val, y_val)
    plot_charts(model)
    validate_model(model, X_val, y_val)
    save_model(model)

    return model


if __name__ == "__main__":
    train()
