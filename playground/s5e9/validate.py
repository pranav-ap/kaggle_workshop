import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from common import load_model, load_val_split_data, store_metrics


def validate_model(model, X_val, y_val):
    dval = xgb.DMatrix(X_val)
    preds = model.predict(dval)

    metrics = {
        "mse": float(mean_squared_error(y_val, preds)),
        "mae": float(mean_absolute_error(y_val, preds)),
        "r2": float(r2_score(y_val, preds)),
        "num_samples": len(y_val),
    }

    return metrics


def main():
    model = load_model()
    X_val, y_val = load_val_split_data()
    metrics = validate_model(model, X_val, y_val)
    store_metrics(metrics)

    print("Validation Metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")


if __name__ == "__main__":
    main()
