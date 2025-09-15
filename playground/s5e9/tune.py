import optuna
import xgboost as xgb
from common import load_train_split_data, Config, store_best_params


def get_params(trial):
    params = {
        "verbosity": 0,
        "objective": "reg:squarederror",
        "tree_method": "hist",
        "booster": "gbtree",

        # Learning rate + depth tradeoff
        "eta": trial.suggest_float("eta", 0.001, 1, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 9),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 8),

        # Sampling
        "subsample": trial.suggest_float("subsample", 0.3, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),

        # Regularization
        "lambda": trial.suggest_float("lambda", 1e-2, 10.0, log=True),
        "alpha": trial.suggest_float("alpha", 1e-2, 10.0, log=True),

        # Split sensitivity
        "gamma": trial.suggest_float("gamma", 0, 4),
    }

    return params


def objective(trial):
    data, target = load_train_split_data()
    dtrain = xgb.DMatrix(data, label=target)
    params = get_params(trial)

    cv_results = xgb.cv(
        params=params,
        dtrain=dtrain,
        num_boost_round=5000,
        nfold=5,
        metrics="rmse",
        early_stopping_rounds=100,
        seed=Config.RANDOM_SEED,
        shuffle=True,
    )

    return min(cv_results["test-rmse-mean"])


def describe_study(study):
    print("Number of finished trials: ", len(study.trials))
    print("Best trial:")
    trial = study.best_trial
    print("  Value: ", trial.value)
    print("  Params:")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")


def run_optuna():
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=200, timeout=60*10, n_jobs=-1)
    describe_study(study)
    return study


def main():
    study = run_optuna()
    store_best_params(study.best_trial.params)


if __name__ == "__main__":
    main()
