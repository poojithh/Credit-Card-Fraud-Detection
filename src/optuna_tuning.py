import optuna
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

def tune_xgboost(X_train, y_train):
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train,
        y_train,
        test_size=0.2,
        random_state=42,
        stratify=y_train
    )

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 200),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "eval_metric": "logloss",
            "random_state": 42,
            "n_jobs": -1
        }

        model = XGBClassifier(**params)
        model.fit(X_tr, y_tr)

        probs = model.predict_proba(X_val)[:, 1]
        score = roc_auc_score(y_val, probs)

        return score

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=10)

    print("\n===== Optuna Best Parameters =====")
    print(study.best_params)
    print("Best ROC-AUC:", study.best_value)

    return study.best_params