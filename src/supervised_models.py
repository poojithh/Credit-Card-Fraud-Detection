from xgboost import XGBClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

def train_models(X_train, y_train, xgb_params=None):
    if xgb_params is None:
        xgb_params = {
            "n_estimators": 100,
            "max_depth": 4,
            "learning_rate": 0.1
        }

    xgb = XGBClassifier(
        **xgb_params,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1
    )

    bag = BaggingClassifier(
        estimator=DecisionTreeClassifier(max_depth=6),
        n_estimators=20,
        random_state=42,
        n_jobs=-1
    )

    xgb.fit(X_train, y_train)
    bag.fit(X_train, y_train)

    return xgb, bag