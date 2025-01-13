import numpy as np
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb


# 1. RandomizedSearchCV でパラメータ範囲を絞る
def run_randomized_search(X_train, y_train, model_type):

    if model_type == "rf":
        random_grid = {
            "n_estimators": [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000],
            "max_depth": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110],
            "max_features": ["log2", "sqrt"],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "criterion": ["gini", "entropy", "log_loss"],
            "bootstrap": [True, False],
        }
        model = RandomForestClassifier(random_state=42)
    elif model_type == "xgb":
        random_grid = {
            "learning_rate": [0.01, 0.05, 0.1, 0.2, 0.3],
            "max_depth": [2, 4, 6, 8],
            "min_child_weight": [2, 4, 6, 8, 10],
            "gamma": [0, 0.001, 0.005, 0.01, 0.05, 0.1],
            "subsample": [0.5, 0.6, 0.7, 0.8, 0.9],
            "colsample_bytree": [0.5, 0.6, 0.7, 0.8, 0.9],
            "reg_alpha": [0, 0.001, 0.005, 0.01, 0.05, 0.1],
            "reg_lambda": [0, 0.001, 0.005, 0.01, 0.05, 0.1],
        }
        model = xgb.XGBClassifier(random_state=42, objective="multi:softprob")
    

    # RandomizedSearchCV の設定
    model_random = RandomizedSearchCV(
        estimator=model,
        param_distributions=random_grid,
        n_iter=100,
        cv=3,
        scoring="neg_log_loss",
        verbose=2,
        random_state=42,
        n_jobs=-1,
    )

    print("Running RandomizedSearchCV...")
    model_random.fit(X_train, y_train)

    print("Best Parameters from RandomizedSearchCV:")
    print(model_random.best_params_)

    return model_random.best_params_


def run_grid_search(X_train, y_train, random_params, model_type):
    # グリッドサーチ用のパラメータ範囲
    if model_type == "rf":
        param_grid = {
            "n_estimators": [
                max(50, random_params["n_estimators"] - 100),
                random_params["n_estimators"],
                random_params["n_estimators"] + 100,
            ],
            "criterion": [random_params["criterion"]],
            "max_depth": [
                max(10, random_params["max_depth"] - 10),
                random_params["max_depth"],
                random_params["max_depth"] + 10,
            ],
            "min_samples_split": [
                max(2, random_params["min_samples_split"] - 1),
                random_params["min_samples_split"],
                random_params["min_samples_split"] + 1,
            ],
            "min_samples_leaf": [random_params["min_samples_leaf"]],
            "max_features": [random_params["max_features"]],
        }
        model = RandomForestClassifier(random_state=42)
    elif model_type == "xgb":
        param_grid = {
            "learning_rate": [
                max(0.01, random_params["learning_rate"] - 0.01),
                random_params["learning_rate"],
                random_params["learning_rate"] + 0.01,
            ],
            "max_depth": [
                max(1, random_params["max_depth"] - 1),
                random_params["max_depth"],
                random_params["max_depth"] + 1,
            ],
            "min_child_weight": [
                max(1, random_params["min_child_weight"] - 1),
                random_params["min_child_weight"],
                random_params["min_child_weight"] + 1,
            ],
            "gamma": [
                max(0, random_params["gamma"] - 0.01),
                random_params["gamma"],
                random_params["gamma"] + 0.01,
            ],
            "subsample": [
                max(0.1, random_params["subsample"] - 0.1),
                random_params["subsample"],
                min(1.0, random_params["subsample"] + 0.1),
            ],
            "colsample_bytree": [
                max(0.1, random_params["colsample_bytree"] - 0.1),
                random_params["colsample_bytree"],
                min(1.0, random_params["colsample_bytree"] + 0.1),
            ],
            "reg_alpha": [
                max(0, random_params["reg_alpha"] - 0.001),
                random_params["reg_alpha"],
                random_params["reg_alpha"] + 0.001,
            ],
        }
        model = xgb.XGBClassifier(random_state=42, objective="multi:softprob")

    # GridSearchCV の設定
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=3,
        scoring="neg_log_loss",
        verbose=2,
        n_jobs=-1,
    )

    print("Running GridSearchCV...")
    grid_search.fit(X_train, y_train)

    print("Best Parameters from GridSearchCV:")
    print(grid_search.best_params_)

    return grid_search.best_estimator_, grid_search.best_params_


# 3. Hyperparameter Tuning の統合関数
def tune_hyperparameters(X_train, y_train, model_type):
    # 1. RandomizedSearchCV を実行してパラメータ範囲を取得
    random_params = run_randomized_search(X_train, y_train, model_type)

    # 2. GridSearchCV を実行して最適なパラメータを取得
    best_model, best_params = run_grid_search(X_train, y_train, random_params, model_type)

    print("Final Best Parameters:")
    print(best_params)

    return best_model, best_params
