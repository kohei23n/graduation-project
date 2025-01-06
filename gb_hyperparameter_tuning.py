import pandas as pd
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

# 1. RandomizedSearchCV でパラメータ範囲を絞る
def run_randomized_search(X_train, y_train):

    random_grid = {
        "learning_rate": [0.01, 0.02, 0.05, 0.1, 0.2, 0.3],
        "min_child_weight": [2, 3, 4, 5, 6, 7, 8],
        "max_depth": [1, 2, 3, 4],
        "colsample_bytree": [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        "subsample": [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        "reg_alpha": [0.001, 0.003, 0.01, 0.03, 0.1],
        "reg_lambda": [0.001, 0.003, 0.01, 0.03, 0.1],
        "gamma": [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1],
    }

    # 最適なパラメータで Gradient Boosting モデルを構築
    gb_model = xgb.XGBClassifier(random_state=42, objective="multi:softprob")

    # ランダムサーチのインスタンス作成
    gb_random = RandomizedSearchCV(
        estimator=gb_model,
        param_distributions=random_grid,
        n_iter=100,
        cv=3,
        scoring="neg_log_loss",
        verbose=2,
        random_state=42,
        n_jobs=-1,
    )

    print("Running RandomizedSearchCV...")
    gb_random.fit(X_train, y_train)

    print("Best Parameters from RandomizedSearchCV:")
    print(gb_random.best_params_)

    return gb_random.best_params_


def run_grid_search(X_train, y_train, random_params):
    # グリッドサーチ用のパラメータ範囲
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
        "colsample_bytree": [
            max(0.1, random_params["colsample_bytree"] - 0.1),
            random_params["colsample_bytree"],
            min(1.0, random_params["colsample_bytree"] + 0.1),
        ],
    }

    # XGBoostモデル
    gb_model = xgb.XGBClassifier(random_state=42, objective="multi:softprob")

    # GridSearchCV の設定
    grid_search = GridSearchCV(
        estimator=gb_model,
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
def tune_hyperparameters(X_train, y_train):
    # 1. RandomizedSearchCV を実行してパラメータ範囲を取得
    random_params = run_randomized_search(X_train, y_train)

    print("Randomized Search Best Parameters:")
    print(random_params)

    # 2. GridSearchCV を実行して最適なパラメータを取得
    best_model, best_params = run_grid_search(X_train, y_train, random_params)

    print("Final Best Parameters:")
    print(best_params)

    return best_model, best_params
