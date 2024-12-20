import numpy as np
from components.model_evaluation import evaluate_rps
from sklearn.metrics import make_scorer
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.ensemble import RandomForestClassifier

rps_scorer = make_scorer(evaluate_rps, greater_is_better=False, response_method="predict_proba")

# 1. RandomizedSearchCV でパラメータ範囲を絞る
def run_randomized_search(X_train, y_train):
    
    n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)] # 決定木の数
    max_depth = [int(x) for x in np.linspace(10, 110, num = 11)] # 木の深さ
    max_depth.append(None) # 木の深さの最大値
    max_features = ['log2', 'sqrt'] # 分岐の際に考慮する特徴量の数
    min_samples_split = [2, 5, 10] # 分岐を許すためのサンプル数
    min_samples_leaf = [1, 2, 4] # 葉ノードを許すためのサンプル数
    criterion = ["gini", "entropy"] # 分岐の品質を評価する指標
    bootstrap = [True, False] # ブートストラップサンプリングを行うかどうか

    random_grid = {
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "min_samples_split": min_samples_split,
        "min_samples_leaf": min_samples_leaf,
        "max_features": max_features,
        "criterion": criterion,
        "bootstrap": bootstrap,
    }

    # ランダムフォレストモデル
    rf_model = RandomForestClassifier(random_state=42)

    # RandomizedSearchCV の設定
    rf_random = RandomizedSearchCV(
        estimator=rf_model,
        param_distributions=random_grid,
        n_iter=100,
        cv=3,
        scoring=rps_scorer,
        verbose=2,
        random_state=42,
        n_jobs=-1,
    )

    print("Running RandomizedSearchCV...")
    rf_random.fit(X_train, y_train)

    print("Best Parameters from RandomizedSearchCV:")
    print(rf_random.best_params_)

    return rf_random.best_params_

def run_grid_search(X_train, y_train, random_params):
    # グリッドサーチ用のパラメータ範囲
    param_grid = {
        "n_estimators": [
            max(50, random_params["n_estimators"] - 100),  # 最小50を保証
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

    # ランダムフォレストモデル
    rf_model = RandomForestClassifier(random_state=42)

    # GridSearchCV の設定
    grid_search = GridSearchCV(
        estimator=rf_model,
        param_grid=param_grid,
        cv=3,
        scoring=rps_scorer,
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

    # 2. GridSearchCV を実行して最適なパラメータを取得
    best_model, best_params = run_grid_search(X_train, y_train, random_params)

    print("Final Best Parameters:")
    print(best_params)

    return best_model, best_params