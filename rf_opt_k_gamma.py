import pandas as pd
import numpy as np
from components.data_processing import (
    calculate_form,
    add_streaks,
    add_team_performance_to_matches,
    merge_ratings,
    add_goal_difference,
    add_differentials,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV

# データの読み込みと準備
match_data_df = pd.read_csv("./csv/match_data_10yr.csv")
ratings_df = pd.read_csv("./csv/ratings_data.csv")
match_data_df["Date"] = pd.to_datetime(
    match_data_df["Date"], format="%d/%m/%Y", dayfirst=True
)

# 訓練データとテストデータに分割
latest_season = match_data_df["Season"].max()
test_data = match_data_df[match_data_df["Season"] >= (latest_season - 1)].copy()
train_data = match_data_df[match_data_df["Season"] < (latest_season - 1)].copy()

teams = set(match_data_df["HomeTeam"]).union(set(match_data_df["AwayTeam"]))

features = [
    "HForm",
    "AForm",
    "HSt",
    "ASt",
    "HSTKPP",
    "ASTKPP",
    "HCKPP",
    "HGKPP",
    "AGKPP",
    "ACKPP",
    "HAttack",
    "AAttack",
    "HMidField",
    "AMidField",
    "HDefence",
    "ADefence",
    "HOverall",
    "AOverall",
    "HTGD",
    "ATGD",
    "HStWeighted",
    "AStWeighted",
    "FormDifferential",
    "StDifferential",
    "STKPP",
    "GKPP",
    "CKPP",
    "RelAttack",
    "RelMidField",
    "RelDefence",
    "RelOverall",
    "GDDifferential",
    "StWeightedDifferential",
]

# k と gamma のデフォルト設定で特徴量生成
default_k = 6
default_gamma = 0.33
teams = set(match_data_df["HomeTeam"]).union(set(match_data_df["AwayTeam"]))
train_data = calculate_form(train_data, default_gamma, teams)
train_data = add_streaks(train_data, default_k)
train_data = add_team_performance_to_matches(train_data, default_k)
train_data = merge_ratings(train_data, ratings_df)
train_data = add_goal_difference(train_data)
train_data = add_differentials(train_data)

# RandomizedSearchCVのハイパーパラメータ範囲
n_estimators = [int(x) for x in np.linspace(200, 2000, 10)]
max_depth = [int(x) for x in np.linspace(10, 110, 11)]
min_samples_split = [2, 5, 10]
min_samples_leaf = [1, 2, 4]
max_features = ["log2", "sqrt"]
random_grid = {
    "n_estimators": n_estimators,
    "max_depth": max_depth,
    "min_samples_split": min_samples_split,
    "min_samples_leaf": min_samples_leaf,
    "max_features": max_features,
}

rf_model = RandomForestClassifier(random_state=42)
rf_random = RandomizedSearchCV(
    estimator=rf_model,
    param_distributions=random_grid,
    n_iter=50,
    cv=3,
    n_jobs=-1,
    random_state=42,
    verbose=1
)
rf_random.fit(train_data[features], train_data["FTR"])
best_random_params = rf_random.best_params_

# 最適なパラメータを保存するための変数
k_values = range(3, 10)
gamma_values = [0.1 * i for i in range(1, 10)]

best_k = None
best_gamma = None
best_score = 0

# kとgammaのすべての組み合わせをループ
for k in k_values:
    for gamma in gamma_values:
        calculate_form(train_data, gamma, teams)
        add_streaks(train_data, k)
        add_team_performance_to_matches(train_data, k)
        merge_ratings(train_data, ratings_df)
        add_goal_difference(train_data)
        add_differentials(train_data)

        X_train = train_data[features]
        y_train = train_data["FTR"]

        rf_model_temp = RandomForestClassifier(**best_random_params, random_state=42)
        rf_model_temp.fit(X_train, y_train)
        score = rf_model_temp.score(X_train, y_train)

        if score > best_score:
            best_k = k
            best_gamma = gamma
            best_score = score

print(f"Best k: {best_k}, Best gamma: {best_gamma}")
