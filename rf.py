import pandas as pd
from components.data_processing import (
    calculate_form,
    add_streaks,
    add_team_performance_to_matches,
    merge_ratings,
    add_goal_difference,
    add_Diffs,
)
from components.model_evaluation import evaluate_rps
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

# データの読み込みと準備
match_data_df = pd.read_csv("./csv/match_data_10yr.csv")
ratings_df = pd.read_csv("./csv/ratings_data.csv")
match_data_df["Date"] = pd.to_datetime(
    match_data_df["Date"], format="%d/%m/%Y", dayfirst=True
)  # 日付を datetime 型に変換

# 最新のシーズンのデータを取得し、テストデータと訓練データに分割
latest_season = match_data_df["Season"].max()
test_data = match_data_df[
    match_data_df["Season"] >= (latest_season - 1)
].copy()  # テストデータを最新の2シーズンに設定
train_data = match_data_df[match_data_df["Season"] < (latest_season - 1)].copy()

# データ加工1：Form の計算
gamma = 0.33  # γ の設定
teams = set(match_data_df["HomeTeam"]).union(
    set(match_data_df["AwayTeam"])
)  # 各チームの一覧
train_data = calculate_form(train_data, gamma, teams)
test_data = calculate_form(test_data, gamma, teams)

# データ加工2： Streak, Weighted Streak の計算
k = 6  # k の設定
train_data = add_streaks(train_data, k)
test_data = add_streaks(test_data, k)

# データ加工3: "PAwayStreak k..." データの追加
train_data = add_team_performance_to_matches(train_data, k)
test_data = add_team_performance_to_matches(test_data, k)

# データ加工4: Ratings
train_data = merge_ratings(train_data, ratings_df)
test_data = merge_ratings(test_data, ratings_df)

# データ加工5: Goal Difference
train_data = add_goal_difference(train_data)
test_data = add_goal_difference(test_data)

# データ加工6: Diff Data
train_data = add_Diffs(train_data)
test_data = add_Diffs(test_data)

## これまでのデータを HTML で表示
train_data.to_html("./htmldata/train_data.html")
test_data.to_html("./htmldata/test_data.html")

### モデルの学習と評価
features = [
    "HomeForm",
    "AwayForm",
    "HomeStreak",
    "AwayStreak",
    "HomeSOT",
    "AwaySOT",
    "HomeGoals",
    "AwayGoals",
    "HomeCorners",
    "AwayCorners",
    "HomeAttackR",
    "AwayAttackR",
    "HomeMidfieldR",
    "AwayMidfieldR",
    "HomeDefenceR",
    "AwayDefenceR",
    "HomeOverallR",
    "AwayOverallR",
    "HomeGD",
    "AwayGD",
    "HomeStreakWeighted",
    "AwayStreakWeighted",
    "FormDiff",
    "StreakDiff",
    "SOTDiff",
    "GoalsDiff",
    "CornersDiff",
    "ARDiff",
    "MRDiff",
    "DRDiff",
    "ORDiff",
    "GDDiff",
    "StreakWeightedDiff",
]

X_train = train_data[features]
y_train = train_data["FTR"]
X_test = test_data[features]
y_test = test_data["FTR"]

# Hyperparameter Tuning

# ## 1. まず、RandomizedSearchCV を使って Hyperparameter の範囲を絞る
# n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=10)] # Number of trees in random forest
# criterion = ["gini", "entropy"] # Function to measure the quality of a split
# max_depth = [int(x) for x in np.linspace(10, 110, num=11)] # Maximum number of levels in tree
# min_samples_split = [2, 5, 10] # Minimum number of samples required to split a node
# min_samples_leaf = [1, 2, 4] # Minimum number of samples required at each leaf node
# max_features = ["log2", "sqrt"] # Number of features to consider at every split
# # Create the random grid
# random_grid = {
#     "n_estimators": n_estimators,
#     "criterion": criterion,
#     "max_depth": max_depth,
#     "min_samples_split": min_samples_split,
#     "min_samples_leaf": min_samples_leaf,
#     "max_features": max_features,
# }

# ### Random search of parameters, using 3 fold cross validation, search across 100 different combinations
# rf_random = RandomizedSearchCV(estimator = rf_model, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
# ### Fit the random search model
# rf_random.fit(X_train, y_train)

### print(rf_random.best_params_)
# # {'n_estimators': 2000, 'min_samples_split': 5, 'min_samples_leaf': 1, 'max_features': 'log2', 'max_depth': 10, 'criterion': 'entropy'}

# ## 2. GridSearchCV による Hyperparameter Tuning

# ### ハイパーパラメータのグリッド設定
# param_grid = {
#     "n_estimators": [200, 500, 1000, 2000],
#     "min_samples_split": [2, 5, 10],
#     "min_samples_leaf": [1, 2, 4],
#     "max_features": ["log2"],
#     "max_depth": [10, 20, 30, 40],
#     "criterion": ["entropy"],
# }

# ### Grid Search
# grid_search = GridSearchCV(
#     estimator=rf_model,
#     param_grid=param_grid,
#     cv=3,
#     scoring="accuracy",
#     verbose=2,
#     n_jobs=-1,
# )
# grid_search.fit(X_train, y_train)

# ### 最適なモデルとハイパーパラメータ
# best_model = grid_search.best_estimator_
# best_params = grid_search.best_params_
# print(f"Best Model: {best_model}")
# print(f"Best Parameters: {best_params}")

### 得られた best_params を使ってモデルを構築
best_params = {
    "criterion": "entropy",
    "max_depth": 10,
    "max_features": "log2",
    "min_samples_leaf": 1,
    "min_samples_split": 5,
    "n_estimators": 1000,
}

# 最適なパラメータでランダムフォレストモデルを構築
rf_model = RandomForestClassifier(**best_params, random_state=42, verbose=2)
rf_model.fit(X_train, y_train)

# テストデータでの予測
y_pred = rf_model.predict(X_test)

# RPSの計算
mean_rps = evaluate_rps(rf_model, X_test, y_test)
print(f"Mean RPS: {mean_rps:.3f}")

# 特徴量の重要度
feature_importances = rf_model.feature_importances_
feature_importance_df = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': feature_importances
})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
print(feature_importance_df)
