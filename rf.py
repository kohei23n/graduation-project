import pandas as pd
from components.model_evaluation import evaluate_rps
from rf_hyperparameter_tuning import tune_hyperparameters
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# データの読み込み
train_data = pd.read_csv("./csv/rf_train_data.csv")
test_data = pd.read_csv("./csv/rf_test_data.csv")

## これまでのデータを HTML で表示
train_data.to_html("./htmldata/train_data.html")
test_data.to_html("./htmldata/test_data.html")

### モデルの学習と評価
features = [
    # Elo
    "HT_Elo",
    "AT_Elo",
    # Points
    "HT_RecentPoints",
    "HT_HomeRecentPoints",
    "HT_AwayRecentPoints",
    "HT_TotalPoints",
    "AT_RecentPoints",
    "AT_HomeRecentPoints",
    "AT_AwayRecentPoints",
    "AT_TotalPoints",
    # Goals
    "HT_RecentGoals",
    "HT_HomeRecentGoals",
    "HT_AwayRecentGoals",
    "HT_RecentGD",
    "HT_HomeRecentGD",
    "HT_AwayRecentGD",
    "HT_TotalGoals",
    "HT_TotalGD",
    "AT_RecentGoals",
    "AT_HomeRecentGoals",
    "AT_AwayRecentGoals",
    "AT_RecentGD",
    "AT_HomeRecentGD",
    "AT_AwayRecentGD",
    "AT_TotalGoals",
    "AT_TotalGD",
    # Shots
    "HT_RecentShots",
    "HT_HomeRecentShots",
    "HT_AwayRecentShots",
    "HT_RecentSOT",
    "HT_HomeRecentSOT",
    "HT_AwayRecentSOT",
    "HT_TotalShots",
    "HT_TotalSOT",
    "AT_RecentShots",
    "AT_HomeRecentShots",
    "AT_AwayRecentShots",
    "AT_RecentSOT",
    "AT_HomeRecentSOT",
    "AT_AwayRecentSOT",
    "AT_TotalShots",
    "AT_TotalSOT",
    # Ratings 
    "HomeAttackR",
    "HomeMidfieldR",
    "HomeDefenceR",
    "HomeOverallR",
    "AwayAttackR",
    "AwayMidfieldR",
    "AwayDefenceR",
    "AwayOverallR",
    # Differences
    "EloDiff",
    "PointsDiff",
    "RecentPointsDiff",
    "HomeAwayPointsDiff",
    "GoalsDiff",
    "RecentGoalsDiff",
    "HomeAwayGoalsDiff",
    "GDDiff",
    "RecentGDDiff",
    "HomeAwayGDDiff",
    "ShotsDiff",
    "RecentShotsDiff",
    "HomeAwayShotsDiff",
    "SOTDiff",
    "RecentSOTDiff",
    "HomeAwaySOTDiff",
    "ARDiff",
    "MRDiff",
    "DRDiff",
    "ORDiff",
    # Betting Odds
    "B365H",
    "B365D",
    "B365A",
]

X_train = train_data[features]
y_train = train_data["FTR"]
X_test = test_data[features]
y_test = test_data["FTR"]

# チューニングの実行
best_model, best_params = tune_hyperparameters(X_train, y_train)
print(f"Best Parameters: {best_params}")

# 最適なパラメータでランダムフォレストモデルを構築
rf_model = RandomForestClassifier(**best_params)
rf_model.fit(X_train, y_train)

# モデルの予測確率を取得
y_pred = rf_model.predict(X_test)

# RPSの計算
y_probs = rf_model.predict_proba(X_test)
mean_rps = evaluate_rps(y_test, y_probs)
print(f"Mean RPS: {mean_rps:.3f}")

# Accuracyの計算
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.3f}")

# 特徴量の重要度
feature_importances = rf_model.feature_importances_
feature_importance_df = pd.DataFrame(
    {"Feature": X_train.columns, "Importance": feature_importances}
)
feature_importance_df = feature_importance_df.sort_values(
    by="Importance", ascending=False
)
print(feature_importance_df)

## Results when hyperparameter tuning is applied for RPS:
# Mean RPS: 0.193
# Accuracy: 0.545

## Accuracy when hyperparameter tuning is applied for accuracy:
# Mean RPS: 0.192
# Accuracy: 0.554