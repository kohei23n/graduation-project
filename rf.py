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
    "HomeForm",
    "AwayForm",
    "HT_RecentGoals",
    "HT_HomeRecentGoals",
    "HT_AwayRecentGoals",
    "AT_RecentGoals",
    "AT_HomeRecentGoals",
    "AT_AwayRecentGoals",
    "HT_RecentShots",
    "HT_HomeRecentShots",
    "HT_AwayRecentShots",
    "AT_RecentShots",
    "AT_HomeRecentShots",
    "AT_AwayRecentShots",
    "HT_RecentSOT",
    "HT_HomeRecentSOT",
    "HT_AwayRecentSOT",
    "AT_RecentSOT",
    "AT_HomeRecentSOT",
    "AT_AwayRecentSOT",
    "HT_RecentGD",
    "HT_HomeRecentGD",
    "HT_AwayRecentGD",
    "AT_RecentGD",
    "AT_HomeRecentGD",
    "AT_AwayRecentGD",
    "HT_TotalPoints",
    "AT_TotalPoints",
    "HT_RecentPoints",
    "AT_RecentPoints",
    "HT_HomeRecentPoints",
    "HT_AwayRecentPoints",
    "AT_HomeRecentPoints",
    "AT_AwayRecentPoints",
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
    "FormDiff",
    "GoalsDiff",
    "HomeGoalsDiff",
    "AwayGoalsDiff",
    "ShotsDiff",
    "HomeShotsDiff",
    "AwayShotsDiff",
    "SOTDiff",
    "HomeSOTDiff",
    "AwaySOTDiff",
    "GDDiff",
    "HomeGDDiff",
    "AwayGDDiff",
    "ARDiff",
    "MRDiff",
    "DRDiff",
    "ORDiff",
]

X_train = train_data[features]
y_train = train_data["FTR"]
X_test = test_data[features]
y_test = test_data["FTR"]

# チューニングの実行
best_model, best_params = tune_hyperparameters(X_train, y_train)
print(f"Best Parameters: {best_params}")

# 最適なパラメータでランダムフォレストモデルを構築
rf_model = RandomForestClassifier(**best_params, random_state=42, verbose=0)
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
