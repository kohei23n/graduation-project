import pandas as pd
from components.feature_engineering import (
    calculate_form,
    add_streaks,
    add_team_performance_to_matches,
    add_diffs,
    add_home_factor,
)
from components.model_evaluation import evaluate_rps
from rf_hyperparameter_tuning import tune_hyperparameters
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# データの読み込みと準備
match_data_df = pd.read_csv("./csv/rf_engineered_data.csv")
ratings_df = pd.read_csv("./csv/ratings_data.csv")

# 訓練データとテストデータに分割
train_data = match_data_df[
    match_data_df["Season"].isin(match_data_df["Season"].unique()[:-2])
].copy()
test_data = match_data_df[
    match_data_df["Season"].isin(match_data_df["Season"].unique()[-2:])
].copy()

# データ加工1：Form の計算
gamma = 0.5  # γ の設定
teams = set(match_data_df["HomeTeam"]).union(
    set(match_data_df["AwayTeam"])
)  # 各チームの一覧
train_data = calculate_form(train_data, gamma, teams)
test_data = calculate_form(test_data, gamma, teams)

k = 3  # k の設定

# シーズンごとの最初の k 試合を除外する
def exclude_first_k_matches(df, k):
    # 各シーズンごとに試合の順番をカウント
    df = df.sort_values(by=["Season", "Date"])  # シーズンと日付でソート
    return df[df.groupby("Season").cumcount() >= k]

# train_data と test_data から最初の k 試合を除外
train_data = exclude_first_k_matches(train_data, k)
test_data = exclude_first_k_matches(test_data, k)

# データ加工2： Streak, Weighted Streak の計算
train_data = add_streaks(train_data, k)
test_data = add_streaks(test_data, k)

# データ加工3: "Past k..." データの追加
train_data = add_team_performance_to_matches(train_data, k)
test_data = add_team_performance_to_matches(test_data, k)

# データ加工6: Diff Data
train_data = add_diffs(train_data)
test_data = add_diffs(test_data)

# データ加工7：isHome
train_data = add_home_factor(train_data)
test_data = add_home_factor(test_data)

## これまでのデータを HTML で表示
train_data.to_html("./htmldata/train_data.html")
test_data.to_html("./htmldata/test_data.html")

### モデルの学習と評価
features = [
    "HomeForm",
    "AwayForm",
    "HomeAttackR",
    "AwayAttackR",
    "HomeMidfieldR",
    "AwayMidfieldR",
    "HomeDefenceR",
    "AwayDefenceR",
    "HomeOverallR",
    "AwayOverallR",
    "HomeStreak",
    "AwayStreak",
    "HomeGoals",
    "HomeShots",
    "HomeSOT",
    "AwayGoals",
    "AwayShots",
    "AwaySOT",
    "HomeGD",
    "AwayGD",
    "HomeStreakWeighted",
    "AwayStreakWeighted",
    "FormDiff",
    "StreakDiff",
    "GoalsDiff",
    "ShotsDiff",
    "SOTDiff",
    "ARDiff",
    "MRDiff",
    "DRDiff",
    "ORDiff",
    "GDDiff",
    "StreakWeightedDiff",
    "HomeIsHome",
    "AwayIsHome",
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

# テストデータでの予測
y_pred = rf_model.predict(X_test)

# RPSの計算
y_probs = rf_model.predict_proba(X_test)
mean_rps = evaluate_rps(y_test.to_numpy(), y_probs)
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
