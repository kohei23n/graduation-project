import pandas as pd
import numpy as np
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score
from components.model_evaluation import evaluate_rps
from components.data_processing import (
    calculate_form,
    add_streaks,
    add_team_performance_to_matches,
    merge_ratings,
    add_goal_difference,
    add_Diffs,
)

# 進捗状況を表示するための設定
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logging.info("Loading data...")

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
    "HomeForm",
    "AwayForm",
    "HomeStreak",
    "AwayStreak",
    "HomeSOT",
    "AwaySOT",
    "HomeCorners",
    "HomeGoals",
    "AwayGoals",
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

# k と gamma の最適化の範囲
k_values = range(3, 8)
gamma_values = [0.1 * i for i in range(1, 6)]

best_k = None
best_gamma = None
best_rps = float("inf")

# Time-series cross-validation
n_splits = 5
tscv = TimeSeriesSplit(n_splits=n_splits)

# k と gamma の最適化ループ
logging.info("Starting hyperparameter tuning...")
for k in k_values:
    for gamma in gamma_values:
        logging.info(f"Testing k={k}, gamma={gamma}")
        rps_scores = []
        accuracy_scores = []

        # 時系列クロスバリデーション
        for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(train_data)):
            logging.info(f"Fold {fold_idx + 1}/{n_splits}")
            temp_train_data = train_data.iloc[train_idx].copy()
            temp_val_data = train_data.iloc[val_idx].copy()

            # 特徴量生成
            temp_train_data = calculate_form(temp_train_data, gamma, teams)
            temp_train_data = add_streaks(temp_train_data, k)
            temp_train_data = add_team_performance_to_matches(temp_train_data, k)
            temp_train_data = merge_ratings(temp_train_data, ratings_df)
            temp_train_data = add_goal_difference(temp_train_data)
            temp_train_data = add_Diffs(temp_train_data)

            temp_val_data = calculate_form(temp_val_data, gamma, teams)
            temp_val_data = add_streaks(temp_val_data, k)
            temp_val_data = add_team_performance_to_matches(temp_val_data, k)
            temp_val_data = merge_ratings(temp_val_data, ratings_df)
            temp_val_data = add_goal_difference(temp_val_data)
            temp_val_data = add_Diffs(temp_val_data)

            # モデル学習
            X_train = temp_train_data[features]
            y_train = temp_train_data["FTR"]
            X_val = temp_val_data[features]
            y_val = temp_val_data["FTR"]

            rf_model = RandomForestClassifier(random_state=42, verbose=0)
            rf_model.fit(X_train, y_train)

            # RPSとAccuracyの評価
            rps_score = evaluate_rps(rf_model, X_val, y_val)
            y_pred = rf_model.predict(X_val)
            acc_score = accuracy_score(y_val, y_pred)

            rps_scores.append(rps_score)
            accuracy_scores.append(acc_score)
            logging.info(f"Fold {fold_idx + 1} - RPS: {rps_score:.4f}, Accuracy: {acc_score:.4f}")

        # 平均 RPS と Accuracy を計算
        avg_rps = np.mean(rps_scores)
        avg_accuracy = np.mean(accuracy_scores)
        logging.info(f"Average RPS: {avg_rps:.4f}, Average Accuracy: {avg_accuracy:.4f} for k={k}, gamma={gamma}")

        # 最良の k と gamma を更新
        if avg_rps < best_rps:
            best_k = k
            best_gamma = gamma
            best_rps = avg_rps
            best_accuracy = avg_accuracy
            logging.info(f"New best parameters: k={best_k}, gamma={best_gamma}, RPS={best_rps:.4f}, Accuracy={best_accuracy:.4f}")

# 最適な結果を出力
print(f"Best k: {best_k}, Best gamma: {best_gamma}")
print(f"Best RPS: {best_rps:.4f}, Best Accuracy: {best_accuracy:.4f}")