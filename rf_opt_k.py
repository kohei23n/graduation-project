import pandas as pd
import numpy as np
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from components.model_evaluation import evaluate_rps
from components.feature_engineering import (
    add_ratings,
    add_elo_rating,
    add_team_stats,
    add_diffs,
)

# 進捗状況を表示するための設定
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logging.info("Loading data...")

# データの読み込みと準備
match_data_df = pd.read_csv("./csv/match_data.csv")
ratings_df = pd.read_csv("./csv/ratings_data.csv")
match_data_df["Date"] = pd.to_datetime(
    match_data_df["Date"], format="%d/%m/%Y", dayfirst=True
)

# k をチューニングするときは訓練データのみで行うため、テストデータは除外
train_data = match_data_df[
    match_data_df["Season"].isin(match_data_df["Season"].unique()[:-2])
].copy()
test_data = match_data_df[
    match_data_df["Season"].isin(match_data_df["Season"].unique()[-2:])
].copy()

features = [
    # Elo
    "HomeElo",
    "AwayElo",
    # Points
    "HT_RecentPoints",
    "HT_HomeRecentPoints",
    "HT_AwayRecentPoints",
    "HT_TotalPoints" "AT_RecentPoints",
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
]

teams = set(match_data_df["HomeTeam"]).union(set(match_data_df["AwayTeam"]))

# 必要な特徴量を追加し、カラムも絞る
match_data_df = add_ratings(match_data_df, ratings_df)


# k の最適化の範囲
k_values = range(3, 8)
best_k, best_rps, best_accuracy = None, float("inf"), 0.0


# Time-series cross-validation
logging.info("Generating features independent of k...")


# 各シーズンの最初の k 試合は IsPrediction=False を設定
def mark_prediction_flag(df, k):
    df = df.sort_values(by=["Season", "Date"]).copy()
    df["IsPrediction"] = df.groupby("Season").cumcount() >= k
    return df


# 学習データには過去データすべてを使用つつ、検証データからは最初の k 試合を除外
def seasonwise_split(data, n_splits, k):
    seasons = sorted(data["Season"].unique())
    for i in range(n_splits):
        # 学習データの設定（最初の i+3 シーズン）
        train_data = data[data["Season"].isin(seasons[: i + 3])]

        # 検証データの設定（次のシーズン、最初の k 試合を除外）
        val_data = data[data["Season"] == seasons[i + 3]]
        val_data = mark_prediction_flag(val_data, k)
        val_data = val_data[val_data["IsPrediction"]]

        # インデックスを返す
        yield train_data.index, val_data.index


# ハイパーパラメータ最適化
logging.info("Starting hyperparameter tuning...")
n_splits = 5
for k in k_values:
    logging.info(f"Testing k={k}")
    rps_scores, accuracy_scores = [], []

    # クロスバリデーション
    for fold_idx, (train_idx, val_idx) in enumerate(
        seasonwise_split(train_data, n_splits, k)
    ):
        logging.info(f"Fold {fold_idx + 1}/{n_splits}")
        temp_train_data = train_data.loc[train_idx].copy()
        temp_val_data = train_data.loc[val_idx].copy()

        # 特徴量生成
        temp_train_data = add_elo_rating(temp_train_data)
        temp_train_data = add_team_stats(temp_train_data, k)
        temp_train_data = add_ratings(temp_train_data, ratings_df)
        temp_train_data = add_diffs(temp_train_data)

        temp_train_data = add_elo_rating(temp_train_data)
        temp_val_data = add_team_stats(temp_val_data, k)
        temp_val_data = add_ratings(temp_val_data, ratings_df)
        temp_val_data = add_diffs(temp_val_data)

        # モデル学習
        X_train = temp_train_data[features]
        y_train = temp_train_data["FTR"]
        X_val = temp_val_data[features]
        y_val = temp_val_data["FTR"]

        rf_model = RandomForestClassifier(random_state=42, verbose=0)
        rf_model.fit(X_train, y_train)

        # 評価
        # モデルの予測確率を取得
        y_probs = rf_model.predict_proba(X_val)

        # RPSを計算する
        rps_score = evaluate_rps(y_val, y_probs)
        y_pred = rf_model.predict(X_val)
        acc_score = accuracy_score(y_val, y_pred)

        rps_scores.append(rps_score)
        accuracy_scores.append(acc_score)

        logging.info(
            f"Fold {fold_idx + 1} - RPS: {rps_score:.4f}, Accuracy: {acc_score:.4f}"
        )

    # 平均スコアを計算
    avg_rps = np.mean(rps_scores)
    avg_accuracy = np.mean(accuracy_scores)
    logging.info(
        f"Average RPS: {avg_rps:.4f}, Average Accuracy: {avg_accuracy:.4f} for k={k}"
    )

    # 最良の k を更新
    if avg_rps < best_rps:
        best_k, best_rps, best_accuracy = (
            k,
            avg_rps,
            avg_accuracy,
        )
        logging.info(
            f"New best parameters: k={best_k}, RPS={best_rps:.4f}, Accuracy={best_accuracy:.4f}"
        )

# 最終結果出力
print(f"Best k: {best_k}")
print(f"Best RPS: {best_rps:.4f}, Best Accuracy: {best_accuracy:.4f}")

# 最終的な特徴量生成
logging.info("Generating final engineered data with optimized k...")

train_data = add_ratings(train_data, ratings_df)
train_data = add_elo_rating(train_data)
train_data = add_team_stats(train_data, best_k)
train_data = add_diffs(train_data)

test_data = add_ratings(test_data, ratings_df)
test_data = add_elo_rating(test_data)
test_data = add_team_stats(test_data, best_k)
test_data = add_diffs(test_data)

# 不要なカラムを削除
required_columns = features + ["Season", "FTR"]
train_data = train_data[required_columns]
test_data = test_data[required_columns]

# 最終データを保存
train_output_path = "./csv/rf_train_data.csv"
test_output_path = "./csv/rf_test_data.csv"

train_data.to_csv(train_output_path, index=False)
test_data.to_csv(test_output_path, index=False)

logging.info(f"Train data saved to {train_output_path}")
logging.info(f"Test data saved to {test_output_path}")
