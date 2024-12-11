import pandas as pd
import numpy as np
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from components.model_evaluation import evaluate_rps
from components.feature_engineering import (
    add_form,
    add_streaks,
    add_team_performance_to_matches,
    add_ratings,
    add_goal_difference,
    add_diffs,
    add_home_factor,
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

# 訓練データとテストデータに分割
train_data = match_data_df[
    match_data_df["Season"].isin(match_data_df["Season"].unique()[:-2])
]
test_data = match_data_df[
    match_data_df["Season"].isin(match_data_df["Season"].unique()[-2:])
]

teams = set(match_data_df["HomeTeam"]).union(set(match_data_df["AwayTeam"]))

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

# k と gamma の最適化の範囲
k_values = range(3, 8)
gamma_values = [0.1 * i for i in range(1, 6)]
best_k, best_gamma, best_rps, best_accuracy = None, None, float("inf"), 0.0

# k と gamma に依存しない特徴量生成
logging.info("Generating features independent of k and gamma...")
match_data_df = add_ratings(match_data_df, ratings_df)
match_data_df = add_goal_difference(match_data_df)
match_data_df = add_home_factor(match_data_df)
intermediate_path = "./csv/intermediate_engineered_data.csv"
match_data_df.to_csv(intermediate_path, index=False)
logging.info(f"Intermediate data saved to {intermediate_path}")


# Time-series cross-validation
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
    for gamma in gamma_values:
        logging.info(f"Testing k={k}, gamma={gamma}")
        rps_scores, accuracy_scores = [], []

        # クロスバリデーション
        for fold_idx, (train_idx, val_idx) in enumerate(
            seasonwise_split(train_data, n_splits, k)
        ):
            logging.info(f"Fold {fold_idx + 1}/{n_splits}")
            temp_train_data = train_data.loc[train_idx].copy()
            temp_val_data = train_data.loc[val_idx].copy()

            # 特徴量生成
            temp_train_data = add_form(temp_train_data, gamma, teams)
            temp_train_data = add_streaks(temp_train_data, k)
            temp_train_data = add_team_performance_to_matches(temp_train_data, k)
            temp_train_data = add_ratings(temp_train_data, ratings_df)
            temp_train_data = add_goal_difference(temp_train_data)
            temp_train_data = add_diffs(temp_train_data)
            temp_train_data = add_home_factor(temp_train_data)

            temp_val_data = add_form(temp_val_data, gamma, teams)
            temp_val_data = add_streaks(temp_val_data, k)
            temp_val_data = add_team_performance_to_matches(temp_val_data, k)
            temp_val_data = add_ratings(temp_val_data, ratings_df)
            temp_val_data = add_goal_difference(temp_val_data)
            temp_val_data = add_diffs(temp_val_data)
            temp_val_data = add_home_factor(temp_val_data)

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
            f"Average RPS: {avg_rps:.4f}, Average Accuracy: {avg_accuracy:.4f} for k={k}, gamma={gamma}"
        )

        # 最良の k と gamma を更新
        if avg_rps < best_rps:
            best_k, best_gamma, best_rps, best_accuracy = (
                k,
                gamma,
                avg_rps,
                avg_accuracy,
            )
            logging.info(
                f"New best parameters: k={best_k}, gamma={best_gamma}, RPS={best_rps:.4f}, Accuracy={best_accuracy:.4f}"
            )

# 最終結果出力
print(f"Best k: {best_k}, Best gamma: {best_gamma}")
print(f"Best RPS: {best_rps:.4f}, Best Accuracy: {best_accuracy:.4f}")

# 最終的な特徴量生成
logging.info("Generating final engineered data with optimized k and gamma...")
match_data_df = add_form(match_data_df, best_gamma, teams)
match_data_df = add_streaks(match_data_df, best_k)
match_data_df = add_team_performance_to_matches(match_data_df, best_k)
match_data_df = add_diffs(match_data_df)

# 最終データを保存
final_output_path = "./csv/rf_engineered_data.csv"
match_data_df.to_csv(final_output_path, index=False)
logging.info(f"Final engineered data saved to {final_output_path}")
