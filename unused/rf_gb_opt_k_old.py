import logging

from components.opt_k import load_and_prepare_data, split_data, optimise_k

# 進捗状況を表示するための設定
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

# データの読み込みと準備
logging.info("Loading and preparing data...")
match_data_df, ratings_df = load_and_prepare_data(
    "./csv/match_data.csv", "./csv/ratings_data.csv"
)

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
    # Betting Odds
    "B365H",
    "B365D",
    "B365A",
]

# データの分割
logging.info("Splitting data...")
train_data, test_data = split_data(match_data_df)

# K の最適化
logging.info("Running optimization for K with Random Forest...")
best_k, best_log_loss, best_accuracy = optimise_k(
    train_data, ratings_df, features, model_type="rf"
)

logging.info(
    f"(RF) Best k: {best_k}, Log Loss: {best_log_loss}, Accuracy: {best_accuracy}"
)

# (RF) Best k: 5, Log Loss: 1.0113558680002364, Accuracy: 0.5212121212121212

logging.info("Running optimization for K with Gradient Boosting...")
best_k, best_log_loss, best_accuracy = optimise_k(
    train_data, ratings_df, features, model_type="xgb"
)

logging.info(
    f"(GB) Best k: {best_k}, Log Loss: {best_log_loss}, Accuracy: {best_accuracy}"
)

# (GB) Best k: 6, Log Loss: 1.2042224155566064, Accuracy: 0.508984375