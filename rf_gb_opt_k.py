import logging
from components.common import load_and_prepare_data, split_data
from components.opt_k import optimise_k

# 進捗状況を表示するための設定
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

# データの読み込みと準備
logging.info("Loading and preparing data...")
match_data_df, ratings_df = load_and_prepare_data(
    "./csv/match_data.csv", "./csv/ratings_data.csv"
)

features = [
    # Ability
    "HT_AttackR",
    "HT_MidfieldR",
    "HT_DefenceR",
    "HT_OverallR",
    "AT_AttackR",
    "AT_MidfieldR",
    "AT_DefenceR",
    "AT_OverallR",
    "HT_AveragePPG",
    "AT_AveragePPG",
    # Recent Performance
    "HT_RecentShots",
    "HT_RecentSOT",
    "HT_RecentShotsConceded",
    "HT_RecentSOTConceded",
    "AT_RecentShots",
    "AT_RecentSOT",
    "AT_RecentShotsConceded",
    "AT_RecentSOTConceded",
    # Home Advantage
    "HT_HomeWinRate",
    "HT_HomeDrawRate",
    "HT_HomeLossRate",
    "AT_AwayWinRate",
    "AT_AwayDrawRate",
    "AT_AwayLossRate",
    # Elo
    "HT_Elo",
    "AT_Elo",
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

# (RF) Best k: 3, Log Loss: 1.0061962760112702, Accuracy: 0.5139285714285714

logging.info("Running optimization for K with Gradient Boosting...")
best_k, best_log_loss, best_accuracy = optimise_k(
    train_data, ratings_df, features, model_type="xgb"
)

logging.info(
    f"(GB) Best k: {best_k}, Log Loss: {best_log_loss}, Accuracy: {best_accuracy}"
)

# (GB) Best k: 5, Log Loss: 1.2136760862953202, Accuracy: 0.5018939393939394
