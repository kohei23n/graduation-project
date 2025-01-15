import logging
from components.common import load_and_prepare_data, split_data, features, remove_first_k_gameweeks
from components.feature_engineering import add_team_stats
from components.opt_k import optimise_k

# 進捗状況を表示するための設定
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

# データの読み込みと準備
logging.info("Loading and preparing data...")
match_data_df, ratings_df = load_and_prepare_data(
    "./csv/match_data.csv", "./csv/ratings_data.csv"
)

# データの分割
logging.info("Splitting data...")
train_data, test_data = split_data(match_data_df)

# K の最適化（一回でOK）
# logging.info("Running optimization for K with Random Forest...")
# best_k, best_log_loss, best_accuracy = optimise_k(
#     train_data, ratings_df, features, model_type="rf"
# )

# logging.info(
#     f"(RF) Best k: {best_k}, Log Loss: {best_log_loss}, Accuracy: {best_accuracy}"
# )

# # (RF) Best k: 3, Log Loss: 1.0061962760112702, Accuracy: 0.5139285714285714
# # Updated: (RF) Best k: 5, Log Loss: 0.9993317628384265, Accuracy: 0.5253787878787879

# logging.info("Running optimization for K with Gradient Boosting...")
# best_k, best_log_loss, best_accuracy = optimise_k(
#     train_data, ratings_df, features, model_type="xgb"
# )

# logging.info(
#     f"(GB) Best k: {best_k}, Log Loss: {best_log_loss}, Accuracy: {best_accuracy}"
# )

# (GB) Best k: 5, Log Loss: 1.2136760862953202, Accuracy: 0.5018939393939394
# Updated: (GB) Best k: 7, Log Loss: 1.2439276466838884, Accuracy: 0.494758064516129

# 最終的な特徴量生成
logging.info("Generating final engineered data with k=6")

train_data = add_team_stats(train_data, ratings_df, k=6)
test_data = add_team_stats(test_data, ratings_df, k=6)

# 最適なkを用いて各シーズンの最初のk試合を除外
train_data = remove_first_k_gameweeks(train_data, k=6)
test_data = remove_first_k_gameweeks(test_data, k=6)

# 不要なカラムを削除
required_columns = ["HomeTeam", "AwayTeam"] + features + ["Season", "FTR"]
train_data = train_data[required_columns]
test_data = test_data[required_columns]

# 最終データを保存（CSV, HTML）
logging.info("Saving data...")
train_data.to_csv("./csv/train_data.csv", index=False)
test_data.to_csv("./csv/test_data.csv", index=False)

train_data.to_html("./htmldata/train_data.html")
test_data.to_html("./htmldata/test_data.html")

logging.info("Data saved successfully!")
