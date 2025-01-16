import logging
import pandas as pd
from components.common_v3 import split_data, features, remove_first_k_gameweeks
from components.feature_engineering_v3 import add_team_stats
from components.opt_k_v3 import optimise_k

# 進捗状況を表示するための設定
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

# データの読み込みと準備
logging.info("Loading and preparing data...")
match_data_df = pd.read_csv("./csv/final_match_data.csv")
logging.info("Data loaded successfully.")

# データの分割
logging.info("Splitting data into training and test datasets...")
train_data, test_data = split_data(match_data_df)

# # K の最適化（一回でOK）
# logging.info("Running optimization for K with Random Forest...")
# best_k, best_log_loss, best_accuracy = optimise_k(train_data, features, model_type="rf")

# logging.info(
#     f"(RF) Best k: {best_k}, Log Loss: {best_log_loss}, Accuracy: {best_accuracy}"
# )

# # Best k: 5, Log Loss: 0.9993317628384265, Accuracy: 0.5253787878787879
# # Best k (after adding xG): 6, Log Loss: 1.0054352680960217, Accuracy: 0.527734375
# # Best k (using only diffs): 7, Log Loss: 1.0156607786366902, Accuracy: 0.5149193548387097

# logging.info("Running optimization for K with Gradient Boosting...")
# best_k, best_log_loss, best_accuracy = optimise_k(
#     train_data, features, model_type="xgb"
# )

# logging.info(
#     f"(GB) Best k: {best_k}, Log Loss: {best_log_loss}, Accuracy: {best_accuracy}"
# )

# # Best k: 7, Log Loss: 1.2439276466838884, Accuracy: 0.494758064516129
# # Best k (after adding xG): : 6, Log Loss: 1.2540531823914112, Accuracy: 0.507421875
# # Best k (using only diffs): 6, Log Loss: 1.2065054356294063, Accuracy: 0.4953125


# 最終的な特徴量生成
logging.info("Generating final engineered data with k=6")
train_data = add_team_stats(train_data, k=6)
test_data = add_team_stats(test_data, k=6)
logging.info("Data generated successfully.")

# 最適なkを用いて各シーズンの最初のk試合を除外
logging.info("Removing first k gameweeks...")
train_data = remove_first_k_gameweeks(train_data, k=6)
test_data = remove_first_k_gameweeks(test_data, k=6)
logging.info("Removed successfully.")

# 最終データを保存（CSV, HTML）
logging.info("Saving data to CSV files...")
train_data.to_csv("./csv/train_data_v3.csv", index=False)
test_data.to_csv("./csv/test_data_v3.csv", index=False)
logging.info("Data saved successfully!")

logging.info("Making HTML Files with data...")
train_data.to_html("./htmldata/train_data_v3.html")
test_data.to_html("./htmldata/test_data_v3.html")
logging.info("Files created successfully!")
