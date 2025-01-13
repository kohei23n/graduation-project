import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from sklearn.ensemble import RandomForestClassifier
from components.common import load_and_prepare_data, split_data, remove_first_k_gameweeks
from components.feature_engineering import add_team_stats
from components.hyperparameter_tuning import tune_hyperparameters
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 進捗状況を表示するための設定
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

# データの読み込みと準備
logging.info("Loading and preparing data...")
match_data_df, ratings_df = load_and_prepare_data(
    "./csv/match_data.csv", "./csv/ratings_data.csv"
)

train_data, test_data = split_data(match_data_df)

# 最終的な特徴量生成
logging.info("Generating final engineered data with k=4")

train_data = add_team_stats(train_data, ratings_df, k=4)
test_data = add_team_stats(test_data, ratings_df, k=4)

# 最適なkを用いて各シーズンの最初のk試合を除外
train_data = remove_first_k_gameweeks(train_data, k=4)
test_data = remove_first_k_gameweeks(test_data, k=4)

# 不要なカラムを削除
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

required_columns = features + ["Season", "FTR"]
train_data = train_data[required_columns]
test_data = test_data[required_columns]

# 最終データを保存
train_data.to_html("./htmldata/train_data.html")
test_data.to_html("./htmldata/test_data.html")

X_train = train_data[features]
y_train = train_data["FTR"]
X_test = test_data[features]
y_test = test_data["FTR"]

# チューニングの実行
best_model, best_params = tune_hyperparameters(X_train, y_train, model_type="rf")
print(f"Best Parameters: {best_params}")

# 最適なパラメータでランダムフォレストモデルを構築
rf_model = RandomForestClassifier(**best_params)
rf_model.fit(X_train, y_train)

# モデルの予測確率を取得
y_pred = rf_model.predict(X_test)

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

# Results after hyperparameter tuning:
# Accuracy: 0.566

# Confusion Matrix 作成
conf_matrix = confusion_matrix(y_test, y_pred, labels=rf_model.classes_)

# Confusion Matrix を DataFrame に変換
conf_matrix_df = pd.DataFrame(
    conf_matrix, index=rf_model.classes_, columns=rf_model.classes_
)

print("\nConfusion Matrix:")
print(conf_matrix_df)

# Confusion Matrix のプロット
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_df, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.show()

# Classification Report の作成
report_dict = classification_report(
    y_test, y_pred, target_names=rf_model.classes_, output_dict=True
)

# Classification Report を DataFrame に変換
report_df = pd.DataFrame(report_dict).transpose()

# Classification Report の表示
print("Classification Report:")
print(report_df)

# Classification Report のプロット
plt.figure(figsize=(10, 6))
sns.heatmap(report_df.iloc[:-3, :-1], annot=True, cmap="Blues", fmt=".3f")
plt.title("Classification Metrics")
plt.xlabel("Metrics")
plt.ylabel("Classes")
plt.show()
