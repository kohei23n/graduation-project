import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import logging
from gb_hyperparameter_tuning import tune_hyperparameters
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from components.opt_k import load_and_prepare_data, split_data, mark_prediction_flag
from components.feature_engineering import add_team_stats

# 進捗状況を表示するための設定
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

# ラベルエンコーダーの初期化
label_encoder = LabelEncoder()

# データの読み込みと準備
logging.info("Loading and preparing data...")
match_data_df, ratings_df = load_and_prepare_data(
    "./csv/match_data.csv", "./csv/ratings_data.csv"
)

train_data, test_data = split_data(match_data_df)

### モデルの学習と評価
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

# 最終的な特徴量生成
logging.info("Generating final engineered data with k=5")

train_data = add_team_stats(train_data, ratings_df, k=5)
test_data = add_team_stats(test_data, ratings_df, k=5)

# 最適なkを用いて各シーズンの最初のk試合を除外
train_data = mark_prediction_flag(train_data, k=5)
train_data = train_data[train_data["IsPrediction"]]

test_data = mark_prediction_flag(test_data, k=5)
test_data = test_data[test_data["IsPrediction"]]

# 不要なカラムを削除
required_columns = features + ["Season", "FTR"]
train_data = train_data[required_columns]
test_data = test_data[required_columns]

X_train = train_data[features]
y_train = train_data["FTR"]
X_test = test_data[features]
y_test = test_data["FTR"]

# ターゲットラベルを数値にエンコード
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

# チューニングの実行
best_model, best_params = tune_hyperparameters(X_train, y_train_encoded)
print(f"Best Parameters: {best_params}")

# 最適なパラメータでランダムフォレストモデルを構築
gb_model = xgb.XGBClassifier(**best_params)
gb_model.fit(X_train, y_train_encoded)

# モデルの予測確率を取得
y_pred = gb_model.predict(X_test)

# Accuracyの計算
accuracy = accuracy_score(y_test_encoded, y_pred)
print(f"Accuracy: {accuracy:.3f}")

# 特徴量の重要度
feature_importances = gb_model.feature_importances_
feature_importance_df = pd.DataFrame(
    {"Feature": X_train.columns, "Importance": feature_importances}
)
feature_importance_df = feature_importance_df.sort_values(
    by="Importance", ascending=False
)
print(feature_importance_df)

# Results after hyperparameter tuning:
# Accuracy: 0.577

# Confusion Matrix 作成
conf_matrix = confusion_matrix(
    y_test_encoded, y_pred, labels=range(len(label_encoder.classes_))
)

# Confusion Matrix を DataFrame に変換
conf_matrix_df = pd.DataFrame(
    conf_matrix,
    index=label_encoder.classes_,
    columns=label_encoder.classes_,
)

# Confusion Matrix のプロット
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_df, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.show()

# Classification Report の生成
report = classification_report(
    y_test_encoded, y_pred, target_names=label_encoder.classes_, digits=3
)

# Classification Report の表示
print("Classification Report:")
print(report)

# Classification Report を DataFrame に変換（表形式にする場合）
report_dict = classification_report(
    y_test_encoded, y_pred, target_names=label_encoder.classes_, output_dict=True
)
report_df = pd.DataFrame(report_dict).transpose()

# Classification Report のプロット
plt.figure(figsize=(8, 6))
sns.heatmap(report_df.iloc[:-3, :-1], annot=True, cmap="Blues", fmt=".3f")
plt.title("Classification Metrics")
plt.show()
