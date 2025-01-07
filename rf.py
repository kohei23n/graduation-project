import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from rf_hyperparameter_tuning import tune_hyperparameters
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# データの読み込み
train_data = pd.read_csv("./csv/rf_train_data.csv")
test_data = pd.read_csv("./csv/rf_test_data.csv")

## これまでのデータを HTML で表示
train_data.to_html("./htmldata/train_data.html")
test_data.to_html("./htmldata/test_data.html")

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

X_train = train_data[features]
y_train = train_data["FTR"]
X_test = test_data[features]
y_test = test_data["FTR"]

# チューニングの実行
best_model, best_params = tune_hyperparameters(X_train, y_train)
print(f"Best Parameters: {best_params}")

# best_params = {'criterion': 'entropy', 'max_depth': 10, 'max_features': 'log2', 'min_samples_leaf': 4, 'min_samples_split': 2, 'n_estimators': 1500}

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
# Accuracy: 0.562

# Confusion Matrix 作成
conf_matrix = confusion_matrix(y_test, y_pred, labels=rf_model.classes_)

# Confusion Matrix を DataFrame に変換
conf_matrix_df = pd.DataFrame(
    conf_matrix, index=rf_model.classes_, columns=rf_model.classes_
)

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

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred, labels=rf_model.classes_)
conf_matrix_df = pd.DataFrame(
    conf_matrix,
    index=rf_model.classes_,
    columns=rf_model.classes_,
)
print("\nConfusion Matrix:")
print(conf_matrix_df)
