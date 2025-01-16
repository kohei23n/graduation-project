import pandas as pd
import logging
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from components.common_v3 import features
from components.tune_hyperparameters import tune_hyperparameters
from components.evaluate_model import evaluate_model

# 進捗状況を表示するための設定
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

# データの読み込みと準備
logging.info("Loading and preparing data...")

train_data = pd.read_csv("./csv/train_data_v3.csv")
test_data = pd.read_csv("./csv/test_data_v3.csv")

# データを分割し、label encoding を実行
label_encoder = LabelEncoder()

X_train = train_data[features]
y_train = label_encoder.fit_transform(train_data["FTR"])
X_test = test_data[features]
y_test = label_encoder.transform(test_data["FTR"])

# チューニングの実行
logging.info("Tuning hyperparameters for Random Forest...")
rf_best_model, rf_best_params = tune_hyperparameters(X_train, y_train, model_type="rf")

logging.info("Tuning hyperparameters for Gradient Boosting...")
xgb_best_model, xgb_best_params = tune_hyperparameters(
    X_train, y_train, model_type="xgb"
)

# print(f"(RF) Best Parameters: {rf_best_params}")
# print(f"(GB) Best Parameters: {xgb_best_params}")

# rf_best_params = {
#     "criterion": "gini",
#     "max_depth": 10,
#     "max_features": "sqrt",
#     "min_samples_leaf": 4,
#     "min_samples_split": 4,
#     "n_estimators": 500,
# }

# xgb_best_params = {
#     "colsample_bytree": 0.8,
#     "gamma": 0,
#     "learning_rate": 0.02,
#     "max_depth": 2,
#     "min_child_weight": 5,
#     "reg_alpha": 0.001,
#     "subsample": 0.77999999999999999,
# }

# 最適なパラメータでモデルを構築
logging.info("Training Random Forest model with best parameters...")
rf_model = RandomForestClassifier(**rf_best_params, random_state=42)
rf_model.fit(X_train, y_train)
rf_y_pred = rf_model.predict(X_test)
logging.info("Done.")

logging.info("Training Gradient Boosting model with best parameters...")
xgb_model = xgb.XGBClassifier(**xgb_best_params, random_state=42)
xgb_model.fit(X_train, y_train)
xgb_y_pred = xgb_model.predict(X_test)
logging.info("Done.")


# ランダムフォレストと勾配ブースティングの評価
logging.info("Evaluating Random Forest model...")
rf_feature_importance = evaluate_model(
    rf_model, X_train, y_test, rf_y_pred, label_encoder
)

logging.info("Evaluating Gradient Boosting model...")
xgb_feature_importance = evaluate_model(
    xgb_model, X_train, y_test, xgb_y_pred, label_encoder
)

# (RF) Accuracy: 0.571
# (RF) Updated Accuracy: 0.58
# (GB) Accuracy: 0.571
# (GB) Updated Accuracy: 0.57

print(f"(RF) Best Parameters: {rf_best_params}")
print(f"(GB) Best Parameters: {xgb_best_params}")
