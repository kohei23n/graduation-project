import pandas as pd
import numpy as np
import logging
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import KFold
import xgboost as xgb
from components.common import remove_first_k_gameweeks
from components.feature_engineering import add_team_stats


# モデルの学習と最適な k の評価
def optimise_k(train_data, ratings, features, model_type, params=None):
    k_values = range(3, 8)
    n_splits = 5
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    label_encoder = LabelEncoder()
    best_k, best_log_loss, best_accuracy = None, float("inf"), 0.0

    for k in k_values:
        logging.info(f"Testing k={k}")
        accuracy_scores, log_losses = [], []

        temp_data = train_data.copy()
        temp_data = remove_first_k_gameweeks(temp_data, k)

        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(temp_data)):
            logging.info(f"Fold {fold_idx + 1}/{n_splits}")

            temp_train_data = temp_data.iloc[train_idx].copy()
            temp_val_data = temp_data.iloc[val_idx].copy()

            temp_train_data = add_team_stats(temp_train_data, ratings, k)
            temp_val_data = add_team_stats(temp_val_data, ratings, k)

            X_train = temp_train_data[features]
            y_train = label_encoder.fit_transform(temp_train_data["FTR"])
            X_val = temp_val_data[features]
            y_val = label_encoder.transform(temp_val_data["FTR"])

            if model_type == "xgb":
                model = xgb.XGBClassifier(
                    random_state=42, objective="multi:softprob", **(params or {})
                )
            elif model_type == "rf":
                model = RandomForestClassifier(random_state=42, **(params or {}))
            else:
                raise ValueError("Unsupported model type")

            model.fit(X_train, y_train)

            y_probs = model.predict_proba(X_val)
            current_log_loss = log_loss(y_val, y_probs)
            y_pred = model.predict(X_val)
            acc_score = accuracy_score(y_val, y_pred)

            log_losses.append(current_log_loss)
            accuracy_scores.append(acc_score)

            logging.info(
                f"Fold {fold_idx + 1} - Log Loss: {current_log_loss:.4f}, Accuracy: {acc_score:.4f}"
            )

        avg_log_loss = np.mean(log_losses)
        avg_accuracy = np.mean(accuracy_scores)

        logging.info(
            f"Average Log Loss: {avg_log_loss:.4f}, Average Accuracy: {avg_accuracy:.4f} for k={k}"
        )

        if avg_log_loss < best_log_loss:
            best_k, best_log_loss, best_accuracy = k, avg_log_loss, avg_accuracy
            logging.info(
                f"New best parameters: k={best_k}, Log Loss={best_log_loss:.4f}, Accuracy={best_accuracy:.4f}"
            )

    return best_k, best_log_loss, best_accuracy
