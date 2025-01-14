import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# モデルを評価する関数
def evaluate_model(model, X_train, y_test, y_pred):
    # Accuracy 計算
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {accuracy:.3f}")

    # 特徴量重要度の取得
    feature_importance = pd.DataFrame(
        {"Feature": X_train.columns, "Importance": model.feature_importances_}
    ).sort_values(by="Importance", ascending=False)
    
    print(f"\nFeature Importance: {feature_importance}")
    
    # Confusion Matrix の作成
    label_encoder = LabelEncoder
    conf_matrix = confusion_matrix(
        y_test, y_pred, labels=range(len(label_encoder.classes_))
    )
    conf_matrix_df = pd.DataFrame(
        conf_matrix,
        index=label_encoder.classes_,
        columns=label_encoder.classes_,
    )
    print(f"\nConfusion Matrix:")
    print(conf_matrix_df)
    
    # Confusion Matrix のプロット
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix_df, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title(f"Confusion Matrix")
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.show()

    # Classification Report の生成
    report = classification_report(
        y_test, y_pred, target_names=label_encoder.classes_, digits=3
    )
    print(f"Classification Report:")
    print(report)

    # Classification Report を DataFrame に変換
    report_dict = classification_report(
        y_test, y_pred, target_names=label_encoder.classes_, output_dict=True
    )
    report_df = pd.DataFrame(report_dict).transpose()

    # Classification Report のプロット
    plt.figure(figsize=(8, 6))
    sns.heatmap(report_df.iloc[:-3, :-1], annot=True, cmap="Blues", fmt=".3f")
    plt.title(f"Classification Metrics")
    plt.show()

    return accuracy, feature_importance, conf_matrix_df, report_df

