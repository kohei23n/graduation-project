import numpy as np
from sklearn.preprocessing import OneHotEncoder

# RPSを計算する関数
def calculate_rps(probs, outcome):
    cum_probs = np.cumsum(probs)
    cum_outcomes = np.cumsum(outcome)
    sum_rps = 0
    for i in range(len(outcome)):
        sum_rps += (cum_probs[i] - cum_outcomes[i])**2
    return sum_rps / (len(outcome) - 1)

# 全サンプルのRPSを計算する関数
def evaluate_rps(model, X_test, y_test):
    # 予測確率を取得
    y_probs = model.predict_proba(X_test)
    # 正解ラベルをワンホットエンコード
    encoder = OneHotEncoder()
    y_test_onehot = encoder.fit_transform(y_test.reshape(-1, 1)).toarray()
    # 各サンプルのRPSを計算
    rps_scores = [
        calculate_rps(y_probs[i], y_test_onehot[i]) for i in range(len(y_test))
    ]
    # 平均RPSを返す
    return np.mean(rps_scores)