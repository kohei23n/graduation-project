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
def evaluate_rps(y_true, y_probs):
    # pandasのSeriesをNumPyの1次元配列に変換する（例: [1, 0, 2]）
    y_true_array = y_true.to_numpy()

    # 1列の2次元配列に変形（例: [[1], [0], [2]]）
    y_true_reshaped = y_true_array.reshape(-1, 1)

    # OneHotEncoderでエンコードして、配列に変換（例: [[1, 0, 0], [0, 1, 0], [0, 0, 1]]）
    encoder = OneHotEncoder()
    y_true_onehot = encoder.fit_transform(y_true_reshaped).toarray()

    # 各サンプルのRPS（Ranked Probability Score）を計算
    # 予測確率 y_probs と実際のOneHotエンコードされたラベルを使ってRPSを求める
    rps_scores = [
        calculate_rps(y_probs[i], y_true_onehot[i]) for i in range(len(y_true))
    ]

    # 平均RPSを返す
    return np.mean(rps_scores)
