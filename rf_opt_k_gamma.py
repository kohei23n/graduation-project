import pandas as pd
from components.data_processing import (
    calculate_form,
    add_streaks,
    add_team_performance_to_matches,
    merge_ratings,
    add_goal_difference,
    add_differentials,
)
from sklearn.ensemble import RandomForestClassifier

# データの読み込みと準備
match_data_df = pd.read_csv("./csv/match_data_10yr.csv")
ratings_df = pd.read_csv("./csv/ratings_data.csv")
match_data_df["Date"] = pd.to_datetime(
    match_data_df["Date"], format="%d/%m/%Y", dayfirst=True
)

# 訓練データとテストデータに分割
latest_season = match_data_df["Season"].max()
test_data = match_data_df[match_data_df["Season"] >= (latest_season - 1)].copy()
train_data = match_data_df[match_data_df["Season"] < (latest_season - 1)].copy()

teams = set(match_data_df["HomeTeam"]).union(set(match_data_df["AwayTeam"]))

features = [
    "HForm",
    "AForm",
    "HSt",
    "ASt",
    "HSTKPP",
    "ASTKPP",
    "HCKPP",
    "HGKPP",
    "AGKPP",
    "ACKPP",
    "HAttack",
    "AAttack",
    "HMidField",
    "AMidField",
    "HDefence",
    "ADefence",
    "HOverall",
    "AOverall",
    "HTGD",
    "ATGD",
    "HStWeighted",
    "AStWeighted",
    "FormDifferential",
    "StDifferential",
    "STKPP",
    "GKPP",
    "CKPP",
    "RelAttack",
    "RelMidField",
    "RelDefence",
    "RelOverall",
    "GDDifferential",
    "StWeightedDifferential",
]

# k と gamma の最適化の範囲
k_values = range(3, 10)
gamma_values = [0.1 * i for i in range(1, 10)]

# 最適な k と gamma を格納する変数
best_k = None
best_gamma = None
best_score = 0

# デフォルト値のランダムフォレストモデル
rf_model = RandomForestClassifier(random_state=42)  # デフォルト値を使用

# kとgammaのすべての組み合わせをループ
for k in k_values:
    for gamma in gamma_values:
        # 特徴量生成
        temp_train_data = train_data.copy()  # データの再利用のためコピー
        temp_train_data = calculate_form(temp_train_data, gamma, teams)
        temp_train_data = add_streaks(temp_train_data, k)
        temp_train_data = add_team_performance_to_matches(temp_train_data, k)
        temp_train_data = merge_ratings(temp_train_data, ratings_df)
        temp_train_data = add_goal_difference(temp_train_data)
        temp_train_data = add_differentials(temp_train_data)

        # モデル学習
        X_train = temp_train_data[features]
        y_train = temp_train_data["FTR"]
        rf_model.fit(X_train, y_train)

        # スコア計算
        score = rf_model.score(X_train, y_train)

        # 最良の k と gamma を更新
        if score > best_score:
            best_k = k
            best_gamma = gamma
            best_score = score

# 最適な k と gamma の結果を出力
print(f"Best k: {best_k}, Best gamma: {best_gamma}, Best score: {best_score}")